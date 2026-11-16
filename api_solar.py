"""Servicio SolarScan: imágenes, serie temporal, búsqueda en manuales y recomendación integrada.

Endpoints que usa el flujo principal de n8n (Hito 2):
  POST /predict                   modelo de imágenes (EfficientNet-B0)
  GET  /serie/estado              modelo de serie temporal (regresión + LSTM) para un inversor y día
  GET  /serie/alertas             inversores con anomalía en un día (flujo de monitoreo diario)
  POST /rag/buscar                búsqueda de contexto en el manual de O&M (embeddings)
  POST /recomendacion-integrada   reglas + Gemini con controles contra alucinaciones
Además: /serie/info, /serie/historial y /tablero para el tablero publicado.
"""
import hashlib
import io
import logging
import math
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from typing import Annotated, Literal

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from recomendacion_gemini import GeminiNoDisponible, consultar_gemini
from recomendacion_integrada import recomendar

CLASES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000
FORMATOS = {'JPEG', 'PNG', 'WEBP', 'BMP'}
logger = logging.getLogger('solarscan')
_peticiones_gemini = deque()
_limite_gemini = threading.Lock()


def permitir_recomendacion():
    """Límite global sencillo para proteger la cuota gratuita del prototipo."""
    ahora = time.monotonic()
    with _limite_gemini:
        while _peticiones_gemini and ahora - _peticiones_gemini[0] >= 60:
            _peticiones_gemini.popleft()
        if len(_peticiones_gemini) >= 8:
            return False
        _peticiones_gemini.append(ahora)
        return True

class Resultado(BaseModel):
    inspeccion_id: str
    fecha_utc: str
    modelo_version: str
    clase: Literal['Bird-drop','Clean','Dusty','Electrical-damage','Physical-Damage','Snow-Covered']
    confianza: float = Field(ge=0, le=1)
    baja_confianza: bool
    revision_humana: bool
    probabilidades: dict[str, float]
    nota: str
    tiempo_servicio_ms: float = Field(ge=0)


class SolicitudRecomendacion(BaseModel):
    inspeccion_id: str = Field(min_length=1, max_length=80)
    clase: Literal['Bird-drop','Clean','Dusty','Electrical-damage','Physical-Damage','Snow-Covered']
    confianza: float = Field(ge=0, le=1)
    revision_humana: bool

class ConsultaManual(BaseModel):
    consulta: str = Field(min_length=3, max_length=500)
    clase: str | None = None
    anomalia: bool = False
    k: int = Field(default=3, ge=1, le=5)


class SolicitudIntegrada(BaseModel):
    """Resumen único que arma n8n con las salidas de los tres modelos."""
    inspeccion_id: str = Field(min_length=1, max_length=80)
    imagen: dict
    serie: dict
    contexto: list[dict] = Field(default_factory=list, max_length=5)


def verificar_token(token: str | None):
    """Si el servicio tiene un token configurado, solo n8n (que lo conoce) puede pedir recomendaciones."""
    esperado = os.environ.get('N8N_WEBHOOK_TOKEN', '').strip()
    if esperado and token != esperado:
        raise HTTPException(401, 'Token inválido.')


async def enviar_evento_n8n(evento: dict):
    """Envía solo metadatos y la recomendación, nunca imágenes ni claves al cliente."""
    url = os.environ.get('N8N_WEBHOOK_URL', '').strip()
    token = os.environ.get('N8N_WEBHOOK_TOKEN', '').strip()
    if not url or not token:
        return
    if not url.startswith('https://'):
        logger.error('N8N_WEBHOOK_URL debe usar HTTPS.')
        return
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            respuesta = await client.post(
                url, json=evento, headers={'X-SolarScan-Token': token})
            respuesta.raise_for_status()
    except httpx.HTTPError:
        logger.exception('No se pudo entregar la inspección %s a n8n.',
                         evento['inspeccion_id'])


def validar_imagen(data: bytes, max_bytes=MAX_BYTES, max_pixels=MAX_PIXELS):
    if not data:
        raise HTTPException(400, 'El archivo está vacío.')
    if len(data) > max_bytes:
        raise HTTPException(413, 'El archivo supera el límite de tamaño.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in FORMATOS:
                raise HTTPException(415, 'Formato no admitido. Utilice JPEG, PNG, WEBP o BMP.')
            if getattr(im, 'n_frames', 1) != 1:
                raise HTTPException(415, 'Se requiere una imagen estática de un solo fotograma.')
            if im.width * im.height > max_pixels:
                raise HTTPException(413, 'La resolución supera el límite permitido.')
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(413, 'La resolución supera el límite permitido.')
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(422, 'No se pudo decodificar una imagen válida.')

class PredictorSolar:
    def __init__(self, model_dir):
        # Se importa al iniciar el servicio, no durante las pruebas con dobles.
        from inferencia_solar import cargar_modelo, predecir
        model_dir = Path(model_dir)
        self.modelo, self.config, self.tf = cargar_modelo(model_dir, 'cpu')
        if self.config['clases'] != CLASES:
            raise ValueError('El orden de clases no corresponde al contrato de SolarScan.')
        if not 0 <= self.config['umbral_revision'] <= 1:
            raise ValueError('Umbral inválido.')
        self.predecir = predecir
        h = hashlib.sha256()
        for file in [model_dir/'efficientnet_b0_solar.pt', model_dir/'config.json']:
            with file.open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    h.update(chunk)
        self.version = h.hexdigest()[:16]
        self.lock = threading.Lock()

    def __call__(self, data):
        # Conserva los bytes originales y el procesamiento de inferencia existente.
        # El archivo temporal se elimina también si la predicción falla.
        with tempfile.TemporaryDirectory(prefix='solarscan_') as temp:
            path = Path(temp)/'entrada.img'
            path.write_bytes(data)
            with self.lock:
                return self.predecir(path, self.modelo, self.config, self.tf, 'cpu')

def create_app(model_dir=None, predictor=None, max_bytes=MAX_BYTES, max_pixels=MAX_PIXELS):
    # Solo una recomendación auténtica por cada predicción reciente puede generar un evento.
    # El registro es temporal, acotado y no almacena la fotografía.
    inspecciones_recientes = {}
    vigencia_segundos = 30 * 60
    max_inspecciones = 500

    def recordar_inspeccion(resultado: Resultado):
        ahora = time.monotonic()
        vencidos = [clave for clave, (_, instante) in inspecciones_recientes.items()
                    if ahora - instante > vigencia_segundos]
        for clave in vencidos:
            inspecciones_recientes.pop(clave, None)
        if len(inspecciones_recientes) >= max_inspecciones:
            inspecciones_recientes.pop(next(iter(inspecciones_recientes)))
        inspecciones_recientes[resultado.inspeccion_id] = (resultado, ahora)

    def consumir_inspeccion(datos: SolicitudRecomendacion):
        registro = inspecciones_recientes.get(datos.inspeccion_id)
        if registro is None:
            return None
        prediccion, instante = registro
        if (time.monotonic() - instante > vigencia_segundos
                or prediccion.clase != datos.clase
                or not math.isclose(prediccion.confianza, datos.confianza, abs_tol=1e-9)
                or prediccion.revision_humana != datos.revision_humana):
            return None
        inspecciones_recientes.pop(datos.inspeccion_id, None)
        return prediccion

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        try:
            app.state.predictor = predictor if predictor is not None else await run_in_threadpool(
                PredictorSolar, model_dir or Path(__file__).parent/'modelo')
        except Exception:
            logger.exception('No fue posible cargar el modelo.')
        # Serie temporal y manuales: si fallan, el resto del servicio sigue disponible.
        app.state.serie = None
        app.state.manuales = None
        try:
            from serie_temporal.servicio_serie import ServicioSerie
            app.state.serie = await run_in_threadpool(ServicioSerie)
        except Exception:
            logger.exception('No fue posible cargar la serie temporal.')
        try:
            from rag.rag_solar import BuscadorManuales
            app.state.manuales = await run_in_threadpool(BuscadorManuales)
        except Exception:
            logger.exception('No fue posible cargar el índice de manuales.')
        yield
        app.state.predictor = None

    app = FastAPI(title='SolarScan', version='1.0.0', lifespan=lifespan,
                  description='Clasificación visual de paneles solares para apoyo a O&M.')

    @app.middleware('http')
    async def identificar(request: Request, call_next):
        request.state.inspeccion_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers['X-Inspection-ID'] = request.state.inspeccion_id
        return response

    @app.exception_handler(HTTPException)
    async def error_controlado(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={
            'inspeccion_id': request.state.inspeccion_id, 'error': exc.detail})

    @app.get('/health')
    def health():
        ready = getattr(app.state, 'predictor', None)
        if ready is None:
            raise HTTPException(503, 'El modelo no está disponible.')
        return {'estado': 'listo', 'modelo_version': ready.version}

    @app.post('/predict', response_model=Resultado)
    async def predict(request: Request, file: Annotated[UploadFile, File()]):
        start = time.perf_counter()
        try:
            data = await file.read(max_bytes + 1)
        finally:
            await file.close()
        await run_in_threadpool(validar_imagen, data, max_bytes, max_pixels)
        loaded = getattr(app.state, 'predictor', None)
        if loaded is None:
            raise HTTPException(503, 'El modelo no está disponible.')
        try:
            result = await run_in_threadpool(loaded, data)
            probs = result['probabilidades']
            if set(probs) != set(CLASES) or not all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values()):
                raise ValueError('Probabilidades inválidas.')
            if abs(sum(probs.values()) - 1) > 1e-4:
                raise ValueError('Probabilidades no normalizadas.')
            salida = Resultado(**result, inspeccion_id=request.state.inspeccion_id,
                fecha_utc=datetime.now(timezone.utc).isoformat(), modelo_version=loaded.version,
                tiempo_servicio_ms=round(1000*(time.perf_counter()-start), 3))
            recordar_inspeccion(salida)
            return salida
        except Exception:
            logger.exception('Error de inferencia. inspeccion_id=%s', request.state.inspeccion_id)
            raise HTTPException(500, 'No fue posible completar la predicción.')

    @app.post('/recommend')
    async def recommend(datos: SolicitudRecomendacion, background_tasks: BackgroundTasks):
        """Gemini explica la predicción; después se informa a n8n sin bloquear la UI."""
        if not os.environ.get('GEMINI_API_KEY', '').strip():
            raise HTTPException(503, 'Gemini aún no está configurado. Agrega GEMINI_API_KEY en Render.')
        if not permitir_recomendacion():
            raise HTTPException(429, 'Se alcanzó el límite temporal de recomendaciones. Espera un minuto.')
        try:
            resultado = await consultar_gemini(datos.clase, datos.confianza, datos.revision_humana)
        except GeminiNoDisponible as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        prediccion = consumir_inspeccion(datos)
        if prediccion is not None:
            evento = {campo: getattr(prediccion, campo) for campo in (
                'inspeccion_id', 'fecha_utc', 'clase', 'confianza',
                'baja_confianza', 'revision_humana')}
            evento['recomendacion'] = resultado
            background_tasks.add_task(enviar_evento_n8n, evento)
        return {'inspeccion_id': datos.inspeccion_id, **resultado}

    # ------------------------------------------------------------ Hito 2
    def _serie():
        servicio = getattr(app.state, 'serie', None)
        if servicio is None:
            raise HTTPException(503, 'La serie temporal no está disponible.')
        return servicio

    @app.get('/serie/info')
    def serie_info():
        return _serie().info()

    @app.get('/serie/estado')
    async def serie_estado(inversor: str = Query(min_length=2, max_length=20),
                           fecha: str | None = Query(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$'),
                           detalle: bool = False):
        from serie_temporal.servicio_serie import FechaFueraDeRango, InversorDesconocido
        try:
            return await run_in_threadpool(_serie().estado, inversor, fecha, detalle)
        except (FechaFueraDeRango, InversorDesconocido) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/serie/alertas')
    async def serie_alertas(fecha: str | None = Query(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$')):
        """Monitoreo diario: inversores bajo el umbral, para decidir cuáles inspeccionar con foto."""
        from serie_temporal.servicio_serie import FechaFueraDeRango
        try:
            return await run_in_threadpool(_serie().alertas, fecha)
        except FechaFueraDeRango as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/serie/historial')
    async def serie_historial():
        return await run_in_threadpool(_serie().tabla_diaria)

    @app.post('/rag/buscar')
    async def rag_buscar(datos: ConsultaManual):
        manuales = getattr(app.state, 'manuales', None)
        if manuales is None:
            # Sin índice el flujo no se detiene: el LLM recibe contexto vacío y debe declarar
            # "sin información en los manuales".
            return {'metodo': 'no disponible', 'consulta': datos.consulta, 'resultados': [], 'sin_contexto': True}
        return await run_in_threadpool(manuales.buscar, datos.consulta, datos.clase, datos.anomalia, datos.k)

    @app.post('/recomendacion-integrada')
    async def recomendacion_integrada(datos: SolicitudIntegrada,
                                      x_solarscan_token: Annotated[str | None, Header()] = None):
        verificar_token(x_solarscan_token)
        if datos.imagen.get('clase') not in CLASES:
            raise HTTPException(422, 'Clase de imagen inválida.')
        try:
            float(datos.imagen.get('confianza'))
        except (TypeError, ValueError):
            raise HTTPException(422, 'Confianza de imagen inválida.')
        if not permitir_recomendacion():
            raise HTTPException(429, 'Se alcanzó el límite temporal de recomendaciones. Espera un minuto.')
        resultado = await run_in_threadpool(recomendar, datos.imagen, datos.serie, datos.contexto)
        return {'inspeccion_id': datos.inspeccion_id, **resultado}
    return app

app = create_app(os.environ.get('SOLARSCAN_MODEL_DIR'))
