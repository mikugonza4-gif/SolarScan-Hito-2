"""Pantallas web de SolarScan sobre la API: interfaz de inspección y tablero de operación."""
import csv
import io
import os
import time
from pathlib import Path

import httpx
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api_solar import app

WEB = Path(__file__).resolve().parent / 'interfaz'
TABLERO = Path(__file__).resolve().parent / 'tablero'
app.mount('/interfaz', StaticFiles(directory=WEB), name='interfaz')
app.mount('/tablero-archivos', StaticFiles(directory=TABLERO), name='tablero-archivos')
_cache = {'t': 0.0, 'datos': None}


@app.get('/', include_in_schema=False)
def inicio():
    return FileResponse(WEB / 'index.html')


@app.get('/tablero', include_in_schema=False)
def tablero():
    return FileResponse(TABLERO / 'index.html')


@app.get('/api/registros')
def registros():
    """Historial de recomendaciones que n8n escribe en Google Sheets (hoja publicada como CSV).

    La URL se configura en Render con TABLERO_CSV_URL. Se guarda en memoria 60 segundos para no
    consultar la planilla en cada recarga del tablero.
    """
    url = os.environ.get('TABLERO_CSV_URL', '').strip()
    if not url.startswith('https://docs.google.com/'):
        return {'configurado': False, 'registros': []}
    if _cache['datos'] is not None and time.monotonic() - _cache['t'] < 60:
        return _cache['datos']
    try:
        r = httpx.get(url, timeout=10.0, follow_redirects=True)
        r.raise_for_status()
        filas = list(csv.DictReader(io.StringIO(r.content.decode('utf-8'))))
    except (httpx.HTTPError, UnicodeDecodeError, csv.Error):
        return {'configurado': True, 'error': 'No se pudo leer la planilla en este momento.', 'registros': []}
    # Solo columnas operativas: el tablero no muestra códigos de inspector.
    visibles = ['fecha_registro_utc', 'inspeccion_id', 'inversor', 'fecha_operacion', 'clase_imagen',
                'confianza_imagen', 'indice_rendimiento', 'anomalia_serie', 'perdida_kwh', 'accion', 'prioridad',
                'recomendacion', 'fuentes', 'generado_por', 'controles_ok', 'estado_aprobacion', 'ot_id']
    datos = {'configurado': True, 'registros': [{k: f.get(k, '') for k in visibles} for f in filas]}
    _cache.update(t=time.monotonic(), datos=datos)
    return datos
