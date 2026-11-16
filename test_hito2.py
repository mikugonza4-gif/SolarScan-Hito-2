"""Pruebas del Hito 2: serie temporal, búsqueda en manuales, reglas y controles del LLM.

No llaman a Gemini: el modelo de lenguaje se reemplaza por funciones de prueba que devuelven
respuestas correctas o con errores, para comprobar que los controles las detectan.
"""
import json

import pytest
from fastapi.testclient import TestClient

# La integración continua del repositorio instala solo dependencias livianas; estas pruebas
# necesitan NumPy y PyTorch, así que se omiten si no están instalados.
pytest.importorskip("numpy")
pytest.importorskip("torch")

import recomendacion_integrada as ri  # noqa: E402
from rag.rag_solar import BM25, BuscadorManuales, tokenizar  # noqa: E402

IMG_POLVO = {"clase": "Dusty", "confianza": 0.91, "revision_humana": False}
IMG_DANO = {"clase": "Physical-Damage", "confianza": 0.76, "revision_humana": True}
IMG_LIMPIO_DUDOSO = {"clase": "Clean", "confianza": 0.557, "revision_humana": True}
SERIE_ANOMALA = {"inversor": "INV-11", "fecha": "2020-06-14", "datos_suficientes": True,
                 "energia_real_kwh": 2108.9, "energia_esperada_kwh": 6945.7, "energia_esperada_lstm_kwh": 6694.7,
                 "indice_rendimiento": 0.304, "indice_rendimiento_lstm": 0.315, "umbral": 0.9, "anomalia": True,
                 "modelos_coinciden": True, "dias_consecutivos_bajo_umbral": 2, "perdida_estimada_kwh": 4836.8,
                 "indice_mediana_planta": 1.074, "problema_general_planta": False}
SERIE_NORMAL = {**SERIE_ANOMALA, "energia_real_kwh": 5592.4, "energia_esperada_kwh": 5543.8,
                "indice_rendimiento": 1.009, "anomalia": False, "dias_consecutivos_bajo_umbral": 0,
                "perdida_estimada_kwh": 0.0}
CONTEXTO = [{"id": "nrel-p30-0", "cita": "NREL (2018), PV O&M Best Practices, p. 30", "pagina": 30,
             "texto": "Module cleaning frequency depends on soiling rate..."}]


# ------------------------------------------------------------- serie temporal
@pytest.fixture(scope="module")
def serie():
    from serie_temporal.servicio_serie import ServicioSerie
    return ServicioSerie()


def test_serie_detecta_anomalia_real(serie):
    e = serie.estado("INV-11", "2020-06-14")
    assert e["anomalia"] and e["indice_rendimiento"] < 0.5
    assert e["perdida_estimada_kwh"] > 0 and e["dias_consecutivos_bajo_umbral"] >= 1


def test_serie_dia_normal(serie):
    e = serie.estado("inv-05", "2020-06-16")
    assert not e["anomalia"] and e["perdida_estimada_kwh"] == 0.0


def test_serie_valida_entradas(serie):
    from serie_temporal.servicio_serie import FechaFueraDeRango, InversorDesconocido
    with pytest.raises(InversorDesconocido):
        serie.estado("INV-99")
    with pytest.raises(FechaFueraDeRango):
        serie.estado("INV-01", "2021-01-01")


def test_serie_alertas_del_dia(serie):
    a = serie.alertas("2020-06-14")
    assert a["inversores_con_anomalia"] == len(a["anomalias"]) > 0
    assert a["anomalias"][0]["perdida_kwh"] >= a["anomalias"][-1]["perdida_kwh"]
    assert any(x["inversor"] == "INV-11" for x in a["anomalias"])
    from datetime import date
    assert serie.fecha_simulada(date(2026, 11, 16)) in serie.fechas[-7:]


# ------------------------------------------------------------------ reglas
def test_reglas_dano_es_alta_y_pide_aprobacion():
    d = ri.decidir(IMG_DANO, SERIE_NORMAL)
    assert d["prioridad"] == "ALTA" and d["requiere_aprobacion"]


def test_reglas_limpio_con_perdida_apunta_al_inversor():
    d = ri.decidir({**IMG_LIMPIO_DUDOSO, "confianza": 0.95}, SERIE_ANOMALA)
    assert d["accion"] == "Revisión eléctrica del inversor" and d["prioridad"] == "ALTA"


def test_reglas_polvo_sin_perdida_no_urgente():
    d = ri.decidir(IMG_POLVO, SERIE_NORMAL)
    assert d["accion"] == "Limpieza programada" and d["prioridad"] == "BAJA"


def test_reglas_baja_confianza_sube_prioridad():
    d = ri.decidir(IMG_LIMPIO_DUDOSO, SERIE_NORMAL)
    assert d["prioridad"] == "MEDIA" and not d["requiere_aprobacion"]


# ------------------------------------------------- controles de alucinación
def respuesta_buena(prioridad="ALTA"):
    return {"diagnostico": "El inversor INV-11 generó el 30,4 % de lo esperado (2108,9 de 6945,7 kWh) y la foto muestra polvo.",
            "recomendacion": "Limpieza prioritaria del string según la frecuencia recomendada [F1].",
            "pasos": ["Programar la limpieza del string [F1].", "Revisar el índice al día siguiente."],
            "prioridad": prioridad, "justificacion": "Pérdida estimada de 4836,8 kWh durante 2 días seguidos.",
            "fuentes": ["NREL (2018), PV O&M Best Practices, p. 30"], "sin_informacion": False}


def test_validacion_acepta_respuesta_correcta():
    dec = ri.decidir(IMG_POLVO, SERIE_ANOMALA)
    perm = ri.cifras_permitidas(IMG_POLVO, SERIE_ANOMALA, CONTEXTO)
    ri.validar(respuesta_buena(dec["prioridad"]), dec, CONTEXTO, perm)


def test_validacion_rechaza_cifra_inventada():
    dec = ri.decidir(IMG_POLVO, SERIE_ANOMALA)
    perm = ri.cifras_permitidas(IMG_POLVO, SERIE_ANOMALA, CONTEXTO)
    mala = respuesta_buena(dec["prioridad"]); mala["justificacion"] = "Se pierden 12000 kWh al mes."
    with pytest.raises(ri.RespuestaInvalida, match="cifras"):
        ri.validar(mala, dec, CONTEXTO, perm)


def test_validacion_rechaza_fuente_inexistente_y_prioridad_cambiada():
    dec = ri.decidir(IMG_POLVO, SERIE_ANOMALA)
    perm = ri.cifras_permitidas(IMG_POLVO, SERIE_ANOMALA, CONTEXTO)
    mala = respuesta_buena(dec["prioridad"]); mala["recomendacion"] = "Limpiar según [F3]."
    with pytest.raises(ri.RespuestaInvalida, match="fuentes"):
        ri.validar(mala, dec, CONTEXTO, perm)
    with pytest.raises(ri.RespuestaInvalida, match="prioridad"):
        ri.validar(respuesta_buena("BAJA"), dec, CONTEXTO, perm)


def test_sin_contexto_exige_declararlo():
    dec = ri.decidir(IMG_POLVO, SERIE_ANOMALA)
    perm = ri.cifras_permitidas(IMG_POLVO, SERIE_ANOMALA, [])
    r = respuesta_buena(dec["prioridad"]); r["recomendacion"] = "Limpiar el string."
    r["pasos"] = ["Limpiar el string.", "Registrar."]
    with pytest.raises(ri.RespuestaInvalida, match="Sin información"):
        ri.validar(r, dec, [], perm)


def test_recomendar_reintenta_y_luego_usa_respaldo(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clave-de-prueba")
    llamadas = []
    def llm_que_inventa(prompt, clave, modelo):
        llamadas.append(prompt)
        r = respuesta_buena("ALTA"); r["diagnostico"] = "Pérdida de 99999 kWh."; return r
    out = ri.recomendar(IMG_POLVO, SERIE_ANOMALA, CONTEXTO, llamar=llm_que_inventa)
    assert len(llamadas) == 2 and "rechazada" in llamadas[1]
    assert out["generado_por"] == "plantilla de respaldo" and not out["controles"]["cifras_verificadas"]
    assert out["prioridad"] == "ALTA" and out["requiere_aprobacion"]


def test_recomendar_ok(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clave-de-prueba")
    out = ri.recomendar(IMG_POLVO, SERIE_ANOMALA, CONTEXTO, llamar=lambda p, c, m: respuesta_buena("ALTA"))
    assert out["generado_por"].startswith("Gemini") and out["controles"]["cifras_verificadas"]
    assert set(["diagnostico", "recomendacion", "prioridad", "justificacion"]) <= set(out)


# ---------------------------------------------------------------- manuales
def test_bm25_ordena_por_relevancia():
    docs = [tokenizar("module cleaning soiling dust"), tokenizar("inverter replacement warranty"),
            tokenizar("vegetation management mowing")]
    s = BM25(docs).puntajes(tokenizar("dust soiling cleaning"))
    assert s.argmax() == 0


def test_buscador_usa_regla_si_no_hay_clave(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    indice = {"fuentes": {"nrel": {"titulo": "Manual", "cita_corta": "NREL (2018)"}},
              "fragmentos": [{"id": "nrel-p1-0", "fuente": "nrel", "pagina": 1, "seccion": "Cleaning",
                              "texto": "Module cleaning removes soiling and dust."},
                             {"id": "nrel-p2-0", "fuente": "nrel", "pagina": 2, "seccion": "Inverters",
                              "texto": "Inverter faults require qualified technicians."}]}
    (tmp_path / "indice_fragmentos.json").write_text(json.dumps(indice))
    r = BuscadorManuales(tmp_path).buscar("¿cómo limpiar polvo?", clase="Dusty", anomalia=False)
    assert r["metodo"].startswith("regla_bm25") and r["resultados"][0]["pagina"] == 1


# --------------------------------------------------------------- endpoints
class PredictorPrueba:
    version = "prueba"
    def __call__(self, data):
        return {}


def test_endpoint_integrada_exige_token(monkeypatch):
    from api_solar import create_app
    monkeypatch.setenv("N8N_WEBHOOK_TOKEN", "secreto")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cuerpo = {"inspeccion_id": "x1", "imagen": IMG_POLVO, "serie": SERIE_ANOMALA, "contexto": CONTEXTO}
    with TestClient(create_app(predictor=PredictorPrueba())) as c:
        assert c.post("/recomendacion-integrada", json=cuerpo).status_code == 401
        r = c.post("/recomendacion-integrada", json=cuerpo, headers={"X-SolarScan-Token": "secreto"})
        assert r.status_code == 200 and r.json()["prioridad"] == "ALTA"
        e = c.get("/serie/estado", params={"inversor": "INV-11", "fecha": "2020-06-14"})
        assert e.status_code == 200 and e.json()["anomalia"]
        assert c.get("/serie/estado", params={"inversor": "INV-11", "fecha": "2019-01-01"}).status_code == 422
        assert c.get("/serie/alertas", params={"fecha": "2020-06-14"}).json()["inversores_con_anomalia"] > 0
