"""Recomendación integrada de SolarScan (Hito 2).

Recibe el resumen único que arma el flujo de n8n con las salidas de los tres modelos:

* imagen  : clase visual, confianza y bandera de revisión humana (EfficientNet-B0).
* serie   : índice de rendimiento del inversor, energía perdida y días seguidos bajo el umbral.
* contexto: fragmentos del manual de O&M recuperados por embeddings (RAG).

La acción y la prioridad se deciden con reglas fijas y auditables (función `decidir`). Gemini
redacta el diagnóstico y la recomendación usando solo esos datos y citando el manual. Antes de
devolver la respuesta se aplican cuatro controles contra alucinaciones:

1. Restricciones en el prompt (no inventar cifras, no cambiar acción ni prioridad, citar [F#]).
2. Esquema JSON obligatorio y validación de tipos, largos y valores permitidos.
3. Verificación de cifras: todo número del texto debe existir en los resultados de los modelos.
4. Verificación de fuentes: solo se aceptan citas a fragmentos realmente recuperados; si no hubo
   contexto, la respuesta debe declarar "sin información en los manuales".

Si la respuesta no pasa los controles se reintenta una vez indicando el error; si vuelve a
fallar se entrega un texto de respaldo armado con plantillas, marcado como tal.
"""
from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger("solarscan.integrada")

PRIORIDADES = ("ALTA", "MEDIA", "BAJA")
DANOS = {"Physical-Damage", "Electrical-damage"}
SUCIEDAD = {"Dusty", "Bird-drop"}
ETIQUETAS = {
    "Bird-drop": "excrementos de aves", "Clean": "panel aparentemente limpio",
    "Dusty": "polvo acumulado", "Electrical-damage": "indicios visuales de daño eléctrico",
    "Physical-Damage": "posible daño físico", "Snow-Covered": "cubierto de nieve",
}
SIN_INFO = "Sin información en los manuales para este caso"

ESQUEMA = {
    "type": "OBJECT",
    "properties": {
        "diagnostico": {"type": "STRING"},
        "recomendacion": {"type": "STRING"},
        "pasos": {"type": "ARRAY", "items": {"type": "STRING"}},
        "prioridad": {"type": "STRING", "enum": list(PRIORIDADES)},
        "justificacion": {"type": "STRING"},
        "fuentes": {"type": "ARRAY", "items": {"type": "STRING"}},
        "sin_informacion": {"type": "BOOLEAN"},
    },
    "required": ["diagnostico", "recomendacion", "pasos", "prioridad", "justificacion", "fuentes", "sin_informacion"],
}


class RespuestaInvalida(Exception):
    pass


# ----------------------------------------------------------------- reglas
def decidir(imagen: dict, serie: dict) -> dict:
    """Acción, prioridad y necesidad de aprobación según reglas explícitas."""
    clase = imagen["clase"]
    conf = float(imagen["confianza"])
    datos = bool(serie.get("datos_suficientes", False))
    anomalia = datos and bool(serie.get("anomalia"))
    indice = serie.get("indice_rendimiento") if datos else None
    general = datos and bool(serie.get("problema_general_planta"))
    razones = []

    if clase in DANOS:
        accion, prioridad = "Inspección técnica en terreno", "ALTA"
        razones.append("la foto sugiere daño físico o eléctrico")
    elif general and anomalia:
        accion, prioridad = "Verificar sensor de irradiancia y estado de la red", "MEDIA"
        razones.append("la caída afecta a la mayoría de los inversores, no solo a este")
    elif anomalia:
        severa = indice is not None and indice < 0.6
        prioridad = "ALTA" if severa else "MEDIA"
        if clase in SUCIEDAD:
            accion = "Limpieza prioritaria del string"
            razones.append("la generación está bajo lo esperado y la foto muestra suciedad")
        elif clase == "Snow-Covered":
            accion = "Remoción de nieve y verificación"
            razones.append("la generación está bajo lo esperado y la foto muestra nieve")
        else:
            accion = "Revisión eléctrica del inversor"
            razones.append("la generación está bajo lo esperado pero la foto no muestra suciedad")
        if severa:
            razones.append("la pérdida supera el 40 % de lo esperado")
    else:
        if clase in SUCIEDAD:
            accion, prioridad = "Limpieza programada", "BAJA"
            razones.append("hay suciedad visible, pero la generación está dentro de lo esperado")
        else:
            accion, prioridad = "Seguimiento", "BAJA"
            razones.append("sin daño visible y generación dentro de lo esperado" if datos
                           else "sin daño visible; no hay datos de generación suficientes")

    if conf < 0.70 and prioridad == "BAJA":
        prioridad = "MEDIA"
        razones.append("la confianza del modelo de imágenes es baja")
    if datos and serie.get("dias_consecutivos_bajo_umbral", 0) >= 3 and prioridad == "MEDIA":
        prioridad = "ALTA"
        razones.append("la pérdida se repite tres días o más")

    requiere_aprobacion = accion != "Seguimiento"
    return {"accion": accion, "prioridad": prioridad, "requiere_aprobacion": requiere_aprobacion,
            "razones": razones}


# --------------------------------------------------------- cifras permitidas
def _numeros(texto: str) -> list[float]:
    salida = []
    for m in re.finditer(r"(?<![\w-])\d+(?:[.,]\d+)?", texto):
        try:
            salida.append(float(m.group().replace(",", ".")))
        except ValueError:
            pass
    return salida


def cifras_permitidas(imagen: dict, serie: dict, contexto: list[dict]) -> set[float]:
    """Todos los números que el texto puede mencionar, en sus formas habituales."""
    base = set(range(0, 11))                               # numeración de pasos y conteos pequeños
    def agregar(v, pct=False):
        if v is None:
            return
        v = float(v)
        for x in ((v * 100,) if pct else (v,)):
            base.update({round(x, 1), round(x), round(x, 2)})
    agregar(imagen.get("confianza"), pct=True)
    for k in ("indice_rendimiento", "indice_rendimiento_lstm", "umbral", "indice_mediana_planta"):
        agregar(serie.get(k), pct=True)
        if serie.get(k) is not None:
            agregar(100 - float(serie[k]) * 100)          # "pérdida de 37 %" a partir de un índice de 0,63
    for k in ("energia_real_kwh", "energia_esperada_kwh", "energia_esperada_lstm_kwh",
              "perdida_estimada_kwh", "dias_consecutivos_bajo_umbral"):
        agregar(serie.get(k))
    inv = re.findall(r"\d+", str(serie.get("inversor", "")))
    base.update(float(x) for x in inv)
    base.update(float(x) for x in re.findall(r"\d+", str(serie.get("fecha", ""))))
    for c in contexto:
        base.add(float(c.get("pagina", 0)))
        base.update(float(x) for x in re.findall(r"\d{4}", c.get("cita", "")))
    return base


def verificar_cifras(textos: list[str], permitidas: set[float]) -> list[float]:
    """Devuelve las cifras del texto que no aparecen en los resultados de los modelos."""
    malas = []
    for t in textos:
        for n in _numeros(t):
            if not any(abs(n - p) <= max(0.6, 0.01 * abs(p)) for p in permitidas):
                malas.append(n)
    return malas


# --------------------------------------------------------------- validación
def validar(gen: dict, decision: dict, contexto: list[dict], permitidas: set[float]) -> dict:
    if not isinstance(gen, dict):
        raise RespuestaInvalida("la respuesta no es un objeto JSON")
    for campo, largo in (("diagnostico", 600), ("recomendacion", 600), ("justificacion", 700)):
        v = gen.get(campo)
        if not isinstance(v, str) or not 10 <= len(v) <= largo:
            raise RespuestaInvalida(f"campo '{campo}' ausente o con largo inválido")
    pasos = gen.get("pasos")
    if not isinstance(pasos, list) or not 2 <= len(pasos) <= 4 or not all(isinstance(p, str) and 5 <= len(p) <= 250 for p in pasos):
        raise RespuestaInvalida("'pasos' debe tener entre 2 y 4 textos")
    if gen.get("prioridad") != decision["prioridad"]:
        raise RespuestaInvalida(f"la prioridad debe ser {decision['prioridad']}")
    textos = [gen["diagnostico"], gen["recomendacion"], gen["justificacion"], *pasos]
    citas = set(re.findall(r"\[F(\d+)\]", " ".join(textos)))
    validas = {str(i + 1) for i in range(len(contexto))}
    if not citas <= validas:
        raise RespuestaInvalida(f"cita fuentes inexistentes: {sorted(citas - validas)}")
    if contexto and not citas:
        raise RespuestaInvalida("debe citar al menos un fragmento del manual con [F#]")
    if not contexto and (not gen.get("sin_informacion") or SIN_INFO.lower() not in gen["recomendacion"].lower()):
        raise RespuestaInvalida(f"sin contexto: la recomendación debe decir '{SIN_INFO}'")
    malas = verificar_cifras(textos, permitidas)
    if malas:
        raise RespuestaInvalida(f"cifras que no están en los resultados de los modelos: {malas[:5]}")
    return {"citas_usadas": sorted(citas, key=int)}


# ------------------------------------------------------------------- prompt
def armar_prompt(imagen, serie, contexto, decision, error_previo=None) -> str:
    datos = {
        "inspeccion_imagen": {"condicion_visual": ETIQUETAS[imagen["clase"]],
                              "confianza_modelo_pct": round(float(imagen["confianza"]) * 100, 1),
                              "requiere_revision_humana": bool(imagen.get("revision_humana"))},
        "serie_generacion": ({"inversor": serie.get("inversor"), "fecha": serie.get("fecha"),
                              "indice_rendimiento_pct": round(serie["indice_rendimiento"] * 100, 1),
                              "umbral_pct": round(serie["umbral"] * 100, 1),
                              "energia_real_kwh": serie["energia_real_kwh"],
                              "energia_esperada_kwh": serie["energia_esperada_kwh"],
                              "perdida_estimada_kwh": serie["perdida_estimada_kwh"],
                              "dias_consecutivos_bajo_umbral": serie["dias_consecutivos_bajo_umbral"],
                              "indice_mediana_planta_pct": (round(serie["indice_mediana_planta"] * 100, 1)
                                                            if serie.get("indice_mediana_planta") is not None else None),
                              "lstm_coincide": serie.get("modelos_coinciden")}
                             if serie.get("datos_suficientes") else {"sin_datos": True}),
        "decision_reglas": {"accion": decision["accion"], "prioridad": decision["prioridad"],
                            "motivos": decision["razones"]},
        "fragmentos_manual": [{"ref": f"F{i + 1}", "cita": c["cita"], "texto": c["texto"][:900]}
                              for i, c in enumerate(contexto)],
    }
    reglas = (
        "Eres el asistente de un supervisor de operación y mantenimiento (O&M) de una planta solar. "
        "Integra las tres fuentes: la inspección de la foto, la serie de generación del inversor y los "
        "fragmentos del manual. No has visto la foto ni el inversor. Reglas obligatorias:\n"
        "1. Usa solo las cifras que aparecen en DATOS, sin redondear distinto ni calcular cifras nuevas.\n"
        "2. La acción y la prioridad ya fueron decididas por reglas: repite exactamente la prioridad "
        f"'{decision['prioridad']}' y explica la acción '{decision['accion']}'. No las cambies.\n"
        "3. Toda indicación técnica debe citar el fragmento que la respalda con [F1], [F2] o [F3]. No "
        "inventes procedimientos que no estén en los fragmentos.\n"
        f"4. Si no hay fragmentos, escribe en 'recomendacion' la frase '{SIN_INFO}', marca "
        "sin_informacion=true y limita los pasos a verificar en terreno y registrar.\n"
        "5. No confirmes fallas eléctricas ni des instrucciones de intervención eléctrica a personal no "
        "calificado. La confianza del modelo no es una probabilidad calibrada.\n"
        "6. Español claro y profesional. diagnostico ≤ 60 palabras, recomendacion ≤ 50, justificacion ≤ 70, "
        "2 a 4 pasos de ≤ 25 palabras. En 'fuentes' lista las citas usadas (texto de 'cita').\n"
    )
    if error_previo:
        reglas += f"\nTu respuesta anterior fue rechazada por este motivo: {error_previo}. Corrígelo.\n"
    return reglas + "\nDATOS:\n" + json.dumps(datos, ensure_ascii=False)


def _llamar_gemini(prompt: str, clave: str, modelo: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
    cuerpo = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "responseSchema": ESQUEMA,
                                   "temperature": 0.2, "maxOutputTokens": 1200}}
    r = httpx.post(url, json=cuerpo, timeout=40.0,
                   headers={"x-goog-api-key": clave, "Content-Type": "application/json"})
    r.raise_for_status()
    partes = r.json()["candidates"][0]["content"]["parts"]
    return json.loads("".join(p.get("text", "") for p in partes))


def _pct(v) -> str:
    return f"{float(v) * 100:.1f}".replace(".", ",") + " %"


def respaldo(imagen, serie, contexto, decision) -> dict:
    """Texto armado con plantillas cuando Gemini no responde o no pasa los controles."""
    if serie.get("datos_suficientes"):
        s = (f"El inversor {serie['inversor']} generó el {_pct(serie['indice_rendimiento'])} de lo "
             f"esperado el {serie['fecha']}")
    else:
        s = "No hay datos de generación suficientes para ese día"
    diag = (f"La foto se clasificó como {ETIQUETAS[imagen['clase']]} "
            f"({_pct(imagen['confianza'])} de confianza). {s}.")
    if contexto:
        rec = f"{decision['accion']}, según el procedimiento del manual [F1]."
        fuentes = [contexto[0]["cita"]]
    else:
        rec = f"{decision['accion']}. {SIN_INFO}."
        fuentes = []
    return {"diagnostico": diag, "recomendacion": rec,
            "pasos": ["Verificar en terreno la condición del panel y del inversor.",
                      "Registrar el resultado en la orden de trabajo."],
            "prioridad": decision["prioridad"], "justificacion": "; ".join(decision["razones"]).capitalize() + ".",
            "fuentes": fuentes, "sin_informacion": not contexto}


def recomendar(imagen: dict, serie: dict, contexto: list[dict], llamar=_llamar_gemini) -> dict:
    """Punto de entrada: decide con reglas, pide el texto a Gemini y aplica los controles."""
    decision = decidir(imagen, serie)
    permitidas = cifras_permitidas(imagen, serie, contexto)
    clave = os.environ.get("GEMINI_API_KEY", "").strip()
    modelo = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    controles = {"esquema_json": False, "prioridad_respetada": False, "cifras_verificadas": False,
                 "fuentes_verificadas": False, "intentos": 0, "motivo_rechazo": None}
    texto, generado_por, error = None, "plantilla de respaldo", None
    if clave:
        for intento in (1, 2):
            controles["intentos"] = intento
            try:
                gen = llamar(armar_prompt(imagen, serie, contexto, decision, error), clave, modelo)
                validar(gen, decision, contexto, permitidas)
                texto, generado_por = gen, f"Gemini ({modelo})"
                controles.update(esquema_json=True, prioridad_respetada=True, cifras_verificadas=True,
                                 fuentes_verificadas=True, motivo_rechazo=None)
                break
            except RespuestaInvalida as exc:
                error = str(exc); controles["motivo_rechazo"] = error
                logger.warning("Respuesta rechazada (intento %s): %s", intento, error)
            except (httpx.HTTPError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                controles["motivo_rechazo"] = f"Gemini no disponible: {type(exc).__name__}"
                break
    else:
        controles["motivo_rechazo"] = "Falta GEMINI_API_KEY"
    if texto is None:
        texto = respaldo(imagen, serie, contexto, decision)
    return {**{k: texto[k] for k in ("diagnostico", "recomendacion", "pasos", "prioridad", "justificacion",
                                    "fuentes", "sin_informacion")},
            "accion": decision["accion"], "requiere_aprobacion": decision["requiere_aprobacion"],
            "motivos_regla": decision["razones"], "generado_por": generado_por, "controles": controles,
            "advertencia": "Orientación preliminar: un responsable debe aprobar antes de intervenir."}
