"""Compara el buscador por embeddings con la regla fija (palabras clave + BM25).

Uso (requiere rag/indice_fragmentos.json; con GEMINI_API_KEY en el entorno evalúa también los
embeddings, sin ella solo la regla fija):
    python rag/evaluar_rag.py

Métricas sobre las 12 preguntas en alcance (top 3):
* Acierto@3: la pregunta tiene al menos un fragmento relevante entre los 3 primeros.
* Precisión@3: proporción de los 3 fragmentos que son relevantes.
* MRR@3: 1 / posición del primer fragmento relevante (0 si no hay).
Además se calibra el umbral de similitud que separa preguntas en alcance de preguntas ajenas al
manual (para responder "sin información" en vez de inventar). Se guarda en config_rag.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))
from rag.rag_solar import BuscadorManuales, EmbeddingsNoDisponibles  # noqa: E402


def relevante(fragmento: dict, criterio: list[list[str]]) -> bool:
    texto = (fragmento["seccion"] + " " + fragmento["texto"]).lower()
    return all(any(t in texto for t in grupo) for grupo in criterio)


def metricas(listas: list[list[bool]]) -> dict:
    acierto = np.mean([any(r) for r in listas])
    precision = np.mean([sum(r) / 3 for r in listas])
    mrr = np.mean([next((1 / (i + 1) for i, v in enumerate(r) if v), 0.0) for r in listas])
    return {"acierto@3": round(float(acierto), 3), "precision@3": round(float(precision), 3), "mrr@3": round(float(mrr), 3)}


def main():
    clave = os.environ.get("GEMINI_API_KEY", "").strip()
    b = BuscadorManuales(RAIZ)
    datos = json.loads((RAIZ / "preguntas_eval.json").read_text(encoding="utf-8"))
    por_id = {f["id"]: f for f in b.fragmentos}
    base_rel = sum(relevante(f, q["criterio"]) for q in datos["en_alcance"] for f in b.fragmentos)
    resultado = {"n_fragmentos": len(b.fragmentos), "n_preguntas": len(datos["en_alcance"]),
                 "fragmentos_relevantes_promedio_por_pregunta": round(base_rel / len(datos["en_alcance"]), 1)}

    # 1) Regla fija (no necesita clave).
    res_regla, detalle = [], []
    for q in datos["en_alcance"]:
        r = b.buscar_bm25(b.consulta_regla(q["clase"], q["anomalia"]), k=3)
        rr = [relevante(por_id[x["id"]], q["criterio"]) for x in r] + [False] * (3 - len(r))
        res_regla.append(rr)
        detalle.append({"pregunta": q["pregunta"], "regla": [(x["cita"], v) for x, v in zip(r, rr)]})
    resultado["regla_fija_bm25"] = metricas(res_regla)

    # 2) Embeddings (requiere GEMINI_API_KEY). Se calculan una vez y quedan en embeddings.npy.
    if clave:
        try:
            b._matriz(clave, reintentos=8)
            res_emb, sim_top = [], []
            for q, d in zip(datos["en_alcance"], detalle):
                e = b.buscar_embeddings(q["pregunta"], k=3, clave=clave)
                re_ = [relevante(por_id[x["id"]], q["criterio"]) for x in e]
                res_emb.append(re_); sim_top.append(e[0]["puntaje"])
                d["embeddings"] = [(x["cita"], x["puntaje"], v) for x, v in zip(e, re_)]
                time.sleep(1)
            sim_fuera = [b.buscar_embeddings(p, k=1, clave=clave)[0]["puntaje"] for p in datos["fuera_de_alcance"]]
            # Umbral: punto medio entre la peor similitud en alcance y la mejor fuera de alcance.
            umbral = (round((min(sim_top) + max(sim_fuera)) / 2, 3) if min(sim_top) > max(sim_fuera)
                      else round(float(np.percentile(sim_top, 10)), 3))
            resultado.update({"embeddings": metricas(res_emb),
                              "similitud_top1_en_alcance": {"min": min(sim_top), "media": round(float(np.mean(sim_top)), 3)},
                              "similitud_top1_fuera_de_alcance": {"max": max(sim_fuera), "valores": sim_fuera},
                              "umbral_sin_informacion": umbral})
            (RAIZ / "config_rag.json").write_text(json.dumps({"puntaje_minimo": umbral}, indent=2), encoding="utf-8")
        except EmbeddingsNoDisponibles as exc:
            # La regla fija queda evaluada igual; el error se registra para repetir la ejecución.
            resultado["embeddings"] = None
            resultado["error_embeddings"] = str(exc)
            print("No se pudieron calcular los embeddings:", exc)
    else:
        resultado["embeddings"] = None
        print("Sin GEMINI_API_KEY: solo se evaluó la regla fija.")
    resultado["detalle"] = detalle
    (RAIZ / "resultados_rag.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resultado.items() if k != "detalle"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
