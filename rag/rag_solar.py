"""Búsqueda de contexto en manuales de O&M para SolarScan (RAG).

Dos métodos, comparados en `rag/evaluar_rag.py`:

* Embeddings (Transformer): cada fragmento del manual y cada consulta se transforman en un
  vector con el modelo `gemini-embedding-001`; se devuelven los fragmentos con mayor similitud
  coseno. Entiende consultas en español sobre un manual en inglés.
* Regla fija (alternativa simple): una lista de palabras clave en inglés por condición del panel
  y búsqueda BM25 por coincidencia de palabras. También se usa como respaldo si la API de
  embeddings no responde, para que el flujo no se detenga.

El índice de fragmentos (`indice_fragmentos.json`) se genera una vez con `construir_indice.py`
a partir del PDF. Los vectores se calculan al primer uso y se guardan en `embeddings.npy`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import httpx
import numpy as np

logger = logging.getLogger("solarscan.rag")
RAIZ = Path(__file__).resolve().parent
MODELO_EMB = "gemini-embedding-001"
DIM = 768
URL_EMB = "https://generativelanguage.googleapis.com/v1beta/models/{m}:batchEmbedContents"
PUNTAJE_MINIMO = 0.55        # valor inicial; evaluar_rag.py lo calibra y lo guarda en config_rag.json

# Regla fija: palabras clave por condición visual y por resultado de la serie.
PALABRAS_CLAVE = {
    "Bird-drop": "bird droppings soiling module cleaning washing",
    "Dusty": "soiling dust module cleaning washing frequency",
    "Snow-Covered": "snow removal module cleaning",
    "Physical-Damage": "module damage cracked glass broken frame replacement inspection",
    "Electrical-damage": "hot spot thermal imaging inverter fault electrical inspection",
    "Clean": "preventive maintenance visual inspection",
    "anomalia": "underperformance inverter performance monitoring production loss",
}

_TOKEN = re.compile(r"[a-záéíóúñ0-9]+", re.I)


def tokenizar(texto: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(texto) if len(t) > 2]


class BM25:
    """BM25 clásico (k1=1,5; b=0,75), sin dependencias externas."""

    def __init__(self, documentos: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = documentos
        self.n = len(documentos)
        self.largo_medio = sum(len(d) for d in documentos) / max(self.n, 1)
        self.df = {}
        for d in documentos:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.tf = [{t: d.count(t) for t in set(d)} for d in documentos]

    def puntajes(self, consulta: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for t in consulta:
            if t not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.largo_medio))
        return s


class EmbeddingsNoDisponibles(Exception):
    pass


def embeber(textos: list[str], tarea: str, clave: str, timeout=30.0, reintentos=0) -> np.ndarray:
    """Vectores normalizados (norma 1) de dimensión DIM, en lotes de 100.

    `reintentos` solo se usa al construir el índice fuera de línea: si la API responde 429 (límite
    de solicitudes por minuto) se espera y se vuelve a intentar. En el servicio se deja en 0 para no
    retrasar la respuesta; ahí el respaldo es la regla fija.
    """
    vectores = []
    for i in range(0, len(textos), 100):
        lote = textos[i:i + 100]
        cuerpo = {"requests": [{"model": f"models/{MODELO_EMB}", "content": {"parts": [{"text": t}]},
                                "taskType": tarea, "outputDimensionality": DIM} for t in lote]}
        if reintentos and i:
            time.sleep(20)                                 # pausa entre lotes al construir el índice
        for intento in range(reintentos + 1):
            try:
                r = httpx.post(URL_EMB.format(m=MODELO_EMB), json=cuerpo, timeout=timeout,
                               headers={"x-goog-api-key": clave, "Content-Type": "application/json"})
                if r.status_code == 429 and intento < reintentos:
                    time.sleep(65)
                    continue
                r.raise_for_status()
                vectores.extend(e["values"] for e in r.json()["embeddings"])
                break
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                if intento < reintentos and isinstance(exc, httpx.TransportError):
                    time.sleep(10)
                    continue
                raise EmbeddingsNoDisponibles(type(exc).__name__) from exc
    m = np.array(vectores, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class BuscadorManuales:
    def __init__(self, carpeta=RAIZ):
        self.carpeta = Path(carpeta)
        datos = json.loads((self.carpeta / "indice_fragmentos.json").read_text(encoding="utf-8"))
        self.fuentes = datos["fuentes"]
        self.fragmentos = datos["fragmentos"]          # [{id, fuente, pagina, seccion, texto}]
        self.bm25 = BM25([tokenizar(f["seccion"] + " " + f["texto"]) for f in self.fragmentos])
        self._emb = None
        self._lock = threading.Lock()
        cfg = self.carpeta / "config_rag.json"
        self.puntaje_minimo = (json.loads(cfg.read_text())["puntaje_minimo"] if cfg.exists() else PUNTAJE_MINIMO)

    # -------------------------------------------------------------- embeddings
    def _matriz(self, clave: str, reintentos=0) -> np.ndarray:
        with self._lock:
            if self._emb is None:
                ruta = self.carpeta / "embeddings.npy"
                if ruta.exists():
                    m = np.load(ruta)
                    if m.shape[0] == len(self.fragmentos):
                        self._emb = m
                if self._emb is None:
                    textos = [f"{f['seccion']}\n{f['texto']}" for f in self.fragmentos]
                    self._emb = embeber(textos, "RETRIEVAL_DOCUMENT", clave, reintentos=reintentos)
                    try:
                        np.save(ruta, self._emb)
                    except OSError:
                        logger.info("No se pudo guardar embeddings.npy; se mantienen en memoria.")
            return self._emb

    def buscar_embeddings(self, consulta: str, k=3, clave: str | None = None) -> list[dict]:
        clave = clave or os.environ.get("GEMINI_API_KEY", "").strip()
        if not clave:
            raise EmbeddingsNoDisponibles("Falta GEMINI_API_KEY")
        m = self._matriz(clave)
        q = embeber([consulta], "RETRIEVAL_QUERY", clave)[0]
        sim = m @ q
        orden = np.argsort(-sim)[:k]
        return [self._salida(i, float(sim[i]), "embeddings") for i in orden]

    # ------------------------------------------------------------- regla fija
    def consulta_regla(self, clase: str | None, anomalia: bool) -> str:
        partes = [PALABRAS_CLAVE.get(clase or "", "")]
        if anomalia:
            partes.append(PALABRAS_CLAVE["anomalia"])
        return " ".join(p for p in partes if p) or PALABRAS_CLAVE["Clean"]

    def buscar_bm25(self, consulta: str, k=3) -> list[dict]:
        s = self.bm25.puntajes(tokenizar(consulta))
        orden = np.argsort(-s)[:k]
        maximo = float(s.max()) if s.max() > 0 else 1.0
        return [self._salida(i, float(s[i] / maximo), "regla_bm25") for i in orden if s[i] > 0]

    # ------------------------------------------------------------- utilidades
    def _salida(self, i, puntaje, metodo):
        f = self.fragmentos[int(i)]
        fuente = self.fuentes[f["fuente"]]
        return {"id": f["id"], "fuente": fuente["titulo"], "pagina": f["pagina"], "seccion": f["seccion"],
                "texto": f["texto"], "puntaje": round(puntaje, 3), "metodo": metodo,
                "cita": f"{fuente['cita_corta']}, p. {f['pagina']}"}

    def buscar(self, consulta: str, clase: str | None = None, anomalia: bool = False, k=3) -> dict:
        """Búsqueda que usa el flujo: embeddings primero; si fallan, regla fija como respaldo."""
        try:
            res = self.buscar_embeddings(consulta, k)
            relevantes = [r for r in res if r["puntaje"] >= self.puntaje_minimo]
            return {"metodo": "embeddings", "consulta": consulta, "resultados": relevantes,
                    "sin_contexto": not relevantes}
        except EmbeddingsNoDisponibles as exc:
            logger.warning("Embeddings no disponibles (%s); se usa la regla fija.", exc)
            q = self.consulta_regla(clase, anomalia)
            res = self.buscar_bm25(q, k)
            return {"metodo": "regla_bm25 (respaldo)", "consulta": q, "resultados": res, "sin_contexto": not res}
