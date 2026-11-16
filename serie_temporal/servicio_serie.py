"""Servicio de la serie temporal de SolarScan.

Calcula, para un inversor y un día, la energía que debió generar según la irradiancia y la
temperatura medidas, y la compara con la energía real. Usa los dos modelos entrenados en
`notebooks/02_serie_temporal.ipynb`:

* Regresión lineal física (modelo elegido en validación para decidir).
* LSTM (segunda opinión; se reporta junto a la regresión).

Los datos corresponden a la planta 2 del dataset público de Kaggle y se reproducen día a día
como si fueran la operación de la planta (supuesto documentado en el informe).
Solo usa la biblioteca estándar, NumPy y PyTorch, que ya están en el servicio de imágenes.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn

RAIZ = Path(__file__).resolve().parent
PASO_HORAS = 0.25            # registros cada 15 minutos


class LSTMSolar(nn.Module):
    """Misma arquitectura que en el notebook de entrenamiento."""

    def __init__(self, n_in=5, oculto=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, oculto, batch_first=True)
        self.salida = nn.Sequential(nn.Linear(oculto, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.salida(h[:, -1]).squeeze(-1)


class FechaFueraDeRango(ValueError):
    pass


class InversorDesconocido(ValueError):
    pass


def _leer_csv_gz(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        yield from csv.DictReader(f)


class ServicioSerie:
    def __init__(self, carpeta_artefactos=RAIZ / "artefactos", carpeta_datos=RAIZ / "datos"):
        self.cfg = json.loads((Path(carpeta_artefactos) / "config_serie.json").read_text(encoding="utf-8"))
        self.umbral = float(self.cfg["umbral_indice"])
        self.g_min = float(self.cfg["g_min"])
        self.ventana = int(self.cfg["ventana"])
        self.coef = np.array(self.cfg["regresion"]["coef"], dtype=np.float64)
        self.alias = self.cfg["alias"]                      # clave original -> INV-XX
        self.cap = self.cfg["capacidad_kw"]                 # INV-XX -> kW de referencia
        self.inversores = sorted(self.cap)
        self._cargar_datos(Path(carpeta_datos))
        oculto = int(self.cfg["lstm"]["oculto"])
        self.lstm = LSTMSolar(oculto=oculto)
        self.lstm.load_state_dict(torch.load(Path(carpeta_artefactos) / "lstm_solar.pt",
                                             map_location="cpu", weights_only=True))
        self.lstm.eval()
        self.lock = threading.Lock()
        self._cache_dias = None

    # ------------------------------------------------------------------ datos
    def _cargar_datos(self, carpeta):
        clima = {}
        for r in _leer_csv_gz(carpeta / "Plant_2_Weather_Sensor_Data.csv.gz"):
            clima[r["DATE_TIME"]] = (float(r["IRRADIATION"]), float(r["MODULE_TEMPERATURE"]),
                                     float(r["AMBIENT_TEMPERATURE"]))
        inicio = datetime.fromisoformat(self.cfg["periodo"][0])
        fin = datetime.fromisoformat(self.cfg["periodo"][1]) + timedelta(hours=23, minutes=45)
        n = int((fin - inicio).total_seconds() // 900) + 1
        self.tiempos = [inicio + timedelta(minutes=15 * i) for i in range(n)]
        self.idx = {t.strftime("%Y-%m-%d %H:%M:%S"): i for i, t in enumerate(self.tiempos)}
        G = np.full(n, np.nan); Tm = np.full(n, np.nan); Ta = np.full(n, np.nan)
        for t, (g, tm, ta) in clima.items():
            i = self.idx.get(t)
            if i is not None:
                G[i], Tm[i], Ta[i] = g, tm, ta
        self.G, self.Tm, self.Ta = G, Tm, Ta
        self.ac = {inv: np.full(n, np.nan) for inv in self.inversores}
        for r in _leer_csv_gz(carpeta / "Plant_2_Generation_Data.csv.gz"):
            inv = self.alias.get(r["SOURCE_KEY"]); i = self.idx.get(r["DATE_TIME"])
            if inv and i is not None:
                self.ac[inv][i] = float(r["AC_POWER"])
        # Entradas normalizadas de la LSTM (mismo orden de características del entrenamiento).
        mu = self.cfg["normalizacion"]["media"]; sd = self.cfg["normalizacion"]["desv"]
        hora = np.array([t.hour + t.minute / 60 for t in self.tiempos])
        cols = [(G - mu["IRRADIATION"]) / sd["IRRADIATION"],
                (Tm - mu["MODULE_TEMPERATURE"]) / sd["MODULE_TEMPERATURE"],
                (Ta - mu["AMBIENT_TEMPERATURE"]) / sd["AMBIENT_TEMPERATURE"],
                np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24)]
        M = np.column_stack(cols)
        # Se arrastra el último valor conocido en los pocos intervalos sin clima (igual que en el notebook).
        for j in range(M.shape[1]):
            col = M[:, j]
            for i in range(len(col)):
                if np.isnan(col[i]):
                    col[i] = col[i - 1] if i > 0 else 0.0
        self.M = M.astype(np.float32)
        self.fechas = sorted({t.date() for t in self.tiempos})

    # --------------------------------------------------------------- modelos
    def _esperado(self, idxs):
        """Potencia esperada normalizada (0-1) por regresión y por LSTM en los intervalos dados."""
        G = self.G[idxs]; T = self.Tm[idxs]
        reg = np.clip(self.coef[0] * G + self.coef[1] * G * (T - 25) + self.coef[2], 0, None)
        pos = np.clip(idxs, self.ventana - 1, None)
        X = np.stack([self.M[p - self.ventana + 1:p + 1] for p in pos])
        with self.lock, torch.inference_mode():
            lstm = np.clip(self.lstm(torch.from_numpy(X)).numpy(), 0, None)
        return reg, lstm

    def _idx_dia(self, d: date):
        ini = self.idx.get(f"{d.isoformat()} 00:00:00")
        if ini is None:
            raise FechaFueraDeRango(f"La fecha debe estar entre {self.fechas[0]} y {self.fechas[-1]}.")
        idxs = np.arange(ini, ini + 96)
        return idxs[(self.G[idxs] > self.g_min) & ~np.isnan(self.G[idxs])]

    def _indice(self, inv, d):
        idxs = self._idx_dia(d)
        real = self.ac[inv][idxs]
        ok = ~np.isnan(real)
        cobertura = float(ok.mean()) if len(idxs) else 0.0
        if ok.sum() < 8:
            return None
        reg, lstm = self._esperado(idxs[ok])
        cap = self.cap[inv]
        e_real = float(real[ok].sum() * PASO_HORAS)
        e_reg = float(reg.sum() * cap * PASO_HORAS)
        e_lstm = float(lstm.sum() * cap * PASO_HORAS)
        return {"energia_real_kwh": e_real, "energia_esperada_regresion_kwh": e_reg,
                "energia_esperada_lstm_kwh": e_lstm,
                "indice_regresion": e_real / e_reg if e_reg > 0 else None,
                "indice_lstm": e_real / e_lstm if e_lstm > 0 else None,
                "cobertura_datos": cobertura}

    # ------------------------------------------------------------- consultas
    def normalizar_inversor(self, inversor: str) -> str:
        inv = (inversor or "").strip().upper()
        if inv in self.cap:
            return inv
        if inversor in self.alias:
            return self.alias[inversor]
        raise InversorDesconocido("Inversor no reconocido. Use INV-01 a INV-22.")

    def estado(self, inversor: str, fecha: str | None = None, detalle: bool = False) -> dict:
        """Diagnóstico de rendimiento de un inversor en un día (por defecto, el último día)."""
        inv = self.normalizar_inversor(inversor)
        d = date.fromisoformat(fecha) if fecha else self.fechas[-1]
        r = self._indice(inv, d)
        if r is None:
            return {"inversor": inv, "fecha": d.isoformat(), "datos_suficientes": False,
                    "mensaje": "Sin registros suficientes de generación ese día (posible corte de datos)."}
        # Días consecutivos bajo el umbral (incluye el día consultado).
        consecutivos = 0
        dia = d
        while dia >= self.fechas[0]:
            ri = r if dia == d else self._indice(inv, dia)
            if ri is None or ri["indice_regresion"] is None or ri["indice_regresion"] >= self.umbral:
                break
            consecutivos += 1
            dia -= timedelta(days=1)
        # Comparación con el resto de la planta el mismo día: separa una falla del inversor de un
        # problema general (nubosidad mal medida, sensor, corte de red).
        pares = [self._indice(o, d) for o in self.inversores if o != inv]
        idx_pares = [p["indice_regresion"] for p in pares if p and p["indice_regresion"] is not None]
        mediana_planta = float(np.median(idx_pares)) if idx_pares else None
        anomalia = r["indice_regresion"] < self.umbral
        general = mediana_planta is not None and mediana_planta < self.umbral
        perdida = max(r["energia_esperada_regresion_kwh"] - r["energia_real_kwh"], 0.0)
        salida = {
            "inversor": inv, "fecha": d.isoformat(), "datos_suficientes": True,
            "modelo_decision": "Regresión lineal física (elegida en validación)",
            "energia_real_kwh": round(r["energia_real_kwh"], 1),
            "energia_esperada_kwh": round(r["energia_esperada_regresion_kwh"], 1),
            "energia_esperada_lstm_kwh": round(r["energia_esperada_lstm_kwh"], 1),
            "indice_rendimiento": round(r["indice_regresion"], 3),
            "indice_rendimiento_lstm": round(r["indice_lstm"], 3) if r["indice_lstm"] else None,
            "umbral": self.umbral, "anomalia": bool(anomalia),
            "modelos_coinciden": bool(anomalia == ((r["indice_lstm"] or 1) < self.umbral)),
            "dias_consecutivos_bajo_umbral": consecutivos,
            "perdida_estimada_kwh": round(perdida, 1) if anomalia else 0.0,
            "indice_mediana_planta": round(mediana_planta, 3) if mediana_planta is not None else None,
            "problema_general_planta": bool(general),
            "cobertura_datos": round(r["cobertura_datos"], 3),
        }
        if detalle:
            idxs = self._idx_dia(d)
            reg, lstm = self._esperado(idxs)
            cap = self.cap[inv]
            salida["curva"] = [{"hora": self.tiempos[i].strftime("%H:%M"),
                                "real_kw": None if math.isnan(self.ac[inv][i]) else round(float(self.ac[inv][i]), 1),
                                "esperado_kw": round(float(reg[k] * cap), 1),
                                "esperado_lstm_kw": round(float(lstm[k] * cap), 1)}
                               for k, i in enumerate(idxs)]
        return salida

    def tabla_diaria(self) -> list[dict]:
        """Índice de rendimiento de todos los inversores y días (se calcula una vez y se guarda)."""
        if self._cache_dias is None:
            filas = []
            for d in self.fechas:
                for inv in self.inversores:
                    r = self._indice(inv, d)
                    if r is None or r["indice_regresion"] is None:
                        continue
                    filas.append({"fecha": d.isoformat(), "inversor": inv,
                                  "indice": round(r["indice_regresion"], 3),
                                  "indice_lstm": round(r["indice_lstm"], 3) if r["indice_lstm"] else None,
                                  "energia_real_kwh": round(r["energia_real_kwh"], 1),
                                  "energia_esperada_kwh": round(r["energia_esperada_regresion_kwh"], 1),
                                  "anomalia": r["indice_regresion"] < self.umbral})
            self._cache_dias = filas
        return self._cache_dias

    def fecha_simulada(self, hoy: date | None = None) -> date:
        """Día del dataset que representa a "hoy" en la reproducción de la operación.

        Supuesto documentado: como los datos son de 2020, el monitoreo diario recorre de forma
        cíclica la semana de prueba (11 al 17 de junio), un día del dataset por cada día real.
        """
        hoy = hoy or date.today()
        semana = self.fechas[-7:]
        return semana[hoy.toordinal() % len(semana)]

    def alertas(self, fecha: str | None = None) -> dict:
        """Inversores con anomalía en un día, ordenados por energía no generada (monitoreo diario)."""
        d = date.fromisoformat(fecha) if fecha else self.fecha_simulada()
        if d not in self.fechas:
            raise FechaFueraDeRango(f"La fecha debe estar entre {self.fechas[0]} y {self.fechas[-1]}.")
        filas = [r for r in self.tabla_diaria() if r["fecha"] == d.isoformat() and r["anomalia"]]
        lista = sorted(({"inversor": r["inversor"], "indice_rendimiento": r["indice"],
                         "indice_rendimiento_lstm": r["indice_lstm"],
                         "perdida_kwh": round(max(r["energia_esperada_kwh"] - r["energia_real_kwh"], 0.0), 1)}
                        for r in filas), key=lambda x: -x["perdida_kwh"])
        return {"fecha": d.isoformat(), "fecha_simulada": fecha is None, "umbral": self.umbral,
                "inversores_evaluados": sum(1 for r in self.tabla_diaria() if r["fecha"] == d.isoformat()),
                "inversores_con_anomalia": len(lista),
                "perdida_total_kwh": round(sum(x["perdida_kwh"] for x in lista), 1), "anomalias": lista}

    def info(self) -> dict:
        return {"inversores": self.inversores, "fecha_inicio": self.fechas[0].isoformat(),
                "fecha_fin": self.fechas[-1].isoformat(), "umbral": self.umbral,
                "modelo_decision": self.cfg["modelo_produccion"],
                "metricas_deteccion_prueba": self.cfg["metricas_deteccion_prueba"],
                "metricas_error_prueba": self.cfg["metricas_error_prueba"]}
