"""Construye rag/indice_fragmentos.json a partir de los manuales PDF de rag/documentos/.

Uso:  python rag/construir_indice.py

Cada página se limpia y se divide en fragmentos de unos 1.500 caracteres con 200 de traslape,
para que una idea que cruza el corte no se pierda. Cada fragmento guarda la fuente, el número de
página del PDF y el último título de sección visto, que luego se usan para citar.
Las páginas de portada, índice y referencias se omiten porque no contienen procedimientos.
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

RAIZ = Path(__file__).resolve().parent
FUENTES = {
    "nrel": {
        "archivo": "73822.pdf",
        "titulo": "Best Practices for Operation and Maintenance of Photovoltaic and Energy Storage Systems; 3rd Edition",
        "cita_corta": "NREL (2018), PV O&M Best Practices",
        "autor": "NREL, Sandia National Laboratories, SunSpec Alliance y SunShot National Laboratory Multiyear Partnership (SuNLaMP) PV O&M Best Practices Working Group",
        "url": "https://docs.nlr.gov/docs/fy19osti/73822.pdf",
        "omitir_paginas": [],
    },
}
TAM, TRASLAPE = 1500, 200
TITULO = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-Z][A-Za-z0-9 ,&/()\-–]{3,80})\s*$")


def limpiar(texto: str) -> str:
    texto = texto.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    texto = re.sub(r"-\n(\w)", r"\1", texto)             # palabras cortadas al final de línea
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto.strip()


def es_pagina_util(texto: str) -> bool:
    t = texto.lower()
    if len(texto) < 300:
        return False
    if "table of contents" in t[:400] or t.count(" ......") > 5 or t.count("....") > 8:
        return False
    if t.lstrip().startswith("references") or "et al." in t and t.count("http") > 6:
        return False
    return True


def main():
    fragmentos, fuentes = [], {}
    for clave, f in FUENTES.items():
        ruta = RAIZ / "documentos" / f["archivo"]
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta}. Descargar desde {f['url']}")
        if ruta.read_bytes()[:5] != b"%PDF-":
            raise SystemExit(f"{ruta} no es un PDF válido (la descarga pudo fallar).")
        lector = PdfReader(str(ruta))
        fuentes[clave] = {k: v for k, v in f.items() if k not in ("archivo", "omitir_paginas")}
        fuentes[clave]["paginas"] = len(lector.pages)
        seccion = "Introducción"
        for n, pagina in enumerate(lector.pages, start=1):
            if n in f["omitir_paginas"]:
                continue
            texto = limpiar(pagina.extract_text() or "")
            for linea in texto.splitlines():
                m = TITULO.match(linea)
                if m:
                    seccion = f"{m.group(1)} {m.group(2).strip()}"
                    break
            if not es_pagina_util(texto):
                continue
            plano = re.sub(r"\s*\n\s*", " ", texto)
            inicio = 0
            while inicio < len(plano):
                trozo = plano[inicio:inicio + TAM]
                if len(trozo) > 200:
                    fragmentos.append({"id": f"{clave}-p{n}-{inicio // (TAM - TRASLAPE)}", "fuente": clave,
                                       "pagina": n, "seccion": seccion, "texto": trozo})
                inicio += TAM - TRASLAPE
    salida = {"fuentes": fuentes, "parametros": {"tam": TAM, "traslape": TRASLAPE}, "fragmentos": fragmentos}
    (RAIZ / "indice_fragmentos.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    emb = RAIZ / "embeddings.npy"
    if emb.exists():
        emb.unlink()                                      # se recalculan con el índice nuevo
    print(f"{len(fragmentos)} fragmentos de {sum(v['paginas'] for v in fuentes.values())} páginas")


if __name__ == "__main__":
    main()
