"""Descarga textos ordenados del BCRA (PDF) y los convierte a Markdown limpio en ./data.

Uso:
    python prepare_data.py            # descarga lo que falte y regenera los .md
    python prepare_data.py --force    # vuelve a descargar los PDF

Los textos ordenados son la normativa vigente consolidada que publica el BCRA en
https://www.bcra.gob.ar (Sistema financiero > Marco legal y normativo > Textos ordenados).
"""
import argparse
import logging
import re
import sys
import urllib.request
from pathlib import Path

import pymupdf
import pymupdf4llm

logger = logging.getLogger("prepare_data")

URL_BASE = "https://www.bcra.gob.ar/archivos/Pdfs/Texord/{codigo}.pdf"

# código del PDF en el sitio del BCRA -> nombre del archivo .md
DOCUMENTOS = {
    "t-pusf": "proteccion_usuarios_servicios_financieros",
    "t-depaho": "depositos_ahorro_cuenta_sueldo",
    "t-tasint": "tasas_interes_operaciones_credito",
    "t-efemin": "efectivo_minimo",
}

DIR_RAW = Path("data/raw")
DIR_DATA = Path("data")

# Líneas del encabezado y pie de página que se repiten en cada hoja
PATRON_ENCABEZADO_PIE = re.compile(
    r"B\.C\.R\.A\.|Versión:|Vigencia:|Página\s+\d+|COMUNICACIÓN\s+“[A-C]”|TEXTO ORDENADO DE LAS NORMAS SOBRE"
)
PATRON_TITULO_NUMERADO = re.compile(r"^(\d+(?:\.\d+)+)\.\s+(.+)$")  # "2.3. Recaudos mínimos..."


# --- Descarga ----------------------------------------------------------------------------------

def descargar(codigo: str, forzar: bool) -> Path:
    destino = DIR_RAW / f"{codigo}.pdf"
    if destino.exists() and not forzar:
        logger.info("%s.pdf ya descargado", codigo)
        return destino
    url = URL_BASE.format(codigo=codigo)
    logger.info("Descargando %s", url)
    destino.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destino)
    return destino


# --- Análisis de la estructura del PDF ---------------------------------------------------------

def texto_plano(linea: str) -> str:
    """Quita la sintaxis Markdown/HTML que agrega pymupdf4llm para poder comparar el texto."""
    linea = re.sub(r"<br>|</?u>|\*\*|_", " ", linea)
    linea = re.sub(r"^[#\-\s|]+", "", linea).replace("|", " ")
    return re.sub(r"\s+", " ", linea).strip()


def leer_caratula(doc: pymupdf.Document) -> dict:
    """La página 1 tiene el título, la última comunicación incorporada y la fecha del texto ordenado."""
    texto = doc[0].get_text()
    titulo = " ".join(texto.split("-Última")[0].split())
    comunicacion = re.search(r"incorporada:\s*(“[A-C]”\s*\d+)", texto)
    fecha = re.search(r"Texto ordenado al\s*([\d/]+)", texto)
    return {
        "titulo": titulo.capitalize(),
        "ultima_comunicacion": comunicacion.group(1) if comunicacion else "s/d",
        "texto_ordenado_al": fecha.group(1) if fecha else "s/d",
    }


def rango_de_contenido(doc: pymupdf.Document) -> tuple[list[int], list[int]]:
    """Devuelve (páginas del índice, páginas con normativa).

    El contenido empieza después del índice y termina antes de la "Tabla de correlaciones"
    (que solo mapea cada punto a la comunicación que le dio origen).
    """
    indice, contenido = [], []
    for i in range(1, doc.page_count):
        texto = doc[i].get_text()
        if "NORMA DE ORIGEN" in texto:
            break
        if "Índice" in texto[:40]:
            indice.append(i)
        else:
            contenido.append(i)
    return indice, contenido


def titulos_de_secciones(doc: pymupdf.Document, paginas_indice: list[int]) -> dict[str, str]:
    """Lee el índice: {"2": "Derechos básicos de los usuarios de servicios financieros"}."""
    texto = "\n".join(doc[i].get_text() for i in paginas_indice)
    return {
        numero: titulo.strip().rstrip(".")
        for numero, titulo in re.findall(r"Sección\s+(\d+)\.\s*(.+)", texto)
    }


def leer_pie(texto_crudo: str) -> dict:
    """El pie de cada página indica la versión, la comunicación y la fecha de vigencia de esa hoja."""
    version = re.search(r"Versión:\s*(\S+?)\.?\s", texto_crudo)
    comunicacion = re.search(r"COMUNICACIÓN\s+“([A-C])”\s+(\d+)", texto_crudo)
    vigencia = re.search(r"Vigencia:\s*([\d/]{8,10})", texto_crudo)
    return {
        "version": version.group(1) if version else "s/d",
        "comunicacion": f"“{comunicacion.group(1)}” {comunicacion.group(2)}" if comunicacion else "s/d",
        "vigencia": vigencia.group(1) if vigencia else "s/d",
    }


def marcador(datos: dict) -> str:
    """Comentario HTML (invisible al renderizar el .md) que ingest.py convierte en metadatos."""
    return "<!-- " + " | ".join(f"{clave}: {valor}" for clave, valor in datos.items()) + " -->"


# --- Limpieza del Markdown ---------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Forma comparable de un texto: minúsculas y solo letras/números."""
    return re.sub(r"[\W_]+", "", texto.casefold())


def seccion_de_pagina(texto_crudo: str, secciones: dict[str, str]) -> str | None:
    """Detecta la sección del encabezado de la página comparando con los títulos del índice.

    Una referencia en el cuerpo ("...de la Sección 6., deberán...") no coincide con el título,
    así que no se confunde con el encabezado.
    """
    for numero, titulo in re.findall(r"Sección\s*(\d+)\.\s*([^\n]*)", texto_crudo):
        if numero in secciones and normalizar(titulo) == normalizar(secciones[numero]):
            return numero
    return None


def limpiar_pagina(md: str, titulo_doc: str, encabezado_seccion: str | None) -> list[str]:
    """Quita de una página las líneas del encabezado y del pie."""
    titulo_doc = normalizar(titulo_doc)
    encabezado_seccion = normalizar(encabezado_seccion or "")
    lineas = []
    for linea in md.splitlines():
        plano = texto_plano(linea)
        comparable = normalizar(plano)
        es_encabezado = (
            PATRON_ENCABEZADO_PIE.search(plano)
            or (len(comparable) > 6 and comparable in titulo_doc)
            or (encabezado_seccion and comparable == encabezado_seccion)
        )
        if plano and es_encabezado:
            continue
        lineas.append(linea)
    return lineas


def normalizar_linea(linea: str) -> str:
    """Ajusta títulos a la jerarquía de la norma y limpia restos de formato."""
    linea = linea.replace("-<br>", "").replace("<br>", " ")
    linea = re.sub(r"</?u>|<!-- (?:Start|End) of picture text -->", "", linea)

    # pymupdf4llm a veces convierte un título numerado en ítem de lista ("- 1.1.2. Sujetos obligados.")
    es_item_titulo = re.match(r"^\s*-\s+\d+\.\d+(?:\.\d+)?\.\s", linea) and len(linea) < 120
    if linea.startswith("#") or es_item_titulo:
        plano = texto_plano(linea)
        if m := PATRON_TITULO_NUMERADO.match(plano):
            # "2.3." -> ###, "2.3.1." -> ####, más profundo -> texto en negrita
            profundidad = m.group(1).count(".") + 1
            if profundidad == 2:
                return f"### {plano}"
            if profundidad == 3:
                return f"#### {plano}"
            return f"**{plano}**"
        return plano  # títulos sin numeración (p. ej. "Anexo") quedan como texto
    return linea.rstrip()


def eliminar_tablas_vacias(texto: str) -> str:
    """Quita tablas que quedaron solo con la fila separadora (|---|) tras limpiar encabezados."""
    texto = re.sub(r"(?m)^\|(?:\s*\|)*\s*$\n?", "", texto)        # filas sin contenido
    texto = re.sub(r"(?m)^\|(?:-{3,}\|)+\s*$\n(?!\|)", "", texto)  # separador huérfano
    return texto


def convertir(codigo: str, pdf: Path) -> Path:
    doc = pymupdf.open(pdf)
    meta = leer_caratula(doc)
    paginas_indice, paginas_contenido = rango_de_contenido(doc)
    secciones = titulos_de_secciones(doc, paginas_indice)
    logger.info(
        "%s: %d páginas de contenido, %d secciones, texto ordenado al %s",
        codigo, len(paginas_contenido), len(secciones), meta["texto_ordenado_al"],
    )

    paginas_md = pymupdf4llm.to_markdown(doc, pages=paginas_contenido, page_chunks=True, show_progress=False)

    salida = [
        marcador({"documento": meta["titulo"], "texto_ordenado_al": meta["texto_ordenado_al"],
                  "ultima_comunicacion": meta["ultima_comunicacion"], "fuente": URL_BASE.format(codigo=codigo)}),
        f"# {meta['titulo']}",
        "",
        f"Texto ordenado del Banco Central de la República Argentina (BCRA) al {meta['texto_ordenado_al']}. "
        f"Última comunicación incorporada: {meta['ultima_comunicacion']}. "
        f"Fuente: {URL_BASE.format(codigo=codigo)}",
        "",
    ]
    seccion_actual = None
    for numero_pagina, pagina in zip(paginas_contenido, paginas_md):
        texto_crudo = doc[numero_pagina].get_text()
        seccion = seccion_de_pagina(texto_crudo, secciones)
        encabezado = f"Sección {seccion}. {secciones[seccion]}" if seccion else None
        lineas = limpiar_pagina(pagina["text"], meta["titulo"], encabezado)
        # El pie se descarta del texto, pero sus datos se conservan como metadatos de la página
        salida += ["", marcador({"pagina": numero_pagina + 1, **leer_pie(texto_crudo)}), ""]
        if seccion and seccion != seccion_actual:
            seccion_actual = seccion
            salida += [f"## Sección {seccion}. {secciones.get(seccion, '')}".rstrip(), ""]
        salida += [normalizar_linea(linea) for linea in lineas]

    texto = "\n".join(salida)
    texto = eliminar_tablas_vacias(texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip() + "\n"

    destino = DIR_DATA / f"{DOCUMENTOS[codigo]}.md"
    destino.write_text(texto, encoding="utf-8")
    logger.info("-> %s (%d caracteres)", destino, len(texto))
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="volver a descargar los PDF")
    args = parser.parse_args()

    DIR_DATA.mkdir(exist_ok=True)
    for codigo in DOCUMENTOS:
        pdf = descargar(codigo, args.force)
        convertir(codigo, pdf)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    main()
