"""Ingesta: lee los documentos de ./data, los fragmenta y los guarda en ChromaDB (./vectorstore).

Uso:
    python ingest.py            # indexa solo si los documentos o la configuración cambiaron
    python ingest.py --force    # borra la colección y vuelve a indexar todo
    python ingest.py --preview  # muestra cómo quedan los chunks sin generar embeddings
"""
import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("ingest")

DIR_DATA = Path("data")
DIR_VECTORSTORE = Path("vectorstore")
COLECCION = "normativa_bcra"

# El MISMO modelo se usa para indexar y para consultar: se guarda en la colección y rag.py lo verifica
MODELO_EMBEDDINGS = "text-embedding-3-small"
# text-embedding-3-small usa el tokenizer cl100k_base: medimos los chunks con ese mismo tokenizer
ENCODING = "cl100k_base"
CHUNK_SIZE = 500     # tokens
CHUNK_OVERLAP = 50   # tokens

PATRON_MARCADOR = re.compile(r"<!--\s*((?:pagina|documento):.*?)\s*-->")  # marcadores de prepare_data.py
PATRON_TITULO = re.compile(r"^(#{2,4})\s+(.*)$")


# --- Lectura de los documentos -----------------------------------------------------------------

def leer_marcador(linea: str) -> dict | None:
    """'<!-- pagina: 7 | vigencia: 23/02/2024 -->' -> {'pagina': '7', 'vigencia': '23/02/2024'}"""
    m = PATRON_MARCADOR.fullmatch(linea.strip())
    if not m:
        return None
    pares = (parte.split(":", 1) for parte in m.group(1).split(" | "))
    return {clave.strip(): valor.strip() for clave, valor in pares}


def dividir_en_bloques(texto: str) -> tuple[dict, list[dict]]:
    """Divide un documento en bloques por sección (##) y punto (###).

    Cada bloque conserva los títulos que lo contienen (para los metadatos) y los marcadores de
    página que tiene adentro (para saber en qué página del PDF está cada fragmento).
    """
    meta_doc: dict = {}
    bloques: list[dict] = []
    titulos = {"seccion": "", "punto": ""}
    actual = {"seccion": "", "punto": "", "lineas": []}

    def cerrar_bloque():
        if any(linea.strip() and not PATRON_MARCADOR.fullmatch(linea.strip()) for linea in actual["lineas"]):
            bloques.append(dict(actual))

    for linea in texto.splitlines():
        marcador = leer_marcador(linea)
        if marcador and "documento" in marcador:
            meta_doc = marcador
            continue
        if linea.startswith("# "):  # título del documento: ya está en los metadatos
            continue

        titulo = PATRON_TITULO.match(linea)
        if titulo and len(titulo.group(1)) in (2, 3):
            cerrar_bloque()
            if len(titulo.group(1)) == 2:
                titulos = {"seccion": titulo.group(2), "punto": ""}
            else:
                titulos["punto"] = titulo.group(2)
            # Si el bloque anterior terminaba con un marcador de página, lo arrastramos al nuevo
            arrastre = [l for l in actual["lineas"][-3:] if PATRON_MARCADOR.fullmatch(l.strip())]
            actual = {**titulos, "lineas": arrastre[-1:] + [linea]}
        else:
            actual["lineas"].append(linea)
    cerrar_bloque()
    return meta_doc, bloques


def crear_chunks(archivo: Path, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    meta_doc, bloques = dividir_en_bloques(archivo.read_text(encoding="utf-8"))
    documento = meta_doc.get("documento", archivo.stem)
    pagina = {"pagina": "s/d", "vigencia": "s/d", "comunicacion": "s/d"}
    chunks = []

    for bloque in bloques:
        for fragmento in splitter.split_text("\n".join(bloque["lineas"])):
            # Página donde empieza el fragmento: la última vista antes de él (o la primera que contiene
            # si arranca con un marcador). Página final: el último marcador dentro del fragmento.
            marcadores = [leer_marcador(m.group(0)) for m in PATRON_MARCADOR.finditer(fragmento)]
            if fragmento.lstrip().startswith("<!--") and marcadores:
                pagina = marcadores[0]
            pagina_inicio = pagina
            if marcadores:
                pagina = marcadores[-1]

            texto = PATRON_MARCADOR.sub("", fragmento)
            texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
            if len(texto) < 40:  # restos sin contenido (por ejemplo, un título suelto)
                continue

            # "Contextual chunk header": el encabezado viaja con el texto, así el embedding y el LLM
            # saben a qué norma y punto pertenece el fragmento aunque el texto no lo diga.
            encabezado = f"[{documento} | {bloque['seccion']} | {bloque['punto']}]".replace(" |  |", " |")
            chunks.append(Document(
                page_content=f"{encabezado}\n{texto}",
                metadata={
                    "fuente": archivo.name,
                    "documento": documento,
                    "seccion": bloque["seccion"],
                    "punto": bloque["punto"],
                    "pagina_inicio": int(pagina_inicio["pagina"]) if pagina_inicio["pagina"].isdigit() else -1,
                    "pagina_fin": int(pagina["pagina"]) if pagina["pagina"].isdigit() else -1,
                    "vigencia": pagina_inicio.get("vigencia", "s/d"),
                    "comunicacion": pagina_inicio.get("comunicacion", "s/d"),
                    "texto_ordenado_al": meta_doc.get("texto_ordenado_al", "s/d"),
                    "url": meta_doc.get("fuente", ""),
                },
            ))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks


def cargar_chunks() -> list[Document]:
    archivos = sorted([*DIR_DATA.glob("*.md"), *DIR_DATA.glob("*.txt")])
    if not archivos:
        raise FileNotFoundError(f"No hay documentos en {DIR_DATA}/. Corré primero: python prepare_data.py")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for archivo in archivos:
        chunks_archivo = crear_chunks(archivo, splitter)
        logger.info("%s: %d chunks", archivo.name, len(chunks_archivo))
        chunks += chunks_archivo
    return chunks


# --- Persistencia ------------------------------------------------------------------------------

def huella_de_datos() -> str:
    """Hash del contenido de ./data y de la configuración de chunking/embeddings.

    Si cambia un documento o un parámetro, cambia la huella y hay que volver a indexar.
    """
    h = hashlib.sha256()
    h.update(json.dumps([MODELO_EMBEDDINGS, ENCODING, CHUNK_SIZE, CHUNK_OVERLAP]).encode())
    for archivo in sorted([*DIR_DATA.glob("*.md"), *DIR_DATA.glob("*.txt")]):
        h.update(archivo.name.encode())
        h.update(archivo.read_bytes())
    return h.hexdigest()[:16]


def crear_embeddings() -> OpenAIEmbeddings:
    load_dotenv()
    return OpenAIEmbeddings(model=MODELO_EMBEDDINGS)


def abrir_vectorstore() -> Chroma:
    """Abre la colección existente verificando que se haya indexado con el mismo modelo de embeddings."""
    cliente = chromadb.PersistentClient(path=str(DIR_VECTORSTORE))
    try:
        coleccion = cliente.get_collection(COLECCION)
    except Exception as e:
        raise RuntimeError("No existe el índice. Corré primero: python ingest.py") from e

    modelo_indexado = (coleccion.metadata or {}).get("embedding_model")
    if modelo_indexado != MODELO_EMBEDDINGS:
        # Error #1 de la consigna: consultar con un modelo distinto al usado para indexar
        raise RuntimeError(
            f"El índice se creó con '{modelo_indexado}' pero se está consultando con "
            f"'{MODELO_EMBEDDINGS}'. Volvé a indexar con: python ingest.py --force"
        )
    return Chroma(client=cliente, collection_name=COLECCION, embedding_function=crear_embeddings())


def indexar(forzar: bool = False) -> Chroma:
    """Crea el índice si no existe o si los documentos cambiaron; si no, reutiliza el existente."""
    huella = huella_de_datos()
    cliente = chromadb.PersistentClient(path=str(DIR_VECTORSTORE))

    existentes = {c.name for c in cliente.list_collections()}
    if COLECCION in existentes:
        coleccion = cliente.get_collection(COLECCION)
        meta = coleccion.metadata or {}
        if not forzar and meta.get("huella") == huella and coleccion.count() > 0:
            logger.info("Índice existente y actualizado (%d chunks, huella %s) -> no se reindexa",
                        coleccion.count(), huella)
            return abrir_vectorstore()
        motivo = "--force" if forzar else "los documentos o la configuración cambiaron"
        logger.info("Reindexando (%s)", motivo)
        cliente.delete_collection(COLECCION)

    chunks = cargar_chunks()
    vectorstore = Chroma(
        client=cliente,
        collection_name=COLECCION,
        embedding_function=crear_embeddings(),
        collection_metadata={
            "embedding_model": MODELO_EMBEDDINGS,
            "huella": huella,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        # Distancia coseno: la similitud no depende del largo del texto, solo de su "dirección"
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    logger.info("Generando embeddings de %d chunks con %s...", len(chunks), MODELO_EMBEDDINGS)
    ids = [f"{c.metadata['fuente']}-{c.metadata['chunk']:04d}" for c in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    logger.info("Índice creado en %s/ (%d chunks)", DIR_VECTORSTORE, vectorstore._collection.count())
    return vectorstore


def mostrar_preview() -> None:
    import tiktoken

    enc = tiktoken.get_encoding(ENCODING)
    chunks = cargar_chunks()
    tamanos = sorted(len(enc.encode(c.page_content)) for c in chunks)
    print(f"\n{len(chunks)} chunks | tokens: mín {tamanos[0]}, mediana {tamanos[len(tamanos) // 2]}, "
          f"máx {tamanos[-1]}")
    ejemplo = next(c for c in chunks if "1,8 milímetros" in c.page_content)
    print("\n--- Ejemplo de chunk ---")
    print(ejemplo.page_content[:700], "...")
    print("\nMetadatos:", json.dumps(ejemplo.metadata, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="reindexar aunque nada haya cambiado")
    parser.add_argument("--preview", action="store_true", help="mostrar los chunks sin indexar")
    args = parser.parse_args()

    if args.preview:
        mostrar_preview()
    else:
        indexar(forzar=args.force)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
