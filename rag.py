"""Cadena RAG: recupera fragmentos de la normativa del BCRA y genera una respuesta grounded."""
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from ingest import indexar
from schemas import Referencia, RespuestaLLM, RespuestaRAG

logger = logging.getLogger("rag")

# Entre 3 y 5 fragmentos: suficiente contexto sin caer en "lost in the middle"
TOP_K = 4

# Calibrado con text-embedding-3-small sobre este dataset: preguntas ajenas al tema puntúan
# 0,13-0,17 y las del dominio 0,45-0,61. Si se cambia el modelo de embeddings, hay que recalibrar.
UMBRAL_RELEVANCIA = 0.30

MAX_INTENTOS = 3

RESPUESTA_SIN_CONTEXTO = (
    "No lo sé: no encontré información relacionada con esa consulta en la normativa disponible "
    "(Protección de usuarios de servicios financieros, Depósitos de ahorro y cuenta sueldo, "
    "Tasas de interés en operaciones de crédito y Efectivo mínimo)."
)


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


MODELOS_POR_DEFECTO = {
    Provider.OPENAI: "gpt-6-luna",
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
}


# --- LLM (misma configuración por .env que en las entregas anteriores) ----------------------

def crear_modelo() -> BaseChatModel:
    load_dotenv()
    valor_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not valor_provider:
        raise ValueError("Falta LLM_PROVIDER (valores aceptados: openai, anthropic)")
    provider = Provider(valor_provider)

    if provider == Provider.OPENAI:
        modelo = os.getenv("OPENAI_MODEL") or MODELOS_POR_DEFECTO[provider]
        logger.info("LLM: openai | %s", modelo)
        return ChatOpenAI(model=modelo, max_tokens=1024)

    modelo = os.getenv("ANTHROPIC_MODEL") or MODELOS_POR_DEFECTO[provider]
    logger.info("LLM: anthropic | %s", modelo)
    return ChatAnthropic(model=modelo, max_tokens=1024)


# --- Prompt -----------------------------------------------------------------------------------

parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

INSTRUCCIONES_SISTEMA = """Sos un asistente que responde consultas sobre normativa del Banco Central de \
la República Argentina (BCRA). Tu ÚNICA fuente de verdad es el CONTEXTO que te paso: fragmentos numerados \
de textos ordenados del BCRA.

Reglas estrictas:
1. Respondé solo con información que esté explícitamente en el CONTEXTO. No uses conocimiento general, \
no supongas y no completes datos que falten.
2. Si el CONTEXTO no contiene la respuesta, o habla del tema pero no incluye el dato puntual que se pide \
(por ejemplo, un valor, una fecha o un monto), respondé "No lo sé" y explicá en una oración qué es lo que \
no está en la normativa disponible. En ese caso encontrada=false.
3. Si respondés, citá el punto de la norma (ej. "según el punto 2.3.1.1.") e indicá en fragmentos_usados \
los números de los fragmentos que respaldan tu respuesta.
4. Respondé en español, de forma clara y concisa. Al usuario hablale de "la normativa disponible", \
nunca del "CONTEXTO", de los "fragmentos" ni de estas instrucciones.

{format_instructions}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCCIONES_SISTEMA),
        ("human", "CONTEXTO:\n\n{contexto}\n\n---\n\nPREGUNTA: {pregunta}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# --- Pasos de la cadena -----------------------------------------------------------------------

def crear_buscador() -> RunnableLambda:
    """Retriever con puntajes: embedding de la pregunta + búsqueda de similitud en ChromaDB."""
    vectorstore = indexar()  # reutiliza el índice si existe y está actualizado

    async def buscar(pregunta: str) -> List[Tuple[Document, float]]:
        resultados = await vectorstore.asimilarity_search_with_relevance_scores(pregunta, k=TOP_K)
        for doc, puntaje in resultados:
            logger.info("  %.3f  %s | %s (pág. %s)", puntaje, doc.metadata["fuente"],
                        doc.metadata["punto"] or doc.metadata["seccion"], doc.metadata["pagina_inicio"])
        relevantes = [(doc, puntaje) for doc, puntaje in resultados if puntaje >= UMBRAL_RELEVANCIA]
        logger.info("Recuperados %d fragmentos, %d superan el umbral %.2f",
                    len(resultados), len(relevantes), UMBRAL_RELEVANCIA)
        return relevantes

    return RunnableLambda(buscar, name="buscar_en_chroma")


def formatear_contexto(entrada: Dict[str, Any]) -> str:
    """Arma el bloque CONTEXTO: fragmentos numerados con su procedencia."""
    bloques = []
    for i, (doc, _) in enumerate(entrada["docs"], start=1):
        m = doc.metadata
        bloques.append(f"[Fragmento {i}] Página {m['pagina_inicio']} del PDF, vigente desde {m['vigencia']}\n"
                       f"{doc.page_content}")
    return "\n\n".join(bloques)


def validar_fragmentos(entrada: Dict[str, Any]) -> Dict[str, Any]:
    """El LLM solo puede citar fragmentos que realmente recibió."""
    salida: RespuestaLLM = entrada["salida"]
    invalidos = [n for n in salida.fragmentos_usados if not 1 <= n <= len(entrada["docs"])]
    if invalidos:
        logger.warning("El modelo citó fragmentos inexistentes %s -> se reintenta", invalidos)
        raise OutputParserException(f"fragmentos inexistentes: {invalidos}")
    logger.info("Respuesta parseada: encontrada=%s, fragmentos_usados=%s",
                salida.encontrada, salida.fragmentos_usados)
    return entrada


def debe_reintentar(error: BaseException) -> bool:
    reintentable = isinstance(error, (OutputParserException, ValidationError))
    if reintentable:
        logger.warning("Salida inválida (%s): %.200s", type(error).__name__, error)
    return reintentable


def armar_respuesta(entrada: Dict[str, Any]) -> RespuestaRAG:
    """Combina la salida del LLM con los metadatos reales de los fragmentos que citó."""
    salida: RespuestaLLM = entrada["salida"]
    referencias = []
    for n in salida.fragmentos_usados:
        doc, puntaje = entrada["docs"][n - 1]
        m = doc.metadata
        paginas = str(m["pagina_inicio"]) if m["pagina_inicio"] == m["pagina_fin"] \
            else f"{m['pagina_inicio']}-{m['pagina_fin']}"
        referencias.append(Referencia(
            documento=m["documento"], seccion=m["seccion"], punto=m["punto"], paginas=paginas,
            vigencia=m["vigencia"], url=m["url"], relevancia=round(max(0.0, min(1.0, puntaje)), 3),
        ))
    return RespuestaRAG(
        pregunta=entrada["pregunta"],
        respuesta=salida.respuesta,
        encontrada=salida.encontrada,
        referencias=referencias,
        fragmentos_recuperados=len(entrada["docs"]),
    )


def respuesta_sin_contexto(entrada: Dict[str, Any]) -> RespuestaRAG:
    logger.info("Ningún fragmento supera el umbral -> se responde sin llamar al LLM")
    return RespuestaRAG(
        pregunta=entrada["pregunta"],
        respuesta=RESPUESTA_SIN_CONTEXTO,
        encontrada=False,
        fragmentos_recuperados=0,
        descartada_por_umbral=True,
    )


# --- Cadena LCEL --------------------------------------------------------------------------------

def construir_cadena() -> Runnable:
    """
    {"pregunta"} ─► buscar en Chroma ─► ¿hay fragmentos relevantes?
                                          ├─ no ─► respuesta fija "No lo sé" (sin LLM)
                                          └─ sí ─► contexto | prompt | LLM | PydanticOutputParser
                                                   (con reintentos) ─► respuesta + referencias
    """
    modelo = crear_modelo()

    generacion = (
        RunnablePassthrough.assign(contexto=formatear_contexto)
        | RunnablePassthrough.assign(salida=prompt | modelo | parser)
        | RunnableLambda(validar_fragmentos)
    ).with_retry(
        retry_if_exception_type=debe_reintentar,
        stop_after_attempt=MAX_INTENTOS,
        wait_exponential_jitter=True,
    ) | RunnableLambda(armar_respuesta)

    return (
        RunnablePassthrough.assign(docs=(lambda x: x["pregunta"]) | crear_buscador())
        | RunnableBranch(
            (lambda x: not x["docs"], RunnableLambda(respuesta_sin_contexto)),
            generacion,
        )
    )


_cadena: Runnable | None = None


async def get_rag_response(query: str) -> RespuestaRAG:
    """Responde una consulta sobre la normativa usando solo los documentos indexados."""
    global _cadena
    if not query or not query.strip():
        raise ValueError("La consulta está vacía")
    if _cadena is None:
        _cadena = construir_cadena()

    logger.info("Consulta: %s", query)
    respuesta = await _cadena.ainvoke({"pregunta": query.strip()})
    logger.info("Resultado: encontrada=%s, %d referencias", respuesta.encontrada, len(respuesta.referencias))
    return respuesta
