"""Pruebas del sistema RAG: pregunta con respuesta, pregunta trampa y trampas sutiles.

Uso:
    python test_rag.py
    python test_rag.py "¿Tu pregunta?"     # consulta libre
"""
import asyncio
import logging
import sys

from rag import get_rag_response
from schemas import RespuestaRAG

PRUEBAS = [
    # (tipo, pregunta, ¿debería encontrar la respuesta?)
    ("Respuesta en los documentos", "¿Qué tamaño mínimo de letra deben tener los contratos financieros?", True),
    ("Respuesta en los documentos", "¿Me pueden cobrar por abrir y mantener una caja de ahorros en pesos?", True),
    ("Trampa (fuera de tema)", "¿Quién ganó el mundial de fútbol 2022?", False),
    ("Trampa sutil (tema presente, dato ausente)", "¿Cuál es la tasa de interés de un plazo fijo hoy?", False),
    ("Trampa sutil (tema presente, dato ausente)", "¿Cuántos dólares puedo comprar por mes como persona humana?", False),
]


def mostrar(r: RespuestaRAG) -> None:
    print(f"\n> {r.pregunta}")
    print(f"\n{r.respuesta}\n")
    origen = "umbral, sin LLM" if r.descartada_por_umbral else "LLM"
    print(f"encontrada={r.encontrada} | fragmentos recuperados={r.fragmentos_recuperados} | respondido por: {origen}")
    for ref in r.referencias:
        print(f"  📎 {ref.documento} — {ref.punto or ref.seccion} (pág. {ref.paginas}, vigente desde "
              f"{ref.vigencia}, relevancia {ref.relevancia:.2f})")


async def main() -> None:
    if len(sys.argv) > 1:
        mostrar(await get_rag_response(" ".join(sys.argv[1:])))
        return

    resultados = []
    for tipo, pregunta, esperado in PRUEBAS:
        print("\n" + "=" * 100 + f"\n{tipo}")
        r = await get_rag_response(pregunta)
        mostrar(r)
        resultados.append((tipo, pregunta, esperado, r.encontrada))

    print("\n" + "=" * 100 + "\nResumen")
    for tipo, pregunta, esperado, obtenido in resultados:
        estado = "✅" if esperado == obtenido else "❌"
        print(f"{estado} {tipo}: esperado encontrada={esperado}, obtenido={obtenido}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for nombre in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(nombre).setLevel(logging.WARNING)
    asyncio.run(main())
