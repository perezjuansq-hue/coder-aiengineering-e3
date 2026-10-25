# coder-aiengineering-e3

**Pre-entrega 3: Sistema de recuperación semántica local (RAG) sobre normativa del BCRA.** Responde consultas sobre normativa bancaria argentina usando **solo** los textos ordenados oficiales del Banco Central (BCRA), indexados en una base vectorial local (**ChromaDB**). Cada respuesta se devuelve como un objeto **Pydantic** con referencias verificables (documento, punto de la norma, página del PDF y vigencia). Si la información no está en los documentos, el sistema responde **"No lo sé"**.

```
INGESTA (una vez)
  PDF del BCRA ─► prepare_data.py ─► data/*.md ─► ingest.py ─► chunks ─► embeddings ─► ChromaDB (./vectorstore)
                  (limpieza)                      (500 tokens / 50 de overlap)   (text-embedding-3-small)

CONSULTA (get_rag_response)
  pregunta ─► embedding ─► top-4 en Chroma ─► ¿algún fragmento con relevancia ≥ 0,30?
                                                ├─ no ─► "No lo sé" (sin llamar al LLM)
                                                └─ sí ─► prompt grounded | LLM | PydanticOutputParser ─► respuesta + referencias
```

## Dataset

Cuatro **textos ordenados** del BCRA (normativa vigente consolidada), descargados de [bcra.gob.ar](https://www.bcra.gob.ar):

| Archivo en `data/` | Norma | Tokens | Chunks |
|---|---|---|---|
| `depositos_ahorro_cuenta_sueldo.md` | [Depósitos de ahorro, cuenta sueldo y especiales](https://www.bcra.gob.ar/archivos/Pdfs/Texord/t-depaho.pdf) | 50.987 | 156 |
| `proteccion_usuarios_servicios_financieros.md` | [Protección de los usuarios de servicios financieros](https://www.bcra.gob.ar/archivos/Pdfs/Texord/t-pusf.pdf) | 18.865 | 57 |
| `efectivo_minimo.md` | [Efectivo mínimo](https://www.bcra.gob.ar/archivos/Pdfs/Texord/t-efemin.pdf) | 17.843 | 61 |
| `tasas_interes_operaciones_credito.md` | [Tasas de interés en las operaciones de crédito](https://www.bcra.gob.ar/archivos/Pdfs/Texord/t-tasint.pdf) | 7.588 | 41 |

Los `.md` ya están en el repo. `prepare_data.py` permite regenerarlos desde los PDF originales.

## Archivos

| Archivo | Qué hace |
|---|---|
| `prepare_data.py` | Descarga los PDF del BCRA y los convierte a Markdown limpio en `data/`. |
| `ingest.py` | **Ingesta:** fragmenta los documentos, genera los embeddings y los persiste en ChromaDB. |
| `schemas.py` | Modelos Pydantic: `RespuestaLLM` (salida del LLM), `Referencia` y `RespuestaRAG` (resultado final). |
| `rag.py` | **Retriever + generación grounded:** cadena LCEL y la función asíncrona `get_rag_response(query)`. |
| `test_rag.py` | Pruebas: preguntas con respuesta, una pregunta trampa y dos trampas sutiles. También acepta consultas libres. |

## Cómo funciona

### 1. Preparación del dataset (`prepare_data.py`)

Los PDF tienen ruido que perjudica la búsqueda: carátula, índice, un pie de página repetido en cada hoja ("B.C.R.A. … Versión … Página N"), palabras cortadas con guion y una "Tabla de correlaciones" final que solo lista comunicaciones. El script:

- Extrae el texto con **pymupdf4llm**, que une párrafos y palabras cortadas y conserva las **tablas en Markdown** (por ejemplo, las tasas de encaje de Efectivo mínimo).
- Descarta la carátula, el índice y las tablas de correlaciones, y elimina el encabezado y el pie de cada página.
- Arma la jerarquía de la norma: `## Sección`, `### 2.3.` y `#### 2.3.1.`. Los títulos de sección se toman del índice del propio PDF.
- **Conserva como metadatos lo que descarta del texto.** Antes de cada página escribe un comentario invisible (`<!-- pagina: 7 | version: 8a | comunicacion: "A" 7744 | vigencia: 28/02/2023 -->`) con los datos del pie, que después usa `ingest.py`.

Así se pasa de unos 167.000 tokens en los PDF a 95.000 de normativa pura.

### 2. Ingesta (`ingest.py`)

- **Chunking por estructura:** primero se corta por sección (`##`) y punto (`###`), para que un fragmento no mezcle dos temas. Si un bloque supera los 500 tokens, se vuelve a cortar con `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)`. Los tokens se cuentan con `cl100k_base`, el mismo tokenizer del modelo de embeddings.
- **Encabezado de contexto:** cada chunk empieza con `[Documento | Sección | Punto]`. De esta manera, tanto el embedding como el LLM saben a qué norma pertenece el fragmento aunque el texto no lo diga.
- **Metadatos por chunk:** `documento`, `seccion`, `punto`, `pagina_inicio`, `pagina_fin`, `vigencia`, `comunicacion`, `texto_ordenado_al` y `url`.
- **Embeddings:** `text-embedding-3-small` de OpenAI (1536 dimensiones, hasta 8.191 tokens, de modo que los chunks nunca se truncan). La colección usa **distancia coseno**.
- **Persistencia:** ChromaDB embebido en `./vectorstore`. Funciona como SQLite: no requiere servidor ni cuenta. La colección guarda una **huella** (un hash del contenido de `data/` y de la configuración de chunking y embeddings). Si nada cambió, no se reindexa. Si cambió un documento o un parámetro, se reindexa automáticamente.
- **Protección contra embeddings no coincidentes:** el nombre del modelo de embeddings queda guardado en la colección. Si se consulta con otro modelo, `abrir_vectorstore()` corta con un error explícito en lugar de devolver resultados al azar.

### 3. Recuperación y generación (`rag.py`)

La función `get_rag_response(query)` ejecuta una sola cadena LCEL con `.ainvoke()`:

```python
generacion = (
    RunnablePassthrough.assign(contexto=formatear_contexto)
    | RunnablePassthrough.assign(salida=prompt | modelo | parser)   # parser = PydanticOutputParser(RespuestaLLM)
    | RunnableLambda(validar_fragmentos)
).with_retry(retry_if_exception_type=debe_reintentar, stop_after_attempt=3) | RunnableLambda(armar_respuesta)

cadena = (
    RunnablePassthrough.assign(docs=(lambda x: x["pregunta"]) | buscador)      # top-4 en Chroma, con puntaje
    | RunnableBranch((lambda x: not x["docs"], RunnableLambda(respuesta_sin_contexto)), generacion)
)
```

**Doble barrera contra las alucinaciones:**

1. **Umbral de relevancia (0,30), sin LLM.** El retriever devuelve los 4 fragmentos más parecidos con su similitud coseno. Los que no superan 0,30 se descartan. Si no queda ninguno, se responde "No lo sé" **sin llamar al LLM**: es más barato, más rápido y no hay riesgo de que el modelo invente. El valor se calibró con preguntas reales: las preguntas ajenas al tema puntúan entre 0,13 y 0,17, y las del dominio entre 0,45 y 0,63.
2. **Prompt de grounding.** Hay preguntas trampa que **pasan el umbral** porque el tema sí está en los documentos, aunque el dato no (por ejemplo, "¿tasa de un plazo fijo hoy?" puntúa 0,60). Para esos casos, el prompt obliga al modelo a responder solo con el contexto y a decir "No lo sé" si falta el dato puntual.

**Referencias verificables:** el LLM no escribe las citas. Solo indica *qué fragmentos numerados* usó (`fragmentos_usados: [1, 3]`), y el código arma cada `Referencia` con los metadatos reales de esos chunks. Además se valida que:
- si `encontrada=true`, haya al menos un fragmento citado;
- los números citados existan entre los fragmentos que recibió el modelo.

Si alguna de estas reglas no se cumple, o el JSON no es válido, la salida se rechaza y `.with_retry()` vuelve a ejecutar la generación (hasta 3 intentos).

**LLM:** se configura con el `.env`, igual que en las entregas anteriores (`LLM_PROVIDER=openai` o `anthropic`). El modelo de embeddings es independiente del LLM: los dos pasos que generan vectores (indexar y consultar) usan siempre `text-embedding-3-small`, y el LLM solo recibe texto.

## Requisitos

- Python **3.12**
- API key de **OpenAI** (obligatoria: se usa para los embeddings). API key de Anthropic solo si el LLM elegido es Claude.

## Instalación

```bash
git clone https://github.com/perezjuansq-hue/coder-aiengineering-e3.git
cd coder-aiengineering-e3

py -3.12 -m venv .venv          # Windows
# python3.12 -m venv .venv      # macOS / Linux

source .venv/Scripts/activate   # Windows (Git Bash)
# .venv\Scripts\Activate.ps1    # Windows (PowerShell)
# source .venv/bin/activate     # macOS / Linux

python -m pip install -r requirements.txt
cp .env.example .env            # y completar las API keys
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | Sí | Embeddings, y también el chat si `LLM_PROVIDER=openai`. |
| `LLM_PROVIDER` | Sí | `openai` o `anthropic`: el modelo que redacta las respuestas. |
| `ANTHROPIC_API_KEY` | Si `LLM_PROVIDER=anthropic` | API key de Anthropic. |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | No | Por defecto, `gpt-6-luna` / `claude-haiku-4-5-20251001`. |

> El `.env` está en `.gitignore` y no se sube al repositorio. Tampoco se suben `vectorstore/` ni los PDF (`data/raw/`): se regeneran con los scripts.

## Ejecución

```bash
python ingest.py              # crea el índice en ./vectorstore (≈8 s, ≈US$0,002 de embeddings)
python test_rag.py            # corre las pruebas
python test_rag.py "¿Qué incluye el costo financiero total?"   # consulta libre
```

Otros comandos:

```bash
python ingest.py --preview    # muestra cantidad y tamaño de los chunks, y un ejemplo, sin generar embeddings
python ingest.py --force      # borra la colección y vuelve a indexar
python prepare_data.py        # regenera data/*.md desde los PDF del BCRA (los descarga si faltan)
```

Correr `ingest.py` antes es opcional: `get_rag_response()` crea el índice la primera vez si no existe. En las corridas siguientes lo reutiliza:

```
19:15:03 | INFO    | Generando embeddings de 315 chunks con text-embedding-3-small...
19:15:08 | INFO    | Índice creado en vectorstore/ (315 chunks)
...
19:15:10 | INFO    | Índice existente y actualizado (315 chunks, huella 1682e7c4e1787528) -> no se reindexa
```

## Pruebas

`test_rag.py` corre 5 consultas. Las 5 dan el resultado esperado con los dos proveedores (OpenAI `gpt-6-luna` y Anthropic `claude-haiku-4-5`):

| Tipo | Pregunta | Resultado |
|---|---|---|
| Respuesta en los documentos | ¿Qué tamaño mínimo de letra deben tener los contratos financieros? | ✅ 1,8 mm, punto 2.3.1.1 |
| Respuesta en los documentos | ¿Me pueden cobrar por abrir y mantener una caja de ahorros en pesos? | ✅ No, punto 1.8 |
| Trampa (fuera de tema) | ¿Quién ganó el mundial de fútbol 2022? | ✅ "No lo sé", descartada por umbral (sin LLM) |
| Trampa sutil | ¿Cuál es la tasa de interés de un plazo fijo hoy? | ✅ "No lo sé" (la norma dice que se pacta libremente, pero no da valores) |
| Trampa sutil | ¿Cuántos dólares puedo comprar por mes como persona humana? | ✅ "No lo sé" (es normativa cambiaria, que no está en el dataset) |

### Ejemplo: pregunta con respuesta

```
> ¿Qué tamaño mínimo de letra deben tener los contratos financieros?

Según el punto 2.3.1.1., los contratos financieros deben tener una tipografía de al menos 1,8 milímetros de altura.

encontrada=True | fragmentos recuperados=4 | respondido por: LLM
  📎 Protección de los usuarios de servicios financieros — 2.3. Recaudos mínimos de la relación de consumo. (pág. 7, vigente desde 28/02/2023, relevancia 0.62)
```

Objeto `RespuestaRAG` devuelto (`model_dump_json()`):

```json
{
  "pregunta": "¿Qué tamaño mínimo de letra deben tener los contratos financieros?",
  "respuesta": "Según el punto 2.3.1.1., los contratos financieros deben tener una tipografía de al menos 1,8 milímetros de altura.",
  "encontrada": true,
  "referencias": [
    {
      "documento": "Protección de los usuarios de servicios financieros",
      "seccion": "Sección 2. Derechos básicos de los usuarios de servicios financieros",
      "punto": "2.3. Recaudos mínimos de la relación de consumo.",
      "paginas": "7",
      "vigencia": "28/02/2023",
      "url": "https://www.bcra.gob.ar/archivos/Pdfs/Texord/t-pusf.pdf",
      "relevancia": 0.624
    }
  ],
  "fragmentos_recuperados": 4,
  "descartada_por_umbral": false
}
```

### Ejemplo: pregunta trampa sutil

```
> ¿Cuál es la tasa de interés de un plazo fijo hoy?

  0.604  depositos_ahorro_cuenta_sueldo.md | 4.4. Tasas de interés. (pág. 66)      <- supera el umbral:
  0.548  depositos_ahorro_cuenta_sueldo.md | 4.4. Tasas de interés. (pág. 67)         el tema está en la norma
  ...

No lo sé: según el punto 4.4.1., las tasas se acuerdan libremente entre las entidades financieras y los
clientes, pero la normativa disponible no informa cuál es la tasa vigente hoy para un plazo fijo.

encontrada=False | fragmentos recuperados=4 | respondido por: LLM
```

### Ejemplo: pregunta fuera de tema

```
> ¿Quién ganó el mundial de fútbol 2022?

  0.169  depositos_ahorro_cuenta_sueldo.md | 2.2. Titulares. (pág. 14)
  ...
Recuperados 4 fragmentos, 0 superan el umbral 0.30
Ningún fragmento supera el umbral -> se responde sin llamar al LLM

No lo sé: no encontré información relacionada con esa consulta en la normativa disponible (...).
```

## Parámetros

| Parámetro | Valor | Dónde | Por qué |
|---|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 tokens | `ingest.py` | Mínimo pedido por la consigna. La mediana real es de 415 tokens, porque muchos puntos de la norma son más cortos. |
| `MODELO_EMBEDDINGS` | `text-embedding-3-small` | `ingest.py` | Buen rendimiento en español y un límite de 8.191 tokens, así que no trunca chunks. |
| `TOP_K` | 4 | `rag.py` | Entre 3 y 5: contexto suficiente sin caer en *lost in the middle*. |
| `UMBRAL_RELEVANCIA` | 0,30 | `rag.py` | Calibrado para este modelo y este dataset. **Hay que recalibrarlo si cambia el modelo de embeddings.** |
| `MAX_INTENTOS` | 3 | `rag.py` | Reintentos ante JSON inválido o referencias incoherentes. |

## Limitaciones

- El sistema conoce solo estas 4 normas. Por ejemplo, el régimen completo de Personas Expuestas Políticamente o la normativa cambiaria están en otros textos ordenados y resoluciones que no forman parte del dataset.
- Los textos ordenados se actualizan con cada comunicación nueva del BCRA. Para actualizar el dataset hay que correr `prepare_data.py --force` y luego `ingest.py`, que detecta el cambio y reindexa solo.
- No es asesoramiento legal: las respuestas citan la norma, pero conviene consultar el texto oficial vigente.
