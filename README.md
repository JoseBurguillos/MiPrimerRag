# RAG autocorrectivo con LangGraph

✍️ (2-3 frases: qué es y qué problema resuelve. Por ejemplo: un asistente que responde
preguntas sobre unos documentos de informática y que, antes de responder, comprueba
si lo que ha encontrado sirve, para no inventarse respuestas.)

## Cómo funciona

```
START → rewrite_question → retrieve → grade_document ──decidir──┬─→ generate  → END
                                                                └─→ no_answer → END
```

| Nodo | Qué hace |
|---|---|
| `rewrite_question` | Reescribe la pregunta del usuario para que sea clara y concreta antes de buscar (p. ej. *"lo de las contraseñas ¿cómo va?"* → *"¿Cómo se gestionan las contraseñas de forma segura?"*) |
| `retrieve` | Busca en ChromaDB los `k` fragmentos más parecidos a la pregunta reescrita |
| `grade_document` | Un LLM juez evalúa si los fragmentos recuperados son relevantes para la pregunta original y devuelve `"si"` o `"no"` |
| `decidir` | Arista condicional: si el juez responde `"si"` pasa a `generate`; si responde `"no"`, a `no_answer` |
| `generate` | Genera la respuesta usando únicamente los fragmentos recuperados y la pregunta original |
| `no_answer` | Devuelve un mensaje indicando que no hay información sobre eso en los documentos |

## Tecnologías

- **LangGraph**: orquestación del flujo con nodos y aristas condicionales
- **LangChain**: mensajes, documentos y salida estructurada
- **ChromaDB**: base de datos vectorial
- **Ollama**: modelos en local (`qwen2.5:7b` como LLM y `embeddinggemma` para los embeddings)
- **Pydantic**: salida estructurada del juez (`Literal["si", "no"]`)

## Instalación

1. Instala [Ollama](https://ollama.com) y descarga los modelos:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull embeddinggemma
   ```
2. Crea un entorno virtual con Python 3.13 e instala las dependencias:
   ```bash
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Ejecuta desde la carpeta del proyecto:
   ```bash
   python Rag.py
   ```

## Decisiones de diseño

✍️ Explica por qué tomaste cada decisión. Algunas ideas de lo que has vivido:

- **Por qué un juez después de la búsqueda:** (similar ≠ relevante; Chroma siempre devuelve `k` fragmentos)
- **Por qué `Literal["si", "no"]` y no `str`:** (qué pasaba con `str`)
- **Por qué reescribir siempre al principio:** (el caso de "lo de las contraseñas ¿cómo va?")
- **Por qué el juez y `generate` usan la pregunta original y no la reescrita:** (la reescrita a veces se desvía)
- **Por qué `qwen2.5:7b` y no `llama3.2:3b`:** (qué fallaba con el 3B)

## Resultados

| Pregunta | Tipo | llama3.2:3b | qwen2.5:7b |
|---|---|---|---|
| lo de las contraseñas ¿cómo va? | Vaga | ❌ "No tengo información" | ✅ Correcta y fiel al documento |
| ¿Quién ganó el Mundial de 2010? | Fuera de tema | ✅ `no_answer` | ✅ `no_answer` |
| ¿Qué es un JOIN en SQL? | Trampa | ✅ `no_answer` | ✅ `no_answer` |
| eso de las IPs | Vaga | — | ✅ Correcta |
| ¿Qué son las propiedades ACID? | Concreta | ⚠️ Añade explicaciones | ⚠️ Añade explicaciones |

## Limitaciones y próximos pasos

✍️ Por ejemplo:
- El modelo todavía añade información que no está en los documentos (caso ACID). Siguiente paso: un nodo `check_hallucination`.
- (¿Qué más mejorarías?)

## Lo que aprendí

✍️ (3-5 puntos con lo más importante que has aprendido haciendo este proyecto.)
