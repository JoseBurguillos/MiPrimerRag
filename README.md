# RAG autocorrectivo con LangGraph

Asistente que responde preguntas sobre un conjunto de documentos de informática (bases de datos, ciberseguridad, Python y redes). A diferencia de un RAG básico, **antes de responder comprueba si los fragmentos encontrados sirven de verdad para la pregunta**: si ninguno sirve, dice que no tiene información en lugar de inventarse una respuesta. Todo funciona en local con Ollama, sin APIs de pago.

## Cómo funciona

```
START → rewrite_question → retrieve → grade_document ──decidir──┬─→ generate  → END
                                                                └─→ no_answer → END
```

| Nodo | Qué hace |
|---|---|
| `rewrite_question` | Reescribe la pregunta del usuario para que sea clara y concreta antes de buscar (p. ej. *"lo de las contraseñas ¿cómo va?"* → *"¿Cómo se gestionan las contraseñas de forma segura?"*) |
| `retrieve` | Busca en ChromaDB los `k` fragmentos más parecidos a la pregunta reescrita |
| `grade_document` | Un LLM juez evalúa **cada fragmento por separado** frente a la pregunta original (`"si"` / `"no"`) y se queda solo con los relevantes |
| `decidir` | Arista condicional: si queda al menos un fragmento relevante pasa a `generate`; si no queda ninguno, a `no_answer` |
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

- **Por qué un juez después de la búsqueda:** ChromaDB siempre devuelve `k=3` fragmentos, aunque ninguno tenga relación con la pregunta. Que un fragmento sea *similar* no significa que sea *relevante*. Sin el juez, el modelo intentaba responder con información que no venía al caso.
- **Por qué evaluar cada fragmento por separado:** así se descartan solo los irrelevantes y `generate` recibe un contexto limpio, en lugar de aceptar o rechazar los tres en bloque.
- **Por qué `Literal["si", "no"]` y no `str`:** con un `str` libre el modelo puede devolver variantes ("Sí", "si, es relevante", frases completas) y la comparación falla. Con `Literal`, Pydantic obliga a que la salida sea exactamente una de las dos opciones.
- **Por qué reescribir siempre al principio:** las preguntas vagas como *"lo de las contraseñas ¿cómo va?"* no recuperaban buenos fragmentos. Reescribiéndolas a una pregunta clara, la búsqueda vectorial mejora mucho. El prompt de reescritura prohíbe añadir temas nuevos para no cambiar el significado.
- **Por qué el juez y `generate` usan la pregunta original y no la reescrita:** la reescritura a veces se desvía de lo que el usuario quería decir. La reescrita sirve para buscar mejor, pero para juzgar y responder la referencia tiene que ser la pregunta real.
- **Por qué `qwen2.5:7b` y no `llama3.2:3b`:** el modelo de 3B no entendía bien las preguntas vagas y respondía "No tengo información" aunque la respuesta estuviera en los documentos. `qwen2.5:7b` las resuelve correctamente sin dejar de rechazar las preguntas fuera de tema.
- **Por qué `temperature=0`:** el juez y la reescritura tienen que ser deterministas; con temperatura alta la misma pregunta podía dar resultados distintos.

## Resultados

| Pregunta | Tipo | llama3.2:3b | qwen2.5:7b |
|---|---|---|---|
| lo de las contraseñas ¿cómo va? | Vaga | ❌ "No tengo información" | ✅ Correcta y fiel al documento |
| ¿Quién ganó el Mundial de 2010? | Fuera de tema | ✅ `no_answer` | ✅ `no_answer` |
| ¿Qué es un JOIN en SQL? | Trampa | ✅ `no_answer` | ✅ `no_answer` |
| eso de las IPs | Vaga | — | ✅ Correcta |
| ¿Qué son las propiedades ACID? | Concreta | ⚠️ Añade explicaciones | ⚠️ Añade explicaciones |

## Limitaciones y próximos pasos

- El modelo todavía añade información que no está en los documentos (caso ACID). Siguiente paso: un nodo `check_hallucination` que verifique que la respuesta se apoya solo en los fragmentos.
- Si no queda ningún fragmento relevante, el sistema se rinde directamente. Mejora: reescribir la pregunta de otra forma y volver a buscar antes de devolver `no_answer`.
- La base vectorial se entrega ya construida; añadir un script de ingesta para indexar documentos nuevos (incluidos PDFs).
- Sustituir la entrada por terminal por una interfaz web o una API (Streamlit o FastAPI).

## Lo que aprendí

- Que un fragmento sea similar no significa que sea relevante: los resultados de una búsqueda vectorial hay que validarlos.
- La salida estructurada (Pydantic + `Literal`) es lo que permite usar un LLM dentro de la lógica de un programa de forma fiable.
- A modelar un flujo con nodos, estado compartido y aristas condicionales en LangGraph.
- El tamaño del modelo importa mucho con preguntas ambiguas, y la forma de comprobarlo es con casos de prueba concretos, no a ojo.
- A montar todo el sistema en local con Ollama, sin depender de APIs de pago.
