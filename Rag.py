from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma #Base de datos vectorial
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from typing import TypedDict,Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

modelo=ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)

embeddings = OllamaEmbeddings( #convierte el texto en vectores
    model="embeddinggemma"
)

base_vectorial=Chroma( #Base de datos de vectores, guardamos todos lo fragmentos de los archivos, nada de conversaciones
    collection_name="documentos_informatica", #el nombre de la coleccion donde se guardaran los fragmentos (conceptalmente una tabla)
    embedding_function=embeddings, #Le entrega a Chroma el objeto que debe utilizar para convertir textos y preguntas en vectores.
    persist_directory="./base_chroma"
)


class EstadoRAG (TypedDict):
    pregunta:str
    pregunta_original:str
    documentos:list[Document]
    respuesta:str

class EvaluacionDocumento(BaseModel):
    relevante:Literal["si", "no"]=Field(description="Escribe 'si' si el documento contiene la respuesta a la pregunta o, si la pregunta es general sobre un tema, si da información útil sobre ese tema. Escribe 'no' si el documento no responde a lo que se pregunta.")

def retrieve(state: EstadoRAG):
    pregunta=state["pregunta"]

    resultado=base_vectorial.similarity_search(
        pregunta,
        k=3
    )

    for doc in resultado:
        print("Recuperado:", doc.metadata["nombre"], "→", doc.page_content[:60])

    return{
        "documentos":resultado
    }

def generate (state:EstadoRAG):
    pregunta=state["pregunta_original"]
    documentos=state["documentos"]

    contexto= ""

    for documento in documentos:
        contexto+=documento.page_content

    instrucciones="""
    Eres un asistente que responde preguntas sobre informática usando solo los documentos proporcionados.

    Reglas:
        - Responde únicamente con la información incluida en el contexto.
        - Si la pregunta es general sobre un tema (por ejemplo "lo de las contraseñas"), resume la información del contexto sobre ese tema
        - No utilices conocimientos externos ni inventes datos.
        - Si el contexto no contiene la respuesta, responde exactamente: "No tengo información suficiente en los documentos para responder a eso."
        - Responde en español, de forma clara y breve.
    """

    mensajes=[
        SystemMessage(
            content=(instrucciones)
        ),
        HumanMessage(
            content=(
                f"Contexto:\n{contexto}\n\nPregunta del usuario: {pregunta}\n"
            )
        )
    ]

    resultado=modelo.invoke(
        mensajes
    )

    return{
        "respuesta":resultado.content
    }

juez=modelo.with_structured_output(EvaluacionDocumento)

PROMPT_JUEZ = """Eres un evaluador estricto. Tu única tarea es decidir si un documento contiene la respuesta a una pregunta.

Reglas:
- Responde 'si' solo si el documento contiene directamente la respuesta o una parte clara de ella.
- Responde 'no' si el documento solo trata del mismo tema general pero no responde a lo que se pregunta.
- No respondas a la pregunta. Solo evalúa el documento.
- Si la pregunta es general sobre un tema, responde ‘si’ cuando el documento dé información útil sobre ese tema
Ejemplos:
- Pregunta: "¿Qué son las propiedades ACID?" / Documento: "Una clave primaria identifica cada fila de una tabla." → no (es de bases de datos, pero no explica ACID)
- Pregunta: "¿Qué son las propiedades ACID?" / Documento: "Las propiedades ACID son atomicidad, consistencia, aislamiento y durabilidad." → si
- Pregunta: "¿Quién ganó el Mundial de 2010?" / Documento: "TCP prioriza una entrega fiable de los datos." → no"""

def grade_documents(state: EstadoRAG):
    documentos=state["documentos"]
    pregunta=state["pregunta_original"]

    relevantes=[]

    for documento in documentos:
        evaluacion=juez.invoke([
            SystemMessage(content=PROMPT_JUEZ),
            HumanMessage(content=f"Documento: {documento.page_content}\nPregunta:{pregunta}\n"
            )
        ])

        if evaluacion.relevante=="si":
            relevantes.append(documento)

    return {"documentos":relevantes}

def no_answer(state:EstadoRAG):
    return {
        "respuesta":"No tenemos suficiente información para responder ese pregunta"
    }


def decidir(state: EstadoRAG):
    if len(state["documentos"])==0:
        return "no_answer"
    else:
        return "generate" 

def rewrite_question(state:EstadoRAG):
    pregunta=state["pregunta_original"]

    prompt="""Eres un experto en reformular preguntas para mejorar búsquedas en una base de documentos de informática.

Reglas:
- Reescribe la pregunta para que sea clara y bien formulada.
- No añadas temas, conceptos ni detalles que no aparezcan en la pregunta original.
- Si la pregunta es vaga o general, mantenla general: solo hazla más clara, no elijas un aspecto concreto.
- No cambies el significado de la pregunta.
- Responde ÚNICAMENTE con la pregunta reescrita, sin explicaciones, sin comillas y sin frases como "Aquí tienes".

Ejemplos:
- "eso de las IPs" → ¿Qué es una dirección IP?
- "lo de las copias de seguridad" → ¿Qué información hay sobre las copias de seguridad?
- "qué pasa si falla algo en la base" → ¿Qué ocurre en una base de datos si falla una operación?
- "python cuando salió" → ¿Cuándo apareció la primera versión de Python?"""

    mensajes=[
            SystemMessage(
                content=(prompt)
            ),
            HumanMessage(
                content=(
                    f"Pregunta:{pregunta}"
                )
            )
        ]

    respuesta=modelo.invoke(
        mensajes
    )

    print(f"Pregunta reescrita:{respuesta.content}")

    return{
        "pregunta":respuesta.content
    }

graph=StateGraph(EstadoRAG)

graph.add_node("retrieve",retrieve)

graph.add_node("generate",generate)

graph.add_node("grade_document",grade_documents)

graph.add_node("no_answer",no_answer)

graph.add_node("rewrite_question",rewrite_question)

graph.add_edge(START,"rewrite_question")
graph.add_edge("rewrite_question","retrieve")
graph.add_edge("retrieve","grade_document")
graph.add_conditional_edges(
    "grade_document",
    decidir
)
graph.add_edge("generate", END)
graph.add_edge("no_answer", END)

app=graph.compile()

pregunta=input("Que quieres saber? ")
resultado=app.invoke({
    "pregunta":pregunta,
    "pregunta_original": pregunta
})

print("Documentos relevantes:",len(resultado["documentos"]) )
print(resultado["respuesta"])





