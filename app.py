"""Asesor RAG de Libertad Financiera.

Un chatbot de Gradio que responde preguntas de finanzas personales desde
documentos locales usando Gemini, LangChain y ChromaDB.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Asesor RAG de Libertad Financiera"
DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

SYSTEM_PROMPT = """Eres el Asesor RAG de Libertad Financiera, un asistente educativo.
Responde la pregunta del usuario en español usando únicamente el contexto de abajo.
Sé práctico, conciso y claro. Si la respuesta no aparece en el contexto, di que no
lo sabes con base en la base de conocimiento proporcionada.

Importante: No des asesoría financiera, fiscal, legal o de inversión personalizada.
Explica principios generales y recomienda consultar a un profesional calificado para
decisiones sobre situaciones personales.

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""


def load_documents() -> List[Document]:
    """Carga todos los documentos .txt desde la carpeta local docs."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError("Falta la carpeta docs.")

    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    if not documents:
        raise FileNotFoundError("No se encontraron documentos .txt en la carpeta docs.")
    return documents


def build_vector_store(api_key: str) -> Chroma:
    """Crea o refresca la base vectorial local de Chroma desde docs."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="financial_freedom_docs",
    )


@lru_cache(maxsize=1)
def create_qa_chain(api_key: str) -> RetrievalQA:
    """Crea una cadena RAG con Gemini y Chroma."""
    vector_store = build_vector_store(api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def format_sources(source_documents: List[Document]) -> str:
    """Formatea los nombres de archivo usados como fuentes."""
    if not source_documents:
        return "Fuentes usadas: no se devolvieron fuentes."

    seen = []
    for doc in source_documents:
        source = Path(str(doc.metadata.get("source", "Unknown source"))).name
        if source not in seen:
            seen.append(source)

    return "Fuentes usadas: " + ", ".join(seen)


def answer_question(message: str, history: List[Tuple[str, str]]) -> str:
    """Responde una pregunta con contexto recuperado y fuentes."""
    del history

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "GEMINI_API_KEY no está configurada. Agrégala como variable de entorno "
            "o como secreto en HuggingFace Spaces, y luego reinicia la app."
        )

    if not message.strip():
        return "Por favor haz una pregunta sobre libertad financiera o finanzas personales."

    try:
        qa_chain = create_qa_chain(api_key)
        result = qa_chain.invoke({"query": message})
        answer = str(result.get("result", "")).strip()
        sources = format_sources(result.get("source_documents", []))
        return f"{answer}\n\n{sources}"
    except Exception as exc:
        return (
            "Lo siento, falló el servicio de IA o el pipeline de recuperación. "
            "Revisa tu API key, los documentos y el acceso a internet.\n\n"
            f"Detalles del error: {exc}"
        )


def build_ui() -> gr.Blocks:
    """Construye la interfaz de Gradio."""
    demo_questions = [
        "¿Qué tan grande debería ser mi fondo de emergencia antes de invertir?",
        "¿Cómo ayuda la diversificación a manejar el riesgo de inversión?",
        "¿En qué orden debería priorizar pagar deudas, ahorrar e invertir?",
    ]

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(
            "Haz preguntas educativas sobre libertad financiera, presupuesto, "
            "fondos de emergencia, deuda, inversión, diversificación y riesgo."
        )

        gr.ChatInterface(
            fn=answer_question,
            examples=demo_questions,
            chatbot=gr.Chatbot(height=460),
            textbox=gr.Textbox(
                placeholder="Haz una pregunta sobre libertad financiera...",
                container=False,
                scale=7,
                submit_btn="Preguntar",
                stop_btn="Detener",
            ),
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft())
