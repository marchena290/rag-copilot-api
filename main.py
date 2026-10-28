"""FastAPI endpoint for generating RAG suggestions for support tickets."""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import Generator
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.api_core.exceptions import ResourceExhausted
from google import genai
from google.genai import types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, insert, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from models import HistorialRAG


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)
logger = logging.getLogger("rag_api")

CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "manuales_ti"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_DISTANCE = 1.2
GEMINI_REQUEST_TIMEOUT_MS = 45_000
DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
VALID_GENERATION_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
}
configured_generation_model = os.getenv("GEMINI_MODEL", "").strip()
GENERATION_MODEL_NAME = (
    configured_generation_model
    if configured_generation_model in VALID_GENERATION_MODELS
    else DEFAULT_GENERATION_MODEL
)
if configured_generation_model and configured_generation_model not in VALID_GENERATION_MODELS:
    logger.warning(
        "[RAG] GEMINI_MODEL no es válido. Se usará "
        f"'{GENERATION_MODEL_NAME}' por defecto."
    )

engine: Engine | None = None
chroma_collection: chromadb.Collection | None = None
embedding_model: SentenceTransformer | None = None
gemini_client: genai.Client | None = None

GEMINI_HIGH_DEMAND_MESSAGE = (
    "El servicio de inteligencia artificial está experimentando una alta "
    "demanda temporal en sus servidores. Por favor, intenta enviar tu "
    "consulta de nuevo en unos segundos."
)


class ConsultaRAGRequest(BaseModel):
    consulta: Optional[str] = None
    ticket_id: Optional[int] = None
    usuario_id: Optional[int] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global chroma_collection, embedding_model, engine, gemini_client

    database_url = os.getenv("DATABASE_URL")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    rag_api_key = os.getenv("RAG_API_KEY", "").strip()
    if not database_url:
        raise RuntimeError("Falta DATABASE_URL en el archivo .env")
    if not gemini_api_key:
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    if not rag_api_key:
        raise RuntimeError("Falta RAG_API_KEY en el archivo .env")

    engine = create_engine(database_url, pool_pre_ping=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    chroma_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    gemini_client = genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
    )

    yield

    engine.dispose()
    engine = None
    chroma_collection = None
    embedding_model = None
    gemini_client = None


app = FastAPI(title="DevDesk RAG Copilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-RAG-API-Key"],
)


def require_rag_api_key(
    x_rag_api_key: str | None = Header(default=None),
) -> None:
    configured_key = os.getenv("RAG_API_KEY", "").strip()
    if (
        not configured_key
        or not x_rag_api_key
        or not secrets.compare_digest(x_rag_api_key, configured_key)
    ):
        raise HTTPException(
            status_code=401,
            detail="Falta una credencial válida para acceder a la API RAG",
        )


def _get_runtime_dependencies() -> tuple[
    Engine,
    chromadb.Collection,
    SentenceTransformer,
    genai.Client,
]:
    if not all((engine, chroma_collection, embedding_model, gemini_client)):
        raise HTTPException(
            status_code=503,
            detail="El servicio RAG todavía no está inicializado",
        )
    return engine, chroma_collection, embedding_model, gemini_client


def get_db() -> Generator[Session, None, None]:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="La base de datos todavía no está inicializada",
        )

    db = Session(engine)
    try:
        yield db
    finally:
        db.close()


@app.get("/api/rag/historial")
def obtener_historial_rag(
    _: None = Depends(require_rag_api_key),
    db: Session = Depends(get_db),
):
    try:
        registros = (
            db.query(HistorialRAG)
            .order_by(HistorialRAG.created_at.desc())
            .limit(20)
            .all()
        )
        return registros
    except Exception as error:
        logger.exception("[RAG] Error obteniendo historial")
        raise HTTPException(
            status_code=502,
            detail="Error al obtener el historial RAG",
        ) from error


def _build_prompt(
    title: str,
    description: str,
    documents: list[str],
    sources: list[str],
) -> str:
    context = "\n\n".join(
        f"[Fuente: {source}]\n{document}"
        for source, document in zip(sources, documents)
    )
    return f"""Eres un especialista de soporte técnico de TI de nivel 2.
Genera una solución clara para el ticket usando EXCLUSIVAMENTE el contexto
de los manuales internos que aparece abajo. No inventes pasos ni información.
Si el contexto no permite resolver el problema, dilo explícitamente.

CONTEXTO DE MANUALES INTERNOS:
{context}

TICKET:
- Título: {title}
- Descripción: {description}

Entrega pasos ordenados y menciona las fuentes utilizadas."""


def _is_gemini_unavailable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    error_text = str(error).upper()
    return status_code == 503 or "503" in error_text or "UNAVAILABLE" in error_text


def _is_gemini_rate_limited(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    error_text = str(error).upper()
    return (
        isinstance(error, ResourceExhausted)
        or status_code == 429
        or error_code == 429
        or "429" in error_text
        or "RESOURCE_EXHAUSTED" in error_text
    )


def _is_gemini_unauthenticated(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    error_text = str(error).upper()
    return (
        status_code == 401
        or error_code == 401
        or "401" in error_text
        or "UNAUTHENTICATED" in error_text
        or "API KEY NOT VALID" in error_text
        or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in error_text
    )


def _is_gemini_deadline_exceeded(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    error_text = str(error).upper()
    return (
        status_code == 504
        or error_code == 504
        or "504" in error_text
        or "DEADLINE_EXCEEDED" in error_text
    )


def _safe_gemini_error(error: Exception) -> str:
    message = str(error)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message[:500]


def _generate_gemini_response(
    gemini: genai.Client,
    prompt: str,
) -> str:
    retry_delays = (1.5, 3.0)

    for attempt in range(3):
        try:
            response = gemini.models.generate_content(
                model=GENERATION_MODEL_NAME,
                contents=prompt,
            )
            generated_answer = response.text
            if not generated_answer:
                raise RuntimeError("Gemini no devolvió una respuesta de texto")
            return generated_answer
        except Exception as error:
            if _is_gemini_unauthenticated(error):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "La credencial de Gemini no es válida para la API "
                        "Generative Language. Crea una API key en Google AI "
                        "Studio y configura GEMINI_API_KEY."
                    ),
                ) from error
            if _is_gemini_deadline_exceeded(error):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Gemini tardó demasiado en responder. "
                        "Intenta nuevamente más tarde."
                    ),
                    headers={"Retry-After": "30"},
                ) from error
            is_rate_limited = _is_gemini_rate_limited(error)
            is_unavailable = _is_gemini_unavailable(error)
            if not (is_rate_limited or is_unavailable):
                raise
            if is_rate_limited:
                raise HTTPException(
                    status_code=429,
                    detail=GEMINI_HIGH_DEMAND_MESSAGE,
                    headers={"Retry-After": "30"},
                ) from error
            if attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])

    raise RuntimeError("No se pudo obtener una respuesta de Gemini")


def _guardar_historial_rag(
    db_engine: Engine,
    *,
    usuario_id: int | None,
    ticket_id: int | None,
    modo: str,
    consulta: str,
    respuesta: str,
    fuentes: list[Any],
) -> int:
    try:
        with db_engine.begin() as connection:
            historial_id = connection.execute(
                insert(HistorialRAG).values(
                    usuario_id=usuario_id,
                    ticket_id=ticket_id,
                    modo=modo,
                    consulta=consulta,
                    respuesta=respuesta,
                    fuentes=fuentes,
                )
            ).inserted_primary_key[0]
    except Exception as error:
        logger.exception("Error al guardar el historial RAG")
        raise HTTPException(
            status_code=502,
            detail="Error al guardar el historial de la consulta",
        ) from error

    if historial_id is None:
        raise HTTPException(
            status_code=502,
            detail="La base de datos no devolvió el id del historial",
        )
    return int(historial_id)


@app.post("/api/rag/consultar")
def consultar_rag(
    request: ConsultaRAGRequest,
    _: None = Depends(require_rag_api_key),
) -> dict[str, Any]:
    consulta = request.consulta.strip() if request.consulta else ""
    title = "Consulta libre"
    description = consulta

    if request.ticket_id is None and not consulta:
        raise HTTPException(
            status_code=400,
            detail="Debe enviar una consulta o un ticket_id válido",
        )

    db_engine, collection, model, gemini = _get_runtime_dependencies()

    if request.ticket_id is not None:
        try:
            with db_engine.connect() as connection:
                ticket = connection.execute(
                    text(
                        """
                        SELECT id, title, description
                        FROM public.tickets
                        WHERE id = :ticket_id
                        """
                    ),
                    {"ticket_id": request.ticket_id},
                ).mappings().one_or_none()
        except Exception as e:
            logger.exception("Error consultando el ticket")
            raise HTTPException(
                status_code=502,
                detail="Error al consultar el ticket en la base de datos",
            ) from e

        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket no encontrado")

        title = str(ticket["title"] or "")
        description = str(ticket["description"] or "")
        query_parts = [title, description, consulta]
        query = "\n".join(part for part in query_parts if part)
    else:
        query = consulta

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Debe enviar una consulta o un ticket_id válido",
        )

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    filtered_results = [
        (document, metadata or {}, distance)
        for document, metadata, distance in zip(
            documents, metadatas, distances
        )
        if float(distance) <= MAX_DISTANCE
    ]

    if not filtered_results:
        response = (
            "No se encontró documentación interna relevante para "
            "responder a esta consulta."
        )
        historial_id = _guardar_historial_rag(
            db_engine,
            usuario_id=request.usuario_id,
            ticket_id=request.ticket_id,
            modo="ticket" if request.ticket_id is not None else "libre",
            consulta=query,
            respuesta=response,
            fuentes=[],
        )
        return {
            "id": historial_id,
            "solucion_propuesta": response,
            "fuentes": [],
        }

    filtered_documents = [
        str(document) for document, _, _ in filtered_results
    ]
    sources = [
        str(metadata.get("fuente", "fuente_desconocida"))
        for _, metadata, _ in filtered_results
    ]
    prompt = _build_prompt(title, description, filtered_documents, sources)

    try:
        generated_answer = _generate_gemini_response(gemini, prompt)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error en Gemini")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini no pudo generar una respuesta: {_safe_gemini_error(e)}",
        ) from e

    unique_sources = list(dict.fromkeys(sources))
    historial_id = _guardar_historial_rag(
        db_engine,
        usuario_id=request.usuario_id,
        ticket_id=request.ticket_id,
        modo="ticket" if request.ticket_id is not None else "libre",
        consulta=query,
        respuesta=generated_answer,
        fuentes=unique_sources,
    )
    return {
        "id": historial_id,
        "solucion_propuesta": generated_answer,
        "fuentes": unique_sources,
    }


@app.post("/api/tickets/{ticket_id}/sugerir-solucion")
def sugerir_solucion(
    ticket_id: int,
    _: None = Depends(require_rag_api_key),
) -> dict[str, Any]:
    db_engine, collection, model, gemini = _get_runtime_dependencies()

    try:
        with db_engine.connect() as connection:
            ticket = connection.execute(
                text(
                    """
                    SELECT id, title, description
                    FROM public.tickets
                    WHERE id = :ticket_id
                    """
                ),
                {"ticket_id": ticket_id},
            ).mappings().one_or_none()
    except Exception as e:
        logger.exception("Error consultando el ticket")
        raise HTTPException(
            status_code=502,
            detail="Error al consultar el ticket en la base de datos",
        ) from e

    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket no encontrado")

    query = f"{ticket['title']}\n{ticket['description']}"
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    filtered_results = [
        (document, metadata, distance)
        for document, metadata, distance in zip(
            documents, metadatas, distances
        )
        if float(distance) <= MAX_DISTANCE
    ]
    filtered_documents = [document for document, _, _ in filtered_results]
    filtered_metadatas = [metadata for _, metadata, _ in filtered_results]
    filtered_distances = [distance for _, _, distance in filtered_results]
    sources = [
        str(metadata.get("fuente", "fuente_desconocida"))
        for metadata in filtered_metadatas
    ]

    if not filtered_results:
        generated_answer = (
            "No se encontró documentación interna relevante para resolver "
            "este ticket."
        )
        average_score = None
        sources_json = "[]"
    else:
        prompt = _build_prompt(
            str(ticket["title"]),
            str(ticket["description"]),
            [str(document) for document in filtered_documents],
            sources,
        )

        try:
            generated_answer = _generate_gemini_response(gemini, prompt)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error en Gemini")
            raise HTTPException(
                status_code=502,
                detail=f"Gemini no pudo generar una respuesta: {_safe_gemini_error(e)}",
            ) from e

        average_score = (
            sum(1 / (1 + float(distance)) for distance in filtered_distances)
            / len(filtered_distances)
            if filtered_distances
            else None
        )
        sources_json = json.dumps(
            [
                {
                    "fuente": source,
                    "chunk_index": metadata.get("chunk_index"),
                }
                for source, metadata in zip(sources, filtered_metadatas)
            ],
            ensure_ascii=False,
        )

    try:
        with db_engine.begin() as connection:
            suggestion_id = connection.execute(
                text(
                    """
                    INSERT INTO public.sugerencias_rag (
                        ticket_id,
                        respuesta_generada,
                        fuentes_consultadas,
                        score_similitud_promedio
                    )
                    VALUES (
                        :ticket_id,
                        :respuesta_generada,
                        :fuentes_consultadas,
                        :score_similitud_promedio
                    )
                    RETURNING id
                    """
                ),
                {
                    "ticket_id": ticket_id,
                    "respuesta_generada": generated_answer,
                    "fuentes_consultadas": sources_json,
                    "score_similitud_promedio": average_score,
                },
            ).scalar_one()
    except Exception as e:
        logger.exception("Error guardando la sugerencia RAG")
        raise HTTPException(
            status_code=502,
            detail="Error al guardar la sugerencia en la base de datos",
        ) from e

    historial_id = _guardar_historial_rag(
        db_engine,
        usuario_id=None,
        ticket_id=ticket_id,
        modo="ticket",
        consulta=query,
        respuesta=generated_answer,
        fuentes=json.loads(sources_json),
    )
    return {
        "id": suggestion_id,
        "historial_id": historial_id,
        "ticket_id": ticket_id,
        "solucion_propuesta": generated_answer,
        "fuentes": json.loads(sources_json),
    }