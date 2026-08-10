import logging
import time
from pathlib import Path

from fastapi import File, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from langchain_core.documents import Document

from src.config import Config
from src.embeddings import EmbeddingError, EmbeddingManager
from src.loader import DocumentLoader, DocumentLoadError
from src.llm_factory import LLMFactoryError
from src.rag import RAGPipeline, RAGPipelineError
from src.schemas import (
    AnswerResponse,
    HealthResponse,
    QuestionRequest,
    UploadResponse,
)
from src.splitter import DocumentSplitter, DocumentSplitError
from src.vectorstore import VectorStoreError, VectorStoreManager


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Research Paper Answer Bot API",
    description="RAG based research paper question answering system",
    version="1.0"
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request paths, status, and timing without exposing sensitive data."""
    start_time = time.perf_counter()
    logger.info("API request started: %s %s", request.method, request.url.path)

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "API request completed: %s %s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "API request failed: %s %s duration_ms=%.2f error=%s",
            request.method,
            request.url.path,
            duration_ms,
            exc,
        )
        raise


config = Config()

embedding_manager = EmbeddingManager(config)

vector_store = VectorStoreManager(
    config,
    embedding_manager
)

rag_pipeline = RAGPipeline(
    config,
    vector_store
)


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(_: Request, exc: ResponseValidationError) -> JSONResponse:
    logger.error("Response validation error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid user input"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP error %s: %s", exc.status_code, exc.detail)
    if exc.status_code == status.HTTP_400_BAD_REQUEST:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid user input"},
        )
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        return JSONResponse(status_code=404, content={"error": "Resource not found"})
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(EmbeddingError)
async def embedding_exception_handler(_: Request, exc: EmbeddingError) -> JSONResponse:
    logger.error("Embedding error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Embedding generation failed"},
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(_: Request, exc: VectorStoreError) -> JSONResponse:
    logger.error("Vector store error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Vector database operation failed"},
    )


@app.exception_handler(LLMFactoryError)
async def llm_exception_handler(_: Request, exc: LLMFactoryError) -> JSONResponse:
    logger.error("LLM factory error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "LLM service unavailable"},
    )


@app.exception_handler(RAGPipelineError)
async def rag_exception_handler(_: Request, exc: RAGPipelineError) -> JSONResponse:
    logger.error("RAG pipeline error: %s", exc)
    message = str(exc).lower()
    if "empty" in message or "must be greater" in message:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid user input"},
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "LLM service unavailable"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


@app.get("/")
def home():
    return {
        "message": "Research Paper Answer Bot API is running",
        "docs": "/docs"
    }
def health_check():
    """Return the health status of the service and its dependencies."""
    try:
        vector_db_status = "connected"
        if not getattr(vector_store, "_collection", None):
            vector_db_status = "disconnected"

        return {
            "status": "healthy",
            "service": "Research Paper Answer Bot",
            "llm_provider": config.llm_provider,
            "model": config.chat_model,
            "database": vector_db_status,
        }
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return {
            "status": "degraded",
            "service": "Research Paper Answer Bot",
            "llm_provider": config.llm_provider,
            "model": config.chat_model,
            "database": "disconnected",
        }



@app.post(
    "/upload",
    response_model=UploadResponse
)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, save it to the data folder, and ingest it into the RAG pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    data_folder = Path(config.data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename).name
    save_path = data_folder / safe_filename

    with save_path.open("wb") as handle:
        handle.write(contents)

    try:
        loader = DocumentLoader(config)
        documents = loader.load_single_document(str(save_path))
    except DocumentLoadError as exc:
        logger.warning("Failed to load uploaded PDF: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    splitter = DocumentSplitter(config)
    chunks = splitter.split_documents(documents)

    # splitter returns dicts; convert back to Document for embedding and storage
    doc_chunks = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks
    ]

    embedding_manager.create_embeddings(doc_chunks)
    vector_store.add_documents(doc_chunks)

    return {
        "message": "PDF processed successfully",
        "filename": safe_filename,
        "chunks_created": len(chunks),
    }


@app.post(
    "/ask",
)
def ask_question(request: QuestionRequest):
    try:
        pipeline = RAGPipeline(config, vector_store)
        result = pipeline.ask(request.question)
        return AnswerResponse(
            answer=str(result.get("answer", "") if isinstance(result, dict) else result),
            sources=result.get("sources", []) if isinstance(result, dict) else [],
            retrieved_chunks=result.get("retrieved_chunks", 0) if isinstance(result, dict) else 0,
        )
    except RAGPipelineError as exc:
        message = str(exc).lower()
        if "empty" in message or "must be greater" in message:
            raise HTTPException(status_code=400, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unhandled error in /ask: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")