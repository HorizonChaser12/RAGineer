# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

# for uvicorn web server
import uvicorn

# For logging
import logging
import datetime
from typing import Optional

# Pydantic Model Imports
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema.pydantic_models import InitializeRequest, RebuildIndexRequest, QueryRequest

# RAG Imports
from rag_system import EnhancedAdaptiveRAGSystem
from rag_system import make_serializable


# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="azure_rag_system_api.log",
    filemode="a",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Azure RAG System API",
    description="Advanced RAG system with multiple retrieval methods and re-ranking",
    version="2.0.0",
)

# This is for passing all the hosts where requests are being sent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the RAG system instance
rag_system: Optional[EnhancedAdaptiveRAGSystem] = None


# Exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc)}
    )


@app.post("/initialize")
async def initialize_system(request: InitializeRequest):
    """Initialize the RAG system with specified configuration"""

    global rag_system

    try:

        logger.info(f"Initializing RAG system with file: {request.excel_file_path}")

        rag_system = EnhancedAdaptiveRAGSystem(
            excel_file_path=request.excel_file_path,
            temperature=request.temperature,
            concise_prompt=request.concise_prompt,
            index_file=request.index_file,
            use_sentence_transformers=request.use_sentence_transformers,
            use_reranker=request.use_reranker,
            sentence_transformer_model=request.sentence_transformer_model,
            reranker_model=request.reranker_model,
        )

        status = rag_system.get_system_status()

        logger.info("RAG system initialized successfully")

        return {"message": "RAG system initialized successfully", "status": status}

    except Exception as e:

        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/query")
async def query_system(request: QueryRequest):
    """Query the RAG system with optional analytical mode"""

    global rag_system

    if rag_system is None:

        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    try:

        logger.info(
            f"Processing query: {request.query[:100]}... (Analytical mode: {request.analytical_mode})"
        )

        # Update temperature if provided

        if request.temperature is not None and rag_system.llm:

            rag_system.llm.temperature = request.temperature

        # Choose processing mode

        if request.analytical_mode:

            # Use analytical thinking mode with system's data

            response = rag_system.analyze_data_directly(query=request.query)

            logger.info("Query processed in analytical thinking mode using system data")

        else:

            # Use traditional RAG mode

            response = rag_system.generate_response(request.query, request.k)

            logger.info("Query processed in traditional RAG mode")

        return response

    except Exception as e:

        logger.error(f"Error processing query: {e}", exc_info=True)

        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.post("/rebuild-index")
async def rebuild_index(request: RebuildIndexRequest):
    """Rebuild the FAISS index and sentence transformer embeddings"""

    global rag_system

    if rag_system is None:

        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    try:

        logger.info("Rebuilding indices...")

        rag_system._build_index()

        status = rag_system.get_system_status()

        logger.info("Indices rebuilt successfully")

        return {"message": "Indices rebuilt successfully", "status": status}

    except Exception as e:

        logger.error(f"Failed to rebuild indices: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")


@app.get("/status")
async def get_status():
    """Get system status information"""

    global rag_system

    if rag_system is None:

        return {"initialized": False, "message": "RAG system not initialized"}

    try:

        status = rag_system.get_system_status()

        status["initialized"] = True

        return status

    except Exception as e:

        logger.error(f"Error getting status: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.post("/retrieve")
async def retrieve_documents(request: QueryRequest):
    """Retrieve relevant documents without generating a response"""

    global rag_system

    if rag_system is None:

        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    # Analytical mode doesn't support document retrieval

    if request.analytical_mode:

        raise HTTPException(
            status_code=400,
            detail="Document retrieval not available in analytical mode. Use /query endpoint instead.",
        )

    try:

        logger.info(f"Retrieving documents for query: {request.query[:100]}...")

        retrieved_docs = rag_system.retrieve(request.query, request.k)

        pattern_analysis = rag_system.analyze_patterns(retrieved_docs)

        return {
            "query": request.query,
            "retrieved_docs": make_serializable(retrieved_docs),
            "pattern_analysis": make_serializable(pattern_analysis),
            "count": len(retrieved_docs),
        }

    except Exception as e:

        logger.error(f"Error retrieving documents: {e}", exc_info=True)

        raise HTTPException(
            status_code=500, detail=f"Document retrieval failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""

    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "system_initialized": rag_system is not None,
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""

    return {
        "message": "Enhanced Azure RAG System API",
        "version": "2.0.0",
        "features": [
            "Azure OpenAI Integration",
            "Sentence Transformers",
            "Cross-Encoder Re-ranking",
            "FAISS Vector Search",
            "Pattern Analysis",
        ],
        "endpoints": {
            "POST /initialize": "Initialize the RAG system",
            "POST /query": "Query the system for responses",
            "POST /retrieve": "Retrieve relevant documents only",
            "POST /rebuild-index": "Rebuild search indices",
            "GET /status": "Get system status",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":

    logger.info("Starting Enhanced Azure RAG System API server...")

    uvicorn.run(
        "api_app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=True,
    )
