import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

app = FastAPI(title="Backend Vectorial RAG con Qdrant")

# --- Configuración de Qdrant y Embeddings ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "documentos_embeddings"

# Modelo de embeddings local
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cliente Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- Verificar y crear la colección si no existe ---
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # Tamaño de vector para 'all-MiniLM-L6-v2'
            distance=Distance.COSINE
        )
    )

# VectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# Directorio temporal
UPLOAD_DIR = "./temp_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Modelos de entrada/salida ---
class SearchQuery(BaseModel):
    query: str
    limit: int = 4

class SearchResult(BaseModel):
    content: str
    metadata: dict

# --- Endpoints ---

@app.post("/upload", summary="Cargar y procesar un archivo PDF o DOCX")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=400, 
            detail="Formato no soportado. Sube un archivo .pdf o .docx"
        )
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
            
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        vector_store.add_documents(chunks)
        
        return {
            "status": "éxito",
            "filename": filename,
            "chunks_procesados": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/search", response_model=List[SearchResult], summary="Buscar contexto relevante")
async def search_documents(payload: SearchQuery):
    try:
        results = vector_store.similarity_search(
            query=payload.query,
            k=payload.limit
        )
        
        return [
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in results
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)