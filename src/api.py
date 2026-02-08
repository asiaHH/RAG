import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import src.rag as rag
from src.rag import generate_response
from src.ingestion.loaders import ingest_pdf, ingest_txt, ingest_pptx, ingest_excel, ingest_csv, ingest_unstructured_files
import shutil
from typing import List
import logging
from fastapi.responses import JSONResponse
from src.ingestion import pipeline, sync

load_dotenv()
logger = logging.getLogger(__name__)

app=FastAPI(title="RAG API", description="API for the RAG system with MistralAI and Postgres")

class RequestModel(BaseModel):
    query: str

class SyncRequest(BaseModel):
    directory: str = "data"

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions and return a generic error response.    
    :param request: The incoming request that caused the exception
    :param exc: The exception that was raised
    :return: A JSON response with a 500 status code and a generic error message
    """
    logger.error("Error: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal error of the server"})

# @app.on_event("startup")
# async def startup_event():
#     """
#     Event handler for application startup. Initializes the vector store and logs the status of the connection.
#     """
#     global vector_store
#     vector_store = init_vector_store()
#     if vector_store is not None:
#         logger.info("PGVector connection successful")
#     else:
#         logger.error("PGVector connection failed")

@app.get("/")
def read_root():
    return {"status": "The API is online"}

@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple files. Each file is saved to the "data" directory.    
    :param files: A list of files uploaded by the user
    :return: A JSON response indicating the success or failure of the upload process
    """
    try:
        os.makedirs("data", exist_ok=True)
        for file in files:
            path = f"data/{file.filename}"
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        return {"status": "Files uploaded and processed successfully"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/sync")
async def sync_collection_endpoint(request: SyncRequest = None):
    """
    Endpoint to synchronize the vector store with the files in the specified directory.
    :param request: A SyncRequest object containing the directory to synchronize
    :return: A JSON response indicating the success or failure of the synchronization process
    """
    directory = request.directory if request else "data"
    try:
        if pipeline.vector_store is None:
            pipeline.init_vector_store()

        sync.sync_collection(directory)

        return {"status": f"Collection {os.path.basename(directory)} synchronised with success"}
    except Exception as e:
        logger.error(f"Error during synchronization: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask")
async def ask_question(request: RequestModel):
    """
    Endpoint to ask a question to the RAG system. The question is processed using the generate_response function, which retrieves relevant documents from the vector store and generates an answer using the MistralAI model.
    :param request: A RequestModel object containing the user's question
    :return: A response containing the answer and the sources with metadata
    """
    pipeline.init_vector_store()
    if pipeline.vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first via /upload")
    
    try:
        result = generate_response(pipeline.vector_store, request.query)
        answer = result["answer"]
        docs = result["sources"]

        sources = []
        for d in docs:
            source_dict = {
                "page_content": d.page_content[:200], 
                "metadata": {
                    "source": d.metadata.get("source", "Inconnu"),
                    "page": d.metadata.get("page", None),
                    "file_type": d.metadata.get("file_type", None),
                    "source_id": d.metadata.get("source_id", None)
                }
            }
            sources.append(source_dict)
        
        return {
            "question": request.query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_response_endpoint(file_path: str, question: str):
    """
    Endpoint to generate a response based on a given file and a question. The file is ingested into the vector store, and the question is processed to retrieve relevant information from the file and generate an answer.
    
    :param file_path: The path to the file to be ingested
    :param question: The question to be answered
    :return: A direct response
    """
    if file_path.endswith(".pdf"):
        vector_store = ingest_pdf(file_path)
    elif file_path.endswith(".txt"):
        vector_store = ingest_txt(file_path)
    elif file_path.endswith(".pptx"):
        vector_store = ingest_pptx(file_path)
    elif file_path.endswith(".xlsx"):
        vector_store = ingest_excel(file_path)
    elif file_path.endswith(".csv"):
        vector_store = ingest_csv(file_path)
    else:
        vector_store = ingest_unstructured_files(file_path)
    
    result = generate_response(vector_store, question)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)