import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag import generate_response
from src.ingestion.loaders import ingest_pdf, ingest_txt, ingest_pptx, ingest_excel, ingest_csv, ingest_docx
from src.auth import create_access_token, get_current_user, require_admin
import shutil
from typing import List, Optional
import logging
from fastapi.responses import JSONResponse
from src.ingestion import pipeline, sync
from src.db.conversation import Conversation
from src.db.users import Users
from src.config import PSYCOPG2_CONNECTION_STRING

load_dotenv()
logger = logging.getLogger(__name__)

conversation_store = Conversation(PSYCOPG2_CONNECTION_STRING)
user_store = Users(PSYCOPG2_CONNECTION_STRING)

app=FastAPI(title="RAG API", description="API for the RAG system with MistralAI and Postgres")

class RequestModel(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class SyncRequest(BaseModel):
    directory: Optional[str] = None

class CreateConversationRequest(BaseModel):
    title: str = "Nouvelle discussion"

class AddMessageRequest(BaseModel):
    role: str
    content: str
    sources: Optional[list] = None

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


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

@app.get("/")
def read_root():
    return {"status": "The API is online"}

# --- Endpoints de gestion des utilisateurs ---
@app.post("/auth/register")
def register(req: RegisterRequest):
    if user_store.get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Cet email est déjà utilisé")
    user_id = user_store.create_user(req.email, req.password)
    token = create_access_token(user_id, req.email, is_admin=False)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/login")
def login(req: LoginRequest):
    user = user_store.verify_password(req.email, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    token = create_access_token(user["id"], user["email"], user["is_admin"])
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return user


# --- Endpoints de gestion des Conversations ---

@app.get("/conversations")
def get_conversations(user: dict = Depends(get_current_user)):
    return conversation_store.list_conversations(user["sub"])

@app.post("/conversations")
def create_conversation(req: CreateConversationRequest, user: dict = Depends(get_current_user)):
    conv_id = conversation_store.create_conversation(user["sub"], title=req.title)
    return {"id": conv_id, "title": req.title}

@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str, user: dict = Depends(get_current_user)):
    return conversation_store.load_messages(conversation_id, user["sub"])

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict = Depends(get_current_user)):
    conversation_store.delete_conversation(conversation_id, user["sub"])
    return {"status": "deleted"}

@app.post("/conversations/{conversation_id}/messages")
def add_message_to_conv(conversation_id: str, req: AddMessageRequest, user: dict = Depends(get_current_user)):
    if not conversation_store.conversation_belongs_to_user(conversation_id, user["sub"]):
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    conversation_store.add_message(conversation_id, user["sub"], req.role, req.content, req.sources)
    return {"status": "added"}


@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    """
    Endpoint to upload multiple files. Each file is saved to the "data" directory.    
    :param files: A list of files uploaded by the user
    :return: A JSON response indicating the success or failure of the upload process
    """
    try:
        user_dir = f"data/{user['sub']}"
        os.makedirs(user_dir, exist_ok=True)
        for file in files:
            path = f"{user_dir}/{file.filename}"
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        return {"status": "Files uploaded and processed successfully"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/sync")
async def sync_collection_endpoint(request: SyncRequest = None, user: dict = Depends(get_current_user)):
    """
    Endpoint to synchronize the vector store with the files in the specified directory.
    :param request: A SyncRequest object containing the directory to synchronize
    :return: A JSON response indicating the success or failure of the synchronization process
    """
    directory = request.directory if (request and request.directory) else f"data/{user['sub']}"
    try:
        if pipeline.vector_store is None:
            pipeline.init_vector_store()

        sync.sync_collection(directory, user["sub"])

        return {"status": f"Collection {os.path.basename(directory)} synchronised with success"}
    except Exception as e:
        logger.error(f"Error during synchronization: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/clear-collection")
async def clear_collection_endpoint(user: dict = Depends(get_current_user)):
    """
    Endpoint to completely clear the vector collection.
    This removes all documents and embeddings from the vector store.
    :return: A JSON response indicating the success or failure of the operation
    """
    try:
        success = pipeline.clear_user_collection(user["sub"])
        if success:
            return {"status": "Collection vidée complètement avec succès"}
        else:
            raise HTTPException(status_code=500, detail="Erreur lors du vidage de la collection")
    except Exception as e:
        logger.error(f"Erreur lors du vidage de la collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask")
async def ask_question(request: RequestModel, user: dict = Depends(get_current_user)):
    """
    Endpoint to ask a question to the RAG system. The question is processed using the generate_response function, which retrieves relevant documents from the vector store and generates an answer using the MistralAI model.
    :param request: A RequestModel object containing the user's question
    :return: A response containing the answer and the sources with metadata
    """
    pipeline.init_vector_store()
    if pipeline.vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first via /upload")
    
    try:
        result = generate_response(pipeline.vector_store, request.query, user["sub"])
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

        if request.conversation_id:
            if conversation_store.conversation_belongs_to_user(request.conversation_id, user["sub"]):
                conversation_store.add_message(request.conversation_id, user["sub"], "user", request.query)
                conversation_store.add_message(request.conversation_id, user["sub"], "assistant", answer, sources)
        
        return {
            "question": request.query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)