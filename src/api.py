import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import src.rag as rag
from src.rag import ingest_pdf, generate_response
import shutil

load_dotenv()

app=FastAPI(title="RAG API", description="API pour le système RAG avec MistralAI et Postgres")
#model for the question input
class RequestModel(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"status": "L'API est en ligne"}

@app.post("/upload")
async def upload(file: UploadFile=File(...)):
    try:
        #save the file locally
        os.makedirs("data", exist_ok=True)
        file_path=f"data/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        #ingest the PDF
        rag.vector_store=ingest_pdf(file_path)
        return {"status": "Fichier téléchargé et traité avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask")
async def ask_question(request: RequestModel):
    if rag.vector_store is None:
        raise HTTPException(status_code=400, detail="Veuillez d'abord uploader un PDF via /upload")
    
    try:
        reponse = generate_response(rag.vector_store, request.query)
        return {"question": request.query, "reponse": reponse}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)