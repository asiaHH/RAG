import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

app_user = os.getenv("APP_DB_USER")
app_password = os.getenv("APP_DB_PASSWORD")
host = os.getenv("POSTGRES_HOST", "localhost")
port = os.getenv("POSTGRES_PORT", "5432")
#for PGVector
CONNECTION_STRING = f"postgresql+psycopg2://{app_user}:{app_password}@{host}:{port}/vector_db"
#for psycopg2
PSYCOPG2_CONNECTION_STRING = f"postgresql://{app_user}:{app_password}@{host}:{port}/vector_db"

embeddings = MistralAIEmbeddings(model="mistral-embed")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50