import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST", "localhost")
port = os.getenv("POSTGRES_PORT", "5432")
#for PGVector
CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/vector_db"
#for psycopg2
PSYCOPG2_CONNECTION_STRING = f"postgresql://{user}:{password}@{host}:{port}/vector_db"

embeddings = MistralAIEmbeddings(model="mistral-embed")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50