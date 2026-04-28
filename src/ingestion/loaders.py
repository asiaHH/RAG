from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import re
import hashlib
import psycopg2
from src.ingestion.pipeline import vector_store, init_vector_store
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, PSYCOPG2_CONNECTION_STRING
from lingua import LanguageDetectorBuilder

# Initialize the language detector once
detector = LanguageDetectorBuilder.from_all_languages().build()

def get_chunk_hash(content: str) -> str:
    """
    Calculate the hash of content of a chunk for the unique identification.
    :param content: Content of the chunk
    :return: Hexadecimal hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def chunk_exists_in_db(chunk_hash: str) -> bool:
    """
    Verify if a chunk_hash already exists in the database via direct SQL query.
    :param chunk_hash: The hash of the chunk to verify
    :return: True if the chunk exists, False otherwise
    """
    try:
        with psycopg2.connect(PSYCOPG2_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM langchain_pg_embedding WHERE cmetadata->>'chunk_hash' = %s LIMIT 1",
                    (chunk_hash,)
                )
                return cur.fetchone() is not None
    except Exception as e:
        print(f"Erreur lors de la vérification du chunk: {e}")
        return False

def delete_chunk_by_hash(chunk_hash: str) -> bool:
    """
    Delete a chunk from the database via its hash.
    :param chunk_hash: The hash of the chunk to delete
    :return: True if deleted, False otherwise
    """
    try:
        with psycopg2.connect(PSYCOPG2_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'chunk_hash' = %s",
                    (chunk_hash,)
                )
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Erreur lors de la suppression du chunk: {e}")
        return False

def upsert_chunks(chunks):
    """
    Inserts or updates chunks in the vector store using the hash as a unique key.
    Uses a direct SQL query to check for existence (without embedding).
    :param chunks: List of document chunks to insert
    :return: Number of chunks inserted/updated
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    inserted_count = 0
    updated_count = 0

    for chunk in chunks:
        chunk_hash = get_chunk_hash(chunk.page_content)
        
        # add the hash in metadata for tracking
        chunk.metadata["chunk_hash"] = chunk_hash
        
        # Verify if this chunk already exists via direct SQL
        if chunk_exists_in_db(chunk_hash):
            # chunk already exists, we delete it and re-insert it
            delete_chunk_by_hash(chunk_hash)
            updated_count += 1
        else:
            inserted_count += 1
        
        # insert the chunk (new or updated)
        vector_store.add_documents([chunk])
    
    print(f"Chunks traités : {inserted_count} insérés, {updated_count} mis à jour")
    return inserted_count + updated_count

def detect_language(text):
    """
    Detects the language of a text.
    :param text: Text to analyze
    :return: ISO code of the language (e.g., 'fr', 'en') or 'unknown' if not detectable
    """
    try:
        if not text or len(text.strip()) < 10:  # Besoin d'au moins 10 caractères
            return "unknown"
        lang = detector.detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else "unknown"
    except Exception as e:
        print(f"Erreur lors de la détection de langue: {e}")
        return "unknown"

def clean_text(text):
    """
    Removes problematic characters and normalizes the text.
    - Null characters
    - Invisible characters (Zero-Width)
    - Multiple spaces
    :param text: Text to clean
    :return: Cleaned text
    """
    if not text:
        return text
    
    #Delete null characters
    text = text.replace('\x00', '').replace('\0', '')

    #Delete control characters ASCII (except for common whitespace)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    
    #Delete invisible characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    #Escape backslashes that are not already escaped
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    
    #normalize multiple spaces to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def clean_documents(docs):
    """
    Cleans all documents by removing NUL characters from content and metadata.
    :param docs: List of documents to clean
    :return: List of cleaned documents
    """
    for d in docs:
        # clean the content
        d.page_content = clean_text(d.page_content)
        
        # Clean the metadata
        for key, value in d.metadata.items():
            if isinstance(value, str):
                d.metadata[key] = clean_text(value)
    
    return docs

def ingest_pdf(path, source_id=None):
    """
    Ingest a PDF file, split it into chunks, and index it in the vector store.
    :param path: Path to the PDF file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    loader = PyPDFLoader(path)
    docs = loader.load()
    docs = clean_documents(docs)
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "pdf"
        d.metadata["language"] = detect_language(d.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    print(f"Indexing of {len(chunks)} chunks...")
    upsert_chunks(chunks)
    print(f"Chunks processed with upsert logic.")
    return vector_store

def ingest_txt(path, source_id=None):
    """
    Ingest a TXT file, split it into chunks, and index it in the vector store.
    :param path: Path to the TXT file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    loader = TextLoader(path)
    docs = loader.load()
    docs = clean_documents(docs)
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "txt"
        d.metadata["language"] = detect_language(d.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    print(f"Indexing of {len(chunks)} chunks...")
    upsert_chunks(chunks)
    print(f"Chunks processed with upsert logic.")
    return vector_store

def ingest_pptx(path, source_id=None):
    """
    Ingest a PPTX file, split it into chunks, and index it in the vector store.
    :param path: Path to the PPTX file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    loader = UnstructuredPowerPointLoader(path, mode ="elements")
    docs = loader.load()
    docs = clean_documents(docs)
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "pptx"
        d.metadata["language"] = detect_language(d.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    print(f"Indexing of {len(chunks)} chunks...")
    upsert_chunks(chunks)
    print(f"Chunks processed with upsert logic.")
    return vector_store

def ingest_excel(path, source_id=None):
    """
    Ingest an Excel file, split it into chunks, and index it in the vector store.
    :param path: Path to the Excel file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    xls = pd.ExcelFile(path)
    docs = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # concat all columns into a single text string
        df['_text'] = df.astype(str).agg(' | '.join, axis=1)
        loader = DataFrameLoader(df, page_content_column="_text")
        sheet_docs = loader.load()
        sheet_docs = clean_documents(sheet_docs)
        for d in sheet_docs:
            d.metadata["source"] = path
            d.metadata["source_id"] = source_id or path
            d.metadata["file_type"] = "excel"
            d.metadata["sheet_name"] = sheet_name
            d.metadata["language"] = detect_language(d.page_content) 
        docs.extend(sheet_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    print(f"Indexing of {len(chunks)} chunks...")
    upsert_chunks(chunks)
    print(f"Chunks processed with upsert logic.")
    return vector_store

def ingest_csv(path, source_id=None):
    """
    Ingest a CSV file, split it into chunks, and index it in the vector store.
    :param path: Path to the CSV file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    loader = CSVLoader(path, encoding="utf-8")
    docs = loader.load()
    docs = clean_documents(docs)
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "csv"
        d.metadata["language"] = detect_language(d.page_content)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    print(f"Indexing of {len(chunks)} chunks...")
    upsert_chunks(chunks)
    print(f"Chunks processed with upsert logic.")
    return vector_store

def ingest_docx(path, source_id=None):
    """
    Ingest a DOCX file, split it into chunks, and index it in the vector store.
    :param path: Path to the DOCX file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    loader = UnstructuredWordDocumentLoader(path)
    docs = loader.load()
    docs = clean_documents(docs)
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "docx"
        d.metadata["language"] = detect_language(d.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    print(f"Indexing of {len(chunks)} chunks...")
    upsert_chunks(chunks)
    print(f"Chunks processed with upsert logic.")
    return vector_store