from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import re
from src.ingestion.pipeline import vector_store, init_vector_store
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from lingua import LanguageDetectorBuilder

# Initialise le détecteur de langue une seule fois
detector = LanguageDetectorBuilder.from_all_languages().build()

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
    Nettoie tous les documents en supprimant les caractères NUL du contenu et métadonnées.
    :param docs: Liste des documents à nettoyer
    :return: Liste des documents nettoyés
    """
    for d in docs:
        # Nettoie le contenu
        d.page_content = clean_text(d.page_content)
        
        # Nettoie les métadonnées
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
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
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
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
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
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
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
        loader = DataFrameLoader(df, page_content_column=None)
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
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
    return vector_store

def ingest_unstructured_files(path, source_id=None):
    """
    Ingest an unstructured file, split it into chunks, and index it in the vector store.
    
    :param path: Path to the unstructured file to be ingested
    :param source_id: Unique identifier for the source file
    """
    global vector_store
    if vector_store is None:
        vector_store = init_vector_store()

    loader = UnstructuredFileLoader(path)
    docs = loader.load()
    docs = clean_documents(docs)
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "unstructured"
        d.metadata["language"] = detect_language(d.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    print(f"Indexing of {len(chunks)} chunks...")
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
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
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
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
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
    return vector_store