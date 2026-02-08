from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
from pipeline import vector_store, init_vector_store
from config import CHUNK_OVERLAP, CHUNK_SIZE

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
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "pdf"

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
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "txt"

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
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "pptx"

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
        for d in sheet_docs:
            d.metadata["source"] = path
            d.metadata["source_id"] = source_id or path
            d.metadata["file_type"] = "excel"
            d.metadata["sheet_name"] = sheet_name 
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
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "unstructured"

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
    for d in docs:
        d.metadata["source"] = path
        d.metadata["source_id"] = source_id or path
        d.metadata["file_type"] = "csv"
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    print(f"Indexing of {len(chunks)} chunks...")
    ids = vector_store.add_documents(chunks)
    print(f"{len(ids)} chunks indexed.")
    return vector_store