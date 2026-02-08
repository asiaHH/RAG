from src.db.catalog import DocumentCatalog
from src.config import PSYCOPG2_CONNECTION_STRING
from src.ingestion.loaders import ingest_pdf, ingest_txt, ingest_pptx, ingest_excel, ingest_csv, ingest_unstructured_files
from src.ingestion import pipeline

def sync_collection(directory: str = "data"):
    """
    Synchronize the vector store with the files in the specified directory. 
    It detects new, modified, and deleted files and updates the vector store accordingly.
    :param directory: The directory to synchronize (default is "data")
    :return: Updated vector store
    """
    global vector_store
    print(f"SYNCHRONIZATION of the directory {directory}...")

    catalog = DocumentCatalog(PSYCOPG2_CONNECTION_STRING)

    current_files = catalog.scan_directory(directory)
    if not current_files:
        print("No files found in the directory.")
        return pipeline.vector_store

    indexed_files = catalog.get_indexed_files()
    indexed_set = {f["source_id"]for f in indexed_files}

    to_add = [] 
    to_delete = []

    for file_info in current_files:
        if file_info["source_id"] not in indexed_set:
            to_add.append(file_info)
            print(f"New File: {file_info['file_path']}")
        else:
            existing_file = next(f for f in indexed_files if f["source_id"] == file_info["source_id"])
            if existing_file["content_hash"] != file_info["content_hash"]:
                to_add.append(file_info)
                to_delete.append(file_info["source_id"])
                print(f"Modified File: {file_info['file_path']}")

    current_set = {f["source_id"] for f in current_files}
    for indexed in indexed_files:
        if indexed["source_id"] not in current_set:
            to_delete.append(indexed["source_id"])
            print(f"Deleted File: {indexed['file_path']}")
    print(f"{len(to_add)} to add, {len(to_delete)} to delete")

    if pipeline.vector_store is None:
        pipeline.init_vector_store()
    
    for source_id in set(to_delete):
        print(f"Removal of {source_id}...")
        pipeline.vector_store.delete(where={"source_id": source_id})
        catalog.delete_file(source_id)

    for file_info in to_add:
        catalog.add_or_update_file(file_info)
        path = file_info["file_path"]

        if path.endswith(".pdf"):
            ingest_pdf(file_info["file_path"], source_id=file_info["source_id"])
        elif path.endswith(".txt"):
            ingest_txt(file_info["file_path"], source_id=file_info["source_id"])
        elif path.endswith(".pptx"):
            ingest_pptx(file_info["file_path"], source_id=file_info["source_id"])
        elif path.endswith(".xlsx"):
            ingest_excel(file_info["file_path"], source_id=file_info["source_id"])
        elif path.endswith(".csv"):
            ingest_csv(file_info["file_path"], source_id=file_info["source_id"])
        else:
            ingest_unstructured_files(file_info["file_path"], source_id=file_info["source_id"])

    print("Synchronization completed.")
    return pipeline.vector_store