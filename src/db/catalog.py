import hashlib
import os
from pathlib import Path
from typing import Dict, List
import psycopg2
from datetime import datetime

class DocumentCatalog:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_db()

    def _init_db(self):
        """
        Initialize the database by creating the document_catalog table if it doesn't exist.
        
        :param self: Description
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_catalog (
                        source_id TEXT NOT NULL,
                        user_id UUID NOT NULL,
                        file_path TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        last_modified TIMESTAMP NOT NULL,
                        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        content_hash TEXT NOT NULL,
                        PRIMARY KEY (user_id, source_id)
                    );
                """)

                cur.execute("""
                    ALTER TABLE document_catalog
                    ADD COLUMN IF NOT EXISTS user_id UUID;
                """)

                conn.commit()
                print("Table document_catalog créée.")

    def get_file_hash(self, file_path: str) -> str:
        """
        Calculates the hash of a file to detect modifications.
        :param file_path: Path to the file
        :return: Hash of the file
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def scan_directory(self, directory: str, user_id: str) -> List[Dict]:
        """
        Scans the directory to find files to index.
        Returns a list of dicts with file metadata.
        :param directory: Path to the directory to scan
        :param user_id: the user this scan belongs to
        """
        print(f"Scan of the directory {directory} for user {user_id}...")
        files_info = []
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.pptx', '.xlsx', '.csv']:
                stat = file_path.stat()
                files_info.append({
                    "source_id": os.path.relpath(file_path, start=directory),
                    "user_id": user_id,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix[1:].lower(),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime),
                    "content_hash": self.get_file_hash(str(file_path))
                })
                print(f"File found : {file_path}")
        print(f"Scan completed. {len(files_info)} files found.")
        return files_info
    
    def get_indexed_files(self, user_id: str) -> List[Dict]:
        """
        Retrieves all indexed files belonging to a given user.
        :param user_id: the user whose files to retrieve
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT source_id, file_path, content_hash FROM document_catalog WHERE user_id = %s;",
                (user_id,)
                )
                print(f"{cur.rowcount} indexed files retrieved for user {user_id}.")
                return [
                    {
                        "source_id": row[0],
                        "file_path": row[1],
                        "content_hash": row[2],
                    }
                    for row in cur.fetchall()
                ]


    def add_or_update_file(self, file_info: Dict):
        """
        Adds a new file or updates the metadata of an existing file in the catalog.
        :param file_info: Dictionary containing file metadata (source_id, file_path, file_type, last_modified, content_hash)
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO document_catalog (source_id, user_id, file_path, file_type, last_modified, content_hash)
                      VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, source_id) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        last_modified = EXCLUDED.last_modified,
                        file_type = EXCLUDED.file_type
                    """, (file_info["source_id"], file_info["user_id"], file_info["file_path"], file_info["file_type"], file_info["last_modified"], file_info["content_hash"]))
                conn.commit()

    def delete_file(self, source_id: str, user_id: str):
        """
        Delete a file from the catalog based on its source_id.
        :param source_id: The unique identifier of the file to delete
        :param user_id: the user this file belongs to
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_catalog WHERE source_id = %s AND user_id = %s;", (source_id, user_id))
                conn.commit()