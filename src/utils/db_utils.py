"""Vector database utilities and connections.

This module provides classes for ChromaDB, FAISS, and Qdrant integration
using langchain.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import chromadb
from langchain_community.vectorstores import Chroma, FAISS, Qdrant
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from src.config.config import (
    DB_PERSIST_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    MODEL_CONFIG,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to meet ChromaDB requirements.

    Args:
        name: Original collection name.

    Returns:
        Sanitized collection name.
    """
    # Replace spaces and invalid characters with underscores
    name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    
    # Ensure it starts and ends with alphanumeric character
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    
    # Ensure minimum length of 3 characters
    if len(name) < 3:
        name = name + "_db"
    
    # Truncate to maximum length of 63 characters
    if len(name) > 63:
        name = name[:63]
        # Ensure it ends with alphanumeric
        name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    
    return name

class VectorDBBase(ABC):
    """Abstract base class for vector database operations."""

    def __init__(self, embedding_model: OllamaEmbeddings, collection_name: str):
        """Initialize the vector database with an embedding model.

        Args:
            embedding_model: OllamaEmbeddings instance for text embeddings.
            collection_name: Name of the collection/database.
        """
        self.embedding_model = embedding_model
        self.collection_name = sanitize_collection_name(collection_name)
        self.vectorstore = None

    @abstractmethod
    def store_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Store documents in the vector database.

        Args:
            texts: List of text strings to store.
            metadatas: Optional list of metadata dictionaries.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search for a query.

        Args:
            query: Query text string.
            k: Number of results to return.

        Returns:
            List of similar documents.
        """
        pass

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get the retriever interface for the vector store.

        Args:
            search_kwargs: Optional search parameters for the retriever.

        Returns:
            Retriever interface for the vector store.
        """
        if search_kwargs is None:
            search_kwargs = {}
        
        # Get total number of documents
        total_docs = len(self.vectorstore.get()) if hasattr(self.vectorstore, 'get') else TOP_K_RESULTS
        TOP_K_RESULTS = 1
        # Ensure k doesn't exceed total documents
        if 'k' in search_kwargs:
            search_kwargs['k'] = min(search_kwargs['k'], total_docs) if total_docs > 0 else search_kwargs['k']
        else:
            search_kwargs['k'] = min(TOP_K_RESULTS, total_docs) if total_docs > 0 else TOP_K_RESULTS
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

class ChromaDBManager(VectorDBBase):
    """ChromaDB vector database manager."""

    def __init__(self, embedding_model: OllamaEmbeddings, collection_name: str):
        """Initialize ChromaDB manager.

        Args:
            embedding_model: OllamaEmbeddings instance for text embeddings.
            collection_name: Name of the collection.
        """
        super().__init__(embedding_model, collection_name)
        persist_dir = DB_PERSIST_DIR / "chroma" / self.collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding_model,
            collection_name=self.collection_name
        )

    def store_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Store documents in ChromaDB."""
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search in ChromaDB."""
        return self.vectorstore.similarity_search(query, k=k)

    @staticmethod
    def list_collections() -> List[str]:
        """List all available ChromaDB collections.

        Returns:
            List of collection names.
        """
        base_dir = DB_PERSIST_DIR / "chroma"
        if not base_dir.exists():
            return []
        return [d.name for d in base_dir.iterdir() if d.is_dir()]

class FAISSManager(VectorDBBase):
    """FAISS vector database manager."""

    def __init__(self, embedding_model: OllamaEmbeddings, collection_name: str):
        """Initialize FAISS manager.

        Args:
            embedding_model: OllamaEmbeddings instance for text embeddings.
            collection_name: Name of the collection.
        """
        super().__init__(embedding_model, collection_name)
        persist_dir = DB_PERSIST_DIR / "faiss" / self.collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = persist_dir / "index.faiss"
        if index_path.exists():
            self.vectorstore = FAISS.load_local(
                str(persist_dir),
                embedding_model,
                index_name=self.collection_name
            )
        else:
            self.vectorstore = FAISS(
                embedding_function=embedding_model,
                docstore={},
                index_name=self.collection_name
            )

    def store_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Store documents in FAISS."""
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        save_dir = DB_PERSIST_DIR / "faiss" / self.collection_name
        self.vectorstore.save_local(str(save_dir))

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search in FAISS."""
        return self.vectorstore.similarity_search(query, k=k)

    @staticmethod
    def list_collections() -> List[str]:
        """List all available FAISS collections.

        Returns:
            List of collection names.
        """
        base_dir = DB_PERSIST_DIR / "faiss"
        if not base_dir.exists():
            return []
        return [d.name for d in base_dir.iterdir() if d.is_dir() and (d / "index.faiss").exists()]

def get_available_databases() -> List[str]:
    """Get a list of all available vector databases."""
    # Check direct ChromaDB directories
    collections = set()
    
    # Look for databases in the main persist directory
    if DB_PERSIST_DIR.exists():
        for path in DB_PERSIST_DIR.iterdir():
            if path.is_dir() and (path / "chroma.sqlite3").exists():
                # Get original name by reversing sanitization if possible
                collections.add(path.name)
    
    # Also check legacy locations if they exist
    if (DB_PERSIST_DIR / "chroma").exists():
        collections.update(ChromaDBManager.list_collections())
    if (DB_PERSIST_DIR / "faiss").exists():
        collections.update(FAISSManager.list_collections())
    
    return sorted(list(collections))

def get_vectordb(name: str, embeddings: Any) -> Any:
    """Get or create a ChromaDB vector store."""
    # Sanitize the collection name first
    sanitized_name = sanitize_collection_name(name)
    
    # Create a persist directory for this specific database
    persist_dir = DB_PERSIST_DIR / sanitized_name
    persist_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create or load ChromaDB
        vectordb = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=sanitized_name,
            collection_metadata={
                "distance_function": "cosine"
            }
        )
        
        # Ensure the database is persisted
        vectordb.persist()
        return vectordb
        
    except Exception as e:
        print(f"Error creating/loading vector database: {str(e)}")
        raise 