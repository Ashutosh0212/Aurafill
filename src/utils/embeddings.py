"""Embedding model utilities for text embedding operations.

This module handles text embedding operations using the nomic-embed-text model
through langchain.
"""

from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from src.config.config import OLLAMA_BASE_URL

class EmbeddingManager:
    """Manages text embedding operations using nomic-embed-text model."""

    def __init__(self):
        """Initialize the EmbeddingManager with nomic-embed-text model."""
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model="nomic-embed-text",
            model_kwargs={
                "dimensions": 768,
                "max_length": 8192
            }
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query text.

        Args:
            query: Text string to embed.

        Returns:
            Embedding vector for the query.
        """
        return self.embeddings.embed_query(query) 