"""Configuration settings for the application.

This module contains configuration settings for the application, including
model parameters and database settings.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "src" / "config"

# Load model configurations
with open(CONFIG_DIR / "models_config.yaml", "r") as f:
    MODEL_CONFIG = yaml.safe_load(f)

# Vector DB configurations
VECTOR_DBS = ["ChromaDB", "FAISS", "Qdrant"]
DB_PERSIST_DIR = ROOT_DIR / "data" / "vector_stores"
DB_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "nomic-embed-text"
CONTEXT_MODEL = "llama3.2"
QUERY_MODELS: List[str] = ["mistral", "deepseek-r1:8b", "llama3.2:latest", "deepseek-r1:70b","gpt-3.5-turbo"]

# Ollama API configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Database configurations
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Application settings
MAX_TEXT_LENGTH = 30000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 4

# Model configurations
MODEL_CONFIG = {
    "context_models": {
        "llama3.2": {
            "temperature": 0.1,
            "max_tokens": 4096
        }
    },
    "query_models": {
        "mistral": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "deepseek-r1:8b": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "deepseek-r1:70b": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "gpt-3.5-turbo": {
            "temperature": 0.5,
            "max_tokens": 4096
        }
    }
} 