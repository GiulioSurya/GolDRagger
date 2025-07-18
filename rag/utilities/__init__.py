"""
Utilities package per Gold D Ragger.

Contiene i moduli per la gestione delle connessioni Ollama,
processing delle query, embedding e retrieval.
"""

from ._process_query import QueryProcessor
from ._connection import OllamaConnection
from ._embending import EmbeddingManager

__all__ = [
    'QueryProcessor',
    'OllamaConnection',
    'EmbeddingManager'
]