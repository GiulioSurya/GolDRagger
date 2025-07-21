"""
Utilities package per Gold D Ragger.

Contiene i moduli per la gestione delle connessioni Ollama,
processing delle query, embedding e retrieval, e conversation memory.
"""

from ._process_query import QueryProcessor
from ._connection import OllamaConnection
from ._embending import EmbeddingManager
from ._conversation_memory import (
    initialize_conversation_db,
    store_conversation_pair,
    retrieve_top_conversations,
    get_conversation_context
)

__all__ = [
    'QueryProcessor',
    'OllamaConnection',
    'EmbeddingManager',
    'initialize_conversation_db',
    'store_conversation_pair',
    'retrieve_top_conversations',
    'get_conversation_context'
]