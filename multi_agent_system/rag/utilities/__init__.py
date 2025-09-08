"""
Utilities package per Gold D Ragger.

Contiene i moduli per la gestione delle connessioni Ollama,
processing delle query, embedding e retrieval con dynamic top-k,
e conversation memory.
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
from ._dynamic_retrieval import (
    calculate_dynamic_k_with_limits,
    classify_query_complexity,
    get_debug_metrics_summary
)

__all__ = [
    'QueryProcessor',
    'OllamaConnection',
    'EmbeddingManager',
    'initialize_conversation_db',
    'store_conversation_pair',
    'retrieve_top_conversations',
    'get_conversation_context',
    'calculate_dynamic_k_with_limits',
    'classify_query_complexity',
    'get_debug_metrics_summary'
]