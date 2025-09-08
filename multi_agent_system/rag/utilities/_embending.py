"""
Modulo: EmbeddingManager
Descrizione: Gestisce embedding query e retrieval da ChromaDB con dynamic top-k
"""

from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb
from ._dynamic_retrieval import calculate_dynamic_k_with_limits


class EmbeddingManager:
    """
    Gestisce l'embedding delle query e la ricerca nel vector store.

    Utilizza OllamaEmbeddings per convertire le query in embedding,
    ChromaDB per la ricerca di similarità e sistema dynamic top-k
    basato su query complexity e context window awareness.
    """

    def __init__(self, chroma_client: chromadb.Client, persist_directory: str = ".rag/chroma_db") -> None:
        """
        Inizializza il gestore degli embedding.

        Args:
            chroma_client: Client ChromaDB
            persist_directory: Directory ChromaDB per debug
        """
        self.chroma_client = chroma_client
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vector_store = None

        # Metriche debug accessibili (per debugging)
        self.last_dynamic_k_info: Optional[Dict[str, Any]] = None

        self._initialize_ollama_embeddings()
        self._initialize_vector_store()

    def _initialize_ollama_embeddings(self) -> None:
        """Inizializza OllamaEmbeddings con modello nomic-embed-text."""
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

    def _initialize_vector_store(self) -> None:
        """Inizializza ChromaDB per accesso al vector store."""
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="rag_chunks_ita",
            embedding_function=self.embeddings
        )

    def _get_total_documents_count(self) -> int:
        """
        Ottiene il numero totale di documenti nel vector store.

        Returns:
            Numero totale di documenti
        """
        try:
            collection = self.chroma_client.get_collection("rag_chunks_ita")
            return collection.count()
        except Exception as e:
            print(f"Warning: Impossibile ottenere conteggio documenti: {e}")
            return 100  # Fallback

    def similarity_search(self,
                         query: str,
                         base_template: str = "",
                         conversation_context: str = "") -> List[Dict[str, Any]]:
        """
        Effettua ricerca di similarità con top-k dinamico basato su query complexity.

        Args:
            query: Query processata da cercare
            base_template: Template base SystemMessage (per calcolo context window)
            conversation_context: Contesto conversazione (per calcolo context window)

        Returns:
            Lista di documenti con contenuto, metadata e relevance_score.
            Numero documenti determinato dinamicamente da query complexity.
        """
        # 1. Ottieni conteggio totale documenti
        total_documents = self._get_total_documents_count()

        # 2. Calcola k dinamico con context window awareness
        k_dynamic, debug_info = calculate_dynamic_k_with_limits(
            query=query,
            total_documents=total_documents,
            base_template=base_template,
            conversation_context=conversation_context,
            model_context_window=2048  # Ollama default
        )

        # 3. Salva debug info per troubleshooting
        self.last_dynamic_k_info = debug_info

        # 4. Effettua similarity search con k dinamico
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query, k=k_dynamic
        )

        # 5. Converte in formato dict compatibile con il sistema esistente
        formatted_results = []
        for doc, score in results_with_scores:
            formatted_results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })

        return formatted_results

    def get_last_dynamic_k_debug_info(self) -> Optional[Dict[str, Any]]:
        """
        Restituisce le ultime metriche debug del sistema dynamic k.

        Utile per debugging e analisi delle performance.

        Returns:
            Dizionario con metriche debug o None se non disponibili
        """
        return self.last_dynamic_k_info

    def get_debug_summary(self) -> str:
        """
        Restituisce un summary leggibile delle ultime metriche debug.

        Returns:
            Stringa formattata con summary debug
        """
        if not self.last_dynamic_k_info:
            return "Nessuna informazione debug disponibile"

        from ._dynamic_retrieval import get_debug_metrics_summary
        return get_debug_metrics_summary(self.last_dynamic_k_info)

    def get_embedding_for_query(self, query: str) -> List[float]:
        """
        Genera embedding per una query.

        Args:
            query: Query da convertire in embedding

        Returns:
            Vettore embedding della query
        """
        return self.embeddings.embed_query(query)