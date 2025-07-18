"""
Modulo: EmbeddingManager
Descrizione: Gestisce embedding query e retrieval da ChromaDB
"""

from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb


class EmbeddingManager:
    """
    Gestisce l'embedding delle query e la ricerca nel vector store.

    Utilizza OllamaEmbeddings per convertire le query in embedding
    e ChromaDB per la ricerca di similarità nel database vettoriale.
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

    def similarity_search(self, query: str, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Effettua ricerca di similarità nel vector store con threshold dinamico.

        Args:
            query: Query processata da cercare
            similarity_threshold: Soglia di similarità (default: 0.7, range: 0.0-1.0)
                                 Valori più bassi = più selettivo, più alti = meno selettivo

        Returns:
            Lista di documenti con contenuto, metadata e relevance_score.
            Minimo 3 risultati, massimo 10, filtrati per threshold.
        """
        # Ottieni più risultati del necessario per poter filtrare (massimo possibile: 50)
        max_candidates = 50
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query, k=max_candidates
        )

        # Filtra per threshold di similarità
        filtered_results = [
            (doc, score) for doc, score in results_with_scores
            if score <= similarity_threshold
        ]

        # Applica vincoli min/max
        if len(filtered_results) < 3:
            # Se meno di 3 risultati, prendi i migliori 3 indipendentemente dal threshold
            filtered_results = results_with_scores[:3]
        elif len(filtered_results) > 10:
            # Se più di 10 risultati, prendi i migliori 10
            filtered_results = filtered_results[:10]

        # Converte in formato dict per compatibilità
        formatted_results = []
        for doc, score in filtered_results:
            formatted_results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })

        return formatted_results

    def get_embedding_for_query(self, query: str) -> List[float]:
        """
        Genera embedding per una query.

        Args:
            query: Query da convertire in embedding

        Returns:
            Vettore embedding della query
        """
        return self.embeddings.embed_query(query)