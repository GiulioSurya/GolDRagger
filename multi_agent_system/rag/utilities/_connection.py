"""
Modulo: OllamaConnection
Descrizione: Gestione configurazione per Ollama e ChromaDB
"""

from typing import Any
import chromadb
from langchain_ollama import ChatOllama
import os


class OllamaConnection:
    """
    Gestisce la configurazione per Ollama e ChromaDB.

    Fornisce accesso al client ChromaDB e al modello LLM Ollama.
    """

    def __init__(self, persist_directory: str = None) -> None:
        """Inizializza la connessione Ollama."""

        self.persist_directory = os.path.abspath(persist_directory)
        self.chroma_client = None
        self.llm_client = None
        self.base_url = "http://localhost:11434"

    def get_chroma_client(self) -> chromadb.Client:
        """
        Restituisce il client ChromaDB.

        Returns:
            Client ChromaDB
        """
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        return self.chroma_client

    def get_llm_client(self) -> ChatOllama:
        """
        Restituisce il client LLM Ollama.

        Returns:
            Client ChatOllama
        """
        if self.llm_client is None:
            self.llm_client = ChatOllama(
                model="llama3.1:8b",
                base_url=self.base_url,
                temperature=0.1
            )
        return self.llm_client

    def close_connections(self) -> None:
        """Chiude tutte le connessioni attive."""
        self.chroma_client = None
        self.llm_client = None