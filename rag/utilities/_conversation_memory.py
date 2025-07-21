"""
Modulo: ConversationMemory
Descrizione: Gestione temporanea della conversation history tramite embedding ChromaDB con chunking delle risposte AI
"""

from typing import List, Dict, Any, Optional
import os
import shutil
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def initialize_conversation_db() -> Optional[str]:
    """
    Crea database temporaneo per conversation history.
    Se esiste già lo elimina e lo ricrea.

    Returns:
        Percorso del database temporaneo creato o None se fallisce
    """
    try:
        temp_dir = "./temp_conversation_db"

        # Se esiste, elimina tutto
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Crea directory pulita
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    except Exception as e:
        print(f"Warning: Impossibile creare DB conversazioni temporaneo: {e}")
        return None


def _chunk_ai_response(ai_response: str, chunk_size: int = 300) -> List[str]:
    """
    Divide la risposta AI in chunk da 300 caratteri usando RecursiveCharacterTextSplitter.

    Args:
        ai_response: Risposta completa dell'AI
        chunk_size: Dimensione dei chunk in caratteri (fisso a 300)

    Returns:
        Lista di chunk della risposta AI
    """
    if not ai_response.strip():
        return []

    # Usa RecursiveCharacterTextSplitter con chunk fisso da 300 caratteri
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(50, chunk_size // 9),  # Stesso overlap dei documenti
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Crea documento temporaneo per splitting
    temp_doc = Document(page_content=ai_response)
    chunks_docs = text_splitter.split_documents([temp_doc])

    # Estrae solo il contenuto testuale
    chunks = [doc.page_content for doc in chunks_docs]
    return chunks


def _validate_conversation_input(user_msg: str, bot_response: str) -> bool:
    """
    Valida input per storage conversazione.

    Args:
        user_msg: Messaggio utente
        bot_response: Risposta del bot

    Returns:
        True se input valido, False altrimenti
    """
    if not isinstance(user_msg, str) or not isinstance(bot_response, str):
        return False

    if not user_msg.strip() or not bot_response.strip():
        return False

    return True


def store_conversation_pair(db_path: str, user_msg: str, bot_response: str, message_order: int) -> bool:
    """
    Salva conversazione dividendo la risposta AI in chunk da 300 caratteri nel vector database temporaneo.

    Args:
        db_path: Percorso database temporaneo
        user_msg: Messaggio dell'utente (query originale)
        bot_response: Risposta del bot da dividere in chunk
        message_order: Ordine progressivo del messaggio nella conversazione

    Returns:
        True se salvato con successo, False altrimenti
    """
    try:
        # Validazione input
        if not _validate_conversation_input(user_msg, bot_response):
            print("Warning: Input conversazione non valido, skip storage")
            return False

        if not db_path or not os.path.exists(db_path):
            return False

        # Inizializza embedding
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        # Inizializza ChromaDB client
        chroma_client = chromadb.PersistentClient(path=db_path)

        # Chunking della risposta AI con dimensione fissa 300 caratteri
        ai_chunks = _chunk_ai_response(bot_response, chunk_size=300)

        if not ai_chunks:
            print("Warning: Nessun chunk generato dalla risposta AI")
            return False

        # Crea documenti per ogni chunk con metadati aggiornati
        documents = []
        for chunk_sequence, chunk_content in enumerate(ai_chunks, 1):
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "message_order": message_order,
                    "user_query": user_msg,
                    "chunk_sequence": chunk_sequence,
                    "total_chunks": len(ai_chunks),
                    "type": "conversation_chunk"
                }
            )
            documents.append(doc)

        # Crea/aggiorna vector store
        vector_store = Chroma(
            client=chroma_client,
            collection_name="temp_conversations",
            embedding_function=embeddings
        )

        # Aggiungi tutti i chunk
        vector_store.add_documents(documents)

        return True

    except Exception as e:
        print(f"Warning: Errore storage conversazione: {e}")
        return False


def retrieve_top_conversations(db_path: str, current_query: str) -> List[Dict[str, Any]]:
    """
    Recupera top-5 chunk di conversazioni più rilevanti tramite similarity search.

    Args:
        db_path: Percorso database temporaneo
        current_query: Query corrente per similarity search

    Returns:
        Lista di max 5 chunk di conversazioni precedenti ordinati per rilevanza
    """
    try:
        if not db_path or not os.path.exists(db_path):
            return []

        if not current_query.strip():
            return []

        # Inizializza embedding
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        # Inizializza ChromaDB client
        chroma_client = chromadb.PersistentClient(path=db_path)

        # Inizializza vector store
        vector_store = Chroma(
            client=chroma_client,
            collection_name="temp_conversations",
            embedding_function=embeddings
        )

        # Verifica se collection ha documenti
        collection = chroma_client.get_collection("temp_conversations")
        if collection.count() == 0:
            return []

        # Similarity search - prendi fino a 5 risultati basati su rilevanza semantica
        max_results = min(5, collection.count())
        results = vector_store.similarity_search(current_query, k=max_results)

        # Converte in formato compatibile
        formatted_results = []
        for doc in results:
            formatted_results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })

        return formatted_results

    except Exception as e:
        print(f"Warning: Errore retrieval conversazioni: {e}")
        return []


def get_conversation_context(db_path: str, current_query: str) -> str:
    """
    Orchestratore principale: recupera chunk rilevanti via similarity e li ordina cronologicamente per message_order.

    Args:
        db_path: Percorso database temporaneo
        current_query: Query corrente per similarity search

    Returns:
        Conversation context formattato per system prompt, ordinato cronologicamente
    """
    try:
        # Recupera top-5 chunk basati su similarity search
        conversation_chunks = retrieve_top_conversations(db_path, current_query)

        if not conversation_chunks:
            return ""

        # Ordina chunk per message_order per mantenere ordine cronologico della conversazione
        sorted_chunks = sorted(conversation_chunks, key=lambda x: (
            x['metadata']['message_order'],
            x['metadata']['chunk_sequence']
        ))

        # Formatta conversazioni per system prompt con ordine cronologico
        context_parts = []
        for chunk in sorted_chunks:
            metadata = chunk['metadata']
            content = chunk['page_content']

            # Formato: Message N - User: question | Assistant: chunk (se multi-chunk)
            message_info = f"Message {metadata['message_order']}"
            if metadata['total_chunks'] > 1:
                message_info += f" (chunk {metadata['chunk_sequence']}/{metadata['total_chunks']})"

            formatted_chunk = f"{message_info}\nUser: {metadata['user_query']}\nAssistant: {content}"
            context_parts.append(formatted_chunk)

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"Warning: Errore formattazione conversation context: {e}")
        return ""