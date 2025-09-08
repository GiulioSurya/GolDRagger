from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from utilities import EmbeddingManager
from utilities import QueryProcessor
from utilities import OllamaConnection
from utilities import (
    initialize_conversation_db,
    store_conversation_pair,
    get_conversation_context
)


class GoldDRagger:
    """
    Classe principale per il chatbot RAG.

    Gestisce il flusso conversazionale con messaggi tipizzati LangChain,
    mantiene la storia della conversazione tramite vector database temporaneo
    e orchestra l'interazione tra query processing, embedding, retrieval con
    dynamic top-k e LLM.
    """

    def __init__(self, embedding_model: str = "nomic-embed-text",
                 llm_model: str = "llama3.1:8b") -> None:
        """
        Inizializza il chatbot RAG.

        Args:
            embedding_model: Modello per embedding (default: nomic-embed-text)
            llm_model: Modello LLM (default: llama3.1:8b)
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Conversation memory temporaneo
        self.conversation_db_path: Optional[str] = None
        self.message_counter: int = 0

        # Componenti principali
        self.ollama_connection = None
        self.query_processor = None
        self.embedding_manager = None

        self._initialize_components()
        self._initialize_conversation_memory()

    def _initialize_components(self) -> None:
        """Inizializza tutti i componenti necessari."""
        # Inizializza connessioni
        self.ollama_connection = OllamaConnection("./chroma_db")

        # Inizializza query processor
        self.query_processor = QueryProcessor()

        # Inizializza embedding core con client ChromaDB
        chroma_client = self.ollama_connection.get_chroma_client()
        self.embedding_manager = EmbeddingManager(chroma_client)

    def _initialize_conversation_memory(self) -> None:
        """Inizializza sistema conversation memory temporaneo."""
        self.conversation_db_path = initialize_conversation_db()
        if self.conversation_db_path:
            print("✓ Conversation memory temporaneo inizializzato")
        else:
            print("⚠ Conversation memory non disponibile, continuo senza")

    def _call_llm(self, messages: List[BaseMessage]) -> str:
        """
        Effettua chiamata al modello LLM Ollama con messaggi tipizzati.

        Args:
            messages: Lista di messaggi tipizzati da inviare al modello

        Returns:
            Risposta generata dal modello
        """
        # Ottieni client LLM
        llm_client = self.ollama_connection.get_llm_client()

        # Effettua chiamata con messaggi tipizzati
        response = llm_client.invoke(messages)

        return response.content

    def _build_context_from_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Costruisce il contesto dai risultati del vector store.

        Args:
            results: Risultati dalla ricerca di similarità

        Returns:
            Contesto formattato per il SystemMessage
        """
        context = ""
        for doc in results:
            page_num = doc['metadata'].get('page', 'N/A')
            content = doc['page_content']
            context += f"[Pagina {page_num}]: {content}\n\n"

        return context.strip()

    def _get_base_template(self) -> str:
        """
        Restituisce il template base del SystemMessage.

        Returns:
            Template base senza XML tags dinamici
        """
        return """"You are an expert assistant specialized in Statistics. Your task is to answer
        user questions based exclusively on the provided context and conversational history.
    
        **Generate Response to User Query**
    
        **Step 1: Parse Context Information**
        Extract and utilize relevant knowledge from the provided context within `<context></context>` XML tags.
    
        **Step 2: Consider Previous Conversation**
        If conversation history is available in <conversation_history></conversation_history> XML tags, use it to 
        understand ongoing discussion context, maintain conversation continuity, and answer specific questions 
        about previous exchanges
    
        **Step 3: Analyze User Query**
        Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the
        question, considering both the document context and conversation history.
    
        **Step 4: Determine Response**
        If the answer to the user's query can be directly inferred from the context information or conversation history,
        provide a concise and accurate response in the same language as the user's query.
    
        **Step 5: Handle Uncertainty**
        If the answer is not clear, ask the user for clarification to ensure an accurate response.
    
        **Step 6: Respond in User's Language**
        Maintain consistency by ensuring the response is in the same language as the user's query.
    
        **Step 7: Provide Response**
        Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above."""

    def _build_dynamic_system_message(self, context: str, conversation_context: str) -> SystemMessage:
        """
        Costruisce il SystemMessage dinamico con template RAG, contesto e conversation history.

        Args:
            context: Contesto dai documenti recuperati
            conversation_context: Contesto delle conversazioni precedenti

        Returns:
            SystemMessage con istruzioni RAG, contesto documenti e conversation history
        """
        # Template base
        system_content = self._get_base_template()

        # Aggiungi conversation history se presente (prima del context)
        if conversation_context.strip():
            system_content += f"""

            <conversation_history>
            {conversation_context}
            </conversation_history>"""

        # Aggiungi context se presente
        if context.strip():
            system_content += f"""

            <context>
            {context}
            </context>"""

        return SystemMessage(content=system_content)

    def _format_response_with_sources(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """
        Formatta la risposta includendo le fonti (pagine).

        Args:
            response: Risposta grezza dal LLM
            sources: Documenti sorgente con metadata

        Returns:
            Risposta formattata con indicazioni delle pagine
        """
        # Estrae numeri pagina unici
        pages = set()
        for source in sources:
            page_num = source['metadata'].get('page')
            if page_num:
                pages.add(page_num)

        # Formatta risposta con fonti
        formatted_response = response

        if pages:
            pages_list = sorted(list(pages))
            if len(pages_list) == 1:
                formatted_response += f"\n\n📄 Fonte: Pagina {pages_list[0]}"
            else:
                pages_str = ", ".join(map(str, pages_list))
                formatted_response += f"\n\n📄 Fonti: Pagine {pages_str}"

        return formatted_response

    def _store_conversation_turn(self, user_msg: str, bot_response: str) -> None:
        """
        Salva coppia conversazione nel vector database temporaneo con message_order.

        Args:
            user_msg: Messaggio dell'utente (query originale)
            bot_response: Risposta del bot (verrà divisa in chunk da 300 caratteri)
        """
        if self.conversation_db_path:
            self.message_counter += 1
            success = store_conversation_pair(
                self.conversation_db_path,
                user_msg,
                bot_response,
                self.message_counter
            )
            if not success:
                print("⚠ Impossibile salvare conversazione, continuo senza memory")

    def _build_messages_for_llm(self, user_query: str, context: str, conversation_context: str) -> List[BaseMessage]:
        """
        Costruisce la lista completa di messaggi per il LLM.

        Args:
            user_query: Query corrente dell'utente
            context: Contesto RAG dai documenti recuperati
            conversation_context: Contesto dalle conversazioni precedenti

        Returns:
            Lista completa di messaggi tipizzati per il LLM
        """
        # Lista messaggi da inviare al LLM
        messages = []

        # 1. SystemMessage dinamico con contesto RAG e conversation history
        system_message = self._build_dynamic_system_message(context, conversation_context)
        messages.append(system_message)

        # 2. Query corrente dell'utente (no conversation history in messaggi)
        current_human_message = HumanMessage(content=user_query)
        messages.append(current_human_message)

        return messages

    def _process_user_input(self, user_input: str) -> str:
        """
        Processa l'input dell'utente attraverso la pipeline RAG completa con conversation memory
        e dynamic top-k basato su query complexity.

        Args:
            user_input: Input dell'utente

        Returns:
            Risposta finale del chatbot con fonti
        """
        # 1. Preprocessing della query
        processed_query = self.query_processor.process_query(user_input)

        # 2. Recupero conversation context tramite similarity search
        conversation_context = ""
        if self.conversation_db_path:
            conversation_context = get_conversation_context(self.conversation_db_path, processed_query)

        # 3. Template base per calcolo context window
        base_template = self._get_base_template()

        # 4. Ricerca semantica nei documenti con dynamic top-k
        results = self.embedding_manager.similarity_search(
            processed_query,
            base_template=base_template,
            conversation_context=conversation_context
        )

        # 5. Costruzione contesto RAG dai documenti
        document_context = self._build_context_from_results(results)

        # 6. Costruzione messaggi completi per LLM
        messages = self._build_messages_for_llm(user_input, document_context, conversation_context)

        # 7. Chiamata LLM con messaggi tipizzati
        response = self._call_llm(messages)

        # 8. Formattazione risposta con fonti
        formatted_response = self._format_response_with_sources(response, results)

        # 9. Storage conversazione nel vector database temporaneo
        self._store_conversation_turn(user_input, formatted_response)

        return formatted_response

    def _handle_special_commands(self, user_input: str) -> Optional[str]:
        """
        Gestisce comandi speciali come /help, /clear, /debug, etc.

        Args:
            user_input: Input dell'utente

        Returns:
            Risposta al comando speciale o None
        """
        user_input = user_input.strip().lower()

        if user_input == '/help':
            return """Comandi disponibili:
            /help - Mostra questo messaggio
            /clear - Cancella la conversation memory
            /debug - Mostra info debug ultimo dynamic top-k
            /quit o /exit - Esci dal chat

            Puoi fare domande sul manuale e riceverai risposte basate sui documenti.
            La conversazione precedente viene utilizzata per mantenere il contesto."""

        elif user_input == '/clear':
            # Reinizializza conversation memory
            self._initialize_conversation_memory()
            self.message_counter = 0
            return "Conversation memory cancellata e reinizializzata."

        elif user_input == '/debug':
            # Mostra info debug ultimo dynamic top-k
            debug_summary = self.embedding_manager.get_debug_summary()
            return f"📊 Debug Dynamic Top-K:\n\n{debug_summary}"

        return None

    def _cleanup_on_exit(self) -> None:
        """Cleanup risorse al termine della conversazione."""
        # Il database temporaneo verrà eliminato al prossimo avvio
        pass

    def chat(self) -> None:
        """
        Avvia il loop principale della conversazione.
        """
        print("Ciao, hai domande sul Rif Estimator?")
        print("Digita /help per vedere i comandi disponibili.")

        try:
            while True:
                user_input = input("\nTu: ")

                if user_input.lower() in ['quit', 'exit', 'bye', '/quit', '/exit']:
                    print("Arrivederci!")
                    break

                # Gestione comandi speciali
                special_response = self._handle_special_commands(user_input)
                if special_response:
                    print(f"Bot: {special_response}")
                    continue

                # Pipeline RAG completa con dynamic top-k
                try:
                    response = self._process_user_input(user_input)
                    print(f"Bot: {response}")
                except Exception as e:
                    print(f"Bot: Ops, ho avuto un problema. Riprova.")
                    print(f"Errore: {e}")

        finally:
            # Cleanup garantito anche in caso di interruzione
            self._cleanup_on_exit()


if __name__ == "__main__":
    ragger = GoldDRagger()
    ragger.chat()