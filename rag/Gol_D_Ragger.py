from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from utilities import EmbeddingManager
from utilities import QueryProcessor
from utilities import OllamaConnection


class GoldDRagger:
    """
    Classe principale per il chatbot RAG.

    Gestisce il flusso conversazionale con messaggi tipizzati LangChain,
    mantiene la storia della conversazione e orchestra l'interazione tra
    query processing, embedding, retrieval e LLM.
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
        # Cronologia solo Human/AI messages (SystemMessage è dinamico)
        self.conversation_history: List[BaseMessage] = []

        # Componenti principali
        self.ollama_connection = None
        self.query_processor = None
        self.embedding_manager = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Inizializza tutti i componenti necessari."""
        # Inizializza connessioni
        self.ollama_connection = OllamaConnection("./chroma_db")

        # Inizializza query processor
        self.query_processor = QueryProcessor()

        # Inizializza embedding manager con client ChromaDB
        chroma_client = self.ollama_connection.get_chroma_client()
        self.embedding_manager = EmbeddingManager(chroma_client)

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

    def _build_dynamic_system_message(self, context: str) -> SystemMessage:
        """
        Costruisce il SystemMessage dinamico con template RAG e contesto.

        Args:
            context: Contesto dai documenti recuperati

        Returns:
            SystemMessage con istruzioni RAG e contesto aggiornato
        """
        system_content = f"""Sei un assistente esperto specializzato in Statistica. Il tuo compito è rispondere
         alle domande degli utenti basandoti esclusivamente sul contesto fornito dal manuale tecnico.

        **Generate Response to User Query**
        
        **Step 1: Parse Context Information**
        Extract and utilize relevant knowledge from the provided context within `<context></context>` XML tags.
        
        **Step 2: Analyze User Query**
        Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the
         question.
        
        **Step 3: Determine Response**
        If the answer to the user's query can be directly inferred from the context information, provide a concise and
         accurate response in the same language as the user's query.
        
        **Step 4: Handle Uncertainty**
        If the answer is not clear, ask the user for clarification to ensure an accurate response.
        
        **Step 5: Avoid Context Attribution**
        When formulating your response, do not indicate that the information was derived from the context.
        
        **Step 6: Respond in User's Language**
        Maintain consistency by ensuring the response is in the same language as the user's query.
        
        **Step 7: Provide Response**
        Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.
        
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

    def _update_conversation_history(self, user_msg: str, bot_response: str) -> None:
        """
        Aggiorna la storia della conversazione con messaggi tipizzati.

        Args:
            user_msg: Messaggio dell'utente
            bot_response: Risposta del bot
        """
        # Aggiunge messaggi tipizzati
        self.conversation_history.append(HumanMessage(content=user_msg))
        self.conversation_history.append(AIMessage(content=bot_response))

        # Mantiene solo ultimi 5 scambi (10 messaggi: 5 Human + 5 AI)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _build_messages_for_llm(self, user_query: str, context: str) -> List[BaseMessage]:
        """
        Costruisce la lista completa di messaggi per il LLM.

        Args:
            user_query: Query corrente dell'utente
            context: Contesto RAG dai documenti recuperati

        Returns:
            Lista completa di messaggi tipizzati per il LLM
        """
        # Lista messaggi da inviare al LLM
        messages = []

        # 1. SystemMessage dinamico con contesto RAG aggiornato
        system_message = self._build_dynamic_system_message(context)
        messages.append(system_message)

        # 2. Cronologia conversazionale (Human/AI alternati)
        messages.extend(self.conversation_history)

        # 3. Query corrente dell'utente
        current_human_message = HumanMessage(content=user_query)
        messages.append(current_human_message)

        return messages

    def _process_user_input(self, user_input: str) -> str:
        """
        Processa l'input dell'utente attraverso la pipeline RAG completa.

        Args:
            user_input: Input dell'utente

        Returns:
            Risposta finale del chatbot con fonti
        """
        # 1. Preprocessing della query
        processed_query = self.query_processor.process_query(user_input)

        # 2. Ricerca semantica
        results = self.embedding_manager.similarity_search(processed_query)

        # 3. Costruzione contesto RAG
        context = self._build_context_from_results(results)

        # 4. Costruzione messaggi completi per LLM
        messages = self._build_messages_for_llm(user_input, context)

        # 5. Chiamata LLM con messaggi tipizzati
        response = self._call_llm(messages)

        # 6. Formattazione risposta con fonti
        formatted_response = self._format_response_with_sources(response, results)

        # 7. Aggiornamento storia conversazione
        self._update_conversation_history(user_input, formatted_response)

        return formatted_response

    def _handle_special_commands(self, user_input: str) -> Optional[str]:
        """
        Gestisce comandi speciali come /help, /clear, etc.

        Args:
            user_input: Input dell'utente

        Returns:
            Risposta al comando speciale o None
        """
        user_input = user_input.strip().lower()

        if user_input == '/help':
            return """Comandi disponibili:
        /help - Mostra questo messaggio
        /clear - Cancella la cronologia conversazione
        /quit o /exit - Esci dal chat
        
        Puoi fare domande sulla tua Ducati e riceverai risposte basate sul manuale."""

        elif user_input == '/clear':
            self.conversation_history = []
            return "Cronologia conversazione cancellata."

        return None

    def chat(self) -> None:
        """
        Avvia il loop principale della conversazione.
        """
        print("Ciao, hai domande sul Rif Estimator?")
        print("Digita /help per vedere i comandi disponibili.")

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

            # Pipeline RAG completa
            try:
                response = self._process_user_input(user_input)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Bot: Ops, ho avuto un problema. Riprova.")
                print(f"Errore: {e}")


if __name__ == "__main__":
    ragger = GoldDRagger()
    ragger.chat()