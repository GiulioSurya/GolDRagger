"""
llm.py - Client IBM WatsonX adattato per il sistema multi-agent
Compatibile con l'interfaccia richiesta: invoke(system, user)
"""

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class LLM(object):
    """
    Wrapper per il modello IBM WatsonX AI utilizzato per il sistema multi-agent.

    Questa classe gestisce la configurazione e l'inizializzazione del modello llm
    attraverso l'API di IBM WatsonX AI, adattata per ReAct pattern.

    Attributes:
        str_api_key (str): Chiave API per l'autenticazione con IBM WatsonX AI.
        str_project_id (str): ID del progetto IBM WatsonX AI.
        model (ModelInference): Istanza del modello configurata per l'inferenza.

    Methods:
        invoke(system, user): Esegue chiamata con system e user prompt separati.
        invoke_legacy(str_prompt): Metodo originale per retrocompatibilità.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLM, cls).__new__(cls)
        return cls._instance

    def __post_init__(self):
        """
        Inizializza il modello IBM WatsonX AI con le credenziali e i parametri necessari.
        Ottimizzato per ReAct pattern nel sistema multi-agent.
        """
        if LLM._initialized:
            return

        self.str_api_key = os.getenv("IBM_WATSONX_API_KEY", "la tua api key")
        self.str_project_id = os.getenv("IBM_WATSONX_PROJECT_ID", "il tuo project id")

        credentials = Credentials(
            url="https://eu-de.ml.cloud.ibm.com",
            api_key=self.str_api_key
        )

        # Parametri ottimizzati per ReAct pattern
        self.model = ModelInference(
            model_id='meta-llama/llama-3-3-70b-instruct',
            params={
                "max_new_tokens": 1500,
                "temperature": 0.1,
                "stop_sequences": ["\n\n\n", "User:", "Human:"]
            },
            credentials=credentials,
            project_id=self.str_project_id,
        )

        LLM._initialized = True
        print(f"[LLM] Inizializzato Llama 3.3 70B per sistema multi-agent")

    def invoke(self, system: str, user: str) -> str:
        """
        Metodo principale per il sistema multi-agent.
        Combina system e user prompt nel formato corretto per Llama.

        Args:
            system (str): System prompt con istruzioni ReAct e tool descriptions.
            user (str): User prompt con la richiesta o conversazione.

        Returns:
            str: La risposta generata dal modello.
        """
        # Formatta per Llama 3.3 instruction format
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # # Log per debug (commentare in produzione)
        # if len(user) < 500:  # Log solo per prompt brevi
        #     print(f"\n[LLM] Invocazione:")
        #     print(f"  System: {len(system)} chars")
        #     print(f"  User: {user[:200]}...")

        # Genera risposta
        str_response = self.model.generate_text(prompt=formatted_prompt)

        # # Log risposta (commentare in produzione)
        # if str_response:
        #     print(f"[LLM] Risposta: {str_response[:200]}...")

        return str_response


    def set_react_mode(self, enabled: bool = True):
        """
        Abilita/disabilita modalità ReAct con parametri ottimizzati.

        Args:
            enabled (bool): True per ReAct mode, False per modo standard.
        """
        if enabled:
            # Parametri ottimizzati per ReAct
            self.model.params = {
                "max_new_tokens": 1500,
                "temperature": 0.1,
                "stop_sequences": ["Observation:", "\n\n\n", "User:"],
                "repetition_penalty": 1.05  # Evita ripetizioni
            }
            print("[LLM] ReAct mode abilitato")
        else:
            # Parametri standard
            self.model.params = {
                "max_new_tokens": 500,
                "temperature": 0,
                "stop_sequences": ["}"]
            }
            print("[LLM] ReAct mode disabilitato")

    def set_temperature(self, temperature: float):
        """
        Modifica la temperature del modello.

        Args:
            temperature (float): Valore tra 0 (deterministico) e 1 (creativo).
        """
        if 0 <= temperature <= 1:
            self.model.params["temperature"] = temperature
            print(f"[LLM] Temperature impostata a {temperature}")
        else:
            print(f"[LLM] Temperature {temperature} non valida (deve essere 0-1)")

    def set_max_tokens(self, max_tokens: int):
        """
        Modifica il numero massimo di token generati.

        Args:
            max_tokens (int): Numero massimo di token (min 1, max 4096).
        """
        if 1 <= max_tokens <= 4096:
            self.model.params["max_new_tokens"] = max_tokens
            print(f"[LLM] Max tokens impostato a {max_tokens}")
        else:
            print(f"[LLM] Max tokens {max_tokens} non valido (deve essere 1-4096)")

    def get_model_info(self) -> dict:
        """
        Restituisce informazioni sul modello configurato.

        Returns:
            dict: Informazioni sul modello e parametri.
        """
        return {
            "model_id": "meta-llama/llama-3-3-70b-instruct",
            "endpoint": "https://eu-de.ml.cloud.ibm.com",
            "project_id": self.str_project_id,
            "parameters": self.model.params,
            "initialized": LLM._initialized
        }


# Test del client se eseguito direttamente
if __name__ == "__main__":
    print("=" * 60)
    print("TEST CLIENT IBM WATSONX PER SISTEMA MULTI-AGENT")
    print("=" * 60)

    # Inizializza client
    llm = LLM()

    # Test con formato sistema multi-agent
    system_prompt = """You are an AI agent using the ReAct pattern.

    AVAILABLE TOOLS:
    - create_notion_page: Create a new page in Notion
    - update_notion_page: Update an existing page

    Follow this format:
    Thought: Your reasoning
    Action: tool_name: {"param": "value"}
    PAUSE
    """

    user_prompt = "Create a meeting notes page for today's standup"

    print("\n📝 Test invocazione con system/user separati:")
    response = llm.invoke(system_prompt, user_prompt)
    print(f"\n✅ Risposta: {response}")

    # Info modello
    print("\n📊 Info modello:")
    info = llm.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")