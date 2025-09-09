"""
llm.py - Client IBM WatsonX adattato per il sistema multi-agent
Implementazione Singleton con Metaclasse
Compatibile con l'interfaccia richiesta: invoke(system, user)
"""

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import threading
from abc import ABCMeta, abstractmethod, ABC
from enum import Enum

load_dotenv()


class SingletonMeta(ABCMeta):
    """
    Metaclasse singleton thread-safe per garantire una sola istanza
    di ogni classe LLM nel sistema multi-agent.
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Pattern di double-checked locking
                if cls not in cls._instances:
                    cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Llm(ABC, metaclass=SingletonMeta):
    """
    Classe astratta base per tutti i client LLM.
    Usa SingletonMeta per garantire istanze uniche.
    """

    @abstractmethod
    def invoke(self, system: str, user: str) -> str:
        """Metodo principale per invocare il modello con system e user prompt."""
        pass

    @abstractmethod
    def set_react_mode(self, enabled: bool = True):
        """Abilita/disabilita modalità ReAct."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Restituisce informazioni sul modello."""
        pass


@dataclass
class LlmIbm(Llm):
    """
    Wrapper singleton per il modello IBM WatsonX AI utilizzato per il sistema multi-agent.

    Questa classe gestisce la configurazione e l'inizializzazione del modello LLM
    attraverso l'API di IBM WatsonX AI, adattata per ReAct pattern.
    Implementa il pattern Singleton attraverso metaclasse.

    Attributes:
        str_api_key (str): Chiave API per l'autenticazione con IBM WatsonX AI.
        str_project_id (str): ID del progetto IBM WatsonX AI.
        model (ModelInference): Istanza del modello configurata per l'inferenza.
        initialized (bool): Flag per prevenire re-inizializzazione.
    """

    str_api_key: str = None
    str_project_id: str = None
    model: ModelInference = None
    initialized: bool = False

    def __post_init__(self):
        """
        Inizializza il modello IBM WatsonX AI con le credenziali e i parametri necessari.
        Ottimizzato per ReAct pattern nel sistema multi-agent.
        Protetto contro re-inizializzazione grazie al pattern singleton.
        """
        if self.initialized:
            print("[LLM] Istanza già inizializzata (singleton)")
            return

        print("[LLM] Inizializzazione singleton IBM WatsonX...")

        # Carica credenziali da environment
        self.str_api_key = os.getenv("IBM_WATSONX_API_KEY", "la tua api key")
        self.str_project_id = os.getenv("IBM_WATSONX_PROJECT_ID", "il tuo project id")

        if self.str_api_key == "la tua api key" or self.str_project_id == "il tuo project id":
            raise ValueError("Configurare IBM_WATSONX_API_KEY e IBM_WATSONX_PROJECT_ID nel file .env")

        # Configura credenziali IBM
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

        self.initialized = True
        print(f"[LLM] ✅ Inizializzato {self.model.model_id} per sistema multi-agent")

    def invoke(self, system: str, user: str) -> str:
        """
        Metodo principale per il sistema multi-agent.
        Combina system e user prompt nel formato corretto per Llama.

        Args:
            system (str): System prompt con istruzioni ReAct e tool descriptions.
            user (str): User prompt con la richiesta o conversazione.

        Returns:
            str: La risposta generata dal modello.

        Raises:
            RuntimeError: Se il modello non è stato inizializzato.
        """
        if not self.initialized:
            raise RuntimeError("Modello non inizializzato. Chiamare __post_init__() prima.")

        # Formatta per Llama 3.3 instruction format
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            {system.strip()}
                            <|eot_id|><|start_header_id|>user<|end_header_id|>
                            {user.strip()}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            """

        try:
            str_response = self.model.generate_text(prompt=formatted_prompt)
            return str_response.strip()
        except Exception as e:
            print(f"[LLM] ❌ Errore durante invocazione: {e}")
            raise


    def set_react_mode(self, enabled: bool = True):
        """
        Abilita/disabilita modalità ReAct con parametri ottimizzati.

        Args:
            enabled (bool): True per ReAct mode, False per modo standard.
        """
        if not self.initialized:
            print("[LLM] ⚠️ Modello non inizializzato")
            return

        if enabled:
            # Parametri ottimizzati per ReAct
            self.model.params.update({
                "max_new_tokens": 1500,
                "temperature": 0.1,
                "stop_sequences": ["Observation:", "\n\n\n", "User:"],
                "repetition_penalty": 1.05  # Evita ripetizioni
            })
            print("[LLM] ✅ ReAct mode abilitato")
        else:
            # Parametri standard
            self.model.params.update({
                "max_new_tokens": 500,
                "temperature": 0,
                "stop_sequences": ["}"]
            })
            print("[LLM] ✅ ReAct mode disabilitato")

    def set_temperature(self, temperature: float):
        """
        Modifica la temperature del modello.

        Args:
            temperature (float): Valore tra 0 (deterministico) e 1 (creativo).
        """
        if not self.initialized:
            print("[LLM] ⚠️ Modello non inizializzato")
            return

        if 0 <= temperature <= 1:
            self.model.params["temperature"] = temperature
            print(f"[LLM] ✅ Temperature impostata a {temperature}")
        else:
            print(f"[LLM] ❌ Temperature {temperature} non valida (deve essere 0-1)")

    def set_max_tokens(self, max_tokens: int):
        """
        Modifica il numero massimo di token generati.

        Args:
            max_tokens (int): Numero massimo di token (min 1, max 4096).
        """
        if not self.initialized:
            print("[LLM] ⚠️ Modello non inizializzato")
            return

        if 1 <= max_tokens <= 4096:
            self.model.params["max_new_tokens"] = max_tokens
            print(f"[LLM] ✅ Max tokens impostato a {max_tokens}")
        else:
            print(f"[LLM] ❌ Max tokens {max_tokens} non valido (deve essere 1-4096)")

    def get_model_info(self) -> dict:
        """
        Restituisce informazioni sul modello configurato.

        Returns:
            dict: Informazioni sul modello e parametri.
        """
        if not self.initialized:
            return {
                "status": "non_inizializzato",
                "initialized": False
            }

        return {
            "model_id": "meta-llama/llama-3-3-70b-instruct",
            "endpoint": "https://eu-de.ml.cloud.ibm.com",
            "project_id": self.str_project_id,
            "parameters": self.model.params.copy(),
            "initialized": self.initialized,
            "singleton_id": id(self),
            "status": "attivo"
        }

    def reset_connection(self):
        """
        Reset della connessione (utile per debugging).
        """
        if self.initialized:
            print("[LLM] 🔄 Reset connessione...")
            self.initialized = False
            self.model = None
            self.__post_init__()


@dataclass
class LlmOllama(Llm):
    """
    Wrapper singleton per il modello Ollama utilizzato per il sistema multi-agent.

    Questa classe gestisce la configurazione e l'inizializzazione del modello LLM
    attraverso l'API di Ollama, adattata per ReAct pattern.
    Implementa il pattern Singleton attraverso metaclasse.

    Attributes:
        base_url (str): URL base per il server Ollama.
        model_name (str): Nome del modello Ollama da utilizzare.
        llm_client (ChatOllama): Istanza del client ChatOllama.
        initialized (bool): Flag per prevenire re-inizializzazione.
    """

    base_url: str = None
    model_name: str = None
    llm_client = None
    initialized: bool = False

    def __post_init__(self):
        """
        Inizializza il modello Ollama con i parametri necessari.
        Ottimizzato per ReAct pattern nel sistema multi-agent.
        Protetto contro re-inizializzazione grazie al pattern singleton.
        """
        if self.initialized:
            print("[LLM] Istanza già inizializzata (singleton)")
            return

        print("[LLM] Inizializzazione singleton Ollama...")

        # Carica configurazione da environment o usa default
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

        try:
            from langchain_ollama import ChatOllama

            # Parametri ottimizzati per ReAct pattern
            self.llm_client = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.1,
                num_predict=1500,
                stop=["Observation:", "\n\n\n", "User:"]
            )

            # Test connessione
            test_response = self.llm_client.invoke("Test")

            self.initialized = True
            print(f"[LLM] ✅ Inizializzato {self.model_name} per sistema multi-agent")

        except ImportError as e:
            raise ImportError("langchain_ollama non installato. Eseguire 'pip install langchain-ollama'") from e
        except Exception as e:
            print(f"[LLM] ❌ Errore durante inizializzazione Ollama: {e}")
            raise RuntimeError(f"Errore durante inizializzazione Ollama: {e}") from e

    def invoke(self, system: str, user: str) -> str:
        """
        Metodo principale per il sistema multi-agent.
        Combina system e user prompt nel formato corretto per Ollama.

        Args:
            system (str): System prompt con istruzioni ReAct e tool descriptions.
            user (str): User prompt con la richiesta o conversazione.

        Returns:
            str: La risposta generata dal modello.

        Raises:
            RuntimeError: Se il modello non è stato inizializzato.
        """
        if not self.initialized:
            raise RuntimeError("Modello non inizializzato. Chiamare __post_init__() prima.")

        # Formatta per Ollama con system/user separation
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system.strip()),
            HumanMessage(content=user.strip())
        ]

        try:
            response = self.llm_client.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"[LLM] ❌ Errore durante invocazione Ollama: {e}")
            raise

    def set_react_mode(self, enabled: bool = True):
        """
        Abilita/disabilita modalità ReAct con parametri ottimizzati.

        Args:
            enabled (bool): True per ReAct mode, False per modo standard.
        """
        if not self.initialized:
            print("[LLM] ⚠️ Modello non inizializzato")
            return

        if enabled:
            # Parametri ottimizzati per ReAct
            self.llm_client.num_predict = 1500
            self.llm_client.temperature = 0.1
            self.llm_client.stop = ["Observation:", "\n\n\n", "User:"]
            self.llm_client.repeat_penalty = 1.05  # Evita ripetizioni
            print("[LLM] ✅ ReAct mode abilitato")
        else:
            # Parametri standard
            self.llm_client.num_predict = 500
            self.llm_client.temperature = 0
            self.llm_client.stop = ["}"]
            print("[LLM] ✅ ReAct mode disabilitato")

    def set_temperature(self, temperature: float):
        """
        Modifica la temperature del modello.

        Args:
            temperature (float): Valore tra 0 (deterministico) e 1 (creativo).
        """
        if not self.initialized:
            print("[LLM] ⚠️ Modello non inizializzato")
            return

        if 0 <= temperature <= 1:
            self.llm_client.temperature = temperature
            print(f"[LLM] ✅ Temperature impostata a {temperature}")
        else:
            print(f"[LLM] ❌ Temperature {temperature} non valida (deve essere 0-1)")

    def set_max_tokens(self, max_tokens: int):
        """
        Modifica il numero massimo di token generati.

        Args:
            max_tokens (int): Numero massimo di token (min 1, max 4096).
        """
        if not self.initialized:
            print("[LLM] ⚠️ Modello non inizializzato")
            return

        if 1 <= max_tokens <= 4096:
            self.llm_client.num_predict = max_tokens
            print(f"[LLM] ✅ Max tokens impostato a {max_tokens}")
        else:
            print(f"[LLM] ❌ Max tokens {max_tokens} non valido (deve essere 1-4096)")

    def get_model_info(self) -> dict:
        """
        Restituisce informazioni sul modello configurato.

        Returns:
            dict: Informazioni sul modello e parametri.
        """
        if not self.initialized:
            return {
                "status": "non_inizializzato",
                "initialized": False
            }

        return {
            "model_id": self.model_name,
            "base_url": self.base_url,
            "provider": "ollama",
            "parameters": {
                "temperature": getattr(self.llm_client, 'temperature', None),
                "num_predict": getattr(self.llm_client, 'num_predict', None),
                "stop": getattr(self.llm_client, 'stop', None)
            },
            "initialized": self.initialized,
            "singleton_id": id(self),
            "status": "attivo"
        }

    def reset_connection(self):
        """
        Reset della connessione (utile per debugging).
        """
        if self.initialized:
            print("[LLM] 🔄 Reset connessione Ollama...")
            self.initialized = False
            self.llm_client = None
            self.__post_init__()


class LlmProvider(Enum):
    IBM_WATSONX = "ibm_watsonx"  #cosi posso configurarlo da env più avanti
    OLLAMA = "ollama"

    def get_instance(self) -> Llm:
        """Lazy loading delle istanze"""
        if self == LlmProvider.IBM_WATSONX:
            return LlmIbm()
        elif self == LlmProvider.OLLAMA:
            return LlmOllama()