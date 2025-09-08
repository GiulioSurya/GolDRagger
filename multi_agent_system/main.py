"""
main.py - Sistema Multi-Agent principale con NotionWriterAgent
"""

import asyncio
from typing import Dict, Any, Optional

from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.manager import AgentManager
from multi_agent_system.core.messages import create_human_message
from multi_agent_system.core.llm import LLM
from multi_agent_system.agents.notion_agent import NotionAgentFactory


class MultiAgentSystem:
    """Sistema multi-agent principale"""

    def __init__(self, notion_token: str,
                 default_parent_type: str = "page_id",
                 default_parent_id: str = "268bf066457c80d1bd9ef6319f32a165"):
        """
        Inizializza il sistema

        Args:
            notion_token: Token di autenticazione Notion
            default_parent_type: Tipo parent di default
            default_parent_id: ID parent di default
        """
        # Core components
        self.blackboard = BlackBoard()
        self.llm_client = LLM()
        self.llm_client.__post_init__()
        self.llm_client.set_react_mode(True)
        self.manager = AgentManager(self.blackboard, self.llm_client)

        # Configurazione Notion
        self.notion_token = notion_token
        self.default_parent_type = default_parent_type
        self.default_parent_id = default_parent_id

        # Inizializza agenti
        self._setup_agents()

    def _setup_agents(self):
        """Configura e registra gli agenti"""
        # Crea NotionWriterAgent
        notion_agent = NotionAgentFactory.create_agent(
            agent_id="notion_writer",
            blackboard=self.blackboard,
            llm_client=self.llm_client,
            notion_token=self.notion_token,
            default_parent_type=self.default_parent_type,
            default_parent_id=self.default_parent_id
        )

        # Registra nel manager
        self.manager.register_agent(notion_agent)

    async def query(self, user_input: str, user_id: str = "user") -> str:
        """
        Elabora una query utente tramite il sistema multi-agent

        Args:
            user_input: Richiesta dell'utente
            user_id: ID dell'utente

        Returns:
            str: Risposta del sistema
        """
        # Crea messaggio utente
        human_message = create_human_message(
            user_id=user_id,
            content=user_input
        )

        # Elabora tramite manager
        result = await self.manager.handle_user_request(human_message)

        if result["success"]:
            return result["response"]
        else:
            return f"Errore: {result.get('error', 'Operazione fallita')}"

    def query_sync(self, user_input: str, user_id: str = "user") -> str:
        """Versione sincrona di query"""
        return asyncio.run(self.query(user_input, user_id))

    def get_status(self) -> Dict[str, Any]:
        """Status del sistema"""
        return self.manager.health_check()


# Funzione di utilità per inizializzazione rapida
def create_system(notion_token: str,
                  default_parent_type: str = "page_id",
                  default_parent_id: str = "268bf066457c80d1bd9ef6319f32a165") -> MultiAgentSystem:
    """
    Crea e inizializza il sistema multi-agent

    Args:
        notion_token: Token Notion
        default_parent_type: Tipo parent di default
        default_parent_id: ID parent di default

    Returns:
        MultiAgentSystem: Sistema inizializzato
    """
    return MultiAgentSystem(notion_token, default_parent_type, default_parent_id)


# Test principale
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Configurazione
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")  # Il tuo token Notion
    DEFAULT_PARENT_ID = os.getenv("NOTION_PARENT_ID")  # ID della pagina o database di default

    # Inizializza sistema
    system = create_system(NOTION_TOKEN, "page_id", DEFAULT_PARENT_ID)

    # Test queries
    query = "scrivo una pagina su notion in cui elenchi le 10 migliori città italiane"

    system.query_sync(query)


