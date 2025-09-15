"""
main.py - Sistema Multi-Agent principale con SyntheticAgent (Acting) + AnalystAgent (ReAct) + NotionWriterAgent
"""

import asyncio
from typing import Dict, Any, Optional

from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.manager import AgentManager
from multi_agent_system.core.messages import create_human_message
from multi_agent_system.core.llm import LlmProvider
from multi_agent_system.agents.notion_agent import NotionAgentFactory
from multi_agent_system.agents.syntetic_agent import SyntheticAgentFactory
from multi_agent_system.agents.analyst_agent import AnalystAgentFactory


class MultiAgentSystem:
    """Sistema multi-agent principale con 3 agent specializzati"""

    def __init__(self, notion_token: str,
                 default_parent_type: str = "page_id",
                 default_parent_id: str = "268bf066457c80d1bd9ef6319f32a165"):
        """
        Inizializza il sistema con tutti gli agent

        Args:
            notion_token: Token di autenticazione Notion
            default_parent_type: Tipo parent di default per Notion
            default_parent_id: ID parent di default per Notion
        """
        # Core components
        self.blackboard = BlackBoard()
        self.llm_client = LlmProvider.IBM_WATSONX.get_instance()
        self.llm_client.set_react_mode(True)  # Default ReAct, override per Acting
        self.manager = AgentManager(self.blackboard, self.llm_client)

        # Configurazione Notion
        self.notion_token = notion_token
        self.default_parent_type = default_parent_type
        self.default_parent_id = default_parent_id

        # Inizializza agenti
        self._setup_agents()

    def _setup_agents(self):
        """Configura e registra tutti gli agenti specializzati"""

        # 1. SyntheticAgent - Acting mode per sintesi descrittive
        synthetic_agent = SyntheticAgentFactory.create_executive_agent(
            agent_id="synthetic_agent",
            blackboard=self.blackboard,
            llm_client=self.llm_client
        )

        # 2. AnalystAgent - ReAct mode per analisi numerica
        analyst_agent = AnalystAgentFactory.create_efficient_analyst_agent(
            agent_id="analyst_agent",
            blackboard=self.blackboard,
            llm_client=self.llm_client
        )

        # 3. NotionWriterAgent - ReAct mode per scrittura su Notion
        notion_agent = NotionAgentFactory.create_agent(
            agent_id="notion_writer",
            blackboard=self.blackboard,
            llm_client=self.llm_client,
            notion_token=self.notion_token,
            default_parent_type=self.default_parent_type,
            default_parent_id=self.default_parent_id
        )

        # Registra tutti gli agent nel manager
        self.manager.register_agent(synthetic_agent)
        self.manager.register_agent(analyst_agent)
        self.manager.register_agent(notion_agent)

        print(f"Sistema inizializzato con {len(self.manager.get_registered_agents())} agent:")
        for agent_id in self.manager.get_registered_agents():
            agent = self.manager.get_agent(agent_id)
            mode = "Acting" if hasattr(agent, '_react_mode') and not agent._react_mode else "ReAct"
            tools_count = len(agent.get_available_tools()) if hasattr(agent, 'get_available_tools') else 0
            print(f"  - {agent_id}: {mode} mode, {tools_count} tools")

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
            return f"Error: {result.get('error', 'Operation failed')}"

    def query_sync(self, user_input: str, user_id: str = "user") -> str:
        """Versione sincrona di query"""
        return asyncio.run(self.query(user_input, user_id))

    def get_status(self) -> Dict[str, Any]:
        """Status completo del sistema"""
        return self.manager.health_check()

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Mostra capabilities di tutti gli agent"""
        capabilities = {}
        for agent_id in self.manager.get_registered_agents():
            agent = self.manager.get_agent(agent_id)
            if hasattr(agent, 'get_capabilities'):
                capabilities[agent_id] = agent.get_capabilities()
        return capabilities

    # ===== FUNZIONI DI DEBUG AGGIORNATE =====

    def debug_blackboard_content(self):
        """Debug completo del contenuto blackboard"""
        print("\n" + "=" * 80)
        print("BLACKBOARD DEBUG - CONTENUTO COMPLETO")
        print("=" * 80)

        # Tutte le chiavi
        all_keys = self.blackboard.keys()
        print(f"Totale chiavi nella blackboard: {len(all_keys)}")

        # Raggruppa per tipo
        task_keys = [k for k in all_keys if k.startswith("task_")]
        obs_keys = [k for k in all_keys if k.startswith("observation_")]
        sys_keys = [k for k in all_keys if k.startswith("system_")]
        other_keys = [k for k in all_keys if not k.startswith(("task_", "observation_", "system_"))]

        print(f"Task keys: {len(task_keys)}")
        print(f"Observation keys: {len(obs_keys)}")
        print(f"System keys: {len(sys_keys)}")
        print(f"Other keys: {len(other_keys)}")

        # Mostra task dettagliatamente per tutti gli agent
        print("\n" + "-" * 60)
        print("TASKS NELLA BLACKBOARD - TUTTI GLI AGENT")
        print("-" * 60)

        for key in task_keys:
            task = self.blackboard.get(key)
            if task:
                agent_id = task.get('assigned_to', 'unknown')
                status = task.get('status', 'unknown')
                task_type = task.get('task_type', 'unknown')
                task_id = task.get('task_id', 'unknown')

                print(f"\nKey: {key}")
                print(f"  Agent: {agent_id}")
                print(f"  Type: {task_type}")
                print(f"  Status: {status}")
                print(f"  Task ID: {task_id}")

                # Mostra task data
                task_data = task.get('task_data', {})
                if isinstance(task_data, dict):
                    request = task_data.get('request', task_data)
                    if isinstance(request, str) and len(request) > 100:
                        print(f"  Request: {request[:100]}...")
                    else:
                        print(f"  Request: {request}")

                # Se completato, mostra risultato
                if status == "completed" and task.get('result'):
                    result = task['result']

                    if isinstance(result, dict) and 'answer' in result:
                        answer = result['answer']
                        preview = answer[:200] + "..." if len(answer) > 200 else answer
                        print(f"  ANSWER (preview): {preview}")
                        print(f"  ANSWER (length): {len(answer)} characters")
                    else:
                        preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                        print(f"  Result: {preview}")

                elif status == "failed":
                    error = task.get('result', {}).get('error', 'Unknown error')
                    print(f"  ERROR: {error}")

        # Mostra observations per agent
        print("\n" + "-" * 60)
        print("OBSERVATIONS NELLA BLACKBOARD")
        print("-" * 60)

        # Raggruppa observations per agent
        agent_observations = {}
        for key in sorted(obs_keys):
            obs = self.blackboard.get(key)
            if obs:
                agent_id = obs.get('agent_id', 'unknown')
                if agent_id not in agent_observations:
                    agent_observations[agent_id] = []
                agent_observations[agent_id].append((key, obs))

        for agent_id, observations in agent_observations.items():
            print(f"\n{agent_id.upper()}:")
            for key, obs in observations:
                step = obs.get('step', 'unknown')
                content = obs.get('observation', 'unknown')
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  Step {step}: {content_preview}")

        print("\n" + "=" * 80)

    def debug_agent_specific_responses(self, agent_id: str):
        """Debug specifico per un agent"""
        print(f"\n" + "=" * 80)
        print(f"{agent_id.upper()} - RISPOSTE DETTAGLIATE")
        print("=" * 80)

        # Trova tutti i task dell'agent specifico
        agent_tasks = []
        for key in self.blackboard.keys():
            if key.startswith(f"task_{agent_id}"):
                task = self.blackboard.get(key)
                if task:
                    agent_tasks.append((key, task))

        print(f"Trovati {len(agent_tasks)} task per {agent_id}")

        for i, (key, task) in enumerate(agent_tasks, 1):
            print(f"\n--- TASK {i} ---")
            print(f"Key: {key}")
            print(f"Status: {task.get('status')}")
            print(f"Task Type: {task.get('task_type')}")
            print(f"Created: {task.get('created_at')}")
            print(f"Completed: {task.get('completed_at', 'N/A')}")

            # Mostra il task data
            task_data = task.get('task_data', {})
            print(f"Task Data: {task_data}")

            # Mostra il risultato completo
            if task.get('result'):
                result = task['result']

                if isinstance(result, dict) and 'answer' in result:
                    answer = result['answer']
                    print(f"RISPOSTA COMPLETA ({len(answer)} caratteri):")
                    print("-" * 40)
                    print(answer)
                    print("-" * 40)
                else:
                    print(f"Result content: {result}")
            else:
                print("Nessun risultato disponibile")

    def debug_all_agent_responses(self):
        """Debug risposte di tutti gli agent"""
        for agent_id in self.manager.get_registered_agents():
            self.debug_agent_specific_responses(agent_id)

    def debug_agent_capabilities(self):
        """Debug delle capabilities di tutti gli agent"""
        print("\n" + "=" * 80)
        print("AGENT CAPABILITIES DEBUG")
        print("=" * 80)

        capabilities = self.get_agent_capabilities()

        for agent_id, caps in capabilities.items():
            print(f"\n--- {agent_id.upper()} ---")
            print(f"Class: {caps.get('class_name', 'Unknown')}")
            print(f"Mode: {caps.get('mode', 'Unknown')}")
            print(f"Description: {caps.get('description', 'No description')}")

            tools = caps.get('tools', [])
            print(f"Tools ({len(tools)}): {', '.join(tools) if tools else 'None'}")

            # Mostra features specifiche
            if 'synthesis_features' in caps:
                features = caps['synthesis_features']
                print(f"Synthesis types: {features.get('synthesis_types', [])[:3]}...")

            if 'analysis_features' in caps:
                features = caps['analysis_features']
                print(f"Analysis types: {features.get('analysis_types', [])[:3]}...")

            if 'notion_features' in caps:
                features = caps['notion_features']
                print(f"Notion operations: {features.get('supported_operations', [])}")

    def debug_task_flow(self):
        """Debug del flusso dei task - mostra l'ordine cronologico"""
        print("\n" + "=" * 80)
        print("TASK FLOW - ORDINE CRONOLOGICO")
        print("=" * 80)

        # Raccogli tutti i task con timestamp
        all_tasks = []
        for key in self.blackboard.keys():
            if key.startswith("task_"):
                task = self.blackboard.get(key)
                if task:
                    created_at = task.get('created_at', '1970-01-01T00:00:00Z')
                    all_tasks.append((created_at, key, task))

        # Ordina per timestamp
        all_tasks.sort(key=lambda x: x[0])

        print(f"Flusso di {len(all_tasks)} task in ordine cronologico:")

        for i, (timestamp, key, task) in enumerate(all_tasks, 1):
            agent = task.get('assigned_to', 'unknown')
            task_type = task.get('task_type', 'unknown')
            status = task.get('status', 'unknown')

            print(f"\n{i}. [{timestamp}] {agent}")
            print(f"   Task: {task_type}")
            print(f"   Status: {status}")
            print(f"   Key: {key}")

            # Se ha risultato, mostra anteprima
            if task.get('result') and isinstance(task['result'], dict) and 'answer' in task['result']:
                answer_preview = task['result']['answer'][:150] + "..."
                print(f"   Result: {answer_preview}")

    def debug_comprehensive(self):
        """Debug completo del sistema"""
        self.debug_blackboard_content()
        self.debug_agent_capabilities()
        self.debug_all_agent_responses()
        self.debug_task_flow()


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
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    DEFAULT_PARENT_ID = os.getenv("NOTION_PARENT_ID")

    if not NOTION_TOKEN or not DEFAULT_PARENT_ID:
        print("ERROR: Missing NOTION_TOKEN or NOTION_PARENT_ID in environment variables")
        exit(1)

    # Inizializza sistema
    system = create_system(NOTION_TOKEN, "page_id", DEFAULT_PARENT_ID)

    print("Sistema multi-agent inizializzato con 3 agent specializzati:")
    capabilities = system.get_agent_capabilities()
    for agent_id, caps in capabilities.items():
        mode = caps.get('mode', 'unknown')
        tools = len(caps.get('tools', []))
        print(f"  - {agent_id}: {mode} mode, {tools} tools - {caps.get('class_name', 'Unknown')}")

    # Query di test che richiede tutti e 3 gli agent
    query = """
        Analyze our Q1-Q3 performance and create executive summary on Notion:

        Q1: Revenue €45K, 12 clients, satisfaction 78%, costs €38K
        Q2: Revenue €52K, 15 clients, satisfaction 82%, costs €41K  
        Q3: Revenue €61K, 18 clients, satisfaction 85%, costs €44K

        Team grew from 8 to 12 people during this period.
        New product launched in Q2 contributes 35% of Q3 revenue.
        Customer retention rate: Q1 85%, Q2 88%, Q3 92%

        Requirements:
        1. Perform statistical analysis on revenue growth and trends
        2. Calculate percentage improvements and growth rates
        3. Generate comprehensive executive summary with insights
        4. Create Notion page titled "Q1-Q3 Executive Performance Review"

        Focus on actionable recommendations for Q4 strategy.
        """

    print("\n" + "=" * 80)
    print("ESEGUENDO QUERY COMPLESSA...")
    print("=" * 80)
    print(f"Query: {query[:200]}...")

    # Esegui la query
    result = system.query_sync(query)

    print(f"\n\nRISPOSTA DEL SISTEMA:")
    print("-" * 40)
    print(result)
    print("-" * 40)

    # DEBUG COMPLETO
    print("\n\n" + "=" * 80)
    print("DEBUG COMPLETO DEL SISTEMA")
    print("=" * 80)

    system.debug_comprehensive()

    # Statistiche finali
    print("\n" + "=" * 80)
    print("STATISTICHE FINALI")
    print("=" * 80)

    health_check = system.get_status()
    print(f"Status manager: {health_check['manager_status']}")
    print(f"Agenti registrati: {health_check['registered_agents']}")
    print(f"Task attivi: {health_check['active_tasks']}")
    print(f"Task completati: {health_check['completed_tasks']}")
    print(f"Entries blackboard: {health_check['blackboard_entries']}")

    # Statistiche dettagliate per agent
    for agent_id in system.manager.get_registered_agents():
        agent = system.manager.get_agent(agent_id)
        if hasattr(agent, 'get_stats'):
            stats = agent.get_stats()
            print(f"\n{agent_id} stats:")
            print(f"  Tasks completed: {stats.get('tasks_completed', 0)}")
            print(f"  Tasks failed: {stats.get('tasks_failed', 0)}")
            print(f"  Tools used: {stats.get('tools_used_count', {})}")

    # Task stats globali
    task_stats = system.blackboard.get_task_stats()
    print(f"\nGlobal task stats:")
    print(f"  Totali: {task_stats['total_tasks']}")
    print(f"  Per status: {task_stats['by_status']}")
    print(f"  Per agent: {task_stats['by_agent']}")
    print(f"  Per tipo: {task_stats['by_type']}")