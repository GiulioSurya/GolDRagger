"""
Manager principale del sistema multi-agent.
Orchestratore che coordina gli agent tramite blackboard usando ReAct pattern.
"""

import re
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum


from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.messages import HumanMessage, create_agent_result
from multi_agent_system.core.base_agent import BaseAgent


class ManagerStatus(Enum):
    """Stati del Manager"""
    IDLE = "idle"
    ORCHESTRATING = "orchestrating"
    WAITING_AGENTS = "waiting_agents"
    ERROR = "error"


class AgentManager:
    """
    Manager principale che orchestra gli agent del sistema.

    NON eredita da BaseAgent ma implementa proprio ReAct loop specifico.
    Gestisce registrazione agent, routing task e monitoring esecuzione.
    """

    # System prompt ReAct per orchestrazione
    MANAGER_REACT_PROMPT = """You are the Manager Agent, the orchestrator of a multi-agent system.

Your role is to analyze user requests and delegate tasks to specialized agents through a blackboard system.

You operate using the ReAct (Reasoning and Acting) pattern:

WORKFLOW:
1. Thought: Analyze the user request and determine which agent(s) to use
2. Action: Create a task for the appropriate agent using the exact format shown
3. PAUSE: Wait for the agent to complete the task
4. Observation: Review the task result from the agent
5. Continue until you have all information needed
6. Answer: Provide the final synthesized response to the user

ACTION FORMAT for creating tasks:
Action: create_task: {{"agent_id": "agent_name", "task_type": "type", "task_data": {{"key": "value"}}}}

ACTION FORMAT for checking task status:
Action: check_task: {{"task_id": "task_uuid", "agent_id": "agent_name"}}

IMPORTANT RULES:
- You can only delegate to agents that are registered (see AVAILABLE AGENTS below)
- Always wait for task completion before creating dependent tasks
- If an agent fails, try alternative approaches or agents
- Synthesize results from multiple agents when needed
- ALWAYS end with Answer: when you have the final response

TERMINATION RULES:
- When you have received all needed information from agents, immediately provide Answer:
- Do NOT create additional actions if you already have the complete answer
- Do NOT use "Action:" unless you need to create_task or check_task
- After receiving Observation with complete results, go directly to Answer:
- If you have sufficient information to answer the user, do not delay - provide Answer:

AVAILABLE AGENTS:
{agents_description}

Remember:
- Each agent has specific capabilities listed above
- Use agent_id exactly as shown in the list
- Match task requirements to agent capabilities
- You can chain multiple agents for complex requests
- TERMINATE with Answer: as soon as you have complete information"""

    def __init__(self, blackboard: BlackBoard, llm_client):
        """
        Inizializza il Manager.

        Args:
            blackboard: Blackboard condivisa per comunicazione
            llm_client: Client LLM per ReAct reasoning
        """
        self.manager_id = "manager"
        self.blackboard = blackboard
        self.llm_client = llm_client

        # Registry degli agent
        self._registered_agents: Dict[str, BaseAgent] = {}
        self._agent_capabilities: Dict[str, Dict] = {}

        # Stato e tracking
        self._status = ManagerStatus.IDLE
        self._active_tasks: Dict[str, Dict] = {}  # task_id -> task_info
        self._completed_tasks: List[str] = []

        # Statistiche
        self._stats = {
            "requests_handled": 0,
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "agents_used": {},
            "total_orchestration_time": 0.0,
            "last_activity": None
        }

    def register_agent(self, agent: BaseAgent) -> bool:
        """
        Registra un agent nel sistema.

        Args:
            agent: Agent da registrare

        Returns:
            bool: True se registrato con successo
        """
        try:
            agent_id = agent.agent_id

            # Verifica duplicati
            if agent_id in self._registered_agents:
                return False

            # Registra agent
            self._registered_agents[agent_id] = agent

            # Estrai e salva capabilities
            capabilities = self._extract_agent_capabilities(agent)
            self._agent_capabilities[agent_id] = capabilities

            # Inizializza agent se necessario
            if hasattr(agent, 'initialize'):
                agent.initialize()

            return True

        except Exception as e:
            print(f"Failed to register agent: {str(e)}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Rimuove un agent dal sistema.

        Args:
            agent_id: ID dell'agent da rimuovere

        Returns:
            bool: True se rimosso con successo
        """
        if agent_id not in self._registered_agents:
            return False

        # Rimuovi da registri
        del self._registered_agents[agent_id]
        del self._agent_capabilities[agent_id]

        return True

    def _extract_agent_capabilities(self, agent: BaseAgent) -> Dict:
        """
        Estrae capabilities di un agent per il system prompt.

        Args:
            agent: Agent da cui estrarre capabilities

        Returns:
            Dict: Capabilities strutturate
        """
        capabilities = {
            "agent_id": agent.agent_id,
            "class_name": agent.__class__.__name__,
            "description": f"{agent.__class__.__name__} - Specialized agent",
            "tools": [],
            "tool_details": []
        }

        # Ottieni lista tool se disponibile
        if hasattr(agent, 'get_available_tools'):
            capabilities["tools"] = agent.get_available_tools()

        # Ottieni descrizione dettagliata se l'agent ha il metodo
        if hasattr(agent, 'get_capabilities'):
            detailed_caps = agent.get_capabilities()
            capabilities.update(detailed_caps)

        # Estrai dettagli tool se disponibili
        if hasattr(agent, '_tools'):
            for tool_name, tool in agent._tools.items():
                if hasattr(tool, 'get_schema'):
                    schema = tool.get_schema()
                    capabilities["tool_details"].append({
                        "name": tool_name,
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", [])
                    })

        return capabilities

    def _build_manager_prompt(self) -> str:
        """
        Costruisce system prompt con agent registrati.

        Returns:
            str: System prompt completo per il Manager
        """
        # Costruisci descrizioni agent
        agent_descriptions = []

        for agent_id, capabilities in self._agent_capabilities.items():
            # Descrizione base
            agent_desc = f"\n- {agent_id}: {capabilities['description']}"

            # Aggiungi tool disponibili
            if capabilities['tools']:
                agent_desc += f"\n  Available tools: {', '.join(capabilities['tools'])}"

            # Aggiungi dettagli tool se disponibili
            if capabilities.get('tool_details'):
                agent_desc += "\n  Tool capabilities:"
                for tool in capabilities['tool_details']:
                    agent_desc += f"\n    • {tool['name']}: {tool['description']}"

            agent_descriptions.append(agent_desc)

        # Se nessun agent registrato
        if not agent_descriptions:
            agents_section = "\nNo agents currently registered."
        else:
            agents_section = "\n".join(agent_descriptions)

        # Costruisci prompt finale
        return self.MANAGER_REACT_PROMPT.format(
            agents_description=agents_section
        )

    async def handle_user_request(self, request: HumanMessage) -> Dict[str, Any]:
        """
        Entry point per gestire richieste utente.

        Args:
            request: Messaggio dall'utente

        Returns:
            Dict: Risultato dell'orchestrazione
        """
        start_time = time.time()
        self._status = ManagerStatus.ORCHESTRATING
        self._stats["requests_handled"] += 1

        try:
            # Esegui orchestrazione con ReAct
            result = await self._orchestrate_with_react(request)

            # Aggiorna statistiche
            orchestration_time = time.time() - start_time
            self._stats["total_orchestration_time"] += orchestration_time
            self._stats["last_activity"] = datetime.now(timezone.utc)

            return {
                "success": result.get("success", False),
                "response": result.get("answer", ""),
                "tasks_executed": result.get("tasks_executed", []),
                "execution_time": orchestration_time,
                "agents_used": result.get("agents_used", [])
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
        finally:
            self._status = ManagerStatus.IDLE

    async def _orchestrate_with_react(self, request: HumanMessage) -> Dict[str, Any]:
        """
        Esegue orchestrazione usando ReAct pattern.

        Args:
            request: Richiesta utente

        Returns:
            Dict: Risultato orchestrazione
        """
        react_steps = 0
        max_steps = 3  # Limite per evitare loop infiniti
        tasks_executed = []
        agents_used = set()

        # Costruisci system prompt con agent disponibili
        system_prompt = self._build_manager_prompt()

        # Conversazione iniziale
        conversation = f"User request: {request.content}\n\nPlease analyze this request and orchestrate the appropriate agents."

        while react_steps < max_steps:
            react_steps += 1

            # Chiama LLM per reasoning
            llm_response = self.llm_client.invoke(
                system=system_prompt,
                user=conversation
            )
            print(f"[Manager] Step {react_steps} LLM response:\n{llm_response}\n")

            # Check se ha risposta finale
            if "Answer:" in llm_response:
                answer = self._extract_answer(llm_response)
                return {
                    "success": True,
                    "answer": answer,
                    "tasks_executed": tasks_executed,
                    "agents_used": list(agents_used),
                    "react_steps": react_steps
                }

            # Check se vuole creare task
            elif "Action: create_task:" in llm_response and "PAUSE" in llm_response:
                result = await self._process_create_task_action(llm_response)

                if result["success"]:
                    task_id = result["task_id"]
                    agent_id = result["agent_id"]

                    tasks_executed.append(task_id)
                    agents_used.add(agent_id)

                    # Attendi completamento task
                    task_result = await self._wait_for_task_completion(task_id, agent_id)

                    if task_result:
                        result_data = task_result.get('result', {})
                        if result_data:
                            observation = f"Task {task_id[:8]} completed by {agent_id}. Result: {json.dumps(result_data, indent=2)}"
                        else:
                            observation = f"Task {task_id[:8]} completed by {agent_id}. No result data available."
                        self._stats["tasks_completed"] += 1
                    else:
                        observation = f"Task {task_id[:8]} failed or timed out"
                        self._stats["tasks_failed"] += 1

                    conversation += f"\n\n{llm_response}\n\nObservation: {observation}"

                else:
                    error_obs = f"Error creating task: {result['error']}"
                    conversation += f"\n\n{llm_response}\n\nObservation: {error_obs}"

            # Check se vuole controllare status task
            elif "Action: check_task:" in llm_response and "PAUSE" in llm_response:
                result = self._process_check_task_action(llm_response)

                if result["success"]:
                    task_status = result["task_status"]
                    observation = f"Task status: {json.dumps(task_status, indent=2)}"
                else:
                    observation = f"Error checking task: {result['error']}"

                conversation += f"\n\n{llm_response}\n\nObservation: {observation}"

            else:
                # Risposta non nel formato atteso
                conversation += f"\n\n{llm_response}\n\nPlease follow the ReAct format: Thought, Action, PAUSE or provide final Answer: if you have sufficient information."

        # Raggiunto limite step
        return {
            "success": False,
            "answer": "Orchestration reached maximum steps without completing",
            "tasks_executed": tasks_executed,
            "agents_used": list(agents_used),
            "react_steps": react_steps
        }

    async def _process_create_task_action(self, llm_response: str) -> Dict[str, Any]:
        """
        Processa azione di creazione task.

        Args:
            llm_response: Risposta LLM con action

        Returns:
            Dict: Risultato creazione task
        """
        try:
            # Estrai parametri JSON
            pattern = r"Action:\s*create_task:\s*(\{(?:[^{}]|{[^}]*})*\})"
            match = re.search(pattern, llm_response, re.DOTALL)

            if not match:
                return {"success": False, "error": "Could not parse create_task action"}

            params_json = match.group(1)
            params = json.loads(params_json)

            # Validazione parametri
            agent_id = params.get("agent_id")
            task_type = params.get("task_type", "generic")
            task_data = params.get("task_data", {})

            if not agent_id:
                return {"success": False, "error": "Missing agent_id in create_task"}

            # Verifica agent registrato
            if agent_id not in self._registered_agents:
                available = list(self._registered_agents.keys())
                return {
                    "success": False,
                    "error": f"Agent '{agent_id}' not registered. Available: {available}"
                }

            # Crea task sulla blackboard
            task_id = self.blackboard.create_task(
                assigned_to=agent_id,
                task_type=task_type,
                task_data=task_data,
                created_by=self.manager_id
            )

            # Traccia task attivo
            self._active_tasks[task_id] = {
                "agent_id": agent_id,
                "created_at": datetime.now(timezone.utc),
                "task_type": task_type
            }

            self._stats["tasks_created"] += 1

            # Update stats per agent
            if agent_id not in self._stats["agents_used"]:
                self._stats["agents_used"][agent_id] = 0
            self._stats["agents_used"][agent_id] += 1

            return {
                "success": True,
                "task_id": task_id,
                "agent_id": agent_id
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _process_check_task_action(self, llm_response: str) -> Dict[str, Any]:
        """
        Processa azione di check status task.

        Args:
            llm_response: Risposta LLM con action

        Returns:
            Dict: Status del task
        """
        try:
            # Estrai parametri
            pattern = r"Action:\s*check_task:\s*({.*?})"
            match = re.search(pattern, llm_response, re.DOTALL)

            if not match:
                return {"success": False, "error": "Could not parse check_task action"}

            params = json.loads(match.group(1))
            task_id = params.get("task_id")
            agent_id = params.get("agent_id")

            if not task_id or not agent_id:
                return {"success": False, "error": "Missing task_id or agent_id"}

            # Recupera status dalla blackboard
            task_status = self.blackboard.get_task_status(task_id, agent_id)

            if task_status:
                return {
                    "success": True,
                    "task_status": task_status
                }
            else:
                return {"success": False, "error": "Task not found"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _wait_for_task_completion(self, task_id: str, agent_id: str,
                                        timeout: int = 60) -> Optional[Dict]:
        """
        Attende completamento di un task.

        Args:
            task_id: ID del task
            agent_id: ID dell'agent
            timeout: Timeout in secondi

        Returns:
            Optional[Dict]: Status finale del task o None se timeout
        """
        start_time = time.time()
        check_interval = 0.5  # Check ogni 500ms

        while (time.time() - start_time) < timeout:
            # Check status
            task_status = self.blackboard.get_task_status(task_id, agent_id)

            if task_status:
                status = task_status.get("status")

                if status == "completed":
                    # Task completato con successo
                    if task_id in self._active_tasks:
                        del self._active_tasks[task_id]
                    self._completed_tasks.append(task_id)
                    return task_status

                elif status == "failed":
                    # Task fallito
                    if task_id in self._active_tasks:
                        del self._active_tasks[task_id]
                    return task_status

            # Attendi prima del prossimo check
            await asyncio.sleep(check_interval)

        # Timeout raggiunto
        return None

    def _extract_answer(self, llm_response: str) -> str:
        """
        Estrae risposta finale dal response LLM.

        Args:
            llm_response: Risposta LLM

        Returns:
            str: Risposta estratta
        """
        # Metodo 1: Cerca "Answer:" all'inizio di linea
        lines = llm_response.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("Answer:"):
                # Prendi tutto dopo "Answer:"
                answer_parts = [line.replace("Answer:", "").strip()]
                # Aggiungi linee successive se fanno parte della risposta
                for j in range(i + 1, len(lines)):
                    answer_parts.append(lines[j])
                return '\n'.join(answer_parts).strip()

        # Metodo 2: Cerca "Answer:" ovunque nel testo
        if "Answer:" in llm_response:
            answer_index = llm_response.find("Answer:")
            if answer_index != -1:
                answer_text = llm_response[answer_index + len("Answer:"):].strip()
                return answer_text

        # Fallback: restituisce tutto
        return llm_response.strip()

    def get_registered_agents(self) -> List[str]:
        """Lista degli agent registrati."""
        return list(self._registered_agents.keys())

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Recupera agent per ID."""
        return self._registered_agents.get(agent_id)

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Tutti gli agent registrati."""
        return self._registered_agents.copy()

    def health_check(self) -> Dict[str, Any]:
        """
        Verifica salute del sistema.

        Returns:
            Dict: Status di salute
        """
        health = {
            "manager_status": self._status.value,
            "registered_agents": len(self._registered_agents),
            "agent_statuses": {},
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "blackboard_entries": self.blackboard.size(),
            "stats": self._stats
        }

        # Check status di ogni agent
        for agent_id, agent in self._registered_agents.items():
            if hasattr(agent, 'get_status'):
                health["agent_statuses"][agent_id] = agent.get_status().value
            else:
                health["agent_statuses"][agent_id] = "unknown"

        return health

    def set_agent_instruction(self, agent_id: str, instruction: str,
                              instruction_type: str = "correction",
                              expires_in_minutes: Optional[int] = 30) -> Optional[str]:
        """
        Imposta system instruction per correggere comportamento agent.

        Args:
            agent_id: ID dell'agent
            instruction: Testo dell'istruzione
            instruction_type: Tipo di istruzione
            expires_in_minutes: Durata in minuti

        Returns:
            Optional[str]: ID dell'istruzione o None
        """
        if agent_id not in self._registered_agents:
            return None

        # Aggiungi system instruction tramite blackboard
        instruction_id = self.blackboard.add_system_instruction(
            agent_id=agent_id,
            instruction_text=instruction,
            instruction_type=instruction_type,
            expires_in_minutes=expires_in_minutes,
            priority_level=1,  # Alta priorità per correzioni
            added_by=self.manager_id
        )

        return instruction_id

    def broadcast_system_instruction(self, instruction: str,
                                     instruction_type: str = "global",
                                     expires_in_minutes: Optional[int] = None) -> List[str]:
        """
        Invia istruzione a tutti gli agent.

        Args:
            instruction: Testo dell'istruzione
            instruction_type: Tipo di istruzione
            expires_in_minutes: Durata in minuti

        Returns:
            List[str]: ID delle istruzioni create
        """
        instruction_ids = []

        for agent_id in self._registered_agents.keys():
            instr_id = self.blackboard.add_system_instruction(
                agent_id=agent_id,
                instruction_text=instruction,
                instruction_type=instruction_type,
                expires_in_minutes=expires_in_minutes,
                priority_level=2,  # Media priorità per broadcast
                added_by=self.manager_id
            )
            instruction_ids.append(instr_id)

        return instruction_ids

    def shutdown_all_agents(self):
        """Shutdown ordinato di tutti gli agent."""
        for agent_id, agent in self._registered_agents.items():
            try:
                # Se l'agent ha metodo di shutdown
                if hasattr(agent, 'shutdown'):
                    agent.shutdown()
            except Exception as e:
                print(f"Error shutting down {agent_id}: {str(e)}")

        # Clear registri
        self._registered_agents.clear()
        self._agent_capabilities.clear()
        self._active_tasks.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Statistiche del Manager."""
        return {
            "status": self._status.value,
            "registered_agents": len(self._registered_agents),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            **self._stats
        }

    def __str__(self) -> str:
        return f"AgentManager(agents={len(self._registered_agents)}, status={self._status.value})"

    def __repr__(self) -> str:
        agents = list(self._registered_agents.keys())
        return f"AgentManager(id='{self.manager_id}', agents={agents}, active_tasks={len(self._active_tasks)})"