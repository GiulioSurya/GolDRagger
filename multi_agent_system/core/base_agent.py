from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import threading
import time
import re
from enum import Enum

# Import dai moduli esistenti
from multi_agent_system.core.messages import (
    BaseMessage, create_agent_response, create_agent_result, AgentResult
)
from multi_agent_system.core.tool_base import ToolBase, ToolResult
from multi_agent_system.core.black_board import BlackBoard, BlackboardChange


class AgentStatus(Enum):
    """Stati essenziali di un agent"""
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"


class BaseAgent(ABC):
    """
    Classe base ABC standardizzata con ReAct loop integrato.

    CARATTERISTICHE AUTOMATICHE:
    - System prompt ReAct generico con tool auto-discovery
    - Loop ReAct standardizzato: Thought → Action → PAUSE → Observation
    - Integration con blackboard per observations
    - LLM client configurabile per ogni agent derivato
    - Completamente sincrono

    GLI AGENT DERIVATI DEVONO SOLO:
    - Implementare setup_tools() per registrare i loro tool specifici
    - Passare LLM client nel constructor
    - Il resto è automatico!
    """

    # ReAct System Prompt STANDARDIZZATO con auto-inject tool info
    BASE_REACT_PROMPT = """You are an AI agent that operates using the ReAct (Reasoning and Acting) pattern.

You run in a loop of Thought, Action, PAUSE, Observation until you can provide a final Answer.

WORKFLOW:
1. Thought: Describe your reasoning about the current task
2. Action: Execute one of your available tools using the exact format shown below  
3. PAUSE: Always return PAUSE after an Action and wait for Observation
4. Observation: Will contain the result of your Action (provided automatically)
5. Repeat steps 1-4 until you have enough information
6. Answer: Provide your final response when ready

ACTION FORMAT:
Action: tool_name: parameters_as_json

EXAMPLE:
Thought: I need to read emails to help the user
Action: gmail_reader: {{"max_results": 10, "query": "is:unread"}}
PAUSE

You will then receive:
Observation: [tool result will be inserted here]

Continue this pattern until you can provide a final Answer.

AVAILABLE TOOLS:
{tools_description}

IMPORTANT:
- Always use exact tool names from the list above
- Follow the JSON parameter format exactly
- Always return PAUSE after Action
- End with Answer: when you have the final response"""

    def __init__(self, agent_id: str, blackboard: BlackBoard, llm_client):
        """
        Inizializza BaseAgent con ReAct standardizzato.

        Args:
            agent_id: ID univoco dell'agent
            blackboard: Blackboard condivisa del sistema
            llm_client: Client LLM (deve avere metodo invoke(system, user))
        """
        self.agent_id = agent_id
        self.blackboard = blackboard
        self.llm_client = llm_client

        # Status e controllo thread-safe
        self._status = AgentStatus.IDLE
        self._lock = threading.RLock()

        # Tool management
        self._tools: Dict[str, ToolBase] = {}

        # Task corrente
        self._current_task: Optional[Dict] = None

        # Statistiche
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_react_steps": 0,
            "tools_used_count": {},
            "last_activity": None
        }

        # Setup observer per task assegnati automaticamente
        self._setup_task_observer()

    def _setup_task_observer(self):
        """Osserva blackboard per task assegnati a questo agent"""

        def on_blackboard_change(change: BlackboardChange):
            try:
                # Controlla se è un task per questo agent
                if (change.key.startswith(f"task_{self.agent_id}_") and
                        change.new_value and
                        change.new_value.get("status") == "pending"):
                    self._handle_assigned_task(change.new_value)

            except Exception as e:
                print(f"Error in task observer for {self.agent_id}: {str(e)}")

        self.blackboard.subscribe_to_changes(on_blackboard_change)

    def _handle_assigned_task(self, task: Dict):
        """Gestisce task assegnato dalla blackboard con ReAct loop"""
        with self._lock:
            if self._status == AgentStatus.WORKING:
                return

            self._status = AgentStatus.WORKING
            self._current_task = task

        try:
            # Esegue ReAct loop per il task
            result = self._execute_react_loop(task["task_data"], task["task_id"])

            # Aggiorna risultato sulla blackboard
            success = self.blackboard.update_task_result(
                task_id=task["task_id"],
                agent_id=self.agent_id,
                result=result.data if result.success else {"error": result.error},
                status="completed" if result.success else "failed",
                execution_time=result.execution_time
            )

            if success and result.success:
                self._stats["tasks_completed"] += 1
                self._stats["total_react_steps"] += result.react_steps
            else:
                self._stats["tasks_failed"] += 1

        except Exception as e:
            print(f"Task execution failed for {self.agent_id}: {str(e)}")

            # Marca task come fallito
            self.blackboard.update_task_result(
                task_id=task["task_id"],
                agent_id=self.agent_id,
                result={"error": str(e)},
                status="failed"
            )
            self._stats["tasks_failed"] += 1

        finally:
            with self._lock:
                self._status = AgentStatus.IDLE
                self._current_task = None
                self._stats["last_activity"] = datetime.now(timezone.utc)

    def _execute_react_loop(self, task_data: Dict, task_id: str) -> AgentResult:
        """
        Esegue il ReAct loop standardizzato.

        Args:
            task_data: Dati del task
            task_id: ID del task per observation storage

        Returns:
            AgentResult: Risultato finale con step ReAct tracciati
        """
        start_time = time.time()
        react_steps = 0
        tools_used = []
        observations = []
        max_steps = 10  # Limite safety per evitare loop infiniti

        try:
            # Costruisce system prompt con tool info auto-injected
            system_prompt = self._build_react_system_prompt()

            # Prompt iniziale per il task
            conversation = f"Task: {task_data}\n\nPlease start with your first Thought about this task."

            while react_steps < max_steps:
                # Chiama LLM
                llm_response = self.llm_client.invoke(
                    system=system_prompt,
                    user=conversation
                )
                print(f"[{self.agent_id}] LLM Response (step {react_steps}): {llm_response[:200]}...")

                # Parsing della risposta LLM
                if "Answer:" in llm_response:
                    # LLM ha dato risposta finale
                    answer = self._extract_answer(llm_response)
                    return create_agent_result(
                        success=True,
                        agent_id=self.agent_id,
                        data={"answer": answer, "task_data": task_data},
                        execution_time=time.time() - start_time,
                        react_steps=react_steps,
                        tools_used=tools_used,
                        observations=observations
                    )

                elif "Action:" in llm_response and "PAUSE" in llm_response:
                    # LLM vuole eseguire un'azione
                    action_result = self._process_action(llm_response, task_id)

                    if action_result["success"]:
                        # Tool eseguito con successo
                        tool_name = action_result["tool_name"]
                        observation = action_result["observation"]

                        # Tracking
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                        observations.append(observation)

                        # Aggiorna stats
                        self._stats["tools_used_count"][tool_name] = (
                                self._stats["tools_used_count"].get(tool_name, 0) + 1
                        )

                        # Salva observation sulla blackboard
                        self._save_observation_to_blackboard(task_id, react_steps, observation)

                        # Continua conversazione con observation
                        conversation += f"\n\n{llm_response}\n\nObservation: {observation}"

                    else:
                        # Tool fallito
                        error_obs = f"Error: {action_result['error']}"
                        observations.append(error_obs)
                        conversation += f"\n\n{llm_response}\n\nObservation: {error_obs}"

                    react_steps += 1

                else:
                    # Risposta LLM non nel formato atteso
                    conversation += f"\n\n{llm_response}\n\nPlease follow the format: Thought: ... Action: tool_name: params PAUSE"

            # Raggiunto limite step
            return create_agent_result(
                success=False,
                agent_id=self.agent_id,
                error=f"Reached maximum ReAct steps ({max_steps})",
                execution_time=time.time() - start_time,
                react_steps=react_steps,
                tools_used=tools_used,
                observations=observations
            )

        except Exception as e:
            return create_agent_result(
                success=False,
                agent_id=self.agent_id,
                error=f"ReAct loop failed: {str(e)}",
                execution_time=time.time() - start_time,
                react_steps=react_steps,
                tools_used=tools_used,
                observations=observations
            )

    def _build_react_system_prompt(self) -> str:
        """Costruisce system prompt ReAct con tool descriptions auto-injected"""

        # Costruisce descrizioni tool automaticamente
        tools_descriptions = []

        for tool_name, tool in self._tools.items():
            schema = tool.get_schema()

            # Estrae parametri con tipi
            params_info = []
            for param in schema.get('parameters', []):
                param_desc = f"{param['name']} ({param['type']})"
                if not param.get('required', True):
                    param_desc += " [optional]"
                if param.get('description'):
                    param_desc += f": {param['description']}"

                # Include valori permessi se disponibili
                if param.get('allowed_values'):
                    allowed_vals = ', '.join(map(str, param['allowed_values']))
                    param_desc += f" [allowed: {allowed_vals}]"

                # Include range se disponibili
                if param.get('min_value') is not None or param.get('max_value') is not None:
                    min_val = param.get('min_value', '')
                    max_val = param.get('max_value', '')
                    param_desc += f" [range: {min_val}-{max_val}]"

                # Include valore default se disponibile
                if param.get('default') is not None:
                    param_desc += f" [default: {param['default']}]"

                params_info.append(param_desc)

            tool_desc = f"""
    - {tool_name}: {schema['description']}
      Parameters: {', '.join(params_info) if params_info else 'none'}
      Example: Action: {tool_name}: {{"param1": "value1", "param2": "value2"}}"""

            tools_descriptions.append(tool_desc)

        # Aggiunge system instructions dinamiche dalla blackboard
        system_instructions = self.blackboard.get_system_instructions_for_agent(
            self.agent_id, active_only=True
        )

        additional_instructions = ""
        if system_instructions:
            instructions_text = [f"- {instr.instruction_text}" for instr in system_instructions]
            additional_instructions = f"""

    ADDITIONAL DYNAMIC INSTRUCTIONS:
    {chr(10).join(instructions_text)}"""

        # Combina tutto
        tools_section = "\n".join(tools_descriptions) if tools_descriptions else "No tools available"

        final_prompt = self.BASE_REACT_PROMPT.format(
            tools_description=tools_section
        ) + additional_instructions

        return final_prompt

    def _process_action(self, llm_response: str, task_id: str) -> Dict[str, Any]:
        """Processa azione richiesta dall'LLM ed esegue tool"""
        try:
            # Estrae azione usando regex
            action_pattern = r"Action:\s*(\w+):\s*({.*?})"
            match = re.search(action_pattern, llm_response, re.DOTALL)

            if not match:
                return {
                    "success": False,
                    "error": "Could not parse action format. Use: Action: tool_name: {params}"
                }

            tool_name = match.group(1).strip()
            params_json = match.group(2).strip()

            # Verifica tool esistente
            if tool_name not in self._tools:
                available_tools = list(self._tools.keys())
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not available. Available: {available_tools}"
                }

            # Parse parametri JSON
            import json
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Invalid JSON parameters: {str(e)}"
                }

            # Esegue tool
            tool_result = self.use_tool(tool_name, **params)

            if tool_result.success:
                observation = f"Tool '{tool_name}' executed successfully. Result: {tool_result.data}"
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "observation": observation,
                    "tool_result": tool_result.data
                }
            else:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' failed: {tool_result.error}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Action processing failed: {str(e)}"
            }

    def _extract_answer(self, llm_response: str) -> str:
        """Estrae risposta finale dal response LLM"""
        lines = llm_response.split('\n')
        for line in lines:
            if line.startswith("Answer:"):
                return line.replace("Answer:", "").strip()
        return llm_response.strip()

    def _save_observation_to_blackboard(self, task_id: str, step: int, observation: str):
        """Salva observation sulla blackboard per tracking"""
        key = f"observation_{self.agent_id}_{task_id}_step_{step}"
        self.blackboard.update(
            key=key,
            value={
                "observation": observation,
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": self.agent_id,
                "task_id": task_id
            },
            updated_by=self.agent_id,
            tags=["observation", "react_step", self.agent_id]
        )

    # ===== METODI ABSTRACT - DA IMPLEMENTARE NEGLI AGENT DERIVATI =====

    @abstractmethod
    def setup_tools(self):
        """
        Registra i tool specifici dell'agent.
        UNICO METODO DA IMPLEMENTARE negli agent derivati.

        Esempio:
        def setup_tools(self):
            gmail_tool = GmailReaderTool(self.credentials)
            self.register_tool(gmail_tool)
        """
        pass

    # ===== METODI CONCRETI EREDITATI =====

    def initialize(self) -> bool:
        """Inizializza l'agent (chiamato automaticamente)"""
        try:
            # Setup tool specifici (implementato da classe derivata)
            self.setup_tools()

            # Verifica dipendenze
            if not self.blackboard or not self.llm_client:
                return False

            self._status = AgentStatus.IDLE
            return True

        except Exception as e:
            print(f"Failed to initialize agent {self.agent_id}: {str(e)}")
            self._status = AgentStatus.ERROR
            return False

    def register_tool(self, tool: ToolBase):
        """Registra un tool nell'agent"""
        tool_name = tool.get_name()
        self._tools[tool_name] = tool

    def use_tool(self, tool_name: str, **params) -> ToolResult:
        """Usa un tool registrato"""
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not available",
                execution_time=0.0
            )

        tool = self._tools[tool_name]
        return tool.safe_execute(**params)

    def handle_message(self, message: BaseMessage) -> BaseMessage:
        """Gestisce messaggi diretti"""
        try:
            return create_agent_response(
                from_agent=self.agent_id,
                to_agent=getattr(message, 'from_agent', 'unknown'),
                response_to_message_id=message.message_id,
                success=True,
                result={
                    "status": f"{self.__class__.__name__} ready",
                    "tools": self.get_available_tools(),
                    "react_enabled": True
                }
            )
        except Exception as e:
            return create_agent_response(
                from_agent=self.agent_id,
                to_agent=getattr(message, 'from_agent', 'unknown'),
                response_to_message_id=message.message_id,
                success=False,
                error=str(e)
            )

    def get_available_tools(self) -> List[str]:
        """Lista tool disponibili"""
        return list(self._tools.keys())

    def get_status(self) -> AgentStatus:
        """Status corrente"""
        with self._lock:
            return self._status

    def get_stats(self) -> Dict[str, Any]:
        """Statistiche agent"""
        with self._lock:
            return {
                **self._stats.copy(),
                "agent_id": self.agent_id,
                "status": self._status.value,
                "tools_count": len(self._tools),
                "current_task_id": (
                    self._current_task["task_id"] if self._current_task else None
                )
            }

    def call_other_agent(self, target_agent_id: str, task_type: str, task_data: Dict) -> str:
        """Chiama altro agent via blackboard"""
        task_id = self.blackboard.create_task(
            assigned_to=target_agent_id,
            task_type=task_type,
            task_data=task_data,
            created_by=self.agent_id
        )

        return task_id

    def add_system_instruction(self, instruction: str,
                               instruction_type: str = "instruction",
                               expires_in_minutes: Optional[int] = None,
                               priority_level: int = 5) -> str:
        """Aggiunge system instruction"""
        return self.blackboard.add_system_instruction(
            agent_id=self.agent_id,
            instruction_text=instruction,
            instruction_type=instruction_type,
            expires_in_minutes=expires_in_minutes,
            priority_level=priority_level,
            added_by=self.agent_id
        )

    # ===== METODI PER CAPABILITIES DISCOVERY =====

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Restituisce capabilities dettagliate dell'agent per il Manager.
        Returns:
            Dict: Descrizione strutturata delle capabilities
        """
        capabilities = {
            "agent_id": self.agent_id,
            "class_name": self.__class__.__name__,
            "description": self._generate_agent_description(),
            "tools": self.get_available_tools(),
            "tool_schemas": [],
            "status": self._status.value,
            "stats": {
                "tasks_completed": self._stats["tasks_completed"],
                "tasks_failed": self._stats["tasks_failed"],
                "total_react_steps": self._stats["total_react_steps"]
            }
        }

        # Aggiungi schema dettagliato di ogni tool
        for tool_name, tool in self._tools.items():
            if hasattr(tool, 'get_schema'):
                schema = tool.get_schema()
                capabilities["tool_schemas"].append({
                    "name": tool_name,
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", []),
                    "tags": schema.get("tags", [])
                })

        return capabilities

    def _generate_agent_description(self) -> str:
        """
        Genera descrizione automatica basata su tool e classe.
        Override in agent specifici per descrizioni custom.
        Returns:
            str: Descrizione dell'agent
        """
        # Descrizione base dal nome classe
        class_name = self.__class__.__name__

        # Mapping nomi comuni
        descriptions = {
            "RAGAgent": "Retrieval-Augmented Generation agent for document search and contextual responses",
            "AnalysisAgent": "Analysis agent for data processing, sentiment analysis, and insights extraction",
            "SynthesisAgent": "Synthesis agent for combining information and generating comprehensive outputs",
            "ValidationAgent": "Validation agent for quality checks and data verification",
            "EmailAgent": "Email processing agent for reading and managing email communications",
            "EmailAgentReAct": "Email processing agent with ReAct reasoning for intelligent email handling",
            "WeatherAgent": "Weather agent for meteorological information and forecasts",
            "MathAgent": "Mathematical agent for calculations and numerical processing"
        }

        # Usa descrizione specifica o genera generica
        if class_name in descriptions:
            base_desc = descriptions[class_name]
        else:
            base_desc = f"{class_name} - Specialized agent with {len(self._tools)} tools"

        # Aggiungi info sui tool principali
        if self._tools:
            tool_names = list(self._tools.keys())[:3]  # Prime 3 tool
            if tool_names:
                base_desc += f". Main capabilities: {', '.join(tool_names)}"
                if len(self._tools) > 3:
                    base_desc += f" and {len(self._tools) - 3} more"

        return base_desc

    def __str__(self) -> str:
        return f"ReactAgent(id={self.agent_id}, status={self._status.value})"

    def __repr__(self) -> str:
        return (f"BaseAgent(id='{self.agent_id}', "
                f"status='{self._status.value}', "
                f"tools={len(self._tools)}, react_enabled=True)")


# ===== ESEMPIO AGENT DERIVATO =====

class EmailAgentReAct(BaseAgent):
    """Esempio di agent che eredita ReAct automaticamente"""

    def __init__(self, agent_id: str, blackboard: BlackBoard,
                 llm_client, gmail_credentials: Dict):
        super().__init__(agent_id, blackboard, llm_client)
        self.gmail_credentials = gmail_credentials
        self.initialize()  # Auto-inizializza con ReAct

    def setup_tools(self):
        """UNICO metodo da implementare - registra tool specifici"""
        # Esempio: registrare GmailReaderTool
        # gmail_tool = GmailReaderTool(self.gmail_credentials)
        # self.register_tool(gmail_tool)
        pass


if __name__ == "__main__":
    print("=== BASE AGENT REACT STANDARDIZZATO ===")
    print("Ogni agent derivato eredita automaticamente:")
    print("- ReAct loop completo")
    print("- System prompt con tool auto-discovery")
    print("- Integration blackboard per observations")
    print("- LLM client configurabile")
    print("\nGli agent derivati devono solo implementare setup_tools()!")
    print("Tutto è completamente sincrono e thread-safe!")