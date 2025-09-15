from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import threading
import re
import json
from enum import Enum

# Import dai moduli esistenti
from multi_agent_system.core.messages import (
    BaseMessage, create_agent_response)
from multi_agent_system.core.tool_base import ToolBase, ToolResult
from multi_agent_system.core.black_board import BlackBoard, BlackboardChange
from multi_agent_system.core.utility.agent_strategies import (_ReactStrategy, _ActingStrategy)


class AgentStatus(Enum):
    """Stati essenziali di un agent"""
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"


class BaseAgent(ABC):
    """
    Classe base ABC standardizzata con ReAct/Acting pattern configurabile.

    CARATTERISTICHE AUTOMATICHE:
    - System prompt ReAct O Acting configurabile via parametro
    - ReAct mode: Tool usage + loop PAUSE/Observation con schema JSON pulito
    - Acting mode: No tool + single-shot diretto
    - LLM client auto-configurato per mode scelto
    - Integration con blackboard per observations
    - Completamente sincrono

    GLI AGENT DERIVATI DEVONO SOLO:
    - Implementare setup_tools() per registrare i loro tool (se react=True)
    - Passare LLM client nel constructor
    - Il resto è automatico!
    """

    def __init__(self, agent_id: str, blackboard: BlackBoard, llm_client, react: bool = True):
        """
        Inizializza BaseAgent con ReAct/Acting configurabile.

        Args:
            agent_id: ID univoco dell'agent
            blackboard: Blackboard condivisa del sistema
            llm_client: Client LLM (deve avere metodo invoke(system, user))
            react: True per ReAct mode (tool+loop), False per Acting mode (direct)
        """
        self.agent_id = agent_id
        self.blackboard = blackboard
        self.llm_client = llm_client
        self._react_mode = react

        # AUTO-configura LLM client per il mode scelto
        self.llm_client.set_react_mode(react)

        # Status e controllo thread-safe
        self._status = AgentStatus.IDLE
        self._lock = threading.RLock()

        # Tool management (solo per ReAct mode)
        self._tools: Dict[str, ToolBase] = {}

        # Strategy interna per gestire prompt/loop
        self._prompt_strategy = _ReactStrategy() if react else _ActingStrategy()

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
            """
            rappresenta il callable che viene chiamato quanto si triggera l'observare, fa l'update della BlackBoard
            e chiama il metodo che la esegue
            :param change:
            :return:
            """
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
        """Gestisce task assegnato dalla blackboard con strategy pattern"""
        with self._lock:
            if self._status == AgentStatus.WORKING:
                return

            self._status = AgentStatus.WORKING
            self._current_task = task

        try:
            # Esegue task con strategy appropriata (ReAct o Acting)
            result = self._prompt_strategy.execute_task_loop(
                self, task["task_data"], task["task_id"]
            )

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

    def _extract_action_json_robust(self, text: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Estrae JSON da Action: {tool_name}: {json} gestendo qualsiasi livello di annidamento.

        Args:
            text: Testo da cui estrarre il JSON
            tool_name: Nome del tool da cercare nel pattern

        Returns:
            Optional[Dict]: JSON estratto o None se non trovato/invalido
        """
        try:
            # Trova l'inizio del pattern
            pattern = f"Action:\\s*{re.escape(tool_name)}:\\s*"
            match = re.search(pattern, text, re.DOTALL)

            if not match:
                return None

            start_pos = match.end()

            # Trova la prima graffa
            while start_pos < len(text) and text[start_pos] != '{':
                start_pos += 1

            if start_pos >= len(text):
                return None

            # Usa JSONDecoder per parsing automatico e robusto
            decoder = json.JSONDecoder()
            try:
                obj, end_idx = decoder.raw_decode(text, start_pos)
                return obj
            except json.JSONDecodeError:
                return None

        except Exception:
            return None

    def _process_action(self, llm_response: str) -> Dict[str, Any]:
        """Processa azione richiesta dall'LLM ed esegue tool (solo ReAct mode) con estrazione JSON robusta"""
        if not self._react_mode:
            return {
                "success": False,
                "error": "Tool usage not available in Acting mode"
            }

        try:
            # Estrae nome del tool prima usando regex semplice
            tool_name_pattern = r"Action:\s*(\w+):"
            tool_match = re.search(tool_name_pattern, llm_response, re.DOTALL)

            if not tool_match:
                return {
                    "success": False,
                    "error": "Could not parse action format. Use: Action: tool_name: {params}"
                }

            tool_name = tool_match.group(1).strip()

            # Verifica tool esistente
            if tool_name not in self._tools:
                available_tools = list(self._tools.keys())
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not available. Available: {available_tools}"
                }

            # Estrae parametri usando il metodo robusto
            params = self._extract_action_json_robust(llm_response, tool_name)

            if params is None:
                return {
                    "success": False,
                    "error": f"Could not parse JSON parameters for tool '{tool_name}'"
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
                "task_id": task_id,
                "mode": "react" if self._react_mode else "acting"
            },
            updated_by=self.agent_id,
            tags=["observation", "react_step" if self._react_mode else "acting_step", self.agent_id]
        )

    # ===== METODI ABSTRACT - DA IMPLEMENTARE NEGLI AGENT DERIVATI =====

    @abstractmethod
    def setup_tools(self):
        """
        Registra i tool specifici dell'agent.
        UNICO METODO DA IMPLEMENTARE negli agent derivati.

        NOTE: Se react=False, questo metodo può essere vuoto o non registrare tool

        Esempio per ReAct mode:
        def setup_tools(self):
            if self._react_mode:  # Solo se in ReAct mode
                gmail_tool = GmailReaderTool(self.credentials)
                self.register_tool(gmail_tool)
        """
        pass

    # ===== METODI CONCRETI EREDITATI =====

    def initialize(self) -> bool:
        """Inizializza l'agent (chiamato automaticamente)"""
        try:
            # Setup tool specifici (implementato da classe derivata)
            # Gli agent Acting possono non registrare tool
            self.setup_tools()

            # Verifica dipendenze
            if not self.blackboard or not self.llm_client:
                return False

            self._status = AgentStatus.IDLE
            print(
                f"[{self.agent_id}] Initialized in {'ReAct' if self._react_mode else 'Acting'} mode with {len(self._tools)} tools")
            return True

        except Exception as e:
            print(f"Failed to initialize agent {self.agent_id}: {str(e)}")
            self._status = AgentStatus.ERROR
            return False

    def register_tool(self, tool: ToolBase):
        """Registra un tool nell'agent (solo per ReAct mode)"""
        if not self._react_mode:
            print(f"[{self.agent_id}] Warning: Tool registration ignored in Acting mode")
            return

        tool_name = tool.get_name()
        self._tools[tool_name] = tool

    def use_tool(self, tool_name: str, **params) -> ToolResult:
        """Usa un tool registrato (solo per ReAct mode)"""
        if not self._react_mode:
            return ToolResult(
                success=False,
                error="Tool usage not available in Acting mode",
                execution_time=0.0
            )

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
                    "mode": "react" if self._react_mode else "acting",
                    "tools": self.get_available_tools() if self._react_mode else [],
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
        """Lista tool disponibili (solo per ReAct mode)"""
        return list(self._tools.keys()) if self._react_mode else []

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
                "mode": "react" if self._react_mode else "acting",
                "tools_count": len(self._tools),
                "current_task_id": (
                    self._current_task["task_id"] if self._current_task else None
                )
            }
    #todo non usato, decidere se tenere o meno
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
            "mode": "react" if self._react_mode else "acting",
            "tools": self.get_available_tools(),
            "tool_schemas": [],
            "status": self._status.value,
            "stats": {
                "tasks_completed": self._stats["tasks_completed"],
                "tasks_failed": self._stats["tasks_failed"],
                "total_steps": self._stats["total_react_steps"]
            }
        }

        # Aggiungi schema dettagliato di ogni tool (solo ReAct mode)
        if self._react_mode:
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

    #todo, poco standardizzata, deve essere più generica e applicabile
    def _generate_agent_description(self) -> str:
        """
        Genera descrizione automatica basata su mode, tool e classe.
        Override in agent specifici per descrizioni custom.
        Returns:
            str: Descrizione dell'agent
        """
        # Descrizione base dal nome classe
        class_name = self.__class__.__name__

        #todo decidere se usare o eliminare
        descriptions = {
            "RAGAgent": "Retrieval-Augmented Generation agent for document search and contextual responses",
            "AnalysisAgent": "Analysis agent for data processing, sentiment analysis, and insights extraction",
            "SyntheticAgent": "Synthesis agent for combining information and generating comprehensive outputs, always require all the query",
            "ValidationAgent": "Validation agent for quality checks and data verification",
            "EmailAgent": "Email processing agent for reading and managing email communications",
            "NotionWriterAgent": "Notion writing agent for creating and updating pages with markdown support",
            "WeatherAgent": "Weather agent for meteorological information and forecasts",
            "MathAgent": "Mathematical agent for calculations and numerical processing"
        }

        # Usa descrizione specifica o genera generica
        if class_name in descriptions:
            base_desc = descriptions[class_name]
        else:
            base_desc = f"{class_name} - Specialized agent"

        # Aggiungi mode info
        mode_desc = "with ReAct pattern and tool usage" if self._react_mode else "with direct Acting pattern for reasoning tasks"
        base_desc += f" operating {mode_desc}"

        # Aggiungi info sui tool principali (solo ReAct)
        if self._react_mode and self._tools:
            tool_names = list(self._tools.keys())[:3]  # Prime 3 tool
            if tool_names:
                base_desc += f". Main tools: {', '.join(tool_names)}"
                if len(self._tools) > 3:
                    base_desc += f" and {len(self._tools) - 3} more"

        return base_desc

    def __str__(self) -> str:
        mode = "ReAct" if self._react_mode else "Acting"
        return f"{mode}Agent(id={self.agent_id}, status={self._status.value})"

    def __repr__(self) -> str:
        return (f"BaseAgent(id='{self.agent_id}', "
                f"status='{self._status.value}', "
                f"mode={'react' if self._react_mode else 'acting'}, "
                f"tools={len(self._tools)})")

