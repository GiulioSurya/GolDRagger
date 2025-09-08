from datetime import timedelta
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator, ValidationInfo


class MessageType(Enum):
    """Tipi di messaggi nel sistema multi-agent"""
    AGENT_CALL = "agent_call"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    AGENT_RESPONSE = "agent_response"
    HUMAN_INPUT = "human_input"
    SYSTEM_INSTRUCTION = "system_instruction"


class MessagePriority(Enum):
    """Priorità dei messaggi per gestire code e processing"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class BaseMessage(BaseModel):
    """Classe base per tutti i messaggi nel sistema"""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    message_type: MessageType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: MessagePriority = MessagePriority.NORMAL
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentMessage(BaseMessage):
    """Messaggi di chiamata tra agent"""
    message_type: MessageType = MessageType.AGENT_CALL
    from_agent: str = Field(..., description="ID dell'agent mittente")
    to_agent: str = Field(..., description="ID dell'agent destinatario")
    method: str = Field(..., description="Metodo da chiamare sull'agent target")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Argomenti per il metodo")
    requires_response: bool = Field(default=True, description="Se richiede una risposta")
    timeout: Optional[int] = Field(default=30, description="Timeout in secondi")

    @field_validator('from_agent', 'to_agent')
    def validate_agent_ids(cls, v: str) -> str:
        """Valida che gli ID agent non siano vuoti"""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v.strip()

    @field_validator('method')
    def validate_method(cls, v: str) -> str:
        """Valida che il metodo non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("Method cannot be empty")
        return v.strip()


class AgentResponseMessage(BaseMessage):
    """Risposte degli agent a chiamate/richieste"""
    message_type: MessageType = MessageType.AGENT_RESPONSE
    from_agent: str = Field(..., description="ID dell'agent che risponde")
    to_agent: str = Field(..., description="ID dell'agent destinatario della risposta")
    response_to_message_id: str = Field(..., description="ID del messaggio a cui si risponde")
    success: bool = Field(..., description="Se l'operazione è riuscita")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Risultato dell'operazione")
    error: Optional[str] = Field(default=None, description="Messaggio di errore se fallito")
    execution_time: Optional[float] = Field(default=None, description="Tempo di esecuzione in secondi")

    @field_validator('response_to_message_id')
    def validate_response_id(cls, v: str) -> str:
        """Valida che l'ID del messaggio di riferimento non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("Response to message ID cannot be empty")
        return v.strip()

    @field_validator('error')
    def validate_error_with_success(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Valida coerenza tra success e error"""
        if 'success' in info.data:
            success_value = info.data['success']
            if not success_value and not v:
                raise ValueError("Error message required when success is False")
        return v


class HumanMessage(BaseMessage):
    """Messaggi dall'utente umano"""
    message_type: MessageType = MessageType.HUMAN_INPUT
    user_id: str = Field(..., description="ID dell'utente")
    content: str = Field(..., description="Contenuto del messaggio")
    session_id: Optional[str] = Field(default=None, description="ID della sessione")
    intent: Optional[str] = Field(default=None, description="Intento rilevato/specificato")
    response_to_message_id: Optional[str] = Field(default=None, description="ID messaggio a cui risponde")

    @field_validator('content')
    def validate_content(cls, v: str) -> str:
        """Valida che il contenuto non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

    @field_validator('user_id')
    def validate_user_id(cls, v: str) -> str:
        """Valida che l'user ID non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty")
        return v.strip()


class SystemMessage(BaseMessage):
    """Istruzioni di sistema per estendere dinamicamente i system prompt degli agent"""
    message_type: MessageType = MessageType.SYSTEM_INSTRUCTION
    target_agent: str = Field(..., description="ID dell'agent destinatario dell'istruzione")
    instruction_type: str = Field(default="instruction", description="Tipo di istruzione")
    instruction_text: str = Field(..., description="Testo dell'istruzione da aggiungere al system prompt")
    expires_at: Optional[datetime] = Field(default=None, description="Quando l'istruzione scade")
    priority_level: int = Field(default=1, description="Priorità dell'istruzione (1=alta, 10=bassa)")

    @field_validator('target_agent')
    def validate_target_agent(cls, v: str) -> str:
        """Valida che l'ID dell'agent target non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("Target agent ID cannot be empty")
        return v.strip()

    @field_validator('instruction_type')
    def validate_instruction_type(cls, v: str) -> str:
        """Valida il tipo di istruzione"""
        valid_types = ['instruction', 'constraint', 'behavior', 'context', 'temporary']
        if v.lower() not in valid_types:
            raise ValueError(f"Instruction type must be one of: {valid_types}")
        return v.lower()

    @field_validator('instruction_text')
    def validate_instruction_text(cls, v: str) -> str:
        """Valida che il testo dell'istruzione non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("Instruction text cannot be empty")
        return v.strip()

    @field_validator('priority_level')
    def validate_priority_level(cls, v: int) -> int:
        """Valida il livello di priorità"""
        if not 1 <= v <= 10:
            raise ValueError("Priority level must be between 1 and 10")
        return v


# ===== NUOVO: AgentResult per standardizzare risultati agent =====

class AgentResult(BaseModel):
    """Risultato standardizzato dell'esecuzione di un agent"""
    success: bool = Field(..., description="Se l'operazione è riuscita")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Dati risultanti")
    error: Optional[str] = Field(default=None, description="Messaggio di errore")
    execution_time: float = Field(..., description="Tempo di esecuzione in secondi")
    agent_id: str = Field(..., description="ID dell'agent che ha generato il risultato")
    react_steps: int = Field(default=0, description="Numero di step ReAct eseguiti")
    tools_used: List[str] = Field(default_factory=list, description="Tool utilizzati")
    observations: List[str] = Field(default_factory=list, description="Observation del ReAct loop")

    @field_validator('agent_id')
    def validate_agent_id(cls, v: str) -> str:
        """Valida che l'agent ID non sia vuoto"""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v.strip()

    @field_validator('error')
    def validate_error_with_success(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Valida coerenza tra success e error"""
        if 'success' in info.data:
            success_value = info.data['success']
            if not success_value and not v:
                raise ValueError("Error message required when success is False")
        return v


def create_message_factory(message_data: Dict[str, Any]) -> BaseMessage:
    """Factory per creare il messaggio del tipo corretto basandosi sui dati"""
    message_type = message_data.get('message_type')

    if not message_type:
        raise ValueError("message_type is required")

    if isinstance(message_type, str):
        try:
            message_type = MessageType(message_type)
        except ValueError:
            raise ValueError(f"Unknown message type: {message_type}")

    message_classes = {
        MessageType.AGENT_CALL: AgentMessage,
        MessageType.AGENT_RESPONSE: AgentResponseMessage,
        MessageType.HUMAN_INPUT: HumanMessage,
        MessageType.SYSTEM_INSTRUCTION: SystemMessage,
        MessageType.TASK_REQUEST: AgentMessage,
        MessageType.TASK_RESPONSE: AgentResponseMessage,
    }

    message_class = message_classes.get(message_type)
    if not message_class:
        raise ValueError(f"No class defined for message type: {message_type}")

    try:
        return message_class(**message_data)
    except Exception as e:
        raise ValueError(f"Failed to create {message_class.__name__}: {str(e)}")


def create_agent_call(from_agent: str, to_agent: str, method: str,
                      arguments: Dict[str, Any] = None,
                      priority: MessagePriority = MessagePriority.NORMAL) -> AgentMessage:
    """Helper per creare una chiamata tra agent"""
    return AgentMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        method=method,
        arguments=arguments or {},
        priority=priority
    )


def create_agent_response(from_agent: str, to_agent: str, response_to_message_id: str,
                          success: bool, result: Dict[str, Any] = None,
                          error: str = None, execution_time: float = None) -> AgentResponseMessage:
    """Helper per creare una risposta da agent"""
    return AgentResponseMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        response_to_message_id=response_to_message_id,
        success=success,
        result=result,
        error=error,
        execution_time=execution_time
    )


def create_human_message(user_id: str, content: str, session_id: str = None,
                         intent: str = None) -> HumanMessage:
    """Helper per creare messaggio dall'utente"""
    return HumanMessage(
        user_id=user_id,
        content=content,
        session_id=session_id,
        intent=intent
    )


def create_system_message(target_agent: str, instruction_text: str,
                          instruction_type: str = "instruction",
                          expires_in_minutes: int = None,
                          priority_level: int = 1) -> SystemMessage:
    """Helper per creare istruzione di sistema per agent"""
    expires_at = None
    if expires_in_minutes:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)

    return SystemMessage(
        target_agent=target_agent,
        instruction_text=instruction_text,
        instruction_type=instruction_type,
        expires_at=expires_at,
        priority_level=priority_level
    )


def create_agent_result(success: bool, agent_id: str,
                        data: Dict = None, error: str = None,
                        execution_time: float = 0.0,
                        react_steps: int = 0,
                        tools_used: List[str] = None,
                        observations: List[str] = None) -> AgentResult:
    """Helper per creare AgentResult standardizzato"""
    return AgentResult(
        success=success,
        agent_id=agent_id,
        data=data,
        error=error,
        execution_time=execution_time,
        react_steps=react_steps,
        tools_used=tools_used or [],
        observations=observations or []
    )


if __name__ == "__main__":
    # Test AgentResult
    result = create_agent_result(
        success=True,
        agent_id="test_agent",
        data={"processed": 5},
        execution_time=1.2,
        react_steps=3,
        tools_used=["gmail_reader", "calculator"],
        observations=["Email found", "Calculation completed", "Task finished"]
    )
    print("AgentResult:", result.model_dump_json(indent=2))

    # Test SystemMessage
    system_msg = create_system_message(
        target_agent="rag_agent",
        instruction_text="Focus only on documents from 2024",
        instruction_type="constraint",
        expires_in_minutes=60,
        priority_level=2
    )
    print("\nSystemMessage:", system_msg.model_dump_json(indent=2))