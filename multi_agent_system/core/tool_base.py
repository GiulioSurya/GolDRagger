from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field
import time
from datetime import datetime, timezone
import json


class ToolStatus(Enum):
    """Stati possibili di un tool"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ParameterType(Enum):
    """Tipi di parametri supportati"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class ParameterSchema(BaseModel):
    """Schema per definire parametri del tool"""
    name: str = Field(..., description="Nome del parametro")
    param_type: ParameterType = Field(..., description="Tipo del parametro")
    required: bool = Field(default=True, description="Se il parametro è obbligatorio")
    description: str = Field(..., description="Descrizione del parametro")
    default_value: Optional[Any] = Field(default=None, description="Valore di default")
    allowed_values: Optional[List[Any]] = Field(default=None, description="Valori permessi")
    min_value: Optional[Union[int, float]] = Field(default=None, description="Valore minimo")
    max_value: Optional[Union[int, float]] = Field(default=None, description="Valore massimo")

    def validate_value(self, value: Any) -> bool:
        """Valida un valore contro questo schema"""
        # Controlla se required
        if self.required and value is None:
            return False

        # Se non required e None, OK
        if not self.required and value is None:
            return True

        # Controlla tipo
        if not self._check_type(value):
            return False

        # Controlla valori permessi
        if self.allowed_values and value not in self.allowed_values:
            return False

        # Controlla range per numeri
        if self.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        return True

    def _check_type(self, value: Any) -> bool:
        """Controlla il tipo del valore"""
        if self.param_type == ParameterType.STRING:
            return isinstance(value, str)
        elif self.param_type == ParameterType.INTEGER:
            return isinstance(value, int)
        elif self.param_type == ParameterType.FLOAT:
            return isinstance(value, (int, float))
        elif self.param_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        elif self.param_type == ParameterType.ARRAY:
            return isinstance(value, list)
        elif self.param_type == ParameterType.OBJECT:
            return isinstance(value, dict)
        elif self.param_type == ParameterType.ANY:
            return True
        return False


class ToolResult(BaseModel):
    """Risultato standardizzato dell'esecuzione di un tool"""
    success: bool = Field(..., description="Se l'operazione è riuscita")
    data: Optional[Any] = Field(default=None, description="Dati risultanti")
    error: Optional[str] = Field(default=None, description="Messaggio di errore")
    execution_time: float = Field(..., description="Tempo di esecuzione in secondi")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadati aggiuntivi")

    @property
    def is_success(self) -> bool:
        """Alias per success"""
        return self.success

    @property
    def has_data(self) -> bool:
        """Se il risultato ha dati"""
        return self.data is not None

    @property
    def has_error(self) -> bool:
        """Se il risultato ha errori"""
        return self.error is not None


class ToolBase(ABC):
    """Classe base astratta per tutti i tool del sistema (SYNC VERSION)"""

    def __init__(
            self,
            name: str,
            description: str,
            parameters_schema: List[ParameterSchema],
            version: str = "1.0.0",
            tags: List[str] = None
    ):
        """
        Inizializza il tool base

        Args:
            name: Nome identificativo del tool
            description: Descrizione human-readable
            parameters_schema: Lista degli schemi dei parametri
            version: Versione del tool
            tags: Tag per categorizzazione
        """
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self.version = version
        self.tags = tags or []
        self.created_at = datetime.now(timezone.utc)
        self._status = ToolStatus.AVAILABLE
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = None

        # Crea mapping nome -> schema per lookup veloce
        self._param_map = {param.name: param for param in parameters_schema}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Esegue il tool con i parametri forniti (SYNC VERSION)

        Args:
            **kwargs: Parametri per l'esecuzione

        Returns:
            ToolResult: Risultato dell'esecuzione

        Raises:
            NotImplementedError: Se non implementato nella classe derivata
        """
        pass

    def safe_execute(self, **kwargs) -> ToolResult:
        """
        Esecuzione sicura con validazione e timing automatici (SYNC VERSION)

        Args:
            **kwargs: Parametri per l'esecuzione

        Returns:
            ToolResult: Risultato dell'esecuzione (sempre, anche in caso di errore)
        """
        start_time = time.time()

        try:
            # Valida parametri
            validation_result = self.validate_parameters(kwargs)
            if not validation_result["valid"]:
                return ToolResult(
                    success=False,
                    error=f"Parameter validation failed: {validation_result['errors']}",
                    execution_time=time.time() - start_time
                )

            # Controlla se il tool è disponibile
            if not self.is_available():
                return ToolResult(
                    success=False,
                    error=f"Tool {self.name} is not available (status: {self._status.value})",
                    execution_time=time.time() - start_time
                )

            # Esegue il tool (SYNC)
            result = self.execute(**kwargs)

            # Aggiorna statistiche
            execution_time = time.time() - start_time
            self._update_stats(execution_time)

            # Assicura che execution_time sia settato
            if result.execution_time == 0:  # Se non settato dall'implementazione
                result.execution_time = execution_time

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(execution_time, error=True)

            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )

    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida i parametri ricevuti contro lo schema definito

        Args:
            params: Dizionario con i parametri da validare

        Returns:
            Dict con risultato validazione: {"valid": bool, "errors": List[str]}
        """
        errors = []

        # Controlla parametri richiesti
        for param_schema in self.parameters_schema:
            param_name = param_schema.name
            param_value = params.get(param_name)

            # Se parametro richiesto ma non fornito
            if param_schema.required and param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
                continue

            # Se parametro fornito, valida
            if param_name in params:
                if not param_schema.validate_value(param_value):
                    errors.append(f"Invalid value for parameter {param_name}: {param_value}")

        # Controlla parametri extra non definiti
        defined_params = {param.name for param in self.parameters_schema}
        extra_params = set(params.keys()) - defined_params
        if extra_params:
            errors.append(f"Unknown parameters: {list(extra_params)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def get_schema(self) -> Dict[str, Any]:
        """
        Restituisce lo schema completo del tool in formato JSON

        Returns:
            Dict: Schema del tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.param_type.value,
                    "required": param.required,
                    "description": param.description,
                    "default": param.default_value,
                    "allowed_values": param.allowed_values,
                    "min_value": param.min_value,
                    "max_value": param.max_value
                }
                for param in self.parameters_schema
            ],
            "status": self._status.value,
            "created_at": self.created_at.isoformat()
        }

    def get_name(self) -> str:
        """Nome del tool"""
        return self.name

    def get_description(self) -> str:
        """Descrizione del tool"""
        return self.description

    def get_version(self) -> str:
        """Versione del tool"""
        return self.version

    def get_tags(self) -> List[str]:
        """Tag del tool"""
        return self.tags.copy()

    def is_available(self) -> bool:
        """
        Controlla se il tool è disponibile per l'esecuzione

        Returns:
            bool: True se disponibile, False altrimenti
        """
        return self._status == ToolStatus.AVAILABLE

    def set_status(self, status: ToolStatus, reason: str = None):
        """
        Imposta lo status del tool

        Args:
            status: Nuovo status
            reason: Motivo del cambio status (opzionale)
        """
        old_status = self._status
        self._status = status

        # Log del cambio status se necessario
        if reason:
            print(f"Tool {self.name} status changed from {old_status.value} to {status.value}: {reason}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Statistiche di utilizzo del tool

        Returns:
            Dict: Statistiche di esecuzione
        """
        avg_execution_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0 else 0
        )

        return {
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_execution_time,
            "last_execution_time": self._last_execution_time,
            "status": self._status.value
        }

    def _update_stats(self, execution_time: float, error: bool = False):
        """Aggiorna statistiche interne"""
        if not error:  # Solo se esecuzione riuscita
            self._execution_count += 1
            self._total_execution_time += execution_time
        self._last_execution_time = datetime.now(timezone.utc)

    def __str__(self) -> str:
        return f"Tool(name={self.name}, version={self.version}, status={self._status.value})"

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description[:50]}...', parameters={len(self.parameters_schema)})"


# ===== UTILITY FUNCTIONS =====

def create_parameter_schema(
        name: str,
        param_type: ParameterType,
        description: str,
        required: bool = True,
        default_value: Any = None,
        allowed_values: List[Any] = None,
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None
) -> ParameterSchema:
    """
    Helper function per creare ParameterSchema facilmente

    Args:
        name: Nome parametro
        param_type: Tipo parametro
        description: Descrizione
        required: Se obbligatorio
        default_value: Valore default
        allowed_values: Valori permessi
        min_value: Valore minimo
        max_value: Valore massimo

    Returns:
        ParameterSchema: Schema del parametro
    """
    return ParameterSchema(
        name=name,
        param_type=param_type,
        description=description,
        required=required,
        default_value=default_value,
        allowed_values=allowed_values,
        min_value=min_value,
        max_value=max_value
    )


def create_success_result(
        data: Any,
        execution_time: float,
        metadata: Dict[str, Any] = None
) -> ToolResult:
    """Helper per creare ToolResult di successo"""
    return ToolResult(
        success=True,
        data=data,
        execution_time=execution_time,
        metadata=metadata or {}
    )


def create_error_result(
        error: str,
        execution_time: float,
        metadata: Dict[str, Any] = None
) -> ToolResult:
    """Helper per creare ToolResult di errore"""
    return ToolResult(
        success=False,
        error=error,
        execution_time=execution_time,
        metadata=metadata or {}
    )


# --- Script principale per test ---
if __name__ == "__main__":
    from datetime import datetime
    from typing import Any, Dict, List


    # --- Tool concreto SYNC ---
    class AddNumbersTool(ToolBase):
        """Tool che somma due numeri (SYNC VERSION)"""

        def execute(self, **kwargs) -> ToolResult:
            start_time = time.time()
            try:
                a = kwargs.get("a")
                b = kwargs.get("b")
                result = a + b
                return ToolResult(
                    success=True,
                    data={"sum": result},
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                    execution_time=time.time() - start_time
                )


    # Creiamo lo schema dei parametri
    parameters = [
        create_parameter_schema("a", ParameterType.FLOAT, "Primo numero"),
        create_parameter_schema("b", ParameterType.FLOAT, "Secondo numero")
    ]

    # Inizializziamo il tool
    add_tool = AddNumbersTool(
        name="AddNumbers",
        description="Somma due numeri",
        parameters_schema=parameters
    )

    print("=== TEST TOOL SYNC ===")

    # Esecuzione corretta (SYNC)
    result = add_tool.safe_execute(a=10, b=25)
    print("Risultato corretto:", result.model_dump_json(indent=2))

    # Esecuzione con parametro mancante
    result2 = add_tool.safe_execute(a=10)
    print("Risultato errore:", result2.model_dump_json(indent=2))

    print("Tool sync funziona perfettamente!")