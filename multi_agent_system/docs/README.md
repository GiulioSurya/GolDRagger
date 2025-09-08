# Sistema Multi-Agent con Pattern ReAct

Un sistema multi-agent orchestrato che utilizza il pattern ReAct (Reasoning and Acting) per coordinare agenti specializzati attraverso una blackboard condivisa.

## Panoramica del Sistema

### Architettura Generale

Il sistema si basa su diversi componenti chiave:

- **BlackBoard**: Centro di comunicazione condiviso per task management e system instructions
- **AgentManager**: Orchestratore principale che coordina gli agent usando ReAct pattern  
- **BaseAgent**: Classe base che implementa automaticamente il loop ReAct per gli agent specializzati
- **ToolBase**: Sistema per creare strumenti che gli agent possono utilizzare
- **LLM Client**: Interfaccia per modelli linguistici (IBM WatsonX/Llama 3.3 70B)

### Flusso di Funzionamento

1. Il Manager riceve una richiesta dall'utente
2. Usa ReAct pattern per ragionare e decidere quali agent coinvolgere
3. Crea task specifici sulla BlackBoard assegnandoli agli agent appropriati
4. Gli Agent ricevono automaticamente i loro task e li eseguono con ReAct
5. I risultati vengono consolidati e restituiti all'utente

## Setup Iniziale

### 1. Inizializzazione della BlackBoard

La BlackBoard è il centro nevralgico per la comunicazione tra agent:

```python
from multi_agent_system.core.black_board import BlackBoard

# Crea la blackboard condivisa
blackboard = BlackBoard()

# La blackboard gestisce automaticamente:
# - Task assignment e tracking
# - System instructions dinamiche  
# - Comunicazione tra agent
# - Storage thread-safe con notifiche
```

### 2. Inizializzazione del Client LLM

```python
from multi_agent_system.core.llm import LLM

# Inizializza client LLM (Singleton pattern)
llm_client = LLM()

# Il client è configurato per:
# - IBM WatsonX con Llama 3.3 70B
# - Parametri ottimizzati per ReAct
# - Formato system/user prompt separati
```

### 3. Inizializzazione del Manager

```python
from multi_agent_system.core.manager import AgentManager

# Crea il manager orchestratore
manager = AgentManager(
    blackboard=blackboard,
    llm_client=llm_client
)

# Il manager è pronto per:
# - Registrare agent
# - Orchestrare richieste utente
# - Gestire task complex multi-step
```

## Creazione di un Tool Custom

I tool sono strumenti che gli agent possono utilizzare. Ecco come crearne uno:

```python
from multi_agent_system.core.tool_base import (
    ToolBase, ToolResult, ParameterSchema, ParameterType,
    create_parameter_schema, create_success_result, create_error_result
)
import asyncio
import json

class WeatherTool(ToolBase):
    """Tool per recuperare informazioni meteorologiche"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
        # Definisce parametri del tool
        parameters = [
            create_parameter_schema(
                name="city",
                param_type=ParameterType.STRING,
                description="Nome della città per cui recuperare il meteo",
                required=True
            ),
            create_parameter_schema(
                name="country",
                param_type=ParameterType.STRING, 
                description="Codice paese (es: IT, US, FR)",
                required=False,
                default_value="IT"
            ),
            create_parameter_schema(
                name="units",
                param_type=ParameterType.STRING,
                description="Unità di misura per temperatura",
                required=False,
                default_value="celsius",
                allowed_values=["celsius", "fahrenheit", "kelvin"]
            )
        ]
        
        # Inizializza classe base
        super().__init__(
            name="weather_lookup",
            description="Recupera informazioni meteorologiche per una città",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["weather", "api", "external_data"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Esegue la ricerca meteo"""
        start_time = time.time()
        
        try:
            city = kwargs.get("city")
            country = kwargs.get("country", "IT")
            units = kwargs.get("units", "celsius")
            
            # Simula chiamata API (sostituire con vera API)
            await asyncio.sleep(0.5)  # Simula latenza rete
            
            # Simula risposta API
            weather_data = {
                "location": f"{city}, {country}",
                "temperature": 22 if units == "celsius" else 72,
                "units": units,
                "condition": "Partly cloudy",
                "humidity": 65,
                "wind_speed": 12,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return create_success_result(
                data=weather_data,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return create_error_result(
                error=f"Errore recupero meteo: {str(e)}",
                execution_time=time.time() - start_time
            )
```

## Creazione di un Agent Custom

Gli agent ereditano da BaseAgent e devono solo implementare `setup_tools()`:

```python
from multi_agent_system.core.base_agent import BaseAgent

class WeatherAgent(BaseAgent):
    """Agent specializzato per informazioni meteorologiche"""
    
    def __init__(self, agent_id: str, blackboard: BlackBoard, 
                 llm_client, api_credentials: dict):
        # Salva credenziali
        self.api_credentials = api_credentials
        
        # Inizializza BaseAgent (auto-setup ReAct)
        super().__init__(agent_id, blackboard, llm_client)
        
        # Inizializza automaticamente
        self.initialize()
    
    def setup_tools(self):
        """UNICO metodo da implementare - registra tool specifici"""
        
        # Crea e registra weather tool
        weather_tool = WeatherTool(
            api_key=self.api_credentials.get("weather_api_key")
        )
        self.register_tool(weather_tool)
        
        # Può registrare più tool
        # location_tool = LocationTool()
        # self.register_tool(location_tool)
        
        print(f"WeatherAgent {self.agent_id}: Tool meteo registrato")
    
    def get_capabilities(self) -> dict:
        """Override per descrizione custom"""
        return {
            "agent_id": self.agent_id,
            "class_name": "WeatherAgent", 
            "description": "Agent specializzato per informazioni meteorologiche e previsioni",
            "tools": self.get_available_tools(),
            "specializations": ["weather_data", "location_lookup", "forecasting"]
        }
```

## Assemblaggio Completo del Sistema

Ecco come mettere insieme tutti i componenti:

```python
import asyncio
from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.manager import AgentManager
from multi_agent_system.core.llm import LLM
from multi_agent_system.core.messages import create_human_message

async def main():
    print("Inizializzazione Sistema Multi-Agent")
    print("=" * 50)
    
    # 1. Inizializza componenti core
    blackboard = BlackBoard()
    llm_client = LLM()
    manager = AgentManager(blackboard, llm_client)
    
    # 2. Crea e registra agent specializzati
    weather_agent = WeatherAgent(
        agent_id="weather_agent",
        blackboard=blackboard,
        llm_client=llm_client,
        api_credentials={"weather_api_key": "your_key_here"}
    )
    
    # Registra agent nel manager
    success = manager.register_agent(weather_agent)
    print(f"Weather Agent registrato: {success}")
    
    # 3. Verifica stato sistema
    health = manager.health_check()
    print(f"Sistema pronto - Agent registrati: {health['registered_agents']}")
    
    # 4. Esegui richiesta utente
    user_request = create_human_message(
        user_id="user123",
        content="Che tempo fa a Roma oggi? Dammi anche qualche dettaglio sul vento"
    )
    
    print(f"\nProcessing: {user_request.content}")
    print("-" * 30)
    
    # 5. Manager orchestra la risposta
    result = await manager.handle_user_request(user_request)
    
    # 6. Mostra risultato
    if result["success"]:
        print(f"Risposta: {result['response']}")
        print(f"Agent utilizzati: {result['agents_used']}")
        print(f"Tempo esecuzione: {result['execution_time']:.2f}s")
    else:
        print(f"Errore: {result.get('error', 'Unknown error')}")
    
    # 7. Statistiche finali
    print(f"\nStatistiche Manager:")
    stats = manager.get_stats()
    print(f"  - Richieste gestite: {stats['requests_handled']}")
    print(f"  - Task creati: {stats['tasks_created']}")
    print(f"  - Task completati: {stats['tasks_completed']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Pattern ReAct Automatico

Gli agent utilizzano automaticamente il pattern ReAct:

### Flusso ReAct Standard
```
1. Thought: "L'utente vuole il meteo di Roma"
2. Action: weather_lookup: {"city": "Roma", "country": "IT"}  
3. PAUSE: (attende esecuzione tool)
4. Observation: "Tool eseguito - temperatura 22°C, nuvoloso"
5. Thought: "Ho i dati, posso rispondere"
6. Answer: "A Roma oggi ci sono 22°C con cielo parzialmente nuvoloso..."
```

### Vantaggi del Sistema

- **ReAct Automatico**: Ogni agent eredita il loop ReAct completo
- **Tool Auto-Discovery**: System prompt generato automaticamente dai tool disponibili  
- **Task Orchestration**: Manager coordina agent multipli per richieste complesse
- **Thread-Safe**: BlackBoard gestisce concorrenza automaticamente
- **Extensible**: Facile aggiungere nuovi agent e tool
- **Observable**: Tracking completo di task, step ReAct e performance

## System Instructions Dinamiche

Il sistema supporta istruzioni dinamiche per modificare il comportamento degli agent:

```python
# Manager può dare istruzioni specifiche
instruction_id = manager.set_agent_instruction(
    agent_id="weather_agent",
    instruction="Focus only on current weather, not forecasts",
    instruction_type="constraint",
    expires_in_minutes=60
)

# Broadcast a tutti gli agent
manager.broadcast_system_instruction(
    instruction="Be more concise in responses",
    instruction_type="behavior"
)
```

## Monitoraggio e Debug

```python
# Health check completo
health = manager.health_check()
print(f"Status sistema: {health}")

# Statistiche blackboard
bb_stats = blackboard.get_stats()
print(f"BlackBoard: {bb_stats['total_entries']} entries")

# Storia task
task_stats = blackboard.get_task_stats()
print(f"Task: {task_stats}")

# Cambiamenti recenti
changes = blackboard.get_change_history(limit=10)
for change in changes:
    print(f"- {change['change_type']}: {change['key']}")
```

## Conclusioni

Questo sistema multi-agent fornisce:

- **Facilità d'uso**: Gli agent derivati implementano solo `setup_tools()`
- **Potenza**: ReAct pattern automatico con tool discovery
- **Scalabilità**: Architettura modulare per sistemi complessi
- **Robustezza**: Gestione errori e thread-safety integrati
- **Osservabilità**: Tracking completo di tutte le operazioni

Per iniziare, segui i passi di assemblaggio completo e personalizza agent e tool per le tue esigenze specifiche.