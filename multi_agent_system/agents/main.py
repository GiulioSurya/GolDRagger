#!/usr/bin/env python3
"""
main.py - Test del sistema multi-agent con WeatherAgent
"""

import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Import dei moduli del sistema multi-agent
from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.manager import AgentManager
from multi_agent_system.core.llm import LLM
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.messages import create_human_message
from multi_agent_system.core.tool_base import (
    ToolBase, ToolResult, ParameterSchema, ParameterType,
    create_parameter_schema, create_success_result, create_error_result
)

# ============================================================================
# CONFIGURAZIONE TEST QUERY
# ============================================================================

# Modifica questa query per testare diversi scenari
TEST_QUERY = "Che tempo fa a Roma oggi?"


# Altre query che puoi provare:
# TEST_QUERY = "Dimmi il meteo di Roma con le previsioni"
# TEST_QUERY = "Come è il tempo a Roma?"
# TEST_QUERY = "Previsioni meteo per Roma per i prossimi giorni"
# TEST_QUERY = "Temperatura e umidità a Roma"


# ============================================================================
# IMPLEMENTAZIONE WEATHER TOOL
# ============================================================================

class WeatherTool(ToolBase):
    """Tool per recuperare informazioni meteorologiche (simulato per demo)"""

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
            ),
            create_parameter_schema(
                name="include_forecast",
                param_type=ParameterType.BOOLEAN,
                description="Se includere previsioni per i prossimi giorni",
                required=False,
                default_value=False
            )
        ]

        # Inizializza classe base
        super().__init__(
            name="weather_lookup",
            description="Recupera informazioni meteorologiche correnti e previsioni per una città specifica",
            parameters_schema=parameters,
            version="1.2.0",
            tags=["weather", "api", "external_data", "location"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Esegue la ricerca meteo"""
        start_time = time.time()

        try:
            city = kwargs.get("city")
            country = kwargs.get("country", "IT")
            units = kwargs.get("units", "celsius")
            include_forecast = kwargs.get("include_forecast", False)

            # Simula chiamata API esterna
            time.sleep(0.5)  # Simula latenza rete

            # Database simulato di città e condizioni
            weather_db = {
                "roma": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
                "milano": {"temp": 18, "condition": "Rainy", "humidity": 80},
                "napoli": {"temp": 25, "condition": "Sunny", "humidity": 55},
                "firenze": {"temp": 20, "condition": "Cloudy", "humidity": 70},
                "london": {"temp": 15, "condition": "Foggy", "humidity": 85},
                "paris": {"temp": 19, "condition": "Clear", "humidity": 60},
                "new york": {"temp": 23, "condition": "Thunderstorm", "humidity": 75}
            }

            # Cerca dati città (case insensitive)
            city_data = weather_db.get(city.lower(), {
                "temp": 20, "condition": "Unknown", "humidity": 60
            })

            # Conversione temperature
            temp_celsius = city_data["temp"]
            if units == "fahrenheit":
                temperature = (temp_celsius * 9 / 5) + 32
            elif units == "kelvin":
                temperature = temp_celsius + 273.15
            else:
                temperature = temp_celsius

            # Costruisce risposta base
            weather_data = {
                "location": f"{city.title()}, {country.upper()}",
                "temperature": round(temperature, 1),
                "units": units,
                "condition": city_data["condition"],
                "humidity": city_data["humidity"],
                "wind_speed": 12,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_source": "SimulatedWeatherAPI"
            }

            # Aggiunge previsioni se richieste
            if include_forecast:
                weather_data["forecast"] = [
                    {"day": "Tomorrow", "temp": temperature + 2, "condition": "Sunny"},
                    {"day": "Day+2", "temp": temperature - 1, "condition": "Cloudy"},
                    {"day": "Day+3", "temp": temperature + 3, "condition": "Clear"}
                ]

            execution_time = time.time() - start_time

            return create_success_result(
                data=weather_data,
                execution_time=execution_time,
                metadata={"city_found": city.lower() in weather_db}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Errore recupero meteo per {city}: {str(e)}"

            return create_error_result(
                error=error_msg,
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )


# ============================================================================
# WEATHER AGENT
# ============================================================================

class WeatherAgent(BaseAgent):
    """Agent specializzato per informazioni meteorologiche"""

    def __init__(self, agent_id: str, blackboard: BlackBoard,
                 llm_client, api_credentials: dict):
        self.api_credentials = api_credentials
        super().__init__(agent_id, blackboard, llm_client)
        self.initialize()

    def setup_tools(self):
        """Registra tool specifici del WeatherAgent"""
        # Crea e registra weather tool
        weather_tool = WeatherTool(
            api_key=self.api_credentials.get("weather_api_key", "demo_key")
        )
        self.register_tool(weather_tool)

    def get_capabilities(self) -> dict:
        """Capabilities custom per WeatherAgent"""
        return {
            "agent_id": self.agent_id,
            "class_name": "WeatherAgent",
            "description": "Agent specializzato per informazioni meteorologiche e previsioni del tempo",
            "tools": self.get_available_tools(),
            "specializations": ["weather_data", "location_lookup", "forecasting", "climate_info"],
            "supported_units": ["celsius", "fahrenheit", "kelvin"],
            "supported_countries": ["IT", "US", "FR", "UK", "DE"]
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Funzione principale - test con WeatherAgent"""

    print("Sistema Multi-Agent - Test Weather Agent")
    print(f"Query: {TEST_QUERY}")
    print("-" * 50)

    try:
        # ====================================================================
        # INIZIALIZZAZIONE SISTEMA
        # ====================================================================

        # Core components
        blackboard = BlackBoard()
        llm_client = LLM()
        manager = AgentManager(blackboard, llm_client)

        # Creazione Weather Agent
        weather_agent = WeatherAgent(
            agent_id="weather_agent",
            blackboard=blackboard,
            llm_client=llm_client,
            api_credentials={
                "weather_api_key": "demo_weather_key_12345",
                "location_api_key": "demo_location_key_67890"
            }
        )

        # Registrazione agent
        weather_success = manager.register_agent(weather_agent)
        if not weather_success:
            raise Exception("Errore nella registrazione del WeatherAgent")

        # System instructions
        manager.set_agent_instruction(
            agent_id="weather_agent",
            instruction="Always include wind speed and humidity in weather reports",
            instruction_type="behavior",
            expires_in_minutes=30
        )

        # ====================================================================
        # ESECUZIONE QUERY
        # ====================================================================

        print("Processamento query...")

        user_request = create_human_message(
            user_id="test_user",
            content=TEST_QUERY,
            session_id="test_session"
        )

        # Esecuzione
        start_time = time.time()
        result = await manager.handle_user_request(user_request)
        execution_time = time.time() - start_time

        # ====================================================================
        # RISULTATI
        # ====================================================================

        if result["success"]:
            print("\n" + "=" * 60)
            print("RISPOSTA:")
            print("=" * 60)
            print(result['response'])
            print("\n" + "-" * 60)
            print("DETTAGLI:")
            print(f"- Agent utilizzati: {result.get('agents_used', [])}")
            print(f"- Task eseguiti: {len(result.get('tasks_executed', []))}")
            print(f"- Tempo esecuzione: {result['execution_time']:.2f}s")
        else:
            print("\nERRORE:")
            print(f"- {result.get('error', 'Unknown error')}")

        # Statistiche finali
        weather_stats = weather_agent.get_stats()
        print(f"\nStatistiche Weather Agent:")
        print(f"- Task completati: {weather_stats['tasks_completed']}")
        print(f"- Tool utilizzati: {weather_stats['tools_used_count']}")

    except Exception as e:
        print(f"\nErrore: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    finally:
        # Cleanup
        if 'manager' in locals():
            manager.shutdown_all_agents()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Esegui test
    asyncio.run(main())

    print("\nTest completato.")
    print("Per testare query diverse, modifica la variabile TEST_QUERY")
    print("Esempi:")
    print('  TEST_QUERY = "Dimmi il meteo di Roma con le previsioni"')
    print('  TEST_QUERY = "Come è il tempo a Roma?"')
    print('  TEST_QUERY = "Temperatura e umidità a Roma"')