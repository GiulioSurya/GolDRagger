# Flusso di Esecuzione Query nel Sistema Multi-Agent

Questo documento traccia il percorso completo di una query utente attraverso il sistema multi-agent, mostrando ogni chiamata di metodo e il suo scopo.

## Scenario di Esempio
**Query utente:** "Che tempo fa a Roma oggi?"

---

## 1. INGRESSO DELLA RICHIESTA

### Modulo: `main.py`
- **Metodo chiamato:** `main()` 
- **Descrizione:** Entry point del sistema, inizializza tutti i componenti e gestisce l'input utente
- **Azione:** Crea oggetto `HumanMessage` con la query dell'utente

### Modulo: `messages.py`
- **Metodo chiamato:** `create_human_message()`
- **Descrizione:** Factory per creare messaggi standardizzati dall'utente
- **Azione:** Valida e formatta la richiesta in un oggetto strutturato con ID univoco, timestamp e metadati

---

## 2. ORCHESTRAZIONE MANAGER

### Modulo: `manager.py`
- **Metodo chiamato:** `handle_user_request()`
- **Descrizione:** Entry point del Manager per elaborare richieste utente
- **Azione:** Riceve HumanMessage, avvia timer per statistiche, delega a orchestrazione ReAct

### Modulo: `manager.py`
- **Metodo chiamato:** `_orchestrate_with_react()`
- **Descrizione:** Cuore dell'orchestrazione - implementa pattern ReAct per decidere quali agent coinvolgere
- **Azione:** Avvia loop iterativo di reasoning usando LLM per analizzare la richiesta

### Modulo: `manager.py`
- **Metodo chiamato:** `_build_manager_prompt()`
- **Descrizione:** Costruisce system prompt dinamico per LLM con lista agent disponibili e loro capabilities
- **Azione:** Scansiona agent registrati, estrae tool e capacità, genera prompt ReAct completo

---

## 3. REASONING LLM

### Modulo: `llm.py`
- **Metodo chiamato:** `invoke()`
- **Descrizione:** Interfaccia per chiamate LLM - gestisce formato prompt e comunicazione con IBM WatsonX
- **Azione:** Invia system prompt (agent disponibili) + user prompt (query) al modello Llama 3.3 70B

### Modulo: `llm.py` (internamente)
- **Metodo chiamato:** `generate_text()`
- **Descrizione:** Chiamata vera e propria al modello IBM WatsonX con parametri ottimizzati
- **Azione:** LLM analizza query meteo e decide di usare weather_agent, genera response ReAct

---

## 4. PARSING RISPOSTA MANAGER

### Modulo: `manager.py`
- **Metodo chiamato:** `_process_create_task_action()`
- **Descrizione:** Parser per azioni "create_task" dal LLM - estrae parametri JSON dal reasoning
- **Azione:** Usa regex per estrarre agent_id="weather_agent", task_type="weather_lookup", task_data con città

### Modulo: `manager.py`
- **Metodo chiamato:** Validazione parametri (interno)
- **Descrizione:** Verifica che l'agent richiesto sia registrato e disponibile
- **Azione:** Controlla se "weather_agent" esiste nel registry, valida formato JSON parametri

---

## 5. CREAZIONE TASK SULLA BLACKBOARD

### Modulo: `black_board.py`
- **Metodo chiamato:** `create_task()`
- **Descrizione:** Crea nuovo task con ID univoco per agent specifico sulla lavagna condivisa
- **Azione:** Genera UUID task, salva con chiave "task_weather_agent_{uuid}", marca status="pending"

### Modulo: `black_board.py`
- **Metodo chiamato:** `update()`
- **Descrizione:** Storage thread-safe per salvare entry sulla blackboard con metadati
- **Azione:** Salva task data, timestamp, tags, notifica observers del cambiamento

### Modulo: `black_board.py`
- **Metodo chiamato:** `_notify_observers()`
- **Descrizione:** Sistema di notifiche per informare agent registrati di cambiamenti
- **Azione:** Triggera callback per weather_agent che ha un task pending

---

## 6. ATTIVAZIONE AGENT

### Modulo: `base_agent.py`
- **Metodo chiamato:** Observer callback `on_blackboard_change()`
- **Descrizione:** Listener automatico che rileva task assegnati a questo agent
- **Azione:** Weather_agent riceve notifica del nuovo task, verifica se è per lui

### Modulo: `base_agent.py`
- **Metodo chiamato:** `_handle_assigned_task()`
- **Descrizione:** Gestisce task ricevuto dalla blackboard, coordina esecuzione
- **Azione:** Cambia status agent a "working", estrae task_data, avvia ReAct loop specifico

### Modulo: `base_agent.py`
- **Metodo chiamato:** `_execute_react_loop()`
- **Descrizione:** Implementazione ReAct per singolo agent - loop Thought → Action → Observation
- **Azione:** Avvia reasoning specifico per task meteo con system prompt auto-generato dai tool

---

## 7. SYSTEM PROMPT AGENT

### Modulo: `base_agent.py`
- **Metodo chiamato:** `_build_react_system_prompt()`
- **Descrizione:** Auto-genera system prompt con descrizioni tool disponibili per questo agent
- **Azione:** Scansiona weather_tool registrato, estrae schema parametri, crea prompt ReAct specializzato

### Modulo: `black_board.py`
- **Metodo chiamato:** `get_system_instructions_for_agent()`
- **Descrizione:** Recupera istruzioni dinamiche attive per questo agent (es. "include wind speed")
- **Azione:** Filtra istruzioni per weather_agent, controlla scadenze, ordina per priorità

---

## 8. REASONING AGENT

### Modulo: `llm.py`
- **Metodo chiamato:** `invoke()` (seconda chiamata)
- **Descrizione:** LLM reasoning specifico per agent con context task meteo
- **Azione:** LLM riceve task "Roma weather" + tool weather_lookup disponibile, genera ReAct response

### Modulo: `base_agent.py`
- **Metodo chiamato:** Parsing LLM response (interno)
- **Descrizione:** Analizza risposta LLM per identificare azioni richieste (Action: tool_name)
- **Azione:** Rileva "Action: weather_lookup: {city: Roma, units: celsius}"

---

## 9. ESECUZIONE TOOL

### Modulo: `base_agent.py`
- **Metodo chiamato:** `_process_action()`
- **Descrizione:** Parser ed executor per azioni tool richieste dall'agent LLM
- **Azione:** Estrae tool_name="weather_lookup" e parametri JSON, valida disponibilità

### Modulo: `base_agent.py`
- **Metodo chiamato:** `use_tool()`
- **Descrizione:** Wrapper per esecuzione sicura tool con validazione parametri
- **Azione:** Verifica tool registrato, delega a safe_execute del tool specifico

### Modulo: `tool_base.py`
- **Metodo chiamato:** `safe_execute()`
- **Descrizione:** Execution wrapper con validazione parametri, timing e error handling
- **Azione:** Valida parametri contro schema, controlla tool availability, misura timing

### Modulo: `tool_base.py`
- **Metodo chiamato:** `validate_parameters()`
- **Descrizione:** Verifica parametri ricevuti contro schema definito nel tool
- **Azione:** Controlla city="Roma" required, units="celsius" valido, include_forecast opzionale

---

## 10. BUSINESS LOGIC TOOL

### Modulo: `main.py` (WeatherTool)
- **Metodo chiamato:** `execute()`
- **Descrizione:** Business logic specifica del tool - core functionality per recupero meteo
- **Azione:** Simula API call, lookup Roma nel database simulato, calcola temperature/forecast

### Modulo: `main.py` (WeatherTool)
- **Metodo chiamato:** Simulazione API (interno)
- **Descrizione:** Logica di business per generare dati meteo realistici
- **Azione:** Recupera dati Roma (22°C, partly cloudy), aggiunge humidity, wind_speed, forecast

---

## 11. RITORNO RISULTATI

### Modulo: `tool_base.py`
- **Metodo chiamato:** `create_success_result()`
- **Descrizione:** Factory per creare ToolResult standardizzato con dati e metadati
- **Azione:** Crea oggetto result con success=True, weather data Roma, execution_time

### Modulo: `base_agent.py`
- **Metodo chiamato:** Observation processing (interno)
- **Descrizione:** Converte tool result in observation text per LLM
- **Azione:** Formatta "Tool weather_lookup executed successfully. Result: {Roma data}"

### Modulo: `base_agent.py`
- **Metodo chiamato:** `_save_observation_to_blackboard()`
- **Descrizione:** Salva step ReAct sulla blackboard per tracking e debug
- **Azione:** Crea entry "observation_weather_agent_{task_id}_step_1" con risultato meteo

---

## 12. FINALIZZAZIONE AGENT

### Modulo: `llm.py`
- **Metodo chiamato:** `invoke()` (terza chiamata)
- **Descrizione:** LLM final reasoning con observation per generare Answer finale
- **Azione:** LLM riceve weather data, genera risposta human-friendly "A Roma oggi..."

### Modulo: `base_agent.py`
- **Metodo chiamato:** `_extract_answer()`
- **Descrizione:** Estrae risposta finale dal LLM response quando contiene "Answer:"
- **Azione:** Parsea "Answer: A Roma oggi ci sono 22°C con cielo parzialmente nuvoloso..."

### Modulo: `black_board.py`
- **Metodo chiamato:** `update_task_result()`
- **Descrizione:** Agent segna task come completato sulla blackboard con risultato
- **Azione:** Aggiorna task status="completed", result={answer, weather_data}, execution_time

---

## 13. RITORNO AL MANAGER

### Modulo: `manager.py`
- **Metodo chiamato:** `_wait_for_task_completion()`
- **Descrizione:** Polling asincrono per attendere completamento task agent
- **Azione:** Check periodici su blackboard, rileva status="completed", recupera risultato

### Modulo: `black_board.py`
- **Metodo chiamato:** `get_task_status()`
- **Descrizione:** Recupera status corrente di un task specifico dalla blackboard
- **Azione:** Ritorna task completo con status, result, execution_time per weather task

### Modulo: `manager.py`
- **Metodo chiamato:** `_extract_answer()` (Manager)
- **Descrizione:** Estrae risposta finale dal Manager LLM quando ha tutti i dati necessari
- **Azione:** Manager vede task completato, genera "Answer: A Roma oggi ci sono 22°C..."

---

## 14. RESPONSE FINALE

### Modulo: `manager.py`
- **Metodo chiamato:** Return to `handle_user_request()`
- **Descrizione:** Ritorno al metodo originale con risultato completo
- **Azione:** Calcola statistics (execution_time, agents_used), formatta response finale

### Modulo: `main.py`
- **Metodo chiamato:** Response processing
- **Descrizione:** Riceve risultato orchestrazione e lo presenta all'utente
- **Azione:** Stampa risposta finale, statistics, agent utilizzati, timing completo

---

## FLUSSO DATI PARALLELI

Durante tutto il processo, questi moduli lavorano in background:

### Modulo: `black_board.py`
- **Funzioni:** Change tracking, observer notifications, statistics
- **Descrizione:** Mantiene log di tutti i cambiamenti, notifica components interessati

### Modulo: `messages.py`
- **Funzioni:** Message validation, serialization
- **Descrizione:** Assicura che tutti i messaggi siano ben formati e tracciabili

### Modulo: `manager.py`
- **Funzioni:** Statistics tracking, health monitoring
- **Descrizione:** Raccoglie metriche performance, monitora stato sistema

---

## TIMING TIPICO

- **Query reception:** < 1ms
- **Manager reasoning:** 200-500ms (LLM call)
- **Task creation:** < 10ms
- **Agent notification:** < 5ms
- **Agent reasoning:** 200-500ms (LLM call)
- **Tool execution:** 50-500ms (API simulation)
- **Result processing:** < 20ms
- **Final assembly:** < 10ms

**Totale tipico:** 0.5-1.5 secondi per query semplice con 1 agent e 1 tool.

---

## PUNTI DI FAILURE E RECOVERY

Ogni step ha error handling che può ritornare errori al chiamante, permettendo graceful degradation e debug dettagliato attraverso logging e blackboard tracking.