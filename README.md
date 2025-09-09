# Multi-Agent System with Blackboard Architecture

A sophisticated multi-agent framework built with ReAct pattern orchestration, designed for complex task delegation and collaborative AI workflows.

## 🏗 Architecture Overview

The system implements a **Blackboard Architecture** with intelligent orchestration, where agents collaborate through shared memory and the Manager coordinates task execution using the ReAct (Reasoning and Acting) pattern.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Manager     │◄──►│   BlackBoard    │◄──►│     Agents      │
│   (ReAct Loop)  │    │  (Shared State) │    │ (Specialized)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Task Creation  │    │  Notifications  │    │  Tool Execution │
│  Status Monitor │    │  Change History │    │  ReAct Pattern  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **Manager**: Orchestrates agent interactions using ReAct pattern
- **BlackBoard**: Thread-safe shared memory with pub/sub notifications
- **BaseAgent**: Abstract base with ReAct/Acting pattern support
- **ToolBase**: Modular tool system for agent capabilities
- **LLM Providers**: IBM WatsonX and Ollama integration

##  Quick Start

### 1. Environment Setup

```bash
# Create environment file
cp .env.example .env

# Configure your credentials
IBM_WATSONX_API_KEY=your_api_key_here
IBM_WATSONX_PROJECT_ID=your_project_id_here
NOTION_TOKEN=your_notion_token_here
NOTION_PARENT_ID=your_parent_page_id_here
```

### 2. Basic Usage Example

```python
import asyncio
from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.manager import AgentManager
from multi_agent_system.core.llm import LlmProvider
from multi_agent_system.core.messages import create_human_message

async def main():
    # Initialize core components
    blackboard = BlackBoard()
    llm_client = LlmProvider.IBM_WATSONX.get_instance()
    llm_client.__post_init__()  # Initialize the model
    manager = AgentManager(blackboard, llm_client)
    
    # Create and register agents (see examples below)
    weather_agent = WeatherAgent("weather_agent", blackboard, llm_client, api_credentials={})
    manager.register_agent(weather_agent)
    
    # Handle user request
    user_request = create_human_message(
        user_id="user123",
        content="What's the weather like in Rome today?",
        session_id="session1"
    )
    
    result = await manager.handle_user_request(user_request)
    print(f"Response: {result['response']}")

asyncio.run(main())
```

##  Available Agents

The system comes with several pre-built specialized agents:

### WeatherAgent
Provides weather information and forecasts.

```python
from multi_agent_system.main import WeatherAgent

weather_agent = WeatherAgent(
    agent_id="weather_agent",
    blackboard=blackboard,
    llm_client=llm_client,
    api_credentials={"weather_api_key": "your_key"}
)
```

**Capabilities:**
- Current weather lookup by city
- Multi-unit temperature support (Celsius/Fahrenheit/Kelvin)
- Weather forecasts
- Multiple country support

### NotionWriterAgent
Creates and updates Notion pages with markdown support.

```python
from multi_agent_system.notion_agent import NotionWriterAgent

notion_agent = NotionWriterAgent(
    agent_id="notion_agent",
    blackboard=blackboard,
    llm_client=llm_client,
    notion_token="your_notion_token",
    default_parent_type="page_id",
    default_parent_id="your_parent_page_id"
)
```

**Capabilities:**
- Create pages with markdown content
- Update existing pages
- Emoji icon support
- Configurable parent locations

### SyntheticAgent
Specializes in information synthesis and analysis.

```python
from multi_agent_system.syntetic_agent import SyntheticAgent

synthetic_agent = SyntheticAgent(
    agent_id="synthetic_agent",
    blackboard=blackboard,
    llm_client=llm_client
)
```

**Capabilities:**
- Executive summaries
- Comparative analysis
- Data fusion from multiple sources
- Trend analysis
- Multi-step synthesis orchestration

##  Creating Custom Agents

### ReAct Mode Agent (with Tools)

```python
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.tool_base import ToolBase, ParameterType, create_parameter_schema

class MyCustomTool(ToolBase):
    def __init__(self):
        parameters = [
            create_parameter_schema(
                name="input_text",
                param_type=ParameterType.STRING,
                description="Text to process",
                required=True
            )
        ]
        super().__init__(
            name="my_custom_tool",
            description="Processes text in a custom way",
            parameters_schema=parameters
        )
    
    def execute(self, **kwargs):
        input_text = kwargs.get("input_text")
        # Your custom logic here
        result = f"Processed: {input_text}"
        
        return create_success_result(
            data={"result": result},
            execution_time=0.1
        )

class CustomReActAgent(BaseAgent):
    def __init__(self, agent_id: str, blackboard, llm_client):
        # react=True enables tool usage and ReAct pattern
        super().__init__(agent_id, blackboard, llm_client, react=True)
        self.initialize()
    
    def setup_tools(self):
        """Register tools specific to this agent"""
        custom_tool = MyCustomTool()
        self.register_tool(custom_tool)
```

### Acting Mode Agent (Direct Reasoning)

```python
class CustomActingAgent(BaseAgent):
    def __init__(self, agent_id: str, blackboard, llm_client):
        # react=False enables direct reasoning without tools
        super().__init__(agent_id, blackboard, llm_client, react=False)
        self.initialize()
    
    def setup_tools(self):
        """No tools needed for Acting mode"""
        pass
```

## 🛠 Creating Custom Tools

### Basic Tool Structure

```python
from multi_agent_system.core.tool_base import ToolBase, ToolResult, ParameterType, create_parameter_schema

class DatabaseQueryTool(ToolBase):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
        parameters = [
            create_parameter_schema(
                name="query",
                param_type=ParameterType.STRING,
                description="SQL query to execute",
                required=True
            ),
            create_parameter_schema(
                name="limit",
                param_type=ParameterType.INTEGER,
                description="Maximum number of results",
                required=False,
                default_value=100,
                min_value=1,
                max_value=1000
            )
        ]
        
        super().__init__(
            name="database_query",
            description="Executes SQL queries against the database",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["database", "sql", "data_access"]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        
        try:
            query = kwargs.get("query")
            limit = kwargs.get("limit", 100)
            
            # Your database logic here
            results = self._execute_query(query, limit)
            
            return create_success_result(
                data={"results": results, "count": len(results)},
                execution_time=time.time() - start_time,
                metadata={"query": query, "limit": limit}
            )
            
        except Exception as e:
            return create_error_result(
                error=f"Database query failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _execute_query(self, query: str, limit: int):
        # Implement your database connection logic
        pass
```

##  BlackBoard Communication

### Task Management

The BlackBoard provides sophisticated task management for agent coordination:

```python
# Manager creates tasks for agents
task_id = blackboard.create_task(
    assigned_to="weather_agent",
    task_type="weather_lookup",
    task_data={
        "city": "Rome",
        "include_forecast": True
    },
    created_by="manager"
)

# Agents automatically receive and process tasks
# Results are stored back on the blackboard

# Check task status
task_status = blackboard.get_task_status(task_id, "weather_agent")
print(f"Status: {task_status['status']}")
print(f"Result: {task_status['result']}")
```

### Dynamic System Instructions

Modify agent behavior at runtime:

```python
# Add temporary instruction to agent
instruction_id = blackboard.add_system_instruction(
    agent_id="weather_agent",
    instruction_text="Always include humidity and wind speed in weather reports",
    instruction_type="behavior",
    expires_in_minutes=30,
    priority_level=1
)

# Remove instruction
blackboard.remove_system_instruction("weather_agent", instruction_id)
```

### Change Notifications

Subscribe to blackboard changes:

```python
def on_change(change):
    print(f"Change detected: {change.change_type} for key {change.key}")

blackboard.subscribe_to_changes(on_change)
```

##  LLM Configuration

### IBM WatsonX Setup

```python
from multi_agent_system.core.llm import LlmProvider

# Automatic initialization from environment variables
llm_client = LlmProvider.IBM_WATSONX.get_instance()
llm_client.__post_init__()

# Configure for ReAct mode
llm_client.set_react_mode(True)
llm_client.set_temperature(0.1)
```

### Ollama Setup

```python
# Configure Ollama endpoint
llm_client = LlmProvider.OLLAMA.get_instance()
llm_client.__post_init__()
```

##  Monitoring and Observability

### Agent Statistics

```python
# Get agent performance metrics
stats = agent.get_stats()
print(f"Tasks completed: {stats['tasks_completed']}")
print(f"Tools used: {stats['tools_used_count']}")
print(f"React steps: {stats['total_react_steps']}")
```

### System Health Check

```python
# Monitor overall system health
health = manager.health_check()
print(f"Registered agents: {health['registered_agents']}")
print(f"Active tasks: {health['active_tasks']}")
print(f"Agent statuses: {health['agent_statuses']}")
```

### BlackBoard Analytics

```python
# BlackBoard usage statistics
bb_stats = blackboard.get_stats()
print(f"Total entries: {bb_stats['total_entries']}")
print(f"Total reads: {bb_stats['total_reads']}")
print(f"Change history: {bb_stats['history_size']}")

# Task analytics
task_stats = blackboard.get_task_stats()
print(f"Tasks by status: {task_stats['by_status']}")
print(f"Tasks by agent: {task_stats['by_agent']}")
```

##  Advanced Configuration

### Manager Orchestration

The Manager uses ReAct pattern for intelligent task delegation:

```python
# The Manager automatically:
# 1. Analyzes user requests
# 2. Determines required agents
# 3. Creates and monitors tasks
# 4. Synthesizes final responses

# Custom manager instructions
manager.set_agent_instruction(
    agent_id="weather_agent",
    instruction="Prioritize current conditions over forecasts",
    expires_in_minutes=60
)
```

### Error Handling and Recovery

```python
# Agents handle failures gracefully
try:
    result = await manager.handle_user_request(user_request)
    if result["success"]:
        print(f"Success: {result['response']}")
    else:
        print(f"Failed: {result['error']}")
except Exception as e:
    print(f"System error: {str(e)}")
```

##  Security Considerations

- **API Keys**: Store in environment variables, never in code
- **Input Validation**: All tools include parameter validation
- **Sandboxing**: Agents operate in isolated contexts
- **Rate Limiting**: Implement at LLM provider level

##  Performance Tips

1. **Agent Mode Selection**:
   - Use ReAct mode only when tools are needed
   - Use Acting mode for reasoning-only tasks

2. **Tool Design**:
   - Keep tools focused and single-purpose
   - Include proper error handling
   - Use efficient parameter validation

3. **BlackBoard Optimization**:
   - Use tags for efficient querying
   - Clean up old tasks periodically
   - Monitor memory usage in long-running systems

##  Contributing

When adding new agents or tools:

1. Follow the established patterns (ReAct/Acting)
2. Include comprehensive parameter schemas
3. Add proper error handling and validation
4. Document capabilities and usage examples
5. Include unit tests for custom components

##  License

This project is designed for educational and research purposes. Ensure compliance with all LLM provider terms of service.

---

**Built with**: Python 3.8+, IBM WatsonX AI, Ollama, Pydantic, AsyncIO