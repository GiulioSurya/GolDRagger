# Multi-Agent System with Blackboard Architecture

A multi-agent framework built from scratch implementing blackboard architecture with ReAct pattern coordination. The system enables specialized agents to collaborate on complex workflows through shared memory and intelligent task delegation.

## Architecture Overview

The system uses a blackboard architecture where agents communicate through shared state and a manager coordinates task execution using the ReAct (Reasoning and Acting) pattern.

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

- **MultiAgentSystem**: Main orchestration class providing simplified interface
- **Manager**: Coordinates agent interactions using ReAct pattern for task delegation
- **BlackBoard**: Thread-safe shared memory with change notifications
- **BaseAgent**: Abstract base class supporting both ReAct (tool-enabled) and Acting (reasoning-only) patterns
- **ToolBase**: Framework for creating modular agent capabilities
- **LLM Providers**: Integration layer for IBM WatsonX and Ollama

## Installation and Setup

### Environment Configuration

```bash
# Create environment file
cp .env.example .env

# Configure credentials
# only ollama and watson supported for the moment
IBM_WATSONX_API_KEY=your_api_key_here
IBM_WATSONX_PROJECT_ID=your_project_id_here
NOTION_TOKEN=your_notion_token_here
NOTION_PARENT_ID=your_parent_page_id_here
```

### Basic Usage

```python
from multi_agent_system.main import create_system
import os

# Initialize system with Notion integration
system = create_system(
    notion_token=os.getenv("NOTION_TOKEN"),
    default_parent_type="page_id", 
    default_parent_id=os.getenv("NOTION_PARENT_ID")
)

# Execute simple task
response = system.query_sync("Create a Notion page titled 'Meeting Notes' with agenda and action items sections")

# Execute complex analysis workflow
analysis_query = """
Analyze Q3 performance data:
- Revenue: €61K (up from €52K in Q2)  
- Clients: 18 (up from 15)
- Satisfaction: 85% (up from 82%)

Create executive summary with strategic recommendations and save to Notion as 'Q3 Performance Review'.
"""

result = system.query_sync(analysis_query)
```


## Available Agents

The system currently includes two specialized agents:

### NotionWriterAgent

Handles document creation and management in Notion workspace.

**Capabilities:**
- Create pages with markdown content conversion
- Update existing pages and properties
- Support for emoji icons and rich formatting
- Configurable parent locations (pages or databases)
- Automated template application

**Example Usage:**
```python
# Agent automatically handles Notion API calls
"Create a project status report with sections for progress, risks, and next steps"
```

### SyntheticAgent (Executive Configuration)

Performs data analysis and information synthesis tasks.

**Capabilities:**
- Executive summary generation
- Comparative analysis between datasets
- Data fusion from multiple sources
- Trend analysis with statistical insights
- Multi-step synthesis orchestration

**Example Usage:**
```python
# Agent processes data and generates insights
"Analyze sales performance across quarters and identify key growth drivers"
```

## Technical Implementation

### Agent Communication

The blackboard serves as the central communication hub:

```python
# Task creation and assignment
task_id = blackboard.create_task(
    assigned_to="notion_writer",
    task_type="document_creation",
    task_data={"title": "Report", "content": "..."},
    created_by="manager"
)

# Status monitoring
task_status = blackboard.get_task_status(task_id, "notion_writer")
```

### System Instructions

Runtime behavior modification through dynamic instructions:

```python
# Add temporary behavioral constraint
instruction_id = blackboard.add_system_instruction(
    agent_id="notion_writer",
    instruction_text="Include creation timestamp in all documents",
    instruction_type="behavior",
    expires_in_minutes=30
)
```

### Change Notifications

Subscribe to system state changes:

```python
def handle_change(change):
    print(f"Change detected: {change.change_type} for {change.key}")

blackboard.subscribe_to_changes(handle_change)
```

## Creating Custom Agents

### ReAct Pattern Agent (Tool-Enabled)

```python
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.tool_base import ToolBase

class CustomTool(ToolBase):
    def __init__(self):
        parameters = [
            create_parameter_schema(
                name="input_data",
                param_type=ParameterType.STRING,
                description="Data to process",
                required=True
            )
        ]
        super().__init__(
            name="custom_processor",
            description="Processes data according to custom logic",
            parameters_schema=parameters
        )
    
    def execute(self, **kwargs):
        input_data = kwargs.get("input_data")
        result = f"Processed: {input_data}"
        
        return create_success_result(
            data={"result": result},
            execution_time=0.1
        )

class CustomAgent(BaseAgent):
    def __init__(self, agent_id, blackboard, llm_client):
        super().__init__(agent_id, blackboard, llm_client, react=True)
        self.initialize()
    
    def setup_tools(self):
        custom_tool = CustomTool()
        self.register_tool(custom_tool)
```

### Acting Pattern Agent (Reasoning-Only)

```python
class ReasoningAgent(BaseAgent):
    def __init__(self, agent_id, blackboard, llm_client):
        super().__init__(agent_id, blackboard, llm_client, react=False)
        self.initialize()
    
    def setup_tools(self):
        # No tools needed for pure reasoning tasks
        pass
```

## Tool Development

### Basic Tool Structure

```python
from multi_agent_system.core.tool_base import ToolBase, create_parameter_schema

class DatabaseTool(ToolBase):
    def __init__(self, connection_string):
        self.connection = connection_string
        
        parameters = [
            create_parameter_schema(
                name="query",
                param_type=ParameterType.STRING,
                description="SQL query to execute",
                required=True
            )
        ]
        
        super().__init__(
            name="database_query",
            description="Executes database queries",
            parameters_schema=parameters
        )
    
    def execute(self, **kwargs):
        query = kwargs.get("query")
        # Implementation details
        results = self._execute_query(query)
        
        return create_success_result(
            data={"results": results},
            execution_time=0.2
        )
```

## LLM Configuration

### IBM WatsonX Setup

```python
from multi_agent_system.core.llm import LlmProvider

llm_client = LlmProvider.IBM_WATSONX.get_instance()
llm_client.__post_init__()  # Initialize with environment variables
llm_client.set_react_mode(True)  # Enable ReAct pattern support
```

### Ollama Setup

```python
llm_client = LlmProvider.OLLAMA.get_instance()
llm_client.__post_init__()
```

## Monitoring and Observability

### System Health Check

```python
health_status = system.get_status()
print(f"Registered agents: {health_status['registered_agents']}")
print(f"Active tasks: {health_status['active_tasks']}")
```

### Agent Performance Metrics

```python
stats = agent.get_stats()
print(f"Tasks completed: {stats['tasks_completed']}")
print(f"Average execution time: {stats.get('average_execution_time', 0):.2f}s")
```

### Blackboard Analytics

```python
bb_stats = blackboard.get_stats()
task_stats = blackboard.get_task_stats()

print(f"Total entries: {bb_stats['total_entries']}")
print(f"Tasks by status: {task_stats['by_status']}")
```

## Current Limitations

- RAG capabilities are implemented as separate module, not integrated with multi-agent workflow
- Limited to IBM WatsonX and Ollama LLM providers
- Notion integration requires manual token configuration
- System is in active development with APIs subject to change

## Development Status

This is an experimental framework built to explore multi-agent coordination patterns. The codebase implements core concepts but lacks production-level error handling, testing coverage, and performance optimization. Contributions and feedback are welcome as the project evolves.

## Configuration Notes

The system requires proper API credentials and may incur costs when using commercial LLM providers. Review provider pricing and terms of service before deployment. Local execution with Ollama is available for development and testing purposes.

## Architecture Decisions

The blackboard pattern was chosen to enable loose coupling between agents while maintaining system observability. The ReAct pattern implementation allows for both tool-enabled and reasoning-only agents within the same framework. These design choices prioritize extensibility over performance optimization.