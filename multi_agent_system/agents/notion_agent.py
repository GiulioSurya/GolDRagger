"""
NotionWriterAgent - Versione sync semplificata per integrazione nel sistema multi-agent.
Supporta solo create page e update page con markdown.
AGGIORNATO: Con default configurabili per parent_type e parent_id.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import requests
import json
import time

# Import dai moduli core del sistema
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.tool_base import (
    ToolBase, ToolResult, ParameterSchema, ParameterType,
    create_parameter_schema, create_success_result, create_error_result
)


# ===== CONFIGURAZIONE NOTION =====

class NotionConfig:
    """Configurazione centralizzata per Notion API"""

    def __init__(self, token: str, api_version: str = "2025-09-03"):
        if not token:
            raise ValueError("Notion token is required")

        self.token = token
        self.api_version = api_version
        self.base_url = "https://api.notion.com/v1"

    def get_headers(self) -> Dict[str, str]:
        """Headers standard per tutte le richieste Notion"""
        return {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": self.api_version,
            "Content-Type": "application/json"
        }


# ===== MARKDOWN CONVERTER =====

class MarkdownToNotionConverter:
    """Converte markdown in blocchi Notion (Pattern Strategy)"""

    @staticmethod
    def convert(markdown: str) -> List[Dict]:
        """
        Converte markdown semplice in blocchi Notion

        Args:
            markdown: Testo markdown da convertire

        Returns:
            List[Dict]: Lista di blocchi Notion
        """
        if not markdown or not markdown.strip():
            return []

        blocks = []
        lines = markdown.split('\n')

        for line in lines:
            line = line.rstrip()

            # Linee vuote - skip
            if not line:
                continue

            # Headers (H1, H2, H3)
            if line.startswith('### '):
                blocks.append(MarkdownToNotionConverter._create_heading_block(
                    line[4:], "heading_3"
                ))
            elif line.startswith('## '):
                blocks.append(MarkdownToNotionConverter._create_heading_block(
                    line[3:], "heading_2"
                ))
            elif line.startswith('# '):
                blocks.append(MarkdownToNotionConverter._create_heading_block(
                    line[2:], "heading_1"
                ))
            # Lista puntata
            elif line.startswith('- '):
                blocks.append(MarkdownToNotionConverter._create_list_block(
                    line[2:], "bulleted_list_item"
                ))
            # Lista numerata (semplice: "1. ", "2. ", etc.)
            elif len(line) > 2 and line[0].isdigit() and line[1:3] == '. ':
                blocks.append(MarkdownToNotionConverter._create_list_block(
                    line[3:], "numbered_list_item"
                ))
            # Paragrafo normale
            else:
                blocks.append(MarkdownToNotionConverter._create_paragraph_block(line))

        return blocks

    @staticmethod
    def _create_heading_block(text: str, heading_type: str) -> Dict:
        """Crea blocco heading"""
        return {
            "object": "block",
            "type": heading_type,
            heading_type: {
                "rich_text": [{"type": "text", "text": {"content": text.strip()}}]
            }
        }

    @staticmethod
    def _create_list_block(text: str, list_type: str) -> Dict:
        """Crea blocco lista"""
        return {
            "object": "block",
            "type": list_type,
            list_type: {
                "rich_text": [{"type": "text", "text": {"content": text.strip()}}]
            }
        }

    @staticmethod
    def _create_paragraph_block(text: str) -> Dict:
        """Crea blocco paragrafo"""
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text.strip()}}]
            }
        }


# ===== NOTION TOOLS (SYNC) =====

class NotionBaseTool(ToolBase):
    """Base class per tool Notion con configurazione condivisa"""

    def __init__(self, name: str, description: str, parameters_schema: List[ParameterSchema],
                 notion_config: NotionConfig):
        super().__init__(name, description, parameters_schema)
        self.notion_config = notion_config

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """
        Esegue richiesta HTTP sync a Notion API

        Args:
            method: Metodo HTTP (GET, POST, PATCH)
            endpoint: Endpoint API (senza base URL)
            data: Body della richiesta

        Returns:
            Dict: Risposta API con status e data
        """
        url = f"{self.notion_config.base_url}{endpoint}"
        headers = self.notion_config.get_headers()

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "PATCH":
                response = requests.patch(url, headers=headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Parse response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"error": "Invalid JSON response"}

            return {
                "status_code": response.status_code,
                "data": response_data,
                "success": 200 <= response.status_code < 300
            }

        except requests.exceptions.Timeout:
            return {
                "status_code": 408,
                "data": {"error": "Request timeout"},
                "success": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "status_code": 500,
                "data": {"error": "Connection error"},
                "success": False
            }
        except Exception as e:
            return {
                "status_code": 500,
                "data": {"error": f"Request failed: {str(e)}"},
                "success": False
            }


class CreatePageTool(NotionBaseTool):
    """Tool per creare pagine Notion (SYNC) con default configurabili"""

    def __init__(self, notion_config: NotionConfig,
                 default_parent_type: Optional[str] = None,
                 default_parent_id: Optional[str] = None):
        """
        Inizializza CreatePageTool con default opzionali

        Args:
            notion_config: Configurazione Notion
            default_parent_type: Tipo parent di default ("page_id" o "data_source_id")
            default_parent_id: ID parent di default
        """
        self.default_parent_type = default_parent_type
        self.default_parent_id = default_parent_id

        # Se ci sono default, parent_type e parent_id diventano opzionali
        parent_type_required = default_parent_type is None
        parent_id_required = default_parent_id is None

        # Valori consentiti - rimuovo "workspace" se ci sono default
        if default_parent_type:
            allowed_values = ["page_id", "data_source_id"]
        else:
            allowed_values = ["page_id", "data_source_id", "workspace"]

        parameters = [
            create_parameter_schema(
                "parent_type", ParameterType.STRING,
                f"Type of parent. Default: {default_parent_type}" if default_parent_type else "Type of parent: 'page_id', 'data_source_id', or 'workspace'",
                required=parent_type_required,
                default_value=default_parent_type,
                allowed_values=allowed_values
            ),
            create_parameter_schema(
                "parent_id", ParameterType.STRING,
                f"ID of the parent. Default: {default_parent_id}" if default_parent_id else "ID of the parent (page or data source). Not needed for workspace",
                required=parent_id_required,
                default_value=default_parent_id
            ),
            create_parameter_schema(
                "title", ParameterType.STRING,
                "Title of the page",
                required=True
            ),
            create_parameter_schema(
                "content", ParameterType.STRING,
                "Content of the page in markdown format",
                required=False,
                default_value=""
            ),
            create_parameter_schema(
                "icon_emoji", ParameterType.STRING,
                "Emoji icon for the page (single emoji)",
                required=False
            )
        ]

        super().__init__(
            name="create_notion_page",
            description="Create a new page in Notion with markdown content",
            parameters_schema=parameters,
            notion_config=notion_config
        )

    def execute(self, **kwargs) -> ToolResult:
        """Crea una nuova pagina Notion (SYNC) usando default quando necessario"""
        start_time = time.time()

        try:
            # Applica default se parametri non forniti
            parent_type = kwargs.get("parent_type") or self.default_parent_type
            parent_id = kwargs.get("parent_id") or self.default_parent_id

            # Valida e costruisci parent object
            parent_result = self._build_parent_object(parent_type, parent_id)

            if not parent_result["success"]:
                return create_error_result(
                    parent_result["error"],
                    time.time() - start_time
                )

            # Costruisci properties con titolo
            title = kwargs.get("title", "Untitled")
            properties = {
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": title}
                        }
                    ]
                }
            }

            # Costruisci body richiesta
            body = {
                "parent": parent_result["parent"],
                "properties": properties
            }

            # Aggiungi icon se specificato
            icon_emoji = kwargs.get("icon_emoji")
            if icon_emoji and icon_emoji.strip():
                body["icon"] = {
                    "type": "emoji",
                    "emoji": icon_emoji.strip()
                }

            # Converti content markdown in blocchi Notion
            content = kwargs.get("content", "")
            if content and content.strip():
                body["children"] = MarkdownToNotionConverter.convert(content)

            # Esegui richiesta API
            api_response = self._make_request("POST", "/pages", body)

            if api_response["success"]:
                data = api_response["data"]
                return create_success_result(
                    {
                        "page_id": data.get("id"),
                        "url": data.get("url"),
                        "title": title,
                        "created_time": data.get("created_time"),
                        "parent_type": parent_type,
                        "parent_id": parent_id,
                        "used_defaults": {
                            "parent_type": parent_type == self.default_parent_type,
                            "parent_id": parent_id == self.default_parent_id
                        },
                        "has_content": bool(content and content.strip()),
                        "has_icon": bool(icon_emoji)
                    },
                    time.time() - start_time
                )
            else:
                error_msg = api_response["data"].get("message",
                                                     f"API error: {api_response['status_code']}")
                return create_error_result(
                    f"Failed to create page: {error_msg}",
                    time.time() - start_time
                )

        except Exception as e:
            return create_error_result(
                f"Create page error: {str(e)}",
                time.time() - start_time
            )

    def _build_parent_object(self, parent_type: str, parent_id: str) -> Dict:
        """Costruisce parent object per richiesta API"""
        if not parent_type:
            return {"success": False, "error": "parent_type is required"}

        # Se parent_type non è "workspace", parent_id è richiesto
        if parent_type != "workspace" and not parent_id:
            return {"success": False, "error": f"parent_id is required for parent_type '{parent_type}'"}

        if parent_type == "page_id":
            return {
                "success": True,
                "parent": {"type": "page_id", "page_id": parent_id}
            }
        elif parent_type == "data_source_id":
            return {
                "success": True,
                "parent": {"type": "data_source_id", "data_source_id": parent_id}
            }
        elif parent_type == "workspace":
            return {
                "success": True,
                "parent": {"type": "workspace", "workspace": True}
            }
        else:
            return {"success": False,
                    "error": f"Invalid parent_type: {parent_type}. Use 'page_id', 'data_source_id', or 'workspace'"}


class UpdatePageTool(NotionBaseTool):
    """Tool per aggiornare pagine Notion esistenti (SYNC)"""

    def __init__(self, notion_config: NotionConfig):
        parameters = [
            create_parameter_schema(
                "page_id", ParameterType.STRING,
                "ID of the page to update",
                required=True
            ),
            create_parameter_schema(
                "title", ParameterType.STRING,
                "New title for the page",
                required=False
            ),
            create_parameter_schema(
                "archived", ParameterType.BOOLEAN,
                "Archive or unarchive the page",
                required=False
            ),
            create_parameter_schema(
                "icon_emoji", ParameterType.STRING,
                "Update emoji icon",
                required=False
            )
        ]

        super().__init__(
            name="update_notion_page",
            description="Update properties of an existing Notion page",
            parameters_schema=parameters,
            notion_config=notion_config
        )

    def execute(self, **kwargs) -> ToolResult:
        """Aggiorna una pagina Notion esistente (SYNC)"""
        start_time = time.time()

        try:
            page_id = kwargs.get("page_id")
            if not page_id:
                return create_error_result(
                    "page_id is required",
                    time.time() - start_time
                )

            # Costruisci body della richiesta
            body = {}
            updates_made = []

            # Aggiorna titolo se specificato
            title = kwargs.get("title")
            if title:
                body["properties"] = {
                    "title": {
                        "title": [
                            {
                                "type": "text",
                                "text": {"content": title.strip()}
                            }
                        ]
                    }
                }
                updates_made.append("title")

            # Aggiorna archived se specificato
            archived = kwargs.get("archived")
            if archived is not None:
                body["archived"] = bool(archived)
                updates_made.append("archived_status")

            # Aggiorna icon se specificato
            icon_emoji = kwargs.get("icon_emoji")
            if icon_emoji:
                if icon_emoji.strip().lower() == "none" or icon_emoji.strip() == "":
                    # Rimuovi icon
                    body["icon"] = None
                    updates_made.append("icon_removed")
                else:
                    # Imposta nuovo icon
                    body["icon"] = {
                        "type": "emoji",
                        "emoji": icon_emoji.strip()
                    }
                    updates_made.append("icon_updated")

            # Verifica che ci sia almeno un update
            if not body:
                return create_error_result(
                    "At least one update field is required (title, archived, or icon_emoji)",
                    time.time() - start_time
                )

            # Esegui richiesta API
            api_response = self._make_request("PATCH", f"/pages/{page_id}", body)

            if api_response["success"]:
                data = api_response["data"]
                return create_success_result(
                    {
                        "page_id": data.get("id"),
                        "url": data.get("url"),
                        "last_edited_time": data.get("last_edited_time"),
                        "archived": data.get("archived", False),
                        "updates_applied": updates_made
                    },
                    time.time() - start_time
                )
            else:
                error_msg = api_response["data"].get("message",
                                                     f"API error: {api_response['status_code']}")
                return create_error_result(
                    f"Failed to update page: {error_msg}",
                    time.time() - start_time
                )

        except Exception as e:
            return create_error_result(
                f"Update page error: {str(e)}",
                time.time() - start_time
            )


# ===== NOTION WRITER AGENT =====

class NotionWriterAgent(BaseAgent):
    """
    Agent specializzato per scrittura su Notion.

    Eredita ReAct pattern da BaseAgent automaticamente.
    Supporta solo create page e update page con markdown.
    Completamente sincrono.
    AGGIORNATO: Con default configurabili per parent_type e parent_id.
    """

    def __init__(self, agent_id: str, blackboard: BlackBoard,
                 llm_client, notion_token: str,
                 default_parent_type: Optional[str] = None,
                 default_parent_id: Optional[str] = None):
        """
        Inizializza NotionWriterAgent.

        Args:
            agent_id: ID univoco dell'agent
            blackboard: Blackboard condivisa
            llm_client: Client LLM per ReAct
            notion_token: Token di autenticazione Notion
            default_parent_type: Tipo parent di default ("page_id" o "data_source_id")
            default_parent_id: ID parent di default
        """
        super().__init__(agent_id, blackboard, llm_client)

        # Configurazione Notion
        try:
            self.notion_config = NotionConfig(notion_token)
        except ValueError as e:
            raise ValueError(f"Invalid Notion configuration: {str(e)}")

        # Salva default per i tool
        self.default_parent_type = default_parent_type
        self.default_parent_id = default_parent_id

        # Inizializza agent (registra tools automaticamente)
        success = self.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize NotionWriterAgent {agent_id}")

    def setup_tools(self):
        """
        Implementazione richiesta da BaseAgent.
        Registra solo i 2 tool essenziali per Notion con default configurati.
        """
        # Crea e registra CreatePageTool con default
        create_tool = CreatePageTool(
            self.notion_config,
            default_parent_type=self.default_parent_type,
            default_parent_id=self.default_parent_id
        )
        self.register_tool(create_tool)

        # UpdatePageTool non ha bisogno di default
        update_tool = UpdatePageTool(self.notion_config)
        self.register_tool(update_tool)

        default_info = ""
        if self.default_parent_type and self.default_parent_id:
            default_info = f" with defaults (type: {self.default_parent_type}, id: {self.default_parent_id[:8]}...)"

        print(f"[{self.agent_id}] Registered {len(self._tools)} Notion tools: "
              f"{', '.join(self.get_available_tools())}{default_info}")

    def _generate_agent_description(self) -> str:
        """
        Override per fornire descrizione specifica dell'agent.

        Returns:
            str: Descrizione per il Manager
        """
        base_desc = (
            f"NotionWriterAgent - Creates and updates Notion pages with markdown support. "
            f"Can create pages in workspace or under parent pages/databases. "
            f"Supports emoji icons and page archiving."
        )

        # Aggiungi info sui default se configurati
        if self.default_parent_type and self.default_parent_id:
            base_desc += f" Configured with default parent ({self.default_parent_type}: {self.default_parent_id[:8]}...)."

        base_desc += f" Tools: {', '.join(self.get_available_tools())}"

        return base_desc

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Override per aggiungere info specifiche di Notion.

        Returns:
            Dict: Capabilities estese con info Notion
        """
        base_capabilities = super().get_capabilities()

        # Aggiungi info specifiche Notion
        base_capabilities["notion_features"] = {
            "api_version": self.notion_config.api_version,
            "sync_mode": True,
            "markdown_support": True,
            "supported_operations": ["create_page", "update_page"],
            "supported_parents": ["workspace", "page_id", "data_source_id"],
            "content_features": ["markdown_conversion", "emoji_icons", "page_archiving"],
            "default_parent_configured": bool(self.default_parent_type and self.default_parent_id),
            "default_parent_type": self.default_parent_type,
            "default_parent_id": self.default_parent_id
        }

        return base_capabilities


# ===== FACTORY PER CREAZIONE AGENT =====

class NotionAgentFactory:
    """Factory per creare NotionWriterAgent con validazione (Pattern Factory)"""

    @staticmethod
    def create_agent(agent_id: str, blackboard: BlackBoard,
                     llm_client, notion_token: str,
                     default_parent_type: Optional[str] = None,
                     default_parent_id: Optional[str] = None) -> NotionWriterAgent:
        """
        Crea NotionWriterAgent con validazione completa

        Args:
            agent_id: ID dell'agent
            blackboard: Blackboard condivisa
            llm_client: Client LLM
            notion_token: Token Notion
            default_parent_type: Tipo parent di default ("page_id" o "data_source_id")
            default_parent_id: ID parent di default

        Returns:
            NotionWriterAgent: Agent configurato e pronto

        Raises:
            ValueError: Se configurazione non valida
            RuntimeError: Se inizializzazione fallisce
        """
        # Validazioni
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id cannot be empty")

        if not blackboard:
            raise ValueError("blackboard is required")

        if not llm_client:
            raise ValueError("llm_client is required")

        if not notion_token or not notion_token.strip():
            raise ValueError("notion_token cannot be empty")

        # Validazione default (se uno è specificato, entrambi devono esserlo)
        if (default_parent_type and not default_parent_id) or (default_parent_id and not default_parent_type):
            raise ValueError("Both default_parent_type and default_parent_id must be specified together")

        # Validazione tipo parent
        if default_parent_type and default_parent_type not in ["page_id", "data_source_id"]:
            raise ValueError("default_parent_type must be 'page_id' or 'data_source_id'")

        # Crea agent
        try:
            agent = NotionWriterAgent(
                agent_id=agent_id.strip(),
                blackboard=blackboard,
                llm_client=llm_client,
                notion_token=notion_token.strip(),
                default_parent_type=default_parent_type,
                default_parent_id=default_parent_id
            )

            default_info = ""
            if default_parent_type and default_parent_id:
                default_info = f" with defaults ({default_parent_type}: {default_parent_id[:8]}...)"

            print(f"[NotionAgentFactory] Created agent '{agent_id}'{default_info} successfully")
            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to create NotionWriterAgent: {str(e)}")


if __name__ == "__main__":
    print("=== NOTION AGENT TEST ===")

    # Setup con LLM IBM WatsonX
    from multi_agent_system.core.llm import LLM
    import os
    from dotenv import load_dotenv
    load_dotenv()

    blackboard = BlackBoard()

    try:
        # Inizializza LLM client
        llm_client = LLM()
        llm_client.__post_init__()  # Forza inizializzazione
        llm_client.set_react_mode(True)  # Abilita ReAct mode

        # Crea agent con default configurati
        agent = NotionAgentFactory.create_agent(
            agent_id="notion_test_agent",
            blackboard=blackboard,
            llm_client=llm_client,
            notion_token=os.getenv("NOTION_TOKEN", ""),
            default_parent_type="page_id",
            default_parent_id=os.getenv("NOTION_PARENT_ID", "")
        )

        print("Agent initialized with IBM WatsonX LLM")

        # Simula Manager che crea task
        task_id = blackboard.create_task(
            assigned_to="notion_test_agent",
            task_type="create_meeting_notes",
            task_data={
                "request": "Create a Notion page titled 'Weekly Team Meeting' with sections for agenda, notes, and action items"
            },
            created_by="manager"
        )

        print(f"Task created: {task_id[:8]}...")

        # Agent automaticamente processera' il task via ReAct loop
        # Attendi processing
        import time

        max_wait = 30  # 30 secondi timeout
        start_time = time.time()

        while time.time() - start_time < max_wait:
            task_status = blackboard.get_task_status(task_id, "notion_test_agent")
            if task_status and task_status.get("status") in ["completed", "failed"]:
                break
            time.sleep(1)

        # Controlla risultato finale
        task_result = blackboard.get_task_status(task_id, "notion_test_agent")
        if task_result:
            status = task_result.get("status")
            if status == "completed":
                result_data = task_result.get("result", {})
                print(f"Task completed: {result_data.get('answer', 'Success')}")
            else:
                print(f"Task failed: {task_result.get('result', {}).get('error', 'Unknown error')}")
        else:
            print("Task timeout or not found")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Check Notion token, parent_id configuration, and IBM WatsonX credentials")