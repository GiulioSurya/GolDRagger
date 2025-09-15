"""
SyntheticAgent - Acting mode agent specialized in descriptive synthesis
Single-shot reasoning for text synthesis and summarization without tools
"""

from typing import Dict, Any
from datetime import datetime, timezone

# Import dai moduli core del sistema
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.black_board import BlackBoard


class SyntheticAgent(BaseAgent):
    """
    Agent specializzato nella sintesi descrittiva usando Acting pattern.
    Single-shot reasoning per sintesi di testi, summaries e reportistica.
    Non usa tool - solo reasoning diretto con LLM.
    """

    def __init__(self, agent_id: str, blackboard: BlackBoard, llm_client):
        """
        Inizializza SyntheticAgent in Acting mode (react=False).

        Args:
            agent_id: ID univoco dell'agent
            blackboard: Blackboard condivisa
            llm_client: Client LLM per direct reasoning
        """
        # Inizializza in Acting mode (react=False)
        super().__init__(agent_id, blackboard, llm_client, react=False)

        # Statistiche specifiche per sintesi
        self._synthesis_stats = {
            'total_syntheses': 0,
            'synthesis_types': {},
            'average_content_length': 0,
            'last_synthesis_time': None
        }

        success = self.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize SyntheticAgent {agent_id}")

    def setup_tools(self):
        """
        Non registra tool - Acting mode usa solo reasoning diretto.
        Metodo richiesto da BaseAgent ma vuoto per Acting pattern.
        """
        print(f"[{self.agent_id}] Acting mode - no tools registered, direct reasoning only")

    def _generate_agent_description(self) -> str:
        """Override per descrizione specifica del SyntheticAgent"""
        return (
            f"SyntheticAgent - Specialized synthesis specialist operating in Acting mode "
            f"for direct text processing and descriptive analysis. Processes entire "
            f"queries and content to generate executive summaries, detailed reports, "
            f"key insights extraction, and strategic recommendations. ."
        )

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche specifiche di sintesi"""
        base_stats = self.get_stats()
        return {
            **base_stats,
            'synthesis_stats': self._synthesis_stats,
            'mode': 'acting',
            'specializations': [
                "Executive summary generation",
                "Key insights extraction",
                "Strategic recommendations",
                "Detailed report creation",
                "Content synthesis and analysis"
            ]
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Capabilities specifiche per Acting mode"""
        base_capabilities = super().get_capabilities()

        base_capabilities["synthesis_features"] = {
            "mode": "acting",
            "synthesis_types": [
                "Executive summaries",
                "Key findings extraction",
                "Strategic recommendations",
                "Detailed analysis reports",
                "Content consolidation",
                "Insight generation"
            ],
            "input_processing": [
                "Full query/content analysis",
                "Multi-document synthesis",
                "Text consolidation",
                "Pattern identification",
                "Priority ranking"
            ],
            "output_formats": [
                "Structured markdown",
                "Executive briefings",
                "Bullet-point summaries",
                "Narrative reports",
                "Action item lists"
            ],
            "strengths": [
                "Single-shot comprehensive analysis",
                "No tool dependency",
                "Fast direct reasoning",
                "Holistic content processing"
            ]
        }

        return base_capabilities

    def _save_observation_to_blackboard(self, task_id: str, step: int, observation: str):
        """Override per tracking sintesi in Acting mode"""
        super()._save_observation_to_blackboard(task_id, step, observation)

        # Update synthesis stats
        if "synthesis" in observation.lower() or "summary" in observation.lower():
            self._synthesis_stats['total_syntheses'] += 1
            self._synthesis_stats['last_synthesis_time'] = datetime.now(timezone.utc)

            # Cerca di estrarre tipo di sintesi dall'observation
            if "executive" in observation.lower():
                synthesis_type = "executive_summary"
            elif "report" in observation.lower():
                synthesis_type = "detailed_report"
            elif "insights" in observation.lower():
                synthesis_type = "key_insights"
            else:
                synthesis_type = "general_synthesis"

            self._synthesis_stats['synthesis_types'][synthesis_type] = (
                    self._synthesis_stats['synthesis_types'].get(synthesis_type, 0) + 1
            )


# ===== FACTORY =====

class SyntheticAgentFactory:
    """Factory per creare SyntheticAgent in Acting mode"""

    @staticmethod
    def create_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> SyntheticAgent:
        """Crea SyntheticAgent standard in Acting mode"""
        if not all([agent_id, blackboard, llm_client]):
            raise ValueError("agent_id, blackboard, and llm_client are required")

        try:
            agent = SyntheticAgent(
                agent_id=agent_id.strip(),
                blackboard=blackboard,
                llm_client=llm_client
            )

            print(f"[SyntheticAgentFactory] Created SyntheticAgent '{agent_id}' in Acting mode")
            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to create SyntheticAgent: {str(e)}")

    @staticmethod
    def create_executive_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> SyntheticAgent:
        """Crea agent con focus executive"""
        agent = SyntheticAgentFactory.create_agent(agent_id, blackboard, llm_client)

        # System instructions per focus executive
        agent.add_system_instruction(
            instruction="Focus on executive-level insights, business impact, and strategic implications. Always provide actionable recommendations.",
            instruction_type="behavior",
            priority_level=1
        )

        agent.add_system_instruction(
            instruction="Structure responses with clear executive summary, key points, and next steps. Use business language appropriate for decision makers.",
            instruction_type="constraint",
            priority_level=2
        )

        return agent

    @staticmethod
    def create_report_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> SyntheticAgent:
        """Crea agent specializzato in reportistica dettagliata"""
        agent = SyntheticAgentFactory.create_agent(agent_id, blackboard, llm_client)

        agent.add_system_instruction(
            instruction="Generate comprehensive, detailed reports with thorough analysis. Include background context, detailed findings, and comprehensive recommendations.",
            instruction_type="behavior",
            priority_level=1
        )

        agent.add_system_instruction(
            instruction="Use structured formatting with clear sections, subsections, and logical flow. Support all claims with evidence from the provided content.",
            instruction_type="constraint",
            priority_level=2
        )

        return agent


# ===== TEST =====

if __name__ == "__main__":
    print("=== SYNTHETIC AGENT - ACTING MODE TEST ===")

    from multi_agent_system.core.black_board import BlackBoard

    blackboard = BlackBoard()


    # Mock LLM per test Acting mode
    class MockLLM:
        def invoke(self, system, user):
            return """Based on the provided content, here is my synthesis:

Executive Summary:
The data shows significant growth trends with strong positive indicators for strategic decision making.

Key Findings:
- Primary metrics demonstrate upward trajectory
- Market conditions favor continued expansion
- Risk factors remain manageable within acceptable parameters

Strategic Recommendations:
- Capitalize on current momentum through increased investment
- Monitor emerging market signals for optimization opportunities
- Implement risk mitigation protocols for identified concerns

This synthesis provides a comprehensive foundation for executive decision making."""

        def set_react_mode(self, mode):
            pass


    try:
        # Test standard agent
        agent = SyntheticAgentFactory.create_agent(
            agent_id="synthetic_acting",
            blackboard=blackboard,
            llm_client=MockLLM()
        )

        print("SyntheticAgent (Acting mode) created successfully")
        print(f"Mode: {agent.get_capabilities()['mode']}")
        print(f"Tools: {agent.get_available_tools()}")  # Should be empty

        # Test capabilities
        capabilities = agent.get_capabilities()
        print(f"Synthesis types: {capabilities['synthesis_features']['synthesis_types']}")

        # Test executive agent
        exec_agent = SyntheticAgentFactory.create_executive_agent(
            agent_id="synthetic_executive",
            blackboard=blackboard,
            llm_client=MockLLM()
        )

        print(f"Executive agent created with system instructions")

    except Exception as e:
        print(f"Test failed: {str(e)}")