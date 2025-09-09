"""
SyntheticAgent - Agent specializzato nella sintesi di informazioni
Modalità ReAct (react=True) con tool modulari per ogni tipo di sintesi
"""

from typing import Dict, Any, List
from enum import Enum
from datetime import datetime, timezone

# Import dai moduli core del sistema
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.tool_base import (
    ToolBase, ToolResult, ParameterType,
    create_parameter_schema, create_success_result, create_error_result
)

import time


class SynthesisComplexity(Enum):
    """Livelli di complessità della sintesi"""
    SIMPLE = "simple"       # Sintesi basic, 2-3 paragrafi
    MEDIUM = "medium"       # Sintesi articolata con sezioni
    COMPLEX = "complex"     # Sintesi completa con analisi approfondita


class OutputFormat(Enum):
    """Formati di output supportati"""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    STRUCTURED_JSON = "json"
    BULLET_POINTS = "bullet_points"


# ===== SYNTHESIS TOOLS =====

class ExecutiveSummaryTool(ToolBase):
    """Tool per creare executive summary concisi e orientati al business"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "topic", ParameterType.STRING,
                "Main topic or title for the executive summary",
                required=True
            ),
            create_parameter_schema(
                "sources", ParameterType.ARRAY,
                "List of data sources or content to summarize",
                required=True
            ),
            create_parameter_schema(
                "findings", ParameterType.ARRAY,
                "Key findings or insights to highlight",
                required=False,
                default_value=[]
            ),
            create_parameter_schema(
                "recommendations", ParameterType.ARRAY,
                "Recommendations or action items",
                required=False,
                default_value=[]
            ),
            create_parameter_schema(
                "complexity", ParameterType.STRING,
                "Complexity level for the summary",
                required=False,
                default_value="medium",
                allowed_values=["simple", "medium", "complex"]
            ),
            create_parameter_schema(
                "output_format", ParameterType.STRING,
                "Output format for the summary",
                required=False,
                default_value="markdown",
                allowed_values=["plain_text", "markdown", "json", "bullet_points"]
            )
        ]

        super().__init__(
            name="executive_summary",
            description="Creates executive summary focused on business impact and strategic implications",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["synthesis", "executive", "business", "summary"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Crea executive summary strutturato"""

        start_time = time.time()

        try:
            # Estrae parametri
            topic = kwargs.get("topic")
            sources = kwargs.get("sources", [])
            findings = kwargs.get("findings", [])
            recommendations = kwargs.get("recommendations", [])
            complexity = kwargs.get("complexity", "medium")
            output_format = kwargs.get("output_format", "markdown")

            # Valida sources
            if not sources or len(sources) == 0:
                return create_error_result(
                    "At least one source is required for executive summary",
                    time.time() - start_time
                )

            # Costruisce prompt basato su complexity
            prompt = self._build_executive_prompt(
                topic, sources, findings, recommendations, complexity, output_format
            )

            execution_time = time.time() - start_time

            return create_success_result(
                {
                    "summary_type": "executive_summary",
                    "prompt": prompt,
                    "configuration": {
                        "topic": topic,
                        "sources_count": len(sources),
                        "complexity": complexity,
                        "output_format": output_format,
                        "has_findings": len(findings) > 0,
                        "has_recommendations": len(recommendations) > 0
                    },
                    "instructions": f"Use this prompt to generate an executive summary in {output_format} format"
                },
                execution_time
            )

        except Exception as e:
            return create_error_result(
                f"Executive summary tool failed: {str(e)}",
                time.time() - start_time
            )

    def _build_executive_prompt(self, topic: str, sources: List, findings: List,
                               recommendations: List, complexity: str, output_format: str) -> str:
        """Costruisce prompt specializzato per executive summary"""

        # Determina struttura basata su complexity
        if complexity == "simple":
            sections = "Overview, Key Points, Conclusion"
            depth = "concise 2-3 paragraph summary"
        elif complexity == "medium":
            sections = "Executive Overview, Key Findings, Impact Analysis, Recommendations"
            depth = "structured analysis with clear sections"
        else:  # complex
            sections = "Executive Overview, Detailed Findings, Impact Analysis, Strategic Recommendations, Next Steps, Risk Assessment"
            depth = "comprehensive executive analysis"

        # Format instructions
        format_instructions = self._get_format_instructions(output_format)

        # Sources formatting
        sources_text = self._format_sources(sources)

        # Findings formatting
        findings_text = ""
        if findings:
            findings_text = f"\n\nKey Findings to highlight:\n" + "\n".join([f"- {finding}" for finding in findings])

        # Recommendations formatting
        recommendations_text = ""
        if recommendations:
            recommendations_text = f"\n\nRecommendations to include:\n" + "\n".join([f"- {rec}" for rec in recommendations])

        return f"""Create a {depth} for: {topic}

Required sections: {sections}

Guidelines:
- Focus on business impact and strategic implications
- Use clear, executive-level language
- Prioritize actionable insights over technical details
- Include quantitative metrics when available
- Structure for quick consumption by decision makers

{format_instructions}

Data sources:
{sources_text}

{findings_text}

{recommendations_text}

Deliver a synthesis that enables informed decision-making."""

    def _get_format_instructions(self, output_format: str) -> str:
        """Istruzioni specifiche per formato di output"""
        format_map = {
            "markdown": "Format as clean markdown with headers (##, ###) and bullet points where appropriate.",
            "json": "Structure as JSON with sections as keys and content as values.",
            "bullet_points": "Use bullet points for all content, organized by section with clear headers.",
            "plain_text": "Use clear paragraph structure with section headers in caps."
        }
        return format_map.get(output_format, format_map["markdown"])

    def _format_sources(self, sources: List) -> str:
        """Formatta sources per inclusione nel prompt"""
        if not sources:
            return "No specific sources provided."

        formatted = []
        for i, source in enumerate(sources[:8], 1):  # Limita a 8 per brevità
            if isinstance(source, dict):
                content = source.get('content', str(source))
                title = source.get('title', f'Source {i}')
                formatted.append(f"{i}. {title}: {content}")
            else:
                formatted.append(f"{i}. {str(source)}")

        return "\n".join(formatted)


class ComparativeAnalysisTool(ToolBase):
    """Tool per analisi comparative tra multiple opzioni o dataset"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "items", ParameterType.ARRAY,
                "Items or options to compare (minimum 2 required)",
                required=True
            ),
            create_parameter_schema(
                "context", ParameterType.STRING,
                "Context or purpose of the comparison",
                required=True
            ),
            create_parameter_schema(
                "criteria", ParameterType.ARRAY,
                "Criteria to use for comparison",
                required=False,
                default_value=["performance", "cost", "feasibility"]
            ),
            create_parameter_schema(
                "complexity", ParameterType.STRING,
                "Analysis complexity level",
                required=False,
                default_value="medium",
                allowed_values=["simple", "medium", "complex"]
            ),
            create_parameter_schema(
                "output_format", ParameterType.STRING,
                "Output format for analysis",
                required=False,
                default_value="markdown",
                allowed_values=["plain_text", "markdown", "json", "bullet_points"]
            )
        ]

        super().__init__(
            name="comparative_analysis",
            description="Performs systematic comparative analysis between multiple options or datasets",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["synthesis", "comparison", "analysis", "decision_support"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Crea analisi comparativa strutturata"""
        start_time = time.time()

        try:
            items = kwargs.get("items", [])
            context = kwargs.get("context")
            criteria = kwargs.get("criteria", ["performance", "cost", "feasibility"])
            complexity = kwargs.get("complexity", "medium")
            output_format = kwargs.get("output_format", "markdown")

            # Validazioni
            if len(items) < 2:
                return create_error_result(
                    "Comparative analysis requires at least 2 items to compare",
                    time.time() - start_time
                )

            if not context:
                return create_error_result(
                    "Context is required for comparative analysis",
                    time.time() - start_time
                )

            # Costruisce prompt
            prompt = self._build_comparison_prompt(
                items, context, criteria, complexity, output_format
            )

            execution_time = time.time() - start_time

            return create_success_result(
                {
                    "analysis_type": "comparative_analysis",
                    "prompt": prompt,
                    "configuration": {
                        "context": context,
                        "items_count": len(items),
                        "criteria": criteria,
                        "complexity": complexity,
                        "output_format": output_format
                    },
                    "comparison_matrix": {
                        "items": items,
                        "criteria": criteria
                    },
                    "instructions": f"Use this prompt to generate comparative analysis in {output_format} format"
                },
                execution_time
            )

        except Exception as e:
            return create_error_result(
                f"Comparative analysis tool failed: {str(e)}",
                time.time() - start_time
            )

    def _build_comparison_prompt(self, items: List, context: str, criteria: List[str],
                               complexity: str, output_format: str) -> str:
        """Costruisce prompt per analisi comparativa"""

        # Depth basata su complexity
        depth_map = {
            "simple": "high-level comparison with key differences",
            "medium": "detailed comparison with pros/cons and scoring",
            "complex": "comprehensive analysis with weighted criteria, risk assessment, and strategic implications"
        }
        analysis_depth = depth_map.get(complexity, depth_map["medium"])

        # Format instructions
        format_instructions = self._get_format_instructions(output_format)

        # Items formatting
        items_desc = "\n".join([f"- {item}" for item in items])

        # Criteria formatting
        criteria_desc = ", ".join(criteria)

        return f"""Perform {analysis_depth} for: {context}

Items to Compare:
{items_desc}

Comparison Criteria: {criteria_desc}

Analysis Requirements:
- Create systematic comparison across all criteria
- Identify strengths and weaknesses of each option
- Provide clear recommendations with reasoning
- Include risk considerations where applicable
- Highlight key differentiators and trade-offs
- Use objective, data-driven approach

{format_instructions}

Deliver an objective comparative analysis that supports decision-making."""

    def _get_format_instructions(self, output_format: str) -> str:
        """Istruzioni formato per comparative analysis"""
        format_map = {
            "markdown": "Format with comparison table using markdown syntax, followed by detailed analysis sections.",
            "json": "Structure as JSON with comparison_matrix and detailed_analysis sections.",
            "bullet_points": "Use comparison table format followed by bullet-point analysis for each item.",
            "plain_text": "Create comparison table followed by detailed narrative analysis."
        }
        return format_map.get(output_format, format_map["markdown"])


class DataFusionTool(ToolBase):
    """Tool per fusione di dati da fonti multiple eterogenee"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "datasets", ParameterType.ARRAY,
                "Multiple datasets to fuse (minimum 2 required)",
                required=True
            ),
            create_parameter_schema(
                "fusion_goal", ParameterType.STRING,
                "Goal or purpose of data fusion",
                required=True
            ),
            create_parameter_schema(
                "common_fields", ParameterType.ARRAY,
                "Common fields or keys for data alignment",
                required=False,
                default_value=[]
            ),
            create_parameter_schema(
                "complexity", ParameterType.STRING,
                "Fusion complexity level",
                required=False,
                default_value="medium",
                allowed_values=["simple", "medium", "complex"]
            ),
            create_parameter_schema(
                "output_format", ParameterType.STRING,
                "Output format for fused data",
                required=False,
                default_value="json",
                allowed_values=["plain_text", "markdown", "json", "bullet_points"]
            )
        ]

        super().__init__(
            name="data_fusion",
            description="Fuses data from multiple heterogeneous sources into unified view",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["synthesis", "data_fusion", "integration", "consolidation"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Fonde dati da multiple fonti in vista unificata"""
        start_time = time.time()

        try:
            datasets = kwargs.get("datasets", [])
            fusion_goal = kwargs.get("fusion_goal")
            common_fields = kwargs.get("common_fields", [])
            complexity = kwargs.get("complexity", "medium")
            output_format = kwargs.get("output_format", "json")

            # Validazioni
            if len(datasets) < 2:
                return create_error_result(
                    "Data fusion requires at least 2 datasets",
                    time.time() - start_time
                )

            if not fusion_goal:
                return create_error_result(
                    "Fusion goal is required for data fusion",
                    time.time() - start_time
                )

            # Costruisce prompt
            prompt = self._build_fusion_prompt(
                datasets, fusion_goal, common_fields, complexity, output_format
            )

            execution_time = time.time() - start_time

            return create_success_result(
                {
                    "fusion_type": "data_fusion",
                    "prompt": prompt,
                    "configuration": {
                        "fusion_goal": fusion_goal,
                        "datasets_count": len(datasets),
                        "common_fields": common_fields,
                        "complexity": complexity,
                        "output_format": output_format
                    },
                    "fusion_metadata": {
                        "datasets_summary": self._summarize_datasets(datasets),
                        "fusion_approach": self._get_fusion_approach(complexity)
                    },
                    "instructions": f"Use this prompt to perform data fusion in {output_format} format"
                },
                execution_time
            )

        except Exception as e:
            return create_error_result(
                f"Data fusion tool failed: {str(e)}",
                time.time() - start_time
            )

    def _build_fusion_prompt(self, datasets: List, fusion_goal: str, common_fields: List[str],
                           complexity: str, output_format: str) -> str:
        """Costruisce prompt per data fusion"""

        # Approach basato su complexity
        fusion_approach = self._get_fusion_approach(complexity)

        # Format instructions
        format_instructions = self._get_format_instructions(output_format)

        # Datasets summary
        datasets_summary = self._summarize_datasets(datasets)

        # Common fields
        fields_desc = ", ".join(common_fields) if common_fields else "identify automatically"

        return f"""Perform {fusion_approach} to achieve: {fusion_goal}

Available Datasets:
{datasets_summary}

Common/Key Fields: {fields_desc}

Fusion Requirements:
- Identify and resolve data conflicts systematically
- Maintain data lineage and source attribution
- Create unified data model with consolidated schema
- Highlight data quality and completeness metrics
- Generate integrated insights not visible in individual datasets
- Flag inconsistencies, gaps, and quality issues
- Provide confidence scores for fused data where appropriate

{format_instructions}

Deliver a comprehensive data fusion that maximizes information value while maintaining transparency about source data quality and integration decisions."""

    def _get_fusion_approach(self, complexity: str) -> str:
        """Determina approccio di fusione basato su complexity"""
        approach_map = {
            "simple": "basic aggregation with overlap identification",
            "medium": "structured integration with conflict resolution and quality assessment",
            "complex": "advanced fusion with quality scoring, uncertainty analysis, and confidence intervals"
        }
        return approach_map.get(complexity, approach_map["medium"])

    def _summarize_datasets(self, datasets: List) -> str:
        """Crea summary dei datasets per il prompt"""
        summaries = []
        for i, dataset in enumerate(datasets, 1):
            if isinstance(dataset, dict):
                name = dataset.get('name', f'Dataset {i}')
                description = dataset.get('description', 'No description available')
                records = dataset.get('records', 'Unknown record count')
                schema = dataset.get('schema', 'Unknown schema')
                summaries.append(f"{i}. {name}: {description} | Records: {records} | Schema: {schema}")
            else:
                # Handle string or other types
                summary = str(dataset)[:200] + "..." if len(str(dataset)) > 200 else str(dataset)
                summaries.append(f"{i}. Dataset {i}: {summary}")

        return "\n".join(summaries)

    def _get_format_instructions(self, output_format: str) -> str:
        """Istruzioni formato per data fusion"""
        format_map = {
            "json": "Structure as JSON with fused_data, quality_metrics, source_mapping, and integration_notes sections.",
            "markdown": "Format with fused dataset tables, quality assessment, and detailed integration analysis.",
            "bullet_points": "Present fusion results as organized bullet points with quality indicators.",
            "plain_text": "Present unified dataset with quality assessment and source tracking in narrative form."
        }
        return format_map.get(output_format, format_map["json"])


class TrendAnalysisTool(ToolBase):
    """Tool per analisi di trend e pattern temporali"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "time_series", ParameterType.ARRAY,
                "Time series data for trend analysis",
                required=True
            ),
            create_parameter_schema(
                "analysis_period", ParameterType.STRING,
                "Period description for the analysis",
                required=False,
                default_value="recent trends"
            ),
            create_parameter_schema(
                "metrics", ParameterType.ARRAY,
                "Specific metrics to analyze in trends",
                required=False,
                default_value=["growth", "patterns", "anomalies"]
            ),
            create_parameter_schema(
                "complexity", ParameterType.STRING,
                "Analysis complexity level",
                required=False,
                default_value="medium",
                allowed_values=["simple", "medium", "complex"]
            ),
            create_parameter_schema(
                "output_format", ParameterType.STRING,
                "Output format for analysis",
                required=False,
                default_value="markdown",
                allowed_values=["plain_text", "markdown", "json", "bullet_points"]
            )
        ]

        super().__init__(
            name="trend_analysis",
            description="Analyzes trends and temporal patterns in time series data",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["synthesis", "trends", "temporal", "patterns", "forecasting"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Analizza trend e pattern nei dati temporali"""
        start_time = time.time()

        try:
            time_series = kwargs.get("time_series", [])
            analysis_period = kwargs.get("analysis_period", "recent trends")
            metrics = kwargs.get("metrics", ["growth", "patterns", "anomalies"])
            complexity = kwargs.get("complexity", "medium")
            output_format = kwargs.get("output_format", "markdown")

            # Validazioni
            if not time_series or len(time_series) == 0:
                return create_error_result(
                    "Time series data is required for trend analysis",
                    time.time() - start_time
                )

            # Costruisce prompt
            prompt = self._build_trend_prompt(
                time_series, analysis_period, metrics, complexity, output_format
            )

            execution_time = time.time() - start_time

            return create_success_result(
                {
                    "analysis_type": "trend_analysis",
                    "prompt": prompt,
                    "configuration": {
                        "analysis_period": analysis_period,
                        "data_points": len(time_series),
                        "metrics": metrics,
                        "complexity": complexity,
                        "output_format": output_format
                    },
                    "trend_metadata": {
                        "data_summary": self._summarize_time_data(time_series),
                        "analysis_scope": self._get_analysis_scope(complexity)
                    },
                    "instructions": f"Use this prompt to perform trend analysis in {output_format} format"
                },
                execution_time
            )

        except Exception as e:
            return create_error_result(
                f"Trend analysis tool failed: {str(e)}",
                time.time() - start_time
            )

    def _build_trend_prompt(self, time_series: List, analysis_period: str, metrics: List[str],
                          complexity: str, output_format: str) -> str:
        """Costruisce prompt per trend analysis"""

        # Scope basato su complexity
        analysis_scope = self._get_analysis_scope(complexity)

        # Format instructions
        format_instructions = self._get_format_instructions(output_format)

        # Data summary
        data_summary = self._summarize_time_data(time_series)

        # Metrics description
        metrics_desc = ", ".join(metrics)

        return f"""Perform {analysis_scope} for {analysis_period}

Time Series Data:
{data_summary}

Analysis Metrics: {metrics_desc}

Analysis Requirements:
- Identify primary trends and assess their statistical significance
- Detect cyclical patterns, seasonality, and periodic behaviors
- Highlight significant changes, inflection points, and anomalies
- Assess trend sustainability and identify risk factors
- Provide forward-looking insights and directional indicators
- Quantify trend strength and confidence levels where possible
- Compare current trends to historical patterns
- Identify leading and lagging indicators

{format_instructions}

Deliver actionable trend insights that support strategic planning and decision-making with quantitative evidence where available."""

    def _get_analysis_scope(self, complexity: str) -> str:
        """Determina scope di analisi basato su complexity"""
        scope_map = {
            "simple": "basic trend identification with direction and strength assessment",
            "medium": "comprehensive trend analysis with seasonality and pattern recognition",
            "complex": "advanced trend analysis with forecasting, causality assessment, and confidence intervals"
        }
        return scope_map.get(complexity, scope_map["medium"])

    def _summarize_time_data(self, time_series: List) -> str:
        """Riassume i dati temporali per il prompt"""
        if not time_series:
            return "No time series data provided"

        # Try to extract meaningful info from time series
        data_points = len(time_series)

        # If it's a simple list, describe it
        if isinstance(time_series[0], (int, float)):
            return f"Numerical time series with {data_points} data points"
        elif isinstance(time_series[0], dict):
            # Try to identify structure
            keys = time_series[0].keys() if time_series else []
            return f"Structured time series with {data_points} records, fields: {list(keys)}"
        else:
            return f"Time series data with {data_points} entries"

    def _get_format_instructions(self, output_format: str) -> str:
        """Istruzioni formato per trend analysis"""
        format_map = {
            "markdown": "Format with trend visualization descriptions, statistical analysis, and structured insight sections.",
            "json": "Structure as JSON with trends, patterns, forecasts, statistical_measures, and insights sections.",
            "bullet_points": "Present analysis as organized bullet points with trend indicators and confidence levels.",
            "plain_text": "Present clear trend narrative with quantitative supporting evidence and actionable insights."
        }
        return format_map.get(output_format, format_map["markdown"])


class MultiStepSynthesisTool(ToolBase):
    """Tool per sintesi multi-step che combina risultati di altri tool"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "synthesis_steps", ParameterType.ARRAY,
                "Sequence of synthesis steps to execute",
                required=True
            ),
            create_parameter_schema(
                "final_goal", ParameterType.STRING,
                "Final synthesis goal or objective",
                required=True
            ),
            create_parameter_schema(
                "integration_strategy", ParameterType.STRING,
                "Strategy for integrating multi-step results",
                required=False,
                default_value="comprehensive",
                allowed_values=["sequential", "parallel", "comprehensive", "hierarchical"]
            ),
            create_parameter_schema(
                "output_format", ParameterType.STRING,
                "Output format for final synthesis",
                required=False,
                default_value="markdown",
                allowed_values=["plain_text", "markdown", "json", "bullet_points"]
            )
        ]

        super().__init__(
            name="multi_step_synthesis",
            description="Orchestrates multi-step synthesis combining results from other synthesis tools",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["synthesis", "orchestration", "multi_step", "composition"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Coordina sintesi multi-step"""
        start_time = time.time()

        try:
            synthesis_steps = kwargs.get("synthesis_steps", [])
            final_goal = kwargs.get("final_goal")
            integration_strategy = kwargs.get("integration_strategy", "comprehensive")
            output_format = kwargs.get("output_format", "markdown")

            # Validazioni
            if not synthesis_steps or len(synthesis_steps) == 0:
                return create_error_result(
                    "At least one synthesis step is required",
                    time.time() - start_time
                )

            if not final_goal:
                return create_error_result(
                    "Final goal is required for multi-step synthesis",
                    time.time() - start_time
                )

            # Costruisce prompt per orchestrazione
            prompt = self._build_orchestration_prompt(
                synthesis_steps, final_goal, integration_strategy, output_format
            )

            execution_time = time.time() - start_time

            return create_success_result(
                {
                    "synthesis_type": "multi_step_synthesis",
                    "prompt": prompt,
                    "configuration": {
                        "final_goal": final_goal,
                        "steps_count": len(synthesis_steps),
                        "integration_strategy": integration_strategy,
                        "output_format": output_format
                    },
                    "orchestration_plan": {
                        "steps": synthesis_steps,
                        "strategy": integration_strategy,
                        "execution_approach": self._get_execution_approach(integration_strategy)
                    },
                    "instructions": f"Use this prompt to orchestrate multi-step synthesis in {output_format} format"
                },
                execution_time
            )

        except Exception as e:
            return create_error_result(
                f"Multi-step synthesis tool failed: {str(e)}",
                time.time() - start_time
            )

    def _build_orchestration_prompt(self, synthesis_steps: List, final_goal: str,
                                  integration_strategy: str, output_format: str) -> str:
        """Costruisce prompt per orchestrazione multi-step"""

        # Execution approach
        execution_approach = self._get_execution_approach(integration_strategy)

        # Format instructions
        format_instructions = self._get_format_instructions(output_format)

        # Steps description
        steps_desc = "\n".join([f"{i+1}. {step}" for i, step in enumerate(synthesis_steps)])

        return f"""Orchestrate multi-step synthesis to achieve: {final_goal}

Synthesis Steps to Execute:
{steps_desc}

Integration Strategy: {execution_approach}

Orchestration Requirements:
- Execute synthesis steps in logical sequence
- Integrate results from each step into coherent whole
- Identify synergies and conflicts between step results  
- Maintain consistency across integrated findings
- Highlight meta-insights that emerge from combination
- Ensure final synthesis addresses the stated goal
- Provide clear traceability to source steps

{format_instructions}

Deliver a unified synthesis that leverages all step results to achieve the final goal while maintaining analytical rigor."""

    def _get_execution_approach(self, integration_strategy: str) -> str:
        """Determina approccio di esecuzione basato su strategy"""
        approach_map = {
            "sequential": "Execute steps in order, with each step building on previous results",
            "parallel": "Execute steps independently, then integrate all results equally",
            "comprehensive": "Execute all steps and synthesize with cross-validation and conflict resolution",
            "hierarchical": "Execute steps with priority weighting and structured integration"
        }
        return approach_map.get(integration_strategy, approach_map["comprehensive"])

    def _get_format_instructions(self, output_format: str) -> str:
        """Istruzioni formato per multi-step synthesis"""
        format_map = {
            "markdown": "Format with step-by-step results, integration analysis, and unified synthesis sections.",
            "json": "Structure as JSON with step_results, integration_analysis, and final_synthesis sections.",
            "bullet_points": "Present orchestrated results as organized bullet points showing step contributions.",
            "plain_text": "Present unified synthesis with clear attribution to contributing steps and integration logic."
        }
        return format_map.get(output_format, format_map["markdown"])


# ===== SYNTHETIC AGENT PRINCIPALE =====

class SyntheticAgent(BaseAgent):
    """
    Agent specializzato nella sintesi di informazioni usando ReAct pattern.

    Utilizza tool modulari per diversi tipi di sintesi:
    - Executive Summary
    - Comparative Analysis
    - Data Fusion
    - Trend Analysis
    - Multi-Step Synthesis (orchestration)

    Opera in modalità ReAct (react=True) con loop PAUSE/Observation.
    """

    def __init__(self, agent_id: str, blackboard: BlackBoard, llm_client):
        """
        Inizializza SyntheticAgent in ReAct mode.

        Args:
            agent_id: ID univoco dell'agent
            blackboard: Blackboard condivisa
            llm_client: Client LLM per ReAct reasoning
        """
        # Inizializza BaseAgent in ReAct mode (react=True)
        super().__init__(agent_id, blackboard, llm_client, react=True)

        # Statistiche specifiche per sintesi
        self._synthesis_stats = {
            'total_syntheses': 0,
            'synthesis_types_used': {},
            'average_synthesis_time': 0.0,
            'total_synthesis_time': 0.0,
            'successful_syntheses': 0,
            'failed_syntheses': 0,
            'most_used_tool': None,
            'complex_syntheses': 0,  # Multi-step o complex complexity
            'last_synthesis_time': None
        }

        # Inizializza agent (registra tools automaticamente)
        success = self.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize SyntheticAgent {agent_id}")

    def setup_tools(self):
        """
        Implementazione richiesta da BaseAgent.
        Registra tutti i synthesis tool nel ReAct system.
        """
        # Registra tutti i synthesis tool
        self.register_tool(ExecutiveSummaryTool())
        self.register_tool(ComparativeAnalysisTool())
        self.register_tool(DataFusionTool())
        self.register_tool(TrendAnalysisTool())
        self.register_tool(MultiStepSynthesisTool())

        print(f"[{self.agent_id}] Registered {len(self._tools)} synthesis tools: "
              f"{', '.join(self.get_available_tools())}")

    def _generate_agent_description(self) -> str:
        """Override per descrizione specifica dell'agent di sintesi"""
        return (
            f"SyntheticAgent - Specialized synthesis agent operating in ReAct mode with tool orchestration. "
            f"Capable of executive summaries, comparative analysis, data fusion, trend analysis, and multi-step synthesis. "
            f"Uses systematic reasoning with modular synthesis tools for complex information processing. "
            f"Supports multiple complexity levels and output formats with full observability. "
            f"Tools: {', '.join(self.get_available_tools())}"
        )

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Statistiche specifiche di sintesi combinate con stats base"""
        base_stats = self.get_stats()

        # Calcola tool più usato
        if self._synthesis_stats['synthesis_types_used']:
            most_used = max(self._synthesis_stats['synthesis_types_used'].items(),
                          key=lambda x: x[1])
            self._synthesis_stats['most_used_tool'] = most_used[0]

        return {
            **base_stats,
            'synthesis_specific': self._synthesis_stats,
            'available_synthesis_types': self.get_available_tools(),
            'synthesis_success_rate': (
                self._synthesis_stats['successful_syntheses'] /
                max(self._synthesis_stats['total_syntheses'], 1) * 100
            )
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Override per capabilities specifiche di sintesi"""
        base_capabilities = super().get_capabilities()

        # Aggiungi capabilities specifiche di sintesi
        base_capabilities["synthesis_features"] = {
            "available_tools": self.get_available_tools(),
            "tool_schemas": [
                {
                    "name": tool_name,
                    "description": tool.description,
                    "parameters": [param.name for param in tool.parameters_schema]
                }
                for tool_name, tool in self._tools.items()
            ],
            "complexity_levels": [level.value for level in SynthesisComplexity],
            "output_formats": [fmt.value for fmt in OutputFormat],
            "react_mode": True,
            "composable_synthesis": True,
            "multi_step_orchestration": True,
            "observability": "full_react_loop_tracking"
        }

        return base_capabilities

    # Override del sistema di tracking per synthesis
    def _save_observation_to_blackboard(self, task_id: str, step: int, observation: str):
        """Override per tracking synthesis-specific"""
        # Chiama il metodo base
        super()._save_observation_to_blackboard(task_id, step, observation)

        # Aggiorna statistiche synthesis se rileva uso di synthesis tool
        if "Tool '" in observation and "executed successfully" in observation:
            # Estrae nome tool dalla observation
            tool_start = observation.find("Tool '") + 6
            tool_end = observation.find("'", tool_start)
            if tool_end > tool_start:
                tool_name = observation[tool_start:tool_end]

                # Aggiorna stats synthesis
                self._synthesis_stats['total_syntheses'] += 1
                self._synthesis_stats['synthesis_types_used'][tool_name] = (
                    self._synthesis_stats['synthesis_types_used'].get(tool_name, 0) + 1
                )
                self._synthesis_stats['successful_syntheses'] += 1
                self._synthesis_stats['last_synthesis_time'] = datetime.now(timezone.utc)

                # Check se synthesis complessa (multi-step o complex)
                if tool_name == 'multi_step_synthesis' or 'complex' in observation.lower():
                    self._synthesis_stats['complex_syntheses'] += 1


# ===== FACTORY PER CREAZIONE AGENT =====

class SyntheticAgentFactory:
    """Factory per creare SyntheticAgent con configurazioni predefinite"""

    @staticmethod
    def create_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> SyntheticAgent:
        """
        Crea SyntheticAgent con validazione completa.

        Args:
            agent_id: ID dell'agent
            blackboard: Blackboard condivisa
            llm_client: Client LLM

        Returns:
            SyntheticAgent: Agent configurato in ReAct mode

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

        try:
            agent = SyntheticAgent(
                agent_id=agent_id.strip(),
                blackboard=blackboard,
                llm_client=llm_client
            )

            print(f"[SyntheticAgentFactory] Created ReAct SyntheticAgent '{agent_id}' with "
                  f"{len(agent.get_available_tools())} synthesis tools")
            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to create SyntheticAgent: {str(e)}")

    @staticmethod
    def create_executive_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> SyntheticAgent:
        """Crea agent con focus su executive synthesis (stessi tool, naming diverso)"""
        agent = SyntheticAgentFactory.create_agent(agent_id, blackboard, llm_client)

        # Aggiunge system instruction per focus executive
        agent.add_system_instruction(
            instruction="Prioritize executive-level insights and business impact in all syntheses",
            instruction_type="behavior",
            priority_level=1
        )

        return agent

    @staticmethod
    def create_analytical_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> SyntheticAgent:
        """Crea agent con focus su analisi complesse"""
        agent = SyntheticAgentFactory.create_agent(agent_id, blackboard, llm_client)

        # Aggiunge system instruction per focus analitico
        agent.add_system_instruction(
            instruction="Emphasize quantitative analysis, statistical rigor, and detailed technical insights",
            instruction_type="behavior",
            priority_level=1
        )

        return agent


# ===== TEST E ESEMPIO D'USO =====

if __name__ == "__main__":
    print("=== SYNTHETIC AGENT REACT TEST ===")

    # Setup con LLM
    from multi_agent_system.core.llm import LlmProvider
    from multi_agent_system.core.black_board import BlackBoard
    import os
    from dotenv import load_dotenv

    load_dotenv()

    blackboard = BlackBoard()

    try:
        # Inizializza LLM client in ReAct mode
        llm_client = LlmProvider.IBM_WATSONX.get_instance()
        llm_client.set_react_mode(True)  # ReAct mode per tool usage

        # Crea agent di sintesi
        agent = SyntheticAgentFactory.create_executive_agent(
            agent_id="synthetic_react_agent",
            blackboard=blackboard,
            llm_client=llm_client
        )

        print("ReAct SyntheticAgent initialized successfully")
        print(f"Available tools: {agent.get_available_tools()}")

        # Simula Manager che crea task per executive summary
        task_id = blackboard.create_task(
            assigned_to="synthetic_react_agent",
            task_type="executive_synthesis",
            task_data={
                "request": """Create an executive summary of our Q4 performance using the following data:
                
                Sources:
                - Sales: $2.3M revenue, 15% growth over Q3
                - Customer satisfaction: 87% (up from 82%)  
                - New product: Contributed 30% of growth
                - Market share: Increased to 12% (from 10%)
                - Team: Hired 5 new engineers
                
                Key findings: Strong momentum, successful launch, improved metrics
                Recommendations: Scale marketing, maintain quality focus
                
                Please use the executive_summary tool with medium complexity and markdown format."""
            },
            created_by="manager"
        )

        print(f"Task created: {task_id[:8]}...")
        print("Agent will process via ReAct loop with executive_summary tool...")

        # Wait for processing (ReAct loop will handle automatically)
        import time
        max_wait = 45
        start_time = time.time()

        while time.time() - start_time < max_wait:
            task_status = blackboard.get_task_status(task_id, "synthetic_react_agent")
            if task_status and task_status.get("status") in ["completed", "failed"]:
                break
            time.sleep(1)

        # Check final result
        final_result = blackboard.get_task_status(task_id, "synthetic_react_agent")
        if final_result:
            if final_result.get("status") == "completed":
                result_data = final_result.get("result", {})
                print(f"\n=== TASK COMPLETED ===")
                print(f"Answer: {result_data.get('answer', 'No answer available')}")
            else:
                print(f"Task failed: {final_result.get('result', {}).get('error', 'Unknown error')}")
        else:
            print("Task timeout or not found")

        # Show agent stats
        stats = agent.get_synthesis_stats()
        print(f"\n=== AGENT STATS ===")
        print(f"Mode: ReAct with {len(agent.get_available_tools())} tools")
        print(f"Tasks completed: {stats['tasks_completed']}")
        print(f"Tools used: {stats['tools_used_count']}")
        print(f"Synthesis stats: {stats['synthesis_specific']}")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Check LLM configuration and dependencies")