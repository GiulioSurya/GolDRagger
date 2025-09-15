"""
AnalystAgent - ReAct mode agent specialized in numerical data analysis
Equipped with statistical analysis tools for comprehensive data processing
"""

from typing import Dict, Any, List
from datetime import datetime, timezone
import re
import statistics
import time

# Import dai moduli core del sistema
from multi_agent_system.core.base_agent import BaseAgent
from multi_agent_system.core.black_board import BlackBoard
from multi_agent_system.core.tool_base import (
    ToolBase, ToolResult, ParameterType,
    create_parameter_schema, create_success_result, create_error_result
)


class StatisticalAnalysisTool(ToolBase):
    """Tool for basic statistical analysis of numerical data"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "data_text", ParameterType.STRING,
                "Text containing numerical data to analyze",
                required=True
            ),
            create_parameter_schema(
                "analysis_type", ParameterType.STRING,
                "Type of statistical analysis to perform",
                required=False,
                default_value="comprehensive",
                allowed_values=["comprehensive", "central_tendency", "variability", "distribution"]
            )
        ]

        super().__init__(
            name="statistical_analysis",
            description="Performs statistical analysis on numerical data including mean, median, mode, standard deviation",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["statistics", "analysis", "math"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Perform statistical analysis on numerical data"""
        start_time = time.time()

        try:
            data_text = kwargs.get("data_text", "")
            analysis_type = kwargs.get("analysis_type", "comprehensive")

            if not data_text:
                return create_error_result(
                    "Data text is required for statistical analysis",
                    time.time() - start_time
                )

            # Extract numbers from text
            numbers = self._extract_numbers(data_text)

            if len(numbers) == 0:
                return create_success_result(
                    {
                        "analysis_type": analysis_type,
                        "data_points": 0,
                        "message": "No numerical data found in the provided text",
                        "extracted_numbers": []
                    },
                    time.time() - start_time
                )

            # Perform statistical analysis
            stats_result = self._calculate_statistics(numbers, analysis_type)

            return create_success_result(
                {
                    "analysis_type": analysis_type,
                    "data_points": len(numbers),
                    "extracted_numbers": numbers,
                    "statistics": stats_result,
                    "insights": self._generate_statistical_insights(stats_result, numbers)
                },
                time.time() - start_time
            )

        except Exception as e:
            return create_error_result(
                f"Statistical analysis failed: {str(e)}",
                time.time() - start_time
            )

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        # Pattern for numbers including decimals, negatives, and percentages
        pattern = r'-?\d+\.?\d*%?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                # Handle percentages
                if match.endswith('%'):
                    numbers.append(float(match[:-1]))
                else:
                    numbers.append(float(match))
            except ValueError:
                continue

        return numbers

    def _calculate_statistics(self, numbers: List[float], analysis_type: str) -> Dict[str, Any]:
        """Calculate statistical measures"""
        if not numbers:
            return {}

        stats = {}

        # Central tendency measures
        if analysis_type in ["comprehensive", "central_tendency"]:
            stats["mean"] = statistics.mean(numbers)
            stats["median"] = statistics.median(numbers)

            # Mode (handle potential StatisticsError)
            try:
                stats["mode"] = statistics.mode(numbers)
            except statistics.StatisticsError:
                stats["mode"] = "No unique mode"

        # Variability measures
        if analysis_type in ["comprehensive", "variability"]:
            if len(numbers) > 1:
                stats["standard_deviation"] = statistics.stdev(numbers)
                stats["variance"] = statistics.variance(numbers)
            else:
                stats["standard_deviation"] = 0
                stats["variance"] = 0

            stats["range"] = max(numbers) - min(numbers)
            stats["min_value"] = min(numbers)
            stats["max_value"] = max(numbers)

        # Distribution analysis
        if analysis_type in ["comprehensive", "distribution"]:
            sorted_nums = sorted(numbers)
            n = len(numbers)

            # Quartiles
            stats["q1"] = statistics.median(sorted_nums[:n // 2])
            stats["q3"] = statistics.median(sorted_nums[(n + 1) // 2:])
            stats["iqr"] = stats["q3"] - stats["q1"]

            # Count analysis
            stats["count"] = n
            stats["sum"] = sum(numbers)

        return stats

    def _generate_statistical_insights(self, stats: Dict[str, Any], numbers: List[float]) -> List[str]:
        """Generate insights based on statistical analysis"""
        insights = []

        if not stats:
            return ["No statistical analysis performed"]

        # Central tendency insights
        if "mean" in stats and "median" in stats:
            mean_val = stats["mean"]
            median_val = stats["median"]

            if abs(mean_val - median_val) / median_val > 0.1:  # 10% difference
                if mean_val > median_val:
                    insights.append("Mean is significantly higher than median, suggesting positive skew in data")
                else:
                    insights.append("Mean is significantly lower than median, suggesting negative skew in data")
            else:
                insights.append("Mean and median are close, indicating relatively symmetric distribution")

        # Variability insights
        if "standard_deviation" in stats and "mean" in stats:
            cv = (stats["standard_deviation"] / abs(stats["mean"])) * 100 if stats["mean"] != 0 else 0
            if cv > 30:
                insights.append(f"High variability detected (CV: {cv:.1f}%) - data shows significant dispersion")
            elif cv < 10:
                insights.append(f"Low variability detected (CV: {cv:.1f}%) - data is relatively consistent")

        # Range insights
        if "range" in stats and "mean" in stats:
            range_to_mean = stats["range"] / abs(stats["mean"]) if stats["mean"] != 0 else float('inf')
            if range_to_mean > 2:
                insights.append("Wide range relative to mean suggests potential outliers or diverse data")

        return insights


class PercentageAnalysisTool(ToolBase):
    """Tool for analyzing percentages, growth rates, and proportional relationships"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "data_text", ParameterType.STRING,
                "Text containing percentage data or values for percentage calculations",
                required=True
            ),
            create_parameter_schema(
                "calculation_type", ParameterType.STRING,
                "Type of percentage calculation to perform",
                required=False,
                default_value="auto_detect",
                allowed_values=["auto_detect", "growth_rate", "proportion", "change_analysis", "percentage_points"]
            )
        ]

        super().__init__(
            name="percentage_analysis",
            description="Analyzes percentages, calculates growth rates, and performs proportional analysis",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["percentage", "growth", "proportion", "analysis"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Perform percentage analysis"""
        start_time = time.time()

        try:
            data_text = kwargs.get("data_text", "")
            calc_type = kwargs.get("calculation_type", "auto_detect")

            if not data_text:
                return create_error_result(
                    "Data text is required for percentage analysis",
                    time.time() - start_time
                )

            # Extract percentages and numbers
            percentages = self._extract_percentages(data_text)
            numbers = self._extract_numbers(data_text)

            # Perform analysis based on type
            analysis_result = self._perform_percentage_analysis(percentages, numbers, calc_type, data_text)

            return create_success_result(
                {
                    "calculation_type": calc_type,
                    "extracted_percentages": percentages,
                    "extracted_numbers": numbers,
                    "analysis": analysis_result,
                    "insights": self._generate_percentage_insights(analysis_result, percentages)
                },
                time.time() - start_time
            )

        except Exception as e:
            return create_error_result(
                f"Percentage analysis failed: {str(e)}",
                time.time() - start_time
            )

    def _extract_percentages(self, text: str) -> List[float]:
        """Extract percentage values from text"""
        pattern = r'-?\d+\.?\d*%'
        matches = re.findall(pattern, text)

        percentages = []
        for match in matches:
            try:
                percentages.append(float(match[:-1]))
            except ValueError:
                continue

        return percentages

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract non-percentage numbers from text"""
        # Remove percentages first
        text_no_percent = re.sub(r'-?\d+\.?\d*%', '', text)

        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text_no_percent)

        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue

        return numbers

    def _perform_percentage_analysis(self, percentages: List[float], numbers: List[float],
                                     calc_type: str, original_text: str) -> Dict[str, Any]:
        """Perform percentage analysis based on type"""
        analysis = {}

        if calc_type == "auto_detect":
            calc_type = self._detect_calculation_type(original_text, percentages, numbers)
            analysis["detected_type"] = calc_type

        if calc_type == "growth_rate" and len(numbers) >= 2:
            analysis["growth_rates"] = self._calculate_growth_rates(numbers)

        elif calc_type == "proportion" and numbers:
            analysis["proportions"] = self._calculate_proportions(numbers)

        elif calc_type == "change_analysis" and percentages:
            analysis["change_analysis"] = self._analyze_changes(percentages)

        elif calc_type == "percentage_points" and len(percentages) >= 2:
            analysis["percentage_points"] = self._calculate_percentage_points(percentages)

        # General percentage statistics
        if percentages:
            analysis["percentage_stats"] = {
                "count": len(percentages),
                "average": sum(percentages) / len(percentages),
                "min": min(percentages),
                "max": max(percentages),
                "range": max(percentages) - min(percentages)
            }

        return analysis

    def _detect_calculation_type(self, text: str, percentages: List[float], numbers: List[float]) -> str:
        """Detect most appropriate calculation type based on context"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["growth", "increase", "decrease", "change"]):
            return "growth_rate"
        elif any(word in text_lower for word in ["proportion", "ratio", "share", "part"]):
            return "proportion"
        elif len(percentages) >= 2:
            return "percentage_points"
        else:
            return "change_analysis"

    def _calculate_growth_rates(self, numbers: List[float]) -> List[Dict[str, Any]]:
        """Calculate growth rates between consecutive numbers"""
        growth_rates = []

        for i in range(1, len(numbers)):
            if numbers[i - 1] != 0:
                rate = ((numbers[i] - numbers[i - 1]) / abs(numbers[i - 1])) * 100
                growth_rates.append({
                    "from_value": numbers[i - 1],
                    "to_value": numbers[i],
                    "growth_rate_percent": rate,
                    "period": f"Period {i}"
                })

        return growth_rates

    def _calculate_proportions(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate proportions of each number relative to total"""
        total = sum(numbers)
        if total == 0:
            return {"error": "Cannot calculate proportions - total is zero"}

        proportions = []
        for i, num in enumerate(numbers):
            proportions.append({
                "value": num,
                "proportion_percent": (num / total) * 100,
                "index": i
            })

        return {
            "total": total,
            "proportions": proportions
        }

    def _analyze_changes(self, percentages: List[float]) -> Dict[str, Any]:
        """Analyze changes in percentage values"""
        if len(percentages) < 2:
            return {"message": "Need at least 2 percentages for change analysis"}

        changes = []
        for i in range(1, len(percentages)):
            change = percentages[i] - percentages[i - 1]
            changes.append({
                "from_percent": percentages[i - 1],
                "to_percent": percentages[i],
                "absolute_change": change,
                "period": f"Period {i}"
            })

        return {
            "changes": changes,
            "total_change": percentages[-1] - percentages[0],
            "average_change": sum(c["absolute_change"] for c in changes) / len(changes)
        }

    def _calculate_percentage_points(self, percentages: List[float]) -> List[Dict[str, Any]]:
        """Calculate percentage point differences"""
        point_changes = []

        for i in range(1, len(percentages)):
            point_change = percentages[i] - percentages[i - 1]
            point_changes.append({
                "from_percent": percentages[i - 1],
                "to_percent": percentages[i],
                "percentage_points": point_change,
                "period": f"Period {i}"
            })

        return point_changes

    def _generate_percentage_insights(self, analysis: Dict[str, Any], percentages: List[float]) -> List[str]:
        """Generate insights from percentage analysis"""
        insights = []

        # Growth rate insights
        if "growth_rates" in analysis:
            rates = [gr["growth_rate_percent"] for gr in analysis["growth_rates"]]
            if rates:
                avg_growth = sum(rates) / len(rates)
                if avg_growth > 10:
                    insights.append(f"Strong positive growth trend with average rate of {avg_growth:.1f}%")
                elif avg_growth < -10:
                    insights.append(f"Significant decline trend with average rate of {avg_growth:.1f}%")
                else:
                    insights.append(f"Moderate growth/decline with average rate of {avg_growth:.1f}%")

        # Percentage statistics insights
        if "percentage_stats" in analysis:
            stats = analysis["percentage_stats"]
            if stats["range"] > 50:
                insights.append(f"High variability in percentages (range: {stats['range']:.1f} percentage points)")
            elif stats["range"] < 5:
                insights.append(f"Low variability in percentages (range: {stats['range']:.1f} percentage points)")

        return insights


class TrendIdentificationTool(ToolBase):
    """Tool for identifying trends, patterns, and directional movements in data"""

    def __init__(self):
        parameters = [
            create_parameter_schema(
                "data_text", ParameterType.STRING,
                "Text containing time series or sequential data",
                required=True
            ),
            create_parameter_schema(
                "trend_sensitivity", ParameterType.STRING,
                "Sensitivity level for trend detection",
                required=False,
                default_value="medium",
                allowed_values=["low", "medium", "high"]
            ),
            create_parameter_schema(
                "min_data_points", ParameterType.INTEGER,
                "Minimum number of data points required for trend analysis",
                required=False,
                default_value=3,
                min_value=2,
                max_value=20
            )
        ]

        super().__init__(
            name="trend_identification",
            description="Identifies trends, patterns, and directional movements in sequential data",
            parameters_schema=parameters,
            version="1.0.0",
            tags=["trends", "patterns", "time_series", "analysis"]
        )

    def execute(self, **kwargs) -> ToolResult:
        """Identify trends in data"""
        start_time = time.time()

        try:
            data_text = kwargs.get("data_text", "")
            sensitivity = kwargs.get("trend_sensitivity", "medium")
            min_points = kwargs.get("min_data_points", 3)

            if not data_text:
                return create_error_result(
                    "Data text is required for trend identification",
                    time.time() - start_time
                )

            # Extract sequential data
            numbers = self._extract_sequential_data(data_text)

            if len(numbers) < min_points:
                return create_success_result(
                    {
                        "trend_sensitivity": sensitivity,
                        "min_data_points": min_points,
                        "data_points": len(numbers),
                        "message": f"Insufficient data points for trend analysis (need at least {min_points})",
                        "extracted_data": numbers
                    },
                    time.time() - start_time
                )

            # Perform trend analysis
            trend_analysis = self._analyze_trends(numbers, sensitivity)

            return create_success_result(
                {
                    "trend_sensitivity": sensitivity,
                    "data_points": len(numbers),
                    "extracted_data": numbers,
                    "trend_analysis": trend_analysis,
                    "insights": self._generate_trend_insights(trend_analysis, numbers)
                },
                time.time() - start_time
            )

        except Exception as e:
            return create_error_result(
                f"Trend identification failed: {str(e)}",
                time.time() - start_time
            )

    def _extract_sequential_data(self, text: str) -> List[float]:
        """Extract numbers in sequential order from text"""
        # Find all numbers including percentages
        pattern = r'-?\d+\.?\d*%?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                if match.endswith('%'):
                    numbers.append(float(match[:-1]))
                else:
                    numbers.append(float(match))
            except ValueError:
                continue

        return numbers

    def _analyze_trends(self, numbers: List[float], sensitivity: str) -> Dict[str, Any]:
        """Analyze trends in the data"""
        if len(numbers) < 2:
            return {"error": "Need at least 2 data points for trend analysis"}

        # Set threshold based on sensitivity
        thresholds = {"low": 0.15, "medium": 0.10, "high": 0.05}  # 15%, 10%, 5%
        threshold = thresholds.get(sensitivity, 0.10)

        trend_analysis = {
            "overall_trend": self._determine_overall_trend(numbers, threshold),
            "segment_trends": self._analyze_segments(numbers, threshold),
            "trend_strength": self._calculate_trend_strength(numbers),
            "change_points": self._detect_change_points(numbers, threshold),
            "volatility": self._calculate_volatility(numbers)
        }

        return trend_analysis

    def _determine_overall_trend(self, numbers: List[float], threshold: float) -> Dict[str, Any]:
        """Determine overall trend direction"""
        if not numbers:
            return {"direction": "unknown", "magnitude": 0}

        start_val = numbers[0]
        end_val = numbers[-1]

        if start_val == 0:
            return {"direction": "undefined", "magnitude": 0}

        change_percent = ((end_val - start_val) / abs(start_val)) * 100

        if abs(change_percent) < threshold * 100:
            direction = "stable"
        elif change_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "direction": direction,
            "magnitude": abs(change_percent),
            "start_value": start_val,
            "end_value": end_val,
            "total_change_percent": change_percent
        }

    def _analyze_segments(self, numbers: List[float], threshold: float) -> List[Dict[str, Any]]:
        """Analyze trends in segments of the data"""
        if len(numbers) < 3:
            return []

        segments = []
        segment_size = max(3, len(numbers) // 3)  # At least 3 points per segment

        for i in range(0, len(numbers) - segment_size + 1, segment_size):
            segment = numbers[i:i + segment_size]
            if len(segment) >= 2:
                start_val = segment[0]
                end_val = segment[-1]

                if start_val != 0:
                    change_percent = ((end_val - start_val) / abs(start_val)) * 100

                    if abs(change_percent) < threshold * 100:
                        direction = "stable"
                    elif change_percent > 0:
                        direction = "increasing"
                    else:
                        direction = "decreasing"

                    segments.append({
                        "segment_index": len(segments) + 1,
                        "start_index": i,
                        "end_index": i + len(segment) - 1,
                        "direction": direction,
                        "change_percent": change_percent,
                        "values": segment
                    })

        return segments

    def _calculate_trend_strength(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate the strength of the trend"""
        if len(numbers) < 2:
            return {"strength": "undefined", "score": 0}

        # Calculate correlation coefficient with time indices
        time_indices = list(range(len(numbers)))
        n = len(numbers)

        # Calculate correlation coefficient manually
        mean_time = sum(time_indices) / n
        mean_values = sum(numbers) / n

        numerator = sum((time_indices[i] - mean_time) * (numbers[i] - mean_values) for i in range(n))

        sum_sq_time = sum((t - mean_time) ** 2 for t in time_indices)
        sum_sq_values = sum((v - mean_values) ** 2 for v in numbers)

        if sum_sq_time == 0 or sum_sq_values == 0:
            correlation = 0
        else:
            correlation = numerator / (sum_sq_time * sum_sq_values) ** 0.5

        # Determine strength based on correlation
        abs_corr = abs(correlation)
        if abs_corr > 0.8:
            strength = "very_strong"
        elif abs_corr > 0.6:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "very_weak"

        return {
            "strength": strength,
            "correlation_coefficient": correlation,
            "score": abs_corr
        }

    def _detect_change_points(self, numbers: List[float], threshold: float) -> List[Dict[str, Any]]:
        """Detect significant change points in the data"""
        if len(numbers) < 3:
            return []

        change_points = []

        for i in range(1, len(numbers) - 1):
            # Look at change before and after this point
            before_change = ((numbers[i] - numbers[i - 1]) / abs(numbers[i - 1])) * 100 if numbers[i - 1] != 0 else 0
            after_change = ((numbers[i + 1] - numbers[i]) / abs(numbers[i])) * 100 if numbers[i] != 0 else 0

            # Detect significant direction change
            if abs(before_change) > threshold * 100 or abs(after_change) > threshold * 100:
                if (before_change > 0 and after_change < 0) or (before_change < 0 and after_change > 0):
                    change_points.append({
                        "index": i,
                        "value": numbers[i],
                        "change_type": "direction_reversal",
                        "before_change": before_change,
                        "after_change": after_change
                    })
                elif abs(before_change - after_change) > threshold * 200:  # Significant acceleration/deceleration
                    change_points.append({
                        "index": i,
                        "value": numbers[i],
                        "change_type": "acceleration_change",
                        "before_change": before_change,
                        "after_change": after_change
                    })

        return change_points

    def _calculate_volatility(self, numbers: List[float]) -> Dict[str, Any]:
        """Calculate volatility/stability of the data"""
        if len(numbers) < 2:
            return {"volatility": "undefined", "coefficient": 0}

        # Calculate percentage changes
        changes = []
        for i in range(1, len(numbers)):
            if numbers[i - 1] != 0:
                change = ((numbers[i] - numbers[i - 1]) / abs(numbers[i - 1])) * 100
                changes.append(change)

        if not changes:
            return {"volatility": "undefined", "coefficient": 0}

        # Calculate standard deviation of changes
        mean_change = sum(changes) / len(changes)
        variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
        std_dev = variance ** 0.5

        # Classify volatility
        if std_dev > 20:
            volatility_level = "very_high"
        elif std_dev > 10:
            volatility_level = "high"
        elif std_dev > 5:
            volatility_level = "moderate"
        elif std_dev > 2:
            volatility_level = "low"
        else:
            volatility_level = "very_low"

        return {
            "volatility": volatility_level,
            "coefficient": std_dev,
            "changes": changes,
            "mean_change": mean_change
        }

    def _generate_trend_insights(self, trend_analysis: Dict[str, Any], numbers: List[float]) -> List[str]:
        """Generate insights from trend analysis"""
        insights = []

        # Overall trend insights
        if "overall_trend" in trend_analysis:
            overall = trend_analysis["overall_trend"]
            direction = overall.get("direction", "unknown")
            magnitude = overall.get("magnitude", 0)

            if direction == "increasing":
                insights.append(f"Strong upward trend detected with {magnitude:.1f}% overall growth")
            elif direction == "decreasing":
                insights.append(f"Declining trend identified with {magnitude:.1f}% overall decrease")
            elif direction == "stable":
                insights.append(f"Stable pattern with minimal change ({magnitude:.1f}%)")

        # Trend strength insights
        if "trend_strength" in trend_analysis:
            strength = trend_analysis["trend_strength"]
            strength_level = strength.get("strength", "unknown")
            correlation = strength.get("correlation_coefficient", 0)

            if strength_level in ["very_strong", "strong"]:
                insights.append(f"High trend consistency (correlation: {correlation:.2f}) - predictable pattern")
            elif strength_level == "very_weak":
                insights.append(f"Low trend consistency (correlation: {correlation:.2f}) - erratic behavior")

        # Change points insights
        if "change_points" in trend_analysis:
            change_points = trend_analysis["change_points"]
            if len(change_points) > 0:
                insights.append(f"Detected {len(change_points)} significant change points - trend instability")
            elif len(numbers) > 5:
                insights.append("No significant change points - consistent trend behavior")

        # Volatility insights
        if "volatility" in trend_analysis:
            volatility = trend_analysis["volatility"]
            vol_level = volatility.get("volatility", "unknown")

            if vol_level in ["very_high", "high"]:
                insights.append(f"High volatility detected - unpredictable fluctuations")
            elif vol_level == "very_low":
                insights.append(f"Low volatility - stable and predictable changes")

        return insights


class AnalystAgent(BaseAgent):
    """
    Agent specializzato nell'analisi numerica usando ReAct pattern.
    Equipaggiato con tool statistici per analisi comprensiva dei dati.
    """

    def __init__(self, agent_id: str, blackboard: BlackBoard, llm_client):
        """
        Inizializza AnalystAgent in ReAct mode.

        Args:
            agent_id: ID univoco dell'agent
            blackboard: Blackboard condivisa
            llm_client: Client LLM per ReAct reasoning
        """
        # Inizializza in ReAct mode (react=True)
        super().__init__(agent_id, blackboard, llm_client, react=True)

        # Statistiche specifiche per analisi
        self._analysis_stats = {
            'total_analyses': 0,
            'statistical_analyses': 0,
            'percentage_analyses': 0,
            'trend_analyses': 0,
            'tool_usage': {},
            'last_analysis_time': None
        }

        success = self.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize AnalystAgent {agent_id}")

    def setup_tools(self):
        """Registra i 3 tool di analisi numerica"""
        self.register_tool(StatisticalAnalysisTool())
        self.register_tool(PercentageAnalysisTool())
        self.register_tool(TrendIdentificationTool())

        print(f"[{self.agent_id}] Registered {len(self._tools)} analysis tools: "
              f"{', '.join(self.get_available_tools())}")

    def _generate_agent_description(self) -> str:
        """Override per descrizione specifica dell'AnalystAgent"""
        return (
            f"AnalystAgent - Advanced numerical analysis specialist operating in ReAct mode "
            f"with {len(self._tools)} statistical tools: {', '.join(self.get_available_tools())}. "
            f"Performs comprehensive data analysis including statistical measures, percentage "
            f"calculations, growth rates, and trend identification. Designed for quantitative "
            f"analysis and data-driven insights generation."
        )

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche specifiche di analisi"""
        base_stats = self.get_stats()
        return {
            **base_stats,
            'analysis_stats': self._analysis_stats,
            'available_tools': self.get_available_tools(),
            'specializations': [
                "Statistical analysis (mean, median, std dev)",
                "Percentage and growth rate calculations",
                "Trend identification and pattern recognition",
                "Data volatility assessment",
                "Change point detection"
            ]
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Capabilities specifiche per analisi numerica"""
        base_capabilities = super().get_capabilities()

        base_capabilities["analysis_features"] = {
            "mode": "react",
            "available_tools": self.get_available_tools(),
            "analysis_types": [
                "Descriptive statistics",
                "Central tendency measures",
                "Variability analysis",
                "Distribution analysis",
                "Growth rate calculations",
                "Percentage analysis",
                "Proportion calculations",
                "Trend identification",
                "Pattern recognition",
                "Volatility assessment"
            ],
            "statistical_measures": [
                "Mean, median, mode",
                "Standard deviation, variance",
                "Range, quartiles, IQR",
                "Correlation coefficients",
                "Percentage changes",
                "Growth rates",
                "Trend strength indicators"
            ],
            "output_formats": [
                "Statistical summaries",
                "Trend analysis reports",
                "Change detection results",
                "Quantitative insights",
                "Data quality assessments"
            ]
        }

        return base_capabilities

    def _save_observation_to_blackboard(self, task_id: str, step: int, observation: str):
        """Override per tracking analisi numeriche"""
        super()._save_observation_to_blackboard(task_id, step, observation)

        # Track tool usage specifico
        if "Tool '" in observation and "executed successfully" in observation:
            tool_start = observation.find("Tool '") + 6
            tool_end = observation.find("'", tool_start)
            if tool_end > tool_start:
                tool_name = observation[tool_start:tool_end]

                # Update general stats
                self._analysis_stats['total_analyses'] += 1
                self._analysis_stats['tool_usage'][tool_name] = (
                        self._analysis_stats['tool_usage'].get(tool_name, 0) + 1
                )

                # Update specific counters
                if tool_name == "statistical_analysis":
                    self._analysis_stats['statistical_analyses'] += 1
                elif tool_name == "percentage_analysis":
                    self._analysis_stats['percentage_analyses'] += 1
                elif tool_name == "trend_identification":
                    self._analysis_stats['trend_analyses'] += 1

                self._analysis_stats['last_analysis_time'] = datetime.now(timezone.utc)


# ===== FACTORY =====

class AnalystAgentFactory:
    """Factory per creare AnalystAgent con configurazioni specifiche"""

    @staticmethod
    def create_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> AnalystAgent:
        """Crea AnalystAgent standard"""
        if not all([agent_id, blackboard, llm_client]):
            raise ValueError("agent_id, blackboard, and llm_client are required")

        try:
            agent = AnalystAgent(
                agent_id=agent_id.strip(),
                blackboard=blackboard,
                llm_client=llm_client
            )

            print(f"[AnalystAgentFactory] Created AnalystAgent '{agent_id}' in ReAct mode with "
                  f"{len(agent.get_available_tools())} analysis tools")
            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to create AnalystAgent: {str(e)}")

    @staticmethod
    def create_statistical_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> AnalystAgent:
        """Crea agent con focus su analisi statistica"""
        agent = AnalystAgentFactory.create_agent(agent_id, blackboard, llm_client)

        agent.add_system_instruction(
            instruction="Prioritize statistical rigor and quantitative analysis. Always provide confidence levels and interpret statistical significance.",
            instruction_type="behavior",
            priority_level=1
        )

        agent.add_system_instruction(
            instruction="When analyzing data, start with descriptive statistics before moving to advanced analysis. Explain statistical concepts clearly.",
            instruction_type="constraint",
            priority_level=2
        )

        return agent

    @staticmethod
    def create_trend_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> AnalystAgent:
        """Crea agent specializzato in analisi di trend"""
        agent = AnalystAgentFactory.create_agent(agent_id, blackboard, llm_client)

        agent.add_system_instruction(
            instruction="Focus on trend identification, pattern recognition, and predictive insights. Always assess trend strength and sustainability.",
            instruction_type="behavior",
            priority_level=1
        )

        agent.add_system_instruction(
            instruction="When detecting trends, provide confidence levels and identify potential change points or inflection points.",
            instruction_type="constraint",
            priority_level=2
        )

        return agent

    def create_efficient_analyst_agent(agent_id: str, blackboard: BlackBoard, llm_client) -> AnalystAgent:
        agent = AnalystAgentFactory.create_agent(agent_id, blackboard, llm_client)

        agent.add_system_instruction(
            instruction="CRITICAL: After using ANY tool, immediately provide Answer with the analysis results. Do not use multiple tools unless absolutely necessary. Maximum 3 actions total.",
            instruction_type="constraint",
            priority_level=1
        )

        agent.add_system_instruction(
            instruction="Prioritize one comprehensive analysis over multiple partial analyses. Use statistical_analysis for most cases unless specific percentage or trend analysis is explicitly requested.",
            instruction_type="behavior",
            priority_level=1
        )

        return agent


# ===== TEST =====

if __name__ == "__main__":
    print("=== ANALYST AGENT - REACT MODE TEST ===")

    from multi_agent_system.core.black_board import BlackBoard

    blackboard = BlackBoard()


    # Mock LLM per test ReAct mode
    class MockLLM:
        def invoke(self, system, user):
            return """Thought: I need to analyze the numerical data provided to identify trends and calculate statistics.

Action: statistical_analysis: {"data_text": "10, 15, 12, 18, 22, 19, 25, 28", "analysis_type": "comprehensive"}
PAUSE"""

        def set_react_mode(self, mode):
            pass


    try:
        # Test standard agent
        agent = AnalystAgentFactory.create_agent(
            agent_id="analyst_react",
            blackboard=blackboard,
            llm_client=MockLLM()
        )

        print("AnalystAgent (ReAct mode) created successfully")
        print(f"Mode: {agent.get_capabilities()['mode']}")
        print(f"Tools: {agent.get_available_tools()}")

        # Test capabilities
        capabilities = agent.get_capabilities()
        print(f"Analysis types: {capabilities['analysis_features']['analysis_types'][:3]}...")

        # Test statistical agent
        stat_agent = AnalystAgentFactory.create_statistical_agent(
            agent_id="analyst_statistical",
            blackboard=blackboard,
            llm_client=MockLLM()
        )

        print(f"Statistical agent created with system instructions")

    except Exception as e:
        print(f"Test failed: {str(e)}")