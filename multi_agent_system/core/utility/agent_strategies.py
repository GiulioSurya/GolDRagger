from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import json

from multi_agent_system.core.prompts import BASE_REACT_PROMPT, BASE_ACTING_PROMPT
import textwrap
from multi_agent_system.core.messages import ( create_agent_result, AgentResult
)


class _PromptStrategy(ABC):
    """Strategy interna per gestire diversi tipi di prompt"""

    @abstractmethod
    def execute_task_loop(self, agent: 'BaseAgent', task_data: Dict, task_id: str) -> AgentResult:
        """Esegue il loop specifico per questa strategy"""
        pass

    @abstractmethod
    def build_system_prompt(self, agent: 'BaseAgent') -> str:
        """Costruisce il system prompt per questa strategy"""
        pass


class _ReactStrategy(_PromptStrategy):
    """Strategy per ReAct pattern con tool usage e loop PAUSE/Observation"""

    def build_system_prompt(self, agent: 'BaseAgent') -> str:
        """Costruisce system prompt ReAct con schema JSON pulito e uniforme"""

        # Costruisce schema tool pulito senza valori inutili
        tools_schema = []

        for tool_name, tool in agent._tools.items():
            schema = tool.get_schema()

            # Schema tool base (sempre presente)
            tool_schema = {
                "name": tool_name,
                "description": schema["description"]
            }

            # Aggiungi parametri solo se esistono
            if schema.get("parameters"):
                clean_parameters = []

                for param in schema["parameters"]:
                    # Schema parametro base (sempre presente)
                    param_schema = {
                        "name": param["name"],
                        "type": param["type"],
                        "description": param["description"],
                        "required": param.get("required", True)
                    }

                    # Aggiungi solo campi con valori significativi
                    if param.get("default_value") is not None and param["default_value"] != "":
                        param_schema["default"] = param["default_value"]

                    if param.get("allowed_values") and len(param["allowed_values"]) > 0:
                        param_schema["options"] = param["allowed_values"]

                    if param.get("min_value") is not None:
                        param_schema["min"] = param["min_value"]

                    if param.get("max_value") is not None:
                        param_schema["max"] = param["max_value"]

                    clean_parameters.append(param_schema)

                if clean_parameters:
                    tool_schema["parameters"] = clean_parameters

            tools_schema.append(tool_schema)

        # Converti schema in JSON formattato
        tools_json = json.dumps(tools_schema, indent=2, separators=(',', ': '))

        # System instructions dinamiche
        system_instructions = agent.blackboard.get_system_instructions_for_agent(
            agent.agent_id, active_only=True
        )

        additional_instructions = ""
        if system_instructions:
            instructions_list = [instr.instruction_text for instr in system_instructions]
            additional_instructions = f"""

    ADDITIONAL INSTRUCTIONS:
    {chr(10).join(f"- {instr}" for instr in instructions_list)}"""

        # Usa template standardizzato
        return BASE_REACT_PROMPT.format(
            tools_description=tools_json
        ) + additional_instructions

    def execute_task_loop(self, agent: 'BaseAgent', task_data: Dict, task_id: str) -> AgentResult:
        """Esegue il ReAct loop con tool usage e PAUSE/Observation"""
        start_time = time.time()
        react_steps = 0
        tools_used = []
        observations = []
        max_steps = 10  # Limite safety per evitare loop infiniti

        try:
            # Costruisce system prompt con tool info auto-injected
            system_prompt = self.build_system_prompt(agent)

            # Prompt iniziale per il task
            conversation = f"Task: {task_data}\n\nPlease start with your first Thought about this task."

            while react_steps < max_steps:
                # Chiama LLM
                llm_response = agent.llm_client.invoke(
                    system=system_prompt,
                    user=conversation
                )
                wrapped_response = textwrap.fill(llm_response, width=80)
                print(f"\n------------------------[{agent.agent_id}]---------------------------\n")
                print(f"Acting Response: {wrapped_response}")

                # Parsing della risposta LLM
                if "Answer:" in llm_response:
                    # LLM ha dato risposta finale
                    answer = agent._extract_answer(llm_response)
                    return create_agent_result(
                        success=True,
                        agent_id=agent.agent_id,
                        data={"answer": answer, "task_data": task_data},
                        execution_time=time.time() - start_time,
                        react_steps=react_steps,
                        tools_used=tools_used,
                        observations=observations
                    )

                elif "Action:" in llm_response and "PAUSE" in llm_response:
                    # LLM vuole eseguire un'azione
                    action_result = agent._process_action(llm_response, task_id)

                    if action_result["success"]:
                        # Tool eseguito con successo
                        tool_name = action_result["tool_name"]
                        observation = action_result["observation"]

                        # Tracking
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                        observations.append(observation)

                        # Aggiorna stats
                        agent._stats["tools_used_count"][tool_name] = (
                                agent._stats["tools_used_count"].get(tool_name, 0) + 1
                        )

                        # Salva observation sulla blackboard
                        agent._save_observation_to_blackboard(task_id, react_steps, observation)

                        # Continua conversazione con observation
                        conversation += f"\n\n{llm_response}\n\nObservation: {observation}"

                    else:
                        # Tool fallito
                        error_obs = f"Error: {action_result['error']}"
                        observations.append(error_obs)
                        conversation += f"\n\n{llm_response}\n\nObservation: {error_obs}"

                    react_steps += 1

                else:
                    # Risposta LLM non nel formato atteso
                    conversation += f"\n\n{llm_response}\n\nPlease follow the format: Thought: ... Action: tool_name: params PAUSE"

            # Raggiunto limite step
            return create_agent_result(
                success=False,
                agent_id=agent.agent_id,
                error=f"Reached maximum ReAct steps ({max_steps})",
                execution_time=time.time() - start_time,
                react_steps=react_steps,
                tools_used=tools_used,
                observations=observations
            )

        except Exception as e:
            return create_agent_result(
                success=False,
                agent_id=agent.agent_id,
                error=f"ReAct loop failed: {str(e)}",
                execution_time=time.time() - start_time,
                react_steps=react_steps,
                tools_used=tools_used,
                observations=observations
            )


class _ActingStrategy(_PromptStrategy):
    """Strategy per Acting pattern: single-shot diretto, no tool, solo reasoning"""

    def build_system_prompt(self, agent: 'BaseAgent') -> str:
        """Costruisce system prompt Acting senza tool descriptions"""

        # Aggiunge system instructions dinamiche dalla blackboard
        system_instructions = agent.blackboard.get_system_instructions_for_agent(
            agent.agent_id, active_only=True
        )

        additional_instructions = ""
        if system_instructions:
            instructions_list = [instr.instruction_text for instr in system_instructions]
            additional_instructions = f"""

ADDITIONAL INSTRUCTIONS:
{chr(10).join(f"- {instr}" for instr in instructions_list)}"""

        return BASE_ACTING_PROMPT + additional_instructions

    def execute_task_loop(self, agent: 'BaseAgent', task_data: Dict, task_id: str) -> AgentResult:
        """Esegue single-shot diretto: una chiamata LLM e basta"""
        start_time = time.time()

        try:
            # Costruisce system prompt senza tool info
            system_prompt = self.build_system_prompt(agent)

            # Prompt diretto per il task
            user_prompt = f"Task: {task_data}\n\nPlease provide your direct response to complete this task efficiently."

            # Singola chiamata LLM
            llm_response = agent.llm_client.invoke(
                system=system_prompt,
                user=user_prompt
            )

            wrapped_response = textwrap.fill(llm_response, width=80)
            print(f"\n------------------------[{agent.agent_id}]---------------------------\n")
            print(f"Acting Response: {wrapped_response}")

            # Estrai answer se presente, altrimenti usa tutto il response
            if "Answer:" in llm_response:
                answer = agent._extract_answer(llm_response)
            else:
                answer = llm_response.strip()

            agent._save_observation_to_blackboard(task_id, 0, f"Direct response: {answer[:100]}...")

            return create_agent_result(
                success=True,
                agent_id=agent.agent_id,
                data={"answer": answer, "task_data": task_data},
                execution_time=time.time() - start_time,
                react_steps=1,  # Un singolo "step"
                tools_used=[],  # Nessun tool usato
                observations=[f"Direct reasoning completed"]
            )

        except Exception as e:
            return create_agent_result(
                success=False,
                agent_id=agent.agent_id,
                error=f"Acting execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                react_steps=0,
                tools_used=[],
                observations=[]
            )