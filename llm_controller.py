import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_classic.chains.llm import LLMChain
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Centralized configuration"""
    
    # Network
    SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://minsky.informatik.uni-oldenburg.de:26129/")
    #SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8000") # http://minsky.informatik.uni-oldenburg.de:26129/
    MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
    
    # Model
    MODEL_NAME = os.getenv("LLM_MODEL", "gpt-oss:20b")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.15"))
    
    # Memory
    MEMORY_WINDOW_SIZE = 10  # Keep last 10 interactions
    
    # Limits
    MAX_TOOL_CALLS_PER_RUN = 40
    MAX_PLAN_STEPS = 20
    MAX_RETRIES = 3
    RETRY_BACKOFF = 0.6
    TOOL_TIMEOUT = 30.0
    
    # Logging
    LOG_LEVEL = os.getenv("LLM_LOG_LEVEL", "INFO").upper()
    LOG_FILE = os.getenv("LLM_LOG_FILE", "logs/deep_agent_enhanced.log")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure structured logging"""
    logger = logging.getLogger("deep_agent_enhanced")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
    logger.propagate = False
    
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# Pydantic Models for Structured Outputs
# ============================================================================

class IntentMode(str, Enum):
    """User intent classification"""
    CONTROL = "control"
    QUERY = "query"
    CHAT = "chat"
    CLARIFY = "clarify"


class Priority(str, Enum):
    """Task priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionStatus(str, Enum):
    """Status of agent actions"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StrategicGoal(BaseModel):
    """Structured output for strategic goals"""
    goal_id: str = Field(description="Unique identifier for the goal")
    description: str = Field(description="What needs to be achieved")
    priority: Priority = Field(default=Priority.MEDIUM)
    constraints: List[str] = Field(default_factory=list, description="Safety/operational constraints")
    success_criteria: List[str] = Field(default_factory=list, description="How to verify success")
    requires_tools: bool = Field(default=False, description="Whether this goal needs tool execution")


class StrategicPlan(BaseModel):
    """Structured output from strategic agent"""
    mode: IntentMode = Field(description="Classified user intent")
    reasoning: str = Field(description="Explanation of the plan")
    goals: List[StrategicGoal] = Field(default_factory=list)
    clarification: Optional[str] = Field(None, description="Question if mode is CLARIFY")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in plan")


class TacticalStep(BaseModel):
    """Structured output for tactical steps"""
    step_id: str = Field(description="Unique step identifier")
    description: str = Field(description="What this step does")
    tool_name: Optional[str] = Field(None, description="Specific MCP tool to use")
    tool_args_hint: Optional[str] = Field(None, description="Hint about tool arguments")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this depends on")
    verify: str = Field(description="How to verify success")
    recovery: str = Field(description="What to do if it fails")
    parallel_safe: bool = Field(default=False, description="Can run in parallel with other steps")


class TacticalPlan(BaseModel):
    """Structured output from tactical agent"""
    approach: str = Field(description="Overall approach description")
    steps: List[TacticalStep] = Field(default_factory=list)
    estimated_duration: Optional[float] = Field(None, description="Estimated seconds")
    risk_level: Literal["low", "medium", "high"] = Field(default="low")


class ReasoningStep(BaseModel):
    """Captures a single reasoning step"""
    layer: str
    step_id: str
    timestamp: float
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentAction(BaseModel):
    """Represents an action at any layer"""
    action_id: str
    layer: str
    intent: str
    status: ActionStatus = ActionStatus.PENDING
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    reasoning: List[ReasoningStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def add_reasoning(
        self,
        thought: str,
        action: Optional[str] = None,
        observation: Optional[str] = None,
        confidence: float = 1.0
    ) -> None:
        """Add a reasoning step"""
        step = ReasoningStep(
            layer=self.layer,
            step_id=f"r{len(self.reasoning) + 1}",
            timestamp=time.time(),
            thought=thought,
            action=action,
            observation=observation,
            confidence=confidence
        )
        self.reasoning.append(step)


# ============================================================================
# Custom Callbacks for Observability
# ============================================================================

class AgentObservabilityCallback(AsyncCallbackHandler):
    """Callback handler for detailed agent observability"""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.logger = logging.getLogger(f"deep_agent_enhanced.callbacks.{run_id}")
        self.events: List[Dict[str, Any]] = []
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Log when LLM starts"""
        self.logger.debug(f"[{self.run_id}] LLM started with {len(prompts)} prompt(s)")
        self.events.append({
            "type": "llm_start",
            "timestamp": time.time(),
            "prompts_count": len(prompts)
        })
    
    async def on_llm_end(self, response: Any, **kwargs) -> None:
        """Log when LLM completes"""
        self.logger.debug(f"[{self.run_id}] LLM completed")
        self.events.append({
            "type": "llm_end",
            "timestamp": time.time()
        })
    
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Log LLM errors"""
        self.logger.error(f"[{self.run_id}] LLM error: {error}")
        self.events.append({
            "type": "llm_error",
            "timestamp": time.time(),
            "error": str(error)
        })
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Log tool starts"""
        tool_name = serialized.get("name", "unknown")
        self.logger.info(f"[{self.run_id}] Tool started: {tool_name}")
        self.events.append({
            "type": "tool_start",
            "timestamp": time.time(),
            "tool_name": tool_name,
            "input": input_str[:200]
        })
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Log tool completion"""
        self.logger.info(f"[{self.run_id}] Tool completed")
        self.events.append({
            "type": "tool_end",
            "timestamp": time.time(),
            "output": str(output)[:200]
        })
    
    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Log tool errors"""
        self.logger.error(f"[{self.run_id}] Tool error: {error}")
        self.events.append({
            "type": "tool_error",
            "timestamp": time.time(),
            "error": str(error)
        })
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Log chain starts"""
        self.logger.debug(f"[{self.run_id}] Chain started")
        self.events.append({
            "type": "chain_start",
            "timestamp": time.time()
        })
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Log chain completion"""
        self.logger.debug(f"[{self.run_id}] Chain completed")
        self.events.append({
            "type": "chain_end",
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of events"""
        tool_starts = [e for e in self.events if e["type"] == "tool_start"]
        tool_errors = [e for e in self.events if e["type"] == "tool_error"]
        
        return {
            "total_events": len(self.events),
            "tools_called": len(tool_starts),
            "tools_failed": len(tool_errors),
            "tool_names": [e["tool_name"] for e in tool_starts],
            "errors": [e["error"] for e in tool_errors]
        }


# ============================================================================
# Enhanced Base Agent with LangChain Features
# ============================================================================

class EnhancedBaseAgent(ABC):
    """Enhanced base agent with LangChain memory and callbacks"""
    
    def __init__(self, layer: str, llm: Any):
        self.layer = layer
        self.llm = llm
        self.logger = logging.getLogger(f"deep_agent_enhanced.{layer}")
        
        # Memory for this agent layer
        self.memory = ConversationBufferWindowMemory(
            k=Config.MEMORY_WINDOW_SIZE,
            return_messages=True,
            memory_key="history"
        )
    
    @abstractmethod
    async def process(
        self,
        task: str,
        context: Dict[str, Any],
        **kwargs
    ) -> List[AgentAction]:
        """Process a task and return actions"""
        pass
    
    def create_action(self, intent: str, **kwargs) -> AgentAction:
        """Helper to create a new action"""
        return AgentAction(
            action_id=f"{self.layer}_{uuid.uuid4().hex[:8]}",
            layer=self.layer,
            intent=intent,
            **kwargs
        )
    
    async def invoke_with_retry(
        self,
        chain: Any,
        inputs: Dict[str, Any],
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        max_retries: int = Config.MAX_RETRIES
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Invoke a chain with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await chain.ainvoke(inputs, config={"callbacks": callbacks or []})
                return result, None
            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"Chain invoke failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(Config.RETRY_BACKOFF * (attempt + 1))
        
        return None, last_error


# ============================================================================
# Enhanced Strategic Agent with Structured Outputs
# ============================================================================

class EnhancedStrategicAgent(EnhancedBaseAgent):
    """
    Strategic agent with Pydantic-validated structured outputs
    
    Uses LangChain's PydanticOutputParser for type-safe planning
    """
    
    def __init__(self, layer: str, llm: Any):
        super().__init__(layer, llm)
        
        # Output parser for structured planning
        self.output_parser = PydanticOutputParser(pydantic_object=StrategicPlan)
        
        # Build prompt template with format instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Strategic Planning Agent for an energy management system.

Your role:
1. Classify user intent: CONTROL (take action), QUERY (get info), CHAT (explain), or CLARIFY (need more info)
2. For CONTROL/QUERY: break down into 1-3 high-level goals
3. For CHAT: identify what to explain
4. For CLARIFY: formulate a clear question

{format_instructions}

Context:
- Total agents in system: {total_agents}
- Recent incidents: {recent_incidents}
- Memory highlights: {memory_highlights}

Be concise and focus on WHAT needs to be achieved, not HOW."""),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{user_request}")
        ])
        
        # Create chain with parser
        self.chain = (
            {
                "user_request": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions(),
                "total_agents": lambda x: x.get("total_agents", 0),
                "recent_incidents": lambda x: json.dumps(x.get("recent_incidents", [])),
                "memory_highlights": lambda x: json.dumps(x.get("memory_highlights", [])),
                "history": lambda _: self.memory.load_memory_variables({})["history"]
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )
    
    async def process(
        self,
        task: str,
        context: Dict[str, Any],
        run_id: Optional[str] = None,
        **kwargs
    ) -> List[AgentAction]:
        """Create strategic plan with structured output"""
        
        self.logger.info(f"Strategic planning: {task[:100]}")
        
        # Build context
        chain_input = {
            "user_request": task,
            "total_agents": context.get("topology_summary", {}).get("agents_total", 0),
            "recent_incidents": context.get("incidents", [])[-3:],
            "memory_highlights": context.get("memory", [])[-3:]
        }
        
        # Create callback
        callback = AgentObservabilityCallback(run_id or "strategic")
        
        # Invoke chain with structured output
        plan, error = await self.invoke_with_retry(
            self.chain,
            chain_input,
            callbacks=[callback]
        )
        
        if plan is None or error:
            self.logger.error(f"Strategic planning failed: {error}")
            action = self.create_action(intent=task)
            action.status = ActionStatus.FAILED
            action.error = error
            return [action]
        
        # Convert StrategicPlan to AgentActions
        actions = []
        
        for goal in plan.goals:
            action = self.create_action(
                intent=goal.description,
                metadata={
                    "goal_id": goal.goal_id,
                    "priority": goal.priority.value,
                    "constraints": goal.constraints,
                    "success_criteria": goal.success_criteria,
                    "mode": plan.mode.value,
                    "requires_tools": goal.requires_tools,
                    "plan_confidence": plan.confidence
                }
            )
            
            action.add_reasoning(
                thought=plan.reasoning,
                action=f"Strategic goal: {goal.description}",
                confidence=plan.confidence
            )
            
            actions.append(action)
        
        # Handle clarification
        if plan.mode == IntentMode.CLARIFY and not actions:
            action = self.create_action(
                intent="clarification_needed",
                metadata={
                    "mode": "clarify",
                    "question": plan.clarification or "Could you clarify your request?"
                }
            )
            actions.append(action)
        
        # Save to memory
        self.memory.save_context(
            {"input": task},
            {"output": f"Planned {len(actions)} strategic goal(s)"}
        )
        
        # Log callback summary
        summary = callback.get_summary()
        self.logger.info(f"Strategic callback summary: {summary}")
        
        return actions


# ============================================================================
# Enhanced Tactical Agent with Parallel Execution Planning
# ============================================================================

class EnhancedTacticalAgent(EnhancedBaseAgent):
    """
    Tactical agent with structured outputs and parallel execution detection
    """
    
    def __init__(self, layer: str, llm: Any):
        super().__init__(layer, llm)
        
        self.output_parser = PydanticOutputParser(pydantic_object=TacticalPlan)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Tactical Coordination Agent.

Your role:
- Decompose goals into 1-5 concrete executable steps
- Select specific MCP tools for each step
- Identify dependencies between steps
- Mark steps that can run in parallel (parallel_safe=true)

{format_instructions}

Context:
- Goal priority: {priority}
- Constraints: {constraints}
- Success criteria: {success_criteria}
- Topology: {topology_summary}

Be specific about tool names and execution order."""),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{goal}")
        ])
        
        self.chain = (
            {
                "goal": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions(),
                "priority": lambda x: x.get("priority", "medium"),
                "constraints": lambda x: json.dumps(x.get("constraints", [])),
                "success_criteria": lambda x: json.dumps(x.get("success_criteria", [])),
                "topology_summary": lambda x: json.dumps(x.get("topology_summary", {})),
                "history": lambda _: self.memory.load_memory_variables({})["history"]
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )
    
    async def process(
        self,
        task: str,
        context: Dict[str, Any],
        strategic_action: Optional[AgentAction] = None,
        run_id: Optional[str] = None,
        **kwargs
    ) -> List[AgentAction]:
        """Decompose strategic goal into tactical steps"""
        
        self.logger.info(f"Tactical decomposition: {task[:100]}")
        
        # Build context
        chain_input = {
            "goal": task,
            "priority": strategic_action.metadata.get("priority", "medium") if strategic_action else "medium",
            "constraints": strategic_action.metadata.get("constraints", []) if strategic_action else [],
            "success_criteria": strategic_action.metadata.get("success_criteria", []) if strategic_action else [],
            "topology_summary": context.get("topology_summary", {})
        }
        
        callback = AgentObservabilityCallback(run_id or "tactical")
        
        # Invoke chain
        plan, error = await self.invoke_with_retry(
            self.chain,
            chain_input,
            callbacks=[callback]
        )
        
        if plan is None or error:
            self.logger.error(f"Tactical planning failed: {error}")
            action = self.create_action(intent=task)
            action.status = ActionStatus.FAILED
            action.error = error
            return [action]
        
        # Convert TacticalPlan to AgentActions
        actions = []
        
        for step in plan.steps:
            action = self.create_action(
                intent=step.description,
                tool_name=step.tool_name,
                metadata={
                    "step_id": step.step_id,
                    "tool_args_hint": step.tool_args_hint,
                    "verify": step.verify,
                    "depends_on": step.depends_on,
                    "recovery": step.recovery,
                    "parallel_safe": step.parallel_safe,
                    "risk_level": plan.risk_level
                }
            )
            
            action.add_reasoning(
                thought=plan.approach,
                action=f"Tactical step: {step.description}"
            )
            
            actions.append(action)
        
        # Save to memory
        self.memory.save_context(
            {"input": task},
            {"output": f"Planned {len(actions)} tactical step(s), risk={plan.risk_level}"}
        )
        
        self.logger.info(f"Tactical callback summary: {callback.get_summary()}")
        
        return actions


# ============================================================================
# Enhanced Operational Agent with Parallel Execution
# ============================================================================

class EnhancedOperationalAgent(EnhancedBaseAgent):
    """
    Operational agent with:
    - Parallel tool execution for independent steps
    - Detailed callback tracking
    - Smart retry with exponential backoff
    """
    
    def __init__(self, layer: str, llm: Any, mcp_client: Any):
        super().__init__(layer, llm)
        self.mcp_client = mcp_client
        self.agent = None
        self.agent_executor = None
    
    async def initialize(self):
        """Initialize with MCP tools and create agent executor"""
        if self.mcp_client:
            tools = await self.mcp_client.get_tools()
            self.agent = create_agent(model=self.llm, tools=tools)
            
            # Create AgentExecutor for better control
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=tools,
                verbose=True,
                max_iterations=15,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            self.logger.info(f"Operational agent initialized with {len(tools)} tools")
    
    async def rebuild_agent(self):
        """Rebuild agent (for schema drift recovery)"""
        await self.initialize()
    
    async def process_parallel(
        self,
        actions: List[AgentAction],
        context: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> List[AgentAction]:
        """
        Process multiple actions, executing parallel-safe ones concurrently
        """
        # Separate parallel-safe from sequential
        parallel_actions = [a for a in actions if a.metadata.get("parallel_safe", False)]
        sequential_actions = [a for a in actions if not a.metadata.get("parallel_safe", False)]
        
        results = []
        
        # Execute parallel-safe actions concurrently
        if parallel_actions:
            self.logger.info(f"Executing {len(parallel_actions)} actions in parallel")
            parallel_results = await asyncio.gather(*[
                self._execute_single(action, context, run_id)
                for action in parallel_actions
            ], return_exceptions=True)
            
            for result in parallel_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Parallel execution error: {result}")
                else:
                    results.extend(result)
        
        # Execute sequential actions one by one
        for action in sequential_actions:
            result = await self._execute_single(action, context, run_id)
            results.extend(result)
        
        return results
    
    async def process(
        self,
        task: str,
        context: Dict[str, Any],
        tactical_action: Optional[AgentAction] = None,
        run_id: Optional[str] = None,
        **kwargs
    ) -> List[AgentAction]:
        """Execute a single tactical step"""
        
        action = self.create_action(
            intent=task,
            tool_name=tactical_action.tool_name if tactical_action else None
        )
        
        return await self._execute_single(action, context, run_id)
    
    async def _execute_single(
        self,
        action: AgentAction,
        context: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> List[AgentAction]:
        """Execute a single action with retries and callbacks"""
        
        if not self.agent_executor:
            self.logger.error("Operational agent not initialized")
            action.status = ActionStatus.FAILED
            action.error = "Agent not initialized"
            return [action]
        
        action.status = ActionStatus.RUNNING
        callback = AgentObservabilityCallback(run_id or "operational")
        
        # Execute with retries
        for attempt in range(Config.MAX_RETRIES):
            try:
                action.add_reasoning(
                    thought=f"Executing attempt {attempt + 1}",
                    action=f"Tool: {action.tool_name or 'agent reasoning'}"
                )
                
                # Build execution input
                if action.tool_name:
                    execution_input = f"Execute tool: {action.tool_name}\nTask: {action.intent}"
                else:
                    execution_input = action.intent
                
                # Execute with timeout and callbacks
                result = await asyncio.wait_for(
                    self.agent_executor.ainvoke(
                        {"input": execution_input},
                        config={"callbacks": [callback]}
                    ),
                    timeout=Config.TOOL_TIMEOUT
                )
                
                # Check for errors in output
                output = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])
                
                if self._has_error(output, intermediate_steps):
                    error_msg = self._extract_error(output, intermediate_steps)
                    action.error = error_msg
                    
                    action.add_reasoning(
                        thought=f"Execution failed: {self._classify_error(error_msg)}",
                        observation=error_msg[:200],
                        confidence=0.3
                    )
                    
                    # Retry logic
                    if attempt < Config.MAX_RETRIES - 1:
                        error_class = self._classify_error(error_msg)
                        
                        # Schema drift: rebuild agent
                        if error_class == "tool_not_found":
                            await self.rebuild_agent()
                        
                        await asyncio.sleep(Config.RETRY_BACKOFF * (attempt + 1))
                        continue
                    else:
                        action.status = ActionStatus.FAILED
                        break
                else:
                    # Success
                    action.status = ActionStatus.SUCCESS
                    action.result = {
                        "output": output,
                        "intermediate_steps": len(intermediate_steps),
                        "callback_summary": callback.get_summary()
                    }
                    
                    action.add_reasoning(
                        thought="Execution successful",
                        observation=str(output)[:200],
                        confidence=0.9
                    )
                    break
                    
            except asyncio.TimeoutError:
                action.error = "Execution timed out"
                action.add_reasoning(
                    thought=f"Timeout on attempt {attempt + 1}",
                    observation=f"Exceeded {Config.TOOL_TIMEOUT}s",
                    confidence=0.2
                )
                
                if attempt < Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Config.RETRY_BACKOFF * (attempt + 1))
                else:
                    action.status = ActionStatus.FAILED
                    
            except Exception as e:
                action.error = str(e)
                action.add_reasoning(
                    thought=f"Exception on attempt {attempt + 1}",
                    observation=str(e)[:200],
                    confidence=0.1
                )
                
                if attempt < Config.MAX_RETRIES - 1:
                    await asyncio.sleep(Config.RETRY_BACKOFF * (attempt + 1))
                else:
                    action.status = ActionStatus.FAILED
        
        # Save to memory
        self.memory.save_context(
            {"input": action.intent},
            {"output": f"Status: {action.status.value}"}
        )
        
        self.logger.info(f"Operational callback summary: {callback.get_summary()}")
        
        return [action]
    
    def _has_error(self, output: str, intermediate_steps: List) -> bool:
        """Check if execution had errors"""
        output_lower = str(output).lower()
        
        if any(err in output_lower for err in ["error", "failed", "exception", "invalid"]):
            return True
        
        # Check intermediate steps for tool errors
        for step in intermediate_steps:
            if len(step) > 1:
                tool_output = str(step[1]).lower()
                if any(err in tool_output for err in ["error", "failed", "exception"]):
                    return True
        
        return False
    
    def _extract_error(self, output: str, intermediate_steps: List) -> str:
        """Extract error message"""
        # Try output first
        if "error" in str(output).lower():
            return str(output)[:500]
        
        # Check intermediate steps
        for step in intermediate_steps:
            if len(step) > 1:
                tool_output = str(step[1])
                if "error" in tool_output.lower():
                    return tool_output[:500]
        
        return "Unknown error"
    
    def _classify_error(self, error_text: str) -> str:
        """Classify error type"""
        text = error_text.lower()
        
        if "timeout" in text:
            return "timeout"
        elif "validation" in text or "invalid" in text:
            return "validation"
        elif "tool" in text and ("not found" in text or "unknown" in text):
            return "tool_not_found"
        elif "connection" in text or "network" in text:
            return "network"
        elif any(code in text for code in ["500", "502", "503"]):
            return "server_error"
        else:
            return "unknown"


# ============================================================================
# Enhanced Deep Agent Orchestrator
# ============================================================================

class EnhancedDeepAgentOrchestrator:
    """
    Enhanced orchestrator with:
    - LangChain memory systems
    - Callback-based observability
    - Parallel execution support
    - Structured outputs throughout
    """
    
    def __init__(self, server_base_url: str, mcp_url: str):
        self.server_base_url = server_base_url
        self.mcp_url = mcp_url
        
        # Initialize LLM
        self.llm = ChatOllama(
            base_url=self.server_base_url,
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            stream=False
        )
        
        # Create enhanced agent layers
        self.strategic = EnhancedStrategicAgent("strategic", self.llm)
        self.tactical = EnhancedTacticalAgent("tactical", self.llm)
        self.operational: Optional[EnhancedOperationalAgent] = None
        
        # Connections
        self.http_client: Optional[httpx.AsyncClient] = None
        self.mcp_client: Optional[MultiServerMCPClient] = None
        
        # Session management
        self.session_pads: Dict[str, Dict[str, Any]] = {}
        
        # Global memory for cross-session learnings
        self.global_memory = ConversationBufferWindowMemory(
            k=20,
            return_messages=True,
            memory_key="global_history"
        )
        
        self.logger = logging.getLogger("deep_agent_enhanced.orchestrator")
    
    async def start(self):
        """Initialize all connections"""
        self.logger.info("Starting Enhanced Deep Agent Orchestrator")
        
        self.http_client = httpx.AsyncClient(
            base_url=self.server_base_url,
            timeout=30.0
        )
        
        self.mcp_client = MultiServerMCPClient({
            "mango": {
                "transport": "http",
                "url": self.mcp_url
            }
        })
        
        self.operational = EnhancedOperationalAgent(
            "operational",
            self.llm,
            self.mcp_client
        )
        await self.operational.initialize()
        
        self.logger.info("Enhanced Deep Agent Orchestrator ready")
    
    async def close(self):
        """Cleanup"""
        if self.mcp_client:
            await self.mcp_client.close()
        if self.http_client:
            await self.http_client.aclose()
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get or create session context"""
        if session_id not in self.session_pads:
            self.session_pads[session_id] = {
                "incident": [],
                "memory": [],
                "topology": {},
                "tool_trace": {}
            }
        
        pads = self.session_pads[session_id]
        
        return {
            "session_id": session_id,
            "memory": pads.get("memory", [])[-10:],
            "incidents": pads.get("incident", [])[-10:],
            "tool_trace": pads.get("tool_trace", {})
        }
    
    async def update_topology(self, context: Dict[str, Any]):
        """Fetch and update topology"""
        if not self.http_client:
            return
        
        try:
            response = await self.http_client.get("/topology")
            topology = response.json()
            
            context["topology_summary"] = {
                "agents_total": len(topology.get("agents", [])) if isinstance(topology.get("agents"), list) else 0,
                "raw_keys": list(topology.keys())[:10] if isinstance(topology, dict) else []
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch topology: {e}")
    
    async def process_request(
        self,
        prompt: str,
        session_id: str = "default",
        include_topology: bool = True,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process request through enhanced deep agent layers
        """
        
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]
        
        self.logger.info(f"[{run_id}] Processing: {prompt[:100]}")
        start_time = time.time()
        
        # Build context
        context = self.get_session_context(session_id)
        if include_topology:
            await self.update_topology(context)
        
        # LAYER 1: Strategic Planning (with structured output)
        strategic_actions = await self.strategic.process(prompt, context, run_id=run_id)
        
        if not strategic_actions:
            return self._build_response(
                run_id, "I couldn't formulate a plan.", time.time() - start_time,
                [], [], []
            )
        
        # Check for clarification
        if strategic_actions[0].metadata.get("mode") == "clarify":
            question = strategic_actions[0].metadata.get("question", "Could you clarify?")
            return self._build_response(
                run_id, question, time.time() - start_time,
                strategic_actions, [], []
            )
        
        # LAYER 2: Tactical Decomposition (with structured output)
        tactical_actions = []
        for strategic_action in strategic_actions:
            actions = await self.tactical.process(
                strategic_action.intent,
                context,
                strategic_action=strategic_action,
                run_id=run_id
            )
            tactical_actions.extend(actions)
        
        # LAYER 3: Operational Execution (with parallel support)
        operational_results = await self.operational.process_parallel(
            tactical_actions,
            context,
            run_id=run_id
        )
        
        # SYNTHESIS
        reply = await self._synthesize_response(
            prompt,
            strategic_actions,
            tactical_actions,
            operational_results
        )
        
        # Update session memory
        pads = self.session_pads.get(session_id, {})
        
        for op in operational_results:
            if op.status == ActionStatus.SUCCESS:
                pads.setdefault("memory", []).append(f"Successfully: {op.intent}")
            elif op.status == ActionStatus.FAILED:
                pads.setdefault("incident", []).append(f"Failed: {op.intent} - {op.error}")
        
        pads["memory"] = pads.get("memory", [])[-50:]
        pads["incident"] = pads.get("incident", [])[-50:]
        
        # Update global memory
        self.global_memory.save_context(
            {"input": prompt},
            {"output": reply}
        )
        
        wall_time = time.time() - start_time
        
        return self._build_response(
            run_id,
            reply,
            wall_time,
            strategic_actions,
            tactical_actions,
            operational_results
        )
    
    async def _synthesize_response(
        self,
        user_request: str,
        strategic: List[AgentAction],
        tactical: List[AgentAction],
        operational: List[AgentAction]
    ) -> str:
        """Synthesize final response using LangChain chain"""
        
        # Build synthesis prompt
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize a clear, concise response to the user based on the execution results.

Be natural and helpful. Report what was accomplished or what went wrong.

Strategic goals: {strategic_summary}
Tactical steps: {tactical_summary}
Execution results: {operational_summary}

Provide a brief, user-friendly response."""),
            ("user", "{user_request}")
        ])
        
        synthesis_chain = synthesis_prompt | self.llm
        
        try:
            result = await synthesis_chain.ainvoke({
                "user_request": user_request,
                "strategic_summary": f"{len(strategic)} goal(s)",
                "tactical_summary": f"{len(tactical)} step(s)",
                "operational_summary": f"{sum(1 for op in operational if op.status == ActionStatus.SUCCESS)}/{len(operational)} succeeded"
            })
            
            return result.content.strip()
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            
            # Fallback
            successes = sum(1 for op in operational if op.status == ActionStatus.SUCCESS)
            if successes == len(operational) and operational:
                return f"Successfully completed {successes} operation(s)."
            elif successes > 0:
                return f"Completed {successes} of {len(operational)} operations."
            else:
                return "Unable to complete the requested operations."
    
    def _build_response(
        self,
        run_id: str,
        reply: str,
        wall_time: float,
        strategic: List[AgentAction],
        tactical: List[AgentAction],
        operational: List[AgentAction]
    ) -> Dict[str, Any]:
        """Build response"""
        
        return {
            "run_id": run_id,
            "reply": reply,
            "wall_s": wall_time,
            "tool_trace": {
                "last_tools": [
                    {"name": op.tool_name, "status": op.status.value}
                    for op in operational
                    if op.tool_name
                ][-10:]
            },
            "model_debug": {
                "layers": {
                    "strategic": [json.loads(a.model_dump_json()) for a in strategic],
                    "tactical": [json.loads(a.model_dump_json()) for a in tactical],
                    "operational": [json.loads(a.model_dump_json()) for a in operational]
                }
            },
            "raw": {
                "plan": {
                    "mode": strategic[0].metadata.get("mode") if strategic else "chat",
                    "goal": strategic[0].intent if strategic else "",
                    "confidence": strategic[0].metadata.get("plan_confidence", 1.0) if strategic else 1.0
                },
                "run_summary": {
                    "strategic_goals": len(strategic),
                    "tactical_steps": len(tactical),
                    "operations_total": len(operational),
                    "operations_successful": sum(1 for op in operational if op.status == ActionStatus.SUCCESS),
                    "operations_parallel": sum(1 for t in tactical if t.metadata.get("parallel_safe", False))
                }
            }
        }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Enhanced Deep Agent LLM Service")

orchestrator: Optional[EnhancedDeepAgentOrchestrator] = None
runs: Dict[str, Dict[str, Any]] = {}


class TriggerReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    include_topology: bool = True


@app.on_event("startup")
async def startup():
    global orchestrator
    orchestrator = EnhancedDeepAgentOrchestrator(
        server_base_url=Config.SERVER_BASE_URL,
        mcp_url=Config.MCP_URL
    )
    await orchestrator.start()
    logger.info("Enhanced Deep Agent service started")


@app.on_event("shutdown")
async def shutdown():
    if orchestrator:
        await orchestrator.close()
    logger.info("Enhanced Deep Agent service stopped")


@app.get("/health")
async def health():
    return {"ok": True, "mode": "enhanced_deep_agent", "features": [
        "structured_outputs",
        "memory_systems",
        "parallel_execution",
        "callback_observability"
    ]}


@app.post("/llm/trigger")
async def trigger(req: TriggerReq):
    """Enhanced endpoint with LangChain features"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    
    run_id = uuid.uuid4().hex
    runs[run_id] = {"status": "queued", "run_id": run_id, "session_id": req.session_id}
    
    logger.info(f"[{run_id}] Queued: {req.prompt[:100]}")
    
    async def _job():
        runs[run_id] = {"status": "running", "run_id": run_id, "session_id": req.session_id}
        
        try:
            result = await orchestrator.process_request(
                prompt=req.prompt,
                session_id=req.session_id,
                include_topology=req.include_topology,
                run_id=run_id
            )
            
            runs[run_id] = {
                "status": "done",
                "run_id": run_id,
                "session_id": req.session_id,
                "reply": result["reply"],
                "wall_s": result["wall_s"],
                "tool_trace": result["tool_trace"],
                "model_debug": result["model_debug"],
                "plan": result["raw"]["plan"],
                "run_summary": result["raw"]["run_summary"]
            }
            
            logger.info(f"[{run_id}] Completed in {result['wall_s']:.2f}s")
            
        except Exception as e:
            logger.exception(f"[{run_id}] Failed: {e}")
            runs[run_id] = {
                "status": "error",
                "run_id": run_id,
                "session_id": req.session_id,
                "error": str(e)
            }
    
    asyncio.create_task(_job())
    return {"run_id": run_id, "status": "queued"}


@app.get("/llm/runs/{run_id}")
async def run_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return runs[run_id]


@app.get("/llm/notepads/{session_id}")
async def get_notepads(session_id: str):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    
    pads = orchestrator.session_pads.get(session_id, {})
    return {
        "session_id": session_id,
        "incident": pads.get("incident", []),
        "memory": pads.get("memory", []),
        "topology": pads.get("topology", {}),
        "tool_trace": pads.get("tool_trace", {})
    }


@app.post("/llm/notepads/{session_id}/clear")
async def clear_notepads(session_id: str):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    
    if session_id in orchestrator.session_pads:
        orchestrator.session_pads[session_id] = {
            "incident": [],
            "memory": [],
            "topology": {},
            "tool_trace": {}
        }
    
    # Also clear agent memories for this session
    orchestrator.strategic.memory.clear()
    orchestrator.tactical.memory.clear()
    if orchestrator.operational:
        orchestrator.operational.memory.clear()
    
    logger.info(f"Cleared notepads and memories for session: {session_id}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")