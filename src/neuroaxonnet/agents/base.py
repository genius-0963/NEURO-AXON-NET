"""
Base Agent Abstract Class and Runtime Environment.

Provides state management, message logging, tool registration, and execution loop.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any

from neuroaxonnet.agents.message import AgentMessage, AgentRole, PhaseID, GateVerdict

logger = logging.getLogger("neuroaxonnet.agents")


class BaseAgent(ABC):
    def __init__(self, role: AgentRole, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, func: Callable, description: str):
        """Register a callable tool with documentation."""
        self.tools[name] = {"func": func, "description": description}

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool and return result."""
        if tool_name not in self.tools:
            raise KeyError(f"Agent {self.role.value} does not have tool '{tool_name}' registered.")
        return self.tools[tool_name]["func"](**kwargs)

    def receive_message(self, message: AgentMessage):
        self.inbox.append(message)

    def send_message(
        self,
        recipient: AgentRole | str,
        phase: PhaseID,
        content: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        msg = AgentMessage(
            sender=self.role,
            recipient=recipient,
            phase=phase,
            content=content,
            data=data or {},
        )
        self.outbox.append(msg)
        return msg

    @abstractmethod
    def execute_phase(self, phase: PhaseID, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent-specific responsibilities for the target phase."""
        pass
