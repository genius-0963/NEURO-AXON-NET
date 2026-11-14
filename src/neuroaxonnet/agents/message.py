"""
Agent Inter-Communication and Governance Protocol.

Defines the message envelope, agent roles, phase states, and gate verdicts
for the autonomous multi-agent engineering team.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    CHIEF_ARCHITECT = "Chief Architect & Orchestrator"
    RESEARCH_LEAD = "Scientific Research Lead & Prior-Art Auditor"
    DATA_STEWARD = "Data Steward & Provenance Engineer"
    NEUROHISTOLOGIST = "Domain Expert Neurohistologist"
    ML_ENGINEER = "Deep Learning & Preprocessing Lead"
    QUALITY_GATEKEEPER = "Quality Assurance & Safety Auditor"


class PhaseID(str, Enum):
    PHASE_01 = "Phase 01: Discovery, Scientific Specification & Baseline Plan"
    PHASE_02 = "Phase 02: Data Engineering, Annotation QA & Preprocessing"
    PHASE_03 = "Phase 03: Baseline Model Reproduction & Parity Verification"
    PHASE_04 = "Phase 04: Robustness, Tiling & Automated Failure Detection"
    PHASE_05 = "Phase 05: Evaluation, Calibration, UI & Final Release"


class GateVerdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"
    FAIL = "FAIL"


class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    sender: AgentRole
    recipient: AgentRole | str = "all"
    phase: PhaseID
    content: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PhaseExitGateReport(BaseModel):
    phase: PhaseID
    verdict: GateVerdict
    evaluator: AgentRole
    criteria_evaluated: Dict[str, bool]
    blockers: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    signed_by: List[AgentRole] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
