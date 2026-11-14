"""
Chief Architect and Master Orchestrator Agent.
"""

from __future__ import annotations

from typing import Dict, Any, List
from neuroaxonnet.agents.base import BaseAgent
from neuroaxonnet.agents.message import AgentRole, PhaseID, GateVerdict


class ChiefArchitectAgent(BaseAgent):
    def __init__(self):
        system_prompt = (
            "You are the Chief Architect and Master Orchestrator for NeuroAxonNet. "
            "Your mandate is overall project governance: coordinating the 5-phase execution "
            "lifecycle, enforcing Architecture Decision Records (NAX-ADR-001 through 009), "
            "verifying the Stop/Go exit gates, and delivering a production-grade research platform."
        )
        super().__init__(AgentRole.CHIEF_ARCHITECT, system_prompt)
        self.phase_status: Dict[PhaseID, GateVerdict] = {
            PhaseID.PHASE_01: GateVerdict.REVIEW,
            PhaseID.PHASE_02: GateVerdict.REVIEW,
            PhaseID.PHASE_03: GateVerdict.REVIEW,
            PhaseID.PHASE_04: GateVerdict.REVIEW,
            PhaseID.PHASE_05: GateVerdict.REVIEW,
        }

    def execute_phase(self, phase: PhaseID, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "phase": phase.value,
            "architecture_invariants_enforced": [
                "NAX-ADR-001: PyTorch core + NumPy parity reference adapter",
                "NAX-ADR-002: Density regression integral over segmentation",
                "NAX-ADR-003: Strict animal-level grouping; patch leakage prohibited",
                "NAX-ADR-004: Decoupled Python core powering CLI, API, UI",
                "NAX-ADR-005: First-class automated QC and abstention routing",
                "NAX-ADR-006: Research-use-only non-clinical governance",
                "NAX-ADR-007: Canonical 6-entity relational data contract",
                "NAX-ADR-008: Preprocessing norm and 224x224 valid-core padding",
                "NAX-ADR-009: Multi-annotator consensus and Hungarian point matching",
            ],
            "governance_status": "LOCKED",
        }
        self.send_message(
            recipient="all",
            phase=phase,
            content=f"Chief Architect directive issued for {phase.value}. All ADRs active.",
            data=results,
        )
        return results
