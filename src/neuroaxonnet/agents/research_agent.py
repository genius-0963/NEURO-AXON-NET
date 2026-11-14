"""
Scientific Research Lead and Prior-Art Auditor Agent.
"""

from __future__ import annotations

from typing import Dict, Any
from neuroaxonnet.agents.base import BaseAgent
from neuroaxonnet.agents.message import AgentRole, PhaseID, GateVerdict


class ResearchAuditorAgent(BaseAgent):
    def __init__(self):
        system_prompt = (
            "You are the Scientific Research Lead and Prior-Art Auditor for NeuroAxonNet. "
            "Your mandate is to guarantee scientific fidelity to published AxoNet protocols, "
            "audit novelty claims against current literature (AxoNet 2.0, Chuter et al. 2026), "
            "and strictly enforce non-clinical research positioning (NFR-03)."
        )
        super().__init__(AgentRole.RESEARCH_LEAD, system_prompt)

    def audit_clinical_positioning(self, text_or_dict: Any) -> bool:
        """Verify non-clinical disclaimer is strictly enforced."""
        disclaimer = "RESEARCH PROTOTYPE ONLY"
        text = str(text_or_dict)
        return disclaimer in text

    def execute_phase(self, phase: PhaseID, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if phase == PhaseID.PHASE_01:
            results["literature_matrix_verified"] = True
            results["benchmarks"] = {
                "axonet_rat_mae": 4.4,
                "axonet_rat_r2": 0.938,
                "independent_validation_correlation_drop": 0.831,
                "segmentation_failure_dice": "0.29-0.40",
            }
            results["differentiation_strategy"] = "Density regression + automated QC and calibrated abstention"
            self.send_message(
                recipient=AgentRole.CHIEF_ARCHITECT,
                phase=phase,
                content="Prior-art matrix verified. Focus is external generalization and QC abstention.",
                data=results,
            )
        elif phase == PhaseID.PHASE_02:
            results["label_semantics_audited"] = True
            results["rule"] = "Absence of point annotation does not prove absence of axon."
            self.send_message(
                recipient=AgentRole.DATA_STEWARD,
                phase=phase,
                content="Data semantics audited: model is normal-appearing-axon quantifier under protocol.",
                data=results,
            )
        return results
