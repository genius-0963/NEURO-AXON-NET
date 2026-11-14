"""
Domain Expert Neurohistologist Agent.
"""

from __future__ import annotations

from typing import Dict, Any, List
from neuroaxonnet.agents.base import BaseAgent
from neuroaxonnet.agents.message import AgentRole, PhaseID
from neuroaxonnet.data.annotation_qa import (
    calculate_count_agreement,
    match_points_one_to_one,
    build_consensus_point_map,
)
from neuroaxonnet.data.schema import AnnotationPoint, AnnotationRecord


class NeurohistologistAgent(BaseAgent):
    def __init__(self):
        system_prompt = (
            "You are the Lead Neurohistologist and Domain Expert for NeuroAxonNet. "
            "Your mandate is cytological precision: distinguishing normal-appearing axons from "
            "collapsed myelin figures, astrocytic glia, and capillaries; evaluating inter-rater "
            "agreement; and enforcing half-open boundary intervals to prevent double-counting."
        )
        super().__init__(AgentRole.NEUROHISTOLOGIST, system_prompt)

        self.register_tool("count_agreement", calculate_count_agreement, "Computes count discrepancy and CV.")
        self.register_tool("point_localization", match_points_one_to_one, "Bipartite matching within distance tolerance.")
        self.register_tool("consensus_builder", build_consensus_point_map, "Averages K-annotator indicator maps.")

    def execute_phase(self, phase: PhaseID, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if phase == PhaseID.PHASE_01:
            results["signed_label_protocol"] = "NAX-PROT-001"
            results["scale_um_per_px"] = 0.0625  # 12 um / 192 px
            self.send_message(
                recipient=AgentRole.CHIEF_ARCHITECT,
                phase=phase,
                content="Label protocol NAX-PROT-001 signed. Normal axon criteria defined.",
                data=results,
            )
        elif phase == PhaseID.PHASE_02:
            results["matching_tolerance_px"] = 5.0
            results["adjudication_f1_threshold"] = 0.85
            results["edge_rule"] = "half_open_interval"
            self.send_message(
                recipient=AgentRole.ML_ENGINEER,
                phase=phase,
                content="Annotation QA engine active. Half-open edge policy [0, W) x [0, H) enforced.",
                data=results,
            )
        return results
