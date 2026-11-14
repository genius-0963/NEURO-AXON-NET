"""
Deep Learning and Preprocessing Lead Agent.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from neuroaxonnet.agents.base import BaseAgent
from neuroaxonnet.agents.message import AgentRole, PhaseID
from neuroaxonnet.data.preprocessing import preprocess_baseline
from neuroaxonnet.data.density_builder import build_gaussian_density_target, verify_numeric_invariants


class MLEngineerAgent(BaseAgent):
    def __init__(self):
        system_prompt = (
            "You are the Deep Learning and Preprocessing Lead for NeuroAxonNet. "
            "Your mandate is mathematical and numeric rigor: enforcing Gaussian density target "
            "generation (sigma=8, truncate=2.0 -> 33px kernel, m=1000.0), robust two-sigma "
            "normalization clipped to [-1, 1], 224x224 reflection padding with [16:208, 16:208] "
            "valid-core masking, and verifying all Section 7.1 numeric invariants."
        )
        super().__init__(AgentRole.ML_ENGINEER, system_prompt)

        self.register_tool("preprocess_baseline", preprocess_baseline, "Executes robust norm and 224x224 reflect pad.")
        self.register_tool("build_density", build_gaussian_density_target, "Builds Gaussian density target map.")
        self.register_tool("verify_invariants", verify_numeric_invariants, "Asserts all 5 numeric invariants.")

    def execute_phase(self, phase: PhaseID, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if phase == PhaseID.PHASE_01:
            results["baseline_specification"] = "AxoNet U-Net density regression"
            results["density_scaling_m"] = 1000.0
            results["gaussian_sigma"] = 8.0
            results["filter_size_px"] = 33
            self.send_message(
                recipient=AgentRole.CHIEF_ARCHITECT,
                phase=phase,
                content="Baseline specification locked: U-Net ReLU count density regression.",
                data=results,
            )
        elif phase == PhaseID.PHASE_02:
            invariants = verify_numeric_invariants()
            results["numeric_invariants"] = invariants
            results["all_invariants_pass"] = (
                invariants["invariant_1_pass"]
                and invariants["invariant_2_k_annotator_invariance"]
                and invariants["invariant_3_superposition"]
                and invariants["invariant_4_border_documented"]
                and invariants["invariant_5_pass"]
            )
            self.send_message(
                recipient=AgentRole.QUALITY_GATEKEEPER,
                phase=phase,
                content="Preprocessing contract and 5 numeric invariants verified.",
                data=results,
            )
        return results
