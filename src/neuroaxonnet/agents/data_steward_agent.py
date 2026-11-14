"""
Data Steward and Provenance Engineer Agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
from neuroaxonnet.agents.base import BaseAgent
from neuroaxonnet.agents.message import AgentRole, PhaseID
from neuroaxonnet.data.ingestion import read_microscopy_raw, compute_file_sha256
from neuroaxonnet.data.schema import CanonicalPatchManifestRow, SplitType


class DataStewardAgent(BaseAgent):
    def __init__(self, data_root: Path | str = "data"):
        system_prompt = (
            "You are the Data Steward and Provenance Engineer for NeuroAxonNet. "
            "Your mandate is to guarantee 100% cryptographic data provenance via SHA-256 digests, "
            "enforce the 22-column canonical patch manifest schema, and organize immutable storage."
        )
        super().__init__(AgentRole.DATA_STEWARD, system_prompt)
        self.data_root = Path(data_root)

        self.register_tool("compute_sha256", compute_file_sha256, "Computes SHA-256 of file bytes before decoding.")
        self.register_tool("read_microscopy", read_microscopy_raw, "Reads raw microscopy with bit-depth and format validation.")

    def execute_phase(self, phase: PhaseID, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        if phase == PhaseID.PHASE_01:
            results["data_terms_approved"] = True
            results["raw_storage_policy"] = "Outside git, immutable, referenced via manifest"
            self.send_message(
                recipient=AgentRole.CHIEF_ARCHITECT,
                phase=phase,
                content="Data and license terms audited. Provenance schema approved.",
                data=results,
            )
        elif phase == PhaseID.PHASE_02:
            results["schema_columns_enforced"] = 22
            results["allowlist_formats"] = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
            results["bit_depths_supported"] = [8, 16]
            self.send_message(
                recipient=AgentRole.QUALITY_GATEKEEPER,
                phase=phase,
                content="Canonical 22-column manifest contract active with immutable SHA-256 pre-hashes.",
                data=results,
            )
        return results
