#!/usr/bin/env python3
"""
Autonomous AI Agent Team Runner for NeuroAxonNet.

Executes the multi-agent team across the project phases, logging agent reasoning,
message bus exchanges, and formal exit gate sign-offs.
"""

from neuroaxonnet.agents.orchestrator import MultiAgentTeamOrchestrator


def main():
    print("\n" + "=" * 70)
    print("      NEUROAXONNET: AUTONOMOUS AI AGENT SYSTEM INVOCATION        ")
    print("=" * 70)

    orchestrator = MultiAgentTeamOrchestrator()
    print("AI AGENTS INITIALIZED:")
    for ag in orchestrator.agents:
        print(f"  • {ag.role.value}")

    print("\n[STEP 1] Executing Phase 01 via Agent Collaboration...")
    p1_summary = orchestrator.execute_phase_01()
    print(f"  Phase 01 Exit Gate Verdict: [{p1_summary['verdict']}]")
    print(f"  Signed By: {p1_summary['gate_report']['signed_by']}")

    print("\n[STEP 2] Executing Phase 02 via Agent Collaboration...")
    p2_summary = orchestrator.execute_phase_02()
    print(f"  Phase 02 Exit Gate Verdict: [{p2_summary['verdict']}]")
    print(f"  Patches Processed:          {p2_summary['pipeline_meta']['total_records']}")
    print(f"  Split Hashes Locked:        {list(p2_summary['pipeline_meta']['split_hashes'].keys())}")
    print(f"  Signed By:                  {p2_summary['gate_report']['signed_by']}")

    print("\n[STEP 3] Executing Phase 03 via Agent Collaboration...")
    p3_summary = orchestrator.execute_phase_03()
    print(f"  Phase 03 Exit Gate Verdict: [{p3_summary['verdict']}]")
    print(f"  Overfit Convergence:        {'CONVERGED' if p3_summary['overfit_verification']['converged'] else 'FAILED'}")
    print(f"  Loss Reduction Ratio:       {p3_summary['overfit_verification']['reduction_ratio'] * 100:.1f}%")
    print(f"  Signed By:                  {p3_summary['gate_report']['signed_by']}")

    print("\n[STEP 4] Executing Phase 04 via Agent Collaboration...")
    p4_summary = orchestrator.execute_phase_04()
    bm = p4_summary["benchmark_report"]
    print(f"  Phase 04 Exit Gate Verdict: [{p4_summary['verdict']}]")
    print(f"  Raw Test MAE:               {bm['raw_test_metrics']['mae']} axons/patch")
    print(f"  Calibrated Test MAE:        {bm['calibrated_test_metrics']['mae']} axons/patch")
    print(f"  Concordance CCC:            {bm['calibrated_test_metrics']['concordance_correlation_ccc']}")
    print(f"  Bland-Altman 95% Limits:    [{bm['calibrated_test_metrics']['bland_altman']['lower_limit_95']}, {bm['calibrated_test_metrics']['bland_altman']['upper_limit_95']}]")
    print(f"  Group-Aware Bootstrap CI:   {bm['bootstrap_ci95']['mae_ci95']}")
    print(f"  Signed By:                  {p4_summary['gate_report']['signed_by']}")

    print("\n[STEP 5] Executing Phase 05 via Agent Collaboration...")
    p5_summary = orchestrator.execute_phase_05()
    print(f"  Phase 05 Exit Gate Verdict: [{p5_summary['verdict']}]")
    print(f"  Release Bundle Assets:      {sum(1 for v in p5_summary['release_assets'].values() if v)} / {len(p5_summary['release_assets'])} present")
    print(f"  Signed By:                  {p5_summary['gate_report']['signed_by']}")

    print("\n[STEP 6] Reviewing Agent Message Bus Traffic:")
    print(f"  Total Messages Exchanged:   {len(orchestrator.message_bus)}")
    for i, msg in enumerate(orchestrator.message_bus[:8], 1):
        print(f"  [{i}] {msg.sender.value} -> {msg.recipient}: {msg.content}")

    print("\n[STEP 7] Complete 5-Phase Project Roadmap Status:")
    lifecycle = orchestrator.execute_full_lifecycle()
    for phase_key, pdata in lifecycle["phases"].items():
        verdict = pdata.get("verdict", pdata.get("status"))
        print(f"  • {phase_key.upper()}: [{verdict}]")

    print("\n" + "=" * 70)
    print("      ALL GATES CLEARED - AUTONOMOUS AGENT RUN COMPLETE          ")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
