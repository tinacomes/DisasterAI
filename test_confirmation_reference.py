#!/usr/bin/env python3
"""Smoke tests for the exploiters' confirmation reference and the AI's
stochastic report rounding.

1. confirmation_reference='network' (default): an exploiter whose trusted
   network agrees with an AI report REWARDS the AI (Q[ai] rises) even when
   the report contradicts the exploiter's own stored prior; under the legacy
   'own' reference the same report is punished (Q[ai] falls). Explorers are
   unaffected by the flag (accuracy channel).
2. report_rounding='stochastic' (default): the delivered level is an unbiased
   randomisation of the aligned value (1-α)·truth + α·belief, so the
   delivered confirmation dose is linear in α; 'deterministic' reproduces the
   legacy np.round step behaviour. At α ∈ {0, 1} the two modes coincide.
3. Population metrics: seci_pop_data, aeci_ie_pop_data, seci_ie_pop_data and
   aeci_lockin_pop_data are populated at the metrics cadence.

Run: python3 test_confirmation_reference.py
"""
import random

import numpy as np

from DisasterAI_Model import DisasterModel

BASE = dict(
    share_exploitative=0.5, share_of_disaster=0.15, initial_trust=0.3,
    initial_ai_trust=0.25, number_of_humans=100, share_confirming=0.7,
    disaster_dynamics=2, width=30, height=30, ticks=10,
    learning_rate=0.1, epsilon=0.3, exploit_trust_lr=0.015,
    explor_trust_lr=0.03, ai_alignment_level=1.0,
)


def build_model(seed=42, **overrides):
    random.seed(seed)
    np.random.seed(seed)
    return DisasterModel(**{**BASE, **overrides})


def _setup_exploiter_with_consensus(model, cell, net_level):
    """Return an exploiter with >= 3 trusted friends who all believe
    `net_level` at `cell` with high confidence; the exploiter's own belief
    at the cell is 0."""
    agent = next(a for a in model.humans.values()
                 if a.agent_type == "exploitative" and len(a.friends) >= 3)
    for fid in list(agent.friends)[:5]:
        friend = model.humans.get(fid)
        if friend is None:
            continue
        agent.trust[fid] = 0.7
        friend.beliefs[cell] = {'level': net_level, 'confidence': 0.9}
    agent.beliefs[cell] = {'level': 0, 'confidence': 0.3}
    return agent


def _feed_and_evaluate(model, agent, cell, reported, stored_prior):
    """Inject one matured AI pending item and run evaluate_pending_info;
    return the resulting change in Q['ai']."""
    ai_id = next(iter(model.ais))
    model.tick = 20
    agent.pending_info_evaluations = [
        (16, ai_id, cell, int(reported), int(stored_prior), 0.3, True)
    ]
    agent.q_table['ai'] = 0.0
    q_before = agent.q_table['ai']
    agent.evaluate_pending_info()
    return agent.q_table['ai'] - q_before


def test_network_confirmation_reference():
    cell = (25, 25)  # far from most agents → remote, uncontested by sensing
    # Default: network reference → confirming the NETWORK is rewarded even
    # though the report (4) contradicts the exploiter's own prior (0)
    model = build_model()
    assert model.confirmation_reference == 'network', "default must be 'network'"
    agent = _setup_exploiter_with_consensus(model, cell, net_level=4)
    net_level, net_conf = agent.get_network_consensus(cell)
    assert net_level == 4 and net_conf >= 0.3, (
        f"test setup failed: consensus {net_level} conf {net_conf:.2f}")
    dq_network = _feed_and_evaluate(model, agent, cell, reported=4, stored_prior=0)
    assert dq_network > 0, (
        f"network reference: report agreeing with the trusted network must be "
        f"rewarded, got ΔQ[ai]={dq_network:+.3f}")

    # Legacy: own-prior reference → the same report is punished
    model2 = build_model(confirmation_reference='own')
    agent2 = _setup_exploiter_with_consensus(model2, cell, net_level=4)
    dq_own = _feed_and_evaluate(model2, agent2, cell, reported=4, stored_prior=0)
    assert dq_own < 0, (
        f"own-prior reference: report contradicting the stored prior must be "
        f"punished, got ΔQ[ai]={dq_own:+.3f}")
    print(f"PASS: exploiter confirmation reference — network ΔQ[ai]={dq_network:+.3f}, "
          f"own ΔQ[ai]={dq_own:+.3f}")


def test_stochastic_rounding_linear_dose():
    model = build_model(ai_alignment_level=0.5)
    ai = next(iter(model.ais.values()))
    caller = next(iter(model.humans.values()))
    interest = (15, 15)
    cells = model.grid.get_neighborhood(interest, moore=True, radius=2,
                                        include_center=True)
    # Force a clean construction: AI senses 0 everywhere, caller believes 5
    for cell in cells:
        ai.sensed[cell] = 0
        caller.beliefs[cell] = {'level': 5, 'confidence': 0.9}
    # raw = 0 + 0.5·(5−0) = 2.5 on every reported cell
    vals = []
    for _ in range(200):
        report = ai.report_beliefs(interest, 2, caller)
        vals.extend(report.values())
    vals = np.array(vals, dtype=float)
    assert set(np.unique(vals)) <= {2.0, 3.0}, (
        f"stochastic rounding of 2.5 must yield only 2 or 3, got {np.unique(vals)}")
    assert 2.35 < vals.mean() < 2.65, (
        f"stochastic rounding must be unbiased around 2.5, got mean {vals.mean():.3f}")

    model_det = build_model(ai_alignment_level=0.5, report_rounding='deterministic')
    ai_d = next(iter(model_det.ais.values()))
    caller_d = next(iter(model_det.humans.values()))
    for cell in cells:
        ai_d.sensed[cell] = 0
        caller_d.beliefs[cell] = {'level': 5, 'confidence': 0.9}
    report_d = ai_d.report_beliefs(interest, 2, caller_d)
    assert set(report_d.values()) == {2}, (
        f"deterministic rounding of 2.5 must yield np.round's 2, got "
        f"{set(report_d.values())}")
    print(f"PASS: stochastic rounding unbiased (mean {vals.mean():.3f} ≈ 2.5); "
          f"deterministic reproduces legacy np.round")


def test_population_metrics_recorded():
    model = build_model()
    for _ in range(6):  # crosses one metrics tick (tick % 5 == 0)
        model.step()
    assert model.seci_pop_data, "seci_pop_data empty after a metrics tick"
    assert model.aeci_ie_pop_data and model.seci_ie_pop_data, \
        "IE population series empty after a metrics tick"
    assert model.aeci_lockin_pop_data, "aeci_lockin_pop_data empty"
    tick, seci_pop = model.seci_pop_data[-1]
    assert -1.0 <= seci_pop <= 1.0
    _t, ie_belief, ie_chan = model.aeci_ie_pop_data[-1]
    for v in (ie_belief, ie_chan):
        assert np.isnan(v) or -1.0 <= v <= 1.0
    print(f"PASS: population metrics recorded (SECI-pop={seci_pop:+.3f} at tick {tick})")


if __name__ == '__main__':
    test_network_confirmation_reference()
    test_stochastic_rounding_linear_dose()
    test_population_metrics_recorded()
    print("All confirmation-reference / rounding / population-metric tests passed.")
