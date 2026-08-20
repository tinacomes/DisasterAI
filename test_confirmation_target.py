#!/usr/bin/env python3
"""Smoke tests for the AI confirmation target.

Design contract: the AI must treat exploratory and exploitative callers
IDENTICALLY — it adapts only to the caller's revealed prior beliefs via the
alignment parameter α, never to the caller's cognitive type (a behavioural
pattern that neither an AI nor a human interlocutor can observe).

Tests:
  1. individual (default), α=1: the AI report equals the caller's own current
     belief on every reported cell, for BOTH agent types.
  2. individual, α=1: masking the caller's agent_type does not change the
     report — the AI response is type-agnostic.
  3. consensus (legacy reproduction flag), α=1: exploitative callers receive
     the trusted-network consensus where defined (conf >= 0.2), their own
     prior otherwise; exploratory callers always receive their own prior.
  4. AECI-IE (M1 information-environment index) construct checks:
     a. α=1 with a fully converged community (all members hold the same
        level everywhere): the AI serves back that single level, the served
        pool has zero variance → AECI-IE ≈ −1 for that community's type.
     b. α=0: the AI serves (noisy) truth at scattered interest points; the
        served pool carries the truth's diversity → |AECI-IE| stays far from
        the echo-chamber pole (no truth-convergence confound).

Run: python3 test_confirmation_target.py
"""
import random

import numpy as np

from DisasterAI_Model import DisasterModel, AIAgent

BASE = dict(
    share_exploitative=0.5, share_of_disaster=0.15, initial_trust=0.3,
    initial_ai_trust=0.25, number_of_humans=100, share_confirming=0.7,
    disaster_dynamics=2, width=30, height=30, ticks=10,
    learning_rate=0.1, epsilon=0.3, exploit_trust_lr=0.015,
    explor_trust_lr=0.03, ai_alignment_level=1.0,
    mobility=1, network_type='spatial_bridged', query_scope='network',
)


def build_model(confirmation_target, seed=42, warmup_ticks=8):
    random.seed(seed)
    np.random.seed(seed)
    model = DisasterModel(confirmation_target=confirmation_target, **BASE)
    for _ in range(warmup_ticks):  # let beliefs/trust differentiate
        model.step()
    return model


def one_agent_of(model, agent_type):
    return next(a for a in model.humans.values() if a.agent_type == agent_type)


def get_report(model, caller):
    ai = next(iter(model.ais.values()))
    interest_point = (15, 15)
    return ai.report_beliefs(interest_point, query_radius=2, caller=caller), ai


def test_individual_alpha1_confirms_own_prior():
    model = build_model('individual')
    for agent_type in ("exploitative", "exploratory"):
        caller = one_agent_of(model, agent_type)
        report, _ = get_report(model, caller)
        assert report, f"empty AI report for {agent_type} caller"
        for cell, level in report.items():
            prior = caller.beliefs.get(cell, {'level': 0}).get('level', 0)
            assert level == int(prior), (
                f"individual/α=1: AI reported {level} but {agent_type} caller's "
                f"own prior at {cell} is {prior}")
    print("PASS: individual target, α=1 → report equals caller's own prior (both types)")


def test_individual_is_type_agnostic():
    model = build_model('individual')
    caller = one_agent_of(model, "exploitative")
    state = random.getstate(); np_state = np.random.get_state()
    report_as_exploiter, _ = get_report(model, caller)
    random.setstate(state); np.random.set_state(np_state)
    caller.agent_type = "exploratory"  # mask the type; same RNG stream
    try:
        report_masked, _ = get_report(model, caller)
    finally:
        caller.agent_type = "exploitative"
    assert report_as_exploiter == report_masked, (
        "individual target: AI report changed when the caller's type was "
        "masked — the AI response must not condition on agent_type")
    print("PASS: individual target → AI response independent of caller type")


def test_consensus_flag_reproduces_legacy_targeting():
    model = build_model('consensus')
    caller = one_agent_of(model, "exploitative")
    report, _ = get_report(model, caller)
    assert report, "empty AI report for exploitative caller"
    n_consensus = 0
    for cell, level in report.items():
        prior = caller.beliefs.get(cell, {'level': 0}).get('level', 0)
        net_level, net_conf = caller.get_network_consensus(cell)
        if net_level is not None and net_conf >= 0.2:
            expected = int(round(net_level))
            n_consensus += 1
        else:
            expected = int(prior)
        assert level == expected, (
            f"consensus/α=1: AI reported {level} at {cell}, expected {expected} "
            f"(prior={prior}, consensus={net_level}, conf={net_conf:.2f})")
    assert n_consensus > 0, "no cell had a defined network consensus — test vacuous"
    # Exploratory callers always get their own prior, even under the legacy flag
    explorer = one_agent_of(model, "exploratory")
    report_er, _ = get_report(model, explorer)
    for cell, level in report_er.items():
        prior = explorer.beliefs.get(cell, {'level': 0}).get('level', 0)
        assert level == int(prior)
    print(f"PASS: consensus flag reproduces legacy targeting "
          f"({n_consensus} consensus-targeted cells)")


def _serve_ai_reports_to_community(model, community_idx=0, n_points=8):
    """Have every member of one stored community receive real AI reports at
    scattered interest points, logging them exactly as seek_information does.
    Returns the community's type label."""
    ai = next(iter(model.ais.values()))
    nodes, community_type = model.communities[community_idx]
    # Scattered interest points spanning the grid (deterministic pattern)
    pts = [(3 + 5 * (i % 5), 3 + 5 * (i // 5)) for i in range(n_points)]
    for node in nodes:
        agent = model.humans.get(f"H_{node}")
        if agent is None:
            continue
        for p in pts:
            report = ai.report_beliefs(p, query_radius=2, caller=agent)
            for cell, level in report.items():
                agent.received_report_log['ai'].append((cell, int(round(level))))
    return community_type


def _ie_value_for_type(result, community_type):
    key = 'exploitative' if community_type == 'exploitative' else 'exploratory'
    return result['aeci_ie'][key]


def test_aeci_ie_converged_caller_alpha1():
    """α=1 + fully converged community → AI serves one level → AECI-IE ≈ −1."""
    model = build_model('individual')          # BASE has ai_alignment_level=1.0
    nodes, community_type = model.communities[0]
    # Clear residual logs everywhere so only the served community contributes
    # to its type's per-community average
    for agent in model.humans.values():
        agent.received_report_log = {'human': [], 'ai': []}
    # Converge every member's beliefs on a single level everywhere
    for node in nodes:
        agent = model.humans.get(f"H_{node}")
        if agent is None:
            continue
        for cell in agent.beliefs:
            if isinstance(agent.beliefs[cell], dict):
                agent.beliefs[cell]['level'] = 3
                agent.beliefs[cell]['confidence'] = 0.9
    _serve_ai_reports_to_community(model, 0)
    result = model.calculate_ie_indices()
    val = _ie_value_for_type(result, community_type)
    assert not np.isnan(val), "AECI-IE undefined despite served reports"
    assert val < -0.95, (
        f"α=1 with a converged community must yield AECI-IE ≈ −1 "
        f"(served pool has zero variance), got {val:.3f}")
    print(f"PASS: AECI-IE ≈ −1 for converged caller at α=1 (value {val:.3f})")


def test_aeci_ie_truthful_alpha0():
    """α=0 → AI serves (noisy) truth → served diversity ≈ global belief
    diversity → AECI-IE far from the echo-chamber pole."""
    random.seed(42)
    np.random.seed(42)
    model = DisasterModel(confirmation_target='individual',
                          **{**BASE, 'ai_alignment_level': 0.0})
    for _ in range(8):
        model.step()
    for agent in model.humans.values():
        agent.received_report_log = {'human': [], 'ai': []}
    community_type = _serve_ai_reports_to_community(model, 0)
    result = model.calculate_ie_indices()
    val = _ie_value_for_type(result, community_type)
    assert not np.isnan(val), "AECI-IE undefined despite served reports"
    assert val > -0.5, (
        f"α=0 (truthful AI) must not read as a deep AI echo chamber; "
        f"expected AECI-IE well above −0.5, got {val:.3f}")
    print(f"PASS: AECI-IE stays near 0 for truthful AI at α=0 (value {val:.3f})")


if __name__ == '__main__':
    test_individual_alpha1_confirms_own_prior()
    test_individual_is_type_agnostic()
    test_consensus_flag_reproduces_legacy_targeting()
    test_aeci_ie_converged_caller_alpha1()
    test_aeci_ie_truthful_alpha0()
    print("All confirmation-target smoke tests passed.")
