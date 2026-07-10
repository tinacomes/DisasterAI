# Final Mechanics Review — Pre-Submission Audit

> **ADDENDUM (2026-07-10, same day):** §2.2 and §4 of this review are now
> resolved — the design proposal has been implemented on this branch behind
> three flags defaulting to baseline behaviour (`mobility`, `network_type`,
> `query_scope`), with the §0 stored-communities refactor, broker metrics in
> the results JSON, CLI/workflow wiring, and seeded bitwise regression
> guards. See the status header of DESIGN_PROPOSAL_NETWORK_MOBILITY.md for
> validation results. Still outstanding from §5: the C8 α-noise confound,
> C9/C10 paper documentation, the remaining vestigial parameters
> (`share_confirming` etc. — `exploit_friend_bias`/`exploit_self_bias` are
> now removed), and the paper-scale Stage-4 re-run (now including the
> 3-switch sweep).

**Review date:** 2026-07-10
**Reviewed branch:** `claude/paper-mechanics-review-39dmm1` at `1c547bc`
**Method:** full read of the mechanism code paths, cross-check against
METHODS_PAPER.md / SUPPLEMENTARY.md, REVIEW_PROTOCOL.md, and
DESIGN_PROPOSAL_NETWORK_MOBILITY.md, plus fresh seeded smoke runs
(40 agents, 20×20 grid, 80–100 ticks, α ∈ {0, 0.5, 1}, seeds 1–3) to verify
the Q-learning loop, epsilon decay, and the salience counterfactual
empirically. Verification scripts were run against the current branch head.

This document answers three questions: (1) are all envisioned mechanism
changes implemented, (2) is the Q-learning mechanism working, (3) do the
community (friends) and spatial (movement) mechanisms support meaningful
spatial/network analytics — plus a list of remaining items that still
require change.

---

## 1. Verdict summary

| Question | Answer |
|---|---|
| REVIEW_PROTOCOL fixes (Stages 1–3) implemented? | **Yes** — A1, A2, B1–B7, C1–C6 all verified present in code |
| DESIGN_PROPOSAL (mobility, bridged network, network-gated queries) implemented? | **No — none of it.** The proposal is still status PROPOSAL |
| Q-learning working? | **Yes** — verified empirically on this branch (§3) |
| Friends/communities distinguished? | **Partially** — friendship shapes *acceptance and evaluation* of information, but not *access* to it (§4.1) |
| Movement / spatial patterns distinguished? | **No** — agents are immobile; spatial analytics measure residual inequity only (§4.2) |
| Paper honest about the above? | Yes (METHODS last ¶, SUPPLEMENTARY S5.5 caveat) — the paper matches the code as-is |
| Anything else requiring change? | Yes — C8 α-noise confound (code), C9/C10 undocumented assumptions (paper), Stage-4 paper-scale re-validation, minor cruft (§5) |

**Bottom line.** The codebase is internally consistent with the paper *as
currently written*: the paper explicitly frames periphery gaps as
"structurally narrow" and discloses immobility. But if the goal of the final
paper is meaningful spatial and network-brokerage analytics — which is what
DESIGN_PROPOSAL_NETWORK_MOBILITY.md was written to enable — then the three
mechanism changes in that proposal (mobility, spatially embedded bridged
network, network-gated queries) and the prerequisite SECI metric refactor
are **all still outstanding**. A decision is needed: either implement the
proposal (and re-run the sweeps), or keep the current honest-but-narrow
periphery framing and drop brokerage claims.

---

## 2. Are all envisioned changes implemented?

### 2.1 REVIEW_PROTOCOL items — implemented and verified

Spot-checked in code on this branch (not taken on faith from the status
notes):

- **A1** (triangulation Q-drift): fixed — smoke runs show all Q-values in
  [−0.46, 0.69], inside the reward range; no unbounded drift (§3).
- **A2** (explorer oracle → situation-report verification): present —
  `verification_probability` parameter, 3-tick lag, geometric arrival,
  30-tick expiry with no fallback; documented in METHODS ¶6.
- **B1** (disaster dynamics): `update_disaster` implements drift-toward-
  baseline + patch shocks reading `shock_probability`/`shock_magnitude`.
- **B3** (`==` → `>=`): fixed at the stored-prior branch.
- **B4** (x/y transposition): fixed; `test_filter_bubbles.py` now indexes
  consistently and the stale "agents move every tick" comment is corrected
  to "Agents are immobile" (line ~411).
- **B5/B6/B7**: fixed (AECI-variance guard, shape unpacking, seeded
  `simulate.py` with `--seed_base`).
- **C1/C2** (AECI naming/sign, SECI formula): unified — AECI-Acc/-Err/-Var
  naming and negative-=-echo convention consistent across model, pipeline,
  plots, and both paper files; JSONs carry the `aeci_err_sign` marker.
- **C3/C4/C6** (MAE/precision/unmet-needs definitions): paper matches the
  pipeline definitions; SUPPLEMENTARY S5.0 metrics table present.
- **C5** (AI-reliant classification): both AECI-Err and AECI-Var median-split
  by `cum_accepted_ai`, within type; α*-sensitivity analysis implemented.
- **C12** (salience counterfactual): `salience_weight` parameter live and
  behaviourally effective (verified fresh, §3.3).
- Epsilon decay (recent commits deabac5–1c547bc): implemented and working
  (§3.2), default 1.0 = exact baseline behaviour; compare workflow exists.

### 2.2 DESIGN_PROPOSAL_NETWORK_MOBILITY items — NOT implemented

Verified by direct code inspection (grep + read of the relevant functions):

| Proposal item | Status in code |
|---|---|
| §0 prerequisite: SECI/rumor refactor to stored `model.communities` | **Not done** — `calculate_seci`, rumor assignment, and the component metrics still iterate `nx.connected_components` (DisasterAI_Model.py:3056, 3104, 3128, 3186, 2373, 2700, 2723). Safe only because there are still zero bridges |
| §1 Mobility (returners/explorers, `mobility` flag, `home_pos`, `r_home`, `r_explore`) | **Not done** — no `grid.move_agent` call anywhere; `pos` set once at spawn (`:2354`); no mobility parameter exists |
| §2 Spatial caveman network with weak-tie bridges (`network_type`, `p_bridge`, `bridge_decay`, spatial centroids) | **Not done** — network construction still builds type-homogeneous communities with "NO connections BETWEEN communities" (`:2697`); no bridge/broker code or flags |
| §3 Network-gated queries (`query_scope`) | **Not done** — the human-mode candidate pool is still *all* humans with non-friends at the 0.05 baseline weight (`seek_information`, `:1487–1499`); no friends-of-friends mechanism |
| §3 cleanup: remove dead `exploit_friend_bias` | **Not done** — still accepted and stored (`:2209`, `:2256`), never used |
| §4 validation predictions (H-P1–H-P3), broker flag in results JSON | **Not done** — nothing to validate yet |

None of the proposal's parameters (`mobility`, `network_type`,
`query_scope`, `p_bridge`, `p_within`, `spawn_sigma`) exist anywhere in the
codebase. The proposal header still says "nothing in this document is
implemented yet", and that is accurate.

---

## 3. Is the Q-learning mechanism working? — Yes (verified)

Fresh seeded smoke runs on this branch (40 agents, 20×20, 80 ticks,
seeds 1–2 per condition; script archived in the session scratchpad):

| α | type | Q(self) | Q(human) | Q(ai) | Q range | AI picks (explore/exploit) |
|---|---|---|---|---|---|---|
| 0.0 | exploitative | 0.25/0.16 | 0.27/0.38 | **0.14/0.33** | [−0.38, 0.61] | 147–181 / 327–473 |
| 0.5 | exploitative | 0.28/0.18 | 0.20/0.38 | 0.13/0.34 | [−0.43, 0.65] | 148–154 / 334–507 |
| 1.0 | exploitative | 0.22/0.17 | 0.39/0.42 | **0.46/0.35** | [−0.46, 0.69] | 148–164 / 621–664 |

Findings:

1. **The loop executes and is bounded.** All Q-values across every run stay
   within [−0.46, 0.69] — inside the designed reward range, confirming the
   A1 fix holds (no additive drift; before the fix Q-values exceeded 1.1).
2. **The intended exploiter asymmetry is present.** Exploiter Q(ai) rises
   with α (mean ≈ 0.24 at α = 0 → ≈ 0.41 at α = 1), and exploiters' greedy
   (exploitation-branch) AI selections roughly double (≈ 400 → ≈ 640 per
   run): confirmation-seeking agents learn to prefer the confirming AI.
3. **Explorer flatness at s = 0 is the documented C12 finding, not a bug.**
   Explorer Q(ai) is α-insensitive under uniform verification (0.39 → 0.44
   across α = 0 → 1) because verification reward is base-rate dominated —
   exactly the mechanism diagnosed in REVIEW_PROTOCOL C12.

### 3.2 Epsilon decay (new mechanism, commits deabac5–1c547bc)

Verified live: at `epsilon_decay = 1.0` (default) the exploration share of
mode choices is 0.31 (= ε = 0.3, as designed); at `epsilon_decay = 0.98`
it anneals to 0.13–0.15 over 80 ticks with the 0.05 floor respected. The
`mode_choice_counts` instrumentation (exploration vs exploitation tallies)
records correctly. Note the paper currently states constant ε = 0.3
(METHODS ¶5) — if the compare-epsilon-decay experiment leads you to adopt
decay for the headline runs, METHODS must be updated accordingly.

### 3.3 Salience counterfactual (C12) — mechanism confirmed

Fresh 3-seed runs, 100 ticks: explorer Q(ai) as α goes 0 → 1 moves
**+0.39 → +0.44 at s = 0** (confirming AI mildly preferred) but
**0.35 → 0.22 at s = 1** with AI-trust 0.85 → 0.77 — the severity-weighted
verification flips the gradient sign as intended. The C12 headline
mechanism ("a confirming AI stays trusted by being right about the
unimportant majority") is fully reproducible on this branch.

---

## 4. Communities (friends) vs spatial patterns (movement)

### 4.1 What the friend/community machinery does — and does not — do

**Implemented and working (acceptance/evaluation side):**
- Friends get higher initial trust and a higher trust-decay anchor
  (0.6 vs 0.35 for non-friends/AI) — social loyalty persists.
- Friends get the trust-widened acceptance window
  `D_eff = D·(1 + 0.5·trust)` (`:1210`) — friend info is easier to accept.
- Exploiters' confirmation reference uses `get_network_consensus` over
  friends only (`:664–685`), and the confirming AI targets the *network
  consensus* for exploitative callers (`:2126–2136`) — the community's
  shared narrative is what gets amplified.
- SECI is computed per type-homogeneous community, so the social
  echo-chamber construct is well-defined.

**Not implemented (access side):** whom an agent can *reach* is unaffected
by the network. The human-query candidate pool is the entire population with
non-friends at a 0.05 baseline weight (`:1490–1499`). With ~85 non-friends
× 0.05 ≈ 4.3 aggregate weight versus ~6 friends × ~0.5 ≈ 3, roughly **half
of human queries go to random strangers** — network topology does not gate
information access, which is precisely what DESIGN_PROPOSAL §3 was written
to change. Consequence: betweenness/degree "network periphery" analytics
have no causal channel to operate through, and with zero between-community
edges there are no brokers by construction (every agent in a p = 0.7
near-clique has statistically indistinguishable betweenness — the Q1/Q4
betweenness split in `test_filter_bubbles.py` compares noise).

### 4.2 Spatial patterns

There is **no movement code**: `grid.move_agent` is never called; `pos` is
assigned once at spawn (`:2354`) and `initial_pos == pos` for the whole
run. Sensing radius is 0. Meanwhile query targeting, AI sensing (uniform
random 15% of the grid), and relief targeting all operate grid-wide
independently of position. An agent's spawn location therefore affects
exactly 1 of 900 belief cells directly. The spatial near/far-spawn
analytics are *computed correctly* (B4 fixed) but measure only residual
inequity from local sensing and initial beliefs — as SUPPLEMENTARY S5.5
now honestly states.

### 4.3 So: will the spatial analytics give meaningful results?

**As the code stands, no — by construction.** Both periphery axes are
nearly null: geography barely constrains information access, and network
position doesn't constrain it at all. Expect small, noisy periphery gaps
whose α-dependence is weak. The paper currently *survives* this by scoping
the claim down (the "structural scope … is deliberately narrow" sentence),
but that also means the equity/brokerage questions (H-P1–H-P3 in the
proposal) **cannot be answered by this model version**. Two coherent
options:

- **Option A (bigger paper): implement the proposal.** Follow its own
  5-step validation sequence, in order: (1) stored-communities metric
  refactor — behaviour-neutral, safe to land immediately and *required*
  before any bridges exist, since `calculate_seci` and rumor seeding
  silently mislabel mixed components; (2) `query_scope='network'`;
  (3) bridged spatial network + broker flag; (4) mobility; (5) full
  3-switch sweep. All behind flags defaulting to current behaviour, with
  the α = 0 regression guard the proposal specifies. This changes headline
  results and requires the full paper-scale re-run.
- **Option B (current paper): keep the narrow framing.** Then (i) consider
  dropping or demoting the betweenness-quartile "broker" panels — in a
  bridge-less near-clique they can only show noise and invite reviewer
  attack; (ii) keep the spatial near/far panels with the S5.5 caveat;
  (iii) delete `exploit_friend_bias` and the proposal's promises from any
  paper text. No re-run needed.

The one thing that is *not* coherent is the current in-between: a results
pipeline that computes and plots broker/betweenness gaps on a network that
cannot produce them.

---

## 5. What else still requires change

Ordered by importance.

1. **(Code, results-affecting) C8 — α is confounded with AI sensing noise.**
   `AIAgent.sense_environment` still sets `noise_prob = 0.1 · α`
   (`:1990–1997`): high-α AI is simultaneously more confirming *and*
   noisier, while METHODS ¶ "AI Alignment" claims α "isolates the
   confirmation effect from other AI failure modes". One-line fix
   (constant `noise_prob = 0.1` or 0) + re-run, or a disclosure sentence in
   METHODS. This is the only remaining *mechanism* defect from the review
   protocol.
2. **(Paper) C10 — the network-consensus confirmation target is
   undocumented and contradicts the paper's α formula.** METHODS states
   `r_AI = (1−α)t + αb` with *b* = "the querying agent's current belief",
   but for exploitative callers the code substitutes the *friend-network
   consensus* belief (`:2126–2136`). The code comments call this
   load-bearing for H1. It must be stated and justified in METHODS (the
   social-media-platform analogy), and the formula text amended. Likewise
   METHODS says `D_eff` trust-widening applies to "social contacts" — in
   code it applies to *friends only* (`:1210`).
3. **(Paper) C9 — AI guessing on unsensed cells is undocumented.** Each AI
   senses 15% of the grid but answers ~75% of queries about unsensed cells
   with interpolated/fabricated estimates, evaluated by humans at full
   `source_knowledge_conf = 1.0`. At α = 0 the "truthful" AI is truthful
   only for sensed cells. Document as a deliberate AI-overconfidence
   feature, or restrict reports to sensed cells (results-affecting).
4. **(Process) Stage 4 re-validation has not been done at paper scale on
   this branch.** All numbers above are toy-scale. The three open
   behavioural checks from REVIEW_PROTOCOL §6.16 — (a) exploiter SECI more
   negative at high α (the 60-tick probe showed the opposite), (b) explorer
   α-responsiveness under s ∈ {0, 1}, (c) α* stability across composite
   variants — must be confirmed with the 200-tick / N = 100 / 20-rep sweep
   *after* whatever is decided on items 1 and §4.3, since every re-run
   invalidates cached sweep JSONs (the C5 classification change alone
   already requires a fresh sweep for final numbers, per the Stage-2 note).
5. **(Decision) Epsilon decay: pick one setting for the paper.** The
   mechanism is implemented and off by default; the compare workflow
   exists. Either keep constant ε = 0.3 (paper unchanged) or adopt decay
   and update METHODS ¶5 — don't leave the branch ambiguous.
6. **(Cruft, cosmetic) Vestigial constructor parameters** are still
   accepted-and-ignored: `share_confirming` (set to 0.7 by every caller,
   used in **no** logic — the most misleading one), `lambda_parameter`,
   `low_trust_amplification_factor`, `exploitative_correction_factor`,
   `trust_update_mode`, `exploit_friend_bias`, `exploit_self_bias`. Remove
   in one batch across model + simulate.py + test_filter_bubbles.py +
   notebooks. Also: `AIAgent.memory` still grows unboundedly (prune keys
   older than 1 tick), and the stale comment at `HumanAgent.__init__:98`
   ("Seems used in removed code? Check usage.") should go with the batch.

---

## 6. Suggested order of work

1. Decide §4.3 Option A vs B — everything else sequences behind this.
2. Fix C8 (one line) and make the C9/C10/D_eff paper edits (text only).
3. If Option A: land the stored-communities refactor first (behaviour-
   neutral), then the three flagged mechanisms in the proposal's order.
   If Option B: drop the broker panels, delete `exploit_friend_bias`.
4. Batch-remove vestigial parameters (item 5.6).
5. Single paper-scale re-run (primary α sweep + gap sweep + s ∈ {0,1});
   regenerate every figure from post-fix JSONs only; then the Stage-5
   line-by-line paper reconciliation.
