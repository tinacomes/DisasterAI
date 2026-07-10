# Design Proposal: Spatially Embedded Network, Mobility, and Network-Gated Queries

Status: **IMPLEMENTED** (2026-07-10) behind three switches that all default to
the pre-proposal behaviour: `mobility=0`, `network_type='components'`,
`query_scope='global'`. Seeded regression fingerprints are bitwise identical
to the pre-implementation code at the defaults (under a fixed
`PYTHONHASHSEED`, which the model already required for reproducibility).
Implementation notes:

- §0 prerequisite refactor: done — `model.communities` (list of
  `(set[node_id], type_label)`) is stored at network construction; SECI,
  rumor seeding, and the component-level metrics iterate it instead of
  `nx.connected_components`.
- §1 mobility: done — `HumanAgent.move()`, phase 0 of `step()`. Toy-scale
  validation (30 × 30, 200 ticks): returners cover ~27 cells within max
  excursion radius 3.0; explorers cover ~92 (55–125) cells, home-anchored.
- §2 network: done — `initialize_spatial_bridged_network()`. Validation
  (N = 100, 3 seeds): ~305 edges, mean degree ≈ 6, 8 type-pure communities,
  13–20 bridges; betweenness heavy-tailed (max/median 13–32×; bridge
  endpoints 7–17× non-endpoints). Broker flag (`model.bridge_endpoints`)
  recorded into results JSON (`broker_mae`/`nonbroker_mae` etc.).
- §3 queries: done — `HumanAgent.select_human_source()`; at
  `query_scope='network'`: 92 % friends / 8 % two-hop / 0 % strangers in a
  seeded smoke run, Q(human) still learns for all agents. Dead
  `exploit_friend_bias` / `exploit_self_bias` parameters removed.
- §4 validation: H-P1 confirmed directionally at toy scale (far−near
  disaster-cell MAE gap +0.17 at α = 0 → +0.24 at α = 1 with all three
  switches on); H-P2 within noise at 3 seeds — decide at paper scale
  (step 5 below, still outstanding).

The paper-scale 3-switch sweep (validation step 5) runs via the
`run-primary-sweep` workflow inputs `mobility` / `network_type` /
`query_scope`.

Original proposal follows.
Scope: answers three design questions — (1) how agents should move, (2) what
social-network model supports genuine brokerage and periphery analysis at this
model's scale (30 × 30 grid, 100 humans, 200 ticks, N = 20 replications), and
(3) whether the query mechanism must change for network position to matter.

---

## 0. Why the current design cannot produce periphery/brokerage effects

Diagnosis from the code (see also REVIEW_PROTOCOL A3 and SUPPLEMENTARY S5.5
caveat):

| Mechanism | Current state | Consequence |
|---|---|---|
| Movement | None — `pos` set once at spawn | Spawn distance affects 1 of 900 belief cells |
| Sensing | Radius 0 (own cell only) | Same |
| Human query pool | ALL humans; friends trust-weighted, non-friends at 0.05 baseline (`DisasterAI_Model.py` `seek_information`) | Topology does not gate access |
| AI sensing | Uniform random 15% of grid | Position-independent |
| Relief targeting | Whole belief grid | Position-independent |
| Network | ~25-agent communities, p = 0.7 within, **zero** edges between | Near-cliques, diameter ≈ 2: betweenness/degree variation is noise; no brokers exist by construction |

The friend machinery that *does* exist (higher initial trust, trust-decay
anchor 0.6 vs 0.35, trust-widened acceptance window `D_eff`, friends-only
`get_network_consensus`) shapes **acceptance and evaluation** of reports, but
not **access** to them. `exploit_friend_bias` is accepted as a parameter and
never used (flagged in REVIEW_PROTOCOL for removal).

Design principle for the redesign: **an agent's information access should be
constrained by (a) where it is and (b) whom it knows — and the AI channel
should be the one mechanism that bypasses both.** That makes the equity
question crisp: does truthful AI (α = 0) close the periphery gap that geography
and network structure create, and does confirming AI (α = 1) re-open it?

---

## 1. Mobility: home-anchored returners and explorers

The mobility literature distinguishes two robust profiles — *returners*, whose
movement is dominated by recurrent trips among a few anchor locations, and
*explorers*, who wander among many locations (Pappalardo et al. 2015, *Nature
Communications*; confirmed under natural-hazard conditions in Scientific
Reports 2024). This maps directly onto the model's two cognitive types:

- **Exploitative agents = returners.** Random walk within radius
  `r_home = 3` of `home_pos` (their spawn cell), with a return-home bias
  (step toward home with p = 0.3 when outside radius). Coverage over a run:
  ~25–30 cells around home.
- **Exploratory agents = explorers.** Larger radius of gyration: with
  p = 0.2 per tick, take an excursion step toward their current
  `find_highest_uncertainty_area()` target (capped at `r_explore = 8` from
  home before the return bias kicks in); otherwise local random walk.
  Coverage: ~100–200 cells, but still home-anchored.

Implementation notes:

- Keep sensing radius 0. Movement *is* the sensing mechanism: a far-spawned
  returner genuinely cannot observe the disaster, which is what makes spatial
  periphery real. (The old rationale for radius 0 — "100 agents × r = 2 covers
  94% of the grid per tick" — is solved by movement, not undone by it.)
- `initial_pos` (= home) stays the spatial-periphery classifier, consistent
  with the metrics already in `test_filter_bubbles.py`.
- Movement executes in `HumanAgent.step()` phase 0 (before `phase_observe`),
  one `model.grid.move_agent` call per agent per tick — O(N) per tick,
  negligible against the existing belief-dictionary loops.
- Relief dispatch stays global (tokens represent remote aid requests, not
  physical delivery by the agent). One mechanism change at a time.
- Disaster interaction: no change needed to `update_disaster`; mobile agents
  now discover shocks by covering ground, which finally exercises the
  "non-stationary environment forces ongoing information-seeking" premise.

Parameter: `mobility ∈ {0, 1}` (default 1 after validation; 0 reproduces the
current immobile behaviour for regression comparison).

---

## 2. Network: spatially embedded communities with weak-tie bridges

### Recommendation

A **spatial caveman graph with distance-decayed bridges** — communities are
spatial neighbourhoods; a small fraction of agents carry long-range "weak tie"
links that are the only routes between communities:

1. **Spatial communities.** Per agent type, place `K = 4` community centroids
   on the grid (8 communities total, ~12–13 agents each; centroids sampled
   with a minimum-separation constraint). Agents spawn Gaussian around their
   community centroid (σ = 2.5 cells). This couples network membership to
   geography, as in real settlements, and ties the two periphery notions
   (spatial and network) together instead of leaving them independent.
2. **Within-community edges.** Erdős–Rényi with `p_within = 0.5` →
   mean within-degree ≈ 6 (a Dunbar-scale support clique rather than the
   current ~17). Communities remain type-homogeneous, which the SECI
   construct requires.
3. **Weak-tie bridges.** Each agent independently becomes a *bridge agent*
   with `p_bridge = 0.15` (~15 agents per run). A bridge agent draws ONE
   extra edge to an agent in another community, with endpoint probability
   ∝ (1 + d)^(−2) where d is grid distance (Kleinberg's decay exponent for
   2D lattices, which maximises decentralised routing efficiency). Bridges
   are type-agnostic: most land same-type nearby, a few cross type — rare
   conduits between the exploitative and exploratory worlds.

### Why this specific construction

- **Brokers exist by construction and are identifiable.** Bridge endpoints
  are structural-hole spanners in the sense of Burt (1992, 2004): they sit on
  essentially *all* shortest paths between their two communities, so
  betweenness becomes heavy-tailed instead of uniform (currently every agent
  in a p = 0.7 clique has ≈ the same betweenness). Granovetter's weak-tie
  argument (1973) is literally the mechanism: novel information about distant
  parts of the disaster can only enter a community through a bridge.
  Analysis gains a clean binary *broker flag* (has-bridge vs not) alongside
  the noisy betweenness quartiles.
- **Periphery exists on two coupled axes.** Communities whose centroid is far
  from the epicentre are spatially peripheral; agents without bridges in
  sparsely-bridged communities are network-peripheral. The empirical disaster
  literature consistently finds moderate core–periphery structure in response
  networks (JST *Journal of Disaster Research* 2025; Nowell et al. 2018 on
  wildfire governance networks), so this is descriptively defensible, not
  just analytically convenient.
- **Small-world properties at negligible cost.** High clustering within
  communities + short global paths via bridges (Watts & Strogatz 1998) with
  ~315 edges on 100 nodes. Generation is O(N·K); exact
  `nx.betweenness_centrality` on 100 nodes runs in milliseconds once per run.
  The 900-cell grid constrains the *belief* loops, not the network — nothing
  about this design moves the runtime needle.

### Alternatives considered (and why not)

| Model | Brokers? | Communities (SECI)? | Spatial embedding? | Verdict |
|---|---|---|---|---|
| Current (disconnected caveman) | No — no bridges | Yes | No | Cannot answer the brokerage question |
| Watts–Strogatz ring | Shortcut endpoints, weakly | No natural communities | No | Breaks SECI; unembedded |
| Barabási–Albert scale-free | Hubs ≠ brokers (degree ≠ betweenness role); no clustering | No | No | Wrong mechanism; implausible for offline disaster comms |
| LFR benchmark | Tunable via mixing μ | Yes | No | Overkill; no geography; harder to seed reproducibly |
| Stochastic block model | Yes (rare cross-block edges) | Yes | No | Recommended design ≈ SBM + geography; plain SBM loses the spatial coupling |

### ⚠ Required metric refactor (do this first)

`calculate_seci` (and rumor assignment at init, and `calculate_component_seci`)
iterate `nx.connected_components(self.social_network)` and assume each
component is type-pure — with bridges, the graph becomes (nearly) one
connected component and both silently break (a mixed component gets labelled
by its first member's type). **Store explicit membership at construction**
(`model.communities: list[(set[int], type_label)]`) and compute SECI and rumor
assignment over stored communities, never over connected components. This
refactor is safe to land before any network change — with zero bridges it is
exactly equivalent to the current behaviour.

Parameters: `network_type ∈ {'components', 'spatial_bridged'}` (default
'components' until validated), `p_within = 0.5`, `p_bridge = 0.15`,
`bridge_decay = 2.0`, `n_communities_per_type = 4`, `spawn_sigma = 2.5`.

---

## 3. Queries: yes, they must change

For brokerage to exist, **access must follow edges**. Today the human-mode
candidate pool is every human in the population (non-friends at a 0.05
baseline weight, introduced so Q-learning could explore the "human" category).
That baseline is precisely what deletes the network from the model: with ~85
non-friends × 0.05 ≈ 4.25 total baseline weight against ~6 friends × ~0.5 ≈ 3,
roughly *half* of human queries go to random strangers.

Proposed replacement (`query_scope ∈ {'global', 'network'}`, default
'network' after validation):

1. **Default: friends only**, trust-weighted exactly as the existing friend
   branch already does. This is the "friend version" that exists in the code
   but currently competes with the global pool.
2. **Exploration without teleportation: friends-of-friends.** With
   p = 0.1 per human query, draw from the 2-hop neighbourhood instead
   (uniform). This preserves the Q-learning exploration rationale while
   keeping reach socially plausible — reaching a stranger requires an
   intermediary, which is what makes brokers valuable. It also gives weak
   ties a natural activation mechanism (dormant-tie activation in crises is
   well documented in the disaster-communication literature).
3. **AI stays global.** That is the treatment: the AI channel is the only way
   to bypass both geography and topology. α then controls whether that bypass
   delivers truth to the periphery (α = 0) or echoes the periphery's own
   blind spots back at it (α = 1).
4. No change to report content (`report_beliefs` around the interest point),
   acceptance (`D_eff`), or trust dynamics — friendship already modulates
   those correctly.

Clean-up while there: remove the dead `exploit_friend_bias` parameter
(REVIEW_PROTOCOL housekeeping item).

---

## 4. Expected signatures and validation

Testable predictions once implemented (run the standard α sweep with
`mobility = 1, network_type = 'spatial_bridged', query_scope = 'network'`):

- **H-P1 (spatial gap opens).** Far-spawned agents show materially higher
  disaster-cell MAE than near-spawned agents at α = 1 (confirming AI can only
  echo what the periphery already believes); the gap shrinks toward α = 0.
- **H-P2 (broker advantage).** Bridge agents' MAE beats same-community
  non-bridge agents at high α; the advantage compresses as α → 0 because
  truthful AI substitutes for brokerage — a direct "AI flattens structural
  holes" test (cf. the arXiv "From the Periphery to the Center" brokerage
  dynamics literature).
- **H-P3 (aid follows knowledge).** Peripheral communities' aid-targeting
  precision degrades faster with α than core communities'.
- Sanity: SECI/AECI/MAE baselines at α = 0 should remain in the current
  ballpark with `mobility = 0, network_type = 'components',
  query_scope = 'global'` (regression guard — all three switches default off).

Validation sequence (each step is one PR-sized change with the smoke test):

1. Metric refactor: stored communities for SECI + rumors (behaviour-neutral).
2. `query_scope = 'network'` behind flag; verify Q(human) still learns.
3. Network generator behind flag; verify betweenness heavy tail, broker flag
   recorded into results JSON.
4. Mobility behind flag; verify coverage statistics per type match the
   returner/explorer design.
5. Full 3-switch sweep at paper scale; compare periphery_gap and
   periphery_gap_evolution figures against the current baseline.

Runtime estimate: steps 2–4 add < 5% to a 200-tick run (movement O(N)/tick,
network generation once, queries unchanged in count). The dominant cost
remains the 900-cell belief dictionaries.
