# Revised storyline (paragraph by paragraph) and venue assessment

*2026-08-26. Basis: `PNAS_WRITING_INSTRUCTIONS.md`, `RESULTS_COMPENDIUM.md`
(canonical run `results-mechfix/`, run 32821105202), and the completed
M2–M7 validation chain. This document is the narrative contract for
`DisasterAIFilter_PNAS.tex`; every number cited here traces to the
compendium.*

## What changed relative to the NHB storyline

The NHB draft's story was: *network-bounded access is a precondition for
AI-amplified echo chambers; a truthful AI builds an AECI-Var bubble of its
own; an interior optimum exists only under bounded access, located by the
population's cognitive profile; harms concentrate on the periphery.* Four
of its claims do not survive the mechanism revision and are replaced:

1. **"Consensus recycling" is dead.** The AI never targeted the community
   consensus for exploiters in the executed code; the ablation shows the
   network confirmation reference is behaviorally inert. The AI now has
   ONE response rule for all callers — r = (1−α)t + αb with b = the
   querier's own prior — and everything differential is human. The
   structural-precondition finding survives, but its mechanism is
   acceptance-and-verification behavior under bounded access, not an AI
   that "recycles consensus."
2. **AECI-Var is retired.** The "truthful AI builds a bubble of its own"
   claim rested on an index that is most negative at the truthful endpoint
   because it conflates convergence-on-truth with an echo chamber. The
   AI-side construct is now the channel-baseline information-environment
   index (AECI-IE-chan, per type and population): ≈0-diverse at α=0,
   narrow at α=1 — a true SECI parallel.
3. **The interior optimum is no longer "only under bounded access", and it
   is no longer located by the cognitive profile.** Under the corrected
   (linear) confirmation dose, α* is interior in BOTH configurations
   (α*=0.6 for 7/12 composites in the main model; 0.8–0.9 in the control)
   and interior in EVERY cell of the 132-cell cognitive-profile sweep —
   the old control corner solution and the profile-dependence were both
   artifacts of the near-step delivered dose. Fig. 3b is reframed from a
   dependence claim to a robustness claim.
4. **The story gains two new mechanisms and one counterfactual:**
   *starvation* (confirming AI empties the accuracy-seekers' belief pool
   113→29 rather than persuading anyone), *capture* (confirmation-seekers
   route 0.54→0.69 of queries to the AI with trust flat), and the
   *social-retrenchment counterfactual* (making disconfirmation salient
   removes capture but ejects confirmation-seekers back into their social
   chamber, deepening it −0.37→−0.52 and cutting precision 0.57→0.43 at
   the truthful endpoint — the trap has no purely informational exit).

The through-line of the paper, sharpened: **the harm of a sycophantic AI
at population scale is not persuasion but the silent reorganization of
who learns what from whom — and social network structure, not the AI
policy alone, decides which failure mode a society gets.**

## Main text, paragraph by paragraph

**Title (≤135 chars, declarative, no subtitle):**
*Belief-aligned AI silently starves accuracy-seekers and captures
confirmation-seekers in simulated disaster response* (≈110 chars).
Alternatives: *Social networks decide whether belief-aligned AI amplifies
or dissolves echo chambers in disasters*; *How far should AI agree with
its users? Alignment, echo chambers, and collective action in disasters.*

**Abstract (≤250 words).** (i) People increasingly ask AI, not each
other, what is happening; preference-trained systems drift toward
confirming user beliefs; evidence stops at dyads while echo chambers live
in networks. (ii) ABM of disaster response; single-parameter alignment
dose α from truth to confirmation of the querier's own prior; identical
seeds under network-bounded vs unrestricted access. (iii) Four results:
bounded access is the structural precondition for the confirmation-seekers'
community chamber (dissolves −0.45→+0.05 unrestricted; persists −0.39…−0.18
bounded) while accuracy-seeking communities converge under confirmation in
both; the accuracy cost falls entirely on accuracy-seekers via belief
starvation (MAE 0.54→1.74; pool 113→29) with an interior optimum α*=0.6;
the feedback loop captures confirmation-seekers (query share 0.54→0.69,
trust flat) and the salience counterfactual backfires into social
retrenchment; harms concentrate on the spatial periphery (MAE gap
+0.12→+0.33; aid gap −1.8→−6.6), absent in the immobile control.
(iv) Alignment harm at scale is a property of the sociotechnical system;
monitoring must target community-level convergence, not individual
accuracy.

**Significance (≤120 words, no acronyms).** Three sentences per the
instructions: delegation of situational awareness to AI, most
consequentially in disasters; a mildly confirming AI silently starves
accuracy-seekers and captures confirmation-seekers, with the social
network deciding whether community echo chambers survive; neither maximal
truthfulness nor confirmation is operationally optimal — crisis AI
alignment is a socio-technical, not purely technical, problem.

### Introduction (~900–1,100 words, 8 paragraphs)

- **¶1 Echo chambers.** Three sentences: fragmentation into segregated
  information spaces; definition; polarization/erosion of shared factual
  ground. (mahmoudi2024echo, jeon2024hearhere, cinelli2021echo,
  bail2018exposure, baumann2020modeling)
- **¶2 AI arrives; promise vs construction.** Delegation to AI
  (klingbeil2024trust, angrisani2026gaps); bridging promise
  (jeon2024hearhere); collision with preference-based training →
  sycophancy (christiano2017deep, ouyang2022training, sharma2023sycophancy,
  sharma2024generative, cheng2026sycophantic); rejection of disconfirming
  information erodes trust (glikson2020human); alignment paradox
  (west2025alignment, ekstrom2022self). Ends on the operative question:
  how far should a system align to be used without forfeiting correction?
- **¶3 Disasters as the extreme case.** Urgency, stakes, degraded
  verification (svenson1993time, mendonca2001decision,
  comes2020coordination, levin2012overcoming, sogaard2024evolution); AI
  already deployed in crisis information work (qadir2016crisis,
  reichstein2025early, acharya2025agentic; rest to SI Table S10); explicit
  extreme-case logic: sharpest measurable form; misallocation = unmet needs.
- **¶4 Beliefs form in networks.** One citation per mechanism: bounded
  confidence (hegselmann2002opinion), homophily (mcpherson2001birds),
  confirmation bias/selective exposure (paulus2022influence,
  barbera2015tweeting, paulus2024interplay), weak ties as corrective
  channel (granovetter1973strength); crises tighten each: retreat to close
  groups, infrastructure disruption, silos, premature consensus
  (comes2020coordination, holguin2012unique, pan2012crisis,
  driskell1991group, weick2005organizing).
- **¶5 Evidence gap + research question.** Dyadic human–AI loops amplify
  individual bias (glickman2024human); network structure reshapes
  collective accuracy in ways dyads cannot reveal (lorenz2011social,
  becker2017network). Question: does belief-aligned AI bridge or
  reinforce — and under which structural conditions?
- **¶6 The model in one paragraph.** ABM of decentralized relief; two
  cognitive strategies (march1991exploration); opinion-dynamics lineage
  (deffuant2000mixing, hegselmann2002opinion); Bayesian belief revision;
  learned source selection. **Design-principle sentence:** the AI adapts
  only to what it can observe — the querier's revealed beliefs, through
  α — never to cognitive type; all type differences emerge from human
  information behavior.
- **¶7 The four gaps (one compact paragraph each, ~40% shorter than NHB).**
  Gap 1 structural access conditions; Gap 2 alignment dose–response
  (tatham2010application, glikson2020human, lee2004trust); Gap 3
  beliefs→decisions→feedback at population level (gralla2016problem);
  Gap 4 distribution of harms (holguin2012unique, coleman2024weaving).
- **¶8 Operationalization pointer.** Two configurations on identical
  seeds; one finding per gap; cites Fig. 1. α described as "confirmation
  of the querier's own prior beliefs" (NOT the community's).

### Results (one subsection per finding; all numbers = results-mechfix)

*Short metrics lead-in* (2–3 sentences, no main-text table): SECI /
AECI-IE-chan sign convention stated once; convergence-on-truth vs
convergence-on-error disambiguation kept; per-type reporting is the
default.

1. **Network-bounded access is the structural precondition** (Fig. 2a,b;
   Table S1; Fig. S1). Claim sentence: unrestricted access dissolves the
   confirmation-seekers' chamber at high α (SECI −0.45→+0.05); bounded
   access preserves it at every α (−0.39…−0.18); ΔSECI_exploit at α=1:
   −0.269 [−0.350, −0.189], N=50, Holm p≈5e-08. Accuracy-seeking
   communities deepen 0→−0.33 in BOTH configurations — the α-gradient is
   explorer-driven, the configuration contrast exploiter-driven.
   Societal-layer sentence: the population index (−0.22→0.00 in the
   control) masks the crossing — fragmentation analysis is not optional.
   One sentence on the per-type AI channel (bounded access narrows the
   confirmation-seekers' served pool, ΔAECI-IE-chan −0.341, and
   diversifies the accuracy-seekers', +0.229, both Holm-significant).
2. **Confirming AI harms through starvation, not persuasion; the optimum
   is interior** (Fig. 2c, Fig. 3a; Fig. S2; Table S2). Accuracy cost
   falls entirely on accuracy-seekers (explorer MAE 0.54→1.74; exploiter
   flat ≈1.8); smooth dose–response (delivered dose linear in α;
   effective_alpha manipulation check); mechanism = L1+ pool collapse
   113→29 (part of the SECI deepening is starvation, not shared error —
   said explicitly once). Interior optimum α*=0.6 (7/12 composites,
   spread [0.1, 0.6]); control interior too (0.8–0.9); one sentence: the
   old corner solution was a dose-delivery artifact (rounding ablation,
   Table S2). Societal sentence: population served-information index
   U-shaped in both configurations, shallowest ≈α=0.7, matching the
   operational optimum (unmet needs 1.59→0.25 at α=0.6→2.86).
3. **The feedback loop: capture, lock-in — and the retrenchment
   counterfactual** (Fig. 2d, Fig. 3b; Fig. S3; Table S8). Capture:
   exploiter AI query share 0.54→0.69, trust flat ≈0.47; lock-in:
   AI-heavy accuracy-seekers freeze (LockIn −0.01→−0.13); C12:
   accuracy-seekers never learn to distrust confirming AI (trust
   0.88→0.84; 0.90→0.82 even at full salience) because verification is
   base-rate dominated — a finding, not a failure. Counterfactual: s=1
   removes the capture gradient (share flat ≈0.5) but produces social
   retrenchment — chamber deepens (−0.52 vs −0.37) and precision falls
   (0.43 vs 0.57) at the truthful endpoint; not a remedy, a boundary
   condition. Operational buffering: unmet needs 2.86 vs 10.18 at α=1;
   explorer precision 0.62 vs 0.20. Robustness: interior α* in all 132
   cognitive-profile cells (Fig. 3b) — the optimum is structural, not a
   knife-edge of the assumed cognitive mix.
4. **Harms concentrate on the spatial periphery** (Fig. 4; Fig. S4;
   Table S3). Spatial MAE gap +0.12→+0.33; aid-contribution gap
   −1.8→−6.6; both ≈0 at all α in the immobile control (structural null);
   betweenness/broker gaps small — the periphery is spatial, not
   graph-positional, under mobility.

### Discussion (5 paragraphs)

- **¶1 Mechanism synthesis.** Sycophantic AI captures confirmation-seekers
  and starves accuracy-seekers; social structure decides whether community
  chambers survive and whether operations collapse; extends
  Glickman–Sharot dyadic amplification to networks (docking: Fig. S6),
  parallel to Lorenz/Becker structure-dependence of collective accuracy.
- **¶2 The interior optimum is descriptive, not a design target.**
  Explicit disclaimer; adoption routes that need no confirmation:
  uncertainty communication, transparency, verification support.
- **¶3 The salience counterfactual as a boundary condition.** Removing
  base-rate dilution does not create truth-seeking, it relocates
  confirmation demand into the social network; interventions must widen
  the verified channel, not just penalize the AI.
- **¶4 Type-agnostic AI as a modeling commitment.** One paragraph: the AI
  knows nothing unobservable; the confirmation-target ablation is inert;
  all differential outcomes emerge from human acceptance, verification,
  and reward behavior.
- **¶5 Limitations + generalization.** Reduced-form α; stylized hazard;
  N=100 (robust 100–500); exogenous verification; single uniform AI
  policy; extreme-case logic for transfer to milder settings; α*
  transfer caveat.

### Materials & Methods (final main-text section, concise)

Model in ~600–700 words: environment; agent types (D/δ, learning
channels; exploiters score confirmation against trusted-network consensus
when defined, own prior otherwise); AI rule r=(1−α)t+αb with b = querier's
own current belief for both types, stochastic report rounding (dose
linear; effective_alpha check); network/mobility/query-scope switches;
metrics (SECI, AECI-IE-chan per type + population, LockIn, L1+ pool, MAE,
unmet needs, precision); design (11α × 20 seeds × 200 ticks × 2
configurations, paired seeds); statistics (MixedLM + Holm contrasts;
N=50 boundary). Full ODD, calibration, and metric formalism → SI.

## Venue assessment (2026-08-26)

**Recommendation: PNAS Direct Submission (Social Sciences → Psychological
and Cognitive Sciences) as primary; PNAS Nexus as transfer fallback.**

- The escalation gate defined in the writing instructions is fully
  satisfied: M6 docking reproduces Glickman & Sharot for both agent
  types; M2 shows the interior optimum in all 132 cognitive-profile
  cells; M3 makes every boundary claim seed-robust at N=50 with Holm
  correction; M4 delivers significant U-curvature and configuration
  effects; M5 passes all ten robustness perturbations.
- Lineage: the two closest methodological ancestors (Lorenz et al. 2011;
  Becker et al. 2017) are both PNAS papers; the paper extends a
  Nature-published dyadic result (Glickman & Sharot 2024) to networked
  populations. That is the classic PNAS "general interest + strong
  discipline anchor" profile.
- The message is human–AI behavior at scale, not the ABM: the framing,
  significance statement, and cover letter should lead with the
  behavioral mechanisms (starvation, capture, retrenchment) — the model
  is the instrument.
- Main risk at PNAS: the credibility case rests on a substantial SI
  ablation chain competing with the 6-page format. Mitigated by the
  single-sentence attribution + Table S2 pattern used in the draft.
- Alternatives considered: **PNAS Nexus** (same format family, easier
  bar, direct transfer — the natural fallback, not the first choice now
  the gate is passed); **Science Advances** (good scope fit, more room
  for the SI chain; weaker lineage argument than PNAS); **Nature
  Communications** (viable, but the NHB-family framing was deliberately
  abandoned and its computational-social-science bar for pure-simulation
  papers is high); **Nature Human Behaviour** (dropped per instructions —
  harder than PNAS for this genre); **JASSS / J. Computational Social
  Science** (excluded by the author: too niche/ABM-focused for the
  paper's message).
