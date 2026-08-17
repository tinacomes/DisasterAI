# Paper review — Overleaf draft vs repository results (2026-08-17)

Review of the Overleaf draft *"Breaking the Bubble? How Belief-Aligned AI
Reshapes Collective Opinions and Decisions in Urgent Decisions"* against the
committed results (run 29100134858, 2026-07-10) and the code on this branch.
This file lives in `docs/development/` and is internal working material —
delete with the rest of the directory before submission.

---

## 1. Number-by-number verification

Every quantitative claim in the Results was checked against
`results/comparison/comparison_table.md`, `results/*/summary_table.md`,
and values recomputed from `results/*/experiment_results.json` with the same
code paths the figures use.

**Verified correct** (exact match with the committed run):

| Claim in draft | Source | Status |
|---|---|---|
| Control SECI −0.246±0.023 (α=0) → −0.133±0.037 (α=1) | comparison table | ✓ |
| ΔSECI null for α≤0.6; −0.069 [−0.116,−0.021] at 0.7; −0.182 [−0.250,−0.114] at 1.0 | paired deltas | ✓ |
| Main-model SECI −0.316±0.055 at α=1 | comparison table | ✓ |
| AECI-Var −0.267/−0.254 (α=0) → −0.160/−0.135 (α=1) | comparison table | ✓ |
| ΔAECI-Var seed-robust only at α=0.5 (+0.036 [+0.006,+0.067]) | paired deltas | ✓ (but see issue 2) |
| AECI-Err +0.050±0.028 (main, α=0.9); positive at α≥0.9 both configs | comparison table | ✓ |
| SECI −0.19…−0.24 for α≤0.8, −0.32 at α≥0.9 (main) | summary table | ✓ |
| AECI-Var strongest α=0 (−0.254), weakest α=0.8 (−0.122) | summary table | ✓ |
| Bubble composite min 0.28 at α\*=0.6; near-minima 0.45 at 0.4/0.5; operational min 0.55 at 0.4 | summary table | ✓ |
| α\* sensitivity: main {0.6, 0.4, 0.5} / {0.4, 0.4, 0.3}; control {1.0×3} / {0.7, 0.7, 0.3} | comparison table | ✓ |
| Unmet 0.42 at α=0.5 vs 1.11 at α=0; precision ≈0.78 flat to α=0.6 (main) | summary table | ✓ |
| Lifecycle: peaks 0.44–0.58 / 0.33–0.50; exploit recovery ticks 40–55 (α≤0.8), never at α≥0.9; explor 105→155, never at α≥0.7 | recomputed from JSON | ✓ |
| AI query share: explor 0.50 (α=0) – 0.59 (α=0.6); exploit 0.39 (α=0) → 0.59 (α=1) | recomputed from JSON | ✓ (see issue 4) |
| ΔMAE seed-robust at all 11 levels, −0.077 [−0.118,−0.036] (0.4) to −0.156 [−0.203,−0.109] (1.0) | paired deltas | ✓ |
| Main α=0.6 MAE 1.376 ≈ control α=0.5 (1.441) | comparison table | ✓ |
| ΔPrecision positive except α=0.7; +0.302 [+0.204,+0.399] at 0.9 | paired deltas | ✓ |
| Control unmet 2.5 (α=0.8) → 11.0 (α=0.9); precision 0.62 → 0.21; main 3.2 / 0.51 | comparison table | ✓ means (± labels wrong, issue 1) |
| ΔUnmet −4.5 [−6.0,−2.9] and −4.2 [−5.5,−2.8] at α≥0.9 | paired deltas | ✓ as printed (but see issue 2) |
| Relief volume 402 → 330 tokens/tick; worst-served deficit +0.19 → +0.34 | recomputed from JSON | ✓ (401.7→329.5; 0.186→0.336) |
| Methods parameters (D, δ, ε=0.3, η=0.10, 15 % AI sensing, shock 0.1/±2, p_bridge=0.15, 2-hop 0.1, non-friend floor 0.05, radii 3/8, …) | DisasterAI_Model.py + test_filter_bubbles.py | ✓ |

**Issues found:**

1. **The "±" on unmet needs was not an s.e.m.** In
   `compute_goldilocks_metrics`, every metric's spread was divided by √N
   except `unmet_needs`, whose value was the tick-wise across-run SD (≈SD, not
   s.e.m.). The draft reports "2.5±1.9", "11.0±3.1", "3.2±2.9" as mean±s.e.m. —
   the true s.e.m.s are roughly 0.3–0.6. **Fixed in code on this branch**; the
   text edit is in §4 below. The committed `comparison_table.md` and Fig. 2's
   unmet panel error bars inherit the old value until the workflow is re-run.

2. **Window mismatch between raw means and paired deltas** for the two
   per-tick series (unmet needs, AECI-Var). Per-run steady-state values
   (`*_ss_runs`) covered only the last **15 ticks**, while the reported means
   cover the last **75 ticks**. Consequences in the draft:
   - ΔUnmet "−4.5 / −4.2" sits next to raw means 11.0 vs 3.2 whose difference
     is −7.7 — an internal inconsistency a careful reviewer will catch.
   - The "seed-robust at a single alignment level only (α=0.5)" claim for
     ΔAECI-Var was computed on the 15-tick window; on the 75-tick window the
     raw difference at α=0.5 is +0.061, and the significance pattern may
     change.
   **Fixed in code on this branch.** The corrected deltas require re-running
   the **Compare Baseline vs Network/Mobility Switches** workflow (dispatch
   from the Actions tab on this branch; the session's GitHub integration
   lacks workflow-dispatch rights). Refresh the two affected sentences from
   the new `comparison_table.md` afterwards. The factor-of-three buffer claim
   holds on both windows (11.0/3.2 ≈ 3.4; 6.6/2.15 ≈ 3.1), so the qualitative
   conclusion is safe.

3. **Fraction-of-run-in-chamber range is imprecise.** Draft: "rises for both
   types … to 0.7–0.83 at α≥0.8". Actual values: exploitative 0.60 at α=0.8
   and 0.80/0.82 only at α≥0.9; exploratory 0.80 at α=0.8, 0.70 at α≥0.9.
   Replacement text in §4.

4. **"Rises monotonically" (exploitative AI share) is not strictly true** —
   there is a small dip at α=0.4 (0.413 → 0.409). Use "rises steadily".

5. **Abstract finding 4 is disclaimed by the Results.** The abstract and the
   introduction's fourth contribution promise the periphery result ("harm
   concentrates on those remote from the disaster; central network positions
   offer no protection"), but Sec. 3.3 ends by explicitly *not* building
   claims on the periphery analyses. The committed data **supports** the
   claim: the far−near belief-error gap grows monotonically from +0.14 (α=0)
   to +0.37 (α=1); far-spawned agents' relief contribution falls ≈16→6
   tokens/agent per 5-tick window (near: ≈18.5→13); the high-vs-low
   betweenness error difference stays within ±0.06 at every α. Recommended
   fix: add a short Results §3.4 (draft in §5 below) rather than weakening
   the abstract.

6. **Broken cross-references:** "Figure ??" (conceptual overview, Background
   synthesis), "Table ??" twice in Results (the paired-deltas table — it does
   not exist in the draft; LaTeX provided in §6). ~25 citations render as
   "(?)" — missing BibTeX entries; suggestions in §7.

7. **Count inconsistency:** the abstract reports *four* findings and the
   introduction *four* contributions/gaps, but Results announces "The *three*
   research questions" (Q1–Q3). Adding §3.4 (issue 5) resolves this; then
   change "three" → "four" and add Q4 to the Results preamble.

---

## 2. What was fixed in code (this branch)

- `test_filter_bubbles.py`: per-tick `*_ss_runs` now cover the last 75 ticks
  (window matched to cadence); `unmet` SE is now a true across-replication
  s.e.m. from per-run late-run means.
- Figures (regenerable via `--plots-only` / Replot workflow, no
  re-simulation): panel letters **a, b, c…** on all main-text figures
  (Nature requirement); Goldilocks SECI/AECI-Var panels no longer squashed
  into the theoretical [−1, 1] range; lifecycle figure uses one consistent
  colour pair for the two agent types across panels; AI-query-share figure
  draws s.d. bands only for α=0 and α=1 (was 11 overlapping bands);
  comparison-figure legend now says "control (global access)" / "main model
  (network-bounded)" matching the paper's terminology.
- `tools/compare_configs.py`: display-label mapping + panel letters.

**Action needed:** dispatch *Compare Baseline vs Network/Mobility Switches*
on this branch (Actions tab, defaults) → then *Archive Run Artifacts* with
the run ID to refresh `results/`, and update the two delta-dependent
sentences (issue 2).

Figure-caption updates required by the figure changes:
- Fig. 5 caption: colours no longer differ per panel; describe
  dark/light = exploitative/exploratory once.
- Fig. 6 caption: "shading (±s.d. over replications) shown for α = 0 and
  α = 1 only, for legibility".
- Fig. 2 caption: after the re-run, the unmet panel error bars are genuine
  s.e.m. — the caption's "mean ± s.e.m." then holds for all panels.

---

## 3. Structure review (Nature Human Behaviour)

The draft currently reads Intro → Background → Results → Discussion →
Conclusion → Methods with numbered sections. NHB Articles use:
**Introduction (no heading) → Results → Discussion → Methods**, no separate
Background, no separate Conclusion, unnumbered sections, abstract ≤ ~150
words without citations, main text ~6,000 words (excl. Methods/refs/captions).

Recommendations, in order of impact:

1. **Fold the Background into the Introduction.** Keep the current intro's
   arc (fragmentation → AI arrives → sycophancy → crises as the acute case →
   networks gap) and absorb from the Background only what the reader needs
   before the Results: one paragraph each on (i) networked information
   sharing/homophily/weak ties, (ii) trust & the sycophancy dilemma +
   algorithmic monoculture, (iii) belief→decision feedback under urgency —
   each ending in its gap/question. Target 1,800–2,200 words. The synthesis
   §2.4 becomes the final intro paragraph + the Fig. 1 walk-through.
2. **Move Table 1 (promises/perils of AI) to Supplementary** (or propose it
   as a Box). It is a framing device, not evidence, and it costs ~a page.
3. **Add Results §"Harms concentrate on the spatial periphery"** (draft in
   §5) so the four promised findings are the four Results subsections; this
   also legitimises promoting Fig. S4 (periphery gap) to the main text —
   arguably the paper's most policy-relevant figure.
4. **Merge the Conclusion into the final Discussion paragraph** (NHB has no
   Conclusion section). The current Conclusion is already a compact summary —
   it can replace the Discussion's last paragraph.
5. **Delete the roadmap paragraph** at the end of the Introduction ("The
   remainder of the paper proceeds as follows…") — not used in NHB, and it
   currently skips Results/Methods anyway.
6. **Abstract:** rewrite to ≤150 words, no keywords line (NHB doesn't use
   author keywords). Draft in §4, E1.
7. **The disaster-vs-general tension** is already handled the right way:
   general phenomenon up front, disasters as the paradigmatic, measurable
   testbed, generalisation paragraph in the Discussion (urgency, moving
   ground truth, delayed verification, bounded access → public health,
   financial contagion, security). Two sharpenings: (a) say "urgent
   decision situations" in the title instead of "Urgent Decisions" (avoids
   "Decisions … Decisions"), e.g. *"Breaking the bubble? How belief-aligned
   AI reshapes collective beliefs and decisions in urgent situations"*;
   (b) in the first Results sentence, remind the reader once that the
   disaster grid is the operationalisation of an urgent decision
   environment, then stop re-justifying it.
8. **References**: NHB uses numbered Nature style; the sn-jnl author-year
   format is fine for the working draft, but plan the switch (springer nature
   `sn-nature` option or a natbib→numeric flip) before submission.

---

## 4. Concrete text edits (old → new)

**E1 — Abstract** (replace entirely; ~150 words, no citations):

> AI assistants trained on human feedback drift towards sycophancy:
> confirming what their users already believe. Whether this breaks or
> deepens the echo chambers in which humans form beliefs is unknown, because
> existing evidence comes from individuals interacting with a single AI,
> while echo chambers arise in networks. Here we use an agent-based model of
> disaster response in which people consult peers or an AI whose reports
> vary, through an alignment parameter α, from ground truth to full
> confirmation of prior beliefs. Interaction structure decides the influence
> of AI: a confirming AI deepens social echo chambers only where information
> access follows social ties, while a fully truthful AI creates an echo
> chamber of its own among its most reliant users. Intermediate alignment
> (α ≈ 0.4–0.6) minimises both while sustaining relief performance, and the
> harms of rising alignment concentrate on people remote from the disaster.
> Trustworthy AI is thus a property of the sociotechnical system, not of the
> model alone.

**E2 — Results preamble** ("The three research questions structure…"):
> The four research questions structure the presentation of the results, one
> finding per question…
and append: "…and how the harms of alignment are distributed across space
and the social network (Q4, Sec. 3.4)."

**E3 — Unmet-needs sentence (Sec. 3.3)** — until the re-run lands, drop the
mislabelled ± values:
> In the control, unmet high-need cells jump from 2.5 at α = 0.8 to 11.0 at
> α = 0.9 (means over N = 20 replications), while precision falls from 0.62
> to 0.21 […] The main model degrades at the same alignment levels, and far
> less severely: unmet needs rise to 3.2 and precision falls to 0.51 at
> α = 0.9.
After the re-run, reinstate "±" using the new (true s.e.m.) values from
`comparison_table.md`, and refresh the ΔUnmet numbers in the following
sentence from the same table.

**E4 — Fraction-in-chamber sentence (Sec. 3.2)**:
> Consistent with this, the fraction of the run spent inside a chamber rises
> from roughly 0.6 (exploitative) and 0.4 (exploratory) in the low-to-mid
> range to 0.80–0.82 for exploitative agents at α ≥ 0.9 and 0.70–0.80 for
> exploratory agents at α ≥ 0.8.

**E5 — "monotonically" (Sec. 3.2, Fig. 6 paragraph)**:
> their steady-state AI share rises steadily with alignment, from 0.39 at
> α = 0 to 0.59 at α = 1.

**E6 — AECI-Var α=0.5 delta sentence (Sec. 3.1)**: keep the structure but
re-read the significance from the regenerated table before submission; if
more levels become seed-robust, the sentence "the paired difference is
seed-robust at a single alignment level only" must change accordingly.

**E7 — Typos / grammar** (Introduction & Background):
- "Thus far, however, the use of LLMs so far is far from pervasive" → "So
  far, however, the use of LLMs is far from pervasive" (also fix "Angrisani
  et al. (2026)" → "\citep{...}").
- "How we share information and interacts with others depends on our social
  structures our networks" → "How we share information and interact with
  others depends on our social structures — our networks —".
- "Because rapid the combination of complexity and urgency overwhelm" →
  "Because the combination of complexity and urgency overwhelms".
- "These silos are further is amplified and translated" → "These silos are
  further amplified and translated".
- "it is not know what the structural conditions are of information access"
  → "it is not known under which structural conditions of information
  access".
- "urgency and complextiy" → "complexity".
- "feed back into into sensemaking" → "feed back into sensemaking".
- "yet relationship between the degree of alignment" → "yet the
  relationship…".
- "The urgency of crises make this dilemma" → "makes".
- "How AI and algorithmic influence interacts with these behavioural and
  social structure" → "…with these behavioural patterns and social
  structures".
- Abstract old text "interaction structure decides on influence of AI" →
  "decides the influence of AI" (superseded by E1).
- "offer no protection. The Results follow this logic, presenting one
  finding per gap" — sentence ends without a period in the draft.
- "outcomes homogenise and, and collective outcomes" → drop the duplicated
  "and,".

---

## 5. Draft Results §3.4 (new subsection, supports abstract finding 4)

> **3.4 Harms of alignment concentrate on the spatial periphery**
>
> Q4 asked how the harms of belief-alignment are distributed across space
> and the social network. Supplementary Figs. S4–S5 decompose steady-state
> outcomes by distance between an agent's home location and the epicentre
> (nearest versus furthest quartile) and by betweenness centrality (Q1
> versus Q4). In the main model, the belief-error gap between far- and
> near-spawned agents widens monotonically with alignment, from +0.14 at
> α = 0 to +0.37 at α = 1: agents remote from the events, who depend almost
> entirely on mediated reports, absorb most of the accuracy loss that
> confirmation produces. The behavioural consequence is withdrawal from the
> response: relief contributed by far-spawned agents falls from roughly 16
> to 6 tokens per agent per five-tick window across the sweep, while
> near-spawned agents reduce their contribution far less (from roughly 18
> to 13) — the remote quartile does not merely become less accurate, it
> stops sending assistance. Network position offers no comparable
> protection: the belief-error difference between high- and low-betweenness
> agents remains within ±0.06 at every alignment level, and both groups
> degrade in lockstep as alignment rises. The harms of a confirming AI are
> thus spatially, not topologically, structured: they fall on those who
> cannot verify reports against first-hand observation, regardless of how
> well connected they are.

(If S4 is promoted to the main text as the new Fig. 7, renumber
accordingly. Values are steady-state means over the last 75 ticks of the
committed run; CIs are shown in the figure.)

---

## 6. LaTeX for the missing paired-deltas table ("Table ??")

Refresh the ΔAECI-Var and ΔUnmet columns from `comparison_table.md` after
the workflow re-run; the other columns are unaffected by the window fix.

```latex
\begin{table}[t]
\centering
\caption{Paired per-seed differences (main model $-$ control) across the
alignment sweep. Replicate $i$ uses seed $i$ in both configurations; cells
give the mean difference with the 95\% CI of the per-seed differences
($N=20$). Bold marks CIs excluding zero.}\label{tab:paired-deltas}
\begin{tabular}{lcccc}
\toprule
$\alpha$ & $\Delta$SECI & $\Delta$MAE & $\Delta$Unmet & $\Delta$Precision \\
\midrule
0.0 & $+0.018\,[-0.028, +0.064]$ & $\mathbf{-0.080\,[-0.116, -0.044]}$ & $-0.68\,[-1.37, +0.01]$ & $\mathbf{+0.062\,[+0.002, +0.121]}$ \\
0.1 & $+0.020\,[-0.026, +0.066]$ & $\mathbf{-0.087\,[-0.116, -0.058]}$ & $\mathbf{-1.18\,[-2.13, -0.23]}$ & $\mathbf{+0.070\,[+0.017, +0.124]}$ \\
0.2 & $+0.022\,[-0.018, +0.062]$ & $\mathbf{-0.081\,[-0.117, -0.044]}$ & $-0.19\,[-1.04, +0.65]$ & $\mathbf{+0.064\,[+0.007, +0.122]}$ \\
0.3 & $+0.009\,[-0.039, +0.058]$ & $\mathbf{-0.091\,[-0.129, -0.054]}$ & $-0.30\,[-0.95, +0.35]$ & $\mathbf{+0.069\,[+0.009, +0.130]}$ \\
0.4 & $+0.015\,[-0.032, +0.062]$ & $\mathbf{-0.077\,[-0.118, -0.036]}$ & $-0.46\,[-1.01, +0.10]$ & $\mathbf{+0.071\,[+0.009, +0.133]}$ \\
0.5 & $+0.004\,[-0.048, +0.056]$ & $\mathbf{-0.105\,[-0.140, -0.071]}$ & $-0.16\,[-0.52, +0.19]$ & $\mathbf{+0.081\,[+0.022, +0.140]}$ \\
0.6 & $-0.011\,[-0.057, +0.035]$ & $\mathbf{-0.130\,[-0.163, -0.098]}$ & $-0.16\,[-0.54, +0.23]$ & $\mathbf{+0.093\,[+0.035, +0.151]}$ \\
0.7 & $\mathbf{-0.069\,[-0.116, -0.021]}$ & $\mathbf{-0.123\,[-0.161, -0.085]}$ & $-0.40\,[-0.85, +0.04]$ & $+0.057\,[-0.008, +0.122]$ \\
0.8 & $\mathbf{-0.080\,[-0.127, -0.033]}$ & $\mathbf{-0.116\,[-0.163, -0.068]}$ & $\mathbf{-0.76\,[-1.42, -0.10]}$ & $\mathbf{+0.097\,[+0.030, +0.164]}$ \\
0.9 & $\mathbf{-0.166\,[-0.231, -0.101]}$ & $\mathbf{-0.154\,[-0.214, -0.095]}$ & $\mathbf{-4.46\,[-6.04, -2.87]}$ & $\mathbf{+0.302\,[+0.204, +0.399]}$ \\
1.0 & $\mathbf{-0.182\,[-0.250, -0.114]}$ & $\mathbf{-0.156\,[-0.203, -0.109]}$ & $\mathbf{-4.16\,[-5.48, -2.85]}$ & $\mathbf{+0.276\,[+0.179, +0.373]}$ \\
\bottomrule
\end{tabular}
\end{table}
```

(ΔAECI-Var and ΔAECI-Err can go to a supplementary table to keep the main
table readable — the text quotes them only at single levels.)

---

## 7. Missing citations — suggested BibTeX targets

Every "(?)" in the draft, with a concrete, real reference to consider.
Verify each fits your intended claim before adding.

| Location / claim | Suggestion |
|---|---|
| Echo chambers & polarisation (intro, "?Cinelli…") | Del Vicario et al. 2016, *The spreading of misinformation online*, PNAS 113(3) |
| Humans delegate information collection/interpretation to AI | Rahwan et al. 2019, *Machine behaviour*, Nature 568 |
| AI could bridge divides / correct biases | Argyle et al. 2023, PNAS 120(41) (AI chat interventions improve divisive conversations); Tessler et al. 2024, *AI can help humans find common ground*, Science 386 |
| Contradicting information rejected; rejection erodes trust | Nickerson 1998, *Confirmation bias*, Rev. Gen. Psychol. 2(2); Nyhan & Reifler 2010, Political Behavior 32 |
| Preference-based training / RLHF ("(??)") | Christiano et al. 2017 + Ouyang et al. 2022 (already in your .bib) |
| Raters prefer belief-matching responses → sycophancy | Sharma et al. 2024a (in .bib); Perez et al. 2023, *Discovering language model behaviors with model-written evaluations*, ACL Findings |
| Sycophancy documented across AI assistants | Sharma et al. 2024a; Cheng et al. 2026 (both in .bib) |
| Urgent decision situations ("??Comes et al. 2020") | Svenson & Maule 1993; Mendonça et al. 2001 (both already cited elsewhere) |
| Selective exposure in crises | Garrett 2009, JCMC 14(2); or Stroud 2010, J. Communication 60 |
| Confirmation bias in crises | Nickerson 1998 (above) or keep Paulus et al. 2022 |
| Disasters disrupt information infrastructures | Comfort 2007, *Crisis management in hindsight*, Public Admin. Rev. 67 |
| Polycrisis eroding sustainability | Lawrence et al. 2024, *Global polycrisis*, Global Sustainability 7 |
| Algorithmic-bias opinion model (any pair may meet) | Sîrbu et al. 2019, *Algorithmic bias amplifies opinion fragmentation and polarization*, PLOS ONE 14(3) |
| Algorithm curates timelines in a network | Perra & Rocha 2019, *Modelling opinion dynamics in the age of algorithmic personalisation*, Sci. Rep. 9 |
| Consequences of algorithmic bias depend on structure | Keijzer & Mäs 2022, *The complex link between filter bubbles and opinion polarization*, Data Science 5 |
| Algorithmic monoculture ("(??)") | Kleinberg & Raghavan 2021, PNAS 118(22); Bommasani et al. 2022, NeurIPS (outcome homogenization) |
| Recommender feedback homogenises behaviour | Chaney, Stewart & Engelhardt 2018, RecSys '18 |
| Belief-level counterpart of algorithmic monoculture | Kleinberg & Raghavan 2021 (as above) |
| "? term this the alignment paradox" | The intended source needs to come from you; a candidate is West & Aydin's *AI Alignment Paradox* essay (CACM, 2025) — verify it matches your usage |
| Sycophancy at level of individual responses | Sharma et al. 2024a |
| Remote actors depend on mediated information | Van de Walle & Comes 2015, *On the nature of information management in complex and natural disasters*, Procedia Eng. 107 |
| AI guidelines/standards only partially cover urgency | EU AI Act (Reg. (EU) 2024/1689); NIST AI Risk Management Framework 1.0 (2023) |
| Opinion-dynamics frameworks ("(??)" in intro) | Hegselmann & Krause 2002 + Deffuant et al. 2000 (both in .bib) |
| Belief-aligned generative models emerging ("??Cheng") | Sharma et al. 2024b + Glickman & Sharot 2025 (both in .bib) |

---

## 8. Overleaf access

This environment cannot reach the Overleaf project directly (no
credentials). Two ways to let a session edit it:
1. **Overleaf ⇄ GitHub sync** (Overleaf premium/institutional): link the
   Overleaf project to a GitHub repo (e.g. a `paper/` folder or a dedicated
   repo), then edits pushed from here appear in Overleaf via "Pull from
   GitHub".
2. Paste the **Overleaf git URL + token** (Menu → Sync → Git) into the chat;
   the project can then be cloned, edited and pushed back directly.

Until then, the edits in §4–§6 are copy-paste ready.
