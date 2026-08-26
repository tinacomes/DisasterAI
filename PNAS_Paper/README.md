# PNAS_Paper — Overleaf-ready manuscript package

Redraft of the DisasterAIFilter manuscript to the PNAS format family
(2026-08-26), following `../PNAS_WRITING_INSTRUCTIONS.md`. All results
statements are based on the revised-model canonical run
`../results-mechfix/` (run 32821105202) and the completed M2–M7
validation chain; every claim traces to `../RESULTS_COMPENDIUM.md`.

## Contents (everything needed for Overleaf)

| File | Role |
|---|---|
| `DisasterAIFilter_PNAS.tex` | Main text (Intro, 4 Results findings, Discussion, Materials & Methods; 4 display items; ~4,000 words; Abstract ≤250 w; Significance ≤120 w) |
| `SI_Appendix.tex` | Standalone SI Appendix (model description, metric definitions, experiment designs, Figs. S1–S7, Tables S1–S10, transparency chain) |
| `References.bib` | Bibliography (copied from `tinacomes/DisasterAIFilterPaper`; no new references added; all cited keys verified to resolve) |
| `sn-jnl.cls`, `sn-nature.bst` | Class/style so the main text compiles as-is during drafting |
| `Figures/` | All figure files (see below) |
| `make_figures.py` | Regenerates Figs. 2–4 and S2/S3/S7 from the archived result JSONs (run from the repo root); copies S1/S4/S5/S6 from the archives. Never hand-edits figure data |
| `STORYLINE.md` | Paragraph-by-paragraph storyline (what changed vs. the NHB draft) and the venue assessment |

## Compiling

- Main text: `pdflatex DisasterAIFilter_PNAS.tex` + `bibtex` (uses
  `sn-jnl`/`sn-nature`). Fig. 1 is TikZ (drawn in the .tex).
- SI Appendix: `pdflatex SI_Appendix.tex` + `bibtex` (plain `article`
  + `natbib`, self-contained).
- **At submission**: swap the main text to `pnas-new.cls`
  (`\templatetype{pnasresearcharticle}`), move the Significance text
  (currently in `\abstract*{}`) into `\significancestatement{}`, and
  convert the SI to `\templatetype{pnassupportinginfo}`. Verify current
  PNAS limits (title ≤135 chars, abstract ≤250 w, significance ≤120 w,
  6 pages) against the live author guidelines.

## Figures

Main text: `fig2_configuration` (per-type SECI, per-type MAE, unmet
needs; both configurations), `fig3_goldilocks` (interior optimum +
cognitive-profile robustness), `fig4_periphery` (spatial gaps), each as
`.png` and `.pdf`. SI: `figS1_lifecycle` (archived),
`figS2_starvation`, `figS3_feedback`, `figS4_periphery_evolution`
(archived), `figS5_population_evolution` (archived),
`figS6_dyadic_docking` (archived), `figS7_robustness`.

Sanity checks built into `make_figures.py` assert that the recomputed
α* locations match the archived α* sensitivity tables
(population composite: 0.6 main / 0.9 control; gap-sweep α* within
[0.6, 0.8] in every cell).

## Author actions before submission

1. Venue decision (see `STORYLINE.md`): PNAS Direct Submission primary,
   PNAS Nexus fallback.
2. Approve Significance wording; complete Acknowledgements/Funding.
3. Mint the Zenodo DOI over code + all results directories; update the
   Data availability statement.
4. Resolve the `steinbrink2024transparency` placeholder in
   `References.bib` (currently unused by this manuscript).
