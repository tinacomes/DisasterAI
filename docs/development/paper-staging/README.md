# Paper-repo staging (2026-08-17)

Edits for `tinacomes/DisasterAIFilterPaper` (the Overleaf-synced paper repo),
staged here because the Claude GitHub App has no write access to that repo yet.

- `0001-*.patch` — the full commit (text edits + References.bib fixes +
  figure files), produced with `git format-patch`; apply with `git am` in a
  clone of DisasterAIFilterPaper, or push directly from a session once the
  app has write access (github.com/settings/installations → Claude →
  Repository access → add DisasterAIFilterPaper).
- `DisasterAIFilter_main.tex`, `References.bib`,
  `Supplementary_S1-S9_content.tex` — the edited files, for reference or
  manual upload to Overleaf.
- Figure PNGs are in the patch; they are the regenerated versions of the
  committed run 29100134858 (see `results/` in this repo).

Delete this directory once the changes are on the paper repo.
