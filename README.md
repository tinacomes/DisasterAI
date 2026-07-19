# depacc — potential-deprivation accessibility equity across European cities

A reproducible, config-driven pipeline measuring infrastructure accessibility
across European cities through the lens of **deprivation**, contrasting
**everyday-service access** against **emergency capability**:

- **Everyday services** (GP, pharmacy, supermarket, school, green space —
  chosen, repeated, substitutable): a potential/gravity measure. The effective
  deprivation time is a **soft-minimum** over reachable facilities of travel
  time inflated by a **2SFCA congestion factor** (supply capacity vs demand
  competition); deprivation = DLF(effective time).
- **Emergency capabilities** (emergency-department hospitals, ambulance
  stations — non-substitutable, time-critical): **nearest-facility** travel
  time only; deprivation = convex DCF(nearest time).

The deprivation function *is* the impedance function of a gravity model run in
the opposite direction — increasing and convex in travel time — with all
functional forms and parameters **transferred from the literature** via config
(`config/deprivation.yaml`; the pipeline refuses to run while parameters are
null placeholders and its error names the paper each value must come from).

The **central output** is the *relationship* between the two surfaces:

1. **Cell-level co-location** — a population-weighted bivariate typology
   (everyday hi/lo × emergency hi/lo) mapping *compounding* deprivation;
2. **City-level divergence** — each city as a point in an
   everyday-vs-emergency plane (e.g. Gini vs Gini), off-diagonal spread;
3. **Trajectory** — cities ordered along the size gradient, testing whether
   everyday and emergency deprivation co-evolve or diverge with city size.
   This is **space-for-time, cross-sectional inference** (after Musso et al.,
   PNAS 2026): trajectories are read from the cross-sectional city-size
   gradient, never from observed temporal change. There is deliberately no
   longitudinal component.

## Two-tier data architecture

| | Tier 1 (continental) | Tier 2 (deep dive) |
|---|---|---|
| Cities | all Eurostat-OECD FUAs above config threshold | DE, NL, FR, UK, Nordics + reliable-GTFS cities |
| Population | GHS-POP 100 m; Eurostat Census 2021 1 km | + DE Zensus 2022 100 m, NL CBS 100 m, FR INSEE Filosofi 200 m, UK LSOA+IMD |
| Facilities | OSM (completeness-benchmarked per country) | same |
| Modes | walk + car (harmonised, OSM) | + public transit (r5py + R5 + GTFS) |

## Install

```bash
cd deprivation-accessibility-eu
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"     # core + tests
uv pip install -p .venv/bin/python -e ".[full]"                     # full geospatial/routing stack
```

`r5py` (Tier-2 transit routing) requires a **JDK 21** Java runtime.

## Run

```bash
depacc validate --city hamburg          # config sanity check
depacc run --city hamburg               # full pipeline: ingest → access →
                                        # deprivation → divergence → equity → viz
depacc run --city hamburg --stage access
pytest                                  # unit tests (no downloads needed)
```

No raw data is committed; everything under `data/` is reproduced by the ingest
stage with cached downloads and JSON provenance sidecars (URL, SHA-256,
timestamp, licence). See `data/README.md` for every source, licence,
resolution and native CRS, and `methods.md` for every modelling choice and the
literature source of every parameter.

## Repository layout

```
config/            defaults + services + deprivation functions + per-city YAML
src/depacc/
  ingest/          cached downloaders + provenance logging
  quality/         OSM completeness benchmarking per country
  access/          travel-time matrices (walk/car everywhere; +transit Tier 2)
  deprivation/     DLF/DCF forms · soft-min reducer · 2SFCA congestion · surfaces
  divergence/      bivariate typology · city-level everyday-vs-emergency plane
  equity/          weighted mean · Gini · concentration index · regressions
  cityvector/      per-city features · clustering · size-gradient trajectory
  viz/             maps and cross-city figures
tests/             unit tests on the model mathematics (synthetic fixtures)
data/              (gitignored) raw + derived data, reproduced by ingest
docs/              static results site
```

## Note on hosting

This project currently lives as a self-contained subproject on a development
branch of `tinacomes/DisasterAI` (session constraint). To extract it into its
own repository with history:

```bash
git subtree split --prefix=deprivation-accessibility-eu -b depacc-standalone
# then push that branch to a new empty repo, e.g. deprivation-accessibility-eu
```

After the split: enable GitHub Pages from `docs/`, connect the repository to
Zenodo via the GitHub–Zenodo integration before tagging the first release
(this yields a DOI; `CITATION.cff` is already in place), and configure a DVC
remote for the heavy derived artefacts (`dvc remote add -d <name> <url>`).

## Licence

MIT for code. Derived data redistributed with releases: CC-BY-4.0, with
attribution to JRC/GHSL, Eurostat/GISCO, © OpenStreetMap contributors (ODbL),
national statistical offices and transit agencies. See `LICENSE` and
`data/README.md`.
