"""Command-line interface.

    depacc run --city hamburg                 # full pipeline for one city
    depacc run --city hamburg --stage access  # a single stage
    depacc validate --city hamburg            # config sanity check

Stages run in order: ingest -> access -> deprivation -> divergence -> equity
-> viz. Heavy dependencies (geopandas, r5py, ...) are imported inside each
stage, so the CLI and unit-testable mathematics only need the core install.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from depacc.config import load_config

STAGES = ("ingest", "access", "deprivation", "divergence", "equity", "viz")


def _run_stage(stage: str, cfg: dict, city: str, project_root: Path) -> None:
    if stage == "ingest":
        from depacc.ingest.pipeline import run_ingest

        run_ingest(cfg, city, project_root)
    elif stage == "access":
        from depacc.access.matrices import run_access

        run_access(cfg, city, project_root)
    elif stage == "deprivation":
        from depacc.deprivation.pipeline import run_deprivation

        run_deprivation(cfg, city, project_root)
    elif stage == "divergence":
        from depacc.divergence.pipeline import run_divergence

        run_divergence(cfg, city, project_root)
    elif stage == "equity":
        from depacc.equity.pipeline import run_equity

        run_equity(cfg, city, project_root)
    elif stage == "viz":
        from depacc.viz.pipeline import run_viz

        run_viz(cfg, city, project_root)
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"Unknown stage {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depacc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline for one city")
    run.add_argument("--city", required=True, help="city id (config/cities/<id>.yaml)")
    run.add_argument("--stage", choices=STAGES + ("all",), default="all")
    run.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="subproject root (default: the installed package's repo checkout)",
    )

    val = sub.add_parser("validate", help="load + validate config for a city")
    val.add_argument("--city", required=True)

    make = sub.add_parser(
        "make-city", help="generate a minimal Tier-1 fast-path city config "
                          "(friction engine + Overpass facilities)")
    make.add_argument("--fua-code", required=True, help="URAU FUA code, e.g. SE001L1")
    make.add_argument("--name", required=True)
    make.add_argument("--country", required=True, help="two-letter code, e.g. SE")
    make.add_argument("--city-id", default=None,
                      help="config id/slug (default: name, lowercased)")

    cross = sub.add_parser(
        "cross", help="cross-city cityvector clustering, PNAS-style scaling "
                      "regressions + size-gradient read over cityplane.csv")
    cross.add_argument("--n-clusters", type=int, default=3)
    cross.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[2])

    lf = sub.add_parser(
        "list-fuas", help="list Functional Urban Areas (codes + names) from "
                          "the Eurostat URAU layer, for choosing cities")
    lf.add_argument("--country", action="append", default=None,
                    help="two-letter code; repeatable (e.g. --country DE --country NL)")
    lf.add_argument("--project-root", type=Path,
                    default=Path(__file__).resolve().parents[2])

    args = parser.parse_args(argv)

    if args.command == "list-fuas":
        from depacc.ingest.fua_sample import list_fuas

        fuas = list_fuas(load_config(), args.project_root)
        if args.country:
            fuas = fuas[fuas.country.isin([c.upper() for c in args.country])]
        with_pop = fuas.dropna(subset=["population"]) if fuas.population.notna().any() else fuas
        print(with_pop.sort_values(
            ["country", "population" if fuas.population.notna().any() else "name"],
            ascending=[True, False] if fuas.population.notna().any() else [True, True],
        ).to_string(index=False))
        print(f"\n{len(fuas)} FUAs. Use with: depacc make-city --fua-code CODE "
              f"--name NAME --country CC, or the batch workflow's make_cities "
              f"input: \"CODE,NAME,CC; ...\"")
        return 0

    if args.command == "make-city":
        from depacc.config import CONFIG_DIR

        city_id = args.city_id or "".join(
            ch for ch in args.name.lower().replace(" ", "_") if ch.isalnum() or ch == "_")
        path = CONFIG_DIR / "cities" / f"{city_id}.yaml"
        if path.exists():
            print(f"{path} already exists; not overwriting")
            return 1
        path.write_text(
            f"# Tier-1 fast-path city (generated by `depacc make-city`).\n"
            f"# Facilities via Overpass, travel times via the Weiss et al.\n"
            f"# friction surfaces — no .pbf, no GTFS, no Java.\n"
            f"city:\n"
            f"  id: \"{city_id}\"\n"
            f"  name: \"{args.name}\"\n"
            f"  country: \"{args.country.upper()}\"\n"
            f"  tier: 1\n"
            f"  fua_code: \"{args.fua_code}\"\n"
            f"sources:\n"
            f"  facilities: \"overpass\"\n"
            f"routing:\n"
            f"  engine: \"friction\"\n"
            f"  modes: [\"walk\", \"car\"]\n"
        )
        print(f"wrote {path}; run: depacc run --city {city_id}")
        return 0

    if args.command == "cross":
        from depacc.cityvector.clustering import run_cross_city

        run_cross_city(load_config(), args.project_root, n_clusters=args.n_clusters)
        return 0

    cfg = load_config(args.city)

    if args.command == "validate":
        print(f"Config for '{args.city}' loaded OK "
              f"({len(cfg)} top-level sections: {sorted(cfg)})")
        return 0

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"=== stage: {stage} ===", flush=True)
        _run_stage(stage, cfg, args.city, args.project_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
