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

    cross = sub.add_parser(
        "cross", help="cross-city cityvector clustering + size-gradient read "
                      "over all cities in cityplane.csv")
    cross.add_argument("--n-clusters", type=int, default=3)
    cross.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[2])

    args = parser.parse_args(argv)

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
