"""Command-line entry point: ``smartbuilding <command>``."""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel

from smartbuilding import __version__
from smartbuilding.core.base import Contract
from smartbuilding.core.bundle import check_bundle, load_bundle
from smartbuilding.core.loaders import ContractLoadError, load_model
from smartbuilding.core.schemas import CONTRACTS, check_schemas, export_schemas


def _cmd_validate(args: argparse.Namespace) -> int:
    cls = CONTRACTS.get(args.contract)
    if cls is None:
        print(f"unknown contract {args.contract!r}; try 'smartbuilding contracts'", file=sys.stderr)
        return 2
    obj: BaseModel = load_model(args.path, cls)
    if isinstance(obj, Contract):
        print(f"OK {args.contract} hash={obj.compute_hash()}")
    else:
        print(f"OK {args.contract}")
    return 0


def _cmd_hash(args: argparse.Namespace) -> int:
    cls = CONTRACTS.get(args.contract)
    if cls is None or not issubclass(cls, Contract):
        print(f"{args.contract!r} is not a hashable contract", file=sys.stderr)
        return 2
    print(load_model(args.path, cls).compute_hash())
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.run_config)
    problems = check_bundle(bundle)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    scenarios = bundle.scenarios()
    assumptions = bundle.assumptions()
    print(f"scenarios: {len(scenarios.scenarios)} ({args.run_config})")
    print(f"values marked as assumptions: {len(assumptions)}")
    if args.list_assumptions:
        for a in assumptions:
            print(f"  - {a}")
    return 1 if problems else 0


def _cmd_schemas(args: argparse.Namespace) -> int:
    if args.check:
        problems = check_schemas(args.out)
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0
    for path in export_schemas(args.out):
        print(path)
    return 0


def _cmd_check_layout(args: argparse.Namespace) -> int:
    from smartbuilding.core.layout import Layout
    from smartbuilding.core.spec import BuildingSpec
    from smartbuilding.layout.checker import check_layout

    layout: Layout = load_model(args.layout, Layout)
    spec: BuildingSpec = load_model(args.spec, BuildingSpec)
    bundle_path = args.config or Path("configs/run_default.yaml")
    bundle = load_bundle(bundle_path)

    verification = check_layout(layout, spec, bundle.thresholds, bundle.catalogue)
    passing_hard = [r.constraint_id for r in verification.hard if r.satisfied]
    failing_hard = [r.constraint_id for r in verification.hard if not r.satisfied]

    print(f"Layout: {layout.layout_id} (geometry_hash={layout.geometry_hash[:16]}...)")
    print(f"Hard constraints: {len(passing_hard)} passed, {len(failing_hard)} failed")
    if failing_hard:
        print("Failing checks:")
        for r in verification.hard:
            if not r.satisfied:
                print(f"  - {r.constraint_id}: {r.detail}")
    if verification.soft:
        print("Soft scores:")
        for s in verification.soft:
            print(f"  - {s.name}: cost={s.cost:.3f} (weight={s.weight})")
        print(f"Total weighted soft cost: {verification.total_soft_cost:.3f}")
    return 1 if failing_hard else 0


def _cmd_solve(args: argparse.Namespace) -> int:
    from smartbuilding.core.spec import BuildingSpec
    from smartbuilding.solver.csp import solve

    spec: BuildingSpec = load_model(args.spec, BuildingSpec)
    bundle_path = args.config or Path("configs/run_default.yaml")
    bundle = load_bundle(bundle_path)

    output = solve(
        spec,
        bundle.catalogue,
        bundle.thresholds,
        seed=args.seed,
        max_layouts=args.max_layouts,
        time_limit_s=args.time_limit,
    )
    print(f"CSP Solver finished in {output.solve_time_s:.2f}s (seed={output.seed})")
    print(f"Feasible layouts found: {output.feasible_count}")
    if output.infeasible_count and not output.feasible_count:
        print(f"Infeasible. Conflict: {output.conflict_explanation}")

    for idx, r in enumerate(output.results, start=1):
        status = "FEASIBLE" if r.feasible else "INFEASIBLE"
        print(
            f"  [{idx}] {r.layout.layout_id} | {status} | "
            f"CSR={r.constraint_satisfaction_rate * 100:.1f}% | "
            f"soft_cost={r.layout.verification.total_soft_cost:.3f}"
        )
        if args.out_dir:
            out_p = Path(args.out_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            fname = out_p / f"{r.layout.layout_id}.json"
            fname.write_text(r.layout.model_dump_json(indent=2), encoding="utf-8")
            print(f"       Saved: {fname}")
    return 0 if output.feasible_count > 0 else 1


def _cmd_plan_construction(args: argparse.Namespace) -> int:
    from smartbuilding.core.layout import Layout
    from smartbuilding.planning.construction import plan_construction

    layout: Layout = load_model(args.layout, Layout)
    plan = plan_construction(layout, seed=args.seed)

    print(f"Construction Plan: {plan.plan_id}")
    print(f"Total tasks: {len(plan.tasks)}")
    print(f"Total duration: {plan.total_duration_days:.1f} days")
    print(f"Total cost: {plan.total_cost_units:.1f} units")
    print(f"Critical path length: {len(plan.critical_path)} tasks")
    if args.out:
        out_f = Path(args.out)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved to {out_f}")
    return 0


def _cmd_run_e0(args: argparse.Namespace) -> int:
    from smartbuilding.eval.e0 import run_e0

    res = run_e0(seed=args.seed)
    print("=== E0 sanity benchmark (SafeStep-style toy corridor) ===")
    print("mean dose proxy over seeds, and exit used:")
    for key, dose in res.mean_dose.items():
        print(f"  {key:9s} dose {dose:6.2f}   exits {res.exit_counts[key]}")
    print(f"check passed: {res.check_passed} - {res.note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartbuilding")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the package version")
    sub.add_parser("contracts", help="list contract names")

    v = sub.add_parser("validate", help="validate a YAML/JSON file against a contract")
    v.add_argument("contract")
    v.add_argument("path", type=Path)

    h = sub.add_parser("hash", help="print the content hash of a contract file")
    h.add_argument("contract")
    h.add_argument("path", type=Path)

    c = sub.add_parser("check-config", help="load a run config and cross-check all inputs")
    c.add_argument("run_config", type=Path)
    c.add_argument("--list-assumptions", action="store_true")

    s = sub.add_parser("export-schemas", help="write or check JSON Schemas")
    s.add_argument("--out", default="schemas")
    s.add_argument("--check", action="store_true")

    cl = sub.add_parser("check-layout", help="check a layout against a spec and thresholds")
    cl.add_argument("layout", type=Path, help="path to layout JSON")
    cl.add_argument("--spec", required=True, type=Path, help="path to building spec JSON")
    cl.add_argument("--config", type=Path, default=None, help="path to run config YAML")

    sol = sub.add_parser("solve", help="generate layout candidates using CSP search")
    sol.add_argument("--spec", required=True, type=Path, help="path to building spec JSON")
    sol.add_argument("--config", type=Path, default=None, help="path to run config YAML")
    sol.add_argument("--max-layouts", type=int, default=5, help="maximum layouts to return")
    sol.add_argument("--seed", type=int, default=0, help="random seed")
    sol.add_argument("--time-limit", type=float, default=60.0, help="solve time limit in seconds")
    sol.add_argument("--out-dir", type=Path, default=None, help="directory to save layouts")

    cp = sub.add_parser(
        "plan-construction", help="generate construction schedule and critical path"
    )
    cp.add_argument("layout", type=Path, help="path to layout JSON")
    cp.add_argument("--seed", type=int, default=0, help="random seed")
    cp.add_argument("--out", type=Path, default=None, help="output path for construction plan JSON")

    sm = sub.add_parser("simulate", help="solve a layout, then run baselines on a scenario family")
    sm.add_argument("--spec", required=True, help="building spec JSON")
    sm.add_argument("--config", help="run config YAML")
    sm.add_argument("--family", default="normal_fire", help="scenario family (see --list-families)")
    sm.add_argument("--split", default="test", choices=["repair", "test"])
    sm.add_argument("--n", type=int, default=10, help="number of scenarios")
    sm.add_argument("--methods", default="B1,B2,B3,B4,B5", help="comma-separated baselines")
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--list-families", action="store_true")
    ui = sub.add_parser("ui", help="serve the 3D interface (open the printed address in a browser)")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--root", default=".", help="project folder containing configs/ and data/")
    e0 = sub.add_parser("run-e0", help="run E0 SafeStep benchmark (Dijkstra vs SafeStep)")
    e0.add_argument("--config", type=Path, default=None, help="path to run config YAML")
    e0.add_argument("--seed", type=int, default=42, help="random seed")
    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    from smartbuilding.core.spec import BuildingSpec
    from smartbuilding.eval.simulate import simulate_report

    spec: BuildingSpec = load_model(args.spec, BuildingSpec)
    bundle = load_bundle(args.config or Path("configs/run_default.yaml"))
    text, ok = simulate_report(
        spec,
        bundle,
        args.family,
        args.split,
        args.n,
        args.methods.split(","),
        args.seed,
        args.list_families,
    )
    print(text)
    return 0 if ok else 1


def _cmd_ui(args: argparse.Namespace) -> int:
    from smartbuilding.ui.server import serve

    serve(Path(args.root), args.host, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "version":
            print(__version__)
            return 0
        if args.command == "contracts":
            print("\n".join(sorted(CONTRACTS)))
            return 0
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "hash":
            return _cmd_hash(args)
        if args.command == "check-config":
            return _cmd_check_config(args)
        if args.command == "export-schemas":
            return _cmd_schemas(args)
        if args.command == "check-layout":
            return _cmd_check_layout(args)
        if args.command == "solve":
            return _cmd_solve(args)
        if args.command == "plan-construction":
            return _cmd_plan_construction(args)
        if args.command == "ui":
            return _cmd_ui(args)
        if args.command == "simulate":
            return _cmd_simulate(args)
        if args.command == "run-e0":
            return _cmd_run_e0(args)
    except ContractLoadError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
