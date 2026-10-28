"""Backend of the `smartbuilding simulate` command: solve a layout, run baselines, tabulate."""

from __future__ import annotations

import re
from collections import defaultdict

from smartbuilding.core.bundle import ConfigBundle
from smartbuilding.core.decision import Baseline
from smartbuilding.core.primitives import Split
from smartbuilding.core.spec import BuildingSpec
from smartbuilding.eval.episode import (
    EpisodeSummary,
    navgraph_for,
    resolve_for,
    run_episode,
    summarise,
)
from smartbuilding.sim.resolve import ResolveError
from smartbuilding.solver.csp import solve


def simulate_report(
    spec: BuildingSpec,
    bundle: ConfigBundle,
    family: str,
    split: str,
    n: int,
    methods: list[str],
    seed: int,
    list_families: bool = False,
) -> tuple[str, bool]:
    out = solve(
        spec, bundle.catalogue, bundle.thresholds, seed=seed, max_layouts=1, time_limit_s=30
    )
    if not out.results:
        return f"no layout: {out.conflict_explanation}", False
    graph = navgraph_for(out.results[0].layout, spec)
    scenarios = [s for s in bundle.scenarios().scenarios if s.split == Split(split)]
    if list_families:
        names = sorted(
            {
                re.sub(r"_(repair|test)_\d+$", "", s.scenario_id).removeprefix("SC_")
                for s in scenarios
            }
        )
        return "\n".join(names), True
    chosen = [s for s in scenarios if s.scenario_id.startswith(f"SC_{family}_")][:n]
    if not chosen:
        return f"no {split} scenarios for family {family!r} (try --list-families)", False
    lines: list[str] = []
    acc: dict[str, list[EpisodeSummary]] = defaultdict(list)
    for sc in chosen:
        try:
            resolved = resolve_for(bundle, sc, graph, spec)
        except ResolveError as exc:
            lines.append(f"skip {sc.scenario_id}: {exc}")
            continue
        for m in methods:
            res, _ = run_episode(bundle, graph, resolved, spec, Baseline(m))
            acc[m].append(summarise(res, bundle))
    lines.append(
        f"{spec.building_type.value}, family '{family}', {len(chosen)} {split} scenarios, seed {seed}"
    )
    lines.append("(dose is a proxy, not FED; these are sanity runs, not experiment results)")
    lines.append(
        f"{'method':6s} {'dose':>7s} {'completion':>10s} {'evac_s':>7s} {'queue_delay_s':>13s}"
    )
    for name, rows in acc.items():
        k = len(rows)
        lines.append(
            f"{name:6s} {sum(r.mean_dose for r in rows) / k:7.2f} {sum(r.completion for r in rows) / k:10.2f} "
            f"{sum((r.mean_evac_s or 0.0) for r in rows) / k:7.1f} "
            f"{sum(r.peak_queue_delay_s for r in rows) / k:13.1f}"
        )
    return "\n".join(lines), True
