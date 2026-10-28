import ast
import json
from pathlib import Path

from smartbuilding.cli import main
from smartbuilding.core.schemas import CONTRACTS, check_schemas, export_schemas
from tests.factories import CONFIGS, EXAMPLES, ROOT

SRC = ROOT / "src" / "smartbuilding"


def test_committed_schemas_match_code():
    problems = check_schemas(ROOT / "schemas")
    assert problems == [], f"schemas are stale; run `make schemas`: {problems}"


def test_export_is_deterministic(tmp_path):
    export_schemas(tmp_path / "a")
    export_schemas(tmp_path / "b")
    for f in (tmp_path / "a").iterdir():
        assert f.read_text() == (tmp_path / "b" / f.name).read_text()


def test_check_detects_drift(tmp_path):
    export_schemas(tmp_path)
    victim = tmp_path / "Layout.schema.json"
    victim.write_text(victim.read_text().replace("grid_w", "grid_width"))
    (tmp_path / "Stray.schema.json").write_text("{}")
    (tmp_path / "NavGraph.schema.json").unlink()
    problems = check_schemas(tmp_path)
    assert "outdated: Layout.schema.json" in problems
    assert "unexpected: Stray.schema.json" in problems
    assert "missing: NavGraph.schema.json" in problems


def test_every_schema_is_valid_json():
    for name in CONTRACTS:
        json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text())


# ---- CLI


def test_cli_validate_and_hash(capsys):
    assert main(["validate", "BuildingSpec", str(EXAMPLES / "school_spec.json")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK BuildingSpec hash=")
    assert main(["hash", "BuildingSpec", str(EXAMPLES / "school_spec.json")]) == 0
    assert len(capsys.readouterr().out.strip()) == 64


def test_cli_rejects_bad_input(tmp_path, capsys):
    bad = tmp_path / "x.json"
    bad.write_text('{"floors": 0}')
    assert main(["validate", "BuildingSpec", str(bad)]) == 1
    assert "validation error" in capsys.readouterr().err
    assert main(["validate", "Nope", str(bad)]) == 2
    assert main(["hash", "Observation", str(bad)]) == 2


def test_cli_check_config_and_misc(capsys):
    assert main(["check-config", str(CONFIGS / "run_default.yaml"), "--list-assumptions"]) == 0
    out = capsys.readouterr().out
    assert "scenarios:" in out and "assumptions" in out and "thresholds.min_exit_count" in out
    assert main(["version"]) == 0
    assert main(["contracts"]) == 0
    assert "Layout" in capsys.readouterr().out


def test_cli_export_schemas(tmp_path):
    assert main(["export-schemas", "--out", str(tmp_path)]) == 0
    assert main(["export-schemas", "--out", str(tmp_path), "--check"]) == 0


# ---- information barrier
#
# Ground truth lives in smartbuilding.sim.truth. Decision code (planners, belief, RL, GenAI and
# the contract modules they use) must never import smartbuilding.sim or name GroundTruthState.
# Allowed: the simulator itself, evaluation code, the schema registry and oracle baselines.

DECISION_PACKAGES = ["planning", "rl", "uncertainty", "genai", "layout"]
TRUTH_ALLOWED_PREFIXES = ("sim/", "eval/")
TRUTH_ALLOWED_FILES = {"core/schemas.py"}
ORACLE_FILES = {"oracle.py"}


def _mentions_truth(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("smartbuilding.sim"):
                return True
            if any(a.name == "GroundTruthState" for a in node.names):
                return True
        if isinstance(node, ast.Import) and any(
            a.name.startswith("smartbuilding.sim") for a in node.names
        ):
            return True
        if isinstance(node, ast.Name) and node.id == "GroundTruthState":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "GroundTruthState":
            return True
    return False


def _truth_offenders(root: Path) -> list[str]:
    offenders = []
    for f in root.rglob("*.py"):
        rel = f.relative_to(root).as_posix()
        if f.name in ORACLE_FILES or rel in TRUTH_ALLOWED_FILES:
            continue
        if rel.startswith(TRUTH_ALLOWED_PREFIXES):
            continue
        if _mentions_truth(f):
            offenders.append(rel)
    return sorted(offenders)


def test_only_simulator_and_evaluation_touch_ground_truth():
    assert _truth_offenders(SRC) == []


def test_core_contract_modules_do_not_define_or_import_truth():
    import smartbuilding.core.runtime as runtime

    assert not hasattr(runtime, "GroundTruthState")
    assert not _mentions_truth(SRC / "core" / "runtime.py")


def test_decision_packages_are_scanned_when_they_exist():
    """Documents which packages the barrier covers; missing ones are skipped until built."""
    covered = [p for p in DECISION_PACKAGES if (SRC / p).is_dir()]
    for pkg in covered:
        assert not [f for f in (SRC / pkg).rglob("*.py") if _mentions_truth(f)]


def test_barrier_detector_catches_violations(tmp_path):
    root = tmp_path / "smartbuilding"
    (root / "planning").mkdir(parents=True)
    (root / "sim").mkdir()
    (root / "planning" / "bad_import.py").write_text(
        "from smartbuilding.sim.truth import GroundTruthState\n"
    )
    (root / "planning" / "bad_module.py").write_text("import smartbuilding.sim.engine\n")
    (root / "planning" / "bad_name.py").write_text("x = rt.GroundTruthState\n")
    (root / "planning" / "oracle.py").write_text("from smartbuilding.sim.truth import X\n")
    (root / "planning" / "good.py").write_text(
        "from smartbuilding.core.runtime import BeliefState\n"
    )
    (root / "sim" / "engine.py").write_text(
        "from smartbuilding.sim.truth import GroundTruthState\n"
    )
    assert _truth_offenders(root) == [
        "planning/bad_import.py",
        "planning/bad_module.py",
        "planning/bad_name.py",
    ]


def test_no_absolute_paths_in_configs():
    for f in CONFIGS.glob("*.yaml"):
        text = f.read_text()
        assert "/home/" not in text and "C:\\" not in text, f.name


def test_no_module_shadows_a_stdlib_module():
    """A file named like a stdlib module breaks tools when run from its own folder."""
    import sys

    for f in SRC.rglob("*.py"):
        assert f.stem not in sys.stdlib_module_names or f.stem == "__init__", f
