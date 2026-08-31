import importlib.util
from pathlib import Path
import re

import yaml

from src.data.data import parse_args as parse_data_args
from src.original.experiment import parse_args as parse_run_args
from src.trainer import config_to_argv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_length_matched_checkpoint",
    ROOT / "scripts" / "evaluate_length_matched_checkpoint.py",
)
assert SPEC is not None and SPEC.loader is not None
LENGTH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LENGTH_MODULE)
parse_length_args = LENGTH_MODULE.parse_args

DIRECT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_direct_checkpoint",
    ROOT / "scripts" / "evaluate_direct_checkpoint.py",
)
assert DIRECT_SPEC is not None and DIRECT_SPEC.loader is not None
DIRECT_MODULE = importlib.util.module_from_spec(DIRECT_SPEC)
DIRECT_SPEC.loader.exec_module(DIRECT_MODULE)
parse_direct_args = DIRECT_MODULE.parse_args


def test_length_matched_evaluation_loads_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "runs/example/checkpoint.pt",
                "data_dir": "data/boundary_sweep",
                "device": "cpu",
                "evaluation": {
                    "swaps_per_loop": 1,
                    "splits": ["boundary_11_19"],
                    "fixed_loop_counts": [20, 40],
                    "batch_size": 64,
                    "output": "runs/evaluations/example.json",
                },
            }
        ),
        encoding="utf-8",
    )

    args = parse_length_args(["--config", str(config_path), "--device", "cuda"])

    assert args.checkpoint == ROOT / "runs/example/checkpoint.pt"
    assert args.data_dir == ROOT / "data/boundary_sweep"
    assert args.swaps_per_loop == 1.0
    assert args.splits == ["boundary_11_19"]
    assert args.fixed_loop_counts == [20, 40]
    assert args.eval_batch_size == 64
    assert args.device == "cuda"
    assert args.out == ROOT / "runs/evaluations/example.json"


def test_direct_evaluation_loads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "direct.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "runs/direct/checkpoint.pt",
                "data_dir": "data/boundary_sweep",
                "evaluation": {
                    "splits": ["boundary_11_19"],
                    "batch_size": 32,
                    "output": "runs/evaluations/direct.json",
                },
            }
        ),
        encoding="utf-8",
    )

    args = parse_direct_args(["--config", str(config_path)])

    assert args.checkpoint == ROOT / "runs/direct/checkpoint.pt"
    assert args.data_dir == ROOT / "data/boundary_sweep"
    assert args.splits == ["boundary_11_19"]
    assert args.eval_batch_size == 32
    assert args.out == ROOT / "runs/evaluations/direct.json"


def test_paper_data_configs_use_the_documented_ranges() -> None:
    standard = parse_data_args(
        ["--config", str(ROOT / "configs/data/paper_h10.yaml")]
    )
    boundary = parse_data_args(
        ["--config", str(ROOT / "configs/data/boundary_11_19.yaml")]
    )

    assert standard.l_train == 10
    assert standard.ood_mult == [4, 8]
    assert standard.out == ROOT / "data"
    assert boundary.boundary_sweep
    assert boundary.boundary_min_swaps == 11
    assert boundary.boundary_max_swaps == 19
    assert boundary.out == ROOT / "data/boundary_sweep"


def test_readme_referenced_configs_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"configs/[A-Za-z0-9_./-]+\.ya?ml", readme))

    assert referenced
    assert not [path for path in referenced if not (ROOT / path).is_file()]


def test_paper_training_configs_match_the_training_cli() -> None:
    for config_path in sorted((ROOT / "configs/paper").glob("*.yaml")):
        relative_path = config_path.relative_to(ROOT)
        config = yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))
        args = parse_run_args(config_to_argv(config))

        assert args.online_training
        assert args.curriculum_min_swaps == 2
        assert args.curriculum_max_swaps == 10
        assert args.train_steps == 100_000
