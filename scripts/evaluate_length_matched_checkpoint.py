#!/usr/bin/env python3
"""Run diagnostic length-matched oracle evaluation from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.original.experiment import (
    EVAL_SPLITS,
    evaluate_classifier,
    evaluate_length_matched_classifier,
    make_loader,
    maybe_compile_model,
    resolve_device,
)
from src.original.model import (
    FanRecurrentTransformer,
    OriginalModelConfig,
    RecurrentR0Transformer,
    RecurrentTransformer,
    build_model,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("evaluation config must contain a YAML mapping")
    return loaded


def _config_path(value: object | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _first(*values: object | None, default: object | None = None) -> object | None:
    return next((value for value in values if value is not None), default)


def _apply_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    config: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    if args.config is not None:
        try:
            config = _load_yaml(args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
        raw_evaluation = config.get("evaluation") or {}
        if not isinstance(raw_evaluation, dict):
            parser.error("config 'evaluation' must be a YAML mapping")
        evaluation = raw_evaluation

    config_checkpoint = _first(evaluation.get("checkpoint"), config.get("checkpoint"))
    config_data_dir = _first(evaluation.get("data_dir"), config.get("data_dir"))
    config_out = _first(
        evaluation.get("out"),
        evaluation.get("output"),
        config.get("out"),
        config.get("output"),
    )
    args.checkpoint = args.checkpoint or _config_path(config_checkpoint)
    args.data_dir = args.data_dir or _config_path(config_data_dir) or ROOT / "data"
    args.out = args.out or _config_path(config_out)
    args.swaps_per_loop = _first(
        args.swaps_per_loop,
        evaluation.get("swaps_per_loop"),
        config.get("swaps_per_loop"),
    )
    args.splits = list(
        _first(args.splits, evaluation.get("splits"), default=list(EVAL_SPLITS))
    )
    fixed_loop_counts = _first(
        args.fixed_loop_counts,
        evaluation.get("fixed_loop_counts"),
        default=None,
    )
    args.fixed_loop_counts = (
        None if fixed_loop_counts is None else [int(value) for value in fixed_loop_counts]
    )
    args.device = str(
        _first(
            args.device,
            evaluation.get("device"),
            config.get("device"),
            default="auto",
        )
    )
    args.eval_batch_size = int(
        _first(args.eval_batch_size, evaluation.get("batch_size"), default=128)
    )
    max_eval_samples = _first(
        args.max_eval_samples,
        evaluation.get("max_samples"),
        default=None,
    )
    args.max_eval_samples = (
        None if max_eval_samples is None else int(max_eval_samples)
    )
    args.periodic_atomic_positions = bool(
        _first(
            args.periodic_atomic_positions,
            evaluation.get("periodic_atomic_positions"),
            default=False,
        )
    )
    periodic_position_max_swaps = _first(
        args.periodic_position_max_swaps,
        evaluation.get("periodic_position_max_swaps"),
        default=None,
    )
    args.periodic_position_max_swaps = (
        None
        if periodic_position_max_swaps is None
        else int(periodic_position_max_swaps)
    )
    args.slot_first = _first(
        args.slot_first,
        evaluation.get("slot_first"),
        default=None,
    )

    if args.checkpoint is None:
        parser.error("set --checkpoint or 'checkpoint' in the YAML config")
    if args.swaps_per_loop is None:
        parser.error(
            "set --swaps-per-loop or 'evaluation.swaps_per_loop' in the YAML config"
        )
    args.swaps_per_loop = float(args.swaps_per_loop)
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML evaluation config; explicit CLI options override its values",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--swaps-per-loop", type=float)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="JSONL stems to evaluate relative to --data-dir",
    )
    parser.add_argument(
        "--fixed-loop-counts",
        type=int,
        nargs="+",
        default=None,
        help="also evaluate every sample at these fixed recurrence counts",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--periodic-atomic-positions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "diagnostic eval-only override for atomic+sinusoidal Fan checkpoints: "
            "repeat swap position ids after the trained max swap length"
        ),
    )
    parser.add_argument(
        "--periodic-position-max-swaps",
        type=int,
        default=None,
        help="trained swap-window length for --periodic-atomic-positions; defaults to checkpoint args.json curriculum_max_swaps",
    )
    parser.add_argument(
        "--slot-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override checkpoint run metadata; otherwise auto-detected from sibling args.json",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    return _apply_config(parser, parser.parse_args(argv))


def main() -> None:
    args = parse_args()
    if args.swaps_per_loop <= 0:
        raise SystemExit("--swaps-per-loop must be positive")
    if args.fixed_loop_counts and any(loop_count < 1 for loop_count in args.fixed_loop_counts):
        raise SystemExit("--fixed-loop-counts must all be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config")
    model_state = payload.get("model_state")
    if not isinstance(checkpoint_config, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint must contain config and model_state")
    config = OriginalModelConfig(**checkpoint_config)
    model = build_model(config)
    if not isinstance(
        model,
        (RecurrentTransformer, RecurrentR0Transformer, FanRecurrentTransformer),
    ):
        raise SystemExit("length-matched evaluation requires a recurrent checkpoint")
    model.load_state_dict(model_state, strict=True)

    saved_run_args: dict[str, object] = {}
    checkpoint_run_config = payload.get("run_config")
    if isinstance(checkpoint_run_config, dict):
        saved_run_args = checkpoint_run_config
    for run_args_path in (
        args.checkpoint.parent / "args.json",
        args.checkpoint.parent.parent / "args.json",
    ):
        if run_args_path.is_file():
            loaded_args = json.loads(run_args_path.read_text(encoding="utf-8"))
            if isinstance(loaded_args, dict):
                saved_run_args = loaded_args
            break
    slot_first = (
        bool(saved_run_args.get("slot_first", False))
        if args.slot_first is None
        else args.slot_first
    )
    periodic_position_max_swaps = args.periodic_position_max_swaps
    if args.periodic_atomic_positions:
        if not isinstance(model, FanRecurrentTransformer):
            raise SystemExit("--periodic-atomic-positions requires a fan-recurrent checkpoint")
        if config.fan_input_format != "atomic" or config.position_encoding != "sinusoidal":
            raise SystemExit(
                "--periodic-atomic-positions requires atomic input and sinusoidal positions"
            )
        if periodic_position_max_swaps is None:
            saved_max_swaps = saved_run_args.get("curriculum_max_swaps")
            if isinstance(saved_max_swaps, int) and not isinstance(saved_max_swaps, bool):
                periodic_position_max_swaps = saved_max_swaps
            else:
                raise SystemExit(
                    "--periodic-position-max-swaps is required when args.json does not provide curriculum_max_swaps"
                )
        if periodic_position_max_swaps < 1:
            raise SystemExit("--periodic-position-max-swaps must be positive")
        model.use_atomic_periodic_positions(periodic_position_max_swaps)

    device = resolve_device(args.device)
    model = maybe_compile_model(model.to(device), device)

    splits: dict[str, object] = {}
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing dataset split: {path}")
        loader = make_loader(
            path,
            config.architecture,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=0,
            max_samples=args.max_eval_samples,
            slot_first=slot_first,
            swaps_per_loop=args.swaps_per_loop,
            input_format=(
                config.fan_input_format
                if isinstance(model, FanRecurrentTransformer)
                else "template"
            ),
        )
        metrics = evaluate_length_matched_classifier(
            model,
            loader,
            device,
            swaps_per_loop=args.swaps_per_loop,
        )
        if args.fixed_loop_counts:
            metrics["fixed_loop_sweep"] = {
                str(loop_count): evaluate_classifier(
                    model,
                    loader,
                    device,
                    adaptive_kl=False,
                    num_loops=loop_count,
                )
                for loop_count in sorted(set(args.fixed_loop_counts))
            }
        splits[split] = metrics
    result = {
        "evaluation_mode": (
            "length_matched"
            if isinstance(model, FanRecurrentTransformer)
            else "length_matched_oracle"
        ),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config) if args.config is not None else None,
        "swaps_per_loop": args.swaps_per_loop,
        "fixed_loop_counts": sorted(set(args.fixed_loop_counts or [])),
        "slot_first": slot_first,
        "position_override": {
            "mode": (
                "atomic_periodic_eval_override"
                if args.periodic_atomic_positions
                else "atomic_periodic_config"
                if config.atomic_position_period is not None
                else "default"
            ),
            "max_swaps": (
                periodic_position_max_swaps
                if args.periodic_atomic_positions
                else config.atomic_position_period
            ),
        },
        "splits": splits,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
