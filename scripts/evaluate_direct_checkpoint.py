#!/usr/bin/env python3
"""Evaluate a fixed-depth direct Transformer checkpoint on arbitrary JSONL splits."""

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
    make_loader,
    maybe_compile_model,
    resolve_device,
)
from src.original.model import DirectTransformer, OriginalModelConfig, build_model


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

    args.checkpoint = args.checkpoint or _config_path(
        _first(evaluation.get("checkpoint"), config.get("checkpoint"))
    )
    args.data_dir = (
        args.data_dir
        or _config_path(_first(evaluation.get("data_dir"), config.get("data_dir")))
        or ROOT / "data"
    )
    args.out = args.out or _config_path(
        _first(
            evaluation.get("out"),
            evaluation.get("output"),
            config.get("out"),
            config.get("output"),
        )
    )
    args.splits = list(
        _first(args.splits, evaluation.get("splits"), default=list(EVAL_SPLITS))
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
    if args.checkpoint is None:
        parser.error("set --checkpoint or 'checkpoint' in the YAML config")
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML evaluation config; explicit CLI options override its values",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return _apply_config(parser, parser.parse_args(argv))


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config")
    model_state = payload.get("model_state")
    if not isinstance(checkpoint_config, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint must contain config and model_state")
    config = OriginalModelConfig(**checkpoint_config)
    model = build_model(config)
    if not isinstance(model, DirectTransformer):
        raise SystemExit("this evaluator requires a direct Transformer checkpoint")
    model.load_state_dict(model_state, strict=True)
    device = resolve_device(args.device)
    maybe_compile_model(model.to(device), device)

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
            input_format=config.direct_input_format,
        )
        splits[split] = evaluate_classifier(model, loader, device, adaptive_kl=False)

    result = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config) if args.config is not None else None,
        "evaluation_mode": "fixed_depth_direct",
        "num_layers": config.num_layers,
        "input_format": config.direct_input_format,
        "causal": config.direct_causal,
        "position_encoding": config.position_encoding,
        "splits": splits,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
