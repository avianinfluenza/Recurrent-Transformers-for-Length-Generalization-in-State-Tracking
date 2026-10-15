#!/usr/bin/env python3
"""Paired order probe of the paper's trained Atomic+NoPE Fan model.

No training or production model mutation. Run from any directory, for example::

    .venv/bin/python scripts/analysis/analyze_causal_order_sensitivity.py \
      --checkpoints checkpoints/causal_order_sensitivity/seed{0,1,2}/checkpoint.pt

The output directory must be new. Tensors contain all K=1..max(K), not just
the requested budgets; one forward_all_loops call per batch and condition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.atomic import ATOMIC_SLOT_IDS, encode_atomic_body
from src.data.collate import collate_fn
from src.data.data import GenConfig, sample_problem
from src.original.data import replay_states
from src.original.model import FanRecurrentTransformer, OriginalModelConfig

DEFAULT_LENGTHS = [2, 3, 4, 6, 8, 10, 12, 15, 19]
DEFAULT_BUDGETS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
PAPER_CONFIG = ROOT / "configs/paper/recurrent_d128_l1_atomic_nope.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate(init: list[int], swaps: list[list[int]]) -> list[int]:
    """Independent explicit simulator; cross-checked with repository replay_states."""
    state = list(init)
    for left, right in swaps:
        state[left], state[right] = state[right], state[left]
    return state


def event_ids(swaps: list[list[int]]) -> list[str]:
    """Ordered entity pair is the token identity; rank distinguishes repeats."""
    counts: Counter = Counter()
    result = []
    for swap in swaps:
        key = tuple(swap)
        result.append(f"SWAP_{key[0]}_{key[1]}#{counts[key]}")
        counts[key] += 1
    return result


def validate_pair(pair: dict) -> None:
    a, b, n = pair["a"], pair["b"], pair["n_swaps"]
    assert a["init"] == b["init"] and sorted(a["init"]) == list(range(5))
    assert a["n_swaps"] == b["n_swaps"] == n
    assert len(a["swaps"]) == len(b["swaps"]) == n
    assert Counter(map(tuple, a["swaps"])) == Counter(map(tuple, b["swaps"]))
    assert a["swaps"] != b["swaps"]
    assert sorted(pair["permutation_b_from_a"]) == list(range(n))
    assert b["swaps"] == [a["swaps"][i] for i in pair["permutation_b_from_a"]]
    assert len(set(pair["event_ids_a"])) == n
    assert pair["event_ids_b"] == [pair["event_ids_a"][i] for i in pair["permutation_b_from_a"]]
    assert pair["event_ids_a"] == event_ids(a["swaps"])
    for row in (a, b):
        assert row["labels"] == simulate(row["init"], row["swaps"])
        assert row["labels"] == replay_states(row["init"], row["swaps"])[-1]
    assert a["labels"] != b["labels"]
    batch = collate_fn([a, b], input_format="atomic")
    assert torch.equal(batch["slot_pos"][0], batch["slot_pos"][1])
    assert batch["slot_pos"][0].tolist() == pair["slot_pos"] == list(range(n + 5, n + 10))
    assert batch["input_ids"].shape == (2, n + 10)
    assert batch["attn_mask"].all()
    assert torch.equal(batch["input_ids"][0, :5], batch["input_ids"][1, :5])
    for name, index in (("a", 0), ("b", 1)):
        assert batch["input_ids"][index].tolist() == pair[f"input_ids_{name}"]
        assert pair[f"input_ids_{name}"][-5:] == list(ATOMIC_SLOT_IDS)


def generate_pairs(lengths: list[int], count: int, seed: int) -> tuple[list[dict], dict]:
    """Uniform source swaps, random permutation, reject unchanged final state.

    A fresh source problem is drawn after every rejection (no searching for a
    successful B on a fixed A). Both the pair and its reverse are deduplicated.
    """
    pairs, stats = [], {}
    for n in lengths:
        rng = random.Random(seed + 100003 * n)
        cfg = GenConfig(min_swaps=n, max_swaps=n)
        accepted, attempts, unchanged, duplicates = 0, 0, 0, 0
        seen = set()
        while accepted < count:
            attempts += 1
            if attempts > max(10000, count * 1000):
                raise RuntimeError(f"pair generation exhausted at N={n}")
            problem = sample_problem(rng, cfg)
            init = problem["init_state"]
            swaps = [list(s) for s in problem["swaps"]]
            permutation = list(range(n))
            rng.shuffle(permutation)
            other = [swaps[i] for i in permutation]
            labels_a, labels_b = simulate(init, swaps), simulate(init, other)
            if labels_a == labels_b:
                unchanged += 1
                continue
            a_key, b_key = tuple(map(tuple, swaps)), tuple(map(tuple, other))
            key = (tuple(init), tuple(sorted((a_key, b_key))))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            identities = event_ids(swaps)
            pair = {
                "pair_id": f"N{n:02d}_{accepted:04d}", "n_swaps": n,
                "split": "ID" if n <= 10 else "Boundary",
                "a": {"init": init, "swaps": swaps, "labels": labels_a, "n_swaps": n},
                "b": {"init": init, "swaps": other, "labels": labels_b, "n_swaps": n},
                "permutation_b_from_a": permutation,
                "event_ids_a": identities,
                "event_ids_b": [identities[i] for i in permutation],
                "b_positions_for_a": [permutation.index(i) + 5 for i in range(n)],
                "slot_pos": list(range(n + 5, n + 10)),
                "input_ids_a": encode_atomic_body(init, swaps) + list(ATOMIC_SLOT_IDS),
                "input_ids_b": encode_atomic_body(init, other) + list(ATOMIC_SLOT_IDS),
            }
            validate_pair(pair)
            pairs.append(pair)
            accepted += 1
        stats[str(n)] = dict(accepted=accepted, attempts=attempts,
                             rejected_same_final=unchanged, rejected_duplicate=duplicates)
    return pairs, stats


@contextmanager
def bidirectional_attention(model: FanRecurrentTransformer) -> Iterator[None]:
    """Override only SelfAttention.forward's causal keyword on this instance.

    Hooks are removed in finally, including on exceptions. No global class
    patches, state_dict changes, padding changes, or persistent model edits.
    """
    handles = []

    def override(module, args, kwargs):
        return args, {**kwargs, "causal": False}

    try:
        for layer in model.shared_layers:
            handles.append(layer.attention.register_forward_pre_hook(override, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def gather_slots(hidden: Tensor, positions: Tensor) -> Tensor:
    return hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))


@torch.inference_mode()
def extract(model: FanRecurrentTransformer, batch: dict, loops: int) -> dict[str, Tensor]:
    """Returned shapes [input, K, slot, feature]; SWAP K=1 is token-aligned later."""
    logits, states = model.forward_all_loops(
        batch["input_ids"], batch["attn_mask"], batch["slot_pos"],
        num_loops=loops, return_hidden_states=True,
    )
    raw = torch.stack([gather_slots(h, batch["slot_pos"]) for h in states], dim=1)
    normalized = torch.stack([
        gather_slots(model.final_norm(h) * batch["attn_mask"][..., None], batch["slot_pos"])
        for h in states
    ], dim=1)
    # A classifier hook in tests independently checks the actual classifier input.
    return dict(hidden=raw, normalized=normalized, logits=logits,
                probabilities=logits.softmax(-1), predictions=logits.argmax(-1),
                first_full_hidden=states[0])


def distances(a: Tensor, b: Tensor) -> dict[str, Tensor]:
    # Double accumulation makes measurement noise smaller than forward noise.
    a, b = a.double(), b.double()
    delta = (a - b).norm(dim=-1)
    scale = (a.norm(dim=-1) + b.norm(dim=-1)) / 2
    return dict(l2=delta, relative_l2=delta / scale.clamp_min(1e-12),
                cosine=F.cosine_similarity(a, b, dim=-1, eps=1e-12),
                scale=scale, max_abs=(a - b).abs().amax(dim=-1))


def invariant(a: Tensor, b: Tensor, atol: float, rtol: float) -> Tensor:
    """Per-vector symmetric L2 tolerance, not a coordinatewise relative test."""
    d = distances(a, b)
    return d["l2"] <= atol + rtol * d["scale"]


def load_checkpoint(path: Path, device: str, dtype: torch.dtype) -> tuple[FanRecurrentTransformer, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"trained checkpoint required: {path}")
    # These user-supplied experiment checkpoints contain plain tensors/configs.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = OriginalModelConfig(**payload["config"])
    paper = yaml.safe_load(PAPER_CONFIG.read_text())
    expected = {**paper["model"], **{k: paper[k] for k in (
        "architecture", "position_encoding", "fan_input_format", "fan_positional_control")}}
    mismatches = {k: (expected[k], getattr(config, k)) for k in expected
                  if getattr(config, k) != expected[k]}
    if mismatches:
        raise ValueError(f"checkpoint does not match paper d128 L1 Atomic+NoPE: {mismatches}")
    model = FanRecurrentTransformer(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device=device, dtype=dtype).eval()
    args_path = path.parent / "args.json"
    run_args = payload.get("run_config") or (json.loads(args_path.read_text()) if args_path.exists() else {})
    checks = {
        "online_training": True, "train_steps": 100000, "batch_size": 128,
        "lr": 0.0003, "weight_decay": 0.01, "grad_clip": 1.0,
        "curriculum_min_swaps": 2, "curriculum_max_swaps": 10,
        "curriculum_steps_per_length": 1000, "swaps_per_loop": 1.0,
        "deep_supervision_weight": 0.0, "slot_first": False,
    }
    mismatches = {k: (v, run_args[k]) for k, v in checks.items()
                  if k in run_args and run_args[k] != v}
    if mismatches:
        raise ValueError(f"training metadata differs from paper: {mismatches}")
    seed = run_args.get("seed")
    metadata = dict(path=str(path.resolve()), sha256=sha256(path),
                    bytes=path.stat().st_size, seed=seed, epoch=payload.get("epoch"),
                    config=asdict(config), saved_config=payload["config"],
                    run_args=run_args, config_matches_paper=True,
                    missing_training_metadata=[k for k in checks if k not in run_args],
                    args_sha256=sha256(args_path) if args_path.exists() else None,
                    strict_state_dict_loading=True)
    return model, metadata


METRICS = ["hidden_l2", "hidden_relative_l2", "hidden_cosine", "normalized_l2",
           "normalized_relative_l2", "logits_l2", "probabilities_l2",
           "prediction_disagreement", "state_prediction_disagreement",
           "exact_a", "exact_b", "slot_accuracy_a", "slot_accuracy_b", "both_exact",
           "changed_gt_slot_accuracy", "unchanged_gt_slot_accuracy",
           "changed_gt_prediction_disagreement", "unchanged_gt_prediction_disagreement",
           "swap_k1_relative_l2", "moved_swap_k1_relative_l2"]


def sample_metrics(x: dict, pairs: list[dict], seed: Any, condition: str) -> list[dict]:
    p, loops = len(pairs), x["hidden"].shape[1]
    h = distances(x["hidden"][0::2], x["hidden"][1::2])
    z = distances(x["normalized"][0::2], x["normalized"][1::2])
    logits = distances(x["logits"][0::2], x["logits"][1::2])
    probs = distances(x["probabilities"][0::2], x["probabilities"][1::2])
    pred_a, pred_b = x["predictions"][0::2], x["predictions"][1::2]
    gt_a = torch.tensor([q["a"]["labels"] for q in pairs], device=pred_a.device)
    gt_b = torch.tensor([q["b"]["labels"] for q in pairs], device=pred_b.device)
    correct_a, correct_b = pred_a == gt_a[:, None], pred_b == gt_b[:, None]
    disagreement = pred_a != pred_b
    result = []
    for i in range(p):
        pair = pairs[i]
        n = pair["n_swaps"]
        positions_b = torch.tensor(pair["b_positions_for_a"], device=pred_a.device)
        swap_a = x["first_full_hidden"][2*i, 5:5+n]
        swap_b = x["first_full_hidden"][2*i+1, positions_b]
        sd = distances(swap_a, swap_b)
        moved = positions_b != torch.arange(5, 5+n, device=pred_a.device)
        changed = gt_a[i] != gt_b[i]
        for k in range(loops):
            row = dict(seed=seed, condition=condition, pair_id=pair["pair_id"],
                       n_swaps=n, split=pair["split"], k=k+1)
            for prefix, d in (("hidden", h), ("normalized", z), ("logits", logits), ("probabilities", probs)):
                keys = ("l2", "relative_l2", "cosine") if prefix in ("hidden", "normalized") else ("l2",)
                for key in keys:
                    row[f"{prefix}_{key}"] = d[key][i, k].mean().item()
                row[f"{prefix}_max_abs"] = d["max_abs"][i, k].max().item()
            row.update(
                prediction_disagreement=disagreement[i,k].float().mean().item(),
                state_prediction_disagreement=int(disagreement[i,k].any()),
                exact_a=int(correct_a[i,k].all()), exact_b=int(correct_b[i,k].all()),
                slot_accuracy_a=correct_a[i,k].float().mean().item(),
                slot_accuracy_b=correct_b[i,k].float().mean().item(),
                both_exact=int(correct_a[i,k].all() & correct_b[i,k].all()),
                changed_gt_slot_accuracy=torch.cat((correct_a[i,k,changed], correct_b[i,k,changed])).float().mean().item(),
                unchanged_gt_slot_accuracy=(torch.cat((correct_a[i,k,~changed], correct_b[i,k,~changed])).float().mean().item()
                                            if (~changed).any() else None),
                changed_gt_prediction_disagreement=disagreement[i,k,changed].float().mean().item(),
                unchanged_gt_prediction_disagreement=disagreement[i,k,~changed].float().mean().item() if (~changed).any() else None,
                swap_k1_relative_l2=sd["relative_l2"].mean().item(),
                moved_swap_k1_relative_l2=sd["relative_l2"][moved].mean().item(),
                hidden_relative_l2_by_slot=json.dumps(h["relative_l2"][i,k].tolist()),
                hidden_l2_by_slot=json.dumps(h["l2"][i,k].tolist()),
                hidden_cosine_by_slot=json.dumps(h["cosine"][i,k].tolist()),
                normalized_relative_l2_by_slot=json.dumps(z["relative_l2"][i,k].tolist()),
                logits_l2_by_slot=json.dumps(logits["l2"][i,k].tolist()),
                prediction_a=json.dumps(pred_a[i,k].tolist()), prediction_b=json.dumps(pred_b[i,k].tolist()),
                min_prediction_margin=min(
                    (x["logits"][2*i,k].topk(2).values.diff(dim=-1).abs()).min().item(),
                    (x["logits"][2*i+1,k].topk(2).values.diff(dim=-1).abs()).min().item()),
            )
            result.append(row)
    return result


def invariance_checks(x: dict, condition: str, pairs: list[dict], seed: Any,
                      atol: float, rtol: float) -> list[dict]:
    rows = []
    selected = slice(0, 1) if condition == "causal" else slice(None)
    for key in ("hidden", "normalized", "logits", "probabilities"):
        a, b = x[key][0::2, selected], x[key][1::2, selected]
        d = distances(a, b)
        passed = invariant(a, b, atol, rtol)
        for k in range(a.shape[1]):
            failures = (~passed[:,k]).nonzero().tolist()
            rows.append(dict(seed=seed, condition=condition, n_swaps=pairs[0]["n_swaps"], k=k+1,
                             tensor=key, passed=bool(passed[:,k].all()),
                             failed_slot_vectors=len(failures),
                             max_relative_l2=d["relative_l2"][:,k].max().item(),
                             max_l2=d["l2"][:,k].max().item(),
                             max_abs=d["max_abs"][:,k].max().item(),
                             atol=atol, rtol=rtol,
                             failed_examples=[dict(pair_id=pairs[i]["pair_id"],slot=s) for i,s in failures[:10]]))
    return rows


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["condition"], row["n_swaps"], row["split"], row["k"])].append(row)
    by_seed = []
    for (seed, condition, n, split, k), values in sorted(groups.items()):
        row = dict(seed=seed, condition=condition, n_swaps=n, split=split, k=k, n_pairs=len(values))
        for metric in METRICS:
            a = np.array([v[metric] for v in values if v[metric] is not None])
            row[metric] = float(a.mean()) if len(a) else None
            row[metric+"_sample_sd"] = float(a.std(ddof=1)) if len(a)>1 else 0.0
        row["exact_match"] = (row["exact_a"] + row["exact_b"]) / 2
        row["slot_accuracy"] = (row["slot_accuracy_a"] + row["slot_accuracy_b"]) / 2
        by_seed.append(row)
    pooled = defaultdict(list)
    for row in by_seed:
        pooled[(row["condition"], row["n_swaps"], row["split"], row["k"])].append(row)
    by_length = []
    for (condition,n,split,k), values in sorted(pooled.items()):
        row = dict(condition=condition,n_swaps=n,split=split,k=k,n_seeds=len(values),n_pairs_per_seed=values[0]["n_pairs"])
        for metric in METRICS+["exact_match","slot_accuracy"]:
            a = np.array([v[metric] for v in values if v[metric] is not None])
            row[metric] = float(a.mean()) if len(a) else None
            row[metric+"_seed_sd"] = float(a.std(ddof=1)) if len(a)>1 else 0.0
        by_length.append(row)
    # Equal weights to lengths within ID/Boundary, then equal weights to seeds.
    split_groups = defaultdict(list)
    for row in by_seed:
        split_groups[(row["seed"],row["condition"],row["split"],row["k"])].append(row)
    split_rows = []
    for (seed,condition,split,k), values in sorted(split_groups.items()):
        row = dict(seed=seed,condition=condition,split=split,k=k,n_lengths=len(values))
        for metric in METRICS+["exact_match","slot_accuracy"]:
            a = [v[metric] for v in values if v[metric] is not None]
            row[metric] = float(np.mean(a)) if a else None
        split_rows.append(row)
    return by_seed, by_length, split_rows


def plot_results(output: Path, by_length: list[dict], split_rows: list[dict], budgets: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "savefig.dpi":180, "pdf.fonttype":42})
    splits = [s for s in ("ID", "Boundary") if any(r["split"]==s for r in split_rows)]
    seeds = sorted({r["seed"] for r in split_rows})
    for number,metric,ylabel in [(1,"hidden_relative_l2","Mean SLOT relative L2"),
                                  (2,"prediction_disagreement","SLOT prediction disagreement")]:
        fig, axes = plt.subplots(1,len(splits),figsize=(6*len(splits),4.3),squeeze=False,layout="constrained")
        for ax,split in zip(axes[0],splits):
            for condition,color in [("causal","#1261a0"),("noncausal","#d35f24")]:
                values = []
                for index,seed in enumerate(seeds):
                    selected = sorted([r for r in split_rows if r["split"]==split and r["condition"]==condition and r["seed"]==seed and r["k"] in budgets],key=lambda r:r["k"])
                    xx, yy = [r["k"] for r in selected], [r[metric] for r in selected]
                    values.append(yy)
                    ax.plot(xx, yy,color=color,lw=0.9,alpha=0.38,ls=["-","--",":"][index%3],label=f"{condition}, seed {seed}")
                values = np.array(values)
                mean,sd = values.mean(0),values.std(0,ddof=1) if len(values)>1 else np.zeros(values.shape[1])
                ax.plot(xx,mean,color=color,lw=2.5,marker="o",ms=3,label=f"{condition}, seed mean")
                ax.fill_between(xx,np.maximum(0,mean-sd),mean+sd,color=color,alpha=0.12)
            ax.set(xlabel="Recurrence K",ylabel=ylabel,title=f"{split} lengths; equal length weights",xticks=budgets)
            if number==1:
                ax.set_yscale("symlog",linthresh=1e-6)
            else:
                ax.set_ylim(-0.02,1.02)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7,ncol=2)
        fig.suptitle("Bands: ±1 seed SD; noncausal is an inference-only control",fontsize=10)
        for ext in ("png","pdf"):
            fig.savefig(output/f"figure{number}.{ext}")
        plt.close(fig)
    lengths = sorted({r["n_swaps"] for r in by_length})
    matrix = np.array([[next(r["hidden_relative_l2"] for r in by_length if r["condition"]=="causal" and r["n_swaps"]==n and r["k"]==k) for k in budgets] for n in lengths])
    from matplotlib.colors import SymLogNorm
    fig,ax=plt.subplots(figsize=(10,5.3),layout="constrained")
    im=ax.imshow(matrix,aspect="auto",cmap="viridis",norm=SymLogNorm(linthresh=1e-6,vmin=0,vmax=max(float(matrix.max()),1e-5)))
    ax.set(xticks=range(len(budgets)),xticklabels=budgets,yticks=range(len(lengths)),
           yticklabels=[f"{n} ({'ID' if n<=10 else 'Boundary'})" for n in lengths],
           xlabel="Recurrence K (columns are sampled budgets)",ylabel="SWAP count N",
           title="Causal SLOT relative L2 — seed mean")
    if min(lengths)<=10<max(lengths):
        ax.axhline(sum(n<=10 for n in lengths)-0.5,color="white",lw=1.5)
    fig.colorbar(im,ax=ax,label="Mean relative L2 (symmetric log scale)")
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure3.{ext}")
    plt.close(fig)


def write_report(output: Path, manifest: dict, rows: list[dict], by_length: list[dict], sanity: list[dict]) -> None:
    causal = [r for r in rows if r["condition"]=="causal"]
    noncausal = [r for r in rows if r["condition"]=="noncausal"]
    failures = [r for r in sanity if not r["passed"]]
    mean = lambda rs,key: float(np.mean([r[key] for r in rs]))
    k1 = [r for r in causal if r["k"]==1]
    k2 = [r for r in causal if r["k"]==2]
    text = [
        "# Causal attention 사건 순서 정보 전달 실험", "",
        f"실행 UTC: {manifest['created_utc']}. Git: `{manifest['git_commit']}`.",
        f"학습된 seeds: {manifest['seeds']}. 길이: {manifest['lengths']}. 길이당 {manifest['pairs_per_length']}쌍.",
        f"주요 K: {manifest['budgets']}; 전체 K=1..{manifest['max_loops']}를 같은 forward에서 수집했다.",
        "",
        "## 코드 확인", "",
        "- `SelfAttention.forward`가 PAD key를 제외하고 `causal=True`이면 strict upper triangle을 −∞로 mask한다. 대각선/self-attention은 허용한다.",
        "- `FanRecurrentTransformer.recurrent_step`은 h←h+embedding 후 공유된 pre-LN Transformer block을 적용한다. h₀=0이며 모든 토큰을 동기 갱신한다. 모델의 step은 한 사건의 symbolic transition과 동일하다고 가정하지 않는다.",
        "- `prepare`는 sqrt(d)로 embedding을 스케일한다. NoPE에서는 계산된 position_ids를 attention에 사용하지 않는다. 매 iteration마다 동일 embedding을 reinject한다.",
        "- atomic body는 INIT 5개 + N개의 ordered-pair SWAP token이다. collate는 body 뒤 PAD, 마지막에 SLOT 5개를 붙인다. 분석은 길이별 batch로 PAD가 없다. SLOT은 `slot_pos`로 gather하며 고정 suffix slicing에 의존하지 않는다.",
        "- `_classify`: 전체 hidden에 `final_norm`(LayerNorm)을 적용하고 mask를 곱한 후 SLOT을 gather한다. shared classifier는 Linear(128,256)→ReLU→Linear(256,5)이다.",
        "- checkpoint의 `config`로 모델을 구성하고 `model_state`를 strict load한다. 구버전 config의 추가 필드는 현재 dataclass default로 채운다. sibling args.json과 paper YAML을 대조했다. checkpoint 저장 epoch=100000은 online step 수이다.",
        "- attention module의 instance forward pre-hook에서 causal keyword만 False로 바꾼다. finally에서 hook을 제거한다. 기존 src·학습 파이프라인·체크포인트는 수정하지 않았다.",
        "",
        "## 방법 및 지표", "",
        "기존 source generator처럼 초기 색상 순열과 ordered distinct-entity swap을 무작위 생성한다(no-op=0). B는 A 사건 occurrence들의 무작위 순열이다. symbolic final state가 같으면 pair 전체를 버리고 새 A를 생성한다. 정방향·역방향 중복 pair를 제거했다. 이 선택은 순서 변경으로 정답이 달라지는 문제에 조건화한 진단 분포이며 원래 ID/Boundary benchmark 정확도의 추정값은 아니다.",
        "동일 token이 반복되어도 A의 SWAP_left_right#occurrence ID를 B에 그대로 운반한다. `b_positions_for_a`로 hidden을 대응시킨다. 위치가 같은 사건도 포함한 평균과 이동한 occurrence만의 평균을 둘 다 기록했다. 원본 symbolic simulator와 독립 simulator의 정답을 대조했다.",
        "SLOT별 L2=‖a−b‖₂, relative L2=2‖a−b‖₂/(‖a‖₂+‖b‖₂), cosine=cos(a,b)이다. 먼저 5개 SLOT을 평균하고 pair·길이를 동일 가중한다. normalized representation에도 같은 지표를 계산한다. prediction disagreement는 5개 SLOT 중 argmax가 다른 비율이며, state disagreement는 하나라도 다른 pair의 비율이다. exact_match는 A/B 각각의 정답에 대한 5-SLOT 동시 정답률이다. changed/unchanged ground-truth SLOT 지표도 분리한다.",
        f"forward dtype={manifest['dtype']}, device={manifest['device']}, PyTorch={manifest['torch_version']}. eval+inference_mode이며 AMP/compile을 사용하지 않는다. 거리 누적은 float64. 불변성 판정은 각 SLOT vector마다 ‖a−b‖₂ ≤ {manifest['atol']} + {manifest['rtol']}·(‖a‖₂+‖b‖₂)/2이다. hidden 크기가 K와 함께 커지므로 raw coordinate의 고정 절대 오차만 사용하지 않는다. logits/probabilities/normalized에도 동일 규칙을 적용한다. argmax 차이는 작은 margin에서도 생길 수 있으므로 연속 출력과 별도 기록한다.",
        "seed SD는 독립 학습 seed 평균의 표준편차이며 신뢰구간이 아니다. 동일 pair를 모든 seed/조건에서 재사용하므로 3 seed 결과를 독립 데이터 3배로 해석하지 않는다.",
        "",
        "## 실제 관측", "",
        f"1. **첫 iteration SLOT**: causal K=1 평균 relative L2={mean(k1,'hidden_relative_l2'):.3g}, 최대 pair-SLOT relative L2={max(max(json.loads(r['hidden_relative_l2_by_slot'])) for r in k1):.3g}. 사전 지정 수치 허용 오차의 통과 여부는 아래 sanity 표와 JSON에 기록했다.",
        f"2. **첫 iteration SWAP**: identity/occurrence 대응 후 causal 평균 relative L2={mean(k1,'swap_k1_relative_l2'):.4g}, 이동한 occurrence에서는 {mean(k1,'moved_swap_k1_relative_l2'):.4g}. SLOT보다 먼저 SWAP hidden에 차이가 관측되는지 이 수치를 기준으로 판단한다.",
        f"3. **두 번째 iteration SLOT**: causal K=2 평균 relative L2={mean(k2,'hidden_relative_l2'):.4g}; 평균 cosine={mean(k2,'hidden_cosine'):.5g}. H2는 구조적 가능성이고 모든 파라미터 설정에 차이가 반드시 생긴다는 명제가 아니다.",
        f"4. **최종 예측**: causal K=1 SLOT disagreement={mean(k1,'prediction_disagreement'):.3%}, K=2={mean(k2,'prediction_disagreement'):.3%}. 아래 K=N 표로 정확한 tracking 여부를 별도로 평가한다.",
        f"5. **긴 sequence**: N=12/15/19 결과는 표와 heatmap을 참조한다. 측정한 최대 N={max(manifest['lengths'])}까지만 근거가 있으며 더 긴 OOD 길이에 대한 결론은 내리지 않는다.",
        f"6. **noncausal 통제**: 모든 측정 K의 평균 relative L2={mean(noncausal,'hidden_relative_l2'):.3g}, 최대 pair-SLOT relative L2={max(max(json.loads(r['hidden_relative_l2_by_slot'])) for r in noncausal):.3g}, 평균 SLOT disagreement={mean(noncausal,'prediction_disagreement'):.3%}. 정확도는 distribution shift 상태의 진단값이며 causal의 성능 효과로 비교하지 않는다.",
        "7. **length generalization 해석**: NoPE가 순서 정보를 원천 차단하지는 않는다. causal prefix로 갱신된 SWAP representation이 다음 recurrent step의 SLOT에 전달될 수 있다. hidden 민감성, 학습 모델의 예측 반응, 각 순서에 맞는 정답 예측은 서로 다른 주장이다. K=N의 높은 paired accuracy가 관측되더라도 올바른 일반적 순차 알고리즘을 학습했다는 증명은 아니다.",
        "",
        "### 길이별 causal 결과 (학습 seed 평균)", "",
        "|N|구간|K|raw rel L2|normalized rel L2|SLOT disagreement|A/B exact 평균|SLOT accuracy|양쪽 모두 exact|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in by_length:
        if r["condition"]=="causal" and r["k"] in {1,2,r["n_swaps"],manifest['max_loops']}:
            text.append(f"|{r['n_swaps']}|{r['split']}|{r['k']}|{r['hidden_relative_l2']:.4g}|{r['normalized_relative_l2']:.4g}|{r['prediction_disagreement']:.2%}|{r['exact_match']:.2%}|{r['slot_accuracy']:.2%}|{r['both_exact']:.2%}|")
    text += ["", "## Sanity check", "",
             f"총 {len(sanity)}개 tensor/K/길이/seed 또는 복원 검사 중 실패 {len(failures)}개. 실패를 성공으로 취급하지 않으며, 실패 시 별도 float64 audit 결과를 확인한다.",
             "모든 pair는 초기 상태·token multiset·sequence length·SLOT 위치 동일, SWAP 순서만 변경, symbolic 정답 상이 조건을 검증했다. checkpoint architecture 및 training metadata 검증도 별도로 실행했다.",
             "", "|검사|건수|실패|최대 relative L2|", "|---|---:|---:|---:|"]
    for condition in ("causal","noncausal","restoration"):
        rr=[r for r in sanity if r['condition']==condition]
        text.append(f"|{condition}|{len(rr)}|{sum(not r['passed'] for r in rr)}|{max((r.get('max_relative_l2',0) for r in rr),default=0):.3g}|")
    text += ["", "## 대표 sample", "",
             "선택 규칙: 각 길이의 첫 번째 생성 pair(성공 사례를 찾아 고르지 않음), 첫 번째 checkpoint seed, causal. 전체 K의 표는 `representative_samples.csv`에 있다."]
    seed = manifest["seeds"][0]
    representatives=[r for r in causal if r["seed"]==seed and r["pair_id"].endswith("_0000")]
    write_csv(output/"representative_samples.csv", representatives)
    pairs = [json.loads(line) for line in (output/"paired_inputs.jsonl").read_text().splitlines()]
    for n in [manifest["lengths"][0],manifest["lengths"][-1]]:
        pair=next(p for p in pairs if p['n_swaps']==n)
        text += ["",f"### {pair['pair_id']} (seed {seed})", "",
                 f"INIT={pair['a']['init']}; A swaps={pair['a']['swaps']}; B swaps={pair['b']['swaps']}.",
                 f"정답 A={pair['a']['labels']}, B={pair['b']['labels']}.", "",
                 "|K|raw rel L2|norm rel L2|예측 A|예측 B|exact A/B|", "|---:|---:|---:|---|---|---|"]
        for r in representatives:
            if r['n_swaps']==n:
                text.append(f"|{r['k']}|{r['hidden_relative_l2']:.4g}|{r['normalized_relative_l2']:.4g}|{r['prediction_a']}|{r['prediction_b']}|{r['exact_a']}/{r['exact_b']}|")
    text += ["", "## 그림", "", "![SLOT sensitivity](figure1.png)", "", "![Prediction disagreement](figure2.png)", "", "![Length by K](figure3.png)",
             "", "## 재현성과 한계", "",
             "`execution_config.json`에 Git 상태, source SHA256, 모델 설정, checkpoint 경로/SHA256, 학습 인자, evaluation seed, package version, thread/dtype/device, command를 저장했다. `checkpoint_sources.json`은 Drive 출처이며 내려받은 checkpoint는 gitignored `checkpoints/causal_order_sensitivity/seed*/checkpoint.pt`에 있다. 과거 논문 결과 JSON은 checkpoint SHA256을 제공하지 않으므로 논문 당시 파일과 바이트 단위 동일성까지 입증할 수는 없다. 모델과 학습 설정의 일치를 검증했다.",
             "`tensors/seed*_condition_N*.npz`: pair_ids, k, hidden/normalized/logits/probabilities/predictions(shape [pair, A/B, K, SLOT, feature]), labels, occurrence로 대응한 swap_hidden_k1(shape [pair,A/B,event,d]), event mapping. 원본 logits/probabilities 및 128차원 representation은 CSV에 문자열로 중복하지 않고 NPZ에 보존했다. per_sample_per_k.csv에는 모든 K의 슬롯별 거리 배열과 예측을 포함한다.",
             "H1은 NoPE·L1·동일 INIT/SLOT·eval 모드에 의존한다. L>1이면 첫 recurrent iteration 안에서도 SWAP→SLOT 전달이 가능하므로 같은 명제를 일반화할 수 없다. H3는 padding mask까지 permutation에 맞게 처리한다는 가정하에서 모든 유한 K의 대수적 성질이며, 유한 실험은 수치 구현의 sanity check이다. 반복 동일 token의 occurrence 대응은 명시적으로 운반한 event ID에 따른 것으로, 입력 자체가 occurrence ID를 추가로 포함하지는 않는다.",
             "학습된 intermediate-K classifier 출력은 해당 K로 모두 학습된 독립 모델의 성능이 아니다. K가 증가할수록 정확도가 단조 증가하거나 raw hidden 차이가 의미 있는 순서 알고리즘을 나타낸다고 가정하지 않는다. noncausal accuracy로 attention의 성능 인과 효과를 추정하지 않는다.",
             "제공된 [참고 논문](https://openreview.net/pdf?id=r9gIjLoGX7)은 OpenReview의 browser verification 때문에 본 실행에서 본문을 확보하지 못했다. 이 보고서의 모델 명세와 결과 해석은 로컬 코드·paper YAML·저장된 학습 인자·실제 평가에 근거한다.",
             "", "재실행 명령:", "", "```sh", manifest["command"], "```", ""]
    (output/"report_ko.md").write_text("\n".join(text),encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints",type=Path,nargs="+",required=True)
    parser.add_argument("--lengths",type=int,nargs="+",default=DEFAULT_LENGTHS)
    parser.add_argument("--budgets",type=int,nargs="+",default=DEFAULT_BUDGETS)
    parser.add_argument("--pairs-per-length",type=int,default=200)
    parser.add_argument("--eval-seed",type=int,default=20261015)
    parser.add_argument("--batch-pairs",type=int,default=25)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",default="cpu",choices=["cpu","mps","cuda"])
    parser.add_argument("--dtype",default="float32",choices=["float32","float64"])
    parser.add_argument("--atol",type=float,default=None)
    parser.add_argument("--rtol",type=float,default=None)
    parser.add_argument("--output",type=Path,default=ROOT/"results/analysis/causal_order_sensitivity")
    parser.add_argument("--source-manifest",type=Path,default=ROOT/"checkpoints/causal_order_sensitivity/source_manifest.json")
    parser.add_argument("--no-tensors",action="store_true",help="For quick smoke runs only")
    args=parser.parse_args()
    if min(args.lengths)<2 or min(args.budgets)<1 or max(args.budgets)<2:
        parser.error("N>=2 and maximum K>=2 are required")
    if min(args.pairs_per_length,args.batch_pairs,args.threads)<1:
        parser.error("counts/threads must be positive")
    args.lengths=sorted(set(args.lengths)); args.budgets=sorted(set(args.budgets))
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.eval_seed); random.seed(args.eval_seed); np.random.seed(args.eval_seed)
    dtype=getattr(torch,args.dtype)
    atol=args.atol if args.atol is not None else (1e-5 if dtype==torch.float32 else 1e-10)
    rtol=args.rtol if args.rtol is not None else (1e-5 if dtype==torch.float32 else 1e-10)
    models,metadata=[],[]
    for path in args.checkpoints:
        model,meta=load_checkpoint(path,args.device,dtype)
        if meta['seed'] is None:
            raise ValueError(f"training seed metadata required in checkpoint or sibling args.json: {path}")
        models.append(model);metadata.append(meta)
    seeds=[m['seed'] for m in metadata]
    if len(seeds)!=len(set(seeds)):
        raise ValueError("duplicate training seeds; choose one checkpoint per seed")
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/"tensors").mkdir()
    started=time.perf_counter()
    pairs,stats=generate_pairs(args.lengths,args.pairs_per_length,args.eval_seed)
    with (output/"paired_inputs.jsonl").open("w") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair,ensure_ascii=False)+"\n")
    import shlex
    source_files=[Path(__file__).resolve(),ROOT/"tests/test_causal_order_sensitivity.py",PAPER_CONFIG,
                  ROOT/"src/original/model.py",ROOT/"src/data/atomic.py",ROOT/"src/data/collate.py",ROOT/"src/model/classifier.py"]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
                  command=shlex.join([sys.executable,*sys.argv]),
                  seeds=seeds,checkpoints=metadata,eval_seed=args.eval_seed,lengths=args.lengths,budgets=args.budgets,
                  max_loops=max(args.budgets),pairs_per_length=args.pairs_per_length,pair_generation=stats,
                  dtype=args.dtype,device=args.device,atol=atol,rtol=rtol,
                  tolerance="per SLOT vector: L2 <= atol + rtol * mean(norm(A),norm(B))",
                  torch_version=torch.__version__,numpy_version=np.__version__,python_version=sys.version,
                  platform=platform.platform(),threads=torch.get_num_threads(),deterministic_algorithms=True,
                  amp=False,compile=False,tensors_saved=not args.no_tensors,
                  source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in source_files if p.is_file()},
                  paired_inputs_sha256=sha256(output/"paired_inputs.jsonl"))
    write_json(output/"execution_config.json",manifest)
    if args.source_manifest.is_file():
        write_json(output/"checkpoint_sources.json",json.loads(args.source_manifest.read_text()))
    all_rows,sanity,swap_rows=[],[],[]
    for model,meta in zip(models,metadata):
        seed=meta['seed']
        for n in args.lengths:
            length_pairs=[p for p in pairs if p['n_swaps']==n]
            chunks={c:defaultdict(list) for c in ('causal','noncausal')}
            for start in range(0,len(length_pairs),args.batch_pairs):
                pp=length_pairs[start:start+args.batch_pairs]
                batch={k:v.to(args.device) for k,v in collate_fn([p[s] for p in pp for s in ('a','b')],input_format='atomic').items()}
                causal=extract(model,batch,manifest['max_loops'])
                with bidirectional_attention(model):
                    noncausal=extract(model,batch,manifest['max_loops'])
                restored=extract(model,batch,manifest['max_loops'])
                # Every batch, every K: exact restoration of continuous outputs.
                restored_ok=all(torch.equal(causal[k],restored[k]) for k in causal)
                sanity.append(dict(seed=seed,condition='restoration',n_swaps=n,batch_start=start,
                                   passed=restored_ok,max_relative_l2=0.0 if restored_ok else float(distances(causal['hidden'],restored['hidden'])['relative_l2'].max())))
                del restored
                for condition,x in [('causal',causal),('noncausal',noncausal)]:
                    sanity.extend(invariance_checks(x,condition,pp,seed,atol,rtol))
                    all_rows.extend(sample_metrics(x,pp,seed,condition))
                    aligned_swaps=[]
                    for i,pair in enumerate(pp):
                        aa=x['first_full_hidden'][2*i,5:5+n]
                        bb=x['first_full_hidden'][2*i+1,pair['b_positions_for_a']]
                        aligned_swaps.append(torch.stack((aa,bb)))
                        ds=distances(aa,bb)
                        for j,identity in enumerate(pair['event_ids_a']):
                            swap_rows.append(dict(seed=seed,condition=condition,n_swaps=n,pair_id=pair['pair_id'],event_id=identity,
                                                  position_a=5+j,position_b=pair['b_positions_for_a'][j],
                                                  moved=5+j!=pair['b_positions_for_a'][j],
                                                  hidden_l2=ds['l2'][j].item(),relative_l2=ds['relative_l2'][j].item(),cosine=ds['cosine'][j].item()))
                    if not args.no_tensors:
                        for key in ('hidden','normalized','logits','probabilities','predictions'):
                            shape=x[key].shape
                            chunks[condition][key].append(x[key].cpu().numpy().reshape(len(pp),2,*shape[1:]))
                        chunks[condition]['swap_hidden_k1'].append(torch.stack(aligned_swaps).cpu().numpy())
            for condition,arrays in chunks.items():
                if not args.no_tensors:
                    np.savez_compressed(output/'tensors'/f"seed{seed}_{condition}_N{n:02d}.npz",
                                        **{k:np.concatenate(v) for k,v in arrays.items()},
                                        pair_ids=np.array([p['pair_id'] for p in length_pairs]),
                                        k=np.arange(1,manifest['max_loops']+1),
                                        labels=np.array([[p['a']['labels'],p['b']['labels']] for p in length_pairs]),
                                        b_positions_for_a=np.array([p['b_positions_for_a'] for p in length_pairs]))
            print(f"seed={seed} N={n}: {len(length_pairs)} pairs, K=1..{manifest['max_loops']}, both masks; elapsed={time.perf_counter()-started:.1f}s",flush=True)
            write_json(output/'sanity_checks.json',sanity)
    write_csv(output/'per_sample_per_k.csv',all_rows)
    write_csv(output/'swap_hidden_k1.csv',swap_rows)
    by_seed,by_length,split_rows=aggregate(all_rows)
    write_csv(output/'aggregate_by_seed.csv',by_seed)
    write_csv(output/'aggregate.csv',by_length)
    write_csv(output/'aggregate_by_split_seed.csv',split_rows)
    plot_results(output,by_length,split_rows,args.budgets)
    manifest['elapsed_seconds']=time.perf_counter()-started
    manifest['invariance_failed_checks']=sum(not r['passed'] for r in sanity)
    manifest['complete']=manifest['invariance_failed_checks']==0
    write_json(output/'execution_config.json',manifest)
    write_report(output,manifest,all_rows,by_length,sanity)
    print(f"Results: {output}; failed checks: {manifest['invariance_failed_checks']}",flush=True)
    if not manifest['complete']:
        raise SystemExit("Invariance/restoration check failed. Inspect sanity_checks.json; repeat failing inputs in float64 before interpreting as structural failure.")


if __name__=='__main__':
    main()
