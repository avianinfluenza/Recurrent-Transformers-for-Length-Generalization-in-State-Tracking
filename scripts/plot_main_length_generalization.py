#!/usr/bin/env python3
"""Plot per-swap exact-match and slot-accuracy curves with seed SD bands."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SWAP_COUNTS = list(range(2, 81))
SPLIT_REGIONS = (
    (2, 10, "ID\n(2–10)", "#6abb55"),
    (11, 19, "OOD\n(11–19)", "#edc948"),
    (20, 40, "OOD\n(20–40)", "#b279d2"),
    (41, 80, "OOD\n(40–80)", "#5b74b8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "paper" / "main" / "manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "assets" / "main_length_generalization.png",
    )
    parser.add_argument("--pdf", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "paper" / "main" / "per_swap_mean_sd.csv",
    )
    return parser.parse_args()


def load_curve(path: Path) -> dict[str, dict[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = {int(point["n_swaps"]): point for point in payload["points"]}
    missing = set(SWAP_COUNTS) - set(points)
    if missing:
        raise ValueError(f"{path} is missing swap lengths: {sorted(missing)}")
    return {
        metric: {n: float(points[n][metric]) for n in SWAP_COUNTS}
        for metric in ("exact_match", "slot_accuracy")
    }


def summarize(curves: list[dict[int, float]]) -> tuple[list[float], list[float]]:
    means, errors = [], []
    for n in SWAP_COUNTS:
        values = [curve[n] for curve in curves]
        means.append(fmean(values))
        errors.append(stdev(values))
    return means, errors


def add_regions(ax: plt.Axes) -> None:
    for start, end, label, color in SPLIT_REGIONS:
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.075, linewidth=0, zorder=0)
        ax.text((start + end) / 2, 1.02, label, color=color, fontsize=9.5,
                fontweight="semibold", ha="center", va="bottom")
    for boundary in (10.5, 19.5, 40.5):
        ax.axvline(boundary, color="#6f6f6f", linewidth=1.0, linestyle=(0, (4, 4)), alpha=0.55)


def plot_panel(
    ax: plt.Axes,
    summaries: list[dict[str, object]],
    metric: str,
    title: str,
    y_label: str,
) -> None:
    add_regions(ax)
    for summary in summaries:
        means = summary[metric]["means"]
        errors = summary[metric]["errors"]
        color = summary["color"]
        lower = [max(0.0, mean - err) for mean, err in zip(means, errors)]
        upper = [min(1.0, mean + err) for mean, err in zip(means, errors)]
        ax.fill_between(SWAP_COUNTS, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(SWAP_COUNTS, means, color=color, linewidth=2.5,
                label=f"{summary['label']} (mean ± 1 SD)")
    ax.set_xlim(1.5, 80.5)
    ax.set_ylim(-0.015, 1.075)
    ax.set_title(title, pad=22)
    ax.set_xlabel("Number of swap events (N)")
    ax.set_ylabel(y_label)
    ax.set_xticks([2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summaries = []
    for model in manifest["models"]:
        seed_curves = [load_curve((manifest_path.parent / seed).resolve()) for seed in model["seeds"]]
        summary: dict[str, object] = {"label": model["label"], "color": model["color"], "seed_curves": seed_curves}
        for metric in ("exact_match", "slot_accuracy"):
            means, errors = summarize([curve[metric] for curve in seed_curves])
            summary[metric] = {"means": means, "errors": errors}
        summaries.append(summary)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "semibold", "axes.labelsize": 12,
                         "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.2), sharey=True)
    plot_panel(axes[0], summaries, "exact_match", "Exact match", "Exact match accuracy (%)")
    plot_panel(axes[1], summaries, "slot_accuracy", "Slot accuracy", "Slot accuracy (%)")
    # ``sharey=True`` suppresses duplicate tick labels by default; retain them
    # because the two panels are intended to be readable independently.
    axes[1].tick_params(axis="y", labelleft=True)
    axes[0].legend(loc="center right", bbox_to_anchor=(0.995, 0.76), fontsize=9.5)
    axes[1].legend(loc="center right", bbox_to_anchor=(0.995, 0.76), fontsize=9.5)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("metric", "model", "n_swaps", "mean", "sd", "seed_values"))
            for summary in summaries:
                for metric in ("exact_match", "slot_accuracy"):
                    for index, n in enumerate(SWAP_COUNTS):
                        values = [curve[metric][n] for curve in summary["seed_curves"]]
                        writer.writerow((metric, summary["label"], n,
                                         f"{summary[metric]['means'][index]:.8f}",
                                         f"{summary[metric]['errors'][index]:.8f}",
                                         ";".join(f"{value:.8f}" for value in values)))
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
