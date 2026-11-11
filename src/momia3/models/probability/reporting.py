"""Publication-style training curves and final model-comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def write_publication_figures(
    artifact_root: str | Path,
    models: dict[str, str],
    output_root: str | Path,
) -> None:
    """Write training curves and final test metrics for named artifact folders."""
    import matplotlib.pyplot as plt

    artifact_root = Path(artifact_root)
    output_root = Path(output_root)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 120,
    })
    colors = plt.get_cmap("tab10").colors
    histories = {}
    evaluations = {}
    for index, (label, folder) in enumerate(models.items()):
        root = artifact_root / folder
        histories[label] = json.loads((root / "history.json").read_text())
        evaluations[label] = json.loads((root / "evaluation.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for index, (label, history) in enumerate(histories.items()):
        color = colors[index % len(colors)]
        epochs = np.array([row["epoch"] for row in history])
        loss = np.array([row.get("train_loss", row.get("train", {}).get("total", np.nan)) for row in history])
        axes[0].plot(epochs, loss, label=label, color=color, lw=1.0)
        validated = [row for row in history if "validation" in row or "validation_instances" in row]
        x = [row["epoch"] for row in validated]
        y = [
            row.get("validation", row.get("validation_instances", {}))["instances"]["f1"]
            if "validation" in row else row["validation_instances"]["f1"]
            for row in validated
        ]
        axes[1].plot(x, y, label=label, color=color, lw=1.2)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Optimization")
    axes[1].set(xlabel="Epoch", ylabel="Instance F1", title="Validation performance", ylim=(0, 1))
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, output_root / "training_curves")
    plt.close(fig)

    probability = [label for label in histories if "train_loss" in histories[label][0]]
    if probability:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for label in probability:
            history = histories[label]
            color = colors[list(models).index(label) % len(colors)]
            axes[0].plot(
                [row["epoch"] for row in history],
                [row["train_loss"] for row in history],
                label=label, color=color, lw=1.1,
            )
            validated = [row for row in history if "validation" in row]
            axes[1].plot(
                [row["epoch"] for row in validated],
                [row["validation"]["instances"]["f1"] for row in validated],
                label=label, color=color, lw=1.2,
            )
        axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Probability-model optimization")
        axes[1].set(xlabel="Epoch", ylabel="Instance F1", title="Probability-model validation", ylim=(0, 1))
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        _save(fig, output_root / "probability_training_detail")
        plt.close(fig)

    omni = [label for label in histories if "train" in histories[label][0]]
    if omni:
        fig, axes = plt.subplots(1, len(omni) + 1, figsize=(3.5 * (len(omni) + 1), 3.0))
        for index, label in enumerate(omni):
            history = histories[label]
            color = colors[list(models).index(label) % len(colors)]
            epochs = np.array([row["epoch"] for row in history])
            for key_index, key in enumerate(("flow_mse", "direction", "distance", "boundary")):
                values = np.array([row["train"][key] for row in history])
                window = min(25, len(values))
                smooth = np.convolve(values, np.ones(window) / window, mode="valid")
                axes[index].plot(
                    epochs[window - 1:], smooth, lw=1.0,
                    label=key, color=colors[key_index],
                )
            axes[index].set(
                xlabel="Epoch", ylabel="Loss (25-epoch mean)", title=label
            )
            axes[index].legend(frameon=False, fontsize=7)
            validated = [row for row in history if "validation_instances" in row]
            axes[-1].plot(
                [row["epoch"] for row in validated],
                [row["validation_instances"]["f1"] for row in validated],
                label=label, color=color, lw=1.3,
            )
        axes[-1].set(xlabel="Epoch", ylabel="Instance F1", title="Validation performance", ylim=(0, 1))
        axes[-1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        _save(fig, output_root / "omnipose_training_detail")
        plt.close(fig)

    labels = list(models)
    metrics = ("f1", "precision", "recall", "matched_mean_iou")
    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.1), 3.2))
    x = np.arange(len(labels))
    width = 0.19
    for index, metric in enumerate(metrics):
        values = [evaluations[label]["test"]["instances"][metric] for label in labels]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=metric, color=colors[index])
    ax.set(xticks=x, xticklabels=labels, ylabel="Score", title="Held-out test performance", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    _save(fig, output_root / "model_comparison")
    plt.close(fig)
