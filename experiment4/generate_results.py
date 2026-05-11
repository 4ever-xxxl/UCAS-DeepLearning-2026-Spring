from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_metrics(output_dir: Path) -> dict:
    with open(output_dir / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(metrics: dict, output_dir: Path) -> None:
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_ppl = [h["train_ppl"] for h in history]
    val_ppl = [h["val_ppl"] for h in history]
    val_bleu = [h.get("val_bleu", 0) for h in history]
    lr = [h.get("learning_rate", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", label="Train Loss")
    ax.plot(epochs, val_loss, "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[0, 1]
    ax.plot(epochs, train_ppl, "b-", label="Train PPL")
    ax.plot(epochs, val_ppl, "r-", label="Val PPL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Training & Validation Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BLEU
    ax = axes[1, 0]
    ax.plot(epochs, val_bleu, "g-", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU-4")
    ax.set_title("Validation BLEU-4 Score")
    ax.grid(True, alpha=0.3)
    if val_bleu:
        ax.axhline(y=max(val_bleu), color="g", linestyle="--", alpha=0.5, label=f"Best: {max(val_bleu):.2f}")
        ax.legend()

    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, lr, "m-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_translation_samples(output_dir: Path, num_samples: int = 8) -> None:
    """Generate a table of sample translations.

    This function is a placeholder — actual translations are generated during
    training evaluation. The train script prints sample translations to stdout.
    We create a simple summary figure showing the final metrics.
    """
    metrics = load_metrics(output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    text = (
        f"Neural Machine Translation Results\n"
        f"{'=' * 40}\n\n"
        f"Best Val BLEU-4:  {metrics['best_val_bleu']:.2f}\n"
        f"Test Loss:        {metrics['test_loss']:.4f}\n"
        f"Test Perplexity:  {metrics['test_ppl']:.2f}\n"
        f"Test BLEU-4:      {metrics['test_bleu']:.2f}\n\n"
        f"Target: BLEU-4 >= 14.0"
    )

    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=13, fontfamily="monospace",
            verticalalignment="center", bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8})

    fig.savefig(output_dir / "results_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_sample_translations(output_dir: Path) -> None:
    """Print sample translations from the last evaluation."""
    metrics = load_metrics(output_dir)
    print(f"\n{'=' * 60}")
    print(f"Experiment 4: Transformer NMT Results Summary")
    print(f"{'=' * 60}")
    print(f"Best Validation BLEU-4: {metrics['best_val_bleu']:.2f}")
    print(f"Test Loss:              {metrics['test_loss']:.4f}")
    print(f"Test Perplexity:        {metrics['test_ppl']:.2f}")
    print(f"Test BLEU-4:            {metrics['test_bleu']:.2f}")
    print(f"{'=' * 60}")


def main() -> None:
    output_dir = Path("experiment4/outputs")
    if not (output_dir / "metrics.json").exists():
        print("metrics.json not found. Run train_nmt.py first.")
        return

    print("Loading metrics...")
    metrics = load_metrics(output_dir)

    print("Plotting training curves...")
    plot_training_curves(metrics, output_dir)

    print("Generating results summary...")
    plot_translation_samples(output_dir)

    print_sample_translations(output_dir)
    print(f"\nAll visualizations saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
