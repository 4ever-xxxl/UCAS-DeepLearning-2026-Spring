from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

from train_poetry import ExperimentConfig, PoetryModel, generate_poem, resolve_device


def plot_training_curves(history: list[dict], output_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_ppl = [h["train_perplexity"] for h in history]
    val_ppl = [h["val_perplexity"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_ppl, "b-", label="Train PPL", linewidth=1.5)
    ax2.plot(epochs, val_ppl, "r-", label="Val PPL", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Training & Validation Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Poetry LSTM Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")


def generate_demo_samples(
    model: torch.nn.Module,
    ix2word: dict,
    word2ix: dict,
    device: torch.device,
    output_dir: Path,
) -> None:
    test_starts = [
        "湖光秋月两相和",
        "朝辞白帝彩云间",
        "床前明月光",
        "春眠不觉晓",
        "白日依山尽",
        "独在异乡为异客",
        "两个黄鹂鸣翠柳",
        "日照香炉生紫烟",
    ]

    lines: list[str] = ["# 唐诗生成模型 — 续写示例\n"]
    for start in test_starts:
        poem = generate_poem(model, start, ix2word, word2ix, device)
        lines.append(f"**输入**：{start}\n")
        lines.append(f"**续写**：{poem}\n")
        lines.append("---\n")

    output_path = output_dir / "generated_samples.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved generated samples to {output_path}")


def main() -> None:
    output_dir = Path("experiment3/outputs")
    data_path = Path("experiment3/data/tang.npz")

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"metrics.json not found at {metrics_path}")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics["history"]

    plot_training_curves(history, output_dir)

    print(f"Best val loss: {metrics['best_validation_loss']:.4f}")
    print(f"Best val perplexity: {metrics['best_validation_perplexity']:.1f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test perplexity: {metrics['test_perplexity']:.1f}")

    best_model_path = output_dir / "best_model.pt"
    if not best_model_path.exists():
        print(f"best_model.pt not found at {best_model_path}")
        return

    dataset_npz = np.load(data_path, allow_pickle=True)
    ix2word = dataset_npz["ix2word"].item()
    word2ix = dataset_npz["word2ix"].item()
    vocab_size = len(word2ix)

    device = resolve_device("auto")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    model = PoetryModel(
        vocab_size=vocab_size,
        embedding_dim=config_dict["embedding_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_layers=config_dict["num_layers"],
        lstm_dropout=config_dict["lstm_dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    generate_demo_samples(model, ix2word, word2ix, device, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
