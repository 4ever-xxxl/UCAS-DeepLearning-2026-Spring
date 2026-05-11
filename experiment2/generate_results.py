from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_vit import CIFAR_MEAN, CIFAR_STD, CIFAR10ViT

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "best_model.pt"
DATA_DIR = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 512
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_model() -> CIFAR10ViT:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

    model = CIFAR10ViT().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_test_loader() -> DataLoader:
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=eval_transform)
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )


def get_normalized_test_dataset() -> datasets.CIFAR10:
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=eval_transform)


def get_raw_test_dataset() -> datasets.CIFAR10:
    return datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)

    history = data["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] * 100 for h in history]
    val_acc = [h["val_accuracy"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "o-", color="#1565C0", linewidth=2, markersize=4, label="Train Loss")
    ax1.plot(epochs, val_loss, "s-", color="#C62828", linewidth=2, markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_acc, "o-", color="#1565C0", linewidth=2, markersize=4, label="Train Acc")
    ax2.plot(epochs, val_acc, "s-", color="#C62828", linewidth=2, markersize=4, label="Val Acc")
    ax2.axhline(80, color="#2E7D32", linestyle="--", linewidth=1.5, label="80% Target")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"CIFAR-10 ViT | Best Val Acc: {data['best_validation_accuracy']:.2%} | Test Acc: {data['test_accuracy']:.2%}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        logits = model(inputs)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)

    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def plot_confusion_matrix(model: nn.Module, loader: DataLoader, output_path: Path) -> None:
    preds, targets = collect_predictions(model, loader)
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for target, pred in zip(targets, preds):
        cm[target, pred] += 1

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix - CIFAR-10 Test Set")

    accuracy = np.trace(cm) / np.sum(cm)
    fig.suptitle(f"Overall Accuracy: {accuracy:.2%}", fontsize=12, y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_predictions(
    model: nn.Module,
    normalized_dataset: datasets.CIFAR10,
    raw_dataset: datasets.CIFAR10,
    output_path: Path,
    n_samples: int = 30,
) -> None:
    rng = np.random.RandomState(42)
    indices = rng.choice(len(raw_dataset), size=n_samples, replace=False)
    rows = 5
    cols = int(np.ceil(n_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()

    for i, sample_idx in enumerate(indices):
        image, true_label = raw_dataset[sample_idx]
        normalized_image, _ = normalized_dataset[sample_idx]

        with torch.no_grad():
            logits = model(normalized_image.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            pred_label = logits.argmax(dim=1).item()
            confidence = probs[0, pred_label].item()

        ax = axes[i]
        ax.imshow(image.permute(1, 2, 0).numpy())
        is_correct = pred_label == true_label
        color = "#2E7D32" if is_correct else "#C62828"
        ax.set_title(
            f"T: {CLASS_NAMES[true_label]}\nP: {CLASS_NAMES[pred_label]} ({confidence:.0%})",
            fontsize=8,
            color=color,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    fig.suptitle("CIFAR-10 Prediction Samples - Green = Correct, Red = Wrong", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_feature_map(
    module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, store: dict[str, torch.Tensor]
) -> None:
    store["feature_map"] = output.detach().cpu()


def plot_feature_maps(
    model: CIFAR10ViT,
    normalized_dataset: datasets.CIFAR10,
    raw_dataset: datasets.CIFAR10,
    output_path: Path,
    n_samples: int = 4,
    n_channels: int = 12,
) -> None:
    rng = np.random.RandomState(2026)
    indices = rng.choice(len(raw_dataset), size=n_samples, replace=False)

    store: dict[str, torch.Tensor] = {}
    first_conv = model.patch_embed.proj[0]
    hook = first_conv.register_forward_hook(lambda m, inp, out: save_feature_map(m, inp, out, store))

    fig, axes = plt.subplots(n_samples, n_channels + 1, figsize=((n_channels + 1) * 1.35, n_samples * 1.8))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    try:
        for row_idx, sample_idx in enumerate(indices):
            raw_image, label = raw_dataset[sample_idx]
            normalized_image, _ = normalized_dataset[sample_idx]
            with torch.no_grad():
                model(normalized_image.unsqueeze(0).to(DEVICE))

            ax = axes[row_idx, 0]
            ax.imshow(raw_image.permute(1, 2, 0).numpy())
            ax.set_title(CLASS_NAMES[label], fontsize=8)
            ax.axis("off")

            feature_map = store["feature_map"][0]
            for channel in range(min(n_channels, feature_map.shape[0])):
                ax = axes[row_idx, channel + 1]
                ax.imshow(feature_map[channel].numpy(), cmap="viridis")
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"F{channel + 1}", fontsize=7)
    finally:
        hook.remove()

    fig.suptitle("First Convolution Feature Maps in ConvPatchEmbed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")

    model = load_model()
    print("Model loaded.")

    plot_training_curves(METRICS_PATH, OUTPUT_DIR / "training_curves.png")

    print("Computing confusion matrix...")
    test_loader = get_test_loader()
    plot_confusion_matrix(model, test_loader, OUTPUT_DIR / "confusion_matrix.png")

    print("Generating prediction samples...")
    normalized_dataset = get_normalized_test_dataset()
    raw_dataset = get_raw_test_dataset()
    plot_predictions(model, normalized_dataset, raw_dataset, OUTPUT_DIR / "predictions.png")

    print("Visualizing first convolution feature maps...")
    plot_feature_maps(model, normalized_dataset, raw_dataset, OUTPUT_DIR / "feature_maps.png")

    print("All done.")


if __name__ == "__main__":
    main()
