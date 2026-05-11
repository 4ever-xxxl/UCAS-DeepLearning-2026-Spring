from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_mnist import MNIST_MEAN, MNIST_STD, MNISTCNN

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "best_model.pt"
DATA_DIR = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [str(i) for i in range(10)]


def get_device() -> torch.device:
    return DEVICE


def load_model() -> MNISTCNN:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = MNISTCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_test_loader() -> DataLoader:
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=eval_transform)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def get_test_dataset() -> datasets.MNIST:
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    return datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=eval_transform)


def get_unnormalized_test_dataset() -> datasets.MNIST:
    return datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    with open(metrics_path) as f:
        data = json.load(f)

    history = data["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] * 100 for h in history]
    val_acc = [h["val_accuracy"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "o-", color="#2196F3", linewidth=2, markersize=6, label="Train Loss")
    ax1.plot(epochs, val_loss, "s-", color="#FF5722", linewidth=2, markersize=6, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    ax2.plot(epochs, train_acc, "o-", color="#2196F3", linewidth=2, markersize=6, label="Train Acc")
    ax2.plot(epochs, val_acc, "s-", color="#FF5722", linewidth=2, markersize=6, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim(82, 100.5)

    # Annotate best validation accuracy point
    best_epoch_idx = val_acc.index(max(val_acc))
    ax2.annotate(
        f"Best: {val_acc[best_epoch_idx]:.2f}%",
        xy=(epochs[best_epoch_idx], val_acc[best_epoch_idx]),
        xytext=(epochs[best_epoch_idx] + 0.4, val_acc[best_epoch_idx] - 0.6),
        arrowprops={"arrowstyle": "->", "color": "#FF5722"},
        fontsize=9,
        color="#FF5722",
        fontweight="bold",
    )

    fig.suptitle(
        f"MNIST Training Curves | Best Val Acc: {data['best_validation_accuracy']:.4%} | Test Acc: {data['test_accuracy']:.4%}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_matrix(model: MNISTCNN, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # Per-class metrics
    per_class_acc = {}
    for i in range(10):
        tp = cm[i, i]
        total = cm[i].sum()
        per_class_acc[i] = tp / total if total > 0 else 0.0

    # Build annotation with count + per-class acc on diagonal
    annot = np.empty_like(cm, dtype=object)
    for i in range(10):
        for j in range(10):
            cell = str(cm[i, j])
            if i == j:
                cell += f"\n({per_class_acc[i]:.1%})"
            annot[i, j] = cell

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
        annot_kws={"fontsize": 10},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — MNIST Test Set\n(Diagonal shows per-class accuracy)", fontsize=13)

    accuracy = np.trace(cm) / np.sum(cm)
    fig.suptitle(f"Overall Accuracy: {accuracy:.4%}", fontsize=13, y=0.98, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print per-class metrics
    print(f"Overall accuracy: {accuracy:.4%}")
    print("Per-class accuracy:")
    for i in range(10):
        print(f"  Digit {i}: {per_class_acc[i]:.4%} ({cm[i, i]}/{cm[i].sum()})")
    print(f"Saved: {output_path}")


def plot_predictions(
    model: MNISTCNN,
    dataset_normalized: datasets.MNIST,
    dataset_raw: datasets.MNIST,
    device: torch.device,
    output_path: Path,
    n_samples: int = 25,
) -> None:
    idx = np.random.RandomState(42).choice(len(dataset_raw), size=n_samples, replace=False)

    rows = int(np.ceil(np.sqrt(n_samples)))
    cols = int(np.ceil(n_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i, sample_idx in enumerate(idx):
        image, true_label = dataset_raw[sample_idx]
        norm_image, _ = dataset_normalized[sample_idx]

        with torch.no_grad():
            logits = model(norm_image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)
            pred_label = logits.argmax(dim=1).item()
            confidence = probs[0, pred_label].item()

        ax = axes[i]
        ax.imshow(image.squeeze(), cmap="gray")
        is_correct = pred_label == true_label
        color = "#2E7D32" if is_correct else "#C62828"
        ax.set_title(f"T:{true_label} P:{pred_label}\n{confidence:.2%}", fontsize=9, color=color)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("MNIST Predictions — Green = Correct, Red = Wrong", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def feature_map_hook(module, input, output, store):
    store["feature_map"] = output.detach().cpu()


def plot_feature_maps(
    model: MNISTCNN,
    dataset: datasets.MNIST,
    device: torch.device,
    output_path: Path,
    n_samples: int = 4,
    n_channels: int = 16,
) -> None:
    idx = np.random.RandomState(2026).choice(len(dataset), size=n_samples, replace=False)

    store: dict = {}
    hook = model.features[0].register_forward_hook(lambda m, inp, out: feature_map_hook(m, inp, out, store))

    fig, axes = plt.subplots(n_samples, n_channels + 1, figsize=((n_channels + 1) * 1.5, n_samples * 2))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row_idx, sample_idx in enumerate(idx):
        image, label = dataset[sample_idx]
        with torch.no_grad():
            model(image.unsqueeze(0).to(device))

        ax = axes[row_idx, 0]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Input ({label})", fontsize=8)
        ax.axis("off")

        feature_map = store["feature_map"][0]
        for ch in range(min(n_channels, feature_map.shape[0])):
            ax = axes[row_idx, ch + 1]
            ax.imshow(feature_map[ch].numpy(), cmap="viridis")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"F{ch + 1}", fontsize=7)

    hook.remove()

    fig.suptitle(
        "First Conv Layer Feature Maps (Conv1 → 32 channels, showing first 16)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    model = load_model()
    print("Model loaded.")

    plot_training_curves(METRICS_PATH, OUTPUT_DIR / "training_curves.png")

    print("Computing confusion matrix...")
    test_loader = get_test_loader()
    plot_confusion_matrix(model, test_loader, device, OUTPUT_DIR / "confusion_matrix.png")

    print("Generating prediction samples...")
    raw_dataset = get_unnormalized_test_dataset()
    norm_dataset = get_test_dataset()
    plot_predictions(model, norm_dataset, raw_dataset, device, OUTPUT_DIR / "predictions.png")

    print("Visualizing feature maps...")
    plot_feature_maps(model, norm_dataset, device, OUTPUT_DIR / "feature_maps.png")

    print("All done!")


if __name__ == "__main__":
    main()
