from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@dataclass(slots=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    val_size: int
    num_workers: int
    seed: int
    augment: bool
    amp: bool
    device: str


class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on the MNIST dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiment1/data"),
        help="Directory used to cache the MNIST dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment1/outputs"),
        help="Directory used to store checkpoints and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation set size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--disable-augment",
        action="store_true",
        help="Disable random affine augmentation on the training set.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision training on CUDA.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[Any] = []
    if augment:
        train_steps.append(
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.95, 1.05))
        )
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    eval_steps = [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def build_dataloaders(config: ExperimentConfig, device: torch.device) -> dict[str, DataLoader]:
    train_transform, eval_transform = build_transforms(config.augment)
    train_source = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    eval_source = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_size = len(train_source) - config.val_size
    if train_size <= 0:
        raise ValueError(f"Validation size must be smaller than {len(train_source)}.")

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(eval_source, val_indices)

    common_loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }

    return {
        "train": DataLoader(train_dataset, shuffle=True, **common_loader_args),
        "val": DataLoader(val_dataset, shuffle=False, **common_loader_args),
        "test": DataLoader(test_dataset, shuffle=False, **common_loader_args),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += inputs.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += inputs.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    config: ExperimentConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        },
        path,
    )


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=not args.disable_augment,
        amp=not args.disable_amp,
        device=args.device,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataloaders = build_dataloaders(config, device)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_val_accuracy = 0.0
    best_model_path = config.output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []

    print(f"Using device: {device}")
    print(
        "Train config:",
        json.dumps(
            {
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "val_size": config.val_size,
                "augment": config.augment,
                "amp": use_amp,
            },
            ensure_ascii=False,
        ),
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = evaluate(model=model, loader=dataloaders["val"], criterion=criterion, device=device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                path=best_model_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=val_metrics,
                config=config,
            )

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4%} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4%}"
        )

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model=model, loader=dataloaders["test"], criterion=criterion, device=device)

    metrics_payload = {
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "history": history,
    }
    write_metrics(config.output_dir / "metrics.json", metrics_payload)

    print(
        f"Best validation accuracy: {best_val_accuracy:.4%}\n"
        f"Test loss: {test_metrics['loss']:.4f}\n"
        f"Test accuracy: {test_metrics['accuracy']:.4%}"
    )


if __name__ == "__main__":
    main()
