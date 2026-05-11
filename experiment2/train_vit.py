from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic Depth: randomly drop samples in the residual branch."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# ViT components (from experiment guide)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention exactly as specified in the experiment guide."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation — from the experiment guide."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Image-to-patch embedding: Conv2d projects each patch to embed_dim."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)


class ConvPatchEmbed(nn.Module):
    """CIFAR-sized convolutional tokenizer for compact ViT training from scratch."""

    def __init__(self, img_size: int = 32, in_chans: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        if img_size % 4 != 0:
            raise ValueError("ConvPatchEmbed expects img_size divisible by 4.")
        self.num_patches = (img_size // 4) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Transformer encoder block: LayerNorm → Attention → residual → LayerNorm → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        conv_stem: bool = False,
    ) -> None:
        super().__init__()

        if conv_stem:
            self.patch_embed = ConvPatchEmbed(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic Depth: linearly increasing drop_path per block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]  # Take class token
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# CIFAR-10 specific ViT (smaller capacity for 32×32 images)
# ---------------------------------------------------------------------------


class CIFAR10ViT(VisionTransformer):
    """Compact ViT tuned for CIFAR-10 and fast training on 32x32 inputs."""

    def __init__(self) -> None:
        super().__init__(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=10,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=3.0,
            qkv_bias=True,
            drop_path_rate=0.1,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            conv_stem=True,
        )


# ---------------------------------------------------------------------------
# Experiment infrastructure (mirrors experiment1/train_mnist.py structure)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    num_workers: int
    seed: int
    augment: bool
    amp: bool
    device: str
    label_smoothing: float
    mixup_alpha: float
    random_erasing: float
    target_accuracy: float
    min_epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=Path("experiment2/data"), help="CIFAR-10 cache directory.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiment2/outputs"), help="Checkpoint and metrics directory."
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="AdamW weight decay.")
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Validation split fraction from CIFAR-10 train."
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Cross-entropy label smoothing.")
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.2, help="MixUp beta distribution alpha; 0 disables MixUp."
    )
    parser.add_argument("--random-erasing", type=float, default=0.15, help="RandomErasing probability.")
    parser.add_argument(
        "--target-accuracy", type=float, default=0.82, help="Stop once best validation accuracy reaches this value."
    )
    parser.add_argument("--min-epochs", type=int, default=20, help="Minimum epochs before target-accuracy early stop.")
    parser.add_argument("--disable-augment", action="store_true", help="Disable augmentation.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision.")
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


def build_transforms(augment: bool, random_erasing: float) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[Any] = []
    if augment:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                autoaugment.RandAugment(num_ops=2, magnitude=9),
            ]
        )
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    if augment and random_erasing > 0.0:
        train_steps.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.12), value=0))
    eval_steps = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def build_dataloaders(config: ExperimentConfig, device: torch.device) -> dict[str, DataLoader]:
    train_transform, eval_transform = build_transforms(config.augment, config.random_erasing)

    full_train_dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=train_transform)
    full_val_dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=eval_transform)

    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")

    val_size = int(config.val_fraction * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    split_generator = torch.Generator().manual_seed(config.seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_train_dataset)), [train_size, val_size], generator=split_generator
    )
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices.indices)

    # Use all available CPUs for data loading; prefetch more batches
    worker_count = config.num_workers if config.num_workers > 0 else min(16, os.cpu_count() or 1)
    common_loader_args: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
    }
    if worker_count > 0:
        common_loader_args["persistent_workers"] = True
        common_loader_args["prefetch_factor"] = 4

    return {
        "train": DataLoader(train_dataset, shuffle=True, **common_loader_args),
        "val": DataLoader(val_dataset, shuffle=False, **common_loader_args),
        "test": DataLoader(test_dataset, shuffle=False, **common_loader_args),
    }


class CrossEntropyWithSoftTargets(nn.Module):
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def smooth_one_hot(targets: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target_probs = torch.full(
        (targets.size(0), num_classes),
        off_value,
        dtype=torch.float32,
        device=targets.device,
    )
    return target_probs.scatter_(1, targets.unsqueeze(1), on_value)


def apply_mixup(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        return inputs, targets

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = inputs.mul(lam).add_(inputs[index], alpha=1.0 - lam)

    targets_a = smooth_one_hot(targets, num_classes, label_smoothing)
    targets_b = targets_a[index]
    mixed_targets = targets_a.mul(lam).add_(targets_b, alpha=1.0 - lam)
    return mixed_inputs, mixed_targets


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
    mixup_alpha: float,
    label_smoothing: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        hard_targets = targets

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            inputs, targets = apply_mixup(inputs, targets, num_classes, mixup_alpha, label_smoothing)
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == hard_targets).sum().item()
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
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
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
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=not args.disable_augment,
        amp=not args.disable_amp,
        device=args.device,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        random_erasing=args.random_erasing,
        target_accuracy=args.target_accuracy,
        min_epochs=args.min_epochs,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"

    # Blackwell / Ampere optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataloaders = build_dataloaders(config, device)

    model = CIFAR10ViT().to(device)
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    criterion = CrossEntropyWithSoftTargets(label_smoothing=config.label_smoothing)

    # Separate weight decay: no decay on bias and norm parameters
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name or "cls_token" in name or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay_params}, {"params": no_decay_params, "weight_decay": 0.0}],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Cosine annealing with linear warmup
    warmup_epochs = min(5, max(1, config.epochs // 10))
    cosine_epochs = max(1, config.epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_val_accuracy = 0.0
    best_model_path = config.output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []

    print(f"Using device: {device}")
    print(
        f"Train config: batch_size={config.batch_size} epochs={config.epochs} "
        f"lr={config.learning_rate} wd={config.weight_decay} augment={config.augment} "
        f"mixup_alpha={config.mixup_alpha} label_smoothing={config.label_smoothing} "
        f"random_erasing={config.random_erasing} amp={use_amp}"
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
            num_classes=10,
            mixup_alpha=config.mixup_alpha if config.augment else 0.0,
            label_smoothing=config.label_smoothing,
        )
        val_metrics = evaluate(model=model, loader=dataloaders["val"], criterion=criterion, device=device)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
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
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4%} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if epoch >= config.min_epochs and best_val_accuracy >= config.target_accuracy:
            print(f"Reached target validation accuracy {config.target_accuracy:.2%} at epoch {epoch}; stopping early.")
            break

    # Load best checkpoint and evaluate on test set
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
        f"\nBest validation accuracy: {best_val_accuracy:.4%}\n"
        f"Test loss: {test_metrics['loss']:.4f}\n"
        f"Test accuracy: {test_metrics['accuracy']:.4%}"
    )


if __name__ == "__main__":
    main()
