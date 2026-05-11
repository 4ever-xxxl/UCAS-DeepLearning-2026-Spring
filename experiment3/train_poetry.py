from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset


@dataclass(slots=True)
class ExperimentConfig:
    data_path: Path
    output_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    lstm_dropout: float
    val_split: float
    test_split: float
    max_seq_len: int
    num_workers: int
    seed: int
    amp: bool
    device: str
    grad_clip: float


class PoetryModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        lstm_dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.features = nn.ModuleDict(
            {
                "embedding": nn.Embedding(vocab_size, embedding_dim),
                "lstm": nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=lstm_dropout if num_layers > 1 else 0.0,
                ),
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(lstm_dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        input: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len = input.size()
        embeds = self.features["embedding"](input)
        if hidden is None:
            h_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=input.device,
            )
            c_0 = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=input.device,
            )
            hidden = (h_0, c_0)
        output, hidden = self.features["lstm"](embeds, hidden)
        output = self.classifier(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM poetry generation model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("experiment3/data/tang.npz"),
        help="Path to the tang.npz dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment3/outputs"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size (128 for 12GB VRAM).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--embedding-dim", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=768, help="LSTM hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of LSTM layers.")
    parser.add_argument("--lstm-dropout", type=float, default=0.3, help="Dropout between LSTM layers.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set fraction.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test set fraction.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def load_data(
    data_path: Path,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    dataset_npz = np.load(data_path, allow_pickle=True)
    data = torch.from_numpy(dataset_npz["data"].astype(np.int64))
    ix2word = dataset_npz["ix2word"].item()
    word2ix = dataset_npz["word2ix"].item()

    total = len(data)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        TensorDataset(data),
        [train_size, val_size, test_size],
        generator=generator,
    )

    return train_dataset, val_dataset, test_dataset, ix2word, word2ix


def build_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(train_dataset, shuffle=True, **common_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **common_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **common_kwargs),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for (batch_data,) in loader:
        seq = batch_data.to(device, non_blocking=True)
        inputs = seq[:, :-1]
        targets = seq[:, 1:].reshape(-1)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        non_pad = (targets != criterion.ignore_index).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return {
        "loss": total_loss / max(total_tokens, 1),
        "perplexity": np.exp(total_loss / max(total_tokens, 1)),
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
    total_tokens = 0

    for (batch_data,) in loader:
        seq = batch_data.to(device, non_blocking=True)
        inputs = seq[:, :-1]
        targets = seq[:, 1:].reshape(-1)

        logits, _ = model(inputs)
        loss = criterion(logits, targets)

        non_pad = (targets != criterion.ignore_index).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "perplexity": np.exp(avg_loss)}


def generate_poem(
    model: nn.Module,
    start_words: str,
    ix2word: dict,
    word2ix: dict,
    device: torch.device,
    max_gen_len: int = 125,
) -> str:
    model.eval()
    results: list[str] = list(start_words)
    start_len = len(start_words)
    input_tensor = torch.tensor([[word2ix["<START>"]]], device=device, dtype=torch.long)
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    with torch.no_grad():
        for i in range(max_gen_len):
            output, hidden = model(input_tensor, hidden)
            if i < start_len:
                w = results[i]
                if w not in word2ix:
                    w = random.choice(list(word2ix.keys()))
                input_tensor = torch.tensor([[word2ix[w]]], device=device, dtype=torch.long)
            else:
                top_index = output[0].argmax().item()
                w = ix2word.get(top_index, "<EOP>")
                results.append(w)
                input_tensor = torch.tensor([[top_index]], device=device, dtype=torch.long)
                if w == "<EOP>":
                    del results[-1]
                    break

    return "".join(results)


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
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lstm_dropout=args.lstm_dropout,
        val_split=args.val_split,
        test_split=args.test_split,
        max_seq_len=125,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.disable_amp,
        device=args.device,
        grad_clip=args.grad_clip,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_dataset, val_dataset, test_dataset, ix2word, word2ix = load_data(
        config.data_path,
        config.val_split,
        config.test_split,
        config.seed,
    )
    vocab_size = len(word2ix)
    pad_idx = word2ix["</s>"]
    print(f"Vocabulary size: {vocab_size}, padding index: {pad_idx}")

    dataloaders = build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config.batch_size,
        config.num_workers,
        device,
    )

    model = PoetryModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        lstm_dropout=config.lstm_dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile() enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"torch.compile() skipped: {e}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_val_loss = float("inf")
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
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "lstm_dropout": config.lstm_dropout,
                "grad_clip": config.grad_clip,
                "amp": use_amp,
            },
            ensure_ascii=False,
        ),
    )

    sample_start = "湖光秋月两相和"

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=config.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )
        scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_perplexity": train_metrics["perplexity"],
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
        }
        history.append(epoch_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=best_model_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=val_metrics,
                config=config,
            )

        sample = generate_poem(
            model,
            sample_start,
            ix2word,
            word2ix,
            device,
        )
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_ppl={train_metrics['perplexity']:.1f} | "
            f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['perplexity']:.1f}"
        )
        print(f"  Sample [{sample_start}]: {sample}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )

    metrics_payload = {
        "best_validation_loss": best_val_loss,
        "best_validation_perplexity": float(np.exp(best_val_loss)),
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "history": history,
    }
    write_metrics(config.output_dir / "metrics.json", metrics_payload)

    print(
        f"\nBest validation loss: {best_val_loss:.4f} "
        f"(perplexity: {np.exp(best_val_loss):.1f})\n"
        f"Test loss: {test_metrics['loss']:.4f} "
        f"(perplexity: {test_metrics['perplexity']:.1f})"
    )

    print("\n--- Generated samples ---")
    test_starts = ["湖光秋月两相和", "朝辞白帝彩云间", "床前明月光", "春眠不觉晓", "白日依山尽"]
    for start in test_starts:
        poem = generate_poem(model, start, ix2word, word2ix, device)
        print(f"  [{start}] {poem}")


if __name__ == "__main__":
    main()
