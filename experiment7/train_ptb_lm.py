from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tarfile
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

PTB_FILENAMES = ("ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
DEFAULT_DATA_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
RAW_PTB_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
    "ptb.test.txt": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
}
RAW_PTB_MD5 = {
    "ptb.train.txt": "f26c4b92c5fdc7b3f8c7cdcb991d8420",
    "ptb.valid.txt": "aa0affc06ff7c36e977d7cd49e3839bf",
    "ptb.test.txt": "8b80168b89c18661a38ef683c0dc3721",
}


@dataclass(frozen=True, slots=True)
class ModelPreset:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_steps: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    max_grad_norm: float
    lr_decay: float
    lr_decay_start_epoch: int
    init_scale: float


PRESETS: dict[str, ModelPreset] = {
    "test": ModelPreset(
        batch_size=20,
        epochs=1,
        learning_rate=1.0,
        weight_decay=0.0,
        num_steps=2,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        max_grad_norm=1.0,
        lr_decay=0.5,
        lr_decay_start_epoch=1,
        init_scale=0.1,
    ),
    "small": ModelPreset(
        batch_size=20,
        epochs=13,
        learning_rate=1.0,
        weight_decay=0.0,
        num_steps=20,
        embedding_dim=200,
        hidden_dim=200,
        num_layers=2,
        dropout=0.0,
        max_grad_norm=5.0,
        lr_decay=0.5,
        lr_decay_start_epoch=4,
        init_scale=0.1,
    ),
    "medium": ModelPreset(
        batch_size=20,
        epochs=39,
        learning_rate=1.0,
        weight_decay=0.0,
        num_steps=35,
        embedding_dim=650,
        hidden_dim=650,
        num_layers=2,
        dropout=0.5,
        max_grad_norm=5.0,
        lr_decay=1.0 / 1.2,
        lr_decay_start_epoch=6,
        init_scale=0.05,
    ),
    "large": ModelPreset(
        batch_size=20,
        epochs=55,
        learning_rate=1.0,
        weight_decay=0.0,
        num_steps=35,
        embedding_dim=1500,
        hidden_dim=1500,
        num_layers=2,
        dropout=0.65,
        max_grad_norm=10.0,
        lr_decay=1.0 / 1.15,
        lr_decay_start_epoch=14,
        init_scale=0.04,
    ),
}


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
    preset: str
    num_steps: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    max_grad_norm: float
    lr_decay: float
    lr_decay_start_epoch: int
    init_scale: float
    tied_weights: bool
    compile: bool
    amp_dtype: str
    download_url: str


@dataclass(slots=True)
class PTBData:
    train_ids: list[int]
    valid_ids: list[int]
    test_ids: list[int]
    word_to_id: dict[str, int]
    id_to_word: list[str]
    data_path: Path


class PTBBatchDataset(Dataset):
    def __init__(self, raw_ids: list[int], batch_size: int, num_steps: int) -> None:
        raw_data = torch.tensor(raw_ids, dtype=torch.long)
        batch_len = raw_data.numel() // batch_size
        if batch_len <= 1:
            raise ValueError("PTB split is too small for the requested batch size.")

        self.data = raw_data[: batch_size * batch_len].view(batch_size, batch_len)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = (batch_len - 1) // num_steps
        if self.epoch_size <= 0:
            raise ValueError("epoch_size == 0; decrease batch_size or num_steps.")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.epoch_size:
            raise IndexError(index)
        start = index * self.num_steps
        end = start + self.num_steps
        inputs = self.data[:, start:end]
        targets = self.data[:, start + 1 : end + 1]
        return inputs, targets


class PTBLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        init_scale: float,
        tied_weights: bool,
    ) -> None:
        super().__init__()
        if tied_weights and embedding_dim != hidden_dim:
            raise ValueError("Tied weights require embedding_dim == hidden_dim.")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.features = nn.ModuleDict(
            {
                "embedding": nn.Embedding(vocab_size, embedding_dim),
                "lstm": nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                ),
            }
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, vocab_size))

        if tied_weights:
            self.classifier[0].weight = self.features["embedding"].weight
        self.init_weights(init_scale)

    def init_weights(self, init_scale: float) -> None:
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -init_scale, init_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        embedded = self.dropout(self.features["embedding"](input_ids))
        output, hidden = self.features["lstm"](embedded, hidden)
        output = self.dropout(output)
        logits = self.classifier(output.reshape(batch_size * seq_len, self.hidden_dim))
        return logits, hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM language model on Penn Treebank.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="large", help="Model/training preset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiment7/data"),
        help="Directory used to cache the PTB corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment7/outputs"),
        help="Directory used to store checkpoints, vocab and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial SGD learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="SGD weight decay.")
    parser.add_argument("--num-steps", type=int, default=None, help="BPTT sequence length.")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Token embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="LSTM hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Global gradient clipping norm.")
    parser.add_argument("--lr-decay", type=float, default=None, help="Multiplicative learning-rate decay.")
    parser.add_argument(
        "--lr-decay-start-epoch",
        type=int,
        default=None,
        help="First epoch after which the learning rate starts decaying.",
    )
    parser.add_argument("--init-scale", type=float, default=None, help="Uniform weight init range.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' selects CUDA when available.",
    )
    parser.add_argument("--download-url", default=DEFAULT_DATA_URL, help="PTB simple-examples.tgz URL.")
    parser.add_argument("--tied-weights", action="store_true", help="Tie embedding and softmax weights.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
        help="Autocast dtype. 'auto' prefers bfloat16 on supported CUDA GPUs.",
    )
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile on CUDA.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    preset = PRESETS[args.preset]

    def choose(name: str) -> Any:
        value = getattr(args, name)
        return value if value is not None else getattr(preset, name)

    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=choose("batch_size"),
        epochs=choose("epochs"),
        learning_rate=choose("learning_rate"),
        weight_decay=choose("weight_decay"),
        val_size=0,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=False,
        amp=not args.disable_amp,
        device=args.device,
        preset=args.preset,
        num_steps=choose("num_steps"),
        embedding_dim=choose("embedding_dim"),
        hidden_dim=choose("hidden_dim"),
        num_layers=choose("num_layers"),
        dropout=choose("dropout"),
        max_grad_norm=choose("max_grad_norm"),
        lr_decay=choose("lr_decay"),
        lr_decay_start_epoch=choose("lr_decay_start_epoch"),
        init_scale=choose("init_scale"),
        tied_weights=args.tied_weights,
        compile=not args.disable_compile,
        amp_dtype=args.amp_dtype,
        download_url=args.download_url,
    )


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


def resolve_amp_dtype(requested: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if requested == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP was requested but is not supported by this CUDA device.")
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def find_ptb_data_dir(data_dir: Path) -> Path | None:
    for candidate in (data_dir, data_dir / "simple-examples" / "data"):
        if all((candidate / filename).exists() for filename in PTB_FILENAMES):
            return candidate
    return None


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(tar_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise RuntimeError(f"Refusing to extract path outside data_dir: {member.name}")
        tar.extractall(destination)


def download_file(url: str, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    bytes_read = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        expected_size = response.headers.get("Content-Length")
        with tmp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                bytes_read += len(chunk)

    if expected_size is not None and bytes_read != int(expected_size):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download incomplete: got {bytes_read} bytes out of {int(expected_size)} bytes.")
    tmp_path.replace(output_path)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_ptb_files(data_dir: Path) -> Path:
    for filename in PTB_FILENAMES:
        output_path = data_dir / filename
        print(f"Downloading PTB split {filename} from GitHub mirror")
        download_file(RAW_PTB_URLS[filename], output_path)
        actual_md5 = md5sum(output_path)
        if actual_md5 != RAW_PTB_MD5[filename]:
            raise RuntimeError(f"MD5 mismatch for {filename}: expected {RAW_PTB_MD5[filename]}, got {actual_md5}.")
    return data_dir


def ensure_ptb_files(data_dir: Path, download_url: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = find_ptb_data_dir(data_dir)
    if existing is not None:
        return existing

    archive_path = data_dir / "simple-examples.tgz"
    print(f"Downloading PTB corpus from {download_url}")
    try:
        download_file(download_url, archive_path)
        print(f"Extracting {archive_path}")
        safe_extract_tar(archive_path, data_dir)
    except (OSError, RuntimeError) as exc:
        print(f"Archive download failed ({exc}); falling back to raw PTB files.")
        download_raw_ptb_files(data_dir)

    extracted = find_ptb_data_dir(data_dir)
    if extracted is None:
        raise FileNotFoundError("PTB files were not found after extracting simple-examples.tgz.")
    return extracted


def read_words(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\n", " <eos> ").split()


def build_vocabulary(train_path: Path) -> tuple[dict[str, int], list[str]]:
    counter = Counter(read_words(train_path))
    count_pairs = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    id_to_word = [word for word, _ in count_pairs]
    word_to_id = {word: index for index, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


def file_to_word_ids(path: Path, word_to_id: dict[str, int]) -> list[int]:
    words = read_words(path)
    unk_id = word_to_id.get("<unk>")
    if unk_id is None:
        missing = sorted({word for word in words if word not in word_to_id})
        if missing:
            raise KeyError(f"Words missing from vocabulary and no <unk> token exists: {missing[:5]}")
        return [word_to_id[word] for word in words]
    return [word_to_id.get(word, unk_id) for word in words]


def load_ptb_data(config: ExperimentConfig) -> PTBData:
    data_path = ensure_ptb_files(config.data_dir, config.download_url)
    train_path = data_path / "ptb.train.txt"
    valid_path = data_path / "ptb.valid.txt"
    test_path = data_path / "ptb.test.txt"

    word_to_id, id_to_word = build_vocabulary(train_path)
    return PTBData(
        train_ids=file_to_word_ids(train_path, word_to_id),
        valid_ids=file_to_word_ids(valid_path, word_to_id),
        test_ids=file_to_word_ids(test_path, word_to_id),
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        data_path=data_path,
    )


def build_dataloaders(config: ExperimentConfig, data: PTBData, device: torch.device) -> dict[str, DataLoader]:
    common_loader_args = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    return {
        "train": DataLoader(
            PTBBatchDataset(data.train_ids, config.batch_size, config.num_steps),
            **common_loader_args,
        ),
        "val": DataLoader(
            PTBBatchDataset(data.valid_ids, config.batch_size, config.num_steps),
            **common_loader_args,
        ),
        "test": DataLoader(
            PTBBatchDataset(data.test_ids, config.batch_size, config.num_steps),
            **common_loader_args,
        ),
    }


def detach_hidden(hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return hidden[0].detach(), hidden[1].detach()


def safe_exp(value: float) -> float:
    return float("inf") if value > 88.0 else math.exp(value)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def maybe_compile_model(model: nn.Module, device: torch.device, enabled: bool) -> nn.Module:
    if not enabled or device.type != "cuda":
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("torch.compile() enabled (reduce-overhead mode)")
        return compiled
    except Exception as exc:
        print(f"torch.compile() skipped: {exc}")
        return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if hidden is not None:
            hidden = detach_hidden(hidden)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, hidden = model(inputs, hidden)
            flat_targets = targets.reshape(-1)
            loss = criterion(logits, flat_targets)

        # TensorFlow's PTB sequence_loss averages across batch, then sums
        # across time steps. Match that gradient scale while reporting mean CE.
        training_loss = loss * targets.size(1)
        scaler.scale(training_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_correct += (logits.detach().argmax(dim=1) == flat_targets).sum().item()
        total_tokens += token_count

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "perplexity": safe_exp(avg_loss),
        "next_word_accuracy": total_correct / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if hidden is not None:
            hidden = detach_hidden(hidden)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, hidden = model(inputs, hidden)
            flat_targets = targets.reshape(-1)
            loss = criterion(logits, flat_targets)

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_correct += (logits.argmax(dim=1) == flat_targets).sum().item()
        total_tokens += token_count

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "perplexity": safe_exp(avg_loss),
        "next_word_accuracy": total_correct / max(total_tokens, 1),
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = build_config(args)

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config.amp_dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ptb_data = load_ptb_data(config)
    dataloaders = build_dataloaders(config, ptb_data, device)
    vocab_size = len(ptb_data.word_to_id)

    write_json(
        config.output_dir / "vocab.json",
        {"word_to_id": ptb_data.word_to_id, "id_to_word": ptb_data.id_to_word},
    )

    raw_model = PTBLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        init_scale=config.init_scale,
        tied_weights=config.tied_weights,
    ).to(device)
    model = maybe_compile_model(raw_model, device, config.compile)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        raw_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(device=device.type, enabled=use_amp and amp_dtype == torch.float16)

    best_val_ppl = float("inf")
    best_model_path = config.output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []
    parameter_count = count_parameters(raw_model)

    print(f"Using device: {device}")
    print(f"PTB data path: {ptb_data.data_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model parameters: {parameter_count:,}")
    print(
        "Train config:",
        json.dumps(
            {
                "preset": config.preset,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "num_steps": config.num_steps,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "max_grad_norm": config.max_grad_norm,
                "lr_decay": config.lr_decay,
                "lr_decay_start_epoch": config.lr_decay_start_epoch,
                "amp": use_amp,
                "amp_dtype": str(amp_dtype).replace("torch.", ""),
                "compile": config.compile and device.type == "cuda",
            },
            ensure_ascii=False,
        ),
    )

    for epoch in range(1, config.epochs + 1):
        learning_rate = config.learning_rate * (config.lr_decay ** max(epoch - config.lr_decay_start_epoch, 0))
        set_optimizer_lr(optimizer, learning_rate)

        train_metrics = train_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_grad_norm=config.max_grad_norm,
        )
        val_metrics = evaluate(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_perplexity": train_metrics["perplexity"],
            "train_next_word_accuracy": train_metrics["next_word_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
            "val_next_word_accuracy": val_metrics["next_word_accuracy"],
        }
        history.append(epoch_metrics)

        if val_metrics["perplexity"] < best_val_ppl:
            best_val_ppl = val_metrics["perplexity"]
            save_checkpoint(
                path=best_model_path,
                epoch=epoch,
                model=raw_model,
                optimizer=optimizer,
                metrics=val_metrics,
                config=config,
            )

        print(
            f"Epoch {epoch:02d}/{config.epochs} | lr={learning_rate:.5f} | "
            f"train_loss={train_metrics['loss']:.4f} train_ppl={train_metrics['perplexity']:.2f} "
            f"train_acc={train_metrics['next_word_accuracy']:.2%} | "
            f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['perplexity']:.2f} "
            f"val_acc={val_metrics['next_word_accuracy']:.2%}"
        )

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    metrics_payload = {
        "best_validation_perplexity": best_val_ppl,
        "best_validation_loss": float(math.log(best_val_ppl)),
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "test_next_word_accuracy": test_metrics["next_word_accuracy"],
        "target_test_perplexity": 80.0,
        "meets_requirement": test_metrics["perplexity"] < 80.0,
        "vocab_size": vocab_size,
        "parameter_count": parameter_count,
        "history": history,
    }
    write_json(config.output_dir / "metrics.json", metrics_payload)

    print(
        f"\nBest validation perplexity: {best_val_ppl:.2f}\n"
        f"Test loss: {test_metrics['loss']:.4f}\n"
        f"Test perplexity: {test_metrics['perplexity']:.2f}\n"
        f"Test next-word accuracy: {test_metrics['next_word_accuracy']:.2%}\n"
        f"Meets PPL < 80 requirement: {test_metrics['perplexity'] < 80.0}"
    )


if __name__ == "__main__":
    main()
