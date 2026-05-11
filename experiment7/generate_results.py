from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from train_ptb_lm import (
    PTBLanguageModel,
    build_vocabulary,
    ensure_ptb_files,
    file_to_word_ids,
)


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
MODEL_PATH = OUTPUT_DIR / "best_model.pt"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Run train_ptb_lm.py first; missing {METRICS_PATH}.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_vocab() -> tuple[dict[str, int], list[str]]:
    if VOCAB_PATH.exists():
        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        return vocab["word_to_id"], vocab["id_to_word"]

    data_path = ensure_ptb_files(DATA_DIR, "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz")
    return build_vocabulary(data_path / "ptb.train.txt")


def load_model(vocab_size: int) -> PTBLanguageModel:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Run train_ptb_lm.py first; missing {MODEL_PATH}.")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = PTBLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        init_scale=float(config["init_scale"]),
        tied_weights=bool(config["tied_weights"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def plot_training_curves(metrics: dict, output_path: Path) -> None:
    history = metrics["history"]
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_ppl = [item["train_perplexity"] for item in history]
    val_ppl = [item["val_perplexity"] for item in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, "o-", color="#2563eb", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_loss, "s-", color="#dc2626", linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_ppl, "o-", color="#2563eb", linewidth=2, label="Train PPL")
    ax2.plot(epochs, val_ppl, "s-", color="#dc2626", linewidth=2, label="Val PPL")
    ax2.axhline(80.0, color="#111827", linestyle="--", linewidth=1.5, label="Target PPL 80")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Training & Validation Perplexity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"PTB LSTM Language Model | Best Val PPL: {metrics['best_validation_perplexity']:.2f} | "
        f"Test PPL: {metrics['test_perplexity']:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


@torch.no_grad()
def generate_continuation(
    model: PTBLanguageModel,
    prompt: str,
    word_to_id: dict[str, int],
    id_to_word: list[str],
    max_new_tokens: int = 40,
    temperature: float = 0.9,
) -> str:
    words = prompt.lower().split()
    if not words:
        words = ["<eos>"]
    unk_id = word_to_id.get("<unk>", 0)
    ids = [word_to_id.get(word, unk_id) for word in words]
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    logits: torch.Tensor | None = None

    for token_id in ids:
        input_tensor = torch.tensor([[token_id]], dtype=torch.long, device=DEVICE)
        logits, hidden = model(input_tensor, hidden)

    generated = words.copy()
    assert logits is not None
    for _ in range(max_new_tokens):
        next_logits = logits[-1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_word = id_to_word[next_id]
        generated.append(next_word)
        input_tensor = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
        logits, hidden = model(input_tensor, hidden)

    return " ".join(generated).replace(" <eos> ", "\n")


@torch.no_grad()
def build_next_word_examples(
    model: PTBLanguageModel,
    word_to_id: dict[str, int],
    id_to_word: list[str],
    output_path: Path,
    context_len: int = 8,
    top_k: int = 5,
) -> None:
    data_path = ensure_ptb_files(DATA_DIR, "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz")
    test_ids = file_to_word_ids(data_path / "ptb.test.txt", word_to_id)
    positions = [800, 1600, 3200, 6400, 12800, 25600]

    rows = [
        "# PTB Next-word Prediction Examples",
        "",
        "| Context | Target | Top-5 Predictions | Hit |",
        "|---|---:|---|---:|",
    ]
    for position in positions:
        if position <= context_len or position >= len(test_ids):
            continue
        context_ids = test_ids[position - context_len : position]
        target_id = test_ids[position]
        inputs = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)
        logits, _ = model(inputs)
        probs = torch.softmax(logits[-1], dim=-1)
        top_probs, top_ids = probs.topk(top_k)
        predictions = [
            f"{id_to_word[token_id]} ({prob:.1%})"
            for token_id, prob in zip(top_ids.tolist(), top_probs.tolist())
        ]
        hit = "yes" if target_id in top_ids.tolist() else "no"
        context = " ".join(id_to_word[token_id] for token_id in context_ids)
        rows.append(
            f"| `{context}` | `{id_to_word[target_id]}` | {', '.join(predictions)} | {hit} |"
        )

    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")


def write_generated_samples(
    model: PTBLanguageModel,
    word_to_id: dict[str, int],
    id_to_word: list[str],
    output_path: Path,
) -> None:
    prompts = ["the company", "in the year", "new york", "market analysts"]
    rows = ["# Generated PTB-style Samples", ""]
    for prompt in prompts:
        sample = generate_continuation(model, prompt, word_to_id, id_to_word)
        rows.append(f"## Prompt: `{prompt}`")
        rows.append("")
        rows.append(sample)
        rows.append("")

    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    word_to_id, id_to_word = load_vocab()
    model = load_model(len(id_to_word))

    plot_training_curves(metrics, OUTPUT_DIR / "training_curves.png")
    build_next_word_examples(model, word_to_id, id_to_word, OUTPUT_DIR / "next_word_examples.md")
    write_generated_samples(model, word_to_id, id_to_word, OUTPUT_DIR / "generated_samples.md")


if __name__ == "__main__":
    main()
