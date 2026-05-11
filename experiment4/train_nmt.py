from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Special tokens (IDs are aligned with SentencePiece configuration below).
# ---------------------------------------------------------------------------
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

NIUTRANS_ROOT = Path("/tmp/NiuTrans.SMT/sample-data/sample-submission-version")


# ---------------------------------------------------------------------------
# Data preparation: rebuild train/dev/test from the raw NiuTrans corpus.
# The previously prepared files were corrupted (test.en held Chinese; dev.*
# were misaligned), so we always validate and regenerate from scratch.
# ---------------------------------------------------------------------------


def _has_chinese_chars(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _parse_paired_blocks(path: Path) -> list[tuple[str, str]]:
    """Parse Niu.dev.txt / Niu.test.reference style files (zh, blank, en)."""
    pairs: list[tuple[str, str]] = []
    pending_zh: str | None = None
    saw_blank = False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").strip()
            if not line:
                saw_blank = True
                continue
            if pending_zh is None:
                pending_zh = line
                saw_blank = False
            else:
                if saw_blank and not _has_chinese_chars(line):
                    pairs.append((pending_zh, line))
                    pending_zh = None
                    saw_blank = False
                elif _has_chinese_chars(line):
                    # Two consecutive zh lines — drop the previous one.
                    pending_zh = line
                    saw_blank = False
                else:
                    pairs.append((pending_zh, line))
                    pending_zh = None
                    saw_blank = False
    return pairs


def prepare_data(data_dir: Path, dev_size: int = 1000, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    sentinel = data_dir / ".prepared"
    required = ["train.zh", "train.en", "dev.zh", "dev.en", "test.zh", "test.en"]
    if sentinel.exists() and all((data_dir / f).exists() for f in required) and not force:
        return

    if not NIUTRANS_ROOT.exists():
        raise FileNotFoundError(
            f"NiuTrans corpus not found at {NIUTRANS_ROOT}. Please download/extract it first."
        )

    tm_zh = NIUTRANS_ROOT / "TM-training-set" / "chinese.txt"
    tm_en = NIUTRANS_ROOT / "TM-training-set" / "english.txt"
    test_zh = NIUTRANS_ROOT / "Test-set" / "Niu.test.txt"
    ref_path = NIUTRANS_ROOT / "Reference-for-evaluation" / "Niu.test.reference"

    with open(tm_zh, encoding="utf-8") as fz, open(tm_en, encoding="utf-8") as fe:
        zh_lines = [line.rstrip("\n") for line in fz]
        en_lines = [line.rstrip("\n") for line in fe]
    if len(zh_lines) != len(en_lines):
        raise ValueError(f"TM zh/en line counts differ: {len(zh_lines)} vs {len(en_lines)}")

    paired = [
        (z.strip(), e.strip())
        for z, e in zip(zh_lines, en_lines)
        if z.strip() and e.strip() and not _has_chinese_chars(e)
    ]

    rng = random.Random(0)
    rng.shuffle(paired)
    dev_pairs = paired[:dev_size]
    train_pairs = paired[dev_size:]

    def _write(pairs: list[tuple[str, str]], zh_path: Path, en_path: Path) -> None:
        with open(zh_path, "w", encoding="utf-8") as fz, open(en_path, "w", encoding="utf-8") as fe:
            for z, e in pairs:
                fz.write(z + "\n")
                fe.write(e + "\n")

    _write(train_pairs, data_dir / "train.zh", data_dir / "train.en")
    _write(dev_pairs, data_dir / "dev.zh", data_dir / "dev.en")

    # Test source: 1000 monolingual Chinese sentences.
    with open(test_zh, encoding="utf-8") as f:
        test_src = [line.strip() for line in f if line.strip()]
    # Test reference: parse zh/blank/en blocks.
    test_ref_pairs = _parse_paired_blocks(ref_path)
    if len(test_ref_pairs) < len(test_src):
        raise ValueError(
            f"Parsed {len(test_ref_pairs)} test refs but expected ≥ {len(test_src)}."
        )
    ref_by_src = {z: e for z, e in test_ref_pairs}
    test_refs: list[str] = []
    for src in test_src:
        ref = ref_by_src.get(src)
        if ref is None:
            ref = ""  # leave empty rather than fail; sacrebleu handles it
        test_refs.append(ref)

    with open(data_dir / "test.zh", "w", encoding="utf-8") as fz, open(
        data_dir / "test.en", "w", encoding="utf-8"
    ) as fe:
        for src, ref in zip(test_src, test_refs):
            fz.write(src + "\n")
            fe.write(ref + "\n")

    sentinel.write_text("ok\n", encoding="utf-8")
    print(
        f"Prepared data: train={len(train_pairs)} dev={len(dev_pairs)} test={len(test_src)} "
        f"(missing test refs: {sum(1 for r in test_refs if not r)})"
    )


# ---------------------------------------------------------------------------
# SentencePiece BPE vocab
# ---------------------------------------------------------------------------


def train_sentencepiece(
    input_file: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
) -> None:
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=PAD_IDX,
        unk_id=UNK_IDX,
        bos_id=BOS_IDX,
        eos_id=EOS_IDX,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        normalization_rule_name="identity",  # data is already normalized
        remove_extra_whitespaces=False,
        input_sentence_size=200000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
    )


class SPMVocab:
    """Thin wrapper around SentencePieceProcessor exposing the indices we need."""

    def __init__(self, model_path: Path) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        assert self.sp.pad_id() == PAD_IDX
        assert self.sp.unk_id() == UNK_IDX
        assert self.sp.bos_id() == BOS_IDX
        assert self.sp.eos_id() == EOS_IDX

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode_ids(self, ids: list[int]) -> str:
        cleaned = [
            i for i in ids if i not in (PAD_IDX, BOS_IDX, EOS_IDX) and i < len(self)
        ]
        return self.sp.decode(cleaned)


def ensure_spm_models(
    data_dir: Path,
    spm_dir: Path,
    src_vocab_size: int,
    tgt_vocab_size: int,
    force: bool = False,
) -> tuple[SPMVocab, SPMVocab]:
    spm_dir.mkdir(parents=True, exist_ok=True)
    src_prefix = spm_dir / "zh"
    tgt_prefix = spm_dir / "en"
    src_model = src_prefix.with_suffix(".model")
    tgt_model = tgt_prefix.with_suffix(".model")

    if force or not src_model.exists():
        print(f"Training Chinese SentencePiece BPE (vocab={src_vocab_size})...")
        train_sentencepiece(
            data_dir / "train.zh", src_prefix, src_vocab_size, character_coverage=0.9995
        )
    if force or not tgt_model.exists():
        print(f"Training English SentencePiece BPE (vocab={tgt_vocab_size})...")
        train_sentencepiece(
            data_dir / "train.en", tgt_prefix, tgt_vocab_size, character_coverage=1.0
        )

    return SPMVocab(src_model), SPMVocab(tgt_model)


# ---------------------------------------------------------------------------
# Translation Dataset
# ---------------------------------------------------------------------------


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_path: Path,
        tgt_path: Path,
        src_vocab: SPMVocab,
        tgt_vocab: SPMVocab,
        max_len: int = 128,
        keep_unaligned: bool = False,
    ) -> None:
        self.max_len = max_len
        self.src: list[list[int]] = []
        self.tgt: list[list[int]] = []
        self.raw_tgt: list[str] = []  # detokenized reference for BLEU

        with open(src_path, encoding="utf-8") as fs, open(tgt_path, encoding="utf-8") as ft:
            for s_line, t_line in zip(fs, ft):
                s = s_line.strip()
                t = t_line.strip()
                if not s:
                    continue
                if not t and not keep_unaligned:
                    continue
                src_ids = src_vocab.encode(s)[: max_len - 1] + [EOS_IDX]
                tgt_ids = [BOS_IDX] + tgt_vocab.encode(t)[: max_len - 2] + [EOS_IDX]
                self.src.append(src_ids)
                self.tgt.append(tgt_ids)
                self.raw_tgt.append(t)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"src": self.src[idx], "tgt": self.tgt[idx], "raw_tgt": self.raw_tgt[idx]}


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
    src_list = [torch.tensor(item["src"], dtype=torch.long) for item in batch]
    tgt_list = [torch.tensor(item["tgt"], dtype=torch.long) for item in batch]
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return {
        "src": src_padded,
        "tgt_input": tgt_padded[:, :-1],
        "tgt_output": tgt_padded[:, 1:],
        "raw_tgt": [item["raw_tgt"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Mask utilities — convention here is "True = attend" to match
# F.scaled_dot_product_attention's bool-mask semantics.
# ---------------------------------------------------------------------------


def src_attend_mask(src: torch.Tensor) -> torch.Tensor:
    """Encoder/cross-attention key mask. Shape (B, 1, 1, S_src)."""
    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)


def tgt_attend_mask(tgt: torch.Tensor) -> torch.Tensor:
    """Decoder self-attention mask combining causal + key padding. Shape (B, 1, T, T)."""
    pad_ok = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T) — keys
    causal = torch.tril(
        torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool)
    )  # (T, T) — lower-triangular allows j <= i
    return pad_ok & causal  # broadcasts to (B, 1, T, T)


# ---------------------------------------------------------------------------
# Transformer Model Components
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attend_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = query.size(0)
        Q = self.w_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attend_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.dropout(self.self_attn(n, n, n, src_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.dropout(self.self_attn(n, n, n, tgt_mask))
        n = self.norm2(x)
        x = x + self.dropout(self.cross_attn(n, enc_out, enc_out, src_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.2,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len + 4, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # Tie target embedding with output projection.
        self.output_proj.weight = self.tgt_embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        x = self.dec_norm(x)
        return self.output_proj(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt_input, enc_out, tgt_mask, src_mask)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode(
    model: TransformerNMT,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    model.eval()
    device = src.device
    enc_out = model.encode(src, src_mask)
    B = src.size(0)
    ys = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        tgt_mask = tgt_attend_mask(ys)
        logits = model.decode(ys, enc_out, tgt_mask, src_mask)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # Once finished, keep emitting PAD so we don't pollute future decoding state.
        next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, PAD_IDX), next_tok)
        ys = torch.cat([ys, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == EOS_IDX)
        if finished.all():
            break
    return ys


@torch.no_grad()
def beam_search_decode(
    model: TransformerNMT,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    beam_size: int,
    length_penalty: float = 0.6,
) -> list[int]:
    model.eval()
    device = src.device
    enc_out = model.encode(src, src_mask)  # (1, S, d)
    enc_out = enc_out.expand(beam_size, -1, -1).contiguous()
    src_mask_b = src_mask.expand(beam_size, -1, -1, -1).contiguous()

    beams = torch.full((beam_size, 1), BOS_IDX, dtype=torch.long, device=device)
    scores = torch.full((beam_size,), -1e9, device=device)
    scores[0] = 0.0
    finished: list[tuple[list[int], float]] = []
    vocab_size = model.output_proj.out_features

    for step in range(max_len):
        tgt_mask = tgt_attend_mask(beams)
        logits = model.decode(beams, enc_out, tgt_mask, src_mask_b)  # (B, T, V)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        cand = scores.unsqueeze(1) + log_probs  # (B, V)
        flat = cand.view(-1)
        top_scores, top_idx = torch.topk(flat, beam_size)

        beam_idx = torch.div(top_idx, vocab_size, rounding_mode="floor")
        token_idx = top_idx % vocab_size

        new_beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)

        keep_mask = token_idx != EOS_IDX
        for i in range(beam_size):
            if not keep_mask[i]:
                seq = new_beams[i].tolist()[1:-1]  # strip BOS, EOS
                lp = ((5 + len(seq) + 1) ** length_penalty) / ((5 + 1) ** length_penalty)
                finished.append((seq, top_scores[i].item() / max(lp, 1e-6)))

        if keep_mask.any():
            beams = new_beams[keep_mask]
            scores = top_scores[keep_mask]
            # Re-pad if fewer than beam_size beams remain alive.
            if beams.size(0) < beam_size:
                pad_n = beam_size - beams.size(0)
                pad_seq = beams[:1].expand(pad_n, -1)
                beams = torch.cat([beams, pad_seq], dim=0)
                pad_scores = torch.full((pad_n,), -1e9, device=device)
                scores = torch.cat([scores, pad_scores], dim=0)
        else:
            break

        if len(finished) >= beam_size and step > 4:
            break

    if not finished:
        seq = beams[0].tolist()[1:]
        if seq and seq[-1] == EOS_IDX:
            seq = seq[:-1]
        return seq
    finished.sort(key=lambda x: x[1], reverse=True)
    return finished[0][0]


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    amp: bool
    device: str
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    dropout: float
    label_smoothing: float
    max_seq_len: int
    src_vocab_size: int
    tgt_vocab_size: int
    warmup_steps: int
    grad_clip: float
    beam_size: int
    eval_with_beam: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer for Chinese→English NMT.")
    parser.add_argument("--data-dir", type=Path, default=Path("experiment4/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment4/outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--src-vocab-size", type=int, default=16000)
    parser.add_argument("--tgt-vocab-size", type=int, default=16000)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--eval-with-beam", action="store_true",
                        help="Run beam-search BLEU each validation epoch (slower).")
    parser.add_argument("--rebuild-data", action="store_true",
                        help="Force re-prep of dataset and BPE models.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 1-epoch run on a 1k subset for verification.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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


def build_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
    src_vocab: SPMVocab,
    tgt_vocab: SPMVocab,
    smoke: bool = False,
) -> dict[str, DataLoader]:
    train_ds = TranslationDataset(
        config.data_dir / "train.zh", config.data_dir / "train.en",
        src_vocab, tgt_vocab, config.max_seq_len,
    )
    val_ds = TranslationDataset(
        config.data_dir / "dev.zh", config.data_dir / "dev.en",
        src_vocab, tgt_vocab, config.max_seq_len,
    )
    test_ds = TranslationDataset(
        config.data_dir / "test.zh", config.data_dir / "test.en",
        src_vocab, tgt_vocab, config.max_seq_len,
        keep_unaligned=True,
    )
    if smoke:
        train_ds.src = train_ds.src[:1000]
        train_ds.tgt = train_ds.tgt[:1000]
        train_ds.raw_tgt = train_ds.raw_tgt[:1000]
        val_ds.src = val_ds.src[:64]
        val_ds.tgt = val_ds.tgt[:64]
        val_ds.raw_tgt = val_ds.raw_tgt[:64]

    nw = config.num_workers
    common = {
        "batch_size": config.batch_size,
        "num_workers": nw,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
    }
    if nw > 0:
        common["persistent_workers"] = True
    return {
        "train": DataLoader(train_ds, shuffle=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out = batch["tgt_output"].to(device, non_blocking=True)

        s_mask = src_attend_mask(src)
        t_mask = tgt_attend_mask(tgt_in)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(src, tgt_in, s_mask, t_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        non_pad = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    avg = total_loss / max(total_tokens, 1)
    return {"loss": avg, "ppl": math.exp(min(avg, 20.0))}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tgt_vocab: SPMVocab,
    max_len: int,
    use_beam: bool = False,
    beam_size: int = 5,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hypotheses: list[str] = []
    references: list[str] = []

    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_input"].to(device, non_blocking=True)
        tgt_out = batch["tgt_output"].to(device, non_blocking=True)
        raw_refs: list[str] = batch["raw_tgt"]

        s_mask = src_attend_mask(src)
        t_mask = tgt_attend_mask(tgt_in)
        logits = model(src, tgt_in, s_mask, t_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        non_pad = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

        if use_beam:
            for i in range(src.size(0)):
                single_src = src[i : i + 1]
                single_mask = src_attend_mask(single_src)
                hyp_ids = beam_search_decode(model, single_src, single_mask, max_len, beam_size)
                hypotheses.append(tgt_vocab.decode_ids(hyp_ids))
        else:
            decoded = greedy_decode(model, src, s_mask, max_len)
            for i in range(decoded.size(0)):
                ids = decoded[i].tolist()
                if EOS_IDX in ids:
                    ids = ids[: ids.index(EOS_IDX)]
                hypotheses.append(tgt_vocab.decode_ids(ids))
        references.extend(raw_refs)

    # Filter out samples with empty references for BLEU (test set may have them).
    paired = [(h, r) for h, r in zip(hypotheses, references) if r]
    h_filt = [h for h, _ in paired]
    r_filt = [[r for _, r in paired]]
    bleu = sacrebleu.corpus_bleu(h_filt, r_filt, tokenize="13a", lowercase=True)

    avg = total_loss / max(total_tokens, 1)
    return {
        "loss": avg,
        "ppl": math.exp(min(avg, 20.0)),
        "bleu": bleu.score,
        "hypotheses": hypotheses,
        "references": references,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


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
            "metrics": {k: v for k, v in metrics.items() if not isinstance(v, list)},
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        },
        path,
    )


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.disable_amp,
        device=args.device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        max_seq_len=args.max_seq_len,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        beam_size=args.beam_size,
        eval_with_beam=args.eval_with_beam,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_data:
        if config.data_dir.exists():
            shutil.rmtree(config.data_dir)
    prepare_data(config.data_dir, dev_size=1000, force=args.rebuild_data)
    src_vocab, tgt_vocab = ensure_spm_models(
        config.data_dir,
        config.data_dir / "spm",
        config.src_vocab_size,
        config.tgt_vocab_size,
        force=args.rebuild_data,
    )
    print(f"Source vocab: {len(src_vocab)} | Target vocab: {len(tgt_vocab)}")

    dataloaders = build_dataloaders(config, device, src_vocab, tgt_vocab, smoke=args.smoke_test)
    print(
        f"Train batches: {len(dataloaders['train'])} | "
        f"Val batches: {len(dataloaders['val'])} | "
        f"Test batches: {len(dataloaders['test'])}"
    )

    model = TransformerNMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        s = step + 1
        if s < config.warmup_steps:
            return s / float(max(1, config.warmup_steps))
        return math.sqrt(config.warmup_steps / s)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_val_bleu = -1.0
    best_path = config.output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []
    epochs = 1 if args.smoke_test else config.epochs

    print(f"Device: {device} | AMP: {use_amp}")
    print(
        f"Model: d_model={config.d_model} heads={config.n_heads} "
        f"enc={config.n_encoder_layers} dec={config.n_decoder_layers} "
        f"d_ff={config.d_ff} dropout={config.dropout}"
    )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model, loader=dataloaders["train"], criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            device=device, use_amp=use_amp, grad_clip=config.grad_clip,
        )
        val_metrics = evaluate(
            model=model, loader=dataloaders["val"], criterion=criterion,
            device=device, tgt_vocab=tgt_vocab, max_len=config.max_seq_len,
            use_beam=config.eval_with_beam, beam_size=config.beam_size,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ppl": train_metrics["ppl"],
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
            "val_bleu": val_metrics["bleu"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_metrics)

        if val_metrics["bleu"] > best_val_bleu:
            best_val_bleu = val_metrics["bleu"]
            save_checkpoint(best_path, epoch, model, optimizer, val_metrics, config)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} ppl={train_metrics['ppl']:.2f} | "
            f"val_loss={val_metrics['loss']:.4f} ppl={val_metrics['ppl']:.2f} "
            f"bleu={val_metrics['bleu']:.2f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        # Show one sample translation each epoch for sanity.
        if val_metrics["hypotheses"]:
            print(f"   sample hyp: {val_metrics['hypotheses'][0][:160]}")
            print(f"   sample ref: {val_metrics['references'][0][:160]}")

    # Final evaluation with best model.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print("\nRunning test evaluation (greedy)...")
    test_greedy = evaluate(
        model=model, loader=dataloaders["test"], criterion=criterion,
        device=device, tgt_vocab=tgt_vocab, max_len=config.max_seq_len,
        use_beam=False,
    )

    print(f"\nRunning test evaluation (beam={config.beam_size})...")
    test_beam = evaluate(
        model=model, loader=dataloaders["test"], criterion=criterion,
        device=device, tgt_vocab=tgt_vocab, max_len=config.max_seq_len,
        use_beam=True, beam_size=config.beam_size,
    )

    # Persist a few predictions for the report / PPT.
    samples = []
    for i in range(min(20, len(test_beam["hypotheses"]))):
        samples.append({
            "hypothesis": test_beam["hypotheses"][i],
            "reference": test_beam["references"][i],
        })
    (config.output_dir / "test_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    payload = {
        "best_val_bleu": best_val_bleu,
        "test_loss": test_greedy["loss"],
        "test_ppl": test_greedy["ppl"],
        "test_bleu": test_greedy["bleu"],
        "test_bleu_beam": test_beam["bleu"],
        "history": history,
    }
    write_metrics(config.output_dir / "metrics.json", payload)

    print(
        f"\nBest val BLEU: {best_val_bleu:.2f}\n"
        f"Test loss: {test_greedy['loss']:.4f} | ppl: {test_greedy['ppl']:.2f}\n"
        f"Test BLEU-4 (greedy): {test_greedy['bleu']:.2f}\n"
        f"Test BLEU-4 (beam={config.beam_size}): {test_beam['bleu']:.2f}"
    )


if __name__ == "__main__":
    main()
