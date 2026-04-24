from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


RANDOM_SEED = 42
PROJECT_DIR_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/content/drive/MyDrive/CS156 Final Assignment"),
    Path("/content/drive/MyDrive/Downloads/CS156 Final Assignment"),
    Path(r"C:\Users\Mr. Paul\Downloads\CS156 Final Assignment"),
]

DATA_DIR_NAMES = ("Cleaned Data", "Cleaned_Data", "cleaned_data")
OUTPUT_DIR_NAME = "Final Evaluation Outputs"

MODEL_ORDER = [
    "rnn_lstm_scratch",
    "transformer_scratch",
    "opus_mt_pretrained",
    "opus_mt_fine_tuned",
]

MODEL_DISPLAY_NAMES = {
    "rnn_lstm_scratch": "RNN LSTM from scratch",
    "transformer_scratch": "Transformer from scratch",
    "opus_mt_pretrained": "OPUS-MT pretrained",
    "opus_mt_fine_tuned": "OPUS-MT fine-tuned",
}

MODEL_SHORT_NAMES = {
    "rnn_lstm_scratch": "RNN",
    "transformer_scratch": "Transformer",
    "opus_mt_pretrained": "OPUS pretrained",
    "opus_mt_fine_tuned": "OPUS fine-tuned",
}

MODEL_COLORS = {
    "rnn_lstm_scratch": "#3B6EA8",
    "transformer_scratch": "#2D936C",
    "opus_mt_pretrained": "#9C6ADE",
    "opus_mt_fine_tuned": "#D9843B",
}

# None means the full Combined_test.jsonl file. For a smoke test, run:
#   $env:FINAL_EVAL_LIMIT=25; python final_model_evaluation.py
EVALUATION_LIMIT = int(os.environ.get("FINAL_EVAL_LIMIT", "0")) or None
FORCE_REGENERATE = os.environ.get("FINAL_EVAL_FORCE", "0").lower() in {"1", "true", "yes"}
GENERATE_MISSING_PREDICTIONS = os.environ.get("FINAL_EVAL_GENERATE", "1").lower() not in {"0", "false", "no"}
USE_EXISTING_OPUS_PREVIEWS = os.environ.get("FINAL_EVAL_USE_OPUS_PREVIEWS", "1").lower() not in {"0", "false", "no"}

RNN_BATCH_SIZE = int(os.environ.get("FINAL_EVAL_RNN_BATCH_SIZE", "64"))
TRANSFORMER_BATCH_SIZE = int(os.environ.get("FINAL_EVAL_TRANSFORMER_BATCH_SIZE", "32"))
OPUS_BATCH_SIZE = int(os.environ.get("FINAL_EVAL_OPUS_BATCH_SIZE", "32"))
OPUS_GENERATION_MAX_NEW_TOKENS = 80
OPUS_GENERATION_NUM_BEAMS = 1

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_data_dir(project_dir: Path) -> Path | None:
    for name in DATA_DIR_NAMES:
        candidate = project_dir / name
        if (candidate / "Combined_train.jsonl").exists() and (candidate / "Combined_test.jsonl").exists():
            return candidate
    return None


def resolve_project_dir() -> Path:
    if "google.colab" in sys.modules:
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Warning: could not mount Google Drive automatically: {exc}")

    for candidate in PROJECT_DIR_CANDIDATES:
        if find_data_dir(candidate) is not None:
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the CS156 project directory with Combined_train/test JSONL files.")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_metric(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u02bc", "'")
    text = normalize_whitespace(text)
    return " ".join(TOKEN_PATTERN.findall(text))


def token_count(text: str) -> int:
    normalized = normalize_for_metric(text)
    if not normalized:
        return 0
    return len(normalized.split())


def add_pair_keys(df: pd.DataFrame, english_col: str = "English", reference_col: str = "Reference Igbo") -> pd.DataFrame:
    keyed = df.copy()
    keyed["_metric_english"] = keyed[english_col].map(normalize_for_metric)
    keyed["_metric_reference"] = keyed[reference_col].map(normalize_for_metric)
    keyed["_pair_key"] = keyed["_metric_english"] + "\n" + keyed["_metric_reference"]
    keyed["_pair_occurrence"] = keyed.groupby("_pair_key").cumcount()
    return keyed


def load_test_dataframe(data_dir: Path) -> pd.DataFrame:
    records = read_jsonl(data_dir / "Combined_test.jsonl")
    test_df = pd.DataFrame(records)
    test_df = test_df.rename(columns={"Igbo": "Reference Igbo"})
    test_df.insert(0, "row_id", np.arange(len(test_df), dtype=int))

    source_frames = []
    for source_name, filename in [
        ("Bible", "Bible_test_cleaned.jsonl"),
        ("JHW", "cleaned_JHW_test.jsonl"),
    ]:
        path = data_dir / filename
        if path.exists():
            source_df = pd.DataFrame(read_jsonl(path)).rename(columns={"Igbo": "Reference Igbo"})
            source_df["source"] = source_name
            source_frames.append(source_df[["English", "Reference Igbo", "source"]])

    if source_frames:
        source_lookup = add_pair_keys(pd.concat(source_frames, ignore_index=True))
        source_lookup = source_lookup[["_pair_key", "_pair_occurrence", "source"]]
        test_keyed = add_pair_keys(test_df)
        test_df = test_keyed.merge(source_lookup, on=["_pair_key", "_pair_occurrence"], how="left")
        test_df["source"] = test_df["source"].fillna("Unknown")
        test_df = test_df.drop(columns=["_metric_english", "_metric_reference", "_pair_key", "_pair_occurrence"])
    else:
        test_df["source"] = "Unknown"

    test_df["english_token_count"] = test_df["English"].map(token_count)
    test_df["reference_token_count"] = test_df["Reference Igbo"].map(token_count)
    return test_df


def make_output_dirs(project_dir: Path) -> dict[str, Path]:
    output_dir_name = OUTPUT_DIR_NAME
    if EVALUATION_LIMIT is not None:
        output_dir_name = f"{OUTPUT_DIR_NAME} Smoke Test {EVALUATION_LIMIT}"
    output_dir = project_dir / output_dir_name
    dirs = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def progress_iter(iterable, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def prediction_cache_path(dirs: dict[str, Path], model_slug: str) -> Path:
    return dirs["predictions"] / f"{model_slug}_predictions.csv"


def load_cached_predictions(dirs: dict[str, Path], model_slug: str, expected_rows: int) -> pd.DataFrame | None:
    path = prediction_cache_path(dirs, model_slug)
    if FORCE_REGENERATE or not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) != expected_rows:
        print(f"Ignoring cached {model_slug} predictions because it has {len(df)} rows, expected {expected_rows}.")
        return None
    return df


def save_predictions(dirs: dict[str, Path], model_slug: str, pred_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["row_id", "source", "English", "Reference Igbo", "Predicted Igbo"]
    pred_df = pred_df[keep_cols].sort_values("row_id").reset_index(drop=True)
    pred_df.to_csv(prediction_cache_path(dirs, model_slug), index=False)
    return pred_df


def align_existing_preview(preview_path: Path, test_df: pd.DataFrame, model_slug: str) -> pd.DataFrame:
    preview = pd.read_csv(preview_path)
    required_cols = {"English", "Reference Igbo", "Predicted Igbo"}
    missing = required_cols - set(preview.columns)
    if missing:
        raise ValueError(f"{preview_path} is missing columns: {sorted(missing)}")

    preview_keyed = add_pair_keys(preview)
    preview_keyed = preview_keyed[["_pair_key", "_pair_occurrence", "Predicted Igbo"]]

    test_keyed = add_pair_keys(test_df)
    aligned = test_keyed.merge(preview_keyed, on=["_pair_key", "_pair_occurrence"], how="left")
    missing_predictions = aligned["Predicted Igbo"].isna().sum()
    if missing_predictions:
        raise ValueError(
            f"Could not align {missing_predictions} rows from {preview_path.name} to the active test set for {model_slug}."
        )

    aligned = aligned.drop(columns=["_metric_english", "_metric_reference", "_pair_key", "_pair_occurrence"])
    return aligned


def load_or_align_opus_preview(
    dirs: dict[str, Path],
    model_slug: str,
    preview_path: Path,
    test_df: pd.DataFrame,
) -> pd.DataFrame | None:
    cached = load_cached_predictions(dirs, model_slug, len(test_df))
    if cached is not None:
        return cached
    if not USE_EXISTING_OPUS_PREVIEWS:
        return None
    if not preview_path.exists():
        return None
    print(f"Using existing full-test preview for {MODEL_DISPLAY_NAMES[model_slug]}: {preview_path}")
    pred_df = align_existing_preview(preview_path, test_df, model_slug)
    return save_predictions(dirs, model_slug, pred_df)


def clear_torch_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def generate_rnn_predictions(project_dir: Path, dirs: dict[str, Path], test_df: pd.DataFrame) -> pd.DataFrame:
    cached = load_cached_predictions(dirs, "rnn_lstm_scratch", len(test_df))
    if cached is not None:
        return cached
    if not GENERATE_MISSING_PREDICTIONS:
        raise FileNotFoundError("Missing RNN predictions and generation is disabled.")

    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

    checkpoint_path = project_dir / "RNN Outputs" / "rnn_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing RNN checkpoint: {checkpoint_path}")

    print(f"Loading RNN checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    special_tokens = {"pad": "<pad>", "sos": "<sos>", "eos": "<eos>", "unk": "<unk>"}

    class Vocabulary:
        def __init__(self, tokens):
            self.itos = list(tokens)
            self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        def encode(self, tokens, add_special_tokens=True):
            ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
            if add_special_tokens:
                ids = [self.sos_idx] + ids + [self.eos_idx]
            return ids

        def decode(self, ids):
            out_tokens = []
            special_values = set(special_tokens.values())
            for idx in ids:
                idx = int(idx)
                if idx < 0 or idx >= len(self.itos):
                    continue
                token = self.itos[idx]
                if token == special_tokens["eos"]:
                    break
                if token not in special_values:
                    out_tokens.append(token)
            return " ".join(out_tokens)

        @property
        def pad_idx(self):
            return self.stoi[special_tokens["pad"]]

        @property
        def sos_idx(self):
            return self.stoi[special_tokens["sos"]]

        @property
        def eos_idx(self):
            return self.stoi[special_tokens["eos"]]

        @property
        def unk_idx(self):
            return self.stoi[special_tokens["unk"]]

        def __len__(self):
            return len(self.itos)

    class Encoder(nn.Module):
        def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
            super().__init__()
            self.num_layers = num_layers
            self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
            self.dropout = nn.Dropout(dropout)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            self.hidden_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
            self.cell_bridge = nn.Linear(hidden_dim * 2, hidden_dim)

        def _combine_directions(self, state):
            combined_states = []
            for layer in range(self.num_layers):
                forward_state = state[2 * layer]
                backward_state = state[2 * layer + 1]
                combined_states.append(torch.cat([forward_state, backward_state], dim=1))
            return torch.stack(combined_states, dim=0)

        def forward(self, source, source_lengths):
            embedded = self.dropout(self.embedding(source))
            packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=True)
            _, (hidden, cell) = self.lstm(packed)
            hidden = torch.tanh(self.hidden_bridge(self._combine_directions(hidden)))
            cell = torch.tanh(self.cell_bridge(self._combine_directions(cell)))
            return hidden, cell

    class Decoder(nn.Module):
        def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
            super().__init__()
            self.output_dim = output_dim
            self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=pad_idx)
            self.dropout = nn.Dropout(dropout)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, input_token, hidden, cell):
            input_token = input_token.unsqueeze(1)
            embedded = self.dropout(self.embedding(input_token))
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            prediction = self.fc_out(output.squeeze(1))
            return prediction, hidden, cell

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

    config = checkpoint["config"]
    source_vocab = Vocabulary(checkpoint["source_vocab_itos"])
    target_vocab = Vocabulary(checkpoint["target_vocab_itos"])
    max_source_tokens = int(config.get("max_source_tokens", 40))
    max_decoding_steps = int(config.get("max_target_tokens", 40))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    encoder = Encoder(
        input_dim=len(source_vocab),
        embed_dim=int(config.get("embed_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 384)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.3)),
        pad_idx=source_vocab.pad_idx,
    )
    decoder = Decoder(
        output_dim=len(target_vocab),
        embed_dim=int(config.get("embed_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 384)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.3)),
        pad_idx=target_vocab.pad_idx,
    )
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    def translate_batch(texts: list[str]) -> list[str]:
        source_ids = []
        for text in texts:
            tokens = TOKEN_PATTERN.findall(normalize_for_metric(text))[:max_source_tokens]
            source_ids.append(torch.tensor(source_vocab.encode(tokens), dtype=torch.long))

        lengths = torch.tensor([len(ids) for ids in source_ids], dtype=torch.long)
        order = torch.argsort(lengths, descending=True)
        sorted_ids = [source_ids[i] for i in order.tolist()]
        sorted_lengths = lengths[order]
        source_batch = pad_sequence(sorted_ids, batch_first=True, padding_value=source_vocab.pad_idx).to(device)

        with torch.no_grad():
            hidden, cell = model.encoder(source_batch, sorted_lengths)
            batch_size = len(texts)
            input_token = torch.full((batch_size,), target_vocab.sos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            generated_ids = [[] for _ in range(batch_size)]

            for _ in range(max_decoding_steps):
                output, hidden, cell = model.decoder(input_token, hidden, cell)
                next_token = output.argmax(dim=1)

                for sorted_pos, token_id in enumerate(next_token.detach().cpu().tolist()):
                    if finished[sorted_pos]:
                        continue
                    if token_id == target_vocab.eos_idx:
                        finished[sorted_pos] = True
                    else:
                        generated_ids[sorted_pos].append(token_id)

                input_token = next_token
                if bool(finished.all()):
                    break

        unsorted_outputs = [None] * len(texts)
        for sorted_pos, original_pos in enumerate(order.tolist()):
            unsorted_outputs[original_pos] = target_vocab.decode(generated_ids[sorted_pos])
        return unsorted_outputs

    predictions = []
    texts = test_df["English"].tolist()
    ranges = range(0, len(texts), RNN_BATCH_SIZE)
    for start in progress_iter(ranges, "RNN predictions"):
        predictions.extend(translate_batch(texts[start : start + RNN_BATCH_SIZE]))

    pred_df = test_df[["row_id", "source", "English", "Reference Igbo"]].copy()
    pred_df["Predicted Igbo"] = predictions

    del model, checkpoint, encoder, decoder
    clear_torch_memory()
    return save_predictions(dirs, "rnn_lstm_scratch", pred_df)


def generate_transformer_predictions(project_dir: Path, dirs: dict[str, Path], test_df: pd.DataFrame) -> pd.DataFrame:
    cached = load_cached_predictions(dirs, "transformer_scratch", len(test_df))
    if cached is not None:
        return cached
    if not GENERATE_MISSING_PREDICTIONS:
        raise FileNotFoundError("Missing Transformer predictions and generation is disabled.")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    output_dir_candidates = [project_dir / "Transformers Outputs", project_dir / "Transformer Outputs"]
    output_dir = next((path for path in output_dir_candidates if path.exists()), None)
    if output_dir is None:
        raise FileNotFoundError("Could not locate Transformer output directory.")

    best_model_candidates = sorted(output_dir.glob("*_best.keras"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not best_model_candidates:
        raise FileNotFoundError(f"No *_best.keras checkpoint found in {output_dir}.")

    best_model_path = best_model_candidates[0]
    run_name = best_model_path.name[: -len("_best.keras")]
    config_path = output_dir / f"{run_name}_config.json"
    source_vocab_path = output_dir / f"{run_name}_source_vocab.txt"
    target_vocab_path = output_dir / f"{run_name}_target_vocab.txt"

    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    max_source_tokens = int(run_config["max_source_tokens"])
    max_target_tokens = int(run_config["max_target_tokens"])
    target_sequence_length = max_target_tokens + 2

    source_vocab = source_vocab_path.read_text(encoding="utf-8").splitlines()
    target_vocab = target_vocab_path.read_text(encoding="utf-8").splitlines()
    source_token_lookup = {token: index for index, token in enumerate(source_vocab)}
    target_token_lookup = {token: index for index, token in enumerate(target_vocab)}
    target_index_lookup = dict(enumerate(target_vocab))

    start_token = "[start]"
    end_token = "[end]"
    source_unk_id = source_token_lookup.get("[UNK]", 1)
    target_unk_id = target_token_lookup.get("[UNK]", None)
    start_token_id = target_token_lookup[start_token]
    end_token_id = target_token_lookup[end_token]

    @keras.utils.register_keras_serializable(package="cs156")
    def masked_loss(y_true, y_pred):
        loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

    @keras.utils.register_keras_serializable(package="cs156")
    def masked_accuracy(y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        matches *= mask
        return tf.reduce_sum(matches) / tf.maximum(tf.reduce_sum(mask), 1.0)

    @keras.utils.register_keras_serializable(package="cs156")
    class PaddingMask(layers.Layer):
        def call(self, inputs):
            return tf.not_equal(inputs, 0)

        def get_config(self):
            return super().get_config()

    @keras.utils.register_keras_serializable(package="cs156")
    class PositionalEmbedding(layers.Layer):
        def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.sequence_length = sequence_length
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

        def call(self, inputs):
            length = tf.shape(inputs)[-1]
            positions = tf.range(start=0, limit=length, delta=1)
            return self.token_embeddings(inputs) + self.position_embeddings(positions)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "sequence_length": self.sequence_length,
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                }
            )
            return config

    @keras.utils.register_keras_serializable(package="cs156")
    class TransformerEncoder(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, dropout, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.dense_dim = dense_dim
            self.num_heads = num_heads
            self.dropout = dropout
            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout,
            )
            self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
            self.layernorm_1 = layers.LayerNormalization()
            self.layernorm_2 = layers.LayerNormalization()
            self.dropout_1 = layers.Dropout(dropout)
            self.dropout_2 = layers.Dropout(dropout)

        def call(self, inputs, training=False, padding_mask=None):
            attention_mask = None
            if padding_mask is not None:
                attention_mask = tf.cast(padding_mask[:, tf.newaxis, :], dtype="bool")
            attention_output = self.attention(
                query=inputs,
                value=inputs,
                key=inputs,
                attention_mask=attention_mask,
                training=training,
            )
            attention_output = self.dropout_1(attention_output, training=training)
            proj_input = self.layernorm_1(inputs + attention_output)
            proj_output = self.dense_proj(proj_input)
            proj_output = self.dropout_2(proj_output, training=training)
            return self.layernorm_2(proj_input + proj_output)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "embed_dim": self.embed_dim,
                    "dense_dim": self.dense_dim,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                }
            )
            return config

    @keras.utils.register_keras_serializable(package="cs156")
    class TransformerDecoder(layers.Layer):
        def __init__(self, embed_dim, dense_dim, num_heads, dropout, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim
            self.dense_dim = dense_dim
            self.num_heads = num_heads
            self.dropout = dropout
            self.attention_1 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout,
            )
            self.attention_2 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout,
            )
            self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
            self.layernorm_1 = layers.LayerNormalization()
            self.layernorm_2 = layers.LayerNormalization()
            self.layernorm_3 = layers.LayerNormalization()
            self.dropout_1 = layers.Dropout(dropout)
            self.dropout_2 = layers.Dropout(dropout)
            self.dropout_3 = layers.Dropout(dropout)

        def get_causal_attention_mask(self, inputs):
            batch_size = tf.shape(inputs)[0]
            sequence_length = tf.shape(inputs)[1]
            mask = tf.linalg.band_part(tf.ones((sequence_length, sequence_length), dtype="bool"), -1, 0)
            mask = tf.expand_dims(mask, axis=0)
            return tf.tile(mask, [batch_size, 1, 1])

        def call(self, inputs, encoder_outputs, training=False, decoder_padding_mask=None, encoder_padding_mask=None):
            causal_mask = self.get_causal_attention_mask(inputs)
            self_attention_mask = causal_mask
            if decoder_padding_mask is not None:
                padding_mask = tf.cast(decoder_padding_mask[:, tf.newaxis, :], dtype="bool")
                self_attention_mask = tf.logical_and(causal_mask, padding_mask)

            attention_output_1 = self.attention_1(
                query=inputs,
                value=inputs,
                key=inputs,
                attention_mask=self_attention_mask,
                training=training,
            )
            attention_output_1 = self.dropout_1(attention_output_1, training=training)
            out_1 = self.layernorm_1(inputs + attention_output_1)

            cross_attention_mask = None
            if encoder_padding_mask is not None:
                cross_attention_mask = tf.cast(encoder_padding_mask[:, tf.newaxis, :], dtype="bool")

            attention_output_2 = self.attention_2(
                query=out_1,
                value=encoder_outputs,
                key=encoder_outputs,
                attention_mask=cross_attention_mask,
                training=training,
            )
            attention_output_2 = self.dropout_2(attention_output_2, training=training)
            out_2 = self.layernorm_2(out_1 + attention_output_2)
            proj_output = self.dense_proj(out_2)
            proj_output = self.dropout_3(proj_output, training=training)
            return self.layernorm_3(out_2 + proj_output)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "embed_dim": self.embed_dim,
                    "dense_dim": self.dense_dim,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                }
            )
            return config

    custom_objects = {
        "masked_loss": masked_loss,
        "masked_accuracy": masked_accuracy,
        "PaddingMask": PaddingMask,
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
    }

    print(f"Loading Transformer checkpoint: {best_model_path}")
    model = keras.models.load_model(best_model_path, custom_objects=custom_objects)

    def encode_source_batch(texts):
        encoded = np.zeros((len(texts), max_source_tokens), dtype="int64")
        for row_index, text in enumerate(texts):
            tokens = TOKEN_PATTERN.findall(normalize_for_metric(text))[:max_source_tokens]
            token_ids = [source_token_lookup.get(token, source_unk_id) for token in tokens]
            encoded[row_index, : len(token_ids)] = token_ids
        return tf.convert_to_tensor(encoded, dtype=tf.int64)

    def translate_batch(texts: list[str]) -> list[str]:
        encoder_inputs = encode_source_batch(texts)
        current_batch_size = len(texts)
        decoder_inputs = np.zeros((current_batch_size, target_sequence_length - 1), dtype="int64")
        decoder_inputs[:, 0] = start_token_id
        generated_token_ids = [[] for _ in range(current_batch_size)]
        finished = np.zeros(current_batch_size, dtype=bool)

        for step in range(max_target_tokens):
            decoder_inputs_tensor = tf.convert_to_tensor(decoder_inputs, dtype=tf.int64)
            predictions = model(
                {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs_tensor},
                training=False,
            )
            step_logits = predictions[:, step, :]

            blocked_ids = [0, start_token_id]
            if target_unk_id is not None:
                blocked_ids.append(target_unk_id)
            blocked_ids = tf.constant(blocked_ids, dtype=tf.int32)
            blocked_mask = tf.reduce_sum(
                tf.one_hot(blocked_ids, depth=tf.shape(step_logits)[-1], dtype=step_logits.dtype),
                axis=0,
            )
            step_logits = step_logits - blocked_mask[tf.newaxis, :] * tf.cast(1e9, step_logits.dtype)
            next_token_ids = tf.argmax(step_logits, axis=-1, output_type=tf.int64).numpy()

            for row_index, token_id in enumerate(next_token_ids):
                if finished[row_index]:
                    continue
                token_id = int(token_id)
                if token_id in {0, end_token_id}:
                    finished[row_index] = True
                    continue
                generated_token_ids[row_index].append(token_id)
                if step + 1 < decoder_inputs.shape[1]:
                    decoder_inputs[row_index, step + 1] = token_id

            if finished.all():
                break

        outputs = []
        for token_ids in generated_token_ids:
            tokens = [target_index_lookup.get(token_id, "") for token_id in token_ids]
            tokens = [token for token in tokens if token and token not in {start_token, end_token}]
            outputs.append(" ".join(tokens))
        return outputs

    predictions = []
    texts = test_df["English"].tolist()
    ranges = range(0, len(texts), TRANSFORMER_BATCH_SIZE)
    for start in progress_iter(ranges, "Transformer predictions"):
        predictions.extend(translate_batch(texts[start : start + TRANSFORMER_BATCH_SIZE]))

    pred_df = test_df[["row_id", "source", "English", "Reference Igbo"]].copy()
    pred_df["Predicted Igbo"] = predictions

    del model
    keras.backend.clear_session()
    gc.collect()
    return save_predictions(dirs, "transformer_scratch", pred_df)


def generate_opus_predictions(
    project_dir: Path,
    dirs: dict[str, Path],
    test_df: pd.DataFrame,
    model_slug: str,
    model_dir_name: str,
) -> pd.DataFrame:
    cached = load_cached_predictions(dirs, model_slug, len(test_df))
    if cached is not None:
        return cached
    if not GENERATE_MISSING_PREDICTIONS:
        raise FileNotFoundError(f"Missing {model_slug} predictions and generation is disabled.")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir = project_dir / "OPUS_MT Outputs" / model_dir_name
    model_name_or_path = str(model_dir) if model_dir.exists() else "Helsinki-NLP/opus-mt-en-ig"

    print(f"Generating {MODEL_DISPLAY_NAMES[model_slug]} predictions from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    texts = test_df["English"].tolist()
    ranges = range(0, len(texts), OPUS_BATCH_SIZE)
    for start in progress_iter(ranges, f"{MODEL_SHORT_NAMES[model_slug]} predictions"):
        batch_texts = texts[start : start + OPUS_BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=OPUS_GENERATION_MAX_NEW_TOKENS,
                num_beams=OPUS_GENERATION_NUM_BEAMS,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    pred_df = test_df[["row_id", "source", "English", "Reference Igbo"]].copy()
    pred_df["Predicted Igbo"] = predictions

    del model, tokenizer
    clear_torch_memory()
    return save_predictions(dirs, model_slug, pred_df)


def compute_metric_row(model_slug: str, pred_df: pd.DataFrame, split_label: str) -> dict:
    refs = pred_df["Reference Igbo"].map(normalize_for_metric).tolist()
    preds = pred_df["Predicted Igbo"].fillna("").map(normalize_for_metric).tolist()
    exact_match = float(np.mean([pred == ref for pred, ref in zip(preds, refs)]))
    bleu = BLEU(tokenize="none", effective_order=True).corpus_score(preds, [refs]).score
    chrf = CHRF().corpus_score(preds, [refs]).score

    ref_lengths = np.array([max(token_count(text), 1) for text in pred_df["Reference Igbo"]], dtype=float)
    pred_lengths = np.array([token_count(text) for text in pred_df["Predicted Igbo"].fillna("")], dtype=float)
    length_ratios = pred_lengths / ref_lengths

    return {
        "model_slug": model_slug,
        "model": MODEL_DISPLAY_NAMES[model_slug],
        "split": split_label,
        "rows": len(pred_df),
        "Exact match": round(exact_match, 4),
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "avg_prediction_tokens": round(float(pred_lengths.mean()), 2),
        "avg_reference_tokens": round(float(ref_lengths.mean()), 2),
        "avg_prediction_reference_length_ratio": round(float(length_ratios.mean()), 3),
    }


def compute_metrics(prediction_frames: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    source_rows = []

    for model_slug in MODEL_ORDER:
        pred_df = prediction_frames[model_slug]
        overall_rows.append(compute_metric_row(model_slug, pred_df, "all"))
        for source in sorted(pred_df["source"].dropna().unique()):
            source_df = pred_df[pred_df["source"] == source]
            if len(source_df):
                source_rows.append(compute_metric_row(model_slug, source_df, source))

    overall_df = pd.DataFrame(overall_rows)
    by_source_df = pd.DataFrame(source_rows)

    overall_df.to_csv(dirs["tables"] / "model_metrics_overall.csv", index=False)
    by_source_df.to_csv(dirs["tables"] / "model_metrics_by_source.csv", index=False)
    return overall_df, by_source_df


def build_prediction_tables(prediction_frames: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frames = []
    for model_slug, pred_df in prediction_frames.items():
        frame = pred_df.copy()
        frame["model_slug"] = model_slug
        frame["model"] = MODEL_DISPLAY_NAMES[model_slug]
        frame["prediction_metric_text"] = frame["Predicted Igbo"].fillna("").map(normalize_for_metric)
        frame["reference_metric_text"] = frame["Reference Igbo"].map(normalize_for_metric)
        frame["prediction_token_count"] = frame["Predicted Igbo"].fillna("").map(token_count)
        frame["reference_token_count"] = frame["Reference Igbo"].map(token_count)
        long_frames.append(frame)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df.to_csv(dirs["predictions"] / "all_model_predictions_long.csv", index=False)

    base = next(iter(prediction_frames.values()))[["row_id", "source", "English", "Reference Igbo"]].copy()
    wide_df = base.copy()
    for model_slug in MODEL_ORDER:
        wide_df[MODEL_SHORT_NAMES[model_slug]] = prediction_frames[model_slug]["Predicted Igbo"].values
    wide_df.to_csv(dirs["predictions"] / "all_model_predictions_wide.csv", index=False)
    return long_df, wide_df


def load_training_history(project_dir: Path) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}

    rnn_path = project_dir / "RNN Outputs" / "rnn_history.csv"
    if rnn_path.exists():
        histories["rnn"] = pd.read_csv(rnn_path)

    transformer_path = project_dir / "Transformers Outputs" / "transformer_all_baseline_history.csv"
    if not transformer_path.exists():
        transformer_path = project_dir / "Transformer Outputs" / "transformer_all_baseline_history.csv"
    if transformer_path.exists():
        transformer_history = pd.read_csv(transformer_path)
        if "epoch" in transformer_history.columns:
            transformer_history["epoch"] = transformer_history["epoch"] + 1
        histories["transformer"] = transformer_history

    trainer_state_path = project_dir / "OPUS_MT Outputs" / "2_fine_tuned_checkpoints" / "trainer_state.json"
    if trainer_state_path.exists():
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        log_history = pd.DataFrame(trainer_state.get("log_history", []))
        if not log_history.empty:
            histories["opus_fine_tuned"] = log_history

    return histories


def wrap_text(text: str, width: int = 48, max_chars: int = 180) -> str:
    import textwrap

    text = normalize_whitespace(text)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return "\n".join(textwrap.wrap(text, width=width))


def save_captioned_figure(fig, dirs: dict[str, Path], filename: str, caption: str, captions: list[dict]) -> None:
    path = dirs["figures"] / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    captions.append({"figure": filename, "caption": caption})


def create_visualizations(
    project_dir: Path,
    data_dir: Path,
    dirs: dict[str, Path],
    test_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    by_source_df: pd.DataFrame,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    captions: list[dict] = []

    # Figure 1: dataset composition.
    composition_rows = []
    for split, bible_file, jhw_file, combined_file in [
        ("train", "Bible_train_cleaned.jsonl", "cleaned_JHW_train.jsonl", "Combined_train.jsonl"),
        ("test", "Bible_test_cleaned.jsonl", "cleaned_JHW_test.jsonl", "Combined_test.jsonl"),
    ]:
        for source, filename in [("Bible", bible_file), ("JHW", jhw_file)]:
            path = data_dir / filename
            composition_rows.append({"split": split, "source": source, "rows": count_jsonl(path) if path.exists() else 0})
        combined_path = data_dir / combined_file
        composition_rows.append(
            {"split": split, "source": "Combined total", "rows": count_jsonl(combined_path) if combined_path.exists() else 0}
        )

    composition_df = pd.DataFrame(composition_rows)
    composition_df.to_csv(dirs["tables"] / "dataset_composition.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = composition_df[composition_df["source"] != "Combined total"]
    pivot = plot_df.pivot(index="split", columns="source", values="rows").loc[["train", "test"]]
    pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("Training and Test Set Composition")
    ax.set_xlabel("Split")
    ax.set_ylabel("Sentence pairs")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(title="Source")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "01_dataset_composition.png",
        "Figure 1. Number of English-Igbo sentence pairs from the Bible corpus and JHW Hugging Face corpus in the train and test splits.",
        captions,
    )
    plt.close(fig)

    # Figure 2: length distributions.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, column, title in [
        (axes[0], "english_token_count", "English source length"),
        (axes[1], "reference_token_count", "Igbo reference length"),
    ]:
        for source, color in [("Bible", "#4C78A8"), ("JHW", "#F58518"), ("Unknown", "#999999")]:
            values = test_df.loc[test_df["source"] == source, column]
            if len(values):
                ax.hist(values, bins=40, alpha=0.55, label=source, color=color)
        ax.set_title(title)
        ax.set_xlabel("Metric-normalized tokens")
        ax.set_ylabel("Rows")
        ax.legend(title="Source")
    fig.suptitle("Out-of-Sample Test Sentence Lengths")
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "02_test_sentence_lengths.png",
        "Figure 2. Distribution of source and reference token lengths in the held-out Combined_test set, after applying the same tokenization used for metric calculation.",
        captions,
    )
    plt.close(fig)

    # Figure 3: training curves.
    histories = load_training_history(project_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    if "rnn" in histories:
        h = histories["rnn"]
        ax.plot(h["epoch"], h["train_loss"], marker="o", label="train loss", color="#3B6EA8")
        ax.plot(h["epoch"], h["val_loss"], marker="o", label="validation loss", color="#82A6D1")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "RNN history not found", ha="center", va="center")
    ax.set_title("RNN LSTM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax = axes[1]
    if "transformer" in histories:
        h = histories["transformer"]
        ax.plot(h["epoch"], h["loss"], marker="o", label="train loss", color="#2D936C")
        ax.plot(h["epoch"], h["val_loss"], marker="o", label="validation loss", color="#7BC8A4")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Transformer history not found", ha="center", va="center")
    ax.set_title("Transformer")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax = axes[2]
    if "opus_fine_tuned" in histories:
        h = histories["opus_fine_tuned"]
        train_h = h[h["loss"].notna()] if "loss" in h.columns else pd.DataFrame()
        eval_h = h[h["eval_loss"].notna()] if "eval_loss" in h.columns else pd.DataFrame()
        if not train_h.empty:
            ax.plot(train_h["step"], train_h["loss"], label="training loss", color="#D9843B", linewidth=1.5)
        if not eval_h.empty:
            ax.scatter(eval_h["step"], eval_h["eval_loss"], label="monitor eval loss", color="#9C6ADE", s=28)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "OPUS fine-tune history not found", ha="center", va="center")
    ax.set_title("Fine-tuned OPUS-MT")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    fig.suptitle("Training Curves for Trainable Models")
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "03_training_curves.png",
        "Figure 3. Loss curves for the trainable models: the scratch RNN, scratch Transformer, and fine-tuned OPUS-MT model.",
        captions,
    )
    plt.close(fig)

    # Figure 4: BLEU and chrF comparison.
    metric_df = overall_df.copy()
    metric_df["model_short"] = metric_df["model_slug"].map(MODEL_SHORT_NAMES)
    melted = metric_df.melt(
        id_vars=["model_slug", "model_short"],
        value_vars=["BLEU", "chrF"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    for offset, metric, color in [(-width / 2, "BLEU", "#4C78A8"), (width / 2, "chrF", "#F58518")]:
        values = [float(melted[(melted["model_slug"] == slug) & (melted["metric"] == metric)]["score"].iloc[0]) for slug in MODEL_ORDER]
        bars = ax.bar(x + offset, values, width=width, label=metric, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT_NAMES[slug] for slug in MODEL_ORDER], rotation=20, ha="right")
    ax.set_title("Translation Quality on the Same Held-Out Test Set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="Metric")
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "04_bleu_chrf_comparison.png",
        "Figure 4. Corpus BLEU and chrF for all four models on the same Combined_test rows, using one shared normalization and metric implementation.",
        captions,
    )
    plt.close(fig)

    # Figure 5: exact match comparison.
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [float(overall_df.loc[overall_df["model_slug"] == slug, "Exact match"].iloc[0]) for slug in MODEL_ORDER]
    bars = ax.bar(
        [MODEL_SHORT_NAMES[slug] for slug in MODEL_ORDER],
        values,
        color=[MODEL_COLORS[slug] for slug in MODEL_ORDER],
    )
    ax.bar_label(bars, labels=[f"{value:.3%}" for value in values], padding=3)
    ax.set_title("Exact Match Rate")
    ax.set_xlabel("Model")
    ax.set_ylabel("Share of test rows with exact normalized match")
    ax.set_ylim(0, max(values) * 1.35 if values else 0.01)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "05_exact_match_comparison.png",
        "Figure 5. Exact-match accuracy after shared lowercasing, punctuation tokenization, and whitespace normalization. This metric is strict for translation, so values are expected to be low.",
        captions,
    )
    plt.close(fig)

    # Figure 6: source-specific BLEU/chrF.
    source_metric_df = by_source_df[by_source_df["split"].isin(["Bible", "JHW"])].copy()
    source_metric_df["model_short"] = source_metric_df["model_slug"].map(MODEL_SHORT_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, metric in zip(axes, ["BLEU", "chrF"]):
        pivot = source_metric_df.pivot(index="model_short", columns="split", values=metric)
        pivot = pivot.reindex([MODEL_SHORT_NAMES[slug] for slug in MODEL_ORDER])
        pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
        ax.set_title(f"{metric} by Test Source")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Source")
    fig.suptitle("Performance by Dataset Source")
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "06_metrics_by_source.png",
        "Figure 6. BLEU and chrF separated by whether each held-out sentence pair came from the Bible corpus or the JHW corpus.",
        captions,
    )
    plt.close(fig)

    # Figure 7: prediction length ratio.
    length_df = long_df.copy()
    length_df["length_ratio"] = length_df["prediction_token_count"] / length_df["reference_token_count"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = [length_df.loc[length_df["model_slug"] == slug, "length_ratio"].dropna().clip(0, 3).values for slug in MODEL_ORDER]
    box = ax.boxplot(data, labels=[MODEL_SHORT_NAMES[slug] for slug in MODEL_ORDER], patch_artist=True, showfliers=False)
    for patch, slug in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[slug])
        patch.set_alpha(0.75)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_title("Prediction Length Relative to Reference")
    ax.set_xlabel("Model")
    ax.set_ylabel("Predicted tokens / reference tokens")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_captioned_figure(
        fig,
        dirs,
        "07_prediction_length_ratio.png",
        "Figure 7. Distribution of each model's predicted length divided by the reference length. The dashed line marks equal length.",
        captions,
    )
    plt.close(fig)

    # Qualitative example table.
    example_candidates = test_df[(test_df["english_token_count"] >= 6) & (test_df["english_token_count"] <= 28)].copy()
    selected_ids = []
    rng = np.random.default_rng(RANDOM_SEED)
    for source in ["Bible", "JHW"]:
        source_ids = example_candidates.loc[example_candidates["source"] == source, "row_id"].to_numpy()
        if len(source_ids):
            chosen = rng.choice(source_ids, size=min(4, len(source_ids)), replace=False)
            selected_ids.extend(int(x) for x in chosen)
    if not selected_ids:
        selected_ids = test_df["row_id"].head(8).tolist()

    qualitative_df = wide_df[wide_df["row_id"].isin(selected_ids)].copy().sort_values(["source", "row_id"])
    qualitative_df.to_csv(dirs["tables"] / "qualitative_translation_examples.csv", index=False)
    qualitative_df.to_html(dirs["tables"] / "qualitative_translation_examples.html", index=False, escape=False)

    table_cols = ["source", "English", "Reference Igbo", "RNN", "Transformer", "OPUS pretrained", "OPUS fine-tuned"]
    visible_table = qualitative_df[table_cols].head(6).copy()
    for col in table_cols:
        visible_table[col] = visible_table[col].map(lambda text: wrap_text(text, width=28, max_chars=150))

    fig, ax = plt.subplots(figsize=(18, max(5, 1.45 * len(visible_table) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=visible_table.values,
        colLabels=visible_table.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 2.25)
    ax.set_title("Qualitative Translation Examples", pad=16)
    save_captioned_figure(
        fig,
        dirs,
        "08_qualitative_examples_table.png",
        "Figure 8. A compact qualitative sample showing the English input, reference Igbo sentence, and the four model outputs. The full table is saved as qualitative_translation_examples.csv and .html.",
        captions,
    )
    plt.close(fig)

    captions_df = pd.DataFrame(captions)
    captions_df.to_csv(dirs["tables"] / "figure_captions.csv", index=False)

    readme_lines = ["# Figure Captions", ""]
    for row in captions:
        readme_lines.append(f"## {row['figure']}")
        readme_lines.append(row["caption"])
        readme_lines.append("")
    (dirs["figures"] / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def write_run_manifest(
    project_dir: Path,
    data_dir: Path,
    dirs: dict[str, Path],
    test_df: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    rows = []
    for model_slug in MODEL_ORDER:
        rows.append(
            {
                "model_slug": model_slug,
                "model": MODEL_DISPLAY_NAMES[model_slug],
                "prediction_rows": len(prediction_frames[model_slug]),
                "prediction_file": str(prediction_cache_path(dirs, model_slug)),
            }
        )

    manifest = {
        "project_dir": str(project_dir),
        "data_dir": str(data_dir),
        "output_dir": str(dirs["root"]),
        "evaluation_limit": EVALUATION_LIMIT,
        "test_rows": len(test_df),
        "metric_normalization": "lowercase, normalize curly quotes, tokenize with regex \\w+|[^\\w\\s], join tokens with spaces",
        "sacrebleu_reference_shape": "[references] for one reference stream",
        "use_existing_opus_previews": USE_EXISTING_OPUS_PREVIEWS,
        "opus_note": (
            "OPUS predictions are loaded from existing full-test preview CSVs when available; "
            "if missing or FINAL_EVAL_USE_OPUS_PREVIEWS=0, the script regenerates them with num_beams=1."
        ),
    }
    (dirs["tables"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(dirs["tables"] / "prediction_file_manifest.csv", index=False)


def main() -> None:
    configure_stdout()
    start_time = time.time()
    project_dir = resolve_project_dir()
    data_dir = find_data_dir(project_dir)
    if data_dir is None:
        raise FileNotFoundError(f"Could not find data directory under {project_dir}.")
    dirs = make_output_dirs(project_dir)

    print("Project directory:", project_dir)
    print("Data directory:", data_dir)
    print("Output directory:", dirs["root"])
    print("Evaluation limit:", "full test set" if EVALUATION_LIMIT is None else EVALUATION_LIMIT)

    test_df = load_test_dataframe(data_dir)
    if EVALUATION_LIMIT is not None:
        test_df = test_df.head(EVALUATION_LIMIT).copy()
    test_df.to_csv(dirs["tables"] / "evaluation_dataset.csv", index=False)

    prediction_frames: dict[str, pd.DataFrame] = {}

    opus_output_dir = project_dir / "OPUS_MT Outputs"
    prediction_frames["opus_mt_pretrained"] = load_or_align_opus_preview(
        dirs,
        "opus_mt_pretrained",
        opus_output_dir / "opus_mt_direct_pretrained_external_jw_bible_preview.csv",
        test_df,
    )
    if prediction_frames["opus_mt_pretrained"] is None:
        prediction_frames["opus_mt_pretrained"] = generate_opus_predictions(
            project_dir,
            dirs,
            test_df,
            "opus_mt_pretrained",
            "1_direct_pretrained_model",
        )

    prediction_frames["opus_mt_fine_tuned"] = load_or_align_opus_preview(
        dirs,
        "opus_mt_fine_tuned",
        opus_output_dir / "opus_mt_fine_tuned_combined_test_preview.csv",
        test_df,
    )
    if prediction_frames["opus_mt_fine_tuned"] is None:
        prediction_frames["opus_mt_fine_tuned"] = generate_opus_predictions(
            project_dir,
            dirs,
            test_df,
            "opus_mt_fine_tuned",
            "2_fine_tuned_model",
        )

    prediction_frames["rnn_lstm_scratch"] = generate_rnn_predictions(project_dir, dirs, test_df)
    prediction_frames["transformer_scratch"] = generate_transformer_predictions(project_dir, dirs, test_df)

    # Reorder after generation so every saved table follows the report order.
    prediction_frames = {slug: prediction_frames[slug] for slug in MODEL_ORDER}

    overall_df, by_source_df = compute_metrics(prediction_frames, dirs)
    long_df, wide_df = build_prediction_tables(prediction_frames, dirs)
    create_visualizations(project_dir, data_dir, dirs, test_df, overall_df, by_source_df, long_df, wide_df)
    write_run_manifest(project_dir, data_dir, dirs, test_df, prediction_frames)

    print("\nOverall metrics:")
    print(overall_df[["model", "rows", "Exact match", "BLEU", "chrF"]].to_string(index=False))
    print("\nSaved outputs:")
    print("Predictions:", dirs["predictions"])
    print("Tables:", dirs["tables"])
    print("Figures:", dirs["figures"])
    print(f"Elapsed minutes: {(time.time() - start_time) / 60:.2f}")


if __name__ == "__main__":
    main()
