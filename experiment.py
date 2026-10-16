"""Autoregressive (next-word) text generation on War of the Worlds."""
import os, re, json, time, math, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "war_of_the_worlds.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 20
MIN_FREQ = 2
EMB_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = int(os.environ.get("EPOCHS", "10"))
PATIENCE = 3
MAX_NEW_TOKENS = 40
TEMPERATURES = [0.5, 1.0, 1.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------- data loading
def load_corpus(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    m_start = re.search(r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK .*? \*\*\*", raw)
    m_end = re.search(r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK .*? \*\*\*", raw)
    body = raw[m_start.end():m_end.start()] if (m_start and m_end) else raw
    # normalise curly quotes etc.
    body = body.replace("\u2019", "'").replace("\u2018", "'")
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", body.lower())
    return tokens

def build_vocab(train_tokens, min_freq=MIN_FREQ):
    counts = Counter(train_tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, cnt in counts.items():
        if cnt >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    idx2word = {i: w for w, i in vocab.items()}
    return vocab, idx2word

TOKENS = load_corpus(DATA_PATH)
n = len(TOKENS)
train_end = int(0.8 * n)
val_end = int(0.9 * n)
train_tokens = TOKENS[:train_end]
val_tokens = TOKENS[train_end:val_end]
test_tokens = TOKENS[val_end:]

VOCAB, IDX2WORD = build_vocab(train_tokens)
V = len(VOCAB)
UNK = VOCAB["<unk>"]

def encode(tok_list):
    return [VOCAB.get(w, UNK) for w in tok_list]

train_ids = encode(train_tokens)
val_ids = encode(val_tokens)
test_ids = encode(test_tokens)

class WordDataset(Dataset):
    # Many-to-one: context of SEQ_LEN tokens -> single next token.
    # This avoids target leakage for the Bidirectional model.
    def __init__(self, ids, seq_len=SEQ_LEN):
        self.x = []
        self.y = []
        for i in range(len(ids) - seq_len):
            self.x.append(ids[i:i + seq_len])
            self.y.append(ids[i + seq_len])
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

train_ds = WordDataset(train_ids)
val_ds = WordDataset(val_ids)
test_ds = WordDataset(test_ids)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------------------------------------- models
def make_rnn(rnn_type, bidirectional=False):
    hidden_per_dir = HIDDEN_DIM // (2 if bidirectional else 1)
    if rnn_type == "RNN":
        cell = nn.RNN
    elif rnn_type == "LSTM":
        cell = nn.LSTM
    elif rnn_type == "GRU":
        cell = nn.GRU
    else:
        raise ValueError(rnn_type)
    embedding = nn.Embedding(V, EMB_DIM, padding_idx=0)
    rnn = cell(
        EMB_DIM, hidden_per_dir, NUM_LAYERS,
        batch_first=True, dropout=(DROPOUT if NUM_LAYERS > 1 else 0.0),
        bidirectional=bidirectional,
    )
    fc = nn.Linear(hidden_per_dir * (2 if bidirectional else 1), V)
    return ModelWrapper(embedding, rnn, fc, rnn_type, bidirectional)

class ModelWrapper(nn.Module):
    def __init__(self, embedding, rnn, fc, rnn_type, bidirectional):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.fc = fc
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        # Predict from the final time step only (avoids target leakage for BiRNN).
        logits = self.fc(out[:, -1, :])
        return logits

# Four comparable configurations
MODEL_CONFIGS = [
    ("SimpleRNN", "RNN", False),
    ("BiRNN", "RNN", True),
    ("LSTM", "LSTM", False),
    ("GRU", "GRU", False),
]

# ------------------------------------------------------------- training utils
def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl, acc

def train_model(name, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [],
               "train_acc": [], "val_acc": [], "epoch_time": []}
    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        et0 = time.perf_counter()
        tr_loss, tr_ppl, tr_acc = run_epoch(model, train_loader, optimizer)
        va_loss, va_ppl, va_acc = run_epoch(model, val_loader, None)
        et = time.perf_counter() - et0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_ppl"].append(tr_ppl)
        history["val_ppl"].append(va_ppl)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["epoch_time"].append(et)
        print(f"[{name}] epoch {epoch:02d}/{EPOCHS} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_ppl={va_ppl:.2f} val_acc={va_acc:.4f} time={et:.1f}s", flush=True)
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[{name}] early stopping at epoch {epoch}", flush=True)
                break
    total_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, CHECKPOINT_DIR / f"{name}.pt")
    # test
    te_loss, te_ppl, te_acc = run_epoch(model, test_loader, None)
    summary = {
        "model": name,
        "params": int(sum(p.numel() for p in model.parameters())),
        "best_val_loss": float(best_val),
        "best_val_ppl": float(math.exp(min(best_val, 20))),
        "test_loss": float(te_loss),
        "test_ppl": float(te_ppl),
        "test_acc": float(te_acc),
        "total_train_time_sec": float(total_time),
        "epochs_run": len(history["train_loss"]),
    }
    return model, history, summary

# ------------------------------------------------------------ generation
def generate_text(model, seed_words, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0):
    model.eval()
    ids = [VOCAB.get(w, UNK) for w in seed_words]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-SEQ_LEN:]
            x = torch.tensor([context], dtype=torch.long, device=DEVICE)
            logits = model(x)
            next_logits = logits[0] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
    words = [IDX2WORD.get(i, "<unk>") for i in ids]
    return " ".join(words)

def repetition_stats(text):
    toks = text.split()
    if not toks:
        return {}
    uniq = len(set(toks))
    bigrams = [(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]
    uniq_bi = len(set(bigrams))
    return {
        "n_tokens": len(toks),
        "unique_token_ratio": uniq / len(toks),
        "unique_bigram_ratio": (uniq_bi / len(bigrams)) if bigrams else 1.0,
    }

# ---------------------------------------------------------------- main
def main():
    print("Device:", DEVICE)
    print("Corpus tokens:", n, "train/val/test:", len(train_ids), len(val_ids), len(test_ids))
    print("Vocab size:", V)
    print("Train examples:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

    seeds = [
        ["the", "martians", "had", "landed", "on"],
        ["it", "was", "a", "dark", "and", "stormy"],
        ["the", "cylinder", "opened", "and", "a", "strange"],
    ]
    seeds = [[w for w in s if w in VOCAB or True] for s in seeds]

    results = {"config": {
        "dataset": "The War of the Worlds (Project Gutenberg #36)",
        "seq_len": SEQ_LEN, "min_freq": MIN_FREQ, "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM, "layers": NUM_LAYERS, "dropout": DROPOUT,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
        "train_tokens": len(train_ids), "val_tokens": len(val_ids),
        "test_tokens": len(test_ids), "vocab_size": V,
    }, "models": {}}

    all_models = {}
    all_histories = {}
    summaries = []
    for name, rnn_type, bidir in MODEL_CONFIGS:
        print("\n" + "=" * 20, "Training", name, "=" * 20)
        model = make_rnn(rnn_type, bidir).to(DEVICE)
        model, history, summary = train_model(name, model)
        all_models[name] = model
        all_histories[name] = history
        summaries.append(summary)
        results["models"][name] = {"summary": summary, "history": history}

        # generation
        gen_block = {}
        for seed in seeds:
            seed_text = " ".join(seed)
            for T in TEMPERATURES:
                text = generate_text(model, seed, temperature=T)
                stats = repetition_stats(text)
                gen_block.setdefault(seed_text, {})[str(T)] = {"text": text, "stats": stats}
        results["models"][name]["generation"] = gen_block

    # save metrics json
    with (RESULTS_DIR / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)

    # save generated samples txt
    with (RESULTS_DIR / "generated_samples.txt").open("w") as f:
        for name in results["models"]:
            f.write("=" * 80 + "\n" + name + "\n" + "=" * 80 + "\n")
            for seed_text, tmap in results["models"][name]["generation"].items():
                f.write("\nSEED: " + seed_text + "\n")
                for T, info in tmap.items():
                    f.write(f"  Temperature {T}: {info['text']}\n")
            f.write("\n")

    # plots
    # 1. loss curves
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (name, _, _) in zip(axes.flat, MODEL_CONFIGS):
        h = all_histories[name]
        ax.plot(h["train_loss"], label="train loss")
        ax.plot(h["val_loss"], label="val loss")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Training / validation loss")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)

    # 2. test metrics bars
    names = [s["model"] for s in summaries]
    test_ppl = [s["test_ppl"] for s in summaries]
    test_acc = [s["test_acc"] for s in summaries]
    times = [s["total_train_time_sec"] for s in summaries]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(names, test_ppl); ax[0].set_title("Test perplexity (lower better)")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(names, test_acc); ax[1].set_title("Test accuracy (higher better)")
    ax[1].tick_params(axis="x", rotation=20)
    ax[2].bar(names, times); ax[2].set_title("Training time (s)")
    ax[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_metrics.png", dpi=150)
    plt.close(fig)

    # summary table print
    print("\nSUMMARY")
    print(f"{'Model':12}{'params':>10}{'val_ppl':>10}{'test_ppl':>10}{'test_acc':>10}{'time_s':>10}")
    for s in summaries:
        print(f"{s['model']:12}{s['params']:10d}{s['best_val_ppl']:10.2f}"
              f"{s['test_ppl']:10.2f}{s['test_acc']:10.4f}{s['total_train_time_sec']:10.1f}")

    print("\nSaved metrics:", RESULTS_DIR / "metrics.json")

if __name__ == "__main__":
    main()