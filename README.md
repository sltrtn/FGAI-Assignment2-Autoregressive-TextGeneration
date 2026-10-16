# Autoregressive Next-Word Text Generation: Simple RNN vs. Bidirectional RNN vs. LSTM vs. GRU

**Course:** Foundations of Generative AI (ICT4403) — Assignment 2  
**Student:** Madhavan Komandur (Registration No. 230953298)

## 1. Problem statement and objective
Develop and compare four autoregressive (next-word) text-generation models — Simple RNN,
Bidirectional RNN, LSTM, and GRU — on a selected corpus. All four models are trained
under comparable experimental conditions. The goal is to quantify predictive performance
(loss, perplexity, accuracy, training time) and qualitatively assess generated text
(coherence, diversity, contextual consistency, repetition), including the effect of
sampling temperature.

## 2. Dataset description and source
- **Dataset:** The War of the Worlds, by H. G. Wells (public domain).
- **Source:** https://www.gutenberg.org/files/36/36-0.txt (Project Gutenberg eBook #36)
- After removing Gutenberg boilerplate and tokenising (lowercase words + punctuation), the
  corpus contains ~70k tokens with a vocabulary of ~3.1k words (minimum frequency 2).
- Split: 80/10/10 train/validation/test at the token level.
- Context length \(L=20\): input \(x=[w_t,\dots,w_{t+19}]\), target \(y=w_{t+20}\)
  (many-to-one; this prevents future-token leakage in the Bidirectional model).

## 3. Models implemented
All models share: embedding dim 128, hidden dim 256, 1 layer, dropout 0.2, Adam (lr = 1e-3),
gradient clipping 1.0, CrossEntropyLoss. BiRNN splits the hidden size across directions.
- **Simple RNN** (`nn.RNN`)
- **Bidirectional RNN** (`nn.RNN`, bidirectional=True)
- **LSTM** (`nn.LSTM`)
- **GRU** (`nn.GRU`)

## 4. Execution steps and required dependencies
```bash
pip install -r requirements.txt
python experiment.py            # trains all four models, writes results/...
```
Open `text_generation.ipynb` for the annotated, step-by-step version. Reproducibility seed: 42.

## 5. Results obtained

| Model | Params | Val PPL | Test PPL | Test Acc | Train time (s) | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| SimpleRNN | 1,293,086 | 108.55 | 96.86 | 0.2036 | 91.1 | 6 |
| BiRNN | 1,260,318 | 110.90 | 96.15 | 0.2155 | 121.1 | 8 |
| LSTM | 1,589,534 | 99.75 | **86.72** | **0.2252** | 211.9 | 7 |
| GRU | 1,490,718 | 100.08 | 87.13 | 0.2209 | 165.1 | 6 |

Perplexity \(PPL = \exp(\mathcal{L})\), where \(\mathcal{L}\) is mean token cross-entropy.

## 6. Model comparison and evaluation
- **LSTM** achieves the lowest validation/test perplexity and best accuracy; gating helps capture
  longer dependencies (4–10% lower test PPL than SimpleRNN).
- **GRU** is close to LSTM while training ~22% faster, and is the most parameter-efficient
  gated model.
- **BiRNN** did not outperform the unidirectional gated models here because predictions are
  made causally from the left context; its bidirectional view provided only marginal benefit.
- **Simple RNN** trains fastest but underfits longer dependencies (highest loss).

## 7. Sampling-temperature analysis
Text is sampled from identical seed sequences at \(T \in \{0.5, 1.0, 1.5\}\) via
\(P(w) \propto \exp(z_w / T)\). See `results/generated_samples.txt`.
- \(T=0.5\): more repetitive, conservative, higher `<unk>` repetition.
- \(T=1.0\): balanced coherence and diversity; best readability.
- \(T=1.5\): most diverse but drifts into incoherence/novel words.
Outputs are saved per model and seed for review.

## 8. Conclusion
**LSTM** is the best-performing model on held-out perplexity (test PPL 86.72, accuracy 0.2252).
**GRU** is effectively tied (87.13 PPL) and trains substantially faster, so it is the
recommended practical choice for iterative autoregressive generation on this corpus.
Simple RNN and Bidirectional RNN are viable but less accurate.

## Repository layout
```
├── text_generation.ipynb      # annotated notebook
├── experiment.py              # reproducible training script
├── data/war_of_the_worlds.txt # dataset (Gutenberg #36)
├── results/                   # metrics.json, plots, samples, checkpoints
├── requirements.txt
├── report/assignment2.tex     # LaTeX report
└── README.md
```
