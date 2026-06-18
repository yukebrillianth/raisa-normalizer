---
jupyter:
  kernelspec:
    display_name: .venv
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#61fb834f .cell .code}
``` python
%pip install torch transformers peft datasets pandas tqdm matplotlib seaborn scikit-learn sentence-transformers psycopg2-binary -q
```
:::

::: {#3afc73fd .cell .code}
``` python
from unsloth import FastLanguageModel  # must be first — patches torch/transformers at import time

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import gc
import os
from pathlib import Path
from tqdm.auto import tqdm
import psycopg2

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Clear GPU cache and GPU info
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") 
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
else:
    print("CUDA not available")
```
:::

::: {#6e989a00 .cell .code}
``` python
# --- CONFIGURATION ---
BASE_DIR = Path("/home/teaching-factory/pipeline")
MODEL_DIR = BASE_DIR / "experiments" / "training_r16_2e4" / "models" / "lora_adapters"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_TEST = BASE_DIR / "data" / "data_test_retrieval_2.csv"

BASE_MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128
MAX_NEW_TOKENS = 128

TOP_K = 3                     # jumlah kandidat yang diambil dari pgvector
SIMILARITY_THRESHOLD = 0.75   # threshold cosine similarity top-1 untuk retrieve answer
RERANK_KEYWORD_WEIGHT = 0.2   # bobot keyword overlap pada reranking hybrid
MATCH_THRESHOLD = 0.90        # threshold semantic match untuk evaluasi (prediksi vs ground truth)

FALLBACK_ANSWER = "Maaf, saya belum dapat menemukan jawaban yang sesuai untuk pertanyaan Anda."

print(f"Device: {DEVICE}")
print(f"Threshold retrieval: {SIMILARITY_THRESHOLD}")
```
:::

::: {#b6a63f20 .cell .code}
``` python
conn = psycopg2.connect(
    dbname="d_thesis",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)

# Avoid transaction-aborted state for read-only retrieval queries
conn.autocommit = True
cur = conn.cursor()

print("Database connected")
```
:::

::: {#397a66c3 .cell .code}
``` python
## LOAD MODEL

print("=" * 60)
print("LOADING FINE-TUNED MODEL")
print("=" * 60)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model 8-bit with Unsloth...")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

print("Loading Model....")
base_model, _ = FastLanguageModel.from_pretrained(
    BASE_MODEL_NAME,  # Load base model
    max_seq_length=MAX_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=False,
    load_in_8bit=True,
    quantization_config=quantization_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    )

print("Loading PEFT model...")
adapter_config_path = MODEL_DIR / "adapter_config.json"
if adapter_config_path.exists():
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_DIR
    )
else:
    print(f"[WARNING] adapter_config.json not found at: {adapter_config_path}")
    print("Using base model without LoRA adapter.")
    model = base_model

model.eval()

print(f"Model loaded successfully on: {next(model.parameters()).device}")

# Clear memory after loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print("Fine-tuned model loaded successfully.")
```
:::

::: {#4554759b .cell .code}
``` python
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
embedding_model.eval()

print("Embedding model loaded successfully.")
```
:::

::: {#62a644b5 .cell .code}
``` python
print("Loading test dataset...")
df = pd.read_csv(DATA_TEST)

required_cols = {"input_query", "gt_query", "gt_answer"}
assert required_cols.issubset(df.columns), f"Kolom kurang: {required_cols - set(df.columns)}"

print(f"Test dataset loaded: {len(df)} samples")
df.head()
```
:::

::: {#371dbb4f .cell .code}
``` python
## Normalization Functions

def format_prompt(informal_text):
    return f"""### Instruction:
Anda adalah sistem layanan informasi Kampus Institut Teknologi Sepuluh Nopember (ITS). Ubah input kalimat Bahasa Indonesia tidak baku berikut menjadi Bahasa Indonesia baku dan formal.
Jangan menambahkan informasi baru.
Gunakan kalimat singkat, jelas, baku, dan langsung ke inti.

### Input:
{informal_text}

### Response:
"""


def clean_output(text):
    """
    Clean the output by removing special tokens and extra whitespace
    """
    # Remove <think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove quotes
    cleaned = cleaned.strip().strip('"').strip("'")
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def normalize_text(informal_text, max_length=None):
    if max_length is None:
        max_length = MAX_LENGTH
    
    prompt = format_prompt(informal_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            use_cache=True
        )
    
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )



    del inputs, outputs
    torch.cuda.empty_cache()

    return clean_output(decoded)
```
:::

::: {#7a8552b3 .cell .code}
``` python
normalized_queries = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing"):
    informal_text = row["input_query"]

    try:
        normalized = normalize_text(informal_text)
    except torch.cuda.OutOfMemoryError:
        print(f"\n[WARNING] OOM at sample {idx+1}. Clearing cache and retrying...")
        torch.cuda.empty_cache()
        gc.collect()
        try:
            normalized = normalize_text(informal_text, max_length=256)
        except Exception as e:
            print(f"Failed again at sample {idx+1}: {e}")
            normalized = informal_text  # fallback: pakai input asli

    normalized_queries.append(normalized)

    if (idx + 1) % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

df["normalized_query"] = normalized_queries
print("Normalisasi selesai.")
```
:::

::: {#27599810 .cell .code}
``` python
print("Encoding normalized queries...")
query_vectors = embedding_model.encode(
    df["normalized_query"].astype(str).tolist(),
    batch_size=32,
    show_progress_bar=True
)
df["query_vector"] = [v.tolist() for v in query_vectors]
print(f"Encoded {len(query_vectors)} queries, dim = {len(query_vectors[0])}")
```
:::

::: {#0f9b097d .cell .code}
``` python
def keyword_overlap(query, candidate):
    q_words = set(str(query).lower().split())
    c_words = set(str(candidate).lower().split())
    return len(q_words & c_words) / (len(q_words) + 1e-6)


retrieval_records = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieval"):
    normalized_query = row["normalized_query"]
    vector_literal = "[" + ",".join(map(str, row["query_vector"])) + "]"

    try:
        cur.execute("""
            SELECT question, answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_ground_truth
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_literal, vector_literal, TOP_K))
        results = cur.fetchall()
    except Exception as e:
        conn.rollback()
        results = []
        print(f"DB error at row {idx}: {type(e).__name__}: {e}")

    candidates = [
        {"question": q, "answer": a, "similarity": float(s)}
        for (q, a, s) in results
    ]

    # --- RERANKING HYBRID ---
    for c in candidates:
        c["keyword_score"] = keyword_overlap(normalized_query, c["question"])
        c["rerank_score"] = c["similarity"] + RERANK_KEYWORD_WEIGHT * c["keyword_score"]

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    # --- TOP-1 + THRESHOLD FILTER ---
    if reranked and reranked[0]["similarity"] >= SIMILARITY_THRESHOLD:
        answered = 1
        retrieval_result = reranked[0]["answer"]
    else:
        answered = 0
        retrieval_result = FALLBACK_ANSWER

    retrieval_records.append({
        "candidates": candidates,   # urutan asli dari pgvector
        "reranked": reranked,       # urutan setelah reranking
        "answered": answered,
        "retrieval_result": retrieval_result,
    })

df["retrieval_result"] = [r["retrieval_result"] for r in retrieval_records]
df["answered"] = [r["answered"] for r in retrieval_records]
print("Retrieval selesai.")
print(f"Answered (>= threshold): {df['answered'].sum()}/{len(df)}")
```
:::

::: {#c60dacc5 .cell .code}
``` python
_emb_cache = {}

def get_embedding(text):
    key = str(text).strip()
    if key not in _emb_cache:
        _emb_cache[key] = embedding_model.encode(key, convert_to_tensor=True)
    return _emb_cache[key]


def text_norm(text):
    """Normalisasi teks untuk exact match: lowercase, hapus tanda baca, rapikan spasi."""
    t = str(text).lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def semantic_sim(text_a, text_b):
    if not str(text_a).strip() or not str(text_b).strip():
        return 0.0
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    return float(util.cos_sim(emb_a, emb_b).item())


def is_match(pred, gold, threshold=MATCH_THRESHOLD):
    """Returns (is_match: bool, similarity: float). Exact match dulu, lalu semantic."""
    if pred is None or gold is None:
        return False, 0.0
    if text_norm(pred) == text_norm(gold) and text_norm(pred) != "":
        return True, 1.0
    sim = semantic_sim(pred, gold)
    return sim >= threshold, sim

print("Metric functions ready")
```
:::

::: {#7cba77aa .cell .code}
``` python
eval_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    rec = retrieval_records[idx]
    reranked = rec["reranked"]
    gt_query = row["gt_query"]
    gt_answer = row["gt_answer"]

    # --- Cari rank ground-truth question pada hasil reranking ---
    gt_rank = 0
    for i, cand in enumerate(reranked, start=1):
        matched, _ = is_match(cand["question"], gt_query)
        if matched:
            gt_rank = i
            break

    hit_at_1 = int(gt_rank == 1)
    hit_at_2 = int(0 < gt_rank <= 2)
    hit_at_3 = int(0 < gt_rank <= 3)
    rr = 1.0 / gt_rank if gt_rank > 0 else 0.0
    ndcg_at_3 = 1.0 / np.log2(gt_rank + 1) if 0 < gt_rank <= 3 else 0.0

    # --- Info top-1 setelah reranking ---
    if reranked:
        top1 = reranked[0]
        top1_question = top1["question"]
        top1_similarity = top1["similarity"]
        top1_rerank_score = top1["rerank_score"]
        # benar/tidaknya jawaban top-1 (independen dari threshold, untuk analisis threshold)
        top1_correct_bool, top1_answer_sim = is_match(top1["answer"], gt_answer)
        top1_answer_correct = int(top1_correct_bool)
    else:
        top1_question = ""
        top1_similarity = 0.0
        top1_rerank_score = 0.0
        top1_answer_correct = 0
        top1_answer_sim = 0.0

    # --- Hasil akhir sistem (dengan threshold) ---
    answered = rec["answered"]
    answer_correct = top1_answer_correct if answered else 0
    e2e_correct = int(answered == 1 and answer_correct == 1)

    # --- Kualitas normalisasi: similarity input vs normalized ---
    query_semantic_sim = semantic_sim(row["input_query"], row["normalized_query"])

    eval_rows.append({
        "input_query": row["input_query"],
        "normalized_query": row["normalized_query"],
        "gt_query": gt_query,
        "gt_answer": gt_answer,
        "top1_question": top1_question,
        "top1_similarity": round(top1_similarity, 4),
        "top1_rerank_score": round(top1_rerank_score, 4),
        "retrieval_result": rec["retrieval_result"],
        "answered": answered,
        "gt_rank": gt_rank,
        "hit@1": hit_at_1,
        "hit@2": hit_at_2,
        "hit@3": hit_at_3,
        "rr": round(rr, 4),
        "ndcg@3": round(ndcg_at_3, 4),
        "top1_answer_correct": top1_answer_correct,
        "answer_similarity": round(top1_answer_sim, 4),
        "answer_correct": answer_correct,
        "e2e_correct": e2e_correct,
        "query_semantic_similarity": round(query_semantic_sim, 4),
    })

results_df = pd.DataFrame(eval_rows)

print(f"Evaluation selesai: {len(results_df)} samples")
print("\nDETAILED RESULTS PER TEST CASE:")
display(results_df[[
    "input_query", "normalized_query", "retrieval_result",
    "top1_similarity", "answered", "gt_rank",
    "hit@1", "hit@3", "rr", "ndcg@3", "answer_correct", "e2e_correct"
]].head(50))
```
:::

::: {#5e2c8d8f .cell .code}
``` python
n = len(results_df)
n_answered = int(results_df["answered"].sum())
n_fallback = n - n_answered
n_answer_correct = int(results_df["answer_correct"].sum())
n_e2e_correct = int(results_df["e2e_correct"].sum())

hit1 = int(results_df["hit@1"].sum())
hit2 = int(results_df["hit@2"].sum())
hit3 = int(results_df["hit@3"].sum())
mrr = results_df["rr"].mean()
ndcg3 = results_df["ndcg@3"].mean()

answer_rate = n_answered / n
answer_accuracy = n_answer_correct / n_answered if n_answered > 0 else 0.0
e2e_accuracy = n_e2e_correct / n
# semua query di dataset memiliki ground truth answer -> fallback = false rejection
false_rejection_rate = n_fallback / n

print("=" * 70)
print("OVERALL RETRIEVAL METRICS")
print("=" * 70)
print(f"Jumlah sampel                 : {n}")
print(f"Threshold similarity          : {SIMILARITY_THRESHOLD}")
print("-" * 70)
print(f"Hit@1                         : {hit1}/{n} ({hit1/n:.4f})")
print(f"Hit@2                         : {hit2}/{n} ({hit2/n:.4f})")
print(f"Hit@3                         : {hit3}/{n} ({hit3/n:.4f})")
print(f"MRR                           : {mrr:.4f}")
print(f"nDCG@3                        : {ndcg3:.4f}")
print("-" * 70)
print(f"Answer Rate (>= threshold)    : {n_answered}/{n} ({answer_rate:.4f})")
print(f"Fallback / 'Maaf...'          : {n_fallback}/{n} ({false_rejection_rate:.4f})")
print(f"Answer Accuracy (yg dijawab)  : {n_answer_correct}/{n_answered} ({answer_accuracy:.4f})")
print(f"End-to-End Accuracy           : {n_e2e_correct}/{n} ({e2e_accuracy:.4f})")
print("-" * 70)
print(f"Avg Top-1 Similarity          : {results_df['top1_similarity'].mean():.4f}")
print(f"Avg Answer Similarity         : {results_df['answer_similarity'].mean():.4f}")
print(f"Avg Query Semantic Similarity : {results_df['query_semantic_similarity'].mean():.4f}")
print("=" * 70)

summary_stats = pd.DataFrame({
    "Metric": ["Hit@1", "Hit@2", "Hit@3", "MRR", "nDCG@3",
               "Answer Rate", "Answer Accuracy", "End-to-End Accuracy",
               "Avg Top-1 Similarity", "Avg Query Semantic Similarity"],
    "Value": [hit1/n, hit2/n, hit3/n, mrr, ndcg3,
              answer_rate, answer_accuracy, e2e_accuracy,
              results_df["top1_similarity"].mean(),
              results_df["query_semantic_similarity"].mean()],
    "Count": [f"{hit1}/{n}", f"{hit2}/{n}", f"{hit3}/{n}", "-", "-",
              f"{n_answered}/{n}", f"{n_answer_correct}/{n_answered}", f"{n_e2e_correct}/{n}",
              "-", "-"]
})

print("\nSummary Table:")
display(summary_stats)
```
:::

::: {#d0db025c .cell .code}
``` python
# Save detailed results to CSV
output_csv = RESULTS_DIR / "retrieval_evaluation_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nDetailed results saved to: {output_csv}")
```
:::

::: {#929e9da3 .cell .code}
``` python
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Hit@k bar chart
ax1 = axes[0, 0]
hit_labels = ["Hit@1", "Hit@2", "Hit@3"]
hit_values = [hit1/n, hit2/n, hit3/n]
bars = ax1.bar(hit_labels, hit_values, color=["#3498db", "#2ecc71", "#f39c12"], alpha=0.8, edgecolor="black")
for bar, v, c in zip(bars, hit_values, [hit1, hit2, hit3]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.3f}\n({c}/{n})", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 1.1)
ax1.set_ylabel("Rate", fontsize=12, fontweight="bold")
ax1.set_title("Hit@k", fontsize=14, fontweight="bold")

# 2. Distribusi top-1 similarity + threshold line
ax2 = axes[0, 1]
ax2.hist(results_df["top1_similarity"], bins=30, color="#9b59b6", alpha=0.7, edgecolor="black")
ax2.axvline(SIMILARITY_THRESHOLD, color="red", linestyle="--", linewidth=2,
            label=f"Threshold = {SIMILARITY_THRESHOLD}")
ax2.set_xlabel("Top-1 Cosine Similarity", fontsize=12, fontweight="bold")
ax2.set_ylabel("Jumlah Query", fontsize=12, fontweight="bold")
ax2.set_title("Distribusi Top-1 Similarity", fontsize=14, fontweight="bold")
ax2.legend()

# 3. Overall metrics bar
ax3 = axes[1, 0]
metric_labels = ["MRR", "nDCG@3", "Answer\nRate", "Answer\nAccuracy", "E2E\nAccuracy"]
metric_values = [mrr, ndcg3, answer_rate, answer_accuracy, e2e_accuracy]
bars3 = ax3.bar(metric_labels, metric_values,
                color=["#3498db", "#1abc9c", "#f39c12", "#2ecc71", "#e74c3c"],
                alpha=0.8, edgecolor="black")
for bar, v in zip(bars3, metric_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.set_ylim(0, 1.1)
ax3.set_ylabel("Score", fontsize=12, fontweight="bold")
ax3.set_title("Overall Metrics", fontsize=14, fontweight="bold")

# 4. Breakdown hasil akhir sistem
ax4 = axes[1, 1]
n_wrong = n_answered - n_answer_correct
outcome_labels = ["Dijawab\nBenar", "Dijawab\nSalah", "Fallback\n('Maaf...')"]
outcome_values = [n_answer_correct, n_wrong, n_fallback]
bars4 = ax4.bar(outcome_labels, outcome_values,
                color=["#2ecc71", "#e74c3c", "#95a5a6"], alpha=0.8, edgecolor="black")
for bar, v in zip(bars4, outcome_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v}\n({v/n:.1%})", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax4.set_ylabel("Jumlah Query", fontsize=12, fontweight="bold")
ax4.set_title("Breakdown Hasil Akhir Sistem", fontsize=14, fontweight="bold")

plt.tight_layout()
plot_file = RESULTS_DIR / "retrieval_eval_plots.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()
print(f"Plot saved to: {plot_file}")
```
:::

::: {#87d18d11 .cell .code}
``` python
thresholds = np.arange(0.50, 0.96, 0.025)
sweep_answer_rate = []
sweep_e2e_acc = []
sweep_answer_acc = []

for t in thresholds:
    answered_t = results_df["top1_similarity"] >= t
    correct_t = answered_t & (results_df["top1_answer_correct"] == 1)
    sweep_answer_rate.append(answered_t.mean())
    sweep_e2e_acc.append(correct_t.mean())
    sweep_answer_acc.append(correct_t.sum() / answered_t.sum() if answered_t.sum() > 0 else 0.0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Threshold sweep
ax1 = axes[0]
ax1.plot(thresholds, sweep_answer_rate, "o-", label="Answer Rate", color="#f39c12")
ax1.plot(thresholds, sweep_answer_acc, "s-", label="Answer Accuracy (yg dijawab)", color="#2ecc71")
ax1.plot(thresholds, sweep_e2e_acc, "^-", label="End-to-End Accuracy", color="#e74c3c")
ax1.axvline(SIMILARITY_THRESHOLD, color="red", linestyle="--", alpha=0.7,
            label=f"Threshold terpilih = {SIMILARITY_THRESHOLD}")
ax1.set_xlabel("Similarity Threshold", fontsize=12, fontweight="bold")
ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
ax1.set_title("Trade-off Threshold vs Performa", fontsize=14, fontweight="bold")
ax1.legend(loc="lower left")
ax1.grid(alpha=0.3)

# 2. Boxplot similarity: top-1 benar vs salah
ax2 = axes[1]
plot_df = results_df.copy()
plot_df["Top-1 Relevan?"] = plot_df["hit@1"].map({1: "Relevan", 0: "Tidak Relevan"})
sns.boxplot(data=plot_df, x="Top-1 Relevan?", y="top1_similarity",
            hue="Top-1 Relevan?", palette={"Relevan": "#2ecc71", "Tidak Relevan": "#e74c3c"},
            ax=ax2, legend=False)
ax2.axhline(SIMILARITY_THRESHOLD, color="red", linestyle="--", alpha=0.7,
            label=f"Threshold = {SIMILARITY_THRESHOLD}")
ax2.set_ylabel("Top-1 Cosine Similarity", fontsize=12, fontweight="bold")
ax2.set_title("Separabilitas Similarity: Relevan vs Tidak", fontsize=14, fontweight="bold")
ax2.legend()

plt.tight_layout()
threshold_plot_file = RESULTS_DIR / "threshold_analysis.png"
plt.savefig(threshold_plot_file, dpi=300, bbox_inches="tight")
plt.show()
print(f"Plot saved to: {threshold_plot_file}")
```
:::

::: {#5460e346 .cell .code}
``` python
failed_df = results_df[results_df["e2e_correct"] == 0]
print(f"Jumlah kasus gagal: {len(failed_df)}/{n}")
print(f"  - Fallback (di bawah threshold) : {(failed_df['answered'] == 0).sum()}")
print(f"  - Dijawab tapi salah            : {(failed_df['answered'] == 1).sum()}")

display(failed_df[[
    "input_query", "normalized_query", "gt_query", "top1_question",
    "top1_similarity", "answered", "gt_rank", "retrieval_result"
]])
```
:::
