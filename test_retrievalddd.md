```python
%pip install torch transformers peft datasets pandas tqdm matplotlib seaborn scikit-learn sentence-transformers psycopg2-binary -q
```

```python
import os
os.environ["BNB_CUDA_VERSION"] = "128"
```

```python
from unsloth import FastLanguageModel

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
from sentence_transformers import util
from rank_bm25 import BM25Okapi

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Clear GPU cache and GPU info
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    print("CUDA not available")
```

```text
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
```

```text
/home/teaching-factory/pipeline/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

```text
🦥 Unsloth Zoo will now patch everything to make training faster!
Total GPU Memory: 33.7 GB
GPU Memory Used: 0.01 GB
```

```python
# --- CONFIGURATION ---
BASE_DIR = Path("/home/teaching-factory/pipeline")
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_TEST = BASE_DIR / "data" / "data_test230.csv"

BASE_MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 128
MAX_NEW_TOKENS = 128

TOP_K = 3                     # jumlah kandidat yang diambil dari pgvector
SIMILARITY_THRESHOLD = 0.75   # threshold cosine similarity top-1 untuk retrieve answer
BM25_K = 3
RRF_K = 60                    # parameter k untuk Reciprocal Rank Fusion
# RERANK_KEYWORD_WEIGHT = 0.2   # bobot keyword overlap pada reranking hybrid
# INTENT_WEIGHT = 0.3          # bobot intent match pada reranking hybrid
MATCH_THRESHOLD = 0.85       # threshold semantic match untuk evaluasi (prediksi vs ground truth)

FALLBACK_ANSWER = "Maaf, saya belum dapat menemukan jawaban yang sesuai untuk pertanyaan Anda."

print(f"Device: {DEVICE}")
print(f"Threshold retrieval: {SIMILARITY_THRESHOLD}")
```

```text
Device: cuda
Threshold retrieval: 0.75
```

```python
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

```text
Database connected
```

```python


# Ambil semua pertanyaan dari DB untuk index BM25 (dijalankan sekali)
cur.execute("SELECT question, answer FROM qa_ground_truth ORDER BY id;")
all_qa = cur.fetchall()

db_questions = [row[0] for row in all_qa]
db_answers   = {row[0]: row[1] for row in all_qa}

# Tokenisasi untuk BM25 (lowercase, split by whitespace)
tokenized_corpus = [q.lower().split() for q in db_questions]
bm25_index = BM25Okapi(tokenized_corpus)

print(f"BM25 index built: {len(db_questions)} pertanyaan di database")
```

```text
BM25 index built: 1500 pertanyaan di database
```

```python
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
    attn_implementation="sdpa",
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
```

```text
============================================================
LOADING FINE-TUNED MODEL
============================================================
Loading tokenizer...
Loading model 8-bit with Unsloth...
Loading Model....
Unsloth: WARNING `trust_remote_code` is True.
Are you certain you want to do remote code execution?
==((====))==  Unsloth 2026.6.7: Fast Ministral3 patching. Transformers: 5.2.0.
   \\   /|    NVIDIA GeForce RTX 5090. Num GPUs = 1. Max memory: 31.342 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: Mistral3 does not support SDPA - switching to fast eager.
Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.
```

```text
Loading weights: 100%|██████████| 585/585 [00:15<00:00, 38.98it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weight]               
The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
```

```text
Loading PEFT model...
Model loaded successfully on: cuda:0
```

```python
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
embedding_model.eval()

print("Embedding model loaded successfully.")
```

```text
Loading sentence embedding model...
```

```text
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 2028.49it/s, Materializing param=pooler.dense.weight]                               
```

```text
Embedding model loaded successfully.
```

```python
print("Loading test dataset...")
df = pd.read_csv(DATA_TEST)

required_cols = {"input_query", "gt_query", "gt_answer"}
assert required_cols.issubset(df.columns), f"Kolom kurang: {required_cols - set(df.columns)}"

print(f"Test dataset loaded: {len(df)} samples")
df.head()
```

```text
Loading test dataset...
Test dataset loaded: 230 samples
```

```text
                             input_query  \
0      Durasi KP lamanya berapa bulan ya   
1            KP tuh kredit SKS nya brpa?   
2  cara dapetin surat pengantar KP yaapa   
3           dosbing KP yg nentuin siapa?   
4  batas anggota tim KP tuh berapa orang   

                                            gt_query  \
0       Berapa lama waktu pelaksanaan kerja praktik?   
1  Berapa jumlah kredit mata kuliah Kerja Praktik...   
2  Bagaimana cara mendapatkan Surat Pengantar ker...   
3  Siapa yang menentukan dosen pembimbing kerja p...   
4  Berapa jumlah anggota maksimal dalam satu tim ...   

                                           gt_answer  Unnamed: 3  Unnamed: 4  \
0                      Minimal 1 bulan di perusahaan         NaN         NaN   
1          Jumlah kredit Mata Kuliah KP adalah 2 SKS         NaN         NaN   
2  Meminta persetujuan form pengajuan KP ke Koord...         NaN         NaN   
3  Dosen pembimbing KP ditentukan oleh koordinato...         NaN         NaN   
4                               Maksimal 2 mahasiswa         NaN         NaN   

   Unnamed: 5  Unnamed: 6  
0         NaN         NaN  
1         NaN         NaN  
2         NaN         NaN  
3         NaN         NaN  
4         NaN         NaN  
```

```python
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
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.strip().strip('"').strip("'")
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

```python
# Run normalization

normalized_queries = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    informal_text = row["input_query"]

    normalized = normalize_text(informal_text)

    normalized_queries.append(normalized)

# simpan hasil
df["normalized_query"] = normalized_queries
```

```text
100%|██████████| 230/230 [06:22<00:00,  1.66s/it]
```

```python
print("Encoding normalized queries...")
query_vectors = embedding_model.encode(
    df["normalized_query"].astype(str).tolist(),
    batch_size=32,
    show_progress_bar=True
)
df["query_vector"] = [v.tolist() for v in query_vectors]
print(f"Encoded {len(query_vectors)} queries, dim = {len(query_vectors[0])}")
```

```text
Encoding normalized queries...
```

```text
Batches: 100%|██████████| 8/8 [00:00<00:00, 44.95it/s]
```

```text
Encoded 230 queries, dim = 1024
```

```python
def get_bm25_candidates(query, k):
    """Kembalikan top-k kandidat dari BM25 sebagai list dict."""
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        {"question": db_questions[i], "answer": db_answers[db_questions[i]],
         "bm25_score": float(scores[i]), "bm25_rank": rank + 1}
        for rank, i in enumerate(top_indices)
    ]


retrieval_records = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieval (BM25+Vector RRF)"):
    normalized_query = row["normalized_query"]
    vector_literal   = "[" + ",".join(map(str, row["query_vector"])) + "]"

    # ── 1. VECTOR SEARCH (pgvector) ──────────────────────────────────────
    try:
        cur.execute("""
            SELECT question, answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_ground_truth
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_literal, vector_literal, TOP_K))
        vec_results = cur.fetchall()
    except Exception as e:
        conn.rollback()
        vec_results = []
        print(f"DB error at row {idx}: {type(e).__name__}: {e}")

    # buat dict: question → {similarity, vector_rank}
    vec_map = {}
    for rank, (q, a, s) in enumerate(vec_results, start=1):
        vec_map[q] = {"answer": a, "similarity": float(s), "vector_rank": rank}

    # ── 2. BM25 SEARCH ───────────────────────────────────────────────────
    bm25_cands = get_bm25_candidates(normalized_query, BM25_K)
    bm25_map   = {c["question"]: c for c in bm25_cands}

    # ── 3. RECIPROCAL RANK FUSION (RRF) ──────────────────────────────────
    all_questions = set(vec_map.keys()) | set(bm25_map.keys())
    fused = []
    for q in all_questions:
        vec_rank = vec_map[q]["vector_rank"] if q in vec_map else (TOP_K + 1)
        bm25_rank = bm25_map[q]["bm25_rank"] if q in bm25_map else (BM25_K + 1)
        rrf_score = 1 / (RRF_K + vec_rank) + 1 / (RRF_K + bm25_rank)

        # ambil similarity dari vector search (untuk threshold)
        similarity = vec_map[q]["similarity"] if q in vec_map else 0.0
        answer     = vec_map[q]["answer"] if q in vec_map else bm25_map[q]["answer"]

        fused.append({
            "question":   q,
            "answer":     answer,
            "similarity": similarity,       # cosine similarity asli (untuk threshold)
            "rrf_score":  rrf_score,
            "vector_rank": vec_rank,
            "bm25_rank":  bm25_rank,
            "rerank_score": rrf_score,      # alias agar kompatibel dengan sel evaluasi
        })

    reranked = sorted(fused, key=lambda c: c["rrf_score"], reverse=True)

    # ── 4. TOP-1 + THRESHOLD ─────────────────────────────────────────────
    if reranked and reranked[0]["similarity"] >= SIMILARITY_THRESHOLD:
        answered        = 1
        retrieval_result = reranked[0]["answer"]
    else:
        answered        = 0
        retrieval_result = FALLBACK_ANSWER

    # simpan candidates dalam format konsisten dengan sel ablation
    candidates = [
        {**vec_map[q], "question": q, "keyword_score": 0.0, "rerank_score": 0.0}
        for q in vec_map
    ]

    retrieval_records.append({
        "candidates":      candidates,
        "reranked":        reranked,
        "answered":        answered,
        "retrieval_result": retrieval_result,
    })

df["retrieval_result"] = [r["retrieval_result"] for r in retrieval_records]
df["answered"]         = [r["answered"]         for r in retrieval_records]
print("Retrieval selesai.")
print(f"Answered (>= threshold): {df['answered'].sum()}/{len(df)}")
```

```text
Retrieval (BM25+Vector RRF): 100%|██████████| 230/230 [00:01<00:00, 174.32it/s]
```

```text
Retrieval selesai.
Answered (>= threshold): 227/230
```

```python
# INTENT_GROUPS = {
#     "definisi":  ["dimaksud", "pengertian", "apa itu", "artinya", "definisi"],
#     "cara":      ["cara", "bagaimana", "langkah", "prosedur", "alur"],
#     "syarat":    ["syarat", "persyaratan", "ketentuan", "harus", "wajib"],
#     "waktu":     ["kapan", "berapa lama", "durasi", "deadline", "batas waktu"],
#     "jumlah":    ["berapa", "jumlah", "banyak", "maksimal", "minimal", "sks"],
#     "tujuan":    ["tujuan", "manfaat", "keuntungan", "fungsi"],
#     "siapa":     ["siapa", "pihak", "dosen", "koordinator"],
#     "dokumen":   ["dokumen", "berkas", "surat", "form", "formulir"],
# }

# def get_intent(text):
#     t = text.lower()
#     for intent, keywords in INTENT_GROUPS.items():
#         if any(k in t for k in keywords):
#             return intent
#     return None


# def keyword_overlap(query, candidate):
#     q_words = set(str(query).lower().split())
#     c_words = set(str(candidate).lower().split())
#     return len(q_words & c_words) / (len(q_words) + 1e-6)

# def intent_match_score(query, candidate_question):
#     """Bonus jika intent cocok, penalti jika beda, netral jika tidak terdeteksi."""
#     q_intent = get_intent(query)
#     c_intent = get_intent(candidate_question)
#     if q_intent is None:
#         return 0.0    # query tidak jelas intentnya → netral
#     if q_intent == c_intent:
#         return 1.0    # intent cocok → bonus penuh
#     return -0.5       # intent beda → penalti

# retrieval_records = []

# for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieval"):
#     normalized_query = row["normalized_query"]
#     vector_literal = "[" + ",".join(map(str, row["query_vector"])) + "]"

#     try:
#         cur.execute("""
#             SELECT question, answer,
#                    1 - (embedding <=> %s::vector) AS similarity
#             FROM qa_ground_truth
#             ORDER BY embedding <=> %s::vector
#             LIMIT %s;
#         """, (vector_literal, vector_literal, TOP_K))
#         results = cur.fetchall()
#     except Exception as e:
#         conn.rollback()
#         results = []
#         print(f"DB error at row {idx}: {type(e).__name__}: {e}")

#     candidates = [
#         {"question": q, "answer": a, "similarity": float(s)}
#         for (q, a, s) in results
#     ]

#     # --- RERANKING HYBRID ---
#     for c in candidates:
#         c["keyword_score"] = keyword_overlap(normalized_query, c["question"])
#         c["intent_score"]  = intent_match_score(normalized_query, c["question"])
#         c["rerank_score"]  = (c["similarity"]
#                               + RERANK_KEYWORD_WEIGHT * c["keyword_score"]
#                               + INTENT_WEIGHT         * c["intent_score"])

#     reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

#     # --- TOP-1 + THRESHOLD FILTER ---
#     if reranked and reranked[0]["similarity"] >= SIMILARITY_THRESHOLD:
#         answered = 1
#         retrieval_result = reranked[0]["answer"]
#     else:
#         answered = 0
#         retrieval_result = FALLBACK_ANSWER

#     retrieval_records.append({
#         "candidates": candidates,   # urutan asli dari pgvector
#         "reranked": reranked,       # urutan setelah reranking
#         "answered": answered,
#         "retrieval_result": retrieval_result,
#     })

# df["retrieval_result"] = [r["retrieval_result"] for r in retrieval_records]
# df["answered"] = [r["answered"] for r in retrieval_records]
# print("Retrieval selesai.")
# print(f"Answered (>= threshold): {df['answered'].sum()}/{len(df)}")
```

```python
# # COBA LAGIIIIIIIII

# def keyword_overlap(query, candidate):
#     q_words = set(str(query).lower().split())
#     c_words = set(str(candidate).lower().split())
#     return len(q_words & c_words) / (len(q_words) + 1e-6)


# retrieval_records = []

# for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieval"):
#     normalized_query = row["normalized_query"]
#     vector_literal = "[" + ",".join(map(str, row["query_vector"])) + "]"

#     try:
#         cur.execute("""
#             SELECT question, answer,
#                    1 - (embedding <=> %s::vector) AS similarity
#             FROM qa_ground_truth
#             ORDER BY embedding <=> %s::vector
#             LIMIT %s;
#         """, (vector_literal, vector_literal, TOP_K))
#         results = cur.fetchall()
#     except Exception as e:
#         conn.rollback()
#         results = []
#         print(f"DB error at row {idx}: {type(e).__name__}: {e}")

#     candidates = [
#         {"question": q, "answer": a, "similarity": float(s)}
#         for (q, a, s) in results
#     ]

#     # --- RERANKING HYBRID (cosine + keyword overlap) ---
#     for c in candidates:
#         c["keyword_score"] = keyword_overlap(normalized_query, c["question"])
#         c["rerank_score"]  = c["similarity"] + RERANK_KEYWORD_WEIGHT * c["keyword_score"]

#     reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

#     # --- TOP-1 + THRESHOLD FILTER ---
#     if reranked and reranked[0]["similarity"] >= SIMILARITY_THRESHOLD:
#         answered = 1
#         retrieval_result = reranked[0]["answer"]
#     else:
#         answered = 0
#         retrieval_result = FALLBACK_ANSWER

#     retrieval_records.append({
#         "candidates": candidates,   # urutan asli dari pgvector
#         "reranked": reranked,       # urutan setelah reranking
#         "answered": answered,
#         "retrieval_result": retrieval_result,
#     })

# df["retrieval_result"] = [r["retrieval_result"] for r in retrieval_records]
# df["answered"]         = [r["answered"]         for r in retrieval_records]
# print("Retrieval selesai.")
# print(f"Answered (>= threshold): {df['answered'].sum()}/{len(df)}")
```

```python
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

```text
Metric functions ready
```

```python
eval_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    rec        = retrieval_records[idx]
    reranked   = rec["reranked"]
    gt_query   = row["gt_query"]
    gt_answer  = row["gt_answer"]

    # --- Rank ground-truth dalam hasil reranking ---
    gt_rank = 0
    for i, cand in enumerate(reranked, start=1):
        matched, _ = is_match(cand["question"], gt_query)
        if matched:
            gt_rank = i
            break

    # --- Hit@k dan MRR (dinamis sesuai TOP_K) ---
    hit_at = {k: int(0 < gt_rank <= k) for k in range(1, TOP_K + 1)}
    rr     = round(1.0 / gt_rank if gt_rank > 0 else 0.0, 4)

    # --- Top-1 info (untuk threshold & answer check) ---
    if reranked:
        top1               = reranked[0]
        top1_sim           = top1["similarity"]
        top1_correct_bool, top1_answer_sim = is_match(top1["answer"], gt_answer)
        top1_answer_correct = int(top1_correct_bool)
    else:
        top1_sim            = 0.0
        top1_answer_correct = 0
        top1_answer_sim     = 0.0

    # --- Hasil akhir sistem ---
    answered       = rec["answered"]
    answer_correct = top1_answer_correct if answered else 0
    e2e_correct    = int(answered == 1 and answer_correct == 1)

    # --- Kualitas normalisasi: normalized_query vs gt_query (sentence transformer) ---
    normalization_sim = semantic_sim(row["normalized_query"], gt_query)

    # --- Bangun row data ---
    row_data = {
        "input_query":            row["input_query"],
        "normalized_query":       row["normalized_query"],
        "gt_query":               gt_query,
        "gt_answer":              gt_answer,
    }

    # Top-1 s/d Top-K question + similarity (dinamis)
    for k in range(1, TOP_K + 1):
        if k <= len(reranked):
            row_data[f"top{k}_question"]   = reranked[k-1]["question"]
            row_data[f"top{k}_similarity"] = round(reranked[k-1]["similarity"], 4)
        else:
            row_data[f"top{k}_question"]   = ""
            row_data[f"top{k}_similarity"] = 0.0

    row_data["top1_question_chosen"] = reranked[0]["question"] if reranked else ""
    row_data["retrieval_result"]     = rec["retrieval_result"]
    row_data["normalization_similarity"] = round(normalization_sim, 4)

    # Hit@1 s/d Hit@K (dinamis)
    for k in range(1, TOP_K + 1):
        row_data[f"hit@{k}"] = hit_at[k]

    row_data["mrr"]            = rr
    row_data["answer_correct"] = answer_correct

    # Kolom tambahan untuk analisis & ablation (tidak masuk CSV utama)
    row_data["_gt_rank"]            = gt_rank
    row_data["_top1_similarity"]    = round(top1_sim, 4)
    row_data["_top1_answer_correct"]= top1_answer_correct
    row_data["_answer_similarity"]  = round(top1_answer_sim, 4)
    row_data["_answered"]           = answered
    row_data["_e2e_correct"]        = e2e_correct

    eval_rows.append(row_data)

results_df = pd.DataFrame(eval_rows)

# Kolom tampilan utama
display_cols = (
    ["input_query", "normalized_query", "gt_query", "gt_answer"]
    + [f"top{k}_question" for k in range(1, TOP_K + 1)]
    + ["top1_question_chosen", "retrieval_result", "normalization_similarity"]
    + [f"hit@{k}" for k in range(1, TOP_K + 1)]
    + ["mrr", "answer_correct"]
)

print(f"Evaluation selesai: {len(results_df)} samples | TOP_K={TOP_K}")
print("\nDETAILED RESULTS (5 baris pertama):")
display(results_df[display_cols].head(5))
```

```text
Evaluating: 100%|██████████| 230/230 [00:02<00:00, 83.09it/s] 
```

```text
Evaluation selesai: 230 samples | TOP_K=3

DETAILED RESULTS (5 baris pertama):
```

```text
                             input_query  \
0      Durasi KP lamanya berapa bulan ya   
1            KP tuh kredit SKS nya brpa?   
2  cara dapetin surat pengantar KP yaapa   
3           dosbing KP yg nentuin siapa?   
4  batas anggota tim KP tuh berapa orang   

                                    normalized_query  \
0       Berapa lama waktu pelaksanaan kerja praktik?   
1  Berapa jumlah kredit mata kuliah Kerja Praktik...   
2  Bagaimana cara mendapatkan Surat Pengantar ker...   
3  Siapa yang menentukan dosen pembimbing kerja p...   
4     Berapa jumlah maksimal anggota Kelompok Kerja?   

                                            gt_query  \
0       Berapa lama waktu pelaksanaan kerja praktik?   
1  Berapa jumlah kredit mata kuliah Kerja Praktik...   
2  Bagaimana cara mendapatkan Surat Pengantar ker...   
3  Siapa yang menentukan dosen pembimbing kerja p...   
4  Berapa jumlah anggota maksimal dalam satu tim ...   

                                           gt_answer  \
0                      Minimal 1 bulan di perusahaan   
1          Jumlah kredit Mata Kuliah KP adalah 2 SKS   
2  Meminta persetujuan form pengajuan KP ke Koord...   
3  Dosen pembimbing KP ditentukan oleh koordinato...   
4                               Maksimal 2 mahasiswa   

                                       top1_question  \
0       Berapa lama waktu pelaksanaan kerja praktik?   
1  Berapa jumlah kredit mata kuliah Kerja Praktik...   
2  Bagaimana cara mendapatkan Surat Pengantar ker...   
3  Siapa yang menentukan dosen pembimbing kerja p...   
4  Berapa jumlah anggota maksimal dalam satu tim ...   

                                       top2_question  \
0      Berapa lama waktu penyelesaian pembuatan SKL?   
1                        Apa itu kerja praktik (KP)?   
2  Bagaimana cara mendapatkan penilaian kerja pra...   
3  Siapa yang menjadi pembimbing kerja praktik ma...   
4  Berapa jumlah maksimal prestasi yang dapat dil...   

                                       top3_question  \
0                    Apa tujuan utama kerja praktik?   
1  Berapa kredit SKS mata kuliah pengayaan yang w...   
2  Siapa yang harus menandatangani proposal kerja...   
3  Siapa yang harus menandatangani proposal kerja...   
4  Berapa jumlah maksimal prestasi yang dapat dil...   

                                top1_question_chosen  \
0       Berapa lama waktu pelaksanaan kerja praktik?   
1  Berapa jumlah kredit mata kuliah Kerja Praktik...   
2  Bagaimana cara mendapatkan Surat Pengantar ker...   
3  Siapa yang menentukan dosen pembimbing kerja p...   
4  Berapa jumlah anggota maksimal dalam satu tim ...   

                                    retrieval_result  \
0                      Minimal 1 bulan di perusahaan   
1          Jumlah kredit Mata Kuliah KP adalah 2 SKS   
2  Meminta persetujuan form pengajuan KP ke Koord...   
3  Dosen pembimbing KP ditentukan oleh koordinato...   
4                               Maksimal 2 mahasiswa   

   normalization_similarity  hit@1  hit@2  hit@3  mrr  answer_correct  
0                    1.0000      1      1      1  1.0               1  
1                    1.0000      1      1      1  1.0               1  
2                    1.0000      1      1      1  1.0               1  
3                    1.0000      1      1      1  1.0               1  
4                    0.8079      1      1      1  1.0               1  
```

```python
# eval_rows = []

# for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#     rec = retrieval_records[idx]
#     reranked = rec["reranked"]
#     gt_query = row["gt_query"]
#     gt_answer = row["gt_answer"]

#     # --- Cari rank ground-truth question pada hasil reranking ---
#     gt_rank = 0
#     for i, cand in enumerate(reranked, start=1):
#         matched, _ = is_match(cand["question"], gt_query)
#         if matched:
#             gt_rank = i
#             break

#     hit_at_1 = int(gt_rank == 1)
#     hit_at_2 = int(0 < gt_rank <= 2)
#     hit_at_3 = int(0 < gt_rank <= 3)
#     rr = 1.0 / gt_rank if gt_rank > 0 else 0.0
#     ndcg_at_3 = 1.0 / np.log2(gt_rank + 1) if 0 < gt_rank <= 3 else 0.0

#     # --- Info top-1 setelah reranking ---
#     if reranked:
#         top1 = reranked[0]
#         top1_question = top1["question"]
#         top1_similarity = top1["similarity"]
#         top1_rerank_score = top1["rerank_score"]
#         # benar/tidaknya jawaban top-1 (independen dari threshold, untuk analisis threshold)
#         top1_correct_bool, top1_answer_sim = is_match(top1["answer"], gt_answer)
#         top1_answer_correct = int(top1_correct_bool)
#     else:
#         top1_question = ""
#         top1_similarity = 0.0
#         top1_rerank_score = 0.0
#         top1_answer_correct = 0
#         top1_answer_sim = 0.0

#     # --- Hasil akhir sistem (dengan threshold) ---
#     answered = rec["answered"]
#     answer_correct = top1_answer_correct if answered else 0
#     e2e_correct = int(answered == 1 and answer_correct == 1)

#     # --- Kualitas normalisasi: hasil normalisasi vs pertanyaan ground truth ---
#     normalization_sim = semantic_sim(row["normalized_query"], gt_query)

#     eval_rows.append({
#         "input_query": row["input_query"],
#         "normalized_query": row["normalized_query"],
#         "gt_query": gt_query,
#         "gt_answer": gt_answer,
#         "top1_question": top1_question,
#         "top1_similarity": round(top1_similarity, 4),
#         "top1_rerank_score": round(top1_rerank_score, 4),
#         "retrieval_result": rec["retrieval_result"],
#         "answered": answered,
#         "gt_rank": gt_rank,
#         "hit@1": hit_at_1,
#         "hit@2": hit_at_2,
#         "hit@3": hit_at_3,
#         "rr": round(rr, 4),
#         "ndcg@3": round(ndcg_at_3, 4),
#         "top1_answer_correct": top1_answer_correct,
#         "answer_similarity": round(top1_answer_sim, 4),
#         "answer_correct": answer_correct,
#         "e2e_correct": e2e_correct,
#         "normalization_similarity": round(normalization_sim, 4),
#     })

# results_df = pd.DataFrame(eval_rows)

# print(f"Evaluation selesai: {len(results_df)} samples")
# print("\nDETAILED RESULTS PER TEST CASE:")
# display(results_df[[
#     "input_query", "normalized_query", "retrieval_result",
#     "top1_similarity", "answered", "gt_rank",
#     "hit@1", "hit@3", "rr", "ndcg@3", "answer_correct", "e2e_correct"
# ]].head(5))
```

```python
n               = len(results_df)
n_answered      = int(results_df["_answered"].sum())
n_fallback      = n - n_answered
n_answer_correct= int(results_df["answer_correct"].sum())
n_e2e_correct   = int(results_df["_e2e_correct"].sum())

# Hit@k dinamis
hits = {k: int(results_df[f"hit@{k}"].sum()) for k in range(1, TOP_K + 1)}
mrr  = results_df["mrr"].mean()

# nDCG@k dinamis (dihitung ulang dari _gt_rank)
def ndcg_at(k):
    return results_df["_gt_rank"].apply(
        lambda r: 1.0 / np.log2(r + 1) if 0 < r <= k else 0.0
    ).mean()

answer_rate     = n_answered / n
answer_accuracy = n_answer_correct / n_answered if n_answered > 0 else 0.0
e2e_accuracy    = n_e2e_correct / n

print("=" * 70)
print("OVERALL RETRIEVAL METRICS")
print("=" * 70)
print(f"Jumlah sampel                 : {n}")
print(f"TOP_K                         : {TOP_K}")
print(f"Threshold similarity          : {SIMILARITY_THRESHOLD}")
print("-" * 70)
for k in range(1, TOP_K + 1):
    print(f"Hit@{k}                         : {hits[k]}/{n} ({hits[k]/n:.4f})")
print(f"MRR                           : {mrr:.4f}")
for k in range(1, TOP_K + 1):
    print(f"nDCG@{k}                        : {ndcg_at(k):.4f}")
print("-" * 70)
print(f"Answer Rate (>= threshold)    : {n_answered}/{n} ({answer_rate:.4f})")
print(f"Fallback / 'Maaf...'          : {n_fallback}/{n} ({n_fallback/n:.4f})")
print(f"Answer Accuracy (yg dijawab)  : {n_answer_correct}/{n_answered} ({answer_accuracy:.4f})")
print(f"End-to-End Accuracy           : {n_e2e_correct}/{n} ({e2e_accuracy:.4f})")
print("-" * 70)
print(f"Avg Top-1 Similarity          : {results_df['_top1_similarity'].mean():.4f}")
print(f"Avg Answer Similarity         : {results_df['_answer_similarity'].mean():.4f}")
print(f"Avg Normalization Similarity  : {results_df['normalization_similarity'].mean():.4f}")
print("=" * 70)

# Summary table
metric_names  = ([f"Hit@{k}" for k in range(1, TOP_K + 1)]
                 + ["MRR"]
                 + [f"nDCG@{k}" for k in range(1, TOP_K + 1)]
                 + ["Answer Rate", "Answer Accuracy", "End-to-End Accuracy",
                    "Avg Top-1 Similarity", "Avg Normalization Similarity"])
metric_values = ([hits[k]/n for k in range(1, TOP_K + 1)]
                 + [mrr]
                 + [ndcg_at(k) for k in range(1, TOP_K + 1)]
                 + [answer_rate, answer_accuracy, e2e_accuracy,
                    results_df["_top1_similarity"].mean(),
                    results_df["normalization_similarity"].mean()])
metric_counts = ([f"{hits[k]}/{n}" for k in range(1, TOP_K + 1)]
                 + ["-"]
                 + ["-"] * TOP_K
                 + [f"{n_answered}/{n}", f"{n_answer_correct}/{n_answered}",
                    f"{n_e2e_correct}/{n}", "-", "-"])

summary_stats = pd.DataFrame({
    "Metric": metric_names,
    "Value":  metric_values,
    "Count":  metric_counts,
})

print("\nSummary Table:")
display(summary_stats)
```

```text
======================================================================
OVERALL RETRIEVAL METRICS
======================================================================
Jumlah sampel                 : 230
TOP_K                         : 3
Threshold similarity          : 0.75
----------------------------------------------------------------------
Hit@1                         : 204/230 (0.8870)
Hit@2                         : 211/230 (0.9174)
Hit@3                         : 217/230 (0.9435)
MRR                           : 0.9148
nDCG@1                        : 0.8870
nDCG@2                        : 0.9062
nDCG@3                        : 0.9192
----------------------------------------------------------------------
Answer Rate (>= threshold)    : 227/230 (0.9870)
Fallback / 'Maaf...'          : 3/230 (0.0130)
Answer Accuracy (yg dijawab)  : 197/227 (0.8678)
End-to-End Accuracy           : 197/230 (0.8565)
----------------------------------------------------------------------
Avg Top-1 Similarity          : 0.9648
Avg Answer Similarity         : 0.9428
Avg Normalization Similarity  : 0.9580
======================================================================

Summary Table:
```

```text
                          Metric     Value    Count
0                          Hit@1  0.886957  204/230
1                          Hit@2  0.917391  211/230
2                          Hit@3  0.943478  217/230
3                            MRR  0.914782        -
4                         nDCG@1  0.886957        -
5                         nDCG@2  0.906159        -
6                         nDCG@3  0.919202        -
7                    Answer Rate  0.986957  227/230
8                Answer Accuracy  0.867841  197/227
9            End-to-End Accuracy  0.856522  197/230
10          Avg Top-1 Similarity  0.964804        -
11  Avg Normalization Similarity  0.958019        -
```

```python
# Save detailed results to CSV
output_csv = RESULTS_DIR / "retrieval_evaluation_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nDetailed results saved to: {output_csv}")
```

```text

Detailed results saved to: /home/teaching-factory/pipeline/results/retrieval_evaluation_results.csv
```

```python
# Kolom CSV utama (tanpa kolom internal _*)
csv_cols = (
    ["input_query", "normalized_query", "gt_query", "gt_answer"]
    + [f"top{k}_question"   for k in range(1, TOP_K + 1)]
    + [f"top{k}_similarity" for k in range(1, TOP_K + 1)]
    + ["top1_question_chosen", "retrieval_result", "normalization_similarity"]
    + [f"hit@{k}" for k in range(1, TOP_K + 1)]
    + ["mrr", "answer_correct"]
)

output_csv = RESULTS_DIR / f"retrieval_results_top{TOP_K}.csv"
results_df[csv_cols].to_csv(output_csv, index=False)
print(f"Results saved to: {output_csv}")
print(f"Columns ({len(csv_cols)}): {csv_cols}")
```

```text
Results saved to: /home/teaching-factory/pipeline/results/retrieval_results_top3.csv
Columns (18): ['input_query', 'normalized_query', 'gt_query', 'gt_answer', 'top1_question', 'top2_question', 'top3_question', 'top1_similarity', 'top2_similarity', 'top3_similarity', 'top1_question_chosen', 'retrieval_result', 'normalization_similarity', 'hit@1', 'hit@2', 'hit@3', 'mrr', 'answer_correct']
```

```python


sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Hit@k bar chart — dinamis sesuai TOP_K
ax1 = axes[0, 0]
hit_labels = [f"Hit@{k}" for k in range(1, TOP_K + 1)]
hit_vals   = [hits[k]/n  for k in range(1, TOP_K + 1)]
hit_counts_list = [hits[k] for k in range(1, TOP_K + 1)]
palette    = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c",
              "#e74c3c", "#34495e", "#e67e22", "#16a085", "#8e44ad"][:TOP_K]
bars = ax1.bar(hit_labels, hit_vals, color=palette, alpha=0.8, edgecolor="black")
for bar, v, c in zip(bars, hit_vals, hit_counts_list):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.3f}\n({c}/{n})", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Rate", fontsize=12, fontweight="bold")
ax1.set_title(f"Hit@k  (TOP_K={TOP_K})", fontsize=14, fontweight="bold")

# 2. Distribusi top-1 similarity + threshold
ax2 = axes[0, 1]
ax2.hist(results_df["_top1_similarity"], bins=30, color="#9b59b6", alpha=0.7, edgecolor="black")
ax2.axvline(SIMILARITY_THRESHOLD, color="red", linestyle="--", linewidth=2,
            label=f"Threshold = {SIMILARITY_THRESHOLD}")
ax2.set_xlabel("Top-1 Cosine Similarity", fontsize=12, fontweight="bold")
ax2.set_ylabel("Jumlah Query", fontsize=12, fontweight="bold")
ax2.set_title("Distribusi Top-1 Similarity", fontsize=14, fontweight="bold")
ax2.legend()

# 3. Overall metrics bar
ax3 = axes[1, 0]
metric_labels = ["MRR", f"nDCG@{TOP_K}", "Answer\nRate", "Answer\nAccuracy", "E2E\nAccuracy"]
metric_values = [mrr, ndcg_at(TOP_K), answer_rate, answer_accuracy, e2e_accuracy]
bars3 = ax3.bar(metric_labels, metric_values,
                color=["#3498db", "#1abc9c", "#f39c12", "#2ecc71", "#e74c3c"],
                alpha=0.8, edgecolor="black")
for bar, v in zip(bars3, metric_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Score", fontsize=12, fontweight="bold")
ax3.set_title("Overall Metrics", fontsize=14, fontweight="bold")

# 4. Breakdown hasil akhir
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
# plt.savefig(RESULTS_DIR / f"retrieval_eval_plots_top{TOP_K}.png", dpi=300, bbox_inches="tight")
plt.show()
```

![output image 21-0](images/cell-21-0.png)

```python
thresholds = np.arange(0.50, 0.96, 0.025)
sweep_answer_rate = []
sweep_e2e_acc = []
sweep_answer_acc = []

for t in thresholds:
    answered_t = results_df["top1_similarity"] >= t
    correct_t = answered_t & (results_df["_top1_answer_correct"] == 1)
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
# print(f"Plot saved to: {threshold_plot_file}")
```

![output image 22-0](images/cell-22-0.png)

```python
failed_df = results_df[results_df["_e2e_correct"] == 0]
print(f"Jumlah kasus gagal: {len(failed_df)}/{n}")
print(f"  - Fallback (di bawah threshold) : {(failed_df['_answered'] == 0).sum()}")
print(f"  - Dijawab tapi salah            : {(failed_df['_answered'] == 1).sum()}")

display(failed_df[[
    "input_query", "normalized_query", "gt_query", "top1_question",
    "top1_similarity", "_answered", "_gt_rank", "retrieval_result"
]])
```

```text
Jumlah kasus gagal: 33/230
  - Fallback (di bawah threshold) : 3
  - Dijawab tapi salah            : 30
```

```text
                                           input_query  \
17                office 365 aktivasinya yaapa caranya   
22   eh cara reset pasword myITS yaapa ya, aku lupa...   
27    kalo mau ngajuin TA lewat mythesis yaapa caranya   
28         bisa diubah gak kalo dah ngajuin usulan TA?   
30                          tempatnya GRITS dimana yak   
31                            gedung SCC tuh ndek mana   
40                       kampus MMT tuh daerah mana ya   
41                  di PLT ITS ada layanan apa aja nih   
55                       UKT nya PJJ reguler berapa ya   
66                          kadepnya mesin sapa ya rek   
70                          kaprodi Tekpal siapa gaess   
86             matkul pengayaan tuh kredit sks nya brp   
96         kegiatan yang bisa diklaim SKEM tuh apa aja   
103    maksimal berapa lama dapat beasiswa pegawai ITS   
109              kalo tugasnya DRPM tuh lingkupnya apa   
112                       SIMT ITS tuh yg mimpin siapa   
115  eh sekarang ada program kuliah paruh waktu, it...   
122                         kuliah S2 tuh brp taun sih   
131         ntar teknologi kedokteran gelarnya apa yah   
141  pendaftaran beasiswa freshgrad s2 bakal dibuka...   
143  semester ganjil ini jadwal bayar UKT nya kpn broo   
154                 tes skolastik UTBK nih kaya gimana   
170  dokumen yg harus dipenuhi buat daftar SMITS ap...   
179                   btw IUP nih jalur masuk apa yaa?   
181             syarat kalo mau daftar IUP apa aja rek   
190  eh kalo telat balikin kunci loker perpus denda...   
198     score english BC minimal berapa buat konversi?   
201                 daftar program insinyur berapa sih   
203       SKS nya program insinyur RPL berapa totalnya   
219                        trus ngapain tugasnya senat   
222                         siapa ketuanya senat skrg?   
223                          btw MBKM nih kegiatan apa   
225               nah kita bisa ngambil mbkm pas kapan   

                                      normalized_query  \
17   Bagaimana cara mengaktifkan lisensi Office 365...   
22      Bagaimana cara reset password MyITS jika lupa?   
27   Bagaimana cara mengajukan surat rekomendasi ke...   
28   Apakah mahasiswa bisa mengubah rencana penelit...   
30   Dimana lokasi Gedung Research and Innovation C...   
31                Dimana lokasi Auditorium Gedung SCC?   
40   Dimana lokasi Departemen Manajemen Teknologi (...   
41   Apa saja layanan yang ditawarkan oleh Pusat La...   
55   Berapa biaya uang kuliah PPI ITS program reguler?   
66          Siapa nama Kepala Departemen Teknik Mesin?   
70   Siapa nama Kepala Program Studi Teknik Perkapa...   
86   Berapa beban SKS yang harus ditempuh untuk mat...   
96   Apa saja aspek kegiatan kemahasiswaan yang dia...   
103       Apakah ada batasan lama penerimaan beasiswa?   
109  Apa bidang tugas dari Direktorat Riset dan Pen...   
112  Siapa nama Dekan Sekolah Interdisiplin Manajem...   
115  Apa yang dimaksud dengan program paruh waktu d...   
122       Berapa lama masa studi program Magister ITS?   
131  Apa gelar yang diberikan kepada lulusan prodi ...   
141  Kapan jadwal pendaftaran mahasiswa baru Pascas...   
143  Kapan periode pembayaran SPP bagi mahasiswa lama?   
154       Apa yang dimaksud dengan Tes Akademik SMITS?   
170  Apa saja dokumen yang diperlukan untuk mendaft...   
179  Melalui apa saja jalur pendaftaran yang tersed...   
181  Apa saja persyaratan umum pendaftaran IUP ITS ...   
190  Berapa denda keterlambatan pengembalian kunci ...   
198  Berapa score minimal kelulusan jika menggunaka...   
201      Berapa biaya uang kuliah PPI ITS program RPL?   
203  Berapa beban studi yang harus ditempuh dalam P...   
219  Apa fungsi utama Senat Akademik dalam pergurua...   
222               Siapa nama Ketua Senat Akademik ITS?   
223            Apa yang dimaksud dengan kegiatan MBKM?   
225  Kapan mahasiswa dapat mengambil mata kuliah MBKM?   

                                              gt_query  \
17              Bagaimana cara aktivasi Microsoft 365?   
22   Bagaimana cara reset password email ITS jika l...   
27   Bagaimana cara membuat usulan TA/Tesis/Diserta...   
28   Apakah mahasiswa bisa melakukan perubahan data...   
30                     Dimana lokasi Galeri Riset ITS?   
31   Dimana lokasi Student Community Center (SCC) ITS?   
40   Dimana lokasi kampus Magister Manajemen Teknol...   
41   Apa saja layanan yang tersedia pada Pusat Laya...   
55   Berapa biaya UKT untuk Program Magister Pendid...   
66    Siapa nama Kepala Program Studi S1 Teknik Mesin?   
70   Siapa nama Kepala Program Studi S1 Teknik Perk...   
86   Berapa kredit SKS mata kuliah pengayaan yang w...   
96   Apa saja ruang lingkup kegiatan yang termasuk ...   
103  Berapa lama jangka waktu pemberian Beasiswa IT...   
109  Apa tugas Direktorat Riset dan Pengabdian kepa...   
112  Siapa yang memimpin Sekolah Interdisiplin Mana...   
115  Apa yang dimaksud dengan program skema paruh w...   
122  Berapa batas maksimum masa studi Program Magis...   
131  Apa gelar yang diberikan kepada lulusan prodi ...   
141  Kapan jadwal pendaftaran beasiswa fresh gradua...   
143  Kapan pembayaran UKT/SPP semester ganjil 2026/...   
154  Apa yang dimaksud dengan Tes Potensi Skolastik...   
170  Dokumen apa saja yang wajib diunggah saat pend...   
179  Apa yang dimaksud dengan International Undergr...   
181  Apa saja persyaratan akademik yang harus dipen...   
190  Berapa biaya penggantian kunci loker perpustak...   
198  Berapa score minimal British Council untuk bis...   
201                  Berapa biaya pendaftaran PPI ITS?   
203  Berapa beban studi yang harus ditempuh dalam P...   
219  Apa tugas dan wewenang yang dimiliki oleh Sena...   
222  Siapakah yang memimpin Senat Akademik ITS untu...   
223      Apa yang dimaksud dengan program MBKM di ITS?   
225  Pada semester berapa mahasiswa dapat mengikuti...   

                                         top1_question  top1_similarity  \
17   Bagaimana cara mengaktifkan lisensi Office 365...           0.9986   
22                 Bagaimana jika lupa password MyITS?           0.9647   
27   Bagaimana cara mendapatkan Surat Pengantar ker...           0.8267   
28   Sampai kapan mahasiswa dapat mengubah rencana ...           0.8408   
30           Dimana lokasi Gedung Research Center ITS?           0.8550   
31    Dimana lokasi Auditorium Gedung Research Center?           0.7741   
40   Dimana lokasi Departemen D4 Teknik Infrastrukt...           0.7865   
41            Apa itu Pusat Layanan Terpadu (PLT) ITS?           0.9568   
55   Berapa biaya uang kuliah PPI ITS  program regu...           1.0000   
66          Siapa nama Kepala Departemen Teknik Mesin?           1.0000   
70   Siapa nama Kepala Program Studi Pascasarjana T...           0.9552   
86   Berapa SKS minimal yang harus ditempuh untuk m...           0.0000   
96   Apa saja aspek kegiatan kemahasiswaan yang dia...           1.0000   
103                Apakah ada batasan lama cuti studi?           0.8029   
109  Dimana lokasi Direktorat Riset dan Pengabdian ...           0.8955   
112  Siapa nama Dekan Sekolah Interdisiplin Manajem...           1.0000   
115  Apa maksud diterbitkannya Peraturan Rektor Nom...           0.8006   
122       Berapa lama masa studi program Magister ITS?           1.0000   
131  Apa gelar yang diberikan kepada lulusan prodi ...           0.8794   
141  Kapan jadwal pendaftaran mahasiswa baru Pascas...           1.0000   
143  Kapan periode pembayaran SPP bagi mahasiswa lama?           1.0000   
154                Apa yang dimaksud dengan Tes SMITS?           0.9221   
170           Apa saja persayaratan pendaftaran SMITS?           0.9185   
179  Siapa saja yang dapat mendaftar pada program I...           0.8924   
181       Bagaimana prosedur pendaftaran IUP ITS 2026?           0.9298   
190  Berapa denda keterlambatan pengembalian kunci ...           1.0000   
198  Berapa score minimal kelulusan jika menggunaka...           1.0000   
201     Berapa biaya uang kuliah PPI ITS  program RPL?           1.0000   
203  Program studi apa saja yang memenuhi syarat un...           0.8567   
219  Apa fungsi utama Senat Akademik dalam pergurua...           1.0000   
222               Siapa nama Ketua Senat Akademik ITS?           1.0000   
223       Apa yang dimaksud dengan kegiatan prakuliah?           0.0000   
225  Kapan mahasiswa dapat mengambil mata kuliah KK...           0.0000   

     _answered  _gt_rank                                   retrieval_result  
17           1         1  Sign in di aplikasi Office yang sudah diinstal...  
22           1         1  Jika sudah verifikasi email/telepon, gunakan m...  
27           1         0  Meminta persetujuan form pengajuan KP ke Koord...  
28           1         2  Mahasiswa dapat mengubah rencana studinya pali...  
30           1         2  Gedung Research Center ITS terletak di samping...  
31           1         3                Gedung Research ITS Center Lantai 7  
40           1         2  D4 Teknik Infrastruktur Sipil ITS terletak di ...  
41           1         1  Pusat Layanan Terpadu ITS adalah salah satu bi...  
55           1         0  Uang kuliah reguler sebesar Rp12.500.000 untuk...  
66           1         4                        Dr.Eng. Sutikno, S.T., M.T.  
70           1         1       Prof. Aries Sulisetyono, S.T., M.A.Sc, Ph.D.  
86           0         3  Maaf, saya belum dapat menemukan jawaban yang ...  
96           1         1  Aspek kegiatan kemahasiswaan dalam SKEM melipu...  
103          1         0  Ada, biasanya maksimal dua semester selama mas...  
109          1         3  Gedung Research Center Lt. L, Kampus ITS Sukol...  
112          1         3      Prof. Dr.rer.pol. Heri Kuswanto, S.Si., M.Si.  
115          1         0  Peraturan Rektor ini diterbitkan agar setiap p...  
122          1         0  Lama studi normal adalah 4 semester (2 tahun) ...  
131          1         1  Lulusan Program Studi S1 Sistem Informasi dibe...  
141          1         0                         Tanggal 05-08 Januari 2026  
143          1         2  Pembayaran SPP bagi mahasiswa lama dilaksanaka...  
154          1         0  Tes SMITS adalah tes seleksi yang diselenggara...  
170          1         4  Memilik akun myITS Admission, WNI yang memilik...  
179          1         0  IUP ITS terbuka bagi calon mahasiswa dari dala...  
181          1         5  Pendaftaran IUP ITS dilakukan secara daring (o...  
190          1         3  Untuk Mahasiswa dan sivitas akademik ITS Rp10....  
198          1         3  S1 dan S2 minimal 325. S3, Double degree dan I...  
201          1         2  Uang kuliah RPL sebesar Rp12.500.000 untuk bia...  
203          1         2  Program studi yang memenuhi syarat antara lain...  
219          1         0  Fungsi utama Senat Akademik adalah menetapkan ...  
222          1         1                Prof. Dr. Ir. Adi Soeprijanto, M.T.  
223          0         5  Maaf, saya belum dapat menemukan jawaban yang ...  
225          0         2  Maaf, saya belum dapat menemukan jawaban yang ...  
```

```python
def rank_of_gt(candidate_list, gt_query):
    for i, c in enumerate(candidate_list, start=1):
        matched, _ = is_match(c["question"], gt_query)
        if matched:
            return i
    return 0


def evaluate_pipeline(use_rerank):
    hits1 = hits3 = hits5 = e2e = answered_cnt = 0
    rr_list, ndcg5_list = [], []

    for idx, row in df.iterrows():
        rec = retrieval_records[idx]
        cands = rec["reranked"] if use_rerank else rec["candidates"]
        gt_rank = rank_of_gt(cands, row["gt_query"])

        hits1 += int(gt_rank == 1)
        hits3 += int(0 < gt_rank <= 3)
        hits5 += int(0 < gt_rank <= 5)
        rr_list.append(1.0 / gt_rank if gt_rank > 0 else 0.0)
        ndcg5_list.append(1.0 / np.log2(gt_rank + 1) if 0 < gt_rank <= 5 else 0.0)

        if cands and cands[0]["similarity"] >= SIMILARITY_THRESHOLD:
            answered_cnt += 1
            correct, _ = is_match(cands[0]["answer"], row["gt_answer"])
            e2e += int(correct)

    n_total = len(df)
    return {
        "Hit@1":      hits1 / n_total,
        "Hit@3":      hits3 / n_total,
        "Hit@5":      hits5 / n_total,
        "MRR":        np.mean(rr_list),
        "nDCG@5":     np.mean(ndcg5_list),
        "Answer Rate": answered_cnt / n_total,
        "E2E Accuracy": e2e / n_total,
    }


ablation_df = pd.DataFrame({
    "Tanpa Reranking (murni cosine)": evaluate_pipeline(use_rerank=False),
    "Dengan Reranking (hybrid)":      evaluate_pipeline(use_rerank=True),
}).T

ablation_df["Δ Hit@1"] = ablation_df["Hit@1"] - ablation_df.loc["Tanpa Reranking (murni cosine)", "Hit@1"]

print("ABLATION: DENGAN vs TANPA RERANKING")
display(ablation_df.round(4))

changed = sum(
    1 for rec in retrieval_records
    if rec["candidates"] and rec["reranked"]
    and rec["candidates"][0]["question"] != rec["reranked"][0]["question"]
)
print(f"\nQuery yang top-1-nya berubah akibat reranking: {changed}/{len(df)}")
```

```text
ABLATION: DENGAN vs TANPA RERANKING
```

```text
                                 Hit@1   Hit@3   Hit@5     MRR  nDCG@5  \
Tanpa Reranking (murni cosine)  0.8913  0.9435  0.9435  0.9130  0.9208   
Dengan Reranking (hybrid)       0.8870  0.9435  0.9609  0.9148  0.9263   

                                Answer Rate  E2E Accuracy  Δ Hit@1  
Tanpa Reranking (murni cosine)        1.000        0.8565   0.0000  
Dengan Reranking (hybrid)             0.987        0.8565  -0.0043  
```

```text

Query yang top-1-nya berubah akibat reranking: 6/230
```

```python
# Save failed cases to CSV for error analysis
failed_cases_file = RESULTS_DIR / "failed_cases.csv"
failed_df.to_csv(failed_cases_file, index=False)
print(f"Failed cases saved to: {failed_cases_file}")
```

```text
Failed cases saved to: /home/teaching-factory/pipeline/results/failed_cases.csv
```

```python
# print detail question + similarity top-1, top-2, top-3 for all data, convert to dataframe
detailed_records = []
for idx, rec in enumerate(retrieval_records):
    candidates = rec["candidates"]
    for rank, cand in enumerate(candidates, start=1):
        detailed_records.append({
            "index": idx,
            "input_query": df.loc[idx, "input_query"],
            "top_1_question": candidates[0]["question"] if candidates else "",
            "top1_similarity": candidates[0]["similarity"] if candidates else 0.0,
            "top_2_question": candidates[1]["question"] if len(candidates) > 1 else "",
            "top2_similarity": candidates[1]["similarity"] if len(candidates) > 1 else 0.0,
            "top_3_question": candidates[2]["question"] if len(candidates) > 2 else "",
            "top3_similarity": candidates[2]["similarity"] if len(candidates) > 2 else 0.0,
            "retrieval_result": rec["retrieval_result"],
        })
detailed_df = pd.DataFrame(detailed_records)
print(f"Detailed candidate records: {len(detailed_df)} rows")
display(detailed_df.head(20))

# save detailed candidate records to CSV
# detailed_file = RESULTS_DIR / "detailed_records.csv"
# detailed_df.to_csv(detailed_file, index=False)
# print(f"Detailed candidate records saved to: {detailed_file}")
```

```python
# clean up memory
del model, base_model, embedding_model, tokenizer
torch.cuda.empty_cache()
gc.collect()
```

