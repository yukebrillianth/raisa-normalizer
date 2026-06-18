---
jupyter:
  kernelspec:
    display_name: .venv (3.10.12)
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
  nbformat_minor: 4
---

::: {.cell .markdown}
# Testing Fine-tuned Qwen3 14B Model for Text Normalization

Testing model yang sudah di-finetune dengan LoRA adapter
:::

::: {.cell .code}
``` python
!pip install torch transformers peft datasets nltk rouge-score pandas tqdm matplotlib seaborn scikit-learn -q
```
:::

:::::: {.cell .code execution_count="1"}
``` python
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

from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
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
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
else:
    print("CUDA not available")
```

::: {.output .stream .stdout}
    🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
:::

::: {.output .stream .stderr}
    /home/teaching-factory/train_mistral/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
:::

::: {.output .stream .stdout}
    Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.
    🦥 Unsloth Zoo will now patch everything to make training faster!
    Total GPU Memory: 33.7 GB
    GPU Memory Used: 0.01 GB
:::
::::::

::: {.cell .markdown}
## Configuration
:::

:::: {.cell .code execution_count="2"}
``` python
# --- CONFIGURATION ---
BASE_DIR = Path("/home/teaching-factory/train_mistral")
MODEL_DIR = BASE_DIR / "experiments" / "training_r16_2e4" / "models" / "lora_adapters"  # Fixed path to lora_adapters
DATA_DIR = BASE_DIR / "data" / "split"
DATA_TEST = BASE_DIR / "data" / "data_test230.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 256
MAX_NEW_TOKENS = 128

print(f"Device: {DEVICE}")
print(f"Model Directory: {MODEL_DIR}")
print(f"LoRA adapter exists: {(MODEL_DIR / 'adapter_config.json').exists()}")
```

::: {.output .stream .stdout}
    Device: cuda
    Model Directory: /home/teaching-factory/train_mistral/experiments/training_r16_2e4/models/lora_adapters
    LoRA adapter exists: True
:::
::::

::: {.cell .markdown}
## Load Fine-tuned Model
:::

:::::::: {.cell .code execution_count="3"}
``` python
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
os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"

base_model, _ = FastLanguageModel.from_pretrained(
    BASE_MODEL_NAME,  # Load base model
    max_seq_length=MAX_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=False,
    load_in_8bit=True,
    quantization_config=quantization_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",  # avoid flex_attention mask-shape bug during generate
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

::: {.output .stream .stdout}
    ============================================================
    LOADING FINE-TUNED MODEL
    ============================================================
    Loading tokenizer...
:::

::: {.output .stream .stderr}
    The tokenizer you are loading from 'mistralai/Ministral-3-14B-Instruct-2512' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
:::

::: {.output .stream .stdout}
    Loading model 8-bit with Unsloth...
    Loading Model....
    Unsloth: WARNING `trust_remote_code` is True.
    Are you certain you want to do remote code execution?
    ==((====))==  Unsloth 2026.6.1: Fast Ministral3 patching. Transformers: 5.10.2. vLLM: 0.22.1.
       \\   /|    NVIDIA GeForce RTX 5090. Num GPUs = 1. Max memory: 31.342 GB. Platform: Linux.
    O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 12.0. CUDA Toolkit: 13.0. Triton: 3.6.0
    \        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
     "-____-"     Free license: http://github.com/unslothai/unsloth
    Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
    Unsloth: Mistral3 does not support SDPA - switching to fast eager.
    Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.
:::

::: {.output .stream .stderr}
    Loading weights: 100%|██████████| 585/585 [00:15<00:00, 37.60it/s] 
    The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
    The tokenizer you are loading from 'unsloth/Ministral-3-14B-Instruct-2512' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
:::

::: {.output .stream .stdout}
    Loading PEFT model...
    Model loaded successfully on: cuda:0
    Fine-tuned model loaded successfully.
:::
::::::::

:::::: {.cell .code execution_count="4"}
``` python
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
embedding_model = embedding_model.to(DEVICE)
embedding_model.eval()

print("Embedding model loaded successfully.")
```

::: {.output .stream .stdout}
    Loading sentence embedding model...
:::

::: {.output .stream .stderr}
    Loading weights: 100%|██████████| 199/199 [00:00<00:00, 12672.58it/s]
:::

::: {.output .stream .stdout}
    Embedding model loaded successfully.
:::
::::::

:::: {.cell .code execution_count="5"}
``` python
# Load test dataset csv file
print("Loading test dataset...")
df = pd.read_csv(DATA_TEST)
print(f"Test dataset loaded: {len(df)} samples")
```

::: {.output .stream .stdout}
    Loading test dataset...
    Test dataset loaded: 230 samples
:::
::::

::: {.cell .markdown}
## Normalization Function
:::

::: {.cell .code execution_count="6"}
``` python
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

::: {.cell .markdown}
## Metric Functions
:::

:::: {.cell .code execution_count="7"}
``` python
# Initialize scorers
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
chencherry = SmoothingFunction()


def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score
    """
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    score = sentence_bleu(
        [ref_tokens],
        cand_tokens,
        smoothing_function=chencherry.method1
    )
    return score


def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE scores
    """
    scores = rouge_scorer_obj.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_cosine_similarity(reference, candidate):
    """
    Calculate Cosine Similarity using TF-IDF
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([reference, candidate])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0
    
def calculate_semantic_embeddings(reference, candidate):
    """
    Calculate Semantic Similarity using Sentence Embeddings
    """
    try:
        embeddings = embedding_model.encode([reference, candidate], convert_to_tensor=True)
        similarity = cosine_similarity(embeddings[0].cpu().numpy().reshape(1, -1), embeddings[1].cpu().numpy().reshape(1, -1))[0][0]
        return similarity
    except:
        return 0.0


def calculate_all_metrics(reference, candidate):
    """
    Calculate all metrics at once
    """
    bleu = calculate_bleu(reference, candidate)
    rouge_scores = calculate_rouge(reference, candidate)
    cosine = calculate_cosine_similarity(reference, candidate)
    semantic_sim = calculate_semantic_embeddings(reference, candidate)
    
    return {
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'cosine_similarity': cosine,
        'semantic_similarity': semantic_sim
    }


def compute_ci(series):
    """
    Compute mean, std, and 95% confidence interval margin
    """
    n = len(series)
    mean = series.mean()
    std = series.std(ddof=1)

    z = 1.96  # for 95% CI
    ci_margin = z * (std / np.sqrt(n))

    lower = mean - ci_margin
    upper = mean + ci_margin

    return mean, std, ci_margin, lower, upper

    
print("Metric functions ready!")
```

::: {.output .stream .stdout}
    Metric functions ready!
:::
::::

::: {.cell .markdown}
## Run Testing
:::

::::::: {.cell .code execution_count="8"}
``` python
print(f"Running normalization")

generated_results = []
bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
cosine_scores = []
semantic_scores = []

model.eval()

for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating text"):
    informal = row['informal']

    try:
        prediction = normalize_text(informal)
        generated_results.append(prediction)

        if (index + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"\n[WARNING] OOM at sample {index+1}. Clearing cache and retrying...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            prediction = normalize_text(informal, max_length=256)
            generated_results.append(prediction)
        except Exception as e:
            print(f"Failed again at sample {index+1}: {e}")
            generated_results.append("")

df['prediction'] = generated_results
print("Inference selesai. Hasil disimpan di kolom 'prediction'.")

try:
    del model
    del base_model
    del tokenizer
except NameError:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Model unloaded from memory.")


# Monitor memory usage
for index, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
    formal_ref = row['formal']
    prediction = row['prediction']

    # Calculate all metrics
    metrics = calculate_all_metrics(formal_ref, prediction)
        
    bleu_scores.append(metrics['bleu'])
    rouge1_scores.append(metrics['rouge1'])
    rouge2_scores.append(metrics['rouge2'])
    rougeL_scores.append(metrics['rougeL'])
    cosine_scores.append(metrics['cosine_similarity'])
    semantic_scores.append(metrics['semantic_similarity'])
    
```

::: {.output .stream .stdout}
    Running normalization
:::

::: {.output .stream .stderr}
    Generating text:   0%|          | 0/230 [00:00<?, ?it/s]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   0%|          | 1/230 [00:02<09:11,  2.41s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   1%|          | 2/230 [00:04<07:43,  2.03s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   1%|▏         | 3/230 [00:05<06:24,  1.69s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   2%|▏         | 4/230 [00:07<06:08,  1.63s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   2%|▏         | 5/230 [00:08<05:57,  1.59s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   3%|▎         | 6/230 [00:10<05:54,  1.58s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   3%|▎         | 7/230 [00:11<05:32,  1.49s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   3%|▎         | 8/230 [00:12<05:30,  1.49s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   4%|▍         | 9/230 [00:14<05:10,  1.41s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   4%|▍         | 10/230 [00:15<05:09,  1.41s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   5%|▍         | 11/230 [00:17<05:40,  1.55s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   5%|▌         | 12/230 [00:19<05:54,  1.63s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   6%|▌         | 13/230 [00:20<05:30,  1.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   6%|▌         | 14/230 [00:21<04:50,  1.35s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   7%|▋         | 15/230 [00:22<04:34,  1.28s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   7%|▋         | 16/230 [00:23<04:35,  1.29s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   7%|▋         | 17/230 [00:25<04:41,  1.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   8%|▊         | 18/230 [00:27<05:39,  1.60s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   8%|▊         | 19/230 [00:29<05:53,  1.67s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   9%|▊         | 20/230 [00:30<05:20,  1.53s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:   9%|▉         | 21/230 [00:31<04:41,  1.35s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  10%|▉         | 22/230 [00:32<04:31,  1.30s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  10%|█         | 23/230 [00:33<04:23,  1.27s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  10%|█         | 24/230 [00:35<04:35,  1.34s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  11%|█         | 25/230 [00:36<04:08,  1.21s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  11%|█▏        | 26/230 [00:37<04:30,  1.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  12%|█▏        | 27/230 [00:39<04:28,  1.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  12%|█▏        | 28/230 [00:41<05:16,  1.56s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  13%|█▎        | 29/230 [00:43<05:50,  1.75s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  13%|█▎        | 30/230 [00:45<05:39,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  13%|█▎        | 31/230 [00:46<05:38,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  14%|█▍        | 32/230 [00:47<05:01,  1.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  14%|█▍        | 33/230 [00:49<04:52,  1.49s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  15%|█▍        | 34/230 [00:50<04:46,  1.46s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  15%|█▌        | 35/230 [00:52<04:59,  1.53s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  16%|█▌        | 36/230 [00:53<04:16,  1.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  16%|█▌        | 37/230 [00:54<04:09,  1.29s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  17%|█▋        | 38/230 [00:55<03:58,  1.24s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  17%|█▋        | 39/230 [00:56<03:49,  1.20s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  17%|█▋        | 40/230 [00:58<04:01,  1.27s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  18%|█▊        | 41/230 [00:59<04:28,  1.42s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  18%|█▊        | 42/230 [01:02<05:20,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  19%|█▊        | 43/230 [01:03<05:08,  1.65s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  19%|█▉        | 44/230 [01:05<05:09,  1.67s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  20%|█▉        | 45/230 [01:06<04:55,  1.60s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  20%|██        | 46/230 [01:08<04:58,  1.62s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  20%|██        | 47/230 [01:09<04:46,  1.56s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  21%|██        | 48/230 [01:11<04:20,  1.43s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  21%|██▏       | 49/230 [01:12<04:14,  1.40s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  22%|██▏       | 50/230 [01:14<04:23,  1.46s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  22%|██▏       | 51/230 [01:15<04:44,  1.59s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  23%|██▎       | 52/230 [01:17<04:33,  1.54s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  23%|██▎       | 53/230 [01:18<04:25,  1.50s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  23%|██▎       | 54/230 [01:20<04:35,  1.56s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  24%|██▍       | 55/230 [01:22<05:16,  1.81s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  24%|██▍       | 56/230 [01:24<05:04,  1.75s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  25%|██▍       | 57/230 [01:26<05:10,  1.79s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  25%|██▌       | 58/230 [01:28<05:17,  1.85s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  26%|██▌       | 59/230 [01:30<05:07,  1.80s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  26%|██▌       | 60/230 [01:31<04:31,  1.59s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  27%|██▋       | 61/230 [01:32<04:34,  1.63s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  27%|██▋       | 62/230 [01:34<04:31,  1.61s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  27%|██▋       | 63/230 [01:35<04:09,  1.50s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  28%|██▊       | 64/230 [01:36<03:59,  1.44s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  28%|██▊       | 65/230 [01:40<05:58,  2.17s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  29%|██▊       | 66/230 [01:43<06:00,  2.20s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  29%|██▉       | 67/230 [01:44<05:14,  1.93s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  30%|██▉       | 68/230 [01:45<04:55,  1.83s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  30%|███       | 69/230 [01:47<04:23,  1.64s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  30%|███       | 70/230 [01:48<04:10,  1.57s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  31%|███       | 71/230 [01:49<04:00,  1.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  31%|███▏      | 72/230 [01:51<03:45,  1.43s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  32%|███▏      | 73/230 [01:52<03:50,  1.47s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  32%|███▏      | 74/230 [01:53<03:28,  1.34s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  33%|███▎      | 75/230 [01:55<03:43,  1.44s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  33%|███▎      | 76/230 [01:56<03:40,  1.43s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  33%|███▎      | 77/230 [01:58<03:41,  1.45s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  34%|███▍      | 78/230 [01:59<03:38,  1.44s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  34%|███▍      | 79/230 [02:01<03:45,  1.49s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  35%|███▍      | 80/230 [02:02<03:40,  1.47s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  35%|███▌      | 81/230 [02:04<03:45,  1.51s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  36%|███▌      | 82/230 [02:06<03:59,  1.62s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  36%|███▌      | 83/230 [02:08<04:10,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  37%|███▋      | 84/230 [02:10<04:21,  1.79s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  37%|███▋      | 85/230 [02:11<04:07,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  37%|███▋      | 86/230 [02:14<04:42,  1.96s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  38%|███▊      | 87/230 [02:16<04:37,  1.94s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  38%|███▊      | 88/230 [02:18<04:42,  1.99s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  39%|███▊      | 89/230 [02:19<04:16,  1.82s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  39%|███▉      | 90/230 [02:21<04:13,  1.81s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  40%|███▉      | 91/230 [02:23<04:19,  1.87s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  40%|████      | 92/230 [02:25<04:10,  1.82s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  40%|████      | 93/230 [02:27<04:36,  2.02s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  41%|████      | 94/230 [02:29<04:16,  1.89s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  41%|████▏     | 95/230 [02:31<04:27,  1.98s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  42%|████▏     | 96/230 [02:33<04:32,  2.03s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  42%|████▏     | 97/230 [02:35<04:32,  2.05s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  43%|████▎     | 98/230 [02:37<04:22,  1.99s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  43%|████▎     | 99/230 [02:39<04:13,  1.93s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  43%|████▎     | 100/230 [02:41<04:06,  1.90s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  44%|████▍     | 101/230 [02:42<03:53,  1.81s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  44%|████▍     | 102/230 [02:45<04:13,  1.98s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  45%|████▍     | 103/230 [02:46<03:46,  1.78s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  45%|████▌     | 104/230 [02:47<03:27,  1.65s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  46%|████▌     | 105/230 [02:49<03:38,  1.74s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  46%|████▌     | 106/230 [02:52<03:55,  1.90s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  47%|████▋     | 107/230 [02:53<03:29,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  47%|████▋     | 108/230 [02:54<03:13,  1.59s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  47%|████▋     | 109/230 [02:55<03:05,  1.53s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  48%|████▊     | 110/230 [02:57<03:16,  1.64s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  48%|████▊     | 111/230 [03:00<03:44,  1.89s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  49%|████▊     | 112/230 [03:01<03:16,  1.66s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  49%|████▉     | 113/230 [03:03<03:22,  1.73s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  50%|████▉     | 114/230 [03:04<03:06,  1.61s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  50%|█████     | 115/230 [03:06<03:11,  1.67s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  50%|█████     | 116/230 [03:07<03:01,  1.60s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  51%|█████     | 117/230 [03:09<02:50,  1.51s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  51%|█████▏    | 118/230 [03:10<02:36,  1.39s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  52%|█████▏    | 119/230 [03:11<02:19,  1.26s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  52%|█████▏    | 120/230 [03:12<02:20,  1.28s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  53%|█████▎    | 121/230 [03:13<02:11,  1.21s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  53%|█████▎    | 122/230 [03:15<02:20,  1.30s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  53%|█████▎    | 123/230 [03:16<02:13,  1.25s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  54%|█████▍    | 124/230 [03:17<02:20,  1.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  54%|█████▍    | 125/230 [03:19<02:42,  1.55s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  55%|█████▍    | 126/230 [03:21<02:49,  1.63s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  55%|█████▌    | 127/230 [03:23<03:06,  1.81s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  56%|█████▌    | 128/230 [03:25<03:01,  1.78s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  56%|█████▌    | 129/230 [03:27<03:09,  1.87s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  57%|█████▋    | 130/230 [03:29<03:18,  1.99s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  57%|█████▋    | 131/230 [03:32<03:20,  2.02s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  57%|█████▋    | 132/230 [03:34<03:27,  2.12s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  58%|█████▊    | 133/230 [03:36<03:24,  2.11s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  58%|█████▊    | 134/230 [03:39<03:41,  2.30s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  59%|█████▊    | 135/230 [03:42<03:59,  2.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  59%|█████▉    | 136/230 [03:45<04:06,  2.62s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  60%|█████▉    | 137/230 [03:47<03:59,  2.57s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  60%|██████    | 138/230 [03:50<03:56,  2.57s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  60%|██████    | 139/230 [03:51<03:25,  2.26s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  61%|██████    | 140/230 [03:54<03:28,  2.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  61%|██████▏   | 141/230 [03:57<03:46,  2.54s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  62%|██████▏   | 142/230 [03:59<03:34,  2.43s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  62%|██████▏   | 143/230 [04:02<03:55,  2.71s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  63%|██████▎   | 144/230 [04:04<03:19,  2.32s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  63%|██████▎   | 145/230 [04:05<02:52,  2.03s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  63%|██████▎   | 146/230 [04:08<03:04,  2.19s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  64%|██████▍   | 147/230 [04:10<03:16,  2.36s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  64%|██████▍   | 148/230 [04:13<03:12,  2.34s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  65%|██████▍   | 149/230 [04:15<03:03,  2.27s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  65%|██████▌   | 150/230 [04:17<02:56,  2.21s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  66%|██████▌   | 151/230 [04:19<02:51,  2.18s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  66%|██████▌   | 152/230 [04:22<03:01,  2.33s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  67%|██████▋   | 153/230 [04:24<03:06,  2.43s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  67%|██████▋   | 154/230 [04:26<02:41,  2.13s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  67%|██████▋   | 155/230 [04:27<02:23,  1.92s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  68%|██████▊   | 156/230 [04:29<02:10,  1.77s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  68%|██████▊   | 157/230 [04:31<02:20,  1.92s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  69%|██████▊   | 158/230 [04:33<02:15,  1.88s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  69%|██████▉   | 159/230 [04:34<02:09,  1.83s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  70%|██████▉   | 160/230 [04:36<02:13,  1.91s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  70%|███████   | 161/230 [04:39<02:17,  1.99s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  70%|███████   | 162/230 [04:40<02:03,  1.82s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  71%|███████   | 163/230 [04:42<02:10,  1.95s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  71%|███████▏  | 164/230 [04:44<02:11,  1.99s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  72%|███████▏  | 165/230 [04:46<02:07,  1.96s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  72%|███████▏  | 166/230 [04:48<01:59,  1.87s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  73%|███████▎  | 167/230 [04:50<02:02,  1.95s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  73%|███████▎  | 168/230 [04:52<02:09,  2.08s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  73%|███████▎  | 169/230 [04:54<01:53,  1.86s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  74%|███████▍  | 170/230 [04:56<01:50,  1.84s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  74%|███████▍  | 171/230 [04:58<01:52,  1.91s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  75%|███████▍  | 172/230 [05:00<01:56,  2.02s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  75%|███████▌  | 173/230 [05:02<01:53,  1.98s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  76%|███████▌  | 174/230 [05:04<01:50,  1.97s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  76%|███████▌  | 175/230 [05:06<01:47,  1.95s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  77%|███████▋  | 176/230 [05:08<01:49,  2.02s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  77%|███████▋  | 177/230 [05:09<01:40,  1.90s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  77%|███████▋  | 178/230 [05:11<01:30,  1.73s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  78%|███████▊  | 179/230 [05:13<01:28,  1.73s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  78%|███████▊  | 180/230 [05:14<01:29,  1.78s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  79%|███████▊  | 181/230 [05:16<01:26,  1.76s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  79%|███████▉  | 182/230 [05:18<01:27,  1.82s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  80%|███████▉  | 183/230 [05:19<01:17,  1.64s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  80%|████████  | 184/230 [05:21<01:16,  1.66s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  80%|████████  | 185/230 [05:23<01:15,  1.68s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  81%|████████  | 186/230 [05:25<01:19,  1.81s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  81%|████████▏ | 187/230 [05:26<01:15,  1.76s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  82%|████████▏ | 188/230 [05:28<01:10,  1.67s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  82%|████████▏ | 189/230 [05:29<01:05,  1.60s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  83%|████████▎ | 190/230 [05:31<00:59,  1.49s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  83%|████████▎ | 191/230 [05:32<01:00,  1.56s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  83%|████████▎ | 192/230 [05:34<01:04,  1.69s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  84%|████████▍ | 193/230 [05:36<01:00,  1.64s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  84%|████████▍ | 194/230 [05:39<01:10,  1.95s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  85%|████████▍ | 195/230 [05:40<01:03,  1.82s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  85%|████████▌ | 196/230 [05:42<00:58,  1.73s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  86%|████████▌ | 197/230 [05:43<00:53,  1.61s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  86%|████████▌ | 198/230 [05:45<00:54,  1.69s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  87%|████████▋ | 199/230 [05:46<00:50,  1.64s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  87%|████████▋ | 200/230 [05:48<00:52,  1.76s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  87%|████████▋ | 201/230 [05:50<00:48,  1.69s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  88%|████████▊ | 202/230 [05:51<00:46,  1.66s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  88%|████████▊ | 203/230 [05:53<00:47,  1.76s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  89%|████████▊ | 204/230 [05:55<00:47,  1.83s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  89%|████████▉ | 205/230 [05:57<00:44,  1.76s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  90%|████████▉ | 206/230 [05:59<00:40,  1.69s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  90%|█████████ | 207/230 [06:00<00:36,  1.58s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  90%|█████████ | 208/230 [06:01<00:33,  1.53s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  91%|█████████ | 209/230 [06:03<00:33,  1.61s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  91%|█████████▏| 210/230 [06:04<00:30,  1.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  92%|█████████▏| 211/230 [06:06<00:28,  1.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  92%|█████████▏| 212/230 [06:07<00:26,  1.46s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  93%|█████████▎| 213/230 [06:09<00:24,  1.42s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  93%|█████████▎| 214/230 [06:10<00:25,  1.57s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  93%|█████████▎| 215/230 [06:12<00:23,  1.55s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  94%|█████████▍| 216/230 [06:14<00:23,  1.68s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  94%|█████████▍| 217/230 [06:16<00:22,  1.75s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  95%|█████████▍| 218/230 [06:17<00:19,  1.61s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  95%|█████████▌| 219/230 [06:19<00:16,  1.53s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  96%|█████████▌| 220/230 [06:20<00:15,  1.52s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  96%|█████████▌| 221/230 [06:22<00:14,  1.60s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  97%|█████████▋| 222/230 [06:23<00:12,  1.58s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  97%|█████████▋| 223/230 [06:25<00:11,  1.70s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  97%|█████████▋| 224/230 [06:27<00:09,  1.59s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  98%|█████████▊| 225/230 [06:28<00:08,  1.62s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  98%|█████████▊| 226/230 [06:30<00:06,  1.53s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  99%|█████████▊| 227/230 [06:31<00:04,  1.61s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text:  99%|█████████▉| 228/230 [06:33<00:03,  1.73s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text: 100%|█████████▉| 229/230 [06:35<00:01,  1.66s/it]Both `max_new_tokens` (=128) and `max_length`(=262144) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    Generating text: 100%|██████████| 230/230 [06:36<00:00,  1.73s/it]
:::

::: {.output .stream .stdout}
    Inference selesai. Hasil disimpan di kolom 'prediction'.
    Model unloaded from memory.
:::

::: {.output .stream .stderr}
    Scoring: 100%|██████████| 230/230 [00:01<00:00, 172.62it/s]
:::
:::::::

::: {.cell .markdown}
## Results
:::

::::: {.cell .code execution_count="9"}
``` python
# results to dataframe
df['generated_result'] = generated_results
df['bleu'] = bleu_scores
df['rouge1'] = rouge1_scores
df['rouge2'] = rouge2_scores
df['rougeL'] = rougeL_scores
df['cosine_similarity'] = cosine_scores
df['semantic_similarity'] = semantic_scores

print("\n" + "="*80)
print("ALL RESULTS WITH METRICS")
print("="*80)
display(df[['informal', 'formal', 'generated_result', 'bleu', 'rouge1', 'rougeL', 'cosine_similarity', 'semantic_similarity']])

# Save to CSV
results_path = RESULTS_DIR / "normalization_results.csv"
df.to_csv(results_path, index=False)
```

::: {.output .stream .stdout}

    ================================================================================
    ALL RESULTS WITH METRICS
    ================================================================================
:::

::: {.output .display_data}
                                                  informal  \
    0                    Durasi KP lamanya berapa bulan ya   
    1                          KP tuh kredit SKS nya brpa?   
    2                cara dapetin surat pengantar KP yaapa   
    3                         dosbing KP yg nentuin siapa?   
    4                batas anggota tim KP tuh berapa orang   
    ..                                                 ...   
    225               nah kita bisa ngambil mbkm pas kapan   
    226             kalo konversi magang butuh dokumen apa   
    227  cara ngitung jumlah hari buat konversi mbkm yaapa   
    228   biar magang bisa diconvert harus brp bulan dulu?   
    229  MBKM nih buat sluruh prodi apa prodi tertentu aja   

                                                    formal  \
    0         Berapa lama waktu pelaksanaan kerja praktik?   
    1    Berapa jumlah kredit mata kuliah Kerja Praktik...   
    2    Bagaimana cara mendapatkan Surat Pengantar ker...   
    3    Siapa yang menentukan dosen pembimbing kerja p...   
    4    Berapa jumlah anggota maksimal dalam satu tim ...   
    ..                                                 ...   
    225  Pada semester berapa mahasiswa dapat mengikuti...   
    226  Apa saja dokumen pendukung yang diperlukan unt...   
    227  Bagaimana cara menghitung jumlah hari kegiatan...   
    228  Berapa durasi minimal magang agar dapat dikonv...   
    229  Apakah seluruh program studi di ITS menerapkan...   

                                          generated_result      bleu    rouge1  \
    0         Berapa lama waktu pelaksanaan kerja praktik?  1.000000  1.000000   
    1    Berapa jumlah kredit mata kuliah Kerja Praktik...  1.000000  1.000000   
    2    Bagaimana cara mendapatkan Surat Pengantar ker...  1.000000  1.000000   
    3    Siapa yang menentukan dosen pembimbing kerja p...  1.000000  1.000000   
    4    Berapa jumlah maksimal mahasiswa dalam satu ti...  0.598806  0.888889   
    ..                                                 ...       ...       ...   
    225  Kapan mahasiswa dapat mengambil mata kuliah MBKM?  0.073309  0.400000   
    226  Apa saja dokumen pendukung yang diperlukan unt...  1.000000  1.000000   
    227  Bagaimana cara menghitung jumlah hari kegiatan...  1.000000  1.000000   
    228  Berapa durasi minimal magang agar dapat dikonv...  1.000000  1.000000   
    229  Apakah seluruh program studi di ITS menerapkan...  1.000000  1.000000   

           rougeL  cosine_similarity  semantic_similarity  
    0    1.000000           1.000000             1.000000  
    1    1.000000           1.000000             1.000000  
    2    1.000000           1.000000             1.000000  
    3    1.000000           1.000000             1.000000  
    4    0.888889           0.801978             0.960352  
    ..        ...                ...                  ...  
    225  0.400000           0.253209             0.845278  
    226  1.000000           1.000000             1.000000  
    227  1.000000           1.000000             1.000000  
    228  1.000000           1.000000             1.000000  
    229  1.000000           1.000000             1.000000  

    [230 rows x 8 columns]
:::
:::::

::::: {.cell .code execution_count="10"}
``` python
print("\n" + "="*55)
print("METRICS SUMMARY WITH 95% CONFIDENCE INTERVAL")
print("="*55)

results = []

metrics_series = {
    "BLEU": df["bleu"],
    "ROUGE-1": df["rouge1"],
    "ROUGE-2": df["rouge2"],
    "ROUGE-L": df["rougeL"],
    "Cosine Similarity": df["cosine_similarity"],
    "Semantic Similarity": df["semantic_similarity"]
}

for name, series in metrics_series.items():
    mean, std, ci, lower, upper = compute_ci(series)

    print(f"{name:<25}: {mean:.4f} ± {ci:.4f}")

    results.append({
        "Metric": name,
        "Mean": mean,
        "Std": std,
        "CI ± (95%)": ci,
        "CI Lower": lower,
        "CI Upper": upper,
        "Min": series.min(),
        "Max": series.max()
    })

print("="*55)

# -------------------------------------------------
# Detail Statistics
# -------------------------------------------------
summary_stats = pd.DataFrame(results)

# Optional column for paper-ready format
summary_stats["Mean ± CI95"] = (
    summary_stats["Mean"].round(4).astype(str)
    + " ± "
    + summary_stats["CI ± (95%)"].round(4).astype(str)
)

print("\nDetailed Statistics:")
display(summary_stats)
```

::: {.output .stream .stdout}

    =======================================================
    METRICS SUMMARY WITH 95% CONFIDENCE INTERVAL
    =======================================================
    BLEU                     : 0.7907 ± 0.0415
    ROUGE-1                  : 0.8981 ± 0.0227
    ROUGE-2                  : 0.8285 ± 0.0350
    ROUGE-L                  : 0.8931 ± 0.0235
    Cosine Similarity        : 0.8558 ± 0.0299
    Semantic Similarity      : 0.9673 ± 0.0079
    =======================================================

    Detailed Statistics:
:::

::: {.output .display_data}
                    Metric      Mean       Std  CI ± (95%)  CI Lower  CI Upper  \
    0                 BLEU  0.790684  0.320878    0.041470  0.749214  0.832154   
    1              ROUGE-1  0.898058  0.175498    0.022681  0.875377  0.920739   
    2              ROUGE-2  0.828464  0.270809    0.034999  0.793466  0.863463   
    3              ROUGE-L  0.893139  0.181799    0.023495  0.869643  0.916634   
    4    Cosine Similarity  0.855790  0.231737    0.029949  0.825841  0.885740   
    5  Semantic Similarity  0.967255  0.061438    0.007940  0.959315  0.975195   

            Min  Max      Mean ± CI95  
    0  0.023350  1.0  0.7907 ± 0.0415  
    1  0.266667  1.0  0.8981 ± 0.0227  
    2  0.000000  1.0   0.8285 ± 0.035  
    3  0.266667  1.0  0.8931 ± 0.0235  
    4  0.159764  1.0  0.8558 ± 0.0299  
    5  0.708303  1.0  0.9673 ± 0.0079  
:::
:::::

:::: {.cell .code execution_count="11"}
``` python
# save results to csv file
print(f"Saving results to {results_path}...")
df.to_csv(results_path, index=False)
print("Results saved successfully.")
```

::: {.output .stream .stdout}
    Saving results to /home/teaching-factory/train_mistral/results/normalization_results.csv...
    Results saved successfully.
:::
::::

::: {.cell .markdown}
## Calculate Statistics
:::

::::: {.cell .code execution_count="12"}
``` python
print("\n" + "="*50)
print("METRICS SUMMARY")
print("="*50)
print(f"Average BLEU:             {df['bleu'].mean():.4f}")
print(f"Average ROUGE-1:          {df['rouge1'].mean():.4f}")
print(f"Average ROUGE-2:          {df['rouge2'].mean():.4f}")
print(f"Average ROUGE-L:          {df['rougeL'].mean():.4f}")
print(f"Average Cosine Similarity: {df['cosine_similarity'].mean():.4f}")
print(f"Average Semantic Similarity: {df['semantic_similarity'].mean():.4f}")
print("="*50)

# Create summary dataframe
summary_stats = pd.DataFrame({
    'Metric': ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Cosine Similarity', 'Semantic Similarity'],
    'Mean': [
        df['bleu'].mean(),
        df['rouge1'].mean(),
        df['rouge2'].mean(),
        df['rougeL'].mean(),
        df['cosine_similarity'].mean(),
        df['semantic_similarity'].mean()
    ],
    'Std': [
        df['bleu'].std(),
        df['rouge1'].std(),
        df['rouge2'].std(),
        df['rougeL'].std(),
        df['cosine_similarity'].std(),
        df['semantic_similarity'].std()
    ],
    'Min': [
        df['bleu'].min(),
        df['rouge1'].min(),
        df['rouge2'].min(),
        df['rougeL'].min(),
        df['cosine_similarity'].min(),
        df['semantic_similarity'].min()
    ],
    'Max': [
        df['bleu'].max(),
        df['rouge1'].max(),
        df['rouge2'].max(),
        df['rougeL'].max(),
        df['cosine_similarity'].max(),
        df['semantic_similarity'].max()
    ]
})

print("\nDetailed Statistics:")
display(summary_stats)
```

::: {.output .stream .stdout}

    ==================================================
    METRICS SUMMARY
    ==================================================
    Average BLEU:             0.7907
    Average ROUGE-1:          0.8981
    Average ROUGE-2:          0.8285
    Average ROUGE-L:          0.8931
    Average Cosine Similarity: 0.8558
    Average Semantic Similarity: 0.9673
    ==================================================

    Detailed Statistics:
:::

::: {.output .display_data}
                    Metric      Mean       Std       Min  Max
    0                 BLEU  0.790684  0.320878  0.023350  1.0
    1              ROUGE-1  0.898058  0.175498  0.266667  1.0
    2              ROUGE-2  0.828464  0.270809  0.000000  1.0
    3              ROUGE-L  0.893139  0.181799  0.266667  1.0
    4    Cosine Similarity  0.855790  0.231737  0.159764  1.0
    5  Semantic Similarity  0.967255  0.061438  0.708303  1.0
:::
:::::

::: {.cell .markdown}
## Visualize Results
:::

:::::: {.cell .code execution_count="13"}
``` python
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'cosine_similarity', 'semantic_similarity']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# 1. Bar plot of average scores
ax1 = axes[0, 0]
means = [df[m].mean() for m in metrics_to_plot]
bars = ax1.bar(range(len(metrics_to_plot)), means, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(metrics_to_plot)))
ax1.set_xticklabels([m.upper().replace('_', ' ') for m in metrics_to_plot], rotation=45, ha='right')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Average Scores by Metric', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Box plot
ax2 = axes[0, 1]
bp = ax2.boxplot([df[m] for m in metrics_to_plot], 
                   labels=[m.upper().replace('_', '\n') for m in metrics_to_plot],
                   patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Score Distributions', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Score progression
ax3 = axes[1, 0]
x = range(len(df))
ax3.plot(x, df['bleu'], 'o-', label='BLEU', color=colors[0], alpha=0.7)
ax3.plot(x, df['rougeL'], 's-', label='ROUGE-L', color=colors[3], alpha=0.7)
ax3.plot(x, df['cosine_similarity'], '^-', label='Cosine Sim', color=colors[4], alpha=0.7)
ax3.plot(x, df['semantic_similarity'], 'd-', label='Semantic Sim', color=colors[5], alpha=0.7)
ax3.axhline(y=df['bleu'].mean(), color=colors[0], linestyle='--', alpha=0.5, label='BLEU Mean')
ax3.axhline(y=df['rougeL'].mean(), color=colors[3], linestyle='--', alpha=0.5, label='ROUGE-L Mean')
ax3.axhline(y=df['cosine_similarity'].mean(), color=colors[4], linestyle='--', alpha=0.5, label='Cosine Sim Mean')
ax3.axhline(y=df['semantic_similarity'].mean(), color=colors[5], linestyle='--', alpha=0.5, label='Semantic Sim Mean')
ax3.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Score Progression Across Samples', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# 4. Correlation heatmap
ax4 = axes[1, 1]
corr_matrix = df[metrics_to_plot].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax4, cbar_kws={'shrink': 0.8},
            xticklabels=[m.upper().replace('_', ' ') for m in metrics_to_plot],
            yticklabels=[m.upper().replace('_', ' ') for m in metrics_to_plot])
ax4.set_title('Metric Correlation Heatmap', fontsize=14, fontweight='bold')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.setp(ax4.yaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plot_file = RESULTS_DIR / 'finetuned_model_evaluation.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n Visualization saved to: {plot_file}")
```

::: {.output .stream .stderr}
    /tmp/ipykernel_3411724/3346359823.py:30: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
      bp = ax2.boxplot([df[m] for m in metrics_to_plot],
:::

::: {.output .display_data}
![](34b8f015d3227fe4e38739ad075eb011f46db7e2.png)
:::

::: {.output .stream .stdout}

     Visualization saved to: /home/teaching-factory/train_mistral/results/finetuned_model_evaluation.png
:::
::::::

::: {.cell .code execution_count="14"}
``` python
# release memory
del df
del embedding_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```
:::

::: {.cell .markdown}
## Save Results
:::

::: {.cell .code}
``` python
# Save detailed results
results_csv = RESULTS_DIR / "finetuned_test_results.csv"
df.to_csv(results_csv, index=False)
print(f"Detailed results saved to: {results_csv}")

# Save summary statistics
summary_csv = RESULTS_DIR / "finetuned_summary_stats.csv"
summary_stats.to_csv(summary_csv, index=False)
print(f"Summary statistics saved to: {summary_csv}")

# Save as JSON
summary_json = RESULTS_DIR / "finetuned_summary.json"
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump({
        'model_name': BASE_MODEL_NAME,
        'lora_adapter': str(MODEL_DIR),
        'n_samples': len(df),
        'metrics': {
            'bleu': {
                'mean': float(df['bleu'].mean()),
                'std': float(df['bleu'].std()),
                'min': float(df['bleu'].min()),
                'max': float(df['bleu'].max())
            },
            'rouge1': {
                'mean': float(df['rouge1'].mean()),
                'std': float(df['rouge1'].std()),
                'min': float(df['rouge1'].min()),
                'max': float(df['rouge1'].max())
            },
            'rouge2': {
                'mean': float(df['rouge2'].mean()),
                'std': float(df['rouge2'].std()),
                'min': float(df['rouge2'].min()),
                'max': float(df['rouge2'].max())
            },
            'rougeL': {
                'mean': float(df['rougeL'].mean()),
                'std': float(df['rougeL'].std()),
                'min': float(df['rougeL'].min()),
                'max': float(df['rougeL'].max())
            },
            'cosine_similarity': {
                'mean': float(df['cosine_similarity'].mean()),
                'std': float(df['cosine_similarity'].std()),
                'min': float(df['cosine_similarity'].min()),
                'max': float(df['cosine_similarity'].max())
            },
            'exact_match': {
                'accuracy': float(df['exact_match'].mean()),
                'count': int(df['exact_match'].sum()),
                'total': len(df)
            }
        }
    }, f, indent=2, ensure_ascii=False)
print(f"Summary JSON saved to: {summary_json}")

print(f"\nResults saved in: {RESULTS_DIR}")
print(f"  - {results_csv.name}")
print(f"  - {summary_csv.name}")
print(f"  - {summary_json.name}")
print(f"  - finetuned_model_evaluation.png")
```
:::
