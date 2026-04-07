# Cross-Chunk Hidden State Refinement for Long-Document Q&A

A training-free pipeline that enables small language models (Gemma 2B) to answer questions over documents that exceed their context window, without fine-tuning or an external embedding model.

## How It Works

1. **Phase 1**: The document is split into overlapping chunks. Each chunk is encoded independently to produce hidden states and KV-caches.
2. **Phase 2**: Cross-chunk attention refinement: each chunk's hidden states are enriched with attention contributions from every other chunk using the model's own frozen weights (no RoPE, no training).
3. **Query**: At question time, the top-K most relevant chunks are retrieved via cosine similarity over an ensemble of raw and refined hidden states, then fed to the model for generation.

Phases 1 and 2 run once and are cached. Subsequent queries are answered instantly from the cache.

---

## Requirements

### Python
Python 3.10 or higher is recommended.

### Libraries

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `torch >= 2.2.0`
- `transformers >= 4.40.0`
- `accelerate >= 0.28.0`
- `sentencepiece`

A CUDA-capable GPU is strongly recommended. CPU inference works but will be slow.

### Model - Gemma 2B IT (GGUF)

This project uses **Gemma 2 2B Instruct** in GGUF quantized format (`Q4_K_M`).

#### Step 1: Install the Hugging Face CLI

```bash
pip install huggingface_hub
```

#### Step 2: Accept the Gemma licence

Visit [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) and accept the model licence (requires a free Hugging Face account).

#### Step 3: Download the model files

```bash
# Download tokenizer and config files
huggingface-cli download google/gemma-2-2b-it \
    --include "tokenizer*" "config.json" "special_tokens_map.json" \
    --local-dir ./gemma-2-2b-it

# Download the GGUF quantised weights
huggingface-cli download bartowski/gemma-2-2b-it-GGUF \
    --include "gemma-2-2b-it-Q4_K_M.gguf" \
    --local-dir ./gemma-2-2b-it
```

After these two commands, `./gemma-2-2b-it` should contain both the tokenizer files and the `.gguf` weight file.

> **Alternative:** Download manually from
> [https://huggingface.co/bartowski/gemma-2-2b-it-GGUF](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF)
> and place the `.gguf` file alongside the tokenizer files in the same folder.

---

## Usage

```bash
python model.py --document /path/to/your/document.txt --model /path/to/gemma-2-2b-it
```

**Arguments:**

| Argument | Description |
|---|---|
| `--document` | Path to the plain-text `.txt` document you want to query |
| `--model` | Path to the directory containing the Gemma 2B tokenizer files and `.gguf` weight file |

**Example:**

```bash
python model.py --document ./my_document.txt --model ./gemma-2-2b-it
```

The first run encodes the document and saves a cache. All subsequent runs on the same document load from cache and go straight to the Q&A prompt.

### Interactive Q&A

Once the pipeline is ready, you will see:

```
======================================================
  Ready! Ask anything about the document.
  Type 'quit' or 'exit' to stop.
======================================================

You:
```

Type any natural-language question and press Enter. Type `quit` or `exit` (or press Ctrl-C) to stop.

---

## Project Structure

```
model.py          # Entry point -> CLI + interactive Q&A loop
phase1_chunk.py   # Phase 1: per-chunk encoding
phase2_refine.py  # Phase 2: cross-chunk attention refinement
config.py         # Chunk size, overlap, dtype, device settings
requirements.txt  # Python dependencies
```

---

## Citation

Built as an undergraduate research project (COMP 4900A). If you use this project in your work, please make sure to cite it properly.
Model: [Gemma 2](https://arxiv.org/abs/2408.00118) by Google DeepMind.
