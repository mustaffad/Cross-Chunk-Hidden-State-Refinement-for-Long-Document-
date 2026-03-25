import torch
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_DIR_4B, MODEL_FILE_4B, DTYPE, DEVICE, CACHE_DIR, compute_chunk_params



def load_model_and_tokenizer(model_dir=None):
    mdir = Path(model_dir) if model_dir else Path(MODEL_DIR_4B)
    gguf_files = list(mdir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {mdir}")
    mfile = gguf_files[0].name

    print(f"Loading tokenizer from {mfile} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(mdir), gguf_file=mfile)

    print(f"Loading model from {mfile} (dequantizing GGUF → bfloat16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir),
        gguf_file=mfile,
        torch_dtype=DTYPE,
        device_map="cuda:0" if DEVICE == "cuda" else "cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def chunk_tokens(token_ids, chunk_size, overlap):
    chunks = []
    chunk_starts = []
    stride = chunk_size - overlap
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunks.append(token_ids[start:end])
        chunk_starts.append(start)
        if end == len(token_ids):
            break
        start += stride
    return chunks, chunk_starts

def process_chunk(model, input_ids_tensor, position_offset):
    seq_len = input_ids_tensor.shape[1]

    position_ids = torch.arange(
        position_offset, position_offset + seq_len,
        dtype=torch.long, device=input_ids_tensor.device
    ).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_tensor,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=True,
        )

    # Last Transformer layer hidden states.
    hidden_states = outputs.hidden_states[-1].cpu() 

    # Move every KV tensor to CPU immediately to free VRAM.
    kv_cache = tuple(
        (layer_kv[0].cpu(), layer_kv[1].cpu())
        for layer_kv in outputs.past_key_values
    )
    return hidden_states, kv_cache

def run_phase1(document_path, model, tokenizer):
    doc_stem = Path(document_path).stem
    cache_file = CACHE_DIR / f"phase1_{doc_stem}.pkl"
    if cache_file.exists():
        print("Phase 1 cache found")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return (
            data["all_hidden_states"],
            data["all_kv_caches"],
            data["chunks"],
            data["chunk_starts"],
            data["top_k"],
        )

    # Tokenize 
    text = Path(document_path).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"Document tokenized: {len(token_ids)} tokens.")

    max_ctx = getattr(model.config, "max_position_embeddings", 131072)
    params  = compute_chunk_params(len(token_ids), max_ctx)
    chunk_size, overlap, top_k = params["chunk_size"], params["overlap"], params["top_k"]
    print(f"  Auto chunk params: chunk_size={chunk_size}, overlap={overlap}, top_k={top_k}")

    chunks, chunk_starts = chunk_tokens(token_ids, chunk_size, overlap)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")

    all_hidden_states = []
    all_kv_caches = []

    # position_offset tracks where each chunk starts in the document.
    # first token has the correct absolute position.
    position_offset = 0

    for idx, chunk in enumerate(chunks):
        print(f"  Processing chunk {idx + 1}/{len(chunks)} "
              f"(tokens {chunk_starts[idx]}–{chunk_starts[idx] + len(chunk) - 1}) ...")

        input_ids_tensor = torch.tensor([chunk], dtype=torch.long, device=DEVICE)
        hidden_states, kv_cache = process_chunk(model, input_ids_tensor, position_offset)

        all_hidden_states.append(hidden_states)
        all_kv_caches.append(kv_cache)

        position_offset += len(chunk) - overlap

    # Persist to cache.
    print("Saving Phase 1 outputs to cache ...")
    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "all_hidden_states": all_hidden_states,
                "all_kv_caches": all_kv_caches,
                "chunks": chunks,
                "chunk_starts": chunk_starts,
                "top_k": top_k,
            },
            f,
        )
    print("Phase 1 complete.")
    return all_hidden_states, all_kv_caches, chunks, chunk_starts, top_k
