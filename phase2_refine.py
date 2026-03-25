import torch
import torch.nn.functional as F
import pickle
import math

from pathlib import Path
from config import DEVICE, CACHE_DIR


def repeat_kv(x, num_reps):
    if num_reps == 1:
        return x
    B, num_kv_heads, seq, head_dim = x.shape
    # Interleave each KV head num_reps times.
    x = x.unsqueeze(2).expand(B, num_kv_heads, num_reps, seq, head_dim)
    return x.reshape(B, num_kv_heads * num_reps, seq, head_dim)

# My Algorithm
def cross_chunk_attention(H_i, H_j, attn_module, layernorm, num_heads, num_kv_heads, head_dim):
    hidden_size = num_heads * head_dim

    H_i_norm = layernorm(H_i)   
    H_j_norm = layernorm(H_j)   
    
    Q = attn_module.q_proj(H_i_norm)   
    K = attn_module.k_proj(H_j_norm)   
    V = attn_module.v_proj(H_j_norm)   

    seq_i = H_i.shape[1]
    seq_j = H_j.shape[1]
    
    Q = Q.view(1, seq_i, num_heads, head_dim).transpose(1, 2)       
    K = K.view(1, seq_j, num_kv_heads, head_dim).transpose(1, 2)    
    V = V.view(1, seq_j, num_kv_heads, head_dim).transpose(1, 2)    
    
    num_reps = num_heads // num_kv_heads
    K = repeat_kv(K, num_reps)   
    V = repeat_kv(V, num_reps)   

    scale = math.sqrt(head_dim)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale   
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, V)   

    attn_output = attn_output.transpose(1, 2).contiguous().view(1, seq_i, hidden_size)

    attn_output = attn_module.o_proj(attn_output)  

    return attn_output



def run_phase2(all_hidden_states, model, document_path):
    doc_stem = Path(document_path).stem
    cache_file = CACHE_DIR / f"phase2_{doc_stem}.pkl"
    if cache_file.exists():
        print("Phase 2 cache found")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["refined_hidden_states"]

    cfg = model.config
    num_heads     = cfg.num_attention_heads
    num_kv_heads  = cfg.num_key_value_heads
    head_dim      = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    # Retrieve the last Transformer decoder layer's attention module and layernorm.
    last_layer   = model.model.layers[-1]
    attn_module  = last_layer.self_attn
    layernorm    = last_layer.input_layernorm

    N = len(all_hidden_states)
    refined_hidden_states = []

    for i in range(N):
        print(f"  Refining chunk {i + 1}/{N} ...")

        H_i = all_hidden_states[i].to(DEVICE)   

        if N == 1:
            # Only one chunk 
            refined_hidden_states.append(H_i.cpu())
            continue

        cross_sum = torch.zeros_like(H_i)

        for j in range(N):
            if j == i:
                continue
            H_j = all_hidden_states[j].to(DEVICE)   

            with torch.no_grad():
                contrib = cross_chunk_attention(
                    H_i, H_j, attn_module, layernorm,
                    num_heads, num_kv_heads, head_dim
                )   

            cross_sum = cross_sum + contrib

        H_i_star = H_i + cross_sum / (N - 1)   
        refined_hidden_states.append(H_i_star.cpu())

    
    print("Saving Phase 2 outputs to cache ...")
    with open(cache_file, "wb") as f:
        pickle.dump({"refined_hidden_states": refined_hidden_states}, f)
    print("Phase 2 complete.")
    return refined_hidden_states
