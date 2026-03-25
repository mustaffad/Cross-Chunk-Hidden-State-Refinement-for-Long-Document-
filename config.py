from pathlib import Path
import torch

MODEL_DIR_4B  = "/home/mustafa-dursunoglu/models/gemma-2-2b-it"
MODEL_FILE_4B = "gemma-2-2b-it-Q4_K_M.gguf"
MODEL_DIR_27B  = "/home/mustafa-dursunoglu/models/gemma-3-27b"
MODEL_FILE_27B = "google_gemma-3-27b-it-Q4_K_M.gguf"

def compute_chunk_params(num_tokens: int, max_ctx: int) -> dict:
    TARGET_CHUNKS       = 25    
    TARGET_QUERY_TOKENS = 5000  
    OVERLAP_RATIO       = 0.125 


    stride     = max(128, num_tokens // TARGET_CHUNKS)
    chunk_size = int(stride / (1 - OVERLAP_RATIO))
    chunk_size = max(128, min(chunk_size, max_ctx // 4))
    overlap    = max(16, chunk_size // 8)


    top_k = max(3, min(10, TARGET_QUERY_TOKENS // chunk_size))

    return {"chunk_size": chunk_size, "overlap": overlap, "top_k": top_k}



CHUNK_SIZE = 1024
OVERLAP    = 128
CROSS_ATTN_LAYER = -1  
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
