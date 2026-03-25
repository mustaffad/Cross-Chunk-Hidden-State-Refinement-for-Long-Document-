import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from phase1_chunk import load_model_and_tokenizer, run_phase1
from phase2_refine import run_phase2
from config import DEVICE


def _select_top_k_chunks(model, tokenizer, question, all_hidden_states, refined_hidden_states, k):
    q_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.model(q_ids, output_hidden_states=True)
        q_emb = out.hidden_states[-1][:, -1, :]
    q_emb = F.normalize(q_emb.float(), dim=-1)

    similarities = []
    for H, H_star in zip(all_hidden_states, refined_hidden_states):
        emb_h     = F.normalize(H.to(DEVICE).float().mean(dim=1),      dim=-1)
        emb_hstar = F.normalize(H_star.to(DEVICE).float().mean(dim=1), dim=-1)
        sim = ((q_emb * emb_h).sum() + (q_emb * emb_hstar).sum()).item() / 2
        similarities.append(sim)

    top_indices = sorted(
        sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    )
    return top_indices


def answer_question(model, tokenizer, chunks, all_hidden_states, refined_hidden_states, question, top_k):
    top_indices = _select_top_k_chunks(
        model, tokenizer, question, all_hidden_states, refined_hidden_states, k=top_k
    )
    selected_text = "\n\n".join(
        tokenizer.decode(chunks[i], skip_special_tokens=True) for i in top_indices
    )
    content = (
        f"Here are relevant sections of a document:\n\n{selected_text}\n\n"
        f"Answer the following question based on the document above. "
        f"If the answer is not in the provided text, say so.\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": content}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"].to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=300,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Long-document Q&A with Cross-Chunk Hidden State Refinement"
    )
    parser.add_argument(
        "--document", required=True,
        help="Path to the plain-text document to ask questions about",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the Gemma 2B model directory (must contain a .gguf file)",
    )
    args = parser.parse_args()

    document_path = Path(args.document)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    print("\n" + "=" * 60)
    print("  Cross-Chunk Hidden State Refinement — Q&A")
    print("=" * 60)
    print(f"  Document : {document_path}")
    print(f"  Model    : {args.model}")
    print("=" * 60 + "\n")

    print("[Setup] Loading model ...")
    model, tokenizer = load_model_and_tokenizer(model_dir=args.model)

    print("\n[Phase 1] Encoding document chunks ...")
    all_hidden_states, all_kv_caches, chunks, chunk_starts, top_k = run_phase1(
        document_path, model, tokenizer
    )
    print(f"  {len(chunks)} chunks processed.")

    print("\n[Phase 2] Cross-chunk attention refinement ...")
    refined_hidden_states = run_phase2(all_hidden_states, model, document_path)
    print(f"  {len(refined_hidden_states)} chunks refined.")

    print("\n" + "=" * 60)
    print("  Ready! Ask anything about the document.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        answer = answer_question(
            model, tokenizer, chunks,
            all_hidden_states, refined_hidden_states,
            question, top_k,
        )
        print(answer)
        print()


if __name__ == "__main__":
    main()
