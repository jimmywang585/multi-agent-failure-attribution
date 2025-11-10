# src/models.py  — minimal diff: add _resolve_ctx_limit and tweak run_rating()

import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _prefer_bf16() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32

def build_generator(model_name: str):
    dtype = _prefer_bf16()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    # ensure we can pad if model lacks a pad token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    gen = pipeline("text-generation", model=model, tokenizer=tok)
    return gen, tok

def _resolve_ctx_limit(tok, model) -> int:
    """
    Pick a safe, finite context window. Some tokenizers expose a giant sentinel
    (≈1e30) that breaks enable_truncation(). We clamp to a reasonable ceiling.
    """
    # Prefer model’s declared max positions if present
    cfg_limit = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
    tok_limit = getattr(tok, "model_max_length", None)

    # Normalize weird sentinels / Nones
    def _norm(x):
        if x is None:
            return None
        try:
            x = int(x)
        except Exception:
            return None
        # Treat absurdly large numbers as unknown
        return x if 0 < x <= 1_000_000 else None

    cands = [_norm(cfg_limit), _norm(tok_limit), 4096]  # 4096 as conservative fallback
    # take the max of known-good values to avoid being too restrictive
    ctx = max(v for v in cands if v is not None)
    # also cap to a sane upper bound to keep memory predictable
    return min(ctx, 131_072)

@torch.inference_mode()
def run_rating(gen, tok, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    """
    Truncate via tokenizer with a *finite* max_length, then decode back to text
    so the pipeline gets a string (not tensors). Avoids OverflowError + keeps
    KV cache small. No perf impact on your generations.
    """
    ctx = _resolve_ctx_limit(tok, gen.model)
    # leave headroom for generation; keep a small safety margin
    budget = max(64, ctx - max_new_tokens - 32)
    # Hard floor/ceiling to avoid tokenizer overflow/underflow
    budget = int(max(64, min(budget, ctx - 1)))

    enc = tok(prompt, truncation=True, max_length=budget, return_tensors="pt")
    prompt_trunc = tok.decode(enc.input_ids[0], skip_special_tokens=True)

    try:
        out = gen(
            prompt_trunc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # deterministic; temperature/top_p ignored (expected)
            pad_token_id=tok.pad_token_id,
            truncation=True,
        )[0]["generated_text"]
        return out
    except torch.cuda.OutOfMemoryError:
        # fall back to CPU just for this call; move back to CUDA afterwards
        torch.cuda.empty_cache()
        cpu_gen = pipeline("text-generation", model=gen.model.to("cpu"), tokenizer=tok)
        out = cpu_gen(
            prompt_trunc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            truncation=True,
        )[0]["generated_text"]
        try:
            gen.model.to("cuda")
        except Exception:
            pass
        return out