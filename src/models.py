from __future__ import annotations
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def build_generator(model_name: str, device_map="auto", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=dtype)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device_map=device_map)
    return gen, tok

def run_rating(gen, tok, prompt: str, max_new_tokens=256, temperature=0.0, top_p=1.0):
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature, top_p=top_p)[0]["generated_text"]
    m = re.findall(r"\{.*?\}", out, flags=re.S)
    if not m:
        m2 = re.findall(r"\[(.*?)\]", out, flags=re.S)
        if not m2:
            raise ValueError("Could not parse JSON scores from model output.")
        nums = re.findall(r"-?\d+\.?\d*", m2[-1])
        return [max(0, min(100, float(x))) for x in nums]
    try:
        js = json.loads(m[-1])
        scores = js.get("scores", None)
        if scores is None:
            if isinstance(js, list):
                return [max(0, min(100, float(x))) for x in js]
            raise ValueError("No 'scores' key in JSON")
        return [max(0, min(100, float(x))) for x in scores]
    except Exception:
        nums = re.findall(r"-?\d+\.?\d*", m[-1])
        return [max(0, min(100, float(x))) for x in nums]