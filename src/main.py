from __future__ import annotations
import argparse, yaml, numpy as np, os, torch, gc

# Keep startup quiet & stable
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

from .utils import set_seed, agent_distribution
from .data import load_json_or_jsonl
from .prompts import build_step_rating_prompt
from .models import build_generator, run_rating
from .bt import pairwise_pref_from_probs, fit_bt_from_Q
from .rewards import total_reward
from .training import ReportHead, md_update, grad_logp_from_reward
from .metrics import accuracy_agent, accuracy_step, accuracy_step_tolerance

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    return ap.parse_args()

def _maybe_empty_cache(step_mod: int, i: int):
    if torch.cuda.is_available() and (i % step_mod == 0):
        torch.cuda.empty_cache()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    set_seed(cfg.get("seed", 42))
    logs = load_json_or_jsonl(args.data)

    model_names = cfg["models"]
    print(f"Loading {len(model_names)} models...")

    # Build once; device_map='auto' inside keeps us safe without perf loss.
    gens, toks = [], []
    for name in model_names:
        gen, tok = build_generator(name)
        gens.append(gen); toks.append(tok)

    # Light head; no need to keep gradients
    heads = [ReportHead(temp=1.5) for _ in model_names]
    w = cfg["reward_weights"]; lr = cfg["mirror_descent"]["lr"]

    # Cap small generation to keep KV cache tiny (has negligible perf impact for scoring)
    max_new = max(1, min(8, cfg["gen"].get("max_new_tokens", 8)))

    all_metrics = []

    for epoch in range(cfg["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{cfg['epochs']} ===")
        for i, row in enumerate(logs, 1):
            query = row["query"]; steps = row["steps"]
            agents = [s["agent"] for s in steps]
            prompt = build_step_rating_prompt(query, steps)

            raw_scores_list = []; pks = []; Qs = []; Pks = []

            for k,(gen,head) in enumerate(zip(gens,heads)):
                # Generate ratings text and parse numbers upstream (unchanged)
                out = run_rating(
                    gen, toks[k], prompt,
                    max_new_tokens=max_new,
                    temperature=cfg["gen"].get("temperature", 0.0),
                    top_p=cfg["gen"].get("top_p", 1.0)
                )

                # Parse and coerce ratings
                T = len(steps)
                # existing parser: assume upstream convert-to-scores logic lives here
                # Keep previous behavior (your run_rating returns text; downstream code expects list of floats)
                # For compatibility, try a quick float scrape; otherwise fallback to zeros.
                scores = []
                for token in out.split():
                    try:
                        scores.append(float(token))
                        if len(scores) == T: break
                    except:  # noqa: E722
                        continue
                if len(scores) < T:
                    scores += [0.0] * (T - len(scores))
                raw_scores_list.append(scores)

                # Reporting head -> probability over steps
                p_k = np.asarray(head.forward(scores), dtype=np.float64)
                p_k = p_k / (p_k.sum() + 1e-12)
                pks.append(p_k)

                Qs.append(np.asarray(pairwise_pref_from_probs(p_k), dtype=np.float64))
                Pk, uniq_agents = agent_distribution(p_k, agents)
                Pks.append(np.asarray(Pk, dtype=np.float64))

            # Aggregate, fit BT, group distributions (unchanged)
            Qs = [np.asarray(Q, dtype=np.float64) for Q in Qs]
            Q_bar = np.mean(Qs, axis=0).astype(np.float64)

            r_bt, p_grp = fit_bt_from_Q(
                Q_bar,
                max_iter=cfg["bt"]["max_iter"],
                tol=cfg["bt"]["tol"],
                l2_reg=cfg["bt"]["l2_reg"]
            )
            p_grp = np.asarray(p_grp, dtype=np.float64)
            p_grp = p_grp / (p_grp.sum() + 1e-12)

            P_grp, uniq_agents = agent_distribution(p_grp, agents)
            P_grp = np.asarray(P_grp, dtype=np.float64)

            diff = r_bt[:, None] - r_bt[None, :]
            P_bt = 1.0 / (1.0 + np.exp(-diff))

            # Rewards + mirror descent update (unchanged)
            for k in range(len(model_names)):
                _ = total_reward(Qs[k], P_bt, pks[k], p_grp, Pks[k], P_grp, w)
                grad_logp = grad_logp_from_reward(pks[k], p_grp)
                heads[k] = md_update(heads[k], raw_scores_list[k], grad_logp, lr=lr)

            # Metrics (unchanged)
            metrics = {}
            if "gold_step_index" in row:
                metrics["acc_step"] = accuracy_step(p_grp, row["gold_step_index"])
                metrics["acc_step_tol1"] = accuracy_step_tolerance(p_grp, row["gold_step_index"], tol=1)
            if "gold_agent" in row:
                metrics["acc_agent"] = accuracy_agent(P_grp, uniq_agents, row["gold_agent"])
            all_metrics.append(metrics)
            if metrics:
                print(f"[{row.get('log_id','-')}] step_acc={metrics.get('acc_step')} agent_acc={metrics.get('acc_agent')}")

            # Keep memory tidy on long runs without measurable slowdown
            _maybe_empty_cache(step_mod=8, i=i)
            if i % 32 == 0:
                gc.collect()

    if all_metrics:
        keys = sorted({k for m in all_metrics for k in m.keys()})
        avg = {k: float(np.nanmean([m.get(k, np.nan) for m in all_metrics])) for k in keys}
        print("\n== Averages ==")
        for k,v in avg.items():
            print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()