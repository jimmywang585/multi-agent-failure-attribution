from __future__ import annotations
import argparse, yaml, numpy as np
from .utils import set_seed, agent_distribution
from .data import load_json_or_jsonl
from .prompts import build_step_rating_prompt
from .models import build_generator, run_rating
from .bt import pairwise_pref_from_probs, aggregate_pairwise, fit_bt_from_Q
from .rewards import total_reward
from .training import ReportHead, md_update, grad_logp_from_reward
from .metrics import accuracy_agent, accuracy_step, accuracy_step_tolerance

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    set_seed(cfg.get("seed", 42))
    logs = load_json_or_jsonl(args.data)

    model_names = cfg["models"]
    gens, toks = [], []
    print(f"Loading {len(model_names)} models...")
    for name in model_names:
        gen, tok = build_generator(name)
        gens.append(gen); toks.append(tok)

    heads = [ReportHead(temp=1.5) for _ in model_names]
    w = cfg["reward_weights"]; lr = cfg["mirror_descent"]["lr"]

    all_metrics = []

    for epoch in range(cfg["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{cfg['epochs']} ===")
        for row in logs:
            query = row["query"]; steps = row["steps"]
            agents = [s["agent"] for s in steps]
            prompt = build_step_rating_prompt(query, steps)

            raw_scores_list = []; pks = []; Qs = []; Pks = []
            for k,(gen,head) in enumerate(zip(gens,heads)):
                scores = run_rating(gen, toks[k], prompt,
                                    max_new_tokens=cfg["gen"]["max_new_tokens"],
                                    temperature=cfg["gen"]["temperature"],
                                    top_p=cfg["gen"]["top_p"])
                T = len(steps)
                if len(scores) < T: scores = scores + [0.0]*(T - len(scores))
                scores = scores[:T]
                raw_scores_list.append(scores)
                p_k = head.forward(scores); pks.append(p_k)
                Qs.append(pairwise_pref_from_probs(p_k))
                Pk, uniq_agents = agent_distribution(p_k, agents); Pks.append(Pk)

            Q_bar = aggregate_pairwise(Qs)
            r_bt, p_grp = fit_bt_from_Q(Q_bar, max_iter=cfg["bt"]["max_iter"], tol=cfg["bt"]["tol"], l2_reg=cfg["bt"]["l2_reg"])
            P_grp, uniq_agents = agent_distribution(p_grp, agents)
            import numpy as np
            T = len(steps)
            P_bt = 1/(1 + np.exp(-((r_bt[:,None] - r_bt[None,:]))))

            for k in range(len(model_names)):
                Rk = total_reward(Qs[k], P_bt, pks[k], p_grp, Pks[k], P_grp, w)
                grad_logp = grad_logp_from_reward(pks[k], p_grp)
                heads[k] = md_update(heads[k], raw_scores_list[k], grad_logp, lr=lr)

            metrics = {}
            if "gold_step_index" in row:
                metrics["acc_step"] = accuracy_step(p_grp, row["gold_step_index"])
                metrics["acc_step_tol1"] = accuracy_step_tolerance(p_grp, row["gold_step_index"], tol=1)
            if "gold_agent" in row:
                metrics["acc_agent"] = accuracy_agent(P_grp, uniq_agents, row["gold_agent"])
            all_metrics.append(metrics)
            if metrics:
                print(f"[{row.get('log_id','-')}] step_acc={metrics.get('acc_step')} agent_acc={metrics.get('acc_agent')}")

    if all_metrics:
        keys = sorted({k for m in all_metrics for k in m.keys()})
        import numpy as np
        avg = {k: float(np.mean([m.get(k, np.nan) for m in all_metrics if k in m])) for k in keys}
        print("\n== Averages ==")
        for k,v in avg.items():
            print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()