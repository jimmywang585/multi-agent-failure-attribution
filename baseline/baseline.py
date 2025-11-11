#!/usr/bin/env python3
"""
Baseline evaluator for Multi-Agent Failure Attribution Simulation.

Runs each of the same 5 models individually (no BT aggregation)
on the Who&When dataset you already prepared under `data/{train,val,test}.jsonl`.

Each model runs independently and reports its accuracy metrics;
the script averages results across models to produce a baseline summary.

Usage:
    python tools/baseline_eval.py
"""

import argparse, os, sys, yaml, json, re, subprocess, textwrap
from pathlib import Path
from datetime import datetime

# Regex for parsing printed metrics from src.main output
RX_METRIC = re.compile(
    r"^acc_agent:\s*([\d.]+)\s*$|^acc_step:\s*([\d.]+)\s*$|^acc_step_tol1:\s*([\d.]+)\s*$"
)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_metrics_from_text(text):
    acc = {"acc_agent": None, "acc_step": None, "acc_step_tol1": None}
    for line in text.splitlines():
        m = RX_METRIC.search(line.strip())
        if not m:
            continue
        if m.group(1):
            acc["acc_agent"] = float(m.group(1))
        if m.group(2):
            acc["acc_step"] = float(m.group(2))
        if m.group(3):
            acc["acc_step_tol1"] = float(m.group(3))
    return acc

def short_model_name(m):
    if isinstance(m, dict):
        return (m.get("name") or m.get("id") or "model").split("/")[-1]
    return str(m).split("/")[-1]

def run_one_model(single_cfg_path, data_path, epochs, log_path):
    cmd = [
        sys.executable, "-m", "src.main",
        "--config", str(single_cfg_path),   # <-- cast to str
        "--data",   str(data_path),         # <-- cast to str
        "--epochs", str(epochs),
    ]
    print(f"Running: {' '.join(cmd)}")
    with open(str(log_path), "w", encoding="utf-8") as lf:  # safe to cast too
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        all_out = []
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
            all_out.append(line)
        proc.wait()
        out_text = "".join(all_out)
    return out_text, proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="baseline/baseline.yaml", help="Path to baseline.yaml")
    args = ap.parse_args()

    ctrl = load_yaml(args.cfg)
    data_split = ctrl.get("data_split", "train")
    base_config = ctrl.get("base_config", "config.yaml")
    epochs = int(ctrl.get("epochs", 1))
    out_dir = ctrl.get("out_dir", "runs/baseline")

    # Use the dataset that already exists
    data_path = f"data/{data_split}.jsonl"
    if not Path(data_path).exists():
        sys.exit(f"❌ Missing dataset file: {data_path}")

    # Prepare output folders
    run_root = Path(out_dir)
    cfg_dir = run_root / "configs"
    log_dir = run_root / "logs"
    ensure_dir(cfg_dir)
    ensure_dir(log_dir)

    base = load_yaml(base_config)
    models = base.get("models", [])
    if not models:
        sys.exit("❌ ERROR: base config.yaml has no 'models' list.")

    print(f"""
== Baseline Evaluation (no BT aggregation) ==
Dataset: {data_path}
Models: {len(models)} from {base_config}
Epochs: {epochs}
Logs and configs -> {run_root}
""")

    per_model = []
    for i, m in enumerate(models):
        c = dict(base)
        c["models"] = [m]  # one model only
        name = short_model_name(m)
        cfg_path = cfg_dir / f"single_{i}_{name}.yaml"
        save_yaml(c, cfg_path)

        log_path = log_dir / f"{i}_{name}.log"
        print(f"\n== Running model {i+1}/{len(models)}: {name} ==")
        out_text, code = run_one_model(cfg_path, data_path, epochs, log_path)
        if code != 0:
            print(f"⚠️ WARNING: {name} exited with code {code}")
        acc = parse_metrics_from_text(out_text)
        acc["model"] = name
        per_model.append(acc)

    valid = [m for m in per_model if all(v is not None for k,v in m.items() if k.startswith("acc_"))]
    def mean(key):
        vals = [m[key] for m in valid if key in m]
        return round(sum(vals)/len(vals), 3) if vals else 0.0

    summary = {
        "num_models": len(valid),
        "mean_acc_agent": mean("acc_agent"),
        "mean_acc_step": mean("acc_step"),
        "mean_acc_step_tol1": mean("acc_step_tol1"),
        "per_model": per_model,
        "data": data_path,
        "epochs": epochs,
        "base_config": base_config,
    }

    ensure_dir(run_root)
    with open(run_root / f"summary_{data_split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n== Baseline Summary (No BT Aggregation) ==")
    for row in per_model:
        print(f"- {row['model']}: step={row['acc_step']}, agent={row['acc_agent']}, tol1={row['acc_step_tol1']}")
    print(textwrap.dedent(f"""
        == Mean over {summary['num_models']} models ==
        mean_acc_step      = {summary['mean_acc_step']}
        mean_acc_step_tol1 = {summary['mean_acc_step_tol1']}
        mean_acc_agent     = {summary['mean_acc_agent']}
    """))

if __name__ == "__main__":
    main()
