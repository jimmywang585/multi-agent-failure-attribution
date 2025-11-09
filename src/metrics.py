from __future__ import annotations
import numpy as np

def accuracy_agent(P, agents, gold_agent):
    if gold_agent is None: return None
    pred = agents[int(P.argmax())]
    return 1.0 if pred == gold_agent else 0.0

def accuracy_step(p, gold_step_index):
    if gold_step_index is None: return None
    return 1.0 if int(p.argmax()) == int(gold_step_index) else 0.0

def accuracy_step_tolerance(p, gold_step_index, tol: int = 1):
    if gold_step_index is None: return None
    pred = int(p.argmax())
    return 1.0 if abs(pred - int(gold_step_index)) <= tol else 0.0