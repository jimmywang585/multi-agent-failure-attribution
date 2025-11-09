from __future__ import annotations
import random, numpy as np

def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x, temp: float = 1.0):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    x = x / max(1e-12, temp)
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def kl_div(p, q, eps: float = 1e-12) -> float:
    import numpy as np
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def entropy(p, eps: float = 1e-12) -> float:
    import numpy as np
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def normalize_scores_to_probs(scores, temp: float, bias: float = 0.0):
    import numpy as np
    arr = np.array(scores, dtype=np.float64) + bias
    return softmax(arr, temp=temp)

def agent_distribution(p_steps, agents):
    import numpy as np
    uniq = []
    for a in agents:
        if a not in uniq:
            uniq.append(a)
    idx = {a:i for i,a in enumerate(uniq)}
    A = np.zeros(len(uniq), dtype=np.float64)
    for t,a in enumerate(agents):
        A[idx[a]] += p_steps[t]
    if A.sum() > 0:
        A = A / A.sum()
    return A, uniq

def argmax_idx(p):
    import numpy as np
    return int(np.argmax(p))