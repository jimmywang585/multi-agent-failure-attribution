from __future__ import annotations
import numpy as np

def pairwise_pref_from_probs(p):
    T = len(p)
    Q = np.zeros((T,T), dtype=np.float64)
    for s in range(T):
        for t in range(T):
            if s == t:
                Q[s,t] = 0.5
            else:
                denom = p[s] + p[t]
                Q[s,t] = p[s] / denom if denom > 0 else 0.5
    return Q

def aggregate_pairwise(Qs, weights=None):
    if weights is None:
        weights = [1.0] * len(Qs)
    W = sum(weights)
    agg = np.zeros_like(Qs[0])
    for w,Q in zip(weights, Qs):
        agg += w * Q
    return agg / max(W, 1e-12)

def fit_bt_from_Q(Q, max_iter=200, tol=1e-6, l2_reg=1e-4):
    T = Q.shape[0]
    r = np.zeros(T, dtype=np.float64)
    lr = 0.1
    last_ll = None
    for it in range(max_iter):
        P = 1 / (1 + np.exp(-(r[:,None] - r[None,:])))
        grad = np.zeros(T, dtype=np.float64)
        for s in range(T):
            for t in range(T):
                if s == t: continue
                grad[s] += Q[s,t]*(1 - P[s,t]) - (1 - Q[s,t])*(P[s,t])
        grad -= l2_reg * r
        r = r + lr * grad
        r = r - r.mean()
        ll = 0.0
        for s in range(T):
            for t in range(s+1, T):
                ll += Q[s,t]*np.log(P[s,t] + 1e-12) + (1-Q[s,t])*np.log(1-P[s,t] + 1e-12)
        if last_ll is not None and abs(ll - last_ll) < tol:
            break
        last_ll = ll
    p_grp = np.exp(r - r.max()); p_grp = p_grp / p_grp.sum()
    return r, p_grp