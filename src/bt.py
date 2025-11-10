from __future__ import annotations
import numpy as np

def pairwise_pref_from_probs(p_step):
    p = np.asarray(p_step, dtype=np.float64)
    P = p[:, None] / (p[:, None] + p[None, :] + 1e-12)
    np.fill_diagonal(P, 0.5)  # neutral, not 0; we ignore the diagonal anyway
    return P

def aggregate_pairwise(Qs, weights=None):
    Qs = [np.asarray(Q, dtype=np.float64) for Q in Qs]
    if weights is None:
        weights = [1.0] * len(Qs)
    agg = np.zeros_like(Qs[0], dtype=np.float64)
    for w, Q in zip(weights, Qs):
        agg += float(w) * Q
    return agg / (sum(weights) + 1e-12)

def fit_bt_from_Q(Q_bar, max_iter=200, tol=1e-6, l2_reg=1e-4):
    import numpy as np
    eps = 1e-9

    # Float, valid range, enforce antisymmetry: Q_ji = 1 - Q_ij (diag=0.5)
    Q = np.asarray(Q_bar, dtype=np.float64)
    Q = np.clip(Q, eps, 1.0 - eps)
    Q = 0.5 * (Q + (1.0 - Q.T))
    np.fill_diagonal(Q, 0.5)

    T = Q.shape[0]
    r = np.zeros(T, dtype=np.float64)

    for _ in range(int(max_iter)):
        diff = r[:, None] - r[None, :]
        # logistic with safe range
        diff = np.clip(diff, -40.0, 40.0)
        P = 1.0 / (1.0 + np.exp(-diff))
        np.fill_diagonal(P, 0.5)

        # Gradient: wins minus expected wins, both directions, with L2
        # Here weights are all ones; swap in a weight/count matrix if you have one.
        W = np.ones_like(Q, dtype=np.float64)
        grad = ((Q - P) * W).sum(axis=1) - ((Q.T - (1.0 - P)) * W.T).sum(axis=1) - float(l2_reg) * r

        # Diagonal Hessian approximation for safe Newton step
        H_diag = ((P * (1.0 - P)) * (W + W.T)).sum(axis=1) + float(l2_reg) + 1e-8

        step = grad / H_diag
        r_new = r + step

        # Fix the gauge (identifiability) and check convergence
        r_new -= r_new.mean()
        if np.linalg.norm(r_new - r, ord=np.inf) < float(tol):
            r = r_new
            break
        r = r_new

    # Convert abilities to a step distribution
    z = r - r.max()
    s = np.exp(np.clip(z, -60.0, 60.0))
    p_grp = s / (s.sum() + eps)
    return r.astype(np.float64), p_grp.astype(np.float64)   