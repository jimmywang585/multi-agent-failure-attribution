from __future__ import annotations
import numpy as np
from .utils import entropy

def pairwise_bt_alignment(Q_k, P_bt):
    T = Q_k.shape[0]
    loss = 0.0
    for s in range(T):
        for t in range(s+1, T):
            q = Q_k[s,t]; p = P_bt[s,t]
            loss += q*np.log(p + 1e-12) + (1-q)*np.log(1-p + 1e-12)
    return loss

def distribution_alignment(p_k, p_grp):
    return float(np.sum(p_k * np.log(p_grp + 1e-12)))

def agent_alignment(Pk, Pgrp):
    return float(np.sum(Pk * np.log(Pgrp + 1e-12)))

def total_reward(Q_k, P_bt, p_k, p_grp, Pk, Pgrp, weights):
    T = len(p_k)
    Z = T*(T-1)/2.0
    r_pair = pairwise_bt_alignment(Q_k, P_bt)/max(Z,1e-12)
    r_step = distribution_alignment(p_k, p_grp)
    r_agent = agent_alignment(Pk, Pgrp)
    r_ent = weights.get("entropy_step", 0.0) * entropy(p_k) + weights.get("entropy_agent", 0.0) * entropy(Pk)
    return (weights.get("beta_pair",1.0)*r_pair
            + weights.get("beta_step",1.0)*r_step
            + weights.get("beta_agent",0.0)*r_agent
            + r_ent)