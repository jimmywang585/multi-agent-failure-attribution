from __future__ import annotations
import numpy as np
from .utils import normalize_scores_to_probs
class ReportHead:
    def __init__(self, temp: float = 1.0, bias: float = 0.0, min_temp: float = 0.25, max_temp: float = 4.0):
        self.temp = float(temp)
        self.bias = float(bias)
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
    def forward(self, raw_scores):
        return normalize_scores_to_probs(raw_scores, temp=self.temp, bias=self.bias)
    def step(self, grad_temp: float, grad_bias: float, lr: float):
        self.temp -= lr * grad_temp
        self.bias += lr * grad_bias
        self.temp = float(np.clip(self.temp, self.min_temp, self.max_temp))

def md_update(report_head: ReportHead, raw_scores, grad_logp, lr: float):
    import numpy as np
    p = report_head.forward(raw_scores)
    T = len(p)
    z = (np.array(raw_scores, dtype=np.float64) + report_head.bias)/max(1e-12, report_head.temp)
    dlogp_dz = np.eye(T) - p[None,:]
    dR_dz = dlogp_dz.T @ grad_logp
    dz_dtemp = -(np.array(raw_scores)+report_head.bias)/max(1e-12, report_head.temp**2)
    dz_dbias =  1.0/max(1e-12, report_head.temp)
    grad_temp = float((dR_dz * dz_dtemp).sum())
    grad_bias = float((dR_dz * dz_dbias).sum())
    report_head.step(grad_temp, grad_bias, lr)
    return report_head

def grad_logp_from_reward(p_k, p_grp):
    return p_k