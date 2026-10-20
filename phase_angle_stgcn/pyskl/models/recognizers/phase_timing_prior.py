"""
phase_timing_prior.py

Standalone implementation of the circular cycle-position timing prior,
independent of the model/backbone/training loop.

Formula:
    r = (segment_midpoint - cycle_start) / (cycle_end - cycle_start)
    mu, sigma = empirically-derived per-phase center/spread of r (see
        compute_cycle_r.py) -- NOT assumed equal-quarter centers.
    d(a, b) = min(|a-b|, 1-|a-b|)                      (circular distance)
    q_k = softmax(-d(r, mu_k)^2 / (2 sigma_k^2))       (soft timing prior)
    u = normalized entropy of the phase head's own softmax output
    w = u ** 3
    p*_k = (1 - w) * p_k + w * q_k
    y_hat = argmax_k p*_k
"""

import math
import torch
#determined from running computer_cycle_r.py 
DEFAULT_MU = torch.tensor([0.0898, 0.2056, 0.3115, 0.6952])
DEFAULT_SIGMA = torch.tensor([0.0328, 0.0663, 0.0726, 0.0389])


def normalized_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    k = probs.shape[-1]
    probs = probs.clamp(min=eps)
    ent = -(probs * probs.log()).sum(dim=-1)
    return ent / math.log(k)


def circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = (a - b).abs()
    return torch.minimum(diff, 1.0 - diff)


def timing_prior(r: torch.Tensor, mu: torch.Tensor = DEFAULT_MU, sigma=DEFAULT_SIGMA) -> torch.Tensor:
    mu = mu.to(r.device)
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(r.device)
    d = circular_distance(r.unsqueeze(-1), mu)
    logits = -(d ** 2) / (2 * sigma ** 2)
    return torch.softmax(logits, dim=-1)


def apply_timing_prior(phase_probs: torch.Tensor,
                        r: torch.Tensor,
                        mu: torch.Tensor = DEFAULT_MU,
                        sigma=DEFAULT_SIGMA,
                        weight_power: float = 3.0):
    q = timing_prior(r, mu, sigma)
    u = normalized_entropy(phase_probs)
    w = u ** weight_power
    p_star = (1 - w.unsqueeze(-1)) * phase_probs + w.unsqueeze(-1) * q
    y_hat = p_star.argmax(dim=-1)
    return p_star, y_hat, u, w