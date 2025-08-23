"""
Posterior predictive utilities under a Normal model.

Closed-form CDF helpers + Monte Carlo simulators + MC confidence intervals.
Python 3.9 compatible.
"""
from math import erf, sqrt
from typing import Optional, Tuple
import numpy as np

SQRT2 = sqrt(2.0)

def _phi(z: float) -> float:
    """Standard Normal CDF using erf."""
    return 0.5 * (1.0 + erf(z / SQRT2))

# ---------------------------
# Closed-form Normal helpers
# ---------------------------

def prob_over_normal(mean: float, sigma: float, line: float) -> float:
    """P(X > line) where X ~ N(mean, sigma^2)."""
    if sigma <= 0:
        return 1.0 if mean > line else 0.0 if mean < line else 0.5
    z = (line - mean) / sigma
    return 1.0 - _phi(z)

def prob_under_normal(mean: float, sigma: float, line: float) -> float:
    """P(X < line) where X ~ N(mean, sigma^2)."""
    if sigma <= 0:
        return 1.0 if mean < line else 0.0 if mean > line else 0.5
    z = (line - mean) / sigma
    return _phi(z)

def prob_cover_spread(mean_diff: float, sigma_diff: float, home_spread: float) -> float:
    """
    P(Home covers) when margin = Home - Away ~ N(mean_diff, sigma_diff^2).
    E.g., home_spread = -2.5 => P(margin > -2.5).
    """
    return prob_over_normal(mean_diff, sigma_diff, home_spread)

def prob_total_over(mean_total: float, sigma_total: float, total_line: float) -> float:
    """P(Over total) when total ~ N(mean_total, sigma_total^2)."""
    return prob_over_normal(mean_total, sigma_total, total_line)

# ---------------------------
# Monte Carlo posterior sims
# ---------------------------

def simulate_cover_spread(
    mean_diff: float,
    sigma: float,
    spread: float,
    n: int = 10000,
    seed: Optional[int] = None
) -> float:
    """Monte Carlo estimate of P(home margin > spread)."""
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=mean_diff, scale=sigma, size=n)
    return float(np.mean(sims > spread))

def simulate_total_over(
    total_mean: float,
    sigma_total: float,
    line: float,
    n: int = 10000,
    seed: Optional[int] = None
) -> float:
    """Monte Carlo estimate of P(total points > line)."""
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=total_mean, scale=sigma_total, size=n)
    return float(np.mean(sims > line))

# ---------------------------
# MC confidence intervals
# ---------------------------

def mc_ci_normal(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Normal-approx 1 - alpha CI (default ~95%) for a Monte Carlo probability.
    Clips to [0,1]. For large n (>=1000) this is tight and cheap.

    p_hat: Monte Carlo estimate in [0,1]
    n:     number of simulations
    z:     1.96 => ~95% (two-sided)
    """
    p = max(0.0, min(1.0, p_hat))
    if n <= 0:
        return (p, p)
    se = (p * (1.0 - p) / max(1, n)) ** 0.5
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return (lo, hi)