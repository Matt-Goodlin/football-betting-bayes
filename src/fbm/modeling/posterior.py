"""
Posterior predictive utilities under a Normal model.

If X ~ Normal(mean, sigma), then:
- P(X > a) = 1 - Phi((a - mean)/sigma)
- P(X < a) = Phi((a - mean)/sigma)
- For spreads: a = spread_line (home - away)
- For totals:  a = total_line (home + away)

Includes both closed-form CDF helpers and Monte Carlo simulators.
Compatible with Python 3.9.
"""
from math import erf, sqrt
from typing import Optional
import numpy as np

SQRT2 = sqrt(2.0)

def _phi(z: float) -> float:
    """Standard Normal CDF using erf."""
    return 0.5 * (1.0 + erf(z / SQRT2))

def prob_over_normal(mean: float, sigma: float, line: float) -> float:
    """P(X > line) where X ~ N(mean, sigma^2)."""
    if sigma <= 0:
        # Degenerate: all mass at mean
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
    P(Home covers) under Normal model for (Home - Away) ~ N(mean_diff, sigma_diff^2).
    For example, with home_spread = -2.5, computes P(Home - Away > -2.5).
    """
    return prob_over_normal(mean_diff, sigma_diff, home_spread)

def prob_total_over(mean_total: float, sigma_total: float, total_line: float) -> float:
    """
    P(Over total) under Normal model for (Home + Away) ~ N(mean_total, sigma_total^2).
    """
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
    """
    Monte Carlo estimate of P(home margin > spread) where margin ~ N(mean_diff, sigma^2).
    """
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
    """
    Monte Carlo estimate of P(total points > line) where total ~ N(total_mean, sigma_total^2).
    """
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=total_mean, scale=sigma_total, size=n)
    return float(np.mean(sims > line))