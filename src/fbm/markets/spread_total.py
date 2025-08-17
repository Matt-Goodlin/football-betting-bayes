from math import erf, sqrt

_SQRT2 = sqrt(2.0)

def _norm_cdf(z: float) -> float:
    # Standard normal CDF via error function
    return 0.5 * (1.0 + erf(z / _SQRT2))

def prob_cover(diff_mean: float, spread_line: float, sigma_diff: float = 13.0) -> float:
    """
    Probability the HOME side covers a spread_line (e.g., -2.5 means home -2.5).
    We assume point differential D = Home - Away ~ Normal(diff_mean, sigma_diff).
    P(cover) = P(D > spread_line) = 1 - CDF((spread_line - diff_mean)/sigma)
    Push probability is ignored (small for .5 lines).
    """
    z = (spread_line - diff_mean) / sigma_diff
    return 1.0 - _norm_cdf(z)

def prob_over(total_mean: float, total_line: float, sigma_total: float = 10.0) -> float:
    """
    Probability the game goes over a total_line.
    We assume T = Home + Away ~ Normal(total_mean, sigma_total).
    P(over) = P(T > total_line) = 1 - CDF((total_line - total_mean)/sigma)
    """
    z = (total_line - total_mean) / sigma_total
    return 1.0 - _norm_cdf(z)
