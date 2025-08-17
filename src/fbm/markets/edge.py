from typing import Tuple

def ev_and_edge(model_prob: float, fair_prob: float, odds_decimal: float) -> Tuple[float, float]:
    """
    Returns:
      ev_per_dollar: expected profit per $1 wager (e.g., 0.05 = +5 cents per $1)
      edge_pct: model_prob - fair_prob (absolute percentage points, e.g., +0.04 = +4%)
    """
    if not (0.0 < model_prob < 1.0 and 0.0 < fair_prob < 1.0):
        raise ValueError("model_prob and fair_prob must be in (0,1)")
    if odds_decimal <= 1.0:
        raise ValueError("odds_decimal must be > 1.0")

    b = odds_decimal - 1.0  # net payout ratio
    q = 1.0 - model_prob
    ev_per_dollar = model_prob * b - q
    edge_pct = model_prob - fair_prob
    return ev_per_dollar, edge_pct
