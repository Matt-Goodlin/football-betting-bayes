from typing import Dict, List

def _win_prob_from_rating_diff(diff_pts: float, scale_pts: float = 13.0) -> float:
    """
    Convert rating differential (pts) -> win probability via a smooth logistic
    proxy for a Normal CDF. Larger scale_pts => flatter curve (more conservative).
    """
    import math
    return 1.0 / (1.0 + math.exp(-(diff_pts / scale_pts) * 1.7))

def _mov_multiplier(margin_pts: float, *, scale_pts: float = 7.0, cap: float = 2.0) -> float:
    """
    Margin-of-victory multiplier for Elo updates.
    - Uses a gentle log growth: 1 + log1p(|margin| / scale)
    - Capped by `cap` to avoid blowout inflation.
    """
    import math
    m = abs(margin_pts)
    return min(cap, 1.0 + math.log1p(m / max(1e-9, scale_pts)))

def fit_elo_ratings(
    results: List[dict],
    *,
    start_ratings: Dict[str, float] = None,
    k: float = 20.0,
    hfa_points: float = 2.0,
    iters: int = 2,
    scale_pts: float = 13.0,        # mapping from rating diff to win prob
    use_mov: bool = True,           # enable margin-of-victory weighting
    mov_scale_pts: float = 7.0,     # MOV scale (pts) in the log growth
    mov_cap: float = 2.0,           # maximum MOV multiplier
) -> Dict[str, float]:
    """
    Very small Elo-like fitter on final scores.

    Update per game:
      diff = (R_home + HFA) - R_away
      p_home = win_prob(diff)
      actual = 1 if home_pts > away_pts else 0.5 if tie else 0
      err = actual - p_home
      mult = MOV-multiplier (if enabled)
      R_home += k * mult * err
      R_away -= k * mult * err
    """
    ratings = dict(start_ratings or {})

    def r(team: str) -> float:
        return ratings.get(team, 0.0)

    def add(team: str, delta: float):
        ratings[team] = r(team) + delta

    for _ in range(max(1, iters)):
        for g in results:
            home = g["home_team"]; away = g["away_team"]
            hp = g["home_pts"]; ap = g["away_pts"]

            # Outcome to [0,1]
            if hp > ap:
                actual = 1.0
            elif hp < ap:
                actual = 0.0
            else:
                actual = 0.5

            diff = (r(home) + hfa_points) - r(away)
            p_home = _win_prob_from_rating_diff(diff, scale_pts=scale_pts)
            err = actual - p_home

            margin = float(hp - ap)
            mult = _mov_multiplier(margin, scale_pts=mov_scale_pts, cap=mov_cap) if use_mov else 1.0

            add(home, k * mult * err)
            add(away, -k * mult * err)

    return ratings

def normalize_ratings(
    ratings: Dict[str, float],
    *,
    target_std: float = 3.0,
) -> Dict[str, float]:
    """
    Re-center ratings to mean 0 and shrink std to target_std (default 3 pts).
    Keeps early-season probabilities in a reasonable range.
    """
    import math
    if not ratings:
        return ratings
    vals = list(ratings.values())
    mean = sum(vals) / len(vals)
    centered = {t: v - mean for t, v in ratings.items()}

    vals2 = list(centered.values())
    var = sum(v * v for v in vals2) / len(vals2)
    std = math.sqrt(var) if var > 0 else 0.0

    if std <= 1e-9:
        return centered
    scale = target_std / std
    return {t: v * scale for t, v in centered.items()}