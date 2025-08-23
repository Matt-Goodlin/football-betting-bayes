from typing import Dict, List

def _win_prob_from_rating_diff(diff_pts: float, scale_pts: float = 13.0) -> float:
    """
    Convert rating differential (pts) -> win probability via Normal approx.
    P = CDF(mu/sigma) with sigma≈scale_pts; we use a logistic-like proxy for speed:
    """
    # Smooth, cheap approximation of normal CDF using logistic:
    # You can tweak scale_pts; larger = flatter.
    import math
    return 1.0 / (1.0 + math.exp(-(diff_pts / scale_pts) * 1.7))

def fit_elo_ratings(
    results: List[dict],
    *,
    start_ratings: Dict[str, float] = None,
    k: float = 20.0,
    hfa_points: float = 2.0,
    iters: int = 2,
    scale_pts: float = 13.0,
) -> Dict[str, float]:
    """
    Very small Elo-like fitter on final scores.
    - start_ratings: optional starting dict; unseen teams start at 0
    - k: update size (pts) per game error
    - hfa_points: home-field advantage added to home rating
    - iters: number of passes over the dataset
    - scale_pts: scaling used to map rating diff to win prob

    Update rule (per game):
      diff = (R_home + HFA) - R_away
      p_home = win_prob(diff)
      actual = 1 if home_pts > away_pts else 0.5 if tie else 0
      err = actual - p_home
      R_home += k * err
      R_away -= k * err
    """
    ratings = dict(start_ratings or {})
    def r(team: str) -> float: return ratings.get(team, 0.0)
    def add(team: str, delta: float):
        ratings[team] = r(team) + delta

    for _ in range(max(1, iters)):
        for g in results:
            home = g["home_team"]; away = g["away_team"]
            hp = g["home_pts"]; ap = g["away_pts"]
            # Game outcome to [0,1]
            actual = 0.5
            if hp > ap: actual = 1.0
            elif hp < ap: actual = 0.0

            diff = (r(home) + hfa_points) - r(away)
            p_home = _win_prob_from_rating_diff(diff, scale_pts=scale_pts)
            err = actual - p_home
            add(home, k * err)
            add(away, -k * err)

    return ratings
