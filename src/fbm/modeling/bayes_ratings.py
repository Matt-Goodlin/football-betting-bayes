from typing import Dict, List, Tuple
import numpy as np

def _teams_index(results: List[dict]) -> Dict[str, int]:
    teams = set()
    for g in results:
        teams.add(g["home_team"]); teams.add(g["away_team"])
    return {t: i for i, t in enumerate(sorted(teams))}

def fit_bayes_ratings(
    results: List[dict],
    *,
    hfa_points: float = 2.0,
    l2_lambda: float = 4.0,           # Gaussian prior precision (larger = stronger shrink)
    start_ratings: Dict[str, float] = None,
    enforce_sum_zero: bool = True,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Bayesian (ridge/MAP) fit of team 'net strength' from point differentials:
      y = (home_pts - away_pts - hfa_points) ≈ (R_home - R_away) + ε,  ε ~ N(0, σ^2)
    Prior: R ~ N(0, τ^2 I) -> ridge penalty λ = σ^2 / τ^2 (we use λ=l2_lambda directly).

    Returns: (ratings_dict, team_index_map)
    """
    if not results:
        return {}, {}

    idx = _teams_index(results)
    n_teams = len(idx)
    rows = []
    ys = []

    for g in results:
        h = idx[g["home_team"]]; a = idx[g["away_team"]]
        y = float(g["home_pts"]) - float(g["away_pts"]) - float(hfa_points)
        row = np.zeros(n_teams, dtype=float)
        row[h] = 1.0; row[a] = -1.0
        rows.append(row); ys.append(y)

    X = np.vstack(rows)                 # (G, T)
    y = np.asarray(ys, dtype=float)     # (G,)

    lam = float(l2_lambda)
    A = X.T @ X + lam * np.eye(n_teams)
    b = X.T @ y
    beta = np.linalg.solve(A, b)        # unconstrained

    if enforce_sum_zero:
        beta = beta - beta.mean()

    ratings = {team: float(beta[j]) for team, j in idx.items()}
    return ratings, idx
