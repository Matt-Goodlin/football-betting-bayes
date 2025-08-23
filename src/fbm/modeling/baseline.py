from math import erf, sqrt
from typing import Dict, Optional

_SQRT2 = sqrt(2.0)

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / _SQRT2))

class BaselineModel:
    """
    Super-simple baseline:
      - Team rating dict (points), default 0 for unseen teams.
      - HFA in points, default 2.0.
      - sigma_diff: stdev of point differential (e.g., 13).
      - sigma_total: stdev of totals (e.g., 10).
    """

    def __init__(
        self,
        ratings: Optional[Dict[str, float]] = None,
        hfa_points: float = 2.0,
        sigma_diff: float = 13.0,
        sigma_total: float = 10.0,
    ):
        self.ratings = ratings or {}
        self.hfa = hfa_points
        self.sd_diff = sigma_diff
        self.sd_total = sigma_total

    def _rating(self, team: str) -> float:
        return self.ratings.get(team, 0.0)

    def mean_diff(self, home_team: str, away_team: str) -> float:
        """Expected point differential: Home - Away."""
        return (self._rating(home_team) - self._rating(away_team)) + self.hfa

    def win_prob_home(self, home_team: str, away_team: str) -> float:
        """P(Home wins) under Normal(diff)."""
        mu = self.mean_diff(home_team, away_team)
        z = (0.0 - mu) / self.sd_diff
        return 1.0 - _norm_cdf(z)

    def cover_prob_home(self, home_team: str, away_team: str, spread_line: float) -> float:
        """P(Home - Away > spread_line)."""
        mu = self.mean_diff(home_team, away_team)
        z = (spread_line - mu) / self.sd_diff
        return 1.0 - _norm_cdf(z)

    def over_prob(self, total_line: float, league_total_mean: float = 45.0) -> float:
        """P(Total > line) with a fixed league total mean."""
        z = (total_line - league_total_mean) / self.sd_total
        return 1.0 - _norm_cdf(z)