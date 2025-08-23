import math
from typing import Dict, Optional

class BaselineModel:
    """
    Minimal baseline model used by the CLI:
      - team 'ratings' on a point scale (higher = stronger)
      - hfa_points: home-field advantage added to home rating
      - sigma_diff: stdev (points) of score differential
      - sigma_total: stdev (points) of total score
    """

    def __init__(
        self,
        ratings: Optional[Dict[str, float]] = None,
        hfa_points: float = 2.0,
        sigma_diff: float = 13.0,
        sigma_total: float = 10.0,
    ) -> None:
        self.ratings: Dict[str, float] = ratings or {}
        self.hfa_points: float = float(hfa_points)
        self.sigma_diff: float = float(sigma_diff)
        self.sigma_total: float = float(sigma_total)

    def rating(self, team: str) -> float:
        return float(self.ratings.get(team, 0.0))

    @staticmethod
    def _phi(x: float) -> float:
        """Standard normal CDF Φ(x)."""
        # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def win_prob_home(self, home_team: str, away_team: str) -> float:
        """
        P(home_score - away_score > 0) when diff ~ Normal(mean, sigma_diff)
        mean = (rating_home - rating_away + hfa_points)
        """
        mean = self.rating(home_team) - self.rating(away_team) + self.hfa_points
        if self.sigma_diff <= 0:
            # Degenerate; treat as coin flip if no variance
            return 0.5
        z = mean / self.sigma_diff
        # P(diff > 0) = Φ(z)
        return self._phi(z)

    def __repr__(self) -> str:
        return (
            f"BaselineModel(hfa_points={self.hfa_points}, "
            f"sigma_diff={self.sigma_diff}, sigma_total={self.sigma_total}, "
            f"ratings={len(self.ratings)} teams)"
        )