from typing import Tuple

def american_to_decimal(american: int) -> float:
    """
    +150 -> 2.50,  -120 -> 1.8333...
    """
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 1.0 + (american / 100.0)
    else:
        return 1.0 + (100.0 / abs(american))

def implied_prob_from_american(american: int) -> float:
    """
    +150 -> 100/(150+100)=0.4,  -120 -> 120/(120+100)=0.54545...
    """
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)

def remove_vig_two_way(p_a: float, p_b: float) -> Tuple[float, float]:
    """
    Normalize two implied probabilities (e.g., home vs away) to sum to 1.0.
    Example: 0.54 and 0.50 -> (0.5192, 0.4808)
    """
    if p_a <= 0 or p_b <= 0:
        raise ValueError("Probabilities must be > 0")
    s = p_a + p_b
    return p_a / s, p_b / s
