def kelly_fractional(p_win: float, odds_decimal: float, bankroll: float, fraction: float = 0.33) -> float:
    """
    Fractional Kelly stake.
    p_win: model win probability (0..1)
    odds_decimal: market decimal odds (e.g., 1.91 for -110)
    bankroll: current bankroll in dollars
    fraction: Kelly fraction (0..1), e.g., 0.33 for 1/3 Kelly

    Returns stake in dollars (>= 0). 0 if edge <= 0.
    """
    if not (0 < p_win < 1):
        raise ValueError("p_win must be in (0,1)")
    if odds_decimal <= 1.0:
        raise ValueError("odds_decimal must be > 1.0")
    if bankroll < 0:
        raise ValueError("bankroll must be >= 0")
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0,1]")

    b = odds_decimal - 1.0           # net payout ratio
    q = 1.0 - p_win
    edge = b * p_win - q             # Kelly numerator
    if edge <= 0:
        return 0.0
    kelly_unit = edge / b            # full Kelly fraction of bankroll
    return bankroll * fraction * kelly_unit
