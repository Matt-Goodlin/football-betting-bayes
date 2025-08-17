from fbm.markets.kelly import kelly_fractional

def test_kelly_zero_when_no_edge():
    # p_win equals breakeven prob for 2.0 decimal odds (50%) -> stake 0
    assert kelly_fractional(0.5, 2.0, bankroll=1000.0, fraction=0.5) == 0.0

def test_kelly_positive_with_edge():
    # 2.0 decimal odds, p_win=0.55 -> positive edge, stake > 0
    stake = kelly_fractional(0.55, 2.0, bankroll=1000.0, fraction=0.5)
    assert stake > 0

def test_kelly_monotonic_in_bankroll():
    s1 = kelly_fractional(0.55, 2.0, bankroll=1000.0, fraction=0.5)
    s2 = kelly_fractional(0.55, 2.0, bankroll=2000.0, fraction=0.5)
    assert s2 > s1
