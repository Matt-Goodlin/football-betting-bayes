from fbm.markets.spread_total import prob_cover, prob_over

def test_prob_cover_symmetry():
    # If mean diff equals line, cover prob ~ 0.5
    p = prob_cover(diff_mean=-2.5, spread_line=-2.5, sigma_diff=13.0)
    assert abs(p - 0.5) < 1e-6

def test_prob_over_symmetry():
    # If mean total equals line, over prob ~ 0.5
    p = prob_over(total_mean=45.0, total_line=45.0, sigma_total=10.0)
    assert abs(p - 0.5) < 1e-6

def test_prob_cover_shifts_correctly():
    # If mean diff is much greater than line, cover probability > 0.5
    assert prob_cover(diff_mean=-1.0, spread_line=-3.0, sigma_diff=13.0) > 0.5
