from fbm.modeling.posterior import (
    prob_over_normal, prob_under_normal,
    prob_cover_spread, prob_total_over
)

def test_symmetry_at_mean():
    # If mean == line, over = under = 0.5
    p_over = prob_over_normal(mean=0.0, sigma=10.0, line=0.0)
    p_under = prob_under_normal(mean=0.0, sigma=10.0, line=0.0)
    assert abs(p_over - 0.5) < 1e-6
    assert abs(p_under - 0.5) < 1e-6
    assert abs(p_over + p_under - 1.0) < 1e-6

def test_spread_cover_logic():
    # Home spread -3.0, mean diff +3.0, sigma 13 => cover prob > 0.5
    p_cover = prob_cover_spread(mean_diff=3.0, sigma_diff=13.0, home_spread=-3.0)
    assert p_cover > 0.5

def test_totals_over_logic():
    # Total line 45, mean total 48, sigma 10 => over prob > 0.5
    p_over = prob_total_over(mean_total=48.0, sigma_total=10.0, total_line=45.0)
    assert p_over > 0.5

def test_degenerate_sigma():
    # sigma=0 behaves as point mass at mean
    assert prob_over_normal(mean=50.0, sigma=0.0, line=49.0) == 1.0
    assert prob_over_normal(mean=50.0, sigma=0.0, line=51.0) == 0.0
    assert prob_over_normal(mean=50.0, sigma=0.0, line=50.0) == 0.5
