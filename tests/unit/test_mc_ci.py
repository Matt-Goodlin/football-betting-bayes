from fbm.modeling.posterior import mc_ci_normal

def test_mc_ci_normal_bounds_and_width():
    lo, hi = mc_ci_normal(0.6, n=10000)  # ~0.6 ± 0.0096
    assert 0.55 < lo < 0.6
    assert 0.6 < hi < 0.65
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
