from fbm.modeling.baseline import BaselineModel

def test_baseline_win_prob_moves_with_ratings():
    m = BaselineModel(ratings={"A": 3.0, "B": 0.0}, hfa_points=2.0, sigma_diff=13.0)
    p_home = m.win_prob_home("A", "B")
    p_away_home = m.win_prob_home("B", "A")  # swap home/away
    assert p_home > 0.5
    assert p_away_home < 0.5
