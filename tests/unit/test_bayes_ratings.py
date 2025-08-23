from fbm.modeling.bayes_ratings import fit_bayes_ratings

def test_bayes_ratings_simple_signal():
    results = [
        {"home_team":"A","away_team":"B","home_pts":28,"away_pts":20},
        {"home_team":"A","away_team":"B","home_pts":24,"away_pts":17},
        {"home_team":"B","away_team":"A","home_pts":14,"away_pts":21},
    ]
    r, idx = fit_bayes_ratings(results, hfa_points=2.0, l2_lambda=4.0)
    # Team A should rate above B
    assert r["A"] > r["B"]
    # Sum-to-zero constraint
    assert abs(sum(r.values())) < 1e-6
