from fbm.modeling.ratings_fit import fit_elo_ratings

def test_elo_fit_pushes_better_team_up():
    results = [
        {"home_team":"A","away_team":"B","home_pts":27,"away_pts":10},
        {"home_team":"A","away_team":"B","home_pts":24,"away_pts":20},
    ]
    r = fit_elo_ratings(results, k=20.0, hfa_points=2.0, iters=2)
    assert r["A"] > r["B"]

def test_elo_fit_handles_ties():
    results = [{"home_team":"A","away_team":"B","home_pts":21,"away_pts":21}]
    r = fit_elo_ratings(results, k=20.0, iters=1)
    # Ratings should remain close to start (0) after a tie
    assert abs(r.get("A",0.0)) < 5.0 and abs(r.get("B",0.0)) < 5.0
