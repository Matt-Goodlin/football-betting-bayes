from fbm.modeling.ratings_fit import fit_elo_ratings

def test_mov_multiplier_increases_update_for_blowouts():
    # Same teams, one blowout vs one narrow win; blowout should yield larger rating delta
    narrow = [{"home_team":"A","away_team":"B","home_pts":21,"away_pts":20}]
    blowout = [{"home_team":"A","away_team":"B","home_pts":42,"away_pts":14}]

    r1 = fit_elo_ratings(narrow, k=20.0, use_mov=True, mov_scale_pts=7.0, mov_cap=2.0)
    r2 = fit_elo_ratings(blowout, k=20.0, use_mov=True, mov_scale_pts=7.0, mov_cap=2.0)

    # A should gain more in the blowout case
    assert r2["A"] - r2["B"] > r1["A"] - r1["B"]
