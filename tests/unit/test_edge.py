from fbm.markets.edge import ev_and_edge

def test_ev_and_edge_basic():
    # Even odds (2.0), fair prob 0.50, model says 0.55 -> EV positive, edge +0.05
    ev, edge = ev_and_edge(0.55, 0.50, 2.0)
    assert ev > 0
    assert abs(edge - 0.05) < 1e-12
