from fbm.markets.price_utils import (
    american_to_decimal,
    implied_prob_from_american,
    remove_vig_two_way,
)

def test_american_conversions():
    assert abs(american_to_decimal(+150) - 2.5) < 1e-9
    assert abs(american_to_decimal(-120) - (1 + 100/120)) < 1e-9
    assert abs(implied_prob_from_american(+150) - 0.4) < 1e-9
    # -120 implies 120/(120+100)=0.5454545...
    assert abs(implied_prob_from_american(-120) - (120/220)) < 1e-9

def test_remove_vig_two_way():
    # Suppose a book posts -120 / +110 (roughly .545 and .476 -> sum > 1)
    p_home = implied_prob_from_american(-120)
    p_away = implied_prob_from_american(+110)
    p_home_fair, p_away_fair = remove_vig_two_way(p_home, p_away)
    # they should sum to ~1.0
    assert abs((p_home_fair + p_away_fair) - 1.0) < 1e-9
    # and preserve ordering
    assert p_home_fair > p_away_fair
