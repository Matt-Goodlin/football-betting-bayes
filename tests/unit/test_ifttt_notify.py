from fbm.notify.ifttt import build_title_and_message, select_top_tickets

def test_select_top_tickets_ordering():
    tickets = [
        {"edge": "+0.05", "kelly_stake": "50.00", "game_id": "A"},
        {"edge": "+0.07", "kelly_stake": "10.00", "game_id": "B"},
        {"edge": "+0.07", "kelly_stake": "100.00", "game_id": "C"},
    ]
    top = select_top_tickets(tickets, top_n=2)
    assert [t["game_id"] for t in top] == ["C", "B"]

def test_build_title_and_message_empty():
    title, msg = build_title_and_message([], "NFL", 2025, 2, top_n=3)
    assert "FBM Picks — NFL 2025 W2" in title
    assert "No tickets" in msg