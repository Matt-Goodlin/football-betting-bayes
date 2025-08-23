from pathlib import Path
from fbm.utils.csvout import write_csv

def test_write_csv(tmp_path: Path):
    out = tmp_path / "tickets.csv"
    rows = [{"game_id":"G1","market":"ML"}]
    headers = ["game_id","market"]
    write_csv(out, rows, headers)
    text = out.read_text(encoding="utf-8").strip().splitlines()
    assert text[0] == "game_id,market"
    assert text[1].startswith("G1,ML")
