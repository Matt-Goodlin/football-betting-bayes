from pathlib import Path
from fbm.data.ingest.results_csv import load_results_csv

def test_results_loader_reads_and_normalizes(tmp_path: Path):
    p = tmp_path / "results.csv"
    p.write_text(
        "Date,Home Team,Away Team,Home Pts,Away Pts\n"
        "2024-12-01,Chiefs,Bengals,27,24\n",
        encoding="utf-8",
    )
    rows = load_results_csv(p)
    assert len(rows) == 1
    r = rows[0]
    assert r["home_team"] == "Chiefs"
    assert r["away_team"] == "Bengals"
    assert r["home_pts"] == 27 and r["away_pts"] == 24
