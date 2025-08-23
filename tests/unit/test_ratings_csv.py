from pathlib import Path
from fbm.modeling.ratings_csv import load_ratings_csv

def test_load_ratings_csv_reads_pairs(tmp_path: Path):
    p = tmp_path / "ratings.csv"
    p.write_text("team,rating\nChiefs,3.0\nBengals,1.0\n", encoding="utf-8")
    r = load_ratings_csv(p)
    assert r["Chiefs"] == 3.0
    assert r["Bengals"] == 1.0

def test_load_ratings_csv_missing_file_returns_empty(tmp_path: Path):
    p = tmp_path / "ratings.csv"
    r = load_ratings_csv(p)
    assert r == {}
