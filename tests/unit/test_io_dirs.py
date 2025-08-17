from pathlib import Path
from fbm.utils.io import ensure_dir

def test_ensure_dir_creates_nested(tmp_path: Path):
    p = tmp_path / "data" / "bronze" / "league=NFL" / "season=2025" / "week=1"
    assert not p.exists()
    ensure_dir(p)
    assert p.exists() and p.is_dir()
