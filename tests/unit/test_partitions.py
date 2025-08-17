from fbm.utils.partitions import part_path

def test_part_path_includes_parts():
    p = part_path("./data", "bronze", "NFL", 2025, 1)
    text = str(p)
    assert "league=NFL" in text
    assert "season=2025" in text
    assert "week=1" in text
