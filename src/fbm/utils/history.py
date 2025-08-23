from pathlib import Path
from typing import List, Dict, Any

def append_csv(path: Path, rows: List[Dict[str, Any]], headers: List[str]) -> None:
    """
    Append rows to a CSV; write header if the file doesn't exist.
    Uses UTF-8 and no extra dependencies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        if write_header:
            f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
