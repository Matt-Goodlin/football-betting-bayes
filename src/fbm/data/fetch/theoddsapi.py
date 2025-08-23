from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json, urllib.request, urllib.error, ssl
from collections import Counter, defaultdict

# Sports slugs per The Odds API docs:
# NFL:  americanfootball_nfl
# CFB:  americanfootball_ncaaf

def _get(url: str, timeout: int = 20) -> Any:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "fbm/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _most_common_value(values: List[Any]) -> Optional[Any]:
    if not values:
        return None
    c = Counter(values).most_common(1)
    return c[0][0] if c else None

def _avg(nums: List[float]) -> Optional[float]:
    return sum(nums) / len(nums) if nums else None

def fetch_odds_to_csv(
    api_key: str,
    sport: str,
    out_csv: Path,
    region: str = "us",
) -> Path:
    """
    Fetch ML, spreads, totals for upcoming events; write one CSV with our schema:
    game_id,home_team,away_team,home_ml,away_ml,home_spread,home_spread_price,away_spread_price,total_line,over_price,under_price
    Strategy:
      - ML: choose the BEST price across all books for each side (bettor-friendly)
      - Spreads: choose the MOST COMMON line across books; average the prices for that line
      - Totals: same as spreads (use most common total)
    """
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds?"
        f"apiKey={api_key}&regions={region}&markets=h2h,spreads,totals&oddsFormat=american"
    )
    data = _get(url)

    lines = [
        "game_id,home_team,away_team,home_ml,away_ml,home_spread,home_spread_price,away_spread_price,total_line,over_price,under_price"
    ]

    for ev in data:
        gid = ev.get("id", "")
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        # Accumulators across books
        best_home_ml: Optional[int] = None
        best_away_ml: Optional[int] = None

        spread_lines = []
        spread_price_home_by_line: Dict[float, List[int]] = defaultdict(list)
        spread_price_away_by_line: Dict[float, List[int]] = defaultdict(list)

        total_lines = []
        total_price_over_by_line: Dict[float, List[int]] = defaultdict(list)
        total_price_under_by_line: Dict[float, List[int]] = defaultdict(list)

        books = ev.get("bookmakers", []) or []
        for b in books:
            markets = b.get("markets", []) or []
            for m in markets:
                key = m.get("key")  # 'h2h' | 'spreads' | 'totals'
                outcomes = m.get("outcomes", []) or []

                if key == "h2h":
                    # each outcome: {name: team, price: int american}
                    for o in outcomes:
                        nm = o.get("name", "")
                        price = int(o.get("price")) if o.get("price") is not None else None
                        if nm == home and price is not None:
                            # best (max) for positive odds, least negative for negative odds
                            if best_home_ml is None:
                                best_home_ml = price
                            else:
                                # compare by implied payout; simply choose higher price if both positive,
                                # else choose the one closer to zero if both negative, else prefer positive
                                if (price >= 0 and (best_home_ml < 0 or price > best_home_ml)) or \
                                   (price < 0 and best_home_ml < 0 and price > best_home_ml):
                                    best_home_ml = price
                        if nm == away and price is not None:
                            if best_away_ml is None:
                                best_away_ml = price
                            else:
                                if (price >= 0 and (best_away_ml < 0 or price > best_away_ml)) or \
                                   (price < 0 and best_away_ml < 0 and price > best_away_ml):
                                    best_away_ml = price

                elif key == "spreads":
                    # outcomes: {name: team, point: float, price: int}
                    for o in outcomes:
                        nm = o.get("name", "")
                        pt = o.get("point", None)
                        pr = o.get("price", None)
                        if pt is None or pr is None:
                            continue
                        pt = float(pt); pr = int(pr)
                        spread_lines.append(pt)
                        if nm == home:
                            spread_price_home_by_line[pt].append(pr)
                        elif nm == away:
                            spread_price_away_by_line[pt].append(pr)

                elif key == "totals":
                    # outcomes: {name: 'Over'|'Under', point: float, price: int}
                    for o in outcomes:
                        nm = o.get("name", "")
                        pt = o.get("point", None)
                        pr = o.get("price", None)
                        if pt is None or pr is None:
                            continue
                        pt = float(pt); pr = int(pr)
                        total_lines.append(pt)
                        if nm.lower().startswith("over"):
                            total_price_over_by_line[pt].append(pr)
                        elif nm.lower().startswith("under"):
                            total_price_under_by_line[pt].append(pr)

        # Choose consensus for spreads/totals
        sp_line = _most_common_value(spread_lines)
        ou_line = _most_common_value(total_lines)

        # Average prices at that consensus line (fallback to empty lists)
        def _avg_int(xs: List[int]) -> Optional[int]:
            if not xs:
                return None
            return int(round(sum(xs) / float(len(xs))))

        home_sp_price = _avg_int(spread_price_home_by_line.get(sp_line, [])) if sp_line is not None else None
        away_sp_price = _avg_int(spread_price_away_by_line.get(sp_line, [])) if sp_line is not None else None
        over_price = _avg_int(total_price_over_by_line.get(ou_line, [])) if ou_line is not None else None
        under_price = _avg_int(total_price_under_by_line.get(ou_line, [])) if ou_line is not None else None

        # Fall back to blanks if missing pieces
        def _fmt(x: Optional[float]) -> str:
            return "" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))

        line = ",".join([
            gid,
            home,
            away,
            _fmt(best_home_ml),
            _fmt(best_away_ml),
            _fmt(sp_line),
            _fmt(home_sp_price),
            _fmt(away_sp_price),
            _fmt(ou_line),
            _fmt(over_price),
            _fmt(under_price),
        ])
        lines.append(line)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_csv

def fetch_recent_scores_to_csv(
    api_key: str,
    sport: str,
    out_csv: Path,
    days_from: int = 14,
) -> Path:
    """
    Fetch recent completed game scores (last N days) and write a simple results.csv:
    date,home_team,away_team,home_pts,away_pts
    """
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/scores?"
        f"apiKey={api_key}&daysFrom={days_from}&dateFormat=iso"
    )
    data = _get(url)

    lines = ["date,home_team,away_team,home_pts,away_pts"]
    for ev in data:
        # completed only
        if not ev.get("completed"):
            continue
        # oddsapi gives teams; sometimes 'scores' array exists; else use home_score/away_score
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        commence_time = (ev.get("commence_time") or "").split("T")[0]  # YYYY-MM-DD
        scores = ev.get("scores") or []
        home_pts = None
        away_pts = None
        for s in scores:
            if s.get("name") == home:
                home_pts = s.get("score")
            if s.get("name") == away:
                away_pts = s.get("score")
        # fallback fields
        if home_pts is None:
            home_pts = ev.get("home_score")
        if away_pts is None:
            away_pts = ev.get("away_score")

        if home and away and home_pts is not None and away_pts is not None:
            lines.append(f"{commence_time},{home},{away},{int(home_pts)},{int(away_pts)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_csv
