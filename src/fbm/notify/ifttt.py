from typing import List, Dict, Any, Tuple
import urllib.request
import urllib.parse
import urllib.error

Ticket = Dict[str, Any]

def _parse_float(s: str, default: float = 0.0) -> float:
    try:
        return float(str(s).replace("$", "").replace(",", ""))
    except Exception:
        return default

def select_top_tickets(tickets: List[Ticket], top_n: int = 3) -> List[Ticket]:
    """
    Sort by edge desc, then kelly_stake desc. Returns top_n.
    Strings are like '+0.1234' and '1,234.56' per CSV schema.
    """
    return sorted(
        tickets,
        key=lambda t: (_parse_float(t.get("edge", "0")), _parse_float(t.get("kelly_stake", "0"))),
        reverse=True,
    )[:top_n]

def build_title_and_message(
    tickets: List[Ticket], league: str, season: int, week: int, top_n: int = 3
) -> Tuple[str, str]:
    """
    Returns (title, message) for IFTTT. Message trimmed < ~1000 chars.
    """
    title = f"FBM Picks — {league} {season} W{week}"
    top = select_top_tickets(tickets, top_n=top_n)
    if not top:
        return title, "No tickets passed filters."

    lines = []
    for t in top:
        parts = [
            f"{t.get('game_id','?')}",
            f"{t.get('market','?')} {t.get('side_or_bet','?')}",
        ]
        if t.get("line"):
            parts.append(f"line {t['line']}")
        parts.extend([
            f"odds {t.get('odds_am','?')} ({t.get('odds_dec','?')})",
            f"fair {t.get('fair_prob','?')}",
            f"model {t.get('model_prob','?')}",
        ])
        if t.get("model_prob_lo") and t.get("model_prob_hi"):
            parts.append(f"CI [{t['model_prob_lo']}..{t['model_prob_hi']}]")
        parts.extend([
            f"edge {t.get('edge','?')}",
            f"EV {t.get('ev_per_dollar','?')}",
            f"stake ${t.get('kelly_stake','0')}",
        ])
        lines.append(" · ".join(parts))

    msg = "\n".join(lines)
    if len(msg) > 1000:
        msg = msg[:980] + "\n…(truncated)"
    return title, msg

def post_ifttt(key: str, event: str, title: str, message: str) -> Tuple[bool, str]:
    """
    POST to IFTTT Webhooks:
      https://maker.ifttt.com/trigger/{event}/with/key/{key}
    Sends value1=title, value2=message. Returns (ok, info).
    """
    url = f"https://maker.ifttt.com/trigger/fbm_picks/with/key/btPgWVfuA3QVq6MUJtc0wV"
    data = urllib.parse.urlencode({"value1": title, "value2": message}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = resp.getcode()
            if 200 <= code < 300:
                return True, f"ok ({code})"
            return False, f"HTTP {code}"
    except urllib.error.HTTPError as e:
        return False, f"HTTPError {e.code}"
    except urllib.error.URLError as e:
        return False, f"URLError {e.reason}"
    except Exception as e:
        return False, f"Exception {e}"