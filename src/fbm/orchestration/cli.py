import argparse
from pathlib import Path
from fbm.utils.partitions import part_path
from fbm.utils.io import ensure_dir
from fbm.markets.price_utils import (
    implied_prob_from_american,
    remove_vig_two_way,
    american_to_decimal,
)
from fbm.markets.kelly import kelly_fractional
from fbm.config.loader import load_config
from fbm.markets.edge import ev_and_edge
from fbm.data.ingest.odds_csv import load_odds_csv


def _ensure_sample_csv(bronze_path: Path) -> Path:
    """Create a tiny sample CSV if none exists yet."""
    csv_path = bronze_path / "odds.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "game_id,home_team,away_team,home_ml,away_ml\n"
            "W1-001,Chiefs,Bengals,-120,110\n"
            "W1-002,49ers,Cowboys,-140,120\n",
            encoding="utf-8",
        )
        print(f"[fbm] Created sample odds file at {csv_path}")
    return csv_path


def _placeholder_model_prob(home_fair: float) -> float:
    """Temporary stand-in for a real model:
    nudge fair home prob upward by 2 points (cap to [0.05, 0.95]).
    """
    return max(0.05, min(0.95, home_fair + 0.02))


def daily(season: int, week: int, league: str, config_path: str):
    cfg = load_config(config_path)
    bankroll = cfg["betting"]["bankroll"]
    kelly_frac = cfg["betting"]["kelly_fraction"]

    print(f"[fbm] Running daily pipeline | league={league} season={season} week={week}")
    print(f"[config] file={config_path} | bankroll=${bankroll:,.0f}, kelly_fraction={kelly_frac}")

    dataroot = cfg["paths"]["datalake"]
    bronze = part_path(dataroot, "bronze", league, season, week)
    silver = part_path(dataroot, "silver", league, season, week)
    gold = part_path(dataroot, "gold", league, season, week)

    print("Planned output locations (creating if missing):")
    print(f" - bronze → {bronze}")
    print(f" - silver → {silver}")
    print(f" - gold   → {gold}")
    ensure_dir(bronze)
    ensure_dir(silver)
    ensure_dir(gold)

    # --- Demo: single-game moneyline math (kept from earlier) ---
    home_ml = -120
    away_ml = +110
    p_home_mkt = implied_prob_from_american(home_ml)
    p_away_mkt = implied_prob_from_american(away_ml)
    p_home_fair, p_away_fair = remove_vig_two_way(p_home_mkt, p_away_mkt)

    print("\nOdds demo (moneyline):")
    print(f" - Home ML {home_ml:+d} → implied={p_home_mkt:.4f}, decimal={american_to_decimal(home_ml):.4f}")
    print(f" - Away  ML {away_ml:+d} → implied={p_away_mkt:.4f}, decimal={american_to_decimal(away_ml):.4f}")
    print(f" - De-vigged fair probs → home={p_home_fair:.4f}, away={p_away_fair:.4f} (sum={(p_home_fair + p_away_fair):.4f})")

    # --- Kelly demo (kept) ---
    model_p_home = 0.56
    dec_home = american_to_decimal(home_ml)
    stake_demo = kelly_fractional(model_p_home, dec_home, bankroll=bankroll, fraction=kelly_frac)
    ev_per_dollar_demo = model_p_home * (dec_home - 1.0) - (1.0 - model_p_home)

    print("\nKelly demo:")
    print(f" - Model P(home)={model_p_home:.3f}, Market fair P(home)≈{p_home_fair:.3f}")
    print(f" - EV per $1: {ev_per_dollar_demo:.4f} → Stake ({kelly_frac:.2f} Kelly on ${bankroll:,.0f}) = ${stake_demo:,.2f}")

    # --- Tickets from CSV in bronze ---
    csv_path = _ensure_sample_csv(Path(bronze))
    rows = load_odds_csv(csv_path)

    print("\nTickets from CSV:")
    print("GameID,Market,Side,Odds(Am),Odds(Dec),FairProb,ModelProb,Edge,EV_per_$,KellyStake")
    for r in rows:
        game_id = r["game_id"]
        h_ml = int(r["home_ml"])
        a_ml = int(r["away_ml"])

        # implied + de-vig
        p_h = implied_prob_from_american(h_ml)
        p_a = implied_prob_from_american(a_ml)
        p_h_fair, p_a_fair = remove_vig_two_way(p_h, p_a)

        # placeholder model probability for home
        model_ph = _placeholder_model_prob(p_h_fair)

        # compute EV/edge and Kelly stake (home side)
        dec_h = american_to_decimal(h_ml)
        ev_per_dollar, edge_pct = ev_and_edge(model_ph, p_h_fair, dec_h)
        stake = kelly_fractional(model_ph, dec_h, bankroll=bankroll, fraction=kelly_frac)

        print(
            f"{game_id},ML,HOME,{h_ml:+d},{dec_h:.4f},{p_h_fair:.4f},{model_ph:.4f},"
            f"{edge_pct:+.4f},{ev_per_dollar:+.4f},${stake:,.2f}"
        )

    print("\nPipeline (stub):")
    print(" - ingest odds/schedules -> bronze")
    print(" - normalize -> silver")
    print(" - build features -> gold")
    print(" - sample posterior -> probabilities")
    print(" - compare vs market -> edges + kelly")
    print("Done. (Replace with real steps soon.)")


def main():
    parser = argparse.ArgumentParser(prog="fbm", description="Football Bayesian Model CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_daily = sub.add_parser("daily", help="Run ingest -> features -> fit -> select bets")
    p_daily.add_argument("--season", type=int, required=True, help="Season year, e.g. 2025")
    p_daily.add_argument("--week", type=int, required=True, help="Week number")
    p_daily.add_argument("--league", choices=["NFL", "CFB"], default="NFL", help="League (NFL default)")
    p_daily.add_argument("--config", default="conf/default.yaml", help="Path to YAML config")

    args = parser.parse_args()
    if args.cmd == "daily":
        daily(season=args.season, week=args.week, league=args.league, config_path=args.config)


if __name__ == "__main__":
    main()