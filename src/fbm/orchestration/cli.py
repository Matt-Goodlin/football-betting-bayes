import argparse
from fbm.utils.partitions import part_path
from fbm.markets.price_utils import (
    implied_prob_from_american,
    remove_vig_two_way,
    american_to_decimal,
)

def daily(season: int, week: int, league: str = "NFL"):
    print(f"[fbm] Running daily pipeline | league={league} season={season} week={week}")

    dataroot = "./data"
    bronze = part_path(dataroot, "bronze", league, season, week)
    silver = part_path(dataroot, "silver", league, season, week)
    gold   = part_path(dataroot, "gold",   league, season, week)

    print("Planned output locations:")
    print(f" - bronze → {bronze}")
    print(f" - silver → {silver}")
    print(f" - gold   → {gold}")

    # --- Demo: moneyline two-way market (stub values) ---
    home_ml = -120  # example: home favorite -120
    away_ml = +110  # example: away underdog +110

    p_home_mkt = implied_prob_from_american(home_ml)
    p_away_mkt = implied_prob_from_american(away_ml)
    p_home_fair, p_away_fair = remove_vig_two_way(p_home_mkt, p_away_mkt)

    print("\nOdds demo (moneyline):")
    print(f" - Home ML {home_ml:+d} → implied={p_home_mkt:.4f}, decimal={american_to_decimal(home_ml):.4f}")
    print(f" - Away ML {away_ml:+d} → implied={p_away_mkt:.4f}, decimal={american_to_decimal(away_ml):.4f}")
    print(f" - De-vigged fair probs → home={p_home_fair:.4f}, away={p_away_fair:.4f} (sum={p_home_fair+p_away_fair:.4f})")

    # stubbed pipeline steps
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

    args = parser.parse_args()
    if args.cmd == "daily":
        daily(season=args.season, week=args.week, league=args.league)

if __name__ == "__main__":
    main()
