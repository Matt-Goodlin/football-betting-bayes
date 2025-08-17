import argparse

def daily(season: int, week: int, league: str = "NFL"):
    print(f"[fbm] Running daily pipeline | league={league} season={season} week={week}")
    print(" - (stub) ingest odds/schedules -> bronze")
    print(" - (stub) normalize -> silver")
    print(" - (stub) build features -> gold")
    print(" - (stub) sample posterior -> probabilities")
    print(" - (stub) compare vs market -> edges + kelly")
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
