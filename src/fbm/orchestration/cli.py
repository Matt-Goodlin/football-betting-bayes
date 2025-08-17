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
from fbm.markets.spread_total import prob_cover, prob_over


def _ensure_sample_csv(bronze_path: Path) -> Path:
    """Create a tiny sample CSV if none exists yet."""
    csv_path = bronze_path / "odds.csv"
    if not csv_path.exists():
        csv_path.write_text(
            # game_id,home_team,away_team,home_ml,away_ml,home_spread,home_spread_price,away_spread_price,total_line,over_price,under_price
            "game_id,home_team,away_team,home_ml,away_ml,home_spread,home_spread_price,away_spread_price,total_line,over_price,under_price\n"
            "W1-001,Chiefs,Bengals,-120,110,-2.5,-110,-110,48.5,-110,-110\n"
            "W1-002,49ers,Cowboys,-140,120,-3.5,-110,-110,45.0,-105,-115\n",
            encoding="utf-8",
        )
        print(f"[fbm] Created sample odds file at {csv_path}")
    return csv_path


def _placeholder_model_probs_for_game(home_fair_ml: float, spread_line: float, total_line: float):
    """
    Temporary stand-ins:
      - ML model prob = fair + 0.02 (capped to [0.05, 0.95])
      - Spread model cover uses a Normal with mean = spread_line - 0.5 (i.e., slight lean to fav)
      - Total model over uses a Normal with mean = total_line - 0.5 (slight lean to undercut the line)
    """
    def clip(p): return max(0.05, min(0.95, p))

    from fbm.markets.spread_total import prob_cover, prob_over
    ml_model = clip(home_fair_ml + 0.02)
    spread_model = prob_cover(diff_mean=spread_line - 0.5, spread_line=spread_line, sigma_diff=13.0)
    total_model = prob_over(total_mean=total_line - 0.5, total_line=total_line, sigma_total=10.0)
    return ml_model, spread_model, total_model


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

    # --- Tickets from CSV in bronze ---
    csv_path = _ensure_sample_csv(Path(bronze))
    rows = load_odds_csv(csv_path)

    print("\nTickets from CSV:")
    print("GameID,Market,Side/Bet,Odds(Am),Odds(Dec),Line, FairProb, ModelProb, Edge, EV_per_$, KellyStake")

    for r in rows:
        game_id = r["game_id"]

        # ===== MONEYLINE =====
        h_ml = int(r["home_ml"]); a_ml = int(r["away_ml"])
        p_h = implied_prob_from_american(h_ml)
        p_a = implied_prob_from_american(a_ml)
        p_h_fair, _ = remove_vig_two_way(p_h, p_a)
        ml_model, _, _ = _placeholder_model_probs_for_game(p_h_fair, float(r["home_spread"]), float(r["total_line"]))
        dec_h = american_to_decimal(h_ml)
        ev_ml, edge_ml = ev_and_edge(ml_model, p_h_fair, dec_h)
        stake_ml = kelly_fractional(ml_model, dec_h, bankroll=bankroll, fraction=kelly_frac)
        print(f"{game_id},ML,HOME,{h_ml:+d},{dec_h:.4f},, {p_h_fair:.4f}, {ml_model:.4f}, {edge_ml:+.4f}, {ev_ml:+.4f}, ${stake_ml:,.2f}")

        # ===== SPREAD (ATS) HOME =====
        sp_line = float(r["home_spread"])   # e.g., -2.5 means HOME -2.5
        sp_home_price = int(r["home_spread_price"])
        sp_away_price = int(r["away_spread_price"])
        p_sp_home_imp = implied_prob_from_american(sp_home_price)
        p_sp_away_imp = implied_prob_from_american(sp_away_price)
        p_sp_home_fair, _ = remove_vig_two_way(p_sp_home_imp, p_sp_away_imp)
        # model cover prob using Normal approx; could be replaced by posterior later
        sp_model = prob_cover(diff_mean=sp_line - 0.5, spread_line=sp_line, sigma_diff=13.0)
        dec_sp_home = american_to_decimal(sp_home_price)
        ev_sp, edge_sp = ev_and_edge(sp_model, p_sp_home_fair, dec_sp_home)
        stake_sp = kelly_fractional(sp_model, dec_sp_home, bankroll=bankroll, fraction=kelly_frac)
        print(f"{game_id},ATS,HOME,{sp_home_price:+d},{dec_sp_home:.4f},{sp_line:+.1f}, {p_sp_home_fair:.4f}, {sp_model:.4f}, {edge_sp:+.4f}, {ev_sp:+.4f}, ${stake_sp:,.2f}")

        # ===== TOTALS (OVER) =====
        tot_line = float(r["total_line"])
        over_price = int(r["over_price"]); under_price = int(r["under_price"])
        p_over_imp = implied_prob_from_american(over_price)
        p_under_imp = implied_prob_from_american(under_price)
        p_over_fair, _ = remove_vig_two_way(p_over_imp, p_under_imp)
        tot_model = prob_over(total_mean=tot_line - 0.5, total_line=tot_line, sigma_total=10.0)
        dec_over = american_to_decimal(over_price)
        ev_ou, edge_ou = ev_and_edge(tot_model, p_over_fair, dec_over)
        stake_ou = kelly_fractional(tot_model, dec_over, bankroll=bankroll, fraction=kelly_frac)
        print(f"{game_id},OU,OVER,{over_price:+d},{dec_over:.4f},{tot_line:.1f}, {p_over_fair:.4f}, {tot_model:.4f}, {edge_ou:+.4f}, {ev_ou:+.4f}, ${stake_ou:,.2f}")

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