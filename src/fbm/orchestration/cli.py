# src/fbm/orchestration/cli.py
import argparse
from pathlib import Path

from fbm.utils.partitions import part_path
from fbm.utils.io import ensure_dir
from fbm.config.loader import load_config
from fbm.data.ingest.odds_csv import load_odds_csv

from fbm.markets.price_utils import (
    implied_prob_from_american,
    remove_vig_two_way,
    american_to_decimal,
)
from fbm.markets.kelly import kelly_fractional
from fbm.markets.edge import ev_and_edge
from fbm.modeling.baseline import BaselineModel
from fbm.modeling.ratings_csv import load_ratings_csv
from fbm.utils.csvout import write_csv


def _ensure_sample_csv(bronze_path: Path) -> Path:
    csv_path = bronze_path / "odds.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "game_id,home_team,away_team,home_ml,away_ml,home_spread,home_spread_price,away_spread_price,total_line,over_price,under_price\n"
            "W1-001,Chiefs,Bengals,-120,110,-2.5,-110,-110,48.5,-110,-110\n"
            "W1-002,49ers,Cowboys,-140,120,-3.5,-110,-110,45.0,-105,-115\n",
            encoding="utf-8",
        )
        print(f"[fbm] Created sample odds file at {csv_path}")
    return csv_path


def daily(season: int, week: int, league: str, config_path: str):
    # ---- Config
    cfg = load_config(config_path)
    bankroll = float(cfg["betting"]["bankroll"])
    kelly_frac = float(cfg["betting"]["kelly_fraction"])
    min_edge = float(cfg["betting"].get("min_edge_pct", 0.0))
    min_stake = float(cfg["betting"].get("min_kelly_stake", 0.0))

    def passes(edge: float, stake: float) -> bool:
        return (edge >= min_edge) and (stake >= min_stake)

    print(f"[fbm] Running daily pipeline | league={league} season={season} week={week}")
    print(
        f"[config] file={config_path} | bankroll=${bankroll:,.0f}, "
        f"kelly_fraction={kelly_frac}, min_edge={min_edge:.3f}, min_kelly=${min_stake:,.2f}"
    )

    # ---- Paths
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

    # ---- Load ratings from silver (optional)
    ratings_path = Path(silver) / "teams" / "ratings.csv"
    ratings = load_ratings_csv(ratings_path)
    print(f"[ratings] loaded {len(ratings)} team ratings from {ratings_path}")

    # ---- Baseline model (params from YAML if present)
    model_cfg = cfg.get("model", {})
    model = BaselineModel(
        ratings=ratings,
        hfa_points=float(model_cfg.get("hfa_points", 2.0)),
        sigma_diff=float(model_cfg.get("sigma_diff", 13.0)),
        sigma_total=float(model_cfg.get("sigma_total", 10.0)),
    )
    league_total_mean = float(model_cfg.get("league_total_mean", 45.0))

    # ---- Load odds CSV
    csv_path = _ensure_sample_csv(Path(bronze))
    rows = load_odds_csv(csv_path)  # normalizes keys to lowercase with underscores

    # ---- Collect tickets to print and to CSV
    headers = [
        "game_id",
        "market",
        "side_or_bet",
        "odds_am",
        "odds_dec",
        "line",
        "fair_prob",
        "model_prob",
        "edge",
        "ev_per_dollar",
        "kelly_stake",
    ]
    tickets = []

    print("\nTickets (filtered):")
    print("GameID,Market,Side/Bet,Odds(Am),Odds(Dec),Line,FairProb,ModelProb,Edge,EV_per_$,KellyStake")

    for r in rows:
        game_id = r["game_id"]
        home = r["home_team"]
        away = r["away_team"]

        # ===== MONEYLINE (HOME) =====
        h_ml = int(r["home_ml"])
        a_ml = int(r["away_ml"])
        p_h_imp = implied_prob_from_american(h_ml)
        p_a_imp = implied_prob_from_american(a_ml)
        p_h_fair, _ = remove_vig_two_way(p_h_imp, p_a_imp)
        ml_model = model.win_prob_home(home, away)

        dec_h = american_to_decimal(h_ml)
        ev_ml, edge_ml = ev_and_edge(ml_model, p_h_fair, dec_h)
        stake_ml = kelly_fractional(ml_model, dec_h, bankroll=bankroll, fraction=kelly_frac)
        if passes(edge_ml, stake_ml):
            print(
                f"{game_id},ML,HOME,{h_ml:+d},{dec_h:.4f},,"
                f"{p_h_fair:.4f},{ml_model:.4f},{edge_ml:+.4f},{ev_ml:+.4f},${stake_ml:,.2f}"
            )
            tickets.append(
                {
                    "game_id": game_id,
                    "market": "ML",
                    "side_or_bet": "HOME",
                    "odds_am": f"{h_ml:+d}",
                    "odds_dec": f"{dec_h:.4f}",
                    "line": "",
                    "fair_prob": f"{p_h_fair:.4f}",
                    "model_prob": f"{ml_model:.4f}",
                    "edge": f"{edge_ml:+.4f}",
                    "ev_per_dollar": f"{ev_ml:+.4f}",
                    "kelly_stake": f"{stake_ml:.2f}",
                }
            )

        # ===== SPREAD (ATS) HOME =====
        sp_line = float(r["home_spread"])  # e.g., -2.5 means HOME -2.5
        sp_home_price = int(r["home_spread_price"])
        sp_away_price = int(r["away_spread_price"])
        p_sp_h_imp = implied_prob_from_american(sp_home_price)
        p_sp_a_imp = implied_prob_from_american(sp_away_price)
        p_sp_h_fair, _ = remove_vig_two_way(p_sp_h_imp, p_sp_a_imp)
        sp_model = model.cover_prob_home(home, away, sp_line)

        dec_sp_home = american_to_decimal(sp_home_price)
        ev_sp, edge_sp = ev_and_edge(sp_model, p_sp_h_fair, dec_sp_home)
        stake_sp = kelly_fractional(sp_model, dec_sp_home, bankroll=bankroll, fraction=kelly_frac)
        if passes(edge_sp, stake_sp):
            print(
                f"{game_id},ATS,HOME,{sp_home_price:+d},{dec_sp_home:.4f},{sp_line:+.1f},"
                f"{p_sp_h_fair:.4f},{sp_model:.4f},{edge_sp:+.4f},{ev_sp:+.4f},${stake_sp:,.2f}"
            )
            tickets.append(
                {
                    "game_id": game_id,
                    "market": "ATS",
                    "side_or_bet": "HOME",
                    "odds_am": f"{sp_home_price:+d}",
                    "odds_dec": f"{dec_sp_home:.4f}",
                    "line": f"{sp_line:+.1f}",
                    "fair_prob": f"{p_sp_h_fair:.4f}",
                    "model_prob": f"{sp_model:.4f}",
                    "edge": f"{edge_sp:+.4f}",
                    "ev_per_dollar": f"{ev_sp:+.4f}",
                    "kelly_stake": f"{stake_sp:.2f}",
                }
            )

        # ===== TOTALS (OVER) =====
        tot_line = float(r["total_line"])
        over_price = int(r["over_price"])
        under_price = int(r["under_price"])
        p_over_imp = implied_prob_from_american(over_price)
        p_under_imp = implied_prob_from_american(under_price)
        p_over_fair, _ = remove_vig_two_way(p_over_imp, p_under_imp)
        tot_model = model.over_prob(tot_line, league_total_mean=league_total_mean)

        dec_over = american_to_decimal(over_price)
        ev_ou, edge_ou = ev_and_edge(tot_model, p_over_fair, dec_over)
        stake_ou = kelly_fractional(tot_model, dec_over, bankroll=bankroll, fraction=kelly_frac)
        if passes(edge_ou, stake_ou):
            print(
                f"{game_id},OU,OVER,{over_price:+d},{dec_over:.4f},{tot_line:.1f},"
                f"{p_over_fair:.4f},{tot_model:.4f},{edge_ou:+.4f},{ev_ou:+.4f},${stake_ou:,.2f}"
            )
            tickets.append(
                {
                    "game_id": game_id,
                    "market": "OU",
                    "side_or_bet": "OVER",
                    "odds_am": f"{over_price:+d}",
                    "odds_dec": f"{dec_over:.4f}",
                    "line": f"{tot_line:.1f}",
                    "fair_prob": f"{p_over_fair:.4f}",
                    "model_prob": f"{tot_model:.4f}",
                    "edge": f"{edge_ou:+.4f}",
                    "ev_per_dollar": f"{ev_ou:+.4f}",
                    "kelly_stake": f"{stake_ou:.2f}",
                }
            )

    # ---- Save tickets to gold CSV
    out_csv = Path(gold) / "tickets.csv"
    write_csv(out_csv, tickets, [
        "game_id","market","side_or_bet","odds_am","odds_dec",
        "line","fair_prob","model_prob","edge","ev_per_dollar","kelly_stake",
    ])
    print(f"\nSaved {len(tickets)} tickets to {out_csv}")

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
    p_daily.add_argument("--season", type=int, required=True, help="Season year, e.g., 2025")
    p_daily.add_argument("--week", type=int, required=True, help="Week number")
    p_daily.add_argument("--league", choices=["NFL", "CFB"], default="NFL", help="League (NFL default)")
    p_daily.add_argument("--config", default="conf/default.yaml", help="Path to YAML config")

    args = parser.parse_args()
    if args.cmd == "daily":
        daily(season=args.season, week=args.week, league=args.league, config_path=args.config)


if __name__ == "__main__":
    main()