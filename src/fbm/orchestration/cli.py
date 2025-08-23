import argparse
import os
from pathlib import Path
from typing import Optional

from fbm.utils.partitions import part_path
from fbm.utils.io import ensure_dir
from fbm.config.loader import load_config

from fbm.data.ingest.odds_csv import load_odds_csv
from fbm.data.ingest.results_csv import load_results_dir

from fbm.markets.price_utils import (
    implied_prob_from_american,
    remove_vig_two_way,
    american_to_decimal,
)
from fbm.markets.kelly import kelly_fractional
from fbm.markets.edge import ev_and_edge
from fbm.modeling.baseline import BaselineModel
from fbm.modeling.ratings_csv import load_ratings_csv
from fbm.modeling.ratings_fit import fit_elo_ratings, normalize_ratings
from fbm.modeling.bayes_ratings import fit_bayes_ratings
from fbm.modeling.posterior import (
    simulate_cover_spread,
    simulate_total_over,
    mc_ci_normal,
)
from fbm.notify.ifttt import build_title_and_message, post_ifttt
from fbm.utils.csvout import write_csv


def _ensure_sample_odds_csv(bronze_path: Path) -> Path:
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


def _ensure_sample_results_csv(season_silver_path: Path) -> Path:
    games_dir = season_silver_path / "games"
    ensure_dir(games_dir)
    csv_path = games_dir / "results.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "date,home_team,away_team,home_pts,away_pts\n"
            "2024-12-01,Chiefs,Bengals,27,24\n"
            "2024-12-08,49ers,Cowboys,23,20\n",
            encoding="utf-8",
        )
        print(f"[fbm] Created sample results file at {csv_path}")
    return csv_path


def _write_ratings_csv(path: Path, ratings: dict) -> None:
    lines = ["team,rating"]
    for team in sorted(ratings.keys()):
        lines.append(f"{team},{ratings[team]:.6f}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def daily(
    season: int,
    week: int,
    league: str,
    config_path: str,
    mc_n: Optional[int],
    mc_seed: Optional[int],
    notify_ifttt: bool = False,
    ifttt_key: Optional[str] = None,
    ifttt_event: Optional[str] = None,
    notify_top_n: int = 3,
):
    cfg = load_config(config_path)
    bankroll = float(cfg["betting"]["bankroll"])
    kelly_frac = float(cfg["betting"]["kelly_fraction"])
    min_edge = float(cfg["betting"].get("min_edge_pct", 0.0))
    min_stake = float(cfg["betting"].get("min_kelly_stake", 0.0))

    if mc_n is None:
        mc_n = int(cfg.get("model", {}).get("mc_n", 10000))
    if mc_seed is None:
        seed_cfg = cfg.get("model", {}).get("mc_seed", None)
        mc_seed = int(seed_cfg) if (seed_cfg is not None and str(seed_cfg).strip() != "") else None

    def passes(edge: float, stake: float) -> bool:
        return (edge >= min_edge) and (stake >= min_stake)

    print(f"[fbm] Running daily pipeline | league={league} season={season} week={week}")
    print(
        f"[config] file={config_path} | bankroll=${bankroll:,.0f}, "
        f"kelly_fraction={kelly_frac}, min_edge={min_edge:.3f}, min_kelly=${min_stake:,.2f}"
    )
    print(f"[posterior] Monte Carlo: n={mc_n}, seed={mc_seed}")

    dataroot = cfg["paths"]["datalake"]
    bronze = part_path(dataroot, "bronze", league, season, week)
    silver = part_path(dataroot, "silver", league, season, week)
    gold = part_path(dataroot, "gold", league, season, week)
    print("Planned output locations (creating if missing):")
    print(f" - bronze → {bronze}")
    print(f" - silver → {silver}")
    print(f" - gold   → {gold}")
    ensure_dir(bronze); ensure_dir(silver); ensure_dir(gold)

    season_silver = part_path(dataroot, "silver", league, season, None)
    ensure_dir(Path(season_silver) / "teams")

    ratings_path = Path(season_silver) / "teams" / "ratings.csv"
    starting_ratings = load_ratings_csv(ratings_path)
    print(f"[ratings] starter: {len(starting_ratings)} from {ratings_path}")

    games_dir = Path(season_silver) / "games"
    _ensure_sample_results_csv(Path(season_silver))
    results = load_results_dir(games_dir)
    print(f"[results] loaded {len(results)} games from {games_dir} (*.csv)")

    model_cfg = cfg.get("model", {})
    method = str(model_cfg.get("ratings_method", "elo")).lower()

    if method == "bayes":
        l2_lambda = float(model_cfg.get("bayes_l2_lambda", 4.0))
        hfa_cfg = float(model_cfg.get("hfa_points", 2.0))
        fitted, _ = fit_bayes_ratings(
            results,
            hfa_points=hfa_cfg,
            l2_lambda=l2_lambda,
            start_ratings=starting_ratings or None,
            enforce_sum_zero=True,
        )
        from fbm.modeling.ratings_fit import normalize_ratings as _norm
        target_std = float(model_cfg.get("ratings_target_std", 3.0))
        fitted = _norm(fitted, target_std=target_std)
        method_used = "bayes"
    else:
        hfa_cfg = float(model_cfg.get("hfa_points", 2.0))
        scale_pts = float(model_cfg.get("sigma_diff", 13.0))
        elo_k = float(model_cfg.get("elo_k", 20.0))
        elo_iters = int(model_cfg.get("elo_iters", 2))
        use_mov = bool(model_cfg.get("mov_enabled", True))
        mov_scale = float(model_cfg.get("mov_scale_pts", 7.0))
        mov_cap = float(model_cfg.get("mov_cap", 2.0))
        fitted = fit_elo_ratings(
            results,
            start_ratings=starting_ratings,
            k=elo_k,
            hfa_points=hfa_cfg,
            iters=elo_iters,
            scale_pts=scale_pts,
            use_mov=use_mov,
            mov_scale_pts=mov_scale,
            mov_cap=mov_cap,
        )
        from fbm.modeling.ratings_fit import normalize_ratings as _norm
        target_std = float(model_cfg.get("ratings_target_std", 3.0))
        fitted = _norm(fitted, target_std=target_std)
        method_used = "elo"

    fitted_path = Path(season_silver) / "teams" / f"ratings_fitted_{method_used}.csv"
    _write_ratings_csv(fitted_path, fitted)
    print(f"[ratings] fitted ({method_used}): {len(fitted)} saved to {fitted_path}")

    model = BaselineModel(
        ratings=fitted,
        hfa_points=float(model_cfg.get("hfa_points", 2.0)),
        sigma_diff=float(model_cfg.get("sigma_diff", 13.0)),
        sigma_total=float(model_cfg.get("sigma_total", 10.0)),
    )

    # League total mean from results, fallback to config default
    if results:
        totals = [
            int(g["home_pts"]) + int(g["away_pts"])
            for g in results
            if g.get("home_pts") and g.get("away_pts")
        ]
        league_total_mean = sum(totals) / len(totals) if totals else float(model_cfg.get("league_total_mean", 45.0))
    else:
        league_total_mean = float(model_cfg.get("league_total_mean", 45.0))
    print(f"[totals] league_total_mean = {league_total_mean:.2f}")

    odds_csv = _ensure_sample_odds_csv(Path(bronze))
    rows = load_odds_csv(odds_csv)

    headers = [
        "game_id","market","side_or_bet","odds_am","odds_dec",
        "line","fair_prob","model_prob","model_prob_lo","model_prob_hi",
        "edge","ev_per_dollar","kelly_stake",
    ]
    tickets = []

    print("\nTickets (filtered):")
    print("GameID,Market,Side/Bet,Odds(Am),Odds(Dec),Line,FairProb,ModelProb[lo..hi],Edge,EV_per_$,KellyStake")

    for r in rows:
        game_id = r["game_id"]; home = r["home_team"]; away = r["away_team"]

        # ML HOME (closed-form win prob; CI displayed using mc_ci_normal for readability)
        h_ml = int(r["home_ml"]); a_ml = int(r["away_ml"])
        p_h_imp = implied_prob_from_american(h_ml); p_a_imp = implied_prob_from_american(a_ml)
        p_h_fair, _ = remove_vig_two_way(p_h_imp, p_a_imp)
        ml_model = model.win_prob_home(home, away)
        dec_h = american_to_decimal(h_ml)
        ev_ml, edge_ml = ev_and_edge(ml_model, p_h_fair, dec_h)
        stake_ml = kelly_fractional(ml_model, dec_h, bankroll=bankroll, fraction=kelly_frac)
        if passes(edge_ml, stake_ml):
            lo_ml, hi_ml = mc_ci_normal(ml_model, n=int(mc_n))  # display-only CI
            print(f"{game_id},ML,HOME,{h_ml:+d},{dec_h:.4f},,{p_h_fair:.4f},{ml_model:.4f}[{lo_ml:.3f}..{hi_ml:.3f}],{edge_ml:+.4f},{ev_ml:+.4f},${stake_ml:,.2f}")
            tickets.append({
                "game_id": game_id, "market": "ML", "side_or_bet": "HOME",
                "odds_am": f"{h_ml:+d}", "odds_dec": f"{dec_h:.4f}", "line": "",
                "fair_prob": f"{p_h_fair:.4f}", "model_prob": f"{ml_model:.4f}",
                "model_prob_lo": f"{lo_ml:.4f}", "model_prob_hi": f"{hi_ml:.4f}",
                "edge": f"{edge_ml:+.4f}", "ev_per_dollar": f"{ev_ml:+.4f}", "kelly_stake": f"{stake_ml:.2f}",
            })

        # ATS HOME — Monte Carlo posterior + CI
        sp_line = float(r["home_spread"])
        sp_home_price = int(r["home_spread_price"]); sp_away_price = int(r["away_spread_price"])
        p_sp_h_imp = implied_prob_from_american(sp_home_price); p_sp_a_imp = implied_prob_from_american(sp_away_price)
        p_sp_h_fair, _ = remove_vig_two_way(p_sp_h_imp, p_sp_a_imp)
        mean_diff = fitted.get(home, 0.0) - fitted.get(away, 0.0) + model.hfa_points
        sp_model = simulate_cover_spread(mean_diff, model.sigma_diff, sp_line, n=int(mc_n), seed=mc_seed)
        lo_sp, hi_sp = mc_ci_normal(sp_model, n=int(mc_n))
        dec_sp_home = american_to_decimal(sp_home_price)
        ev_sp, edge_sp = ev_and_edge(sp_model, p_sp_h_fair, dec_sp_home)
        stake_sp = kelly_fractional(sp_model, dec_sp_home, bankroll=bankroll, fraction=kelly_frac)
        if passes(edge_sp, stake_sp):
            print(f"{game_id},ATS,HOME,{sp_home_price:+d},{dec_sp_home:.4f},{sp_line:+.1f},{p_sp_h_fair:.4f},{sp_model:.4f}[{lo_sp:.3f}..{hi_sp:.3f}],{edge_sp:+.4f},{ev_sp:+.4f},${stake_sp:,.2f}")
            tickets.append({
                "game_id": game_id, "market": "ATS", "side_or_bet": "HOME",
                "odds_am": f"{sp_home_price:+d}", "odds_dec": f"{dec_sp_home:.4f}", "line": f"{sp_line:+.1f}",
                "fair_prob": f"{p_sp_h_fair:.4f}", "model_prob": f"{sp_model:.4f}",
                "model_prob_lo": f"{lo_sp:.4f}", "model_prob_hi": f"{hi_sp:.4f}",
                "edge": f"{edge_sp:+.4f}", "ev_per_dollar": f"{ev_sp:+.4f}", "kelly_stake": f"{stake_sp:.2f}",
            })

        # OU OVER — Monte Carlo posterior + CI
        tot_line = float(r["total_line"])
        over_price = int(r["over_price"]); under_price = int(r["under_price"])
        p_over_imp = implied_prob_from_american(over_price); p_under_imp = implied_prob_from_american(under_price)
        p_over_fair, _ = remove_vig_two_way(p_over_imp, p_under_imp)
        tot_model = simulate_total_over(league_total_mean, model.sigma_total, tot_line, n=int(mc_n), seed=mc_seed)
        lo_ou, hi_ou = mc_ci_normal(tot_model, n=int(mc_n))
        dec_over = american_to_decimal(over_price)
        ev_ou, edge_ou = ev_and_edge(tot_model, p_over_fair, dec_over)
        stake_ou = kelly_fractional(tot_model, dec_over, bankroll=bankroll, fraction=kelly_frac)
        if passes(edge_ou, stake_ou):
            print(f"{game_id},OU,OVER,{over_price:+d},{dec_over:.4f},{tot_line:.1f},{p_over_fair:.4f},{tot_model:.4f}[{lo_ou:.3f}..{hi_ou:.3f}],{edge_ou:+.4f},{ev_ou:+.4f},${stake_ou:,.2f}")
            tickets.append({
                "game_id": game_id, "market": "OU", "side_or_bet": "OVER",
                "odds_am": f"{over_price:+d}", "odds_dec": f"{dec_over:.4f}", "line": f"{tot_line:.1f}",
                "fair_prob": f"{p_over_fair:.4f}", "model_prob": f"{tot_model:.4f}",
                "model_prob_lo": f"{lo_ou:.4f}", "model_prob_hi": f"{hi_ou:.4f}",
                "edge": f"{edge_ou:+.4f}", "ev_per_dollar": f"{ev_ou:+.4f}", "kelly_stake": f"{stake_ou:.2f}",
            })

    out_csv = Path(gold) / "tickets.csv"
    write_csv(out_csv, tickets, headers)
    print(f"\nSaved {len(tickets)} tickets to {out_csv}")

    # -------- Optional iPhone push via IFTTT Webhooks --------
    if notify_ifttt:
        key = ifttt_key or os.environ.get("IFTTT_KEY")
        event = ifttt_event or os.environ.get("IFTTT_EVENT", "fbm_picks")
        if not key:
            print("[notify] Skipping IFTTT: missing IFTTT_KEY (arg or env).")
        else:
            title, msg = build_title_and_message(tickets, league, season, week, top_n=notify_top_n)
            ok, info = post_ifttt(key, event, title, msg)
            print(f"[notify] IFTTT: {info}")

    print("\nPipeline (stub):")
    print(" - ingest odds/schedules -> bronze")
    print(" - normalize -> silver")
    print(" - fit ratings from results -> season_silver/teams/ratings_fitted_{method}.csv")
    print(" - build features -> gold")
    print(" - sample posterior -> probabilities")
    print(" - compare vs market -> edges + kelly")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(prog="fbm", description="Football Bayesian Model CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_daily = sub.add_parser("daily", help="Run ingest -> features -> fit -> select bets")
    p_daily.add_argument("--season", type=int, required=True, help="Season year, e.g., 2025")
    p_daily.add_argument("--week", type=int, required=True, help="Week number")
    p_daily.add_argument("--league", choices=["NFL", "CFB"], default="NFL", help="League (NFL default)")
    p_daily.add_argument("--config", default="conf/default.yaml", help="Path to YAML config")
    p_daily.add_argument("--mc-n", type=int, default=None, help="Monte Carlo draws per market (overrides config)")
    p_daily.add_argument("--mc-seed", type=int, default=None, help="RNG seed for Monte Carlo (overrides config)")
    # IFTTT notification flags
    p_daily.add_argument("--notify-ifttt", action="store_true", help="Send iPhone push via IFTTT Webhooks")
    p_daily.add_argument("--ifttt-key", type=str, default=None, help="IFTTT Webhooks key (or env IFTTT_KEY)")
    p_daily.add_argument("--ifttt-event", type=str, default="fbm_picks", help="IFTTT event name (default fbm_picks)")
    p_daily.add_argument("--notify-top-n", type=int, default=3, help="How many top tickets to include")

    args = parser.parse_args()
    if args.cmd == "daily":
        daily(
            season=args.season,
            week=args.week,
            league=args.league,
            config_path=args.config,
            mc_n=args.mc_n,
            mc_seed=args.mc_seed,
            notify_ifttt=args.notify_ifttt,
            ifttt_key=args.ifttt_key,
            ifttt_event=args.ifttt_event,
            notify_top_n=args.notify_top_n,
        )


if __name__ == "__main__":
    main()