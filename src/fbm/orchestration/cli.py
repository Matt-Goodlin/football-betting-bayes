import argparse
import os
from pathlib import Path

from fbm.utils.partitions import part_path
from fbm.utils.io import ensure_dir
from fbm.config.loader import load_config

from fbm.data.ingest.odds_csv import load_odds_csv
from fbm.data.ingest.results_csv import load_results_dir
from fbm.data.fetch.theoddsapi import fetch_odds_to_csv, fetch_recent_scores_to_csv

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
from fbm.modeling.posterior import prob_cover_spread, prob_total_over
from fbm.utils.csvout import write_csv


SPORT_SLUG = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

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

def daily(season: int, week: int, league: str, config_path: str,
          mc_n: int = None, mc_seed: int = None,
          notify_ifttt: bool = False, notify_top_n: int = 3,
          use_live: bool = True, scores_days: int = 14):
    cfg = load_config(config_path)
    bankroll = float(cfg["betting"]["bankroll"])
    kelly_frac = float(cfg["betting"]["kelly_fraction"])
    min_edge = float(cfg["betting"].get("min_edge_pct", 0.0))
    min_stake = float(cfg["betting"].get("min_kelly_stake", 0.0))

    def passes(edge: float, stake: float) -> bool:
        return (edge >= min_edge) and (stake >= min_stake)

    print(f"[fbm] Running daily pipeline | league={league} season={season} week={week}")
    if mc_n is not None:
        print(f"[posterior] Monte Carlo: n={mc_n}, seed={mc_seed if mc_seed is not None else '-'}")
    print(
        f"[config] file={config_path} | bankroll=${bankroll:,.0f}, "
        f"kelly_fraction={kelly_frac}, min_edge={min_edge:.3f}, min_kelly=${min_stake:,.2f}"
    )

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
    ensure_dir(Path(season_silver) / "games")

    # --- Live data (if configured) ---
    odds_key = os.environ.get("ODDS_API_KEY")
    sport = SPORT_SLUG.get(league, "americanfootball_nfl")

    # Scores for ratings fit (recent N days)
    res_csv = Path(season_silver) / "games" / "results.csv"
    if use_live and odds_key:
        try:
            fetch_recent_scores_to_csv(odds_key, sport, res_csv, days_from=scores_days)
            print(f"[live] results fetched -> {res_csv}")
        except Exception as e:
            print(f"[live] results fetch failed: {e}; using sample.")
            _ensure_sample_results_csv(Path(season_silver))
    else:
        _ensure_sample_results_csv(Path(season_silver))

    ratings_path = Path(season_silver) / "teams" / "ratings.csv"
    starting_ratings = load_ratings_csv(ratings_path)
    print(f"[ratings] starter: {len(starting_ratings)} from {ratings_path}")

    games_dir = Path(season_silver) / "games"
    results = load_results_dir(games_dir)
    print(f"[results] loaded {len(results)} games from {games_dir} (*.csv)")

    # Fit ratings
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
        target_std = float(model_cfg.get("ratings_target_std", 3.0))
        fitted = normalize_ratings(fitted, target_std=target_std)
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
        target_std = float(model_cfg.get("ratings_target_std", 3.0))
        fitted = normalize_ratings(fitted, target_std=target_std)
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

    # League totals mean from results
    if results:
        totals = [
            int(g["home_pts"]) + int(g["away_pts"])
            for g in results
            if g.get("home_pts") not in ("", None) and g.get("away_pts") not in ("", None)
        ]
        league_total_mean = sum(totals) / len(totals) if totals else float(model_cfg.get("league_total_mean", 45.0))
    else:
        league_total_mean = float(model_cfg.get("league_total_mean", 45.0))
    print(f"[totals] league_total_mean = {league_total_mean:.2f}")

    # Odds
    odds_csv = Path(bronze) / "odds.csv"
    if use_live and odds_key:
        try:
            fetch_odds_to_csv(odds_key, sport, odds_csv)
            print(f"[live] odds fetched -> {odds_csv}")
        except Exception as e:
            print(f"[live] odds fetch failed: {e}; using sample.")
            odds_csv = _ensure_sample_odds_csv(Path(bronze))
    else:
        odds_csv = _ensure_sample_odds_csv(Path(bronze))

    rows = load_odds_csv(odds_csv)

    headers = [
        "game_id","market","side_or_bet","odds_am","odds_dec",
        "line","fair_prob","model_prob","edge","ev_per_dollar","kelly_stake",
    ]
    tickets = []

    # Printing header
    print("\nTickets (filtered):")
    print("GameID,Market,Side/Bet,Odds(Am),Odds(Dec),Line,FairProb,ModelProb,Edge,EV_per_$,KellyStake")

    for r in rows:
        game_id = r["game_id"]; home = r["home_team"]; away = r["away_team"]

        def _emit(market, side, am, dec, line, fair, modelp, edge, ev, stake):
            if (edge >= min_edge) and (stake >= min_stake):
                print(f"{game_id},{market},{side},{am},{dec:.4f},{line},{fair:.4f},{modelp:.4f},{edge:+.4f},{ev:+.4f},${stake:,.2f}")
                tickets.append({
                    "game_id": game_id, "market": market, "side_or_bet": side,
                    "odds_am": str(am), "odds_dec": f"{dec:.4f}", "line": str(line),
                    "fair_prob": f"{fair:.4f}", "model_prob": f"{modelp:.4f}",
                    "edge": f"{edge:+.4f}", "ev_per_dollar": f"{ev:+.4f}",
                    "kelly_stake": f"{stake:.2f}",
                })

        # Moneyline
        try:
            h_ml = int(r["home_ml"]); a_ml = int(r["away_ml"])
        except Exception:
            h_ml = a_ml = 0  # if missing, skip ML later
        if h_ml != 0 and a_ml != 0:
            p_h_imp = implied_prob_from_american(h_ml); p_a_imp = implied_prob_from_american(a_ml)
            p_h_fair, _ = remove_vig_two_way(p_h_imp, p_a_imp)
            ml_model = BaselineModel(
                ratings=fitted, hfa_points=model.hfa_points, sigma_diff=model.sigma_diff, sigma_total=model.sigma_total
            ).win_prob_home(home, away)
            dec_h = american_to_decimal(h_ml)
            ev_ml, edge_ml = ev_and_edge(ml_model, p_h_fair, dec_h)
            stake_ml = kelly_fractional(ml_model, dec_h, bankroll=bankroll, fraction=kelly_frac)
            _emit("ML", "HOME", h_ml, dec_h, "", p_h_fair, ml_model, edge_ml, ev_ml, stake_ml)

        # Spreads (home cover)
        try:
            sp_line = float(r["home_spread"])
            sp_home_price = int(r["home_spread_price"]); sp_away_price = int(r["away_spread_price"])
            p_sp_h_imp = implied_prob_from_american(sp_home_price); p_sp_a_imp = implied_prob_from_american(sp_away_price)
            p_sp_h_fair, p_sp_a_fair = remove_vig_two_way(p_sp_h_imp, p_sp_a_imp)
            mean_diff = fitted.get(home, 0.0) - fitted.get(away, 0.0) + model.hfa_points
            sp_model_home = prob_cover_spread(mean_diff, model.sigma_diff, sp_line)
            dec_sp_home = american_to_decimal(sp_home_price)
            ev_sp_h, edge_sp_h = ev_and_edge(sp_model_home, p_sp_h_fair, dec_sp_home)
            stake_sp_h = kelly_fractional(sp_model_home, dec_sp_home, bankroll=bankroll, fraction=kelly_frac)
            _emit("ATS", "HOME", sp_home_price, dec_sp_home, f"{sp_line:+.1f}", p_sp_h_fair, sp_model_home, edge_sp_h, ev_sp_h, stake_sp_h)

            # away cover = 1 - home cover, line sign flips for display
            sp_model_away = 1.0 - sp_model_home
            dec_sp_away = american_to_decimal(sp_away_price)
            ev_sp_a, edge_sp_a = ev_and_edge(sp_model_away, p_sp_a_fair, dec_sp_away)
            stake_sp_a = kelly_fractional(sp_model_away, dec_sp_away, bankroll=bankroll, fraction=kelly_frac)
            _emit("ATS", "AWAY", sp_away_price, dec_sp_away, f"{-sp_line:+.1f}", p_sp_a_fair, sp_model_away, edge_sp_a, ev_sp_a, stake_sp_a)
        except Exception:
            pass

        # Totals
        try:
            tot_line = float(r["total_line"])
            over_price = int(r["over_price"]); under_price = int(r["under_price"])
            p_over_imp = implied_prob_from_american(over_price); p_under_imp = implied_prob_from_american(under_price)
            p_over_fair, p_under_fair = remove_vig_two_way(p_over_imp, p_under_imp)
            tot_model_over = prob_total_over(league_total_mean, model.sigma_total, tot_line)
            dec_over = american_to_decimal(over_price)
            ev_ou_o, edge_ou_o = ev_and_edge(tot_model_over, p_over_fair, dec_over)
            stake_ou_o = kelly_fractional(tot_model_over, dec_over, bankroll=bankroll, fraction=kelly_frac)
            _emit("OU", "OVER", over_price, dec_over, f"{tot_line:.1f}", p_over_fair, tot_model_over, edge_ou_o, ev_ou_o, stake_ou_o)

            tot_model_under = 1.0 - tot_model_over
            dec_under = american_to_decimal(under_price)
            ev_ou_u, edge_ou_u = ev_and_edge(tot_model_under, p_under_fair, dec_under)
            stake_ou_u = kelly_fractional(tot_model_under, dec_under, bankroll=bankroll, fraction=kelly_frac)
            _emit("OU", "UNDER", under_price, dec_under, f"{tot_line:.1f}", p_under_fair, tot_model_under, edge_ou_u, ev_ou_u, stake_ou_u)
        except Exception:
            pass

    out_csv = Path(gold) / "tickets.csv"
    write_csv(out_csv, tickets, headers)
    print(f"\nSaved {len(tickets)} tickets to {out_csv}")

    # Optional IFTTT notification (unchanged)
    if notify_ifttt and tickets:
        try:
            import json, urllib.request
            key = os.environ.get("IFTTT_KEY")
            event = os.environ.get("IFTTT_EVENT", "fbm_picks")
            if not key:
                print("[notify] Skipping IFTTT: missing IFTTT_KEY (arg or env).")
            else:
                top_n = max(1, int(notify_top_n))
                msg_title = f"{league} {season} W{week}: Top {min(top_n, len(tickets))} tickets"
                msg_body = "\n".join(
                    f"{t['market']} {t['side_or_bet']} {t['odds_am']} | edge {t['edge']} | stake ${t['kelly_stake']}"
                    for t in tickets[:top_n]
                )
                payload = {"value1": msg_title, "value2": msg_body}
                url = f"https://maker.ifttt.com/trigger/{event}/with/key/{key}"
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    print(f"[notify] IFTTT: ok ({resp.getcode()})")
        except Exception as e:
            print(f"[notify] IFTTT error: {e}")

    print("\nPipeline (live if configured):")
    print(" - fetch results (live or sample) -> season_silver/games/results.csv")
    print(" - fit ratings -> season_silver/teams/ratings_fitted_{method}.csv")
    print(" - fetch odds (live or sample) -> bronze/week/odds.csv")
    print(" - compute edges + kelly -> gold/week/tickets.csv")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(prog="fbm", description="Football Bayesian Model CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_daily = sub.add_parser("daily", help="Run ingest -> features -> fit -> select bets")
    p_daily.add_argument("--season", type=int, required=True, help="Season year, e.g., 2025")
    p_daily.add_argument("--week", type=int, required=True, help="Week number")
    p_daily.add_argument("--league", choices=["NFL", "CFB"], default="NFL", help="League (NFL default)")
    p_daily.add_argument("--config", default="conf/default.yaml", help="Path to YAML config")
    p_daily.add_argument("--mc-n", type=int, default=None)
    p_daily.add_argument("--mc-seed", type=int, default=None)
    p_daily.add_argument("--notify-ifttt", action="store_true")
    p_daily.add_argument("--notify-top-n", type=int, default=3)
    p_daily.add_argument("--no-live", action="store_true", help="Disable live fetch even if ODDS_API_KEY is set")
    p_daily.add_argument("--scores-days", type=int, default=14, help="How many days back to pull scores")

    args = parser.parse_args()
    if args.cmd == "daily":
        daily(
            season=args.season, week=args.week, league=args.league, config_path=args.config,
            mc_n=args.mc_n, mc_seed=args.mc_seed,
            notify_ifttt=args.notify_ifttt, notify_top_n=args.notify_top_n,
            use_live=(not args.no_live), scores_days=args.scores_days
        )

if __name__ == "__main__":
    main()