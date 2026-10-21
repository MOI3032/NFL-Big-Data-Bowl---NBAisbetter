"""End-to-end pipeline: preprocess → metrics → validation → figures → CSVs."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from . import validation as V
from . import viz
from .data_loading import load_pff, load_players
from .preprocessing import build_play_table, process_week, sanity_checks


def compute(weeks=range(1, 9), force: bool = False, verbose: bool = True) -> dict:
    """Build the play-level table (metrics + flags) and the frame tables for `weeks`."""
    t0 = time.time()
    plays, log = build_play_table(verbose=verbose)
    plays = plays[plays.week.isin(list(weeks))]
    per_play, windows, dists, checks = [], [], [], []
    for w in weeks:
        fr, wi = process_week(w, plays, force=force)
        chk = sanity_checks(fr, wi)
        chk["week"] = w
        checks.append(chk)
        ft = M.frame_table(fr)
        per_play.append(M.play_metrics(ft))
        windows.append(wi)
        dists.append(M.rusher_distances(fr))
        if verbose:
            print(f"week {w}: {wi.has_snap.sum()} plays with snap, {len(per_play[-1])} with metrics "
                  f"({time.time() - t0:.0f}s)")
    m = pd.concat(per_play, ignore_index=True)
    windows = pd.concat(windows, ignore_index=True)
    df = plays.merge(m, on=["gameId", "playId"], how="inner")
    df = M.add_pis(df)
    df["ttt_outlier"] = df.ttt_outlier.astype(int)

    n_missing = len(plays) - len(df)
    log = pd.concat([log, pd.DataFrame([
        {"step": f"weeks {min(weeks)}-{max(weeks)}", "plays_remaining": len(plays)},
        {"step": "has ball_snap + end event in tracking", "plays_remaining": len(df)}])], ignore_index=True)
    checks = pd.DataFrame(checks)
    if verbose:
        print(log.to_string(index=False))
        print(checks.to_string(index=False))
        print(f"time_to_throw outliers (outside {C.TTT_MIN}-{C.TTT_MAX}s, flagged not dropped): "
              f"{df.ttt_outlier.sum()} · plays lost in tracking: {n_missing}")
    return {"plays": df, "log": log, "checks": checks, "windows": windows,
            "dist": pd.concat(dists, ignore_index=True)}


def analyse(res: dict, make_figures: bool = True, verbose: bool = True) -> dict:
    df = res["plays"]
    out = C.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Metric 2: Kaplan-Meier tables ----------------
    km_tabs = pd.concat([
        M.km_table(df, "rush_bin", order=C.RUSH_ORDER),
        M.km_table(df, "box_bin", order=C.BOX_ORDER),
        M.km_table(df, "pff_passCoverageType", order=["Man", "Zone", "Other"]),
        M.km_table(df, "offenseFormation"),
    ], ignore_index=True)
    rad_rows = []
    for r in C.RADII:
        tag = str(r).replace(".", "_")
        t = M.km_table(df, "rush_bin", time_col=f"time_to_pressure_r{tag}",
                       event_col=f"pressure_event_r{tag}", order=C.RUSH_ORDER)
        t.insert(0, "radius", r)
        rad_rows.append(t)
    radius_tab = pd.concat(rad_rows, ignore_index=True)
    four, six = df[df.rush_bin.astype(str).eq("4")], df[df.rush_bin.astype(str).eq("6+")]
    p_lr = M.logrank_test(four.time_to_pressure, four.pressure_event, six.time_to_pressure, six.pressure_event)

    # ---------------- Teams + validation ----------------
    teams = M.team_table(df)
    df_noPA = M.add_pis(df[df.play_action.eq(0) & df.pocket_roll.eq(0)])
    teams_noPA = M.team_table(df_noPA)[["possessionTeam", "PIS_adj", "rank_PIS_adj"]]
    teams = teams.merge(teams_noPA.rename(columns={"PIS_adj": "PIS_adj_noPA_noPR",
                                                   "rank_PIS_adj": "rank_PIS_adj_noPA_noPR"}),
                        on="possessionTeam", how="left")
    play_val = V.play_level(df)
    team_corr, disagree = V.team_level(teams)
    sens = V.weight_sensitivity(df, C.MIN_TEAM_PLAYS)
    subsets = V.subset_summary(df)
    confusion = V.tracking_vs_pff(df)
    blockers = M.blocker_table(res["dist"], load_pff(), df, load_players())

    # ---------------- CSV outputs ----------------
    play_cols = ["gameId", "playId", "week", "possessionTeam", "defensiveTeam", "quarter", "gameClock",
                 "down", "yardsToGo", "passResult", "offenseFormation", "defendersInBox", "box_bin",
                 "n_rushers", "rush_bin", "blitz", "pff_passCoverage", "pff_passCoverageType",
                 "play_action", "pocket_roll", "scramble_dropback", "dropBackType",
                 "snap_frame", "end_frame", "time_to_throw", "ttt_outlier",
                 "cushion", "clean_time_pct", "pocket_area", "pocket_area_throw", "d_min_min", "d_min_throw",
                 "PIS", "PIS_adj", "time_to_pressure", "pressure_event",
                 "time_to_pressure_r1_5", "pressure_event_r1_5", "time_to_pressure_r3_0", "pressure_event_r3_0",
                 "pressure_pff", "hit_pff", "sack", "beaten"]
    df[play_cols].round(4).to_csv(out / "play_summary.csv", index=False, encoding="utf-8-sig")
    teams.to_csv(out / "team_ranking.csv", index=False, encoding="utf-8-sig")
    km_tabs.round(3).to_csv(out / "time_to_pressure_by_group.csv", index=False, encoding="utf-8-sig")
    radius_tab.round(3).to_csv(out / "radius_sensitivity.csv", index=False, encoding="utf-8-sig")
    play_val.to_csv(out / "validation_play_level.csv", index=False, encoding="utf-8-sig")
    team_corr.to_csv(out / "validation_team_level.csv", index=False, encoding="utf-8-sig")
    disagree.to_csv(out / "teams_pis_vs_sacks_disagreement.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out / "pis_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    subsets.to_csv(out / "subset_summary_PA_PR.csv", index=False, encoding="utf-8-sig")
    blockers.to_csv(out / "blocker_ranking.csv", index=False, encoding="utf-8-sig")
    res["log"].to_csv(out / "exclusions.csv", index=False, encoding="utf-8-sig")

    figs = {}
    if make_figures:
        auc = play_val.query("label == 'pressure_pff' and score.str.startswith('PIS (')").AUC.iat[0]
        rho = team_corr.query("team_score == 'PIS_adj' and vs == 'pressure_rate_pff'").spearman_rho.iat[0]
        figs["km_rushers"] = viz.km_by_rushers(df)
        figs["km_box_small_multiples"] = viz.km_small_multiples(df)
        figs["km_coverage_formation"] = viz.km_by_category(df)
        figs["radius_sensitivity"] = viz.radius_sensitivity_fig(radius_tab)
        figs["team_ranking"] = viz.team_scatter(teams, rho)
        figs["validation_boxplots"] = viz.validation_boxplots(df, auc)
        figs["pis_vs_time_to_throw"] = viz.pis_vs_ttt(df)
        for name, f in figs.items():
            viz.save(f, name, width=1500 if "small" in name else 1100)
        sack, clean = pick_examples(df)
        figs["replay_sack"] = viz.replay(*sack, title_prefix="Collapsed pocket (sack): ")
        figs["replay_clean"] = viz.replay(*clean, title_prefix="Solid pocket: ")
        viz.save_replay(figs["replay_sack"], "replay_sack")
        viz.save_replay(figs["replay_clean"], "replay_clean")

    if verbose:
        print("\nPlay-level validation\n", play_val.to_string(index=False))
        print("\nTeam-level validation\n", team_corr.to_string(index=False))
        print("\nTime to pressure by group\n", km_tabs.to_string(index=False))
        print(f"\nLog-rank 4 vs 6+ rushers: p = {p_lr:.3g}")
        print("\nRadius sensitivity\n", radius_tab.to_string(index=False))
        print("\nWeight sensitivity\n", sens.to_string(index=False))
        print("\nSubsets\n", subsets.to_string(index=False))
        print("\nTracking vs PFF\n", confusion)
        print("\nTeams\n", teams.to_string(index=False))
        print("\nPIS vs sack disagreement\n", disagree.to_string(index=False))
    return {"km_tables": km_tabs, "radius": radius_tab, "teams": teams, "play_validation": play_val,
            "team_validation": team_corr, "disagree": disagree, "weight_sensitivity": sens,
            "subsets": subsets, "confusion": confusion, "blockers": blockers, "logrank_4_vs_6": p_lr,
            "figures": figs}


def pick_examples(df: pd.DataFrame) -> tuple[tuple[int, int], tuple[int, int]]:
    """A sack with a collapsed pocket and a clean, long-developing pass with a solid pocket."""
    s = df[df.sack.eq(1) & df.pressure_event.eq(1) & df.play_action.eq(0) & df.ttt_outlier.eq(0)
           & df.time_to_throw.between(2.5, 4.5)]
    sack = s.sort_values("PIS").iloc[0]
    c = df[df.pressure_pff.eq(0) & df.pressure_event.eq(0) & df.play_action.eq(0)
           & df.passResult.eq("C") & df.time_to_throw.between(3.0, 4.0) & df.pocket_roll.eq(0)]
    clean = c.sort_values("PIS", ascending=False).iloc[0]
    return (int(sack.gameId), int(sack.playId)), (int(clean.gameId), int(clean.playId))
