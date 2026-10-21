"""Play filtering, snap-to-throw windowing, coordinate normalisation and parquet caching."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data_loading import load_games, load_pff, load_plays, load_tracking_game


# --------------------------------------------------------------------------- #
# Play-level table
# --------------------------------------------------------------------------- #
def _flag_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].fillna(0).eq(1).any(axis=1)


def build_play_table(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept plays with PFF labels and flags, exclusion log)."""
    plays = load_plays()
    games = load_games()
    pff = load_pff()

    p = plays.merge(games[["gameId", "season", "week"]], on="gameId", how="left")

    pff = pff.assign(
        is_rush=pff.pff_role.eq(C.ROLE_RUSH),
        is_block=pff.pff_role.eq(C.ROLE_BLOCK),
        is_pressure=_flag_any(pff, ["pff_hit", "pff_hurry", "pff_sack"]),
        is_hit=_flag_any(pff, ["pff_hit"]),
        is_beaten=_flag_any(pff, ["pff_beatenByDefender", "pff_hitAllowed",
                                  "pff_hurryAllowed", "pff_sackAllowed"]),
        is_pr=pff.pff_blockType.eq("PR"),
    )
    agg = pff.groupby(["gameId", "playId"]).agg(
        n_rushers=("is_rush", "sum"),
        n_blockers=("is_block", "sum"),
        pressure_pff=("is_pressure", "max"),
        hit_pff=("is_hit", "max"),
        beaten=("is_beaten", "max"),
        pocket_roll=("is_pr", "max"),
    ).reset_index()
    qb = (pff.loc[pff.pff_role.eq(C.ROLE_QB), ["gameId", "playId", "nflId"]]
          .drop_duplicates(["gameId", "playId"]).rename(columns={"nflId": "qb_nflId"}))
    p = p.merge(agg, on=["gameId", "playId"], how="left").merge(qb, on=["gameId", "playId"], how="left")
    for c in ["pressure_pff", "hit_pff", "beaten", "pocket_roll"]:
        p[c] = p[c].fillna(False).astype(int)
    p["n_rushers"] = p["n_rushers"].fillna(0).astype(int)
    p["sack"] = p.passResult.eq("S").astype(int)
    p["play_action"] = p.pff_playAction.fillna(0).astype(int)
    p["scramble_dropback"] = p.dropBackType.fillna("").str.contains("SCRAMBLE").astype(int)

    # Grouping variables
    p["rush_bin"] = pd.Categorical(
        np.select([p.n_rushers <= 3, p.n_rushers == 4, p.n_rushers == 5], ["≤3", "4", "5"], "6+"),
        categories=C.RUSH_ORDER, ordered=True)
    box = p.defendersInBox
    p["box_bin"] = pd.Categorical(
        np.select([box <= 5, box == 6, box == 7, box >= 8], ["≤5", "6", "7", "8+"], "NA"),
        categories=C.BOX_ORDER, ordered=True)
    p["blitz"] = (p.n_rushers >= 6).astype(int)

    log = [("all plays in plays.csv", len(p))]
    p = p[p.passResult.isin(C.KEEP_PASS_RESULTS)]
    log.append(("passResult in C/I/S/IN (drops scrambles R)", len(p)))
    p = p[p.n_rushers > 0]
    log.append(("at least one 'Pass Rush' player", len(p)))
    p = p[p.qb_nflId.notna()]
    log.append(("has a 'Pass' (QB) player", len(p)))
    p = p.reset_index(drop=True)
    p["qb_nflId"] = p.qb_nflId.astype("int64")

    log_df = pd.DataFrame(log, columns=["step", "plays_remaining"])
    if verbose:
        print(log_df.to_string(index=False))
    return p, log_df


# --------------------------------------------------------------------------- #
# Tracking: windowing and normalisation
# --------------------------------------------------------------------------- #
def normalise_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Flip plays going left so the offense always moves toward increasing x."""
    left = df.playDirection.eq("left")
    df = df.copy()
    df.loc[left, "x"] = C.FIELD_X - df.loc[left, "x"]
    df.loc[left, "y"] = C.FIELD_Y - df.loc[left, "y"]
    for col in ("o", "dir"):
        if col in df:
            df.loc[left, col] = (df.loc[left, col] + 180.0) % 360.0
    return df


def play_windows(track: pd.DataFrame) -> pd.DataFrame:
    """Snap frame and end frame (first throw/sack event after the snap) per play."""
    ev = track.loc[track.event.notna(), ["gameId", "playId", "frameId", "event"]].drop_duplicates()
    snap = (ev[ev.event.eq(C.SNAP_EVENT)].groupby(["gameId", "playId"]).frameId.min()
            .rename("snap_frame"))
    ev = ev.merge(snap, on=["gameId", "playId"])
    after = ev[ev.frameId > ev.snap_frame]

    end = (after[after.event.isin(C.END_EVENTS)].sort_values("frameId")
           .groupby(["gameId", "playId"]).first()[["frameId", "event"]]
           .rename(columns={"frameId": "end_frame", "event": "end_event"}))
    fb = (after[after.event.eq(C.END_FALLBACK)].groupby(["gameId", "playId"]).frameId.min()
          .rename("fb_frame"))
    w = snap.to_frame().join(end, how="left").join(fb, how="left").reset_index()
    w["end_fallback"] = w.end_frame.isna() & w.fb_frame.notna()
    w.loc[w.end_fallback, "end_event"] = C.END_FALLBACK
    w["end_frame"] = w.end_frame.fillna(w.fb_frame)
    w = w.drop(columns="fb_frame")
    return w


def process_game(game_id: int, plays_g: pd.DataFrame, pff_roles: pd.DataFrame):
    """Window + normalise + attach roles for one game. Returns (frames, windows)."""
    t = load_tracking_game(game_id)
    t = t.merge(plays_g[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"])
    w = play_windows(t)
    all_ids = plays_g[["gameId", "playId"]]
    w = all_ids.merge(w, on=["gameId", "playId"], how="left")
    w["has_snap"] = w.snap_frame.notna()
    w["has_end"] = w.end_frame.notna()
    ok = w[w.has_snap & w.has_end].copy()
    ok["snap_frame"] = ok.snap_frame.astype(int)
    ok["end_frame"] = ok.end_frame.astype(int)

    t = t.merge(ok[["gameId", "playId", "snap_frame", "end_frame"]], on=["gameId", "playId"])
    t = t[(t.frameId >= t.snap_frame) & (t.frameId <= t.end_frame)]
    t = normalise_xy(t)

    t["side"] = np.where(t.team.eq("football"), "ball",
                         np.where(t.team.eq(t.possessionTeam), "off", "def"))
    t = t.merge(pff_roles, on=["gameId", "playId", "nflId"], how="left")
    t["role"] = t.role.fillna(pd.Series(np.where(t.side.eq("ball"), "Ball", "Unknown"), index=t.index))

    # Line of scrimmage = ball x at the snap frame (normalised).
    los = (t[t.side.eq("ball") & t.frameId.eq(t.snap_frame)]
           .groupby(["gameId", "playId"]).x.first().rename("los_x"))
    ok = ok.merge(los, on=["gameId", "playId"], how="left")

    keep = ["gameId", "playId", "frameId", "nflId", "jerseyNumber", "side", "role",
            "x", "y", "s", "snap_frame", "end_frame"]
    return t[keep].reset_index(drop=True), pd.concat([ok, w[~(w.has_snap & w.has_end)]])


def process_week(week: int, play_table: pd.DataFrame, force: bool = False):
    """Process (or load from parquet cache) all games of a week."""
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    f_frames = C.PROCESSED_DIR / f"frames_week{week}.parquet"
    f_win = C.PROCESSED_DIR / f"windows_week{week}.parquet"
    if f_frames.exists() and f_win.exists() and not force:
        return pd.read_parquet(f_frames), pd.read_parquet(f_win)

    pff = load_pff()
    roles = pff[["gameId", "playId", "nflId", "pff_role"]].rename(columns={"pff_role": "role"})
    roles["nflId"] = roles.nflId.astype(float)
    pw = play_table[play_table.week.eq(week)]
    frames, wins = [], []
    for gid, g in pw.groupby("gameId"):
        fr, wi = process_game(gid, g, roles[roles.gameId.eq(gid)])
        frames.append(fr)
        wins.append(wi)
    frames = pd.concat(frames, ignore_index=True)
    wins = pd.concat(wins, ignore_index=True)
    frames.to_parquet(f_frames, index=False)
    wins.to_parquet(f_win, index=False)
    return frames, wins


def sanity_checks(frames: pd.DataFrame, windows: pd.DataFrame, tol: float = 3.0) -> dict:
    """Asserts on normalised coordinates and windows. Returns diagnostics."""
    ok = windows[windows.has_snap & windows.has_end]
    assert ok.snap_frame.notna().all(), "every kept play needs a snap frame"
    assert (ok.end_frame > ok.snap_frame).all(), "end frame must follow the snap"
    # Players can legitimately stand a little beyond the sideline/end line,
    # so the hard assert uses a small tolerance; strict violations are reported.
    assert frames.x.between(-tol, C.FIELD_X + tol).all(), "x outside field after normalisation"
    assert frames.y.between(-tol, C.FIELD_Y + tol).all(), "y outside field after normalisation"
    strict_out = (~frames.x.between(0, C.FIELD_X) | ~frames.y.between(0, C.FIELD_Y)).sum()
    ttt = (ok.end_frame - ok.snap_frame) / C.FPS
    return {
        "frames_rows": len(frames),
        "rows_strictly_outside_field": int(strict_out),
        "plays_without_snap": int((~windows.has_snap).sum()),
        "plays_without_end_event": int((windows.has_snap & ~windows.has_end).sum()),
        "plays_using_autoevent_end": int(ok.end_fallback.sum()),
        "ttt_outliers": int((~ttt.between(C.TTT_MIN, C.TTT_MAX)).sum()),
    }
