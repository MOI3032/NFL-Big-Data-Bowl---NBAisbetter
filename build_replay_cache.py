"""Export a compact, deploy-friendly tracking cache for the replays.

The raw tracking/ folder is ~810 MB (22 players, 10 Hz, every column, whole
play). The web app only needs x/y + role + jersey from the snap to a few frames
after the throw, for the plays we keep. This script writes that slice, already
normalised, to a single compressed parquet:

    data/replay/frames.parquet          (one row per player-frame)

`src.viz.play_frames` loads this file when it exists and falls back to the raw
CSV otherwise, so local runs keep working whether or not the cache is built.

    python build_replay_cache.py
"""
from __future__ import annotations

import time

import pandas as pd

from src import config as C
from src.data_loading import load_pff, load_players
from src.preprocessing import build_play_table, normalise_xy, play_windows

OUT_DIR = C.ROOT / "data" / "replay"
OUT_FILE = OUT_DIR / "frames.parquet"
PAD = 5  # frames kept before the snap and after the throw (matches viz default)

KEEP_COLS = ["gameId", "playId", "frameId", "nflId", "jerseyNumber", "team",
             "playDirection", "x", "y", "event"]


def build() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plays, _ = build_play_table(verbose=False)
    kept = plays[["gameId", "playId", "possessionTeam"]]
    pff = load_pff()
    roles = pff[["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"]]
    names = load_players()[["nflId", "displayName"]]

    parts = []
    games = sorted(kept.gameId.unique())
    for i, gid in enumerate(games, 1):
        raw = pd.read_csv(C.TRACKING_DIR / f"tracking_{gid}.csv", usecols=KEEP_COLS)
        g_plays = kept[kept.gameId.eq(gid)]
        raw = raw.merge(g_plays, on=["gameId", "playId"], how="inner")
        # window: snap - PAD .. end + PAD, using the same event logic as preprocessing
        win = play_windows(raw)[["gameId", "playId", "snap_frame", "end_frame"]].dropna()
        raw = raw.merge(win, on=["gameId", "playId"], how="inner")
        raw = raw[raw.frameId.between(raw.snap_frame - PAD, raw.end_frame + PAD)]
        raw = normalise_xy(raw)
        raw["side"] = (raw.team.eq("football")).map({True: "ball"}).where(
            raw.team.eq("football"), raw.team.eq(raw.possessionTeam).map({True: "off", False: "def"}))
        raw = raw.merge(roles, on=["gameId", "playId", "nflId"], how="left").merge(names, on="nflId", how="left")
        raw["pff_role"] = raw.pff_role.fillna("Ball")
        parts.append(raw[["gameId", "playId", "frameId", "nflId", "jerseyNumber", "side",
                          "pff_role", "pff_positionLinedUp", "displayName", "x", "y", "event",
                          "snap_frame", "end_frame"]])
        if i % 20 == 0 or i == len(games):
            print(f"  {i}/{len(games)} games · {time.time() - t0:.0f}s")

    out = pd.concat(parts, ignore_index=True)
    # Sort by play so each play's rows sit together: a filtered read (used by the
    # web app) then touches only one small row group instead of the whole file.
    out = out.sort_values(["gameId", "playId", "frameId"]).reset_index(drop=True)
    # smaller dtypes
    out["frameId"] = out.frameId.astype("int32")
    out["snap_frame"] = out.snap_frame.astype("int32")
    out["end_frame"] = out.end_frame.astype("int32")
    out["x"] = out.x.astype("float32")
    out["y"] = out.y.astype("float32")
    out["nflId"] = out.nflId.astype("float32")
    out["jerseyNumber"] = out.jerseyNumber.astype("float32")
    # Keep these as plain strings, not category: a filtered read of one play
    # would otherwise still materialise the full category dictionary.
    for c in ["side", "pff_role", "pff_positionLinedUp", "displayName"]:
        out[c] = out[c].astype("string")
    # Small row groups (~50k rows) so a per-play filter reads a tiny slice.
    out.to_parquet(OUT_FILE, index=False, compression="zstd", row_group_size=50_000)

    mb = OUT_FILE.stat().st_size / 1e6
    print(f"\nWrote {OUT_FILE} · {len(out):,} rows · {out.playId.nunique()} play-slices across "
          f"{out.gameId.nunique()} games · {mb:.1f} MB · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    build()
