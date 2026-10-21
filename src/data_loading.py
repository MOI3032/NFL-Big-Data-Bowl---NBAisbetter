"""Raw CSV loading helpers.

Loaders are memoised: callers only derive new frames from them (pandas 3
copy-on-write), so the cached objects are never mutated in place.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import RAW_DIR, TRACKING_DIR


@lru_cache(maxsize=1)
def load_games() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "games.csv")


@lru_cache(maxsize=1)
def load_plays() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "plays.csv")


@lru_cache(maxsize=1)
def load_players() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "players.csv")


@lru_cache(maxsize=1)
def load_pff() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "pffScoutingData.csv")


TRACK_COLS = ["gameId", "playId", "nflId", "frameId", "jerseyNumber", "team",
              "playDirection", "x", "y", "s", "a", "o", "dir", "event"]


@lru_cache(maxsize=8)
def load_tracking_game(game_id: int) -> pd.DataFrame:
    """Load the tracking file of a single game (last 8 games kept in memory)."""
    return pd.read_csv(TRACKING_DIR / f"tracking_{game_id}.csv", usecols=TRACK_COLS)


def describe_raw(n_rows: int = 3) -> None:
    """Print the structure, columns and a sample of every raw file."""
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        for name, fn in [("games", load_games), ("plays", load_plays),
                         ("players", load_players), ("pffScoutingData", load_pff)]:
            df = fn()
            print(f"===== {name}.csv  shape={df.shape}")
            print(list(df.columns))
            print(df.head(n_rows).to_string(), "\n")
        files = sorted(TRACKING_DIR.glob("tracking_*.csv"))
        print(f"===== tracking/: {len(files)} files ({files[0].name} … {files[-1].name})")
        t = pd.read_csv(files[0])
        print(list(t.columns))
        print(t.head(n_rows).to_string())
        print("events in first file:", sorted(t.event.dropna().unique()))
