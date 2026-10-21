"""Pocket Integrity Score (PIS), Blitz Time Pressure (BTP), Kaplan-Meier, blocker stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C

KEY = ["gameId", "playId"]
FKEY = ["gameId", "playId", "frameId"]


# --------------------------------------------------------------------------- #
# Frame-level geometry
# --------------------------------------------------------------------------- #
def rusher_distances(frames: pd.DataFrame) -> pd.DataFrame:
    """Distance from the QB to every rusher at every frame of the window."""
    qb = frames.loc[frames.role.eq(C.ROLE_QB), FKEY + ["x", "y"]].drop_duplicates(FKEY)
    ru = frames.loc[frames.role.eq(C.ROLE_RUSH), FKEY + ["nflId", "x", "y", "snap_frame", "end_frame"]]
    m = ru.merge(qb, on=FKEY, suffixes=("", "_qb"))
    m["dist"] = np.hypot(m.x - m.x_qb, m.y - m.y_qb)
    return m[FKEY + ["nflId", "dist", "snap_frame", "end_frame"]]


def convex_hull(pts) -> list[tuple[float, float]]:
    """Andrew's monotone chain. Same hull as scipy's ConvexHull, but ~50x faster
    than a Qhull call for the 5-8 blocker points of a single frame."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return np.nan
    h = np.asarray(convex_hull(pts))
    if len(h) < 3:
        return 0.0  # collinear blockers: degenerate pocket
    x, y = h[:, 0], h[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def pocket_areas(frames: pd.DataFrame) -> pd.DataFrame:
    """Convex-hull area (yd²) of the pass blockers at every frame."""
    b = frames.loc[frames.role.eq(C.ROLE_BLOCK), FKEY + ["x", "y"]].sort_values(FKEY)
    keys = b[FKEY].to_numpy()
    xy = b[["x", "y"]].to_numpy()
    # split rows into frame groups without a slow pandas groupby-apply
    brk = np.flatnonzero((np.diff(keys, axis=0) != 0).any(axis=1)) + 1
    starts = np.r_[0, brk]
    areas = [_hull_area(chunk) for chunk in np.split(xy, brk)]
    out = pd.DataFrame(keys[starts], columns=FKEY)
    out["pocket_area"] = areas
    return out


def frame_table(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, frame): d_min to nearest rusher and pocket area."""
    d = rusher_distances(frames)
    dmin = d.groupby(FKEY).agg(d_min=("dist", "min"), snap_frame=("snap_frame", "first"),
                               end_frame=("end_frame", "first")).reset_index()
    return dmin.merge(pocket_areas(frames), on=FKEY, how="left")


# --------------------------------------------------------------------------- #
# Play-level metrics
# --------------------------------------------------------------------------- #
def time_to_pressure(ft: pd.DataFrame, radius: float) -> pd.DataFrame:
    """First entry of any rusher into `radius` yards of the QB after the snap.

    Frames at the very start of the play where a rusher is already inside the
    radius (e.g. a nose tackle over an under-center QB) are ignored until the
    first frame outside the radius, so alignment is not counted as pressure.
    Plays with no entry before the throw are right-censored at time_to_throw.
    """
    f = ft.sort_values(FKEY)
    inside = f.d_min.le(radius) & f.frameId.gt(f.snap_frame)
    was_out = (~f.d_min.le(radius)).groupby([f.gameId, f.playId]).cummax()
    hit = f[inside & was_out]
    first = hit.groupby(KEY).frameId.min().rename("p_frame")
    w = f.groupby(KEY).agg(snap_frame=("snap_frame", "first"), end_frame=("end_frame", "first"))
    w = w.join(first)
    tag = str(radius).replace(".", "_")
    out = pd.DataFrame(index=w.index)
    out[f"pressure_event_r{tag}"] = w.p_frame.notna().astype(int)
    out[f"time_to_pressure_r{tag}"] = (w.p_frame.fillna(w.end_frame) - w.snap_frame) / C.FPS
    return out.reset_index()


def play_metrics(ft: pd.DataFrame) -> pd.DataFrame:
    g = ft.groupby(KEY)
    m = g.agg(snap_frame=("snap_frame", "first"), end_frame=("end_frame", "first"),
              cushion=("d_min", "mean"), d_min_min=("d_min", "min"),
              pocket_area=("pocket_area", "mean")).reset_index()
    clean = ft.assign(c=ft.d_min.gt(C.PRESSURE_RADIUS)).groupby(KEY).c.mean().rename("clean_time_pct")
    at_end = (ft[ft.frameId.eq(ft.end_frame)].set_index(KEY)
              [["pocket_area", "d_min"]].rename(columns={"pocket_area": "pocket_area_throw",
                                                         "d_min": "d_min_throw"}))
    m = m.merge(clean, on=KEY).merge(at_end, on=KEY, how="left")
    m["time_to_throw"] = (m.end_frame - m.snap_frame) / C.FPS
    m["ttt_outlier"] = (~m.time_to_throw.between(C.TTT_MIN, C.TTT_MAX)).astype(int)
    for r in C.RADII:
        m = m.merge(time_to_pressure(ft, r), on=KEY)
    m["time_to_pressure"] = m["time_to_pressure_r2_0"]
    m["pressure_event"] = m["pressure_event_r2_0"]
    return m


# --------------------------------------------------------------------------- #
# Pocket Integrity Score
# --------------------------------------------------------------------------- #
def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def pis_score(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """PIS = 100 * Phi(Z / sd(Z)), Z = sum_k w_k * z(metric_k).

    Cushion is winsorised at the 1st/99th percentile before z-scoring so a
    handful of plays with rushers dropping far away do not dominate.
    50 = league-average pocket, ~84 = one standard deviation cleaner.
    """
    w = weights or C.PIS_WEIGHTS
    parts = {
        "cushion": df.cushion.clip(*df.cushion.quantile([0.01, 0.99])),
        "clean_time_pct": df.clean_time_pct,
        "pocket_area": df.pocket_area.fillna(df.pocket_area.median()),
    }
    z = sum(w[k] * _z(parts[k]) for k in w if w[k] != 0)
    return pd.Series(100 * norm.cdf(z / z.std(ddof=0)), index=df.index)


def add_pis(df: pd.DataFrame) -> pd.DataFrame:
    """Add PIS and its time-to-throw-adjusted version (OLS residual + 50)."""
    df = df.copy()
    df["PIS"] = pis_score(df)
    slope, intercept = np.polyfit(df.time_to_throw, df.PIS, 1)
    df["PIS_pred_ttt"] = intercept + slope * df.time_to_throw
    df["PIS_adj"] = 50 + df.PIS - df.PIS_pred_ttt
    df.attrs["ttt_slope"] = slope
    df.attrs["ttt_intercept"] = intercept
    return df


WEIGHT_SETS = {
    "baseline (.45/.40/.15)": C.PIS_WEIGHTS,
    "equal (1/3 each)": {"cushion": 1 / 3, "clean_time_pct": 1 / 3, "pocket_area": 1 / 3},
    "no area (.5/.5/0)": {"cushion": 0.5, "clean_time_pct": 0.5, "pocket_area": 0.0},
    "cushion heavy (.7/.2/.1)": {"cushion": 0.7, "clean_time_pct": 0.2, "pocket_area": 0.1},
    "clean heavy (.2/.7/.1)": {"cushion": 0.2, "clean_time_pct": 0.7, "pocket_area": 0.1},
    "area heavy (.3/.3/.4)": {"cushion": 0.3, "clean_time_pct": 0.3, "pocket_area": 0.4},
}


# --------------------------------------------------------------------------- #
# Kaplan-Meier (manual, no lifelines dependency)
# --------------------------------------------------------------------------- #
def kaplan_meier(durations, events) -> pd.DataFrame:
    """KM survival with Greenwood 95% CI. events=1 pressure observed, 0 censored."""
    t = np.asarray(durations, float)
    e = np.asarray(events, int)
    times = np.unique(t[e == 1])
    n_at_risk = np.array([(t >= u).sum() for u in times])
    d = np.array([((t == u) & (e == 1)).sum() for u in times])
    s = np.cumprod(1 - d / n_at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(d / (n_at_risk * (n_at_risk - d)))
        se = np.nan_to_num(s * np.sqrt(gw), nan=0.0)  # last step with S=0 has an infinite term
    out = pd.DataFrame({"time": np.r_[0.0, times], "survival": np.r_[1.0, s],
                        "at_risk": np.r_[len(t), n_at_risk], "events": np.r_[0, d],
                        "lo": np.r_[1.0, np.clip(s - 1.96 * se, 0, 1)],
                        "hi": np.r_[1.0, np.clip(s + 1.96 * se, 0, 1)]})
    return out


def km_median(km: pd.DataFrame) -> float:
    below = km[km.survival <= 0.5]
    return float(below.time.iloc[0]) if len(below) else np.nan


def km_at(km: pd.DataFrame, t: float) -> float:
    return float(km.loc[km.time <= t, "survival"].iloc[-1])


def logrank_test(d1, e1, d2, e2) -> float:
    """Two-group log-rank p-value."""
    from scipy.stats import chi2
    t = np.r_[d1, d2]; e = np.r_[e1, e2]; g = np.r_[np.zeros(len(d1)), np.ones(len(d2))]
    o_minus_e, var = 0.0, 0.0
    for u in np.unique(t[e == 1]):
        at = t >= u
        n, n1 = at.sum(), (at & (g == 0)).sum()
        dd = ((t == u) & (e == 1)).sum()
        d1u = ((t == u) & (e == 1) & (g == 0)).sum()
        o_minus_e += d1u - dd * n1 / n
        if n > 1:
            var += dd * (n1 / n) * (1 - n1 / n) * (n - dd) / (n - 1)
    return float(chi2.sf(o_minus_e ** 2 / var, 1)) if var > 0 else np.nan


def km_table(df: pd.DataFrame, by: str, time_col="time_to_pressure", event_col="pressure_event",
             order=None) -> pd.DataFrame:
    rows = []
    groups = order if order is not None else sorted(df[by].dropna().unique())
    for gname in groups:
        sub = df[df[by].astype(str).eq(str(gname))]
        if len(sub) < 10:
            continue
        km = kaplan_meier(sub[time_col], sub[event_col])
        rows.append({"variable": by, "group": str(gname), "n_plays": len(sub),
                     "pressure_events": int(sub[event_col].sum()),
                     "censored_pct": round(100 * (1 - sub[event_col].mean()), 1),
                     "median_time_to_pressure_s": km_median(km),
                     "S(1.5s)": round(km_at(km, 1.5), 3), "S(2.5s)": round(km_at(km, 2.5), 3),
                     "mean_time_to_throw_s": round(sub.time_to_throw.mean(), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Team aggregation
# --------------------------------------------------------------------------- #
def team_table(df: pd.DataFrame, min_plays: int = C.MIN_TEAM_PLAYS) -> pd.DataFrame:
    g = df.groupby("possessionTeam")
    t = g.agg(n_plays=("PIS", "size"), PIS=("PIS", "mean"), PIS_adj=("PIS_adj", "mean"),
              time_to_throw=("time_to_throw", "mean"), cushion=("cushion", "mean"),
              clean_time_pct=("clean_time_pct", "mean"), pocket_area=("pocket_area", "mean"),
              pressure_rate_pff=("pressure_pff", "mean"), sack_rate=("sack", "mean"),
              beaten_rate=("beaten", "mean"), tracking_pressure_rate=("pressure_event", "mean"))
    t["median_time_to_pressure_s"] = [
        km_median(kaplan_meier(s.time_to_pressure, s.pressure_event)) for _, s in g]
    t = t[t.n_plays >= min_plays].reset_index()
    t["rank_PIS"] = t.PIS.rank(ascending=False).astype(int)
    t["rank_PIS_adj"] = t.PIS_adj.rank(ascending=False).astype(int)
    return t.sort_values("rank_PIS_adj").round(4)


# --------------------------------------------------------------------------- #
# Stretch: per-blocker analysis
# --------------------------------------------------------------------------- #
def blocker_table(dist: pd.DataFrame, pff: pd.DataFrame, plays: pd.DataFrame,
                  players: pd.DataFrame, radius: float = C.PRESSURE_RADIUS,
                  min_blocks: int = 100) -> pd.DataFrame:
    """For each blocker: how often / how fast his assigned defender reaches the QB."""
    d = dist.sort_values(FKEY)
    inside = d.dist.le(radius) & d.frameId.gt(d.snap_frame)
    was_out = (~d.dist.le(radius)).groupby([d.gameId, d.playId, d.nflId]).cummax()
    first = d[inside & was_out].groupby(KEY + ["nflId"]).frameId.min().rename("p_frame")
    r = d.groupby(KEY + ["nflId"]).agg(snap=("snap_frame", "first"), end=("end_frame", "first")).join(first)
    r["reached"] = r.p_frame.notna().astype(int)
    r["t_reach"] = (r.p_frame.fillna(r.end) - r.snap) / C.FPS
    r = r.reset_index().rename(columns={"nflId": "defenderId"})

    blk = pff.loc[pff.pff_role.eq(C.ROLE_BLOCK) & pff.pff_nflIdBlockedPlayer.notna(),
                  KEY + ["nflId", "pff_nflIdBlockedPlayer", "pff_positionLinedUp"]]
    blk = blk.rename(columns={"nflId": "blockerId", "pff_nflIdBlockedPlayer": "defenderId"})
    m = blk.merge(r, on=KEY + ["defenderId"]).merge(plays[KEY + ["possessionTeam"]], on=KEY)
    rows = []
    for bid, s in m.groupby("blockerId"):
        km = kaplan_meier(s.t_reach, s.reached)
        rows.append({"nflId": bid, "team": s.possessionTeam.mode().iat[0],
                     "position": s.pff_positionLinedUp.mode().iat[0], "n_blocks": len(s),
                     "defender_reached_QB_pct": round(100 * s.reached.mean(), 1),
                     "S(2.5s)": round(km_at(km, 2.5), 3)})
    out = pd.DataFrame(rows).merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
    out = out[out.n_blocks >= min_blocks].sort_values("defender_reached_QB_pct")
    return out.reset_index(drop=True)
