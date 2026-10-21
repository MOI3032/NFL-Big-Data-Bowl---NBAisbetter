"""Plotly figures: animated replay, pressure timeline, Kaplan-Meier curves, team ranking, validation.

Every user-facing string goes through `_t(en, fr)`. The pipeline exports in
English (default); the web app calls `set_lang("fr")` before building figures.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from . import metrics as M
from .data_loading import load_pff, load_players, load_plays, load_tracking_game
from .preprocessing import normalise_xy, play_windows

LANG = "en"


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang


def _t(en: str, fr: str) -> str:
    return fr if LANG == "fr" else en


def _num(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if LANG == "fr" else s


FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#1f2937")
RUSH_COLORS = {"≤3": "#2563eb", "4": "#16a34a", "5": "#f59e0b", "6+": "#dc2626"}
OUTCOME_KEYS = ["clean", "pressure", "sack"]
OUTCOME_COLORS = {"clean": "#16a34a", "pressure": "#f59e0b", "sack": "#dc2626"}
PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def _outcome_label(k: str) -> str:
    return {"clean": _t("Clean (PFF)", "Poche propre (PFF)"),
            "pressure": _t("Pressure, no sack", "Pression sans sack"),
            "sack": "Sack"}[k]


def save(fig: go.Figure, name: str, width=1100, height=650) -> None:
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(C.OUTPUT_DIR / f"{name}.html", include_plotlyjs="cdn")
    fig.write_image(C.OUTPUT_DIR / f"{name}.png", width=width, height=height, scale=2)


def _base_layout(fig, title, **kw):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=18)), font=FONT,
                      template="plotly_white", **kw)
    return fig


# --------------------------------------------------------------------------- #
# 1. Play data for replay + timeline (cached)
# --------------------------------------------------------------------------- #
def _hover(d: pd.DataFrame) -> list[str]:
    return (d.displayName.fillna("") + " #" + d.jerseyNumber.fillna(0).astype(int).astype(str)
            + "<br>" + d.pff_role + "<br>" + d.pff_positionLinedUp.fillna("")).tolist()


@lru_cache(maxsize=64)
def play_frames(game_id: int, play_id: int, pad_frames: int = 5, radius: float = C.PRESSURE_RADIUS):
    """Normalised tracking of one play + per-frame pocket summary.

    Returns dict(t=tracking rows, fs=frame summary, snap, end, end_event, play, los_x).
    """
    plays = load_plays()
    play = plays[(plays.gameId == game_id) & (plays.playId == play_id)].iloc[0]
    t = load_tracking_game(game_id)
    t = t[t.playId == play_id].copy()
    win = play_windows(t).iloc[0]
    snap, end = int(win.snap_frame), int(win.end_frame)
    t = normalise_xy(t)
    pff = load_pff()
    roles = pff[(pff.gameId == game_id) & (pff.playId == play_id)][["nflId", "pff_role", "pff_positionLinedUp"]]
    names = load_players()[["nflId", "displayName"]]
    t = t.merge(roles, on="nflId", how="left").merge(names, on="nflId", how="left")
    t["pff_role"] = t.pff_role.fillna("Ball")
    t["side"] = np.where(t.team.eq("football"), "ball",
                         np.where(t.team.eq(play.possessionTeam), "off", "def"))
    t = t[t.frameId.between(max(1, snap - pad_frames), end + pad_frames)].reset_index(drop=True)
    los_x = float(t[(t.side == "ball") & (t.frameId == snap)].x.iloc[0])

    rows = []
    for fid, f in t.groupby("frameId"):
        qb = f[f.pff_role == C.ROLE_QB]
        ru = f[f.pff_role == C.ROLE_RUSH]
        bl = f[f.pff_role == C.ROLE_BLOCK][["x", "y"]].to_numpy()
        qx, qy = float(qb.x.iloc[0]), float(qb.y.iloc[0])
        dist = np.hypot(ru.x - qx, ru.y - qy).to_numpy()
        j = int(np.argmin(dist)) if len(dist) else None
        rows.append({"frameId": fid, "t_s": (fid - snap) / C.FPS, "qb_x": qx, "qb_y": qy,
                     "d_min": float(dist[j]) if j is not None else np.nan,
                     "nearest": (ru.displayName.iloc[j] if j is not None else ""),
                     "area": M._hull_area(bl), "hull": M.convex_hull(bl)})
    fs = pd.DataFrame(rows)
    fs["in_window"] = fs.frameId.between(snap, end)
    w = fs[fs.in_window]  # same rule as metrics.time_to_pressure: first *entry* after the snap
    inside = (w.frameId > snap) & w.d_min.le(radius)
    was_out = (~w.d_min.le(radius)).cummax()
    hit = w[inside & was_out]
    p_frame = int(hit.frameId.iloc[0]) if len(hit) else None
    return dict(t=t, fs=fs, snap=snap, end=end, end_event=str(win.end_event), play=play,
                los_x=los_x, pressure_frame=p_frame)


# --------------------------------------------------------------------------- #
# 2. Animated play replay
# --------------------------------------------------------------------------- #
FIELD_TICKS = list(range(10, 111, 10))


def _field_ticktext():
    g = _t("G", "En-but")
    return [g, "10", "20", "30", "40", "50", "40", "30", "20", "10", g]


def _field_shapes(los_x: float | None, first_down_x: float | None, numbers: bool = True) -> tuple[list, list]:
    shapes = [dict(type="rect", x0=0, x1=C.FIELD_X, y0=0, y1=C.FIELD_Y, fillcolor="#4f8f3f",
                   line=dict(color="white", width=2), layer="below"),
              dict(type="rect", x0=0, x1=10, y0=0, y1=C.FIELD_Y, fillcolor="#356b2b", line_width=0, layer="below"),
              dict(type="rect", x0=110, x1=120, y0=0, y1=C.FIELD_Y, fillcolor="#356b2b", line_width=0, layer="below")]
    for x in range(10, 111, 5):
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=C.FIELD_Y, layer="below",
                           line=dict(color="rgba(255,255,255,0.7)", width=2 if x % 10 == 0 else 1)))
    # hash marks
    for x in range(11, 110):
        for y in (23.36, 29.96):
            shapes.append(dict(type="line", x0=x, x1=x, y0=y - 0.35, y1=y + 0.35, layer="below",
                               line=dict(color="rgba(255,255,255,0.45)", width=1)))
    ann = []
    if numbers:
        for x in range(20, 101, 10):
            num = x - 10 if x <= 60 else 110 - x
            for y in (7, C.FIELD_Y - 7):
                ann.append(dict(x=x, y=y, text=str(num), showarrow=False,
                                font=dict(color="rgba(255,255,255,0.8)", size=16)))
    if los_x is not None:
        shapes.append(dict(type="line", x0=los_x, x1=los_x, y0=0, y1=C.FIELD_Y,
                           line=dict(color="#1e40af", width=3)))
    if first_down_x is not None and first_down_x < 110:
        shapes.append(dict(type="line", x0=first_down_x, x1=first_down_x, y0=0, y1=C.FIELD_Y,
                           line=dict(color="#facc15", width=3)))
    return shapes, ann


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def _lst(s) -> list:
    return [float(v) for v in s] if s is not None else []


def replay_spec(game_id: int, play_id: int, title_prefix: str = "", radius: float = C.PRESSURE_RADIUS,
                pad_frames: int = 5, view: str = "field", show_title: bool = True) -> tuple[dict, int, list]:
    """Replay as a plain plotly.js figure dict (no Python-side validation, ~10x faster).

    Returns (figure_dict, index_of_end_frame, annotations_of_end_frame).
    """
    P = play_frames(game_id, play_id, pad_frames, radius)
    t, fs, snap, end, play, los_x = P["t"], P["fs"], P["snap"], P["end"], P["play"], P["los_x"]
    zoom = view == "pocket"
    shapes, field_ann = _field_shapes(los_x, los_x + play.yardsToGo, numbers=not zoom)
    msize = 20 if zoom else 13
    tsize = 10 if zoom else 8
    end_word = "sack" if "sack" in P["end_event"] else _t("throw", "lancer")
    names = {"pocket": _t("Pocket (blockers' hull)", "Poche (enveloppe des bloqueurs)"),
             "circle": _t(f"{radius:g}-yd circle around QB", f"Cercle de {_num(radius, 0)} yd autour du QB"),
             "off": _t("Offense", "Attaque") + f" ({play.possessionTeam})",
             "cov": _t("Coverage", "Couverture") + f" ({play.defensiveTeam})",
             "rush": "Pass rushers", "qb": "Quarterback", "ball": _t("Ball", "Ballon")}

    def pts(d, key, color, line, tcolor, size, symbol="circle", text=True):
        tr = dict(type="scatter", x=_lst(d.x), y=_lst(d.y), name=names[key], hoverinfo="text",
                  hovertext=_hover(d), mode="markers+text" if text else "markers",
                  marker=dict(size=size, color=color, symbol=symbol, line=line))
        if text:
            tr.update(text=d.jerseyNumber.fillna(0).astype(int).astype(str).tolist(),
                      textfont=dict(color=tcolor, size=tsize))
        return tr

    frames, frame_ann = [], []
    for (fid, f), s in zip(t.groupby("frameId"), fs.itertuples()):
        qb = f[f.pff_role == C.ROLE_QB]
        ru = f[f.pff_role == C.ROLE_RUSH]
        off = f[(f.side == "off") & (f.pff_role != C.ROLE_QB)]
        cov = f[(f.side == "def") & (f.pff_role != C.ROLE_RUSH)]
        ball = f[f.side == "ball"]
        hull = s.hull
        hx = [float(p[0]) for p in hull] + [float(hull[0][0])] if hull else []
        hy = [float(p[1]) for p in hull] + [float(hull[0][1])] if hull else []
        cx, cy = _circle(s.qb_x, s.qb_y, radius)
        danger = bool(s.in_window and s.d_min <= radius)
        phase = (_t("pre-snap", "avant le snap") if fid < snap
                 else _t(f"after the {end_word}", f"après le {end_word}") if fid > end
                 else f"snap → {end_word}")
        data = [
            dict(type="scatter", x=hx, y=hy, fill="toself", mode="lines", name=names["pocket"], hoverinfo="skip",
                 fillcolor="rgba(59,130,246,0.28)", line=dict(color="#1d4ed8", width=2)),
            dict(type="scatter", x=cx.tolist(), y=cy.tolist(), mode="lines", name=names["circle"], hoverinfo="skip",
                 line=dict(color="#dc2626" if danger else "#fde047", width=3, dash="dot")),
            pts(off, "off", "#2563eb", dict(color="white", width=1.5), "white", msize),
            pts(cov, "cov", "#fecaca", dict(color="#991b1b", width=1.5), "#7f1d1d", msize),
            pts(ru, "rush", "#b91c1c", dict(color="black", width=1.5), "white", msize + 3, "diamond"),
            pts(qb, "qb", "#f59e0b", dict(color="black", width=1.5), None, msize + 8, "star", text=False),
            dict(type="scatter", x=_lst(ball.x), y=_lst(ball.y), mode="markers", name=names["ball"], hoverinfo="skip",
                 marker=dict(size=9 if zoom else 7, color="#78350f", line=dict(color="white", width=1))),
        ]
        txt = (f"<b>t = {'+' if s.t_s >= 0 else ''}{_num(s.t_s)} s</b>  ({phase})<br>"
               + _t("Nearest rusher", "Rusher le plus proche") + f" : <b>{_num(s.d_min)} yd</b><br>"
               + _t("Pocket area", "Aire de la poche") + f" : <b>{_num(s.area, 0)} yd²</b>")
        ann = field_ann + [dict(x=1.0, y=1.01, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                                text=txt, showarrow=False, align="left", bgcolor="rgba(255,255,255,0.97)",
                                bordercolor="#dc2626" if danger else "#1f2937", borderwidth=2,
                                borderpad=6, font=dict(size=13))]
        frames.append(dict(data=data, name=str(fid), layout=dict(annotations=ann)))
        frame_ann.append(ann)

    start = next(i for i, fr in enumerate(frames) if int(fr["name"]) == snap)
    end_i = next(i for i, fr in enumerate(frames) if int(fr["name"]) == end)
    title = None
    if show_title:
        title = (f"{title_prefix}{play.possessionTeam} vs {play.defensiveTeam} — Q{play.quarter} {play.gameClock}, "
                 f"{play.down}&{play.yardsToGo}, result {play.passResult}<br>"
                 f"<sup>{str(play.playDescription)[:110]}</sup>"
                 "<br><sup>" + _t("Offense moves left → right · blue line = line of scrimmage · yellow = first-down line",
                                  "L'attaque va de gauche à droite · bleu = ligne de mêlée · jaune = ligne du 1er down")
                 + "<br>" + _t("shaded polygon = pocket formed by the pass blockers · circle turns red when a rusher "
                               f"is within {radius:g} yd of the QB",
                               "polygone bleu = poche formée par les bloqueurs · le cercle devient rouge quand un "
                               f"rusher est à moins de {_num(radius, 0)} yd du QB") + "</sup>")
    if zoom:
        qy = float(fs.loc[fs.frameId == snap, "qb_y"].iloc[0])
        xr = [los_x - 24, los_x + 14]
        yr = [max(-1, qy - 11), min(C.FIELD_Y + 1, qy + 11)]
    else:
        xr, yr = [-1, 121], [-1, 54.3]

    def lab(fr):
        s = (int(fr["name"]) - snap) / C.FPS
        return f"{_num(s)} s" if int(fr["name"]) % 5 == snap % 5 else ""

    steps = [dict(method="animate", label=lab(fr),
                  args=[[fr["name"]], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))]) for fr in frames]
    layout = dict(
        font=FONT, width=1150, height=800 if show_title else 720, shapes=shapes, annotations=frame_ann[start],
        margin=dict(l=20, r=20, t=150 if show_title else 70, b=150),
        legend=dict(orientation="h", y=-0.06, yanchor="top", x=0, font=dict(size=12)),
        plot_bgcolor="#356b2b", paper_bgcolor="white", hovermode="closest",
        xaxis=dict(range=xr, showgrid=False, zeroline=False, constrain="domain", tickvals=FIELD_TICKS,
                   ticktext=_field_ticktext()),
        yaxis=dict(range=yr, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain",
                   showticklabels=False),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.16, xanchor="left", yanchor="top",
                          showactive=False,
                          buttons=[dict(label=_t("▶ Play", "▶ Lecture"), method="animate",
                                        args=[None, dict(frame=dict(duration=100, redraw=True),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=start, steps=steps, x=0.2, len=0.8, y=-0.13, yanchor="top",
                      currentvalue=dict(visible=False), pad=dict(t=10))],
    )
    if title:
        layout["title"] = dict(text=title, x=0.01, font=dict(size=18))
    spec = dict(data=frames[start]["data"], layout=layout, frames=frames)
    return spec, end_i, frame_ann[end_i]


def replay(game_id: int, play_id: int, title_prefix: str = "", radius: float = C.PRESSURE_RADIUS,
           pad_frames: int = 5, view: str = "field", show_title: bool = True) -> go.Figure:
    """Animated replay of one play with the blockers' pocket and the QB's 2-yard circle.

    view="field": full 0-120 x 0-53.3 field. view="pocket": zoom on the pocket
    (equal aspect is kept in both views so the circle and the hull are not distorted).
    """
    spec, end_i, end_ann = replay_spec(game_id, play_id, title_prefix, radius, pad_frames, view, show_title)
    fig = go.Figure(spec)
    fig._replay_end_index = end_i
    fig._replay_end_ann = end_ann
    return fig


def save_replay(fig: go.Figure, name: str) -> None:
    """HTML keeps the animation; PNG is a still at the throw/sack frame."""
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(C.OUTPUT_DIR / f"{name}.html", include_plotlyjs="cdn", auto_play=False)
    lay = fig.layout.to_plotly_json()
    for k in ("sliders", "updatemenus"):
        lay.pop(k, None)
    lay["annotations"] = fig._replay_end_ann
    lay["margin"] = dict(l=20, r=20, t=150, b=80)
    still = go.Figure(data=fig.frames[fig._replay_end_index].data, layout=lay)
    still.update_layout(height=710)
    still.write_image(C.OUTPUT_DIR / f"{name}.png", width=1150, height=710, scale=2)


def play_timeline(game_id: int, play_id: int, radius: float = C.PRESSURE_RADIUS) -> go.Figure:
    """Distance of the nearest rusher and pocket area, second by second."""
    P = play_frames(game_id, play_id, 5, radius)
    fs, snap, end = P["fs"], P["snap"], P["end"]
    t_end = (end - snap) / C.FPS
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ymax = max(8.0, float(fs.d_min.max()) + 0.5)
    fig.add_hrect(y0=0, y1=radius, fillcolor="rgba(220,38,38,0.10)", line_width=0)
    fig.add_vrect(x0=fs.t_s.min(), x1=0, fillcolor="rgba(107,114,128,0.08)", line_width=0)
    fig.add_vrect(x0=t_end, x1=fs.t_s.max(), fillcolor="rgba(107,114,128,0.08)", line_width=0)
    fig.add_trace(go.Scatter(x=fs.t_s, y=fs.area, name=_t("Pocket area (yd²)", "Aire de la poche (yd²)"),
                             mode="lines", line=dict(color="#93c5fd", width=2, dash="dot"),
                             hovertemplate="%{y:.0f} yd²<extra></extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=fs.t_s, y=fs.d_min, name=_t("Nearest rusher → QB (yd)", "Rusher le plus proche → QB (yd)"),
                             mode="lines+markers", line=dict(color="#1f2937", width=3), marker=dict(size=4),
                             customdata=fs.nearest,
                             hovertemplate="%{x:.1f} s · %{y:.1f} yd<br>%{customdata}<extra></extra>"))
    fig.add_hline(y=radius, line=dict(color="#dc2626", dash="dash"),
                  annotation_text=_t(f"pressure zone (< {radius:g} yd)", f"zone de pression (< {_num(radius, 0)} yd)"),
                  annotation_position="bottom right", annotation_font_color="#dc2626")
    fig.add_vline(x=0, line=dict(color="#6b7280", width=1))
    fig.add_vline(x=t_end, line=dict(color="#111827", width=2),
                  annotation_text=("Sack" if "sack" in P["end_event"] else _t("Throw", "Lancer")) + f" {_num(t_end)} s",
                  annotation_position="top")
    if P["pressure_frame"] is not None:
        tp = (P["pressure_frame"] - snap) / C.FPS
        dp = float(fs.loc[fs.frameId == P["pressure_frame"], "d_min"].iloc[0])
        fig.add_trace(go.Scatter(x=[tp], y=[dp], mode="markers+text", name=_t("First pressure", "Première pression"),
                                 text=[_t(f"pressure at {tp:.1f} s", f"pression à {_num(tp)} s")],
                                 textposition="top right", marker=dict(color="#dc2626", size=13, symbol="x"),
                                 textfont=dict(color="#dc2626")))
    fig.update_layout(font=FONT, template="plotly_white", height=300, margin=dict(l=50, r=50, t=20, b=40),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(title=_t("Seconds after the snap", "Secondes après le snap"), dtick=0.5)
    fig.update_yaxes(title=_t("Distance (yd)", "Distance (yd)"), range=[0, ymax], secondary_y=False)
    fig.update_yaxes(title=_t("Area (yd²)", "Aire (yd²)"), showgrid=False, secondary_y=True, rangemode="tozero",
                     tickformat=".0f", nticks=5)
    return fig


# --------------------------------------------------------------------------- #
# 3. Kaplan-Meier curves
# --------------------------------------------------------------------------- #
def _hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _add_km(fig, sub, label, color, row=None, col=None, show_ci=True, showlegend=True,
            time_col="time_to_pressure", event_col="pressure_event", plain_name=False):
    km = M.kaplan_meier(sub[time_col], sub[event_col])
    med = M.km_median(km)
    kw = dict(row=row, col=col) if row else {}
    if show_ci:
        fig.add_trace(go.Scatter(x=km.time, y=km.hi, mode="lines", line=dict(width=0, shape="hv"),
                                 showlegend=False, hoverinfo="skip", legendgroup=label), **kw)
        fig.add_trace(go.Scatter(x=km.time, y=km.lo, mode="lines", line=dict(width=0, shape="hv"),
                                 fill="tonexty", fillcolor=_hex_rgba(color, 0.15), showlegend=False,
                                 hoverinfo="skip", legendgroup=label), **kw)
    if plain_name:
        name = label
    elif np.isfinite(med):
        name = f"{label} (n={len(sub)}, " + _t("median", "médiane") + f" {_num(med)} s)"
    else:
        name = f"{label} (n={len(sub)})"
    fig.add_trace(go.Scatter(x=km.time, y=km.survival, mode="lines", name=name, legendgroup=label,
                             showlegend=showlegend, line=dict(color=color, width=3, shape="hv"),
                             hovertemplate="%{x:.1f} s : %{y:.0%} " + _t("of pockets still clean", "des poches encore propres")
                                           + "<extra>" + label + "</extra>"), **kw)


def km_by_rushers(df: pd.DataFrame, radius_label="2 yards") -> go.Figure:
    fig = go.Figure()
    for b in C.RUSH_ORDER:
        sub = df[df.rush_bin.astype(str).eq(b)]
        if len(sub) >= 10:
            _add_km(fig, sub, f"{b} rushers", RUSH_COLORS[b])
    fig.add_hline(y=0.5, line=dict(color="grey", dash="dash"),
                  annotation_text=_t("50%: median time to pressure", "50 % : temps médian avant pression"),
                  annotation_position="top right")
    _base_layout(fig, _t("How long does the pocket stay clean? Pocket survival by number of pass rushers",
                         "Combien de temps la poche reste-t-elle propre ? Par nombre de pass rushers")
                 + "<br><sup>" + _t(f"Share of dropbacks with no rusher within {radius_label} of the QB · plays where the "
                                    "ball came out first are censored, not dropped · shaded = 95% CI",
                                    "Part des dropbacks sans rusher à moins de 2 yards du QB · les jeux où le ballon part "
                                    "avant sont censurés, pas supprimés · zone ombrée = IC 95 %") + "</sup>",
                 legend=dict(x=0.6, y=0.98, bgcolor="rgba(255,255,255,0.85)"))
    fig.update_xaxes(title=_t("Seconds after the snap", "Secondes après le snap"), range=[0, 5], dtick=0.5)
    fig.update_yaxes(title=_t(f"P(no rusher within {radius_label})", "Poches encore propres"), tickformat=".0%",
                     range=[0, 1.02])
    return fig


def km_small_multiples(df: pd.DataFrame, by="box_bin", order=C.BOX_ORDER, label=None) -> go.Figure:
    label = label or _t("Defenders in the box", "Défenseurs dans la boîte")
    fig = make_subplots(rows=1, cols=len(order), shared_yaxes=True,
                        subplot_titles=[f"{label} : {o}" for o in order], horizontal_spacing=0.03)
    for j, o in enumerate(order, start=1):
        d = df[df[by].astype(str).eq(o)]
        for b in C.RUSH_ORDER:
            sub = d[d.rush_bin.astype(str).eq(b)]
            if len(sub) >= 15:
                _add_km(fig, sub, f"{b} rushers", RUSH_COLORS[b], row=1, col=j, show_ci=False,
                        showlegend=(j == 1), plain_name=True)
    _base_layout(fig, _t(f"Pocket survival by rushers, split by {label.lower()} (pre-snap look)",
                         "Survie de la poche par nombre de rushers, selon les défenseurs dans la boîte")
                 + "<br><sup>" + _t("Defenders in the box is a secondary, pre-snap indicator; the PFF rusher count "
                                    "(colours) is the real blitz variable · groups with <15 plays hidden",
                                    "La boîte est un indicateur pré-snap secondaire ; le nombre de rushers PFF (couleurs) "
                                    "est la vraie variable de blitz · groupes de moins de 15 jeux masqués") + "</sup>",
                 legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title=_t("Seconds after snap", "Secondes après le snap"), range=[0, 5], dtick=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(title=_t("P(no rusher within 2 yd)", "Poches encore propres"), row=1, col=1)
    return fig


def km_by_category(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[_t("Coverage type (PFF)", "Type de couverture (PFF)"),
                                        _t("Offensive formation", "Formation offensive")])
    for i, c in enumerate(["Man", "Zone"]):
        _add_km(fig, df[df.pff_passCoverageType.eq(c)], c, PALETTE[i * 3], row=1, col=1)
    forms = df.offenseFormation.value_counts()
    for i, f in enumerate(forms[forms >= 50].index):
        _add_km(fig, df[df.offenseFormation.eq(f)], f, PALETTE[(i + 1) % len(PALETTE)], row=1, col=2, show_ci=False)
    _base_layout(fig, _t("Pocket survival by coverage and formation", "Survie de la poche par couverture et formation"),
                 legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title=_t("Seconds after snap", "Secondes après le snap"), range=[0, 5])
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    fig.update_yaxes(title=_t("P(no rusher within 2 yd)", "Poches encore propres"), row=1, col=1)
    return fig


def km_generic(df: pd.DataFrame, by: str, order=None, radius: float = C.PRESSURE_RADIUS,
               label: str | None = None, min_n: int = 20, show_ci: bool = True) -> go.Figure:
    """Pocket-survival curves for any grouping column and any pressure radius."""
    tag = str(float(radius)).replace(".", "_")
    tcol, ecol = f"time_to_pressure_r{tag}", f"pressure_event_r{tag}"
    groups = order or sorted(df[by].dropna().astype(str).unique())
    fig = go.Figure()
    for i, g in enumerate(groups):
        sub = df[df[by].astype(str).eq(str(g))]
        if len(sub) >= min_n:
            col = RUSH_COLORS.get(str(g), PALETTE[i % len(PALETTE)]) if by == "rush_bin" else PALETTE[i % len(PALETTE)]
            _add_km(fig, sub, f"{label or by} : {g}", col, show_ci=show_ci, time_col=tcol, event_col=ecol)
    fig.add_hline(y=0.5, line=dict(color="grey", dash="dash"),
                  annotation_text=_t("median", "médiane"), annotation_position="bottom left")
    _base_layout(fig, _t(f"Pocket survival by {label or by} · pressure radius {radius:g} yd",
                         f"Survie de la poche par {label or by} · rayon {_num(radius)} yd"),
                 legend=dict(x=0.55, y=0.98, bgcolor="rgba(255,255,255,0.85)"))
    fig.update_xaxes(title=_t("Seconds after the snap", "Secondes après le snap"), range=[0, 5], dtick=0.5)
    fig.update_yaxes(title=_t(f"P(no rusher within {radius:g} yd)", f"Poches sans rusher à < {_num(radius)} yd"),
                     tickformat=".0%", range=[0, 1.02])
    return fig


def radius_sensitivity_fig(tab: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for r, col in zip(tab.radius.unique(), ["#93c5fd", "#2563eb", "#1e3a8a"]):
        s = tab[tab.radius.eq(r)]
        fig.add_trace(go.Bar(x=s.group, y=s.median_time_to_pressure_s, name=_t(f"radius {r:g} yd", f"rayon {_num(r)} yd"),
                             marker_color=col, text=s.median_time_to_pressure_s.round(1), textposition="outside"))
    _base_layout(fig, _t("Median time to pressure by number of rushers: sensitivity to the pressure radius",
                         "Temps médian avant pression selon le rayon choisi")
                 + "<br><sup>" + _t("Kaplan-Meier medians; missing bar = pocket held for more than half the plays",
                                    "Médianes Kaplan-Meier ; barre absente = la poche a tenu sur plus de la moitié des jeux")
                 + "</sup>", barmode="group")
    fig.update_xaxes(title=_t("Number of PFF pass rushers", "Nombre de pass rushers (PFF)"))
    fig.update_yaxes(title=_t("Median seconds after snap", "Secondes après le snap (médiane)"))
    return fig


# --------------------------------------------------------------------------- #
# 4. Team ranking
# --------------------------------------------------------------------------- #
def team_scatter(teams: pd.DataFrame, rho: float) -> go.Figure:
    t = teams.copy()
    xm, ym = t.pressure_rate_pff.median() * 100, t.PIS_adj.median()
    fig = go.Figure(go.Scatter(
        x=t.pressure_rate_pff * 100, y=t.PIS_adj, mode="markers+text", text=t.possessionTeam,
        textposition="top center",
        marker=dict(size=15, color=t.sack_rate * 100, colorscale="Reds", showscale=True,
                    colorbar=dict(title=_t("Sack rate %", "Taux de sack %")), line=dict(color="#1f2937", width=1)),
        customdata=np.c_[t.n_plays, t.time_to_throw, t.sack_rate * 100, t.PIS],
        hovertemplate="<b>%{text}</b><br>" + _t("PIS adj", "PIS ajusté") + " : %{y:.1f} (" + _t("raw", "brut")
                      + " %{customdata[3]:.1f})<br>" + _t("PFF pressure allowed", "Pression PFF concédée")
                      + " : %{x:.1f}%<br>" + _t("Sack rate", "Taux de sack") + " : %{customdata[2]:.1f}%<br>"
                      + _t("Avg time to throw", "Lancer moyen") + " : %{customdata[1]:.2f} s<br>"
                      + _t("Plays", "Dropbacks") + " : %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=xm, line=dict(color="grey", dash="dash"))
    fig.add_hline(y=ym, line=dict(color="grey", dash="dash"))
    x0, x1 = t.pressure_rate_pff.min() * 100, t.pressure_rate_pff.max() * 100
    y0, y1 = t.PIS_adj.min(), t.PIS_adj.max()
    quads = [(x0, y1, _t("Clean pocket, few pressures", "Poche propre, peu de pression"), "left", "top"),
             (x1, y1, _t("Clean pocket but pressured<br>(QB/scheme driven?)", "Poche propre mais pressions<br>(QB / schéma ?)"),
              "right", "top"),
             (x0, y0, _t("Tight pocket, PFF sees few pressures", "Poche serrée, peu de pressions PFF"), "left", "bottom"),
             (x1, y0, _t("Pocket breaks down: protection problem", "La poche cède : problème de protection"),
              "right", "bottom")]
    for x, y, txt, xa, ya in quads:
        fig.add_annotation(x=x, y=y, text=f"<i>{txt}</i>", showarrow=False, xanchor=xa, yanchor=ya,
                           font=dict(color="#6b7280", size=12))
    _base_layout(fig, _t("Team pass protection: Pocket Integrity Score (adjusted for time to throw) vs PFF pressure "
                         f"allowed<br><sup>Teams with ≥{C.MIN_TEAM_PLAYS} dropbacks, weeks 1-8 2021 · "
                         f"Spearman ρ = {rho:.2f} · colour = sack rate</sup>",
                         "Protection de passe par équipe : PIS ajusté vs pression concédée (PFF)"))
    fig.update_xaxes(title=_t("PFF pressure allowed (% of dropbacks with a hit, hurry or sack)",
                              "Pression concédée selon PFF (% des dropbacks avec hit, hurry ou sack)"))
    fig.update_yaxes(title=_t("PIS adjusted for time to throw (50 = league average)",
                              "PIS ajusté au temps de lancer (50 = moyenne)"))
    return fig


def team_bars(teams: pd.DataFrame) -> go.Figure:
    """Horizontal ranking bars of adjusted PIS (50 = league average)."""
    t = teams.sort_values("PIS_adj")
    col = np.where(t.PIS_adj >= 50, "#16a34a", "#dc2626")
    fig = go.Figure(go.Bar(x=t.PIS_adj - 50, y=t.possessionTeam, base=50, orientation="h", marker_color=col,
                           text=t.PIS_adj.round(1), textposition="outside",
                           customdata=np.c_[t.rank_PIS_adj, t.PIS, t.time_to_throw, t.pressure_rate_pff * 100],
                           hovertemplate="<b>%{y}</b> · " + _t("rank", "rang") + " %{customdata[0]}<br>"
                                         + _t("PIS adj", "PIS ajusté") + " %{text} · " + _t("raw", "brut")
                                         + " %{customdata[1]:.1f}<br>" + _t("Avg time to throw", "Lancer moyen")
                                         + " %{customdata[2]:.2f} s<br>" + _t("PFF pressure", "Pression PFF")
                                         + " %{customdata[3]:.0f}%<extra></extra>"))
    fig.add_vline(x=50, line=dict(color="#111827", width=1))
    fig.update_layout(font=FONT, template="plotly_white", height=820, margin=dict(l=60, r=40, t=20, b=40),
                      showlegend=False)
    lo, hi = float(t.PIS_adj.min()), float(t.PIS_adj.max())
    fig.update_xaxes(title=_t("Adjusted PIS (50 = league average)", "PIS ajusté (50 = moyenne de la ligue)"),
                     range=[min(lo, 50) - 5, max(hi, 50) + 5])
    return fig


# --------------------------------------------------------------------------- #
# 5. Validation
# --------------------------------------------------------------------------- #
def _outcome(df):
    return np.select([df.sack.eq(1), df.pressure_pff.eq(1)], ["sack", "pressure"], "clean")


def validation_boxplots(df: pd.DataFrame, auc: float) -> go.Figure:
    d = df.assign(outcome=_outcome(df))
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Pocket Integrity Score",
        _t("Time to first rusher within 2 yd (pressured plays)", "Temps avant le 1er rusher à < 2 yd (jeux avec pression)")])
    for k in OUTCOME_KEYS:
        s = d[d.outcome.eq(k)]
        lab = _outcome_label(k)
        fig.add_trace(go.Box(y=s.PIS, name=f"{lab}<br>n={len(s)}", marker_color=OUTCOME_COLORS[k], boxmean=True,
                             showlegend=False), row=1, col=1)
        s2 = s[s.pressure_event.eq(1)]
        fig.add_trace(go.Box(y=s2.time_to_pressure, name=f"{lab}<br>n={len(s2)}", marker_color=OUTCOME_COLORS[k],
                             showlegend=False), row=1, col=2)
    _base_layout(fig, _t("Does the tracking metric agree with PFF charting?", "La métrique tracking est-elle d'accord avec PFF ?")
                 + "<br><sup>" + _t(f"PIS separates PFF-pressured from clean dropbacks with AUC = {auc:.2f} "
                                    "(0.5 = coin flip, 1 = perfect)",
                                    f"Le PIS sépare les jeux sous pression PFF des poches propres avec une AUC de "
                                    f"{_num(auc, 2)} (0,5 = hasard, 1 = parfait)") + "</sup>")
    fig.update_yaxes(title=_t("PIS (0-100, higher = cleaner pocket)", "PIS (0-100, plus haut = poche plus propre)"),
                     row=1, col=1)
    fig.update_yaxes(title=_t("Seconds after snap", "Secondes après le snap"), row=1, col=2)
    return fig


def pis_vs_ttt(df: pd.DataFrame) -> go.Figure:
    d = df.assign(outcome=_outcome(df))
    fig = go.Figure()
    for k in OUTCOME_KEYS:
        s = d[d.outcome.eq(k)]
        fig.add_trace(go.Scattergl(x=s.time_to_throw, y=s.PIS, mode="markers", name=_outcome_label(k),
                                   marker=dict(color=OUTCOME_COLORS[k], size=5, opacity=0.45),
                                   customdata=np.c_[s.gameId, s.playId, s.possessionTeam],
                                   hovertemplate="%{customdata[2]} · game %{customdata[0]} · play %{customdata[1]}"
                                                 "<br>TTT %{x:.1f} s · PIS %{y:.0f}<extra></extra>"))
    xs = np.linspace(d.time_to_throw.min(), min(d.time_to_throw.max(), 6.5), 50)
    slope, icpt = df.attrs.get("ttt_slope"), df.attrs.get("ttt_intercept")
    if slope is not None:
        fig.add_trace(go.Scatter(x=xs, y=icpt + slope * xs, mode="lines",
                                 name=_t(f"Linear fit ({slope:+.1f} PIS / s)", f"Tendance ({_num(slope)} PIS par seconde)"),
                                 line=dict(color="black", width=3)))
    _base_layout(fig, _t("Holding the ball costs pocket integrity: PIS vs time to throw",
                         "Garder le ballon coûte cher : PIS selon le temps de lancer")
                 + "<br><sup>" + _t("Each dot is a dropback · the team ranking uses the residual from the black line "
                                    "(PIS adjusted for time to throw)",
                                    "Chaque point est un dropback · le PIS ajusté mesure l'écart à la droite noire") + "</sup>")
    fig.update_xaxes(title=_t("Time to throw (seconds from snap to release / sack)",
                              "Temps de lancer (secondes entre le snap et le lancer ou le sack)"), range=[0, 6.5])
    fig.update_yaxes(title="PIS (0-100)")
    return fig
