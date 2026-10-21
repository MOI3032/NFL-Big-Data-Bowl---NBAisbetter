"""Interactive web app (Dash) for the Pocket Integrity Score / Blitz Time Pressure project.

    python app.py            ->  http://127.0.0.1:8050

Requires the outputs of `python main.py` (CSV files in ./outputs).
Local tool only: bound to 127.0.0.1, no authentication.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme

from src import config as C
from src import metrics as M
from src import viz
from src.data_loading import load_games, load_plays

viz.set_lang("fr")
OUT = C.OUTPUT_DIR


# =========================================================================== #
# Data
# =========================================================================== #
def _read(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise SystemExit(f"{path} est introuvable : lance d'abord `python main.py`.")
    return pd.read_csv(path, encoding="utf-8-sig")


RESULT_FR = {"C": "Complète", "I": "Incomplète", "S": "Sack", "IN": "Interception"}
COVER_FR = {"Man": "Man", "Zone": "Zone", "Other": "Autre"}


def pis_category(p: float) -> tuple[str, str]:
    if p >= 80:
        return "Forteresse", "#15803d"
    if p >= 60:
        return "Poche propre", "#16a34a"
    if p >= 40:
        return "Poche moyenne", "#d97706"
    if p >= 20:
        return "Poche sous pression", "#ea580c"
    return "Poche effondrée", "#dc2626"


def load_app_data() -> pd.DataFrame:
    df = _read("play_summary.csv")
    df["time_to_pressure_r2_0"] = df.time_to_pressure
    df["pressure_event_r2_0"] = df.pressure_event
    games = load_games()[["gameId", "gameDate", "homeTeamAbbr", "visitorTeamAbbr"]]
    desc = load_plays()[["gameId", "playId", "playDescription"]]
    df = df.merge(games, on="gameId", how="left").merge(desc, on=["gameId", "playId"], how="left")
    df["match"] = ("S" + df.week.astype(str) + " · " + df.visitorTeamAbbr + " @ " + df.homeTeamAbbr
                   + " (" + df.gameDate.str[:5] + ")")
    df["id"] = df.gameId.astype(str) + "-" + df.playId.astype(str)
    clock = df.gameClock.str.split(":", expand=True).astype(float)
    df["chrono"] = df.quarter * 10000 - (clock[0] * 60 + clock[1])
    df = df.sort_values(["week", "gameId", "chrono"]).reset_index(drop=True)
    df["down_dist"] = df.down.astype(str) + "&" + df.yardsToGo.astype(str)
    df["result_txt"] = df.passResult.map(RESULT_FR)
    df["cover_txt"] = df.pff_passCoverageType.map(COVER_FR).fillna("?")
    df["ttp_display"] = np.where(df.pressure_event.eq(1), df.time_to_pressure.round(1), np.nan)
    df["pressure_txt"] = np.where(df.pressure_pff.eq(1), "oui", "non")
    df["pa_txt"] = np.where(df.play_action.eq(1), "oui", "non")
    df["category"] = [pis_category(p)[0] for p in df.PIS]
    df["pis_pct"] = df.PIS.rank(pct=True) * 100
    df.attrs["ttt_slope"], df.attrs["ttt_intercept"] = np.polyfit(df.time_to_throw, df.PIS, 1)
    return df


DF = load_app_data()
TEAMS = _read("team_ranking.csv")
TEAM_VAL = _read("validation_team_level.csv")
PLAY_VAL = _read("validation_play_level.csv")
BLOCKERS = _read("blocker_ranking.csv")
RADIUS_TAB = _read("radius_sensitivity.csv")
SENS = _read("pis_weight_sensitivity.csv")
SUBSETS = _read("subset_summary_PA_PR.csv")
EXCL = _read("exclusions.csv")

GAMES = DF.drop_duplicates("gameId")[["gameId", "week", "match", "homeTeamAbbr", "visitorTeamAbbr"]]
ALL_TEAMS = sorted(set(DF.possessionTeam))
AVG = {"ttt": DF.time_to_throw.mean(), "cushion": DF.cushion.mean(), "clean": DF.clean_time_pct.mean(),
       "area": DF.pocket_area.mean(), "pressure": DF.pressure_pff.mean()}
SLOPE = DF.attrs["ttt_slope"]


def fr(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


# Headline numbers (computed, not hard-coded)
_km_rush = M.km_table(DF, "rush_bin", order=C.RUSH_ORDER).set_index("group")
_auc = PLAY_VAL.query("label == 'pressure_pff' and score.str.startswith('PIS (')").AUC.iat[0]
_auc_sack = PLAY_VAL.query("label == 'sack' and score.str.startswith('PIS (')").AUC.iat[0]
_rho = TEAM_VAL.query("team_score == 'PIS_adj' and vs == 'pressure_rate_pff'").spearman_rho.iat[0]
_rho_beaten = TEAM_VAL.query("team_score == 'PIS_adj' and vs == 'beaten_rate'").spearman_rho.iat[0]


# =========================================================================== #
# Static figures (built once, French labels, titles moved into the page)
# =========================================================================== #
def _clean(fig, height=None):
    fig.update_layout(title=None, autosize=True, width=None, margin=dict(t=40))
    if height:
        fig.update_layout(height=height)
    return fig


FIG_TEAMS = _clean(viz.team_scatter(TEAMS, _rho), 620)
FIG_BARS = viz.team_bars(TEAMS)
FIG_BOX = _clean(viz.validation_boxplots(DF, _auc), 560)
FIG_TTT = _clean(viz.pis_vs_ttt(DF), 560)
FIG_SMALL = _clean(viz.km_small_multiples(DF), 460)
FIG_RADIUS = _clean(viz.radius_sensitivity_fig(RADIUS_TAB), 420)


def _blocker_fig():
    import plotly.graph_objects as go
    b = BLOCKERS.nsmallest(15, "defender_reached_QB_pct").iloc[::-1]
    fig = go.Figure(go.Bar(x=b.defender_reached_QB_pct, y=b.displayName + " (" + b.team + ", " + b.position + ")",
                           orientation="h", marker_color="#16a34a", text=b.defender_reached_QB_pct.map(lambda v: f"{fr(v)} %"),
                           textposition="outside", customdata=b.n_blocks,
                           hovertemplate="%{y}<br>%{x:.1f} % des blocs · %{customdata} blocs<extra></extra>"))
    fig.update_layout(font=viz.FONT, template="plotly_white", height=520, margin=dict(l=10, r=40, t=10, b=40))
    fig.update_xaxes(title="% des blocs où son défenseur arrive à < 2 yd du QB (plus bas = mieux)")
    return fig


FIG_BLK = _blocker_fig()


@lru_cache(maxsize=96)
def replay_dict(game_id: int, play_id: int, view: str) -> dict:
    spec, _, _ = viz.replay_spec(game_id, play_id, view=view, show_title=False)
    spec["layout"].pop("width", None)
    spec["layout"].update(autosize=True, height=720)
    return spec


@lru_cache(maxsize=96)
def timeline_dict(game_id: int, play_id: int) -> dict:
    fig = viz.play_timeline(game_id, play_id)
    fig.update_layout(width=None, autosize=True)
    return fig.to_plotly_json()


# =========================================================================== #
# Layout helpers
# =========================================================================== #
def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records (NaN -> None)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def col(cid, name, kind="text"):
    c = {"id": cid, "name": name, "type": "text"}
    if kind in ("int", "dec1", "dec2", "pct"):
        c["type"] = "numeric"
        c["format"] = {"int": Format(precision=0, scheme=Scheme.fixed),
                       "dec1": Format(precision=1, scheme=Scheme.fixed),
                       "dec2": Format(precision=2, scheme=Scheme.fixed),
                       "pct": Format(precision=1, scheme=Scheme.percentage)}[kind]
    return c


TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={"fontFamily": "Inter, Segoe UI, Arial", "fontSize": 13, "padding": "6px 10px", "textAlign": "left",
                "maxWidth": 280, "overflow": "hidden", "textOverflow": "ellipsis", "border": "none",
                "borderBottom": "1px solid #e2e8f0"},
    style_header={"backgroundColor": "#0f172a", "color": "white", "fontWeight": "600", "border": "none"},
    style_filter={"backgroundColor": "#f8fafc"},
    tooltip_delay=0, tooltip_duration=None,
    css=[{"selector": ".dash-table-tooltip", "rule": "background:#0f172a;color:white;font-size:12px;max-width:320px"}],
)
BASE_COND = [{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
             {"if": {"state": "active"}, "backgroundColor": "#dbeafe", "border": "1px solid #2563eb"}]


def table(id_, df, columns, page_size=15, tips=None, cond=None, **kw):
    return dash_table.DataTable(
        id=id_, data=records(df), columns=columns, sort_action="native", filter_action="native",
        page_size=page_size, tooltip_header=tips or {}, style_data_conditional=BASE_COND + (cond or []),
        **TABLE_STYLE, **kw)


def ctrl(id_, options, value, label, **kw):
    return html.Div([html.Label(label), dcc.Dropdown(id=id_, options=options, value=value, clearable=False, **kw)],
                    className="ctrl")


def card(*children, **kw):
    return html.Div(list(children), className="card", **kw)


def explain(title, *items, open_=False):
    return html.Details([html.Summary(title), *items], open=open_, className="explain")


def kpi(value, label):
    return html.Div([html.Div(value, className="v"), html.Div(label, className="l")], className="kpi")


def metric(label, value, hint, color="#0f172a"):
    return html.Div([html.Div(label, className="l"), html.Div(value, className="v", style={"color": color}),
                     html.Div(hint, className="h")], className="metric")


def graph(fig_or_id, **kw):
    if isinstance(fig_or_id, str):
        return dcc.Graph(id=fig_or_id, config={"displaylogo": False}, **kw)
    return dcc.Graph(figure=fig_or_id, config={"displaylogo": False}, **kw)


WEEK_OPTS = [{"label": "Toutes les semaines", "value": 0}] + [{"label": f"Semaine {w}", "value": int(w)}
                                                              for w in sorted(DF.week.unique())]
TEAM_OPTS = [{"label": "Toutes les équipes", "value": "ALL"}] + [{"label": t, "value": t} for t in ALL_TEAMS]

# =========================================================================== #
# Tab 0: guide
# =========================================================================== #
_cat_share = DF.category.value_counts(normalize=True)
SCALE = html.Div([
    html.Div([cat, html.Span(f"PIS {rng} · {fr(100 * _cat_share.get(cat, 0), 0)} % des jeux")], style={"background": c})
    for cat, rng, c in [("Poche effondrée", "0-20", "#dc2626"), ("Poche sous pression", "20-40", "#ea580c"),
                        ("Poche moyenne", "40-60", "#d97706"), ("Poche propre", "60-80", "#16a34a"),
                        ("Forteresse", "80-100", "#15803d")]
], className="scale")

GLOSSARY = [
    ("Dropback", "Action où le QB recule pour passer. Ici : passes complètes, incomplètes, interceptions et sacks. "
                 "Les scrambles (le QB part courir) sont exclus."),
    ("Snap → lancer", "La fenêtre analysée : du snap jusqu'au lancer du ballon ou au sack. 10 images par seconde."),
    ("Pass rusher", "Défenseur qui attaque le QB (rôle « Pass Rush » noté par PFF)."),
    ("Blitz", "6 pass rushers ou plus. 4 est la norme ; 3 ou moins = défense qui recule en couverture."),
    ("Poche", "L'espace protégé autour du QB. On la dessine comme le polygone qui relie les bloqueurs."),
    ("Pression (tracking)", "Un rusher entre dans le cercle de 2 yards autour du QB avant le lancer."),
    ("Pression (PFF)", "Hit, hurry ou sack noté à la vidéo par les scouts de Pro Football Focus. Sert de référence."),
    ("Bloqueur battu", "PFF indique qu'un bloqueur a été battu ou a concédé un hit, un hurry ou un sack."),
    ("Censuré", "Le ballon part avant toute pression : on sait seulement que la poche a tenu au moins jusque-là. "
                "Ces jeux ne sont ni supprimés ni comptés comme pression."),
    ("Kaplan-Meier", "Méthode qui trace la part de poches encore propres seconde après seconde en tenant compte "
                     "des jeux censurés."),
    ("Défenseurs dans la boîte", "Nombre de défenseurs près de la ligne avant le snap : indice de blitz visible "
                                 "avant l'action, mais moins fiable que le nombre réel de rushers."),
    ("Play-action / pocket-roll", "Fausse course avant la passe / poche qui se déplace avec le QB. Signalés, et les "
                                  "résultats sont donnés avec et sans eux."),
    ("AUC", "Capacité d'un score à distinguer deux groupes : 0,5 = hasard, 1 = parfait, 0,8 = bon."),
]

tab_guide = html.Div([
    html.Div([
        kpi(fr(len(DF), 0).replace(",", " "), "dropbacks analysés (2021, semaines 1-8)"),
        kpi(str(DF.gameId.nunique()), "matchs · 32 équipes"),
        kpi(fr(_auc, 2), "AUC du PIS pour repérer une pression PFF"),
        kpi(fr(_auc_sack, 2), "AUC du PIS pour repérer un sack"),
        kpi(f"{fr(_km_rush.loc['4', 'median_time_to_pressure_s'])} → {fr(_km_rush.loc['6+', 'median_time_to_pressure_s'])} s",
            "temps médian avant pression : 4 rushers → blitz 6+"),
    ], className="kpis"),

    card(html.H2("Le Pocket Integrity Score (PIS) en 30 secondes"),
         html.P("Une note de 0 à 100 qui dit à quel point la poche du quarterback est restée propre entre le snap et "
                "le lancer. 50 = poche moyenne de la ligue. Elle est calculée image par image à partir des positions "
                "GPS des 22 joueurs.", className="lead"),
         SCALE,
         html.Div("Répartition des 8 057 dropbacks par niveau de PIS.", style={"color": "#64748b", "fontSize": 12}),
         html.H3("Trois ingrédients", style={"marginTop": 16}),
         html.Div([
             html.Div([html.Div("45 %", className="w"), html.B("Le coussin"),
                       html.P(f"Distance moyenne entre le QB et le rusher le plus proche. Moyenne ligue : "
                              f"{fr(AVG['cushion'])} yards. Plus il y a d'espace, plus le QB est tranquille.")],
                      className="ingredient"),
             html.Div([html.Div("40 %", className="w"), html.B("Le temps propre"),
                       html.P(f"Part du temps où aucun rusher n'est à moins de 2 yards du QB. Moyenne ligue : "
                              f"{fr(100 * AVG['clean'], 0)} %. Un seul rusher qui perce fait chuter ce chiffre.")],
                      className="ingredient"),
             html.Div([html.Div("15 %", className="w"), html.B("L'aire de la poche"),
                       html.P(f"Surface du polygone formé par les bloqueurs (moyenne {fr(AVG['area'], 0)} yd²). "
                              "Poids faible : une grande surface peut aussi être une ligne étirée.")],
                      className="ingredient"),
         ], className="grid-3"),
         html.P("Chaque ingrédient est comparé à la moyenne de la ligue (en écarts-types), on fait la somme pondérée, "
                "puis on ramène sur 0-100 : 50 = moyenne, 84 ≈ un écart-type au-dessus.",
                style={"color": "#64748b", "marginTop": 10})),

    html.Div([
        card(html.H2("PIS ajusté : corriger l'effet du temps de lancer"),
             html.P([f"Un QB qui lance en 1,8 s subit moins de pression qu'un QB qui garde le ballon 3,5 s. Sur nos "
                     f"données, le PIS perd en moyenne ", html.B(f"{fr(-SLOPE)} points par seconde"),
                     " passée avec le ballon."]),
             html.P(["Le ", html.B("PIS ajusté"), " compare chaque action à ce qu'on attend pour son temps de lancer : "
                     "50 = normal pour ce temps, 60 = mieux que prévu, 40 = moins bien. C'est lui qui sert à classer "
                     "les équipes, pour ne pas récompenser une ligne juste parce que son QB lâche vite le ballon. "
                     "Il peut sortir de 0-100 sur des jeux extrêmes."])),
        card(html.H2("Temps avant pression (Blitz Time Pressure)"),
             html.P(["On trace un cercle de ", html.B("2 yards"), " autour du QB. Le temps avant pression est le "
                     "moment où un rusher y entre pour la première fois après le snap."]),
             html.P(["Si le ballon part avant, l'action est ", html.B("censurée"), " : la poche a tenu au moins jusque-là. "
                     "On ne la supprime pas (ce serait ne garder que les jeux ratés) et on n'invente pas de temps. "
                     "Les courbes de Kaplan-Meier gèrent ces deux cas."]),
             html.P(f"Médiane ligue : {fr(_km_rush.loc['4', 'median_time_to_pressure_s'])} s avec 4 rushers, "
                    f"{fr(_km_rush.loc['6+', 'median_time_to_pressure_s'])} s avec un blitz à 6+.")),
    ], className="grid-2"),

    card(html.H2("Comment utiliser l'application"),
         html.Ul([
             html.Li([html.B("Actions & replays : "), "retrouve n'importe quelle action (semaine, match, équipe, résultat, "
                      "blitz…), lis son diagnostic et regarde-la en animation avec la poche et le cercle de pression."]),
             html.Li([html.B("Survie de la poche : "), "compare combien de temps la poche tient selon le nombre de "
                      "rushers, la couverture, la formation, le down, pour la ligue ou une équipe."]),
             html.Li([html.B("Classement équipes : "), "quelles lignes offensives protègent le mieux, une fois corrigé "
                      "le temps de lancer de leur QB."]),
             html.Li([html.B("Validation PFF : "), "la preuve que la métrique tracking colle au travail des scouts PFF."]),
             html.Li([html.B("Bloqueurs : "), "quels joueurs empêchent le mieux leur défenseur d'atteindre le QB."]),
         ])),

    card(html.H2("Glossaire"),
         html.Table([html.Tr([html.Td(a), html.Td(b)]) for a, b in GLOSSARY], className="glossary")),

    card(html.H2("Les données"),
         html.P("NFL Big Data Bowl 2023 : tracking Next Gen Stats (10 images/s) et notes PFF, saison 2021, semaines 1 à 8. "
                "Coordonnées retournées pour que l'attaque aille toujours de gauche à droite."),
         html.Ul([html.Li(f"{r.step} : {fr(r.plays_remaining, 0)} actions") for r in EXCL.itertuples()]),
         html.P("Limites : 8 semaines seulement ; le PIS dépend fortement du temps de lancer ; les pressions PFF "
                "incluent aussi celles provoquées par le QB, pas uniquement par la ligne.", style={"color": "#64748b"})),
])

# =========================================================================== #
# Tab 1: actions & replays
# =========================================================================== #
ACTION_COLUMNS = [col("week", "Sem.", "int"), col("match", "Match"), col("quarter", "Qt", "int"),
                  col("gameClock", "Horloge"), col("down_dist", "Down"), col("possessionTeam", "Attaque"),
                  col("defensiveTeam", "Défense"), col("result_txt", "Résultat"), col("n_rushers", "Rushers", "int"),
                  col("cover_txt", "Couverture"), col("offenseFormation", "Formation"),
                  col("time_to_throw", "Lancer (s)", "dec1"), col("PIS", "PIS", "dec1"),
                  col("PIS_adj", "PIS ajusté", "dec1"), col("ttp_display", "1re pression (s)", "dec1"),
                  col("pressure_txt", "Pression PFF"), col("pa_txt", "Play-action")]
ACTION_TIPS = {
    "week": "Semaine de la saison 2021", "match": "Visiteur @ domicile (date)",
    "quarter": "Quart-temps", "gameClock": "Temps restant dans le quart-temps",
    "down_dist": "Down & yards à gagner", "result_txt": "Résultat de la passe",
    "n_rushers": "Nombre de défenseurs qui attaquent le QB (PFF). 6+ = blitz",
    "cover_txt": "Couverture défensive (PFF)", "time_to_throw": "Secondes entre le snap et le lancer (ou le sack)",
    "PIS": "Pocket Integrity Score 0-100 : propreté de la poche. 50 = moyenne",
    "PIS_adj": "PIS corrigé du temps de lancer. 50 = normal pour ce temps de lancer",
    "ttp_display": "Moment où un rusher entre à moins de 2 yd du QB. Vide = la poche a tenu jusqu'au lancer",
    "pressure_txt": "Hit, hurry ou sack noté par PFF", "pa_txt": "Fausse course avant la passe",
}
ACTION_COND = [
    {"if": {"filter_query": "{PIS} < 20", "column_id": "PIS"}, "backgroundColor": "#fee2e2", "color": "#991b1b", "fontWeight": "700"},
    {"if": {"filter_query": "{PIS} >= 20 && {PIS} < 40", "column_id": "PIS"}, "backgroundColor": "#ffedd5", "color": "#9a3412"},
    {"if": {"filter_query": "{PIS} >= 60 && {PIS} < 80", "column_id": "PIS"}, "backgroundColor": "#dcfce7", "color": "#166534"},
    {"if": {"filter_query": "{PIS} >= 80", "column_id": "PIS"}, "backgroundColor": "#bbf7d0", "color": "#14532d", "fontWeight": "700"},
    {"if": {"filter_query": '{result_txt} = "Sack"', "column_id": "result_txt"}, "color": "#dc2626", "fontWeight": "700"},
    {"if": {"filter_query": '{pressure_txt} = "oui"', "column_id": "pressure_txt"}, "color": "#dc2626", "fontWeight": "700"},
    {"if": {"filter_query": "{n_rushers} >= 6", "column_id": "n_rushers"}, "color": "#dc2626", "fontWeight": "700"},
]

tab_actions = html.Div([
    explain("Comment utiliser cette page",
            html.Ul([
                html.Li("Choisis tes filtres : tout se combine (ex. semaine 3 + KC + sacks)."),
                html.Li("Le tableau se trie en cliquant sur un en-tête ; la ligne grise sous les en-têtes filtre une colonne "
                        "(ex. tape >3 sous « Lancer (s) »). Survole un en-tête pour sa définition."),
                html.Li("Clique une ligne pour afficher le diagnostic et le replay de l'action ; ◀ ▶ passent à la "
                        "précédente / suivante dans l'ordre du tableau."),
                html.Li("Couleurs du PIS : rouge = poche effondrée, orange = sous pression, vert = propre."),
            ]), open_=True),
    card(html.Div([
        ctrl("f-week", WEEK_OPTS, 0, "Semaine"),
        ctrl("f-team", TEAM_OPTS, "ALL", "Équipe"),
        ctrl("f-side", [{"label": "En attaque ou en défense", "value": "both"},
                        {"label": "En attaque (sa protection)", "value": "off"},
                        {"label": "En défense (son pass rush)", "value": "def"}], "both", "Côté"),
        ctrl("f-game", [], "ALL", "Match"),
        ctrl("f-result", [{"label": "Tous les résultats", "value": "ALL"}] +
             [{"label": v, "value": k} for k, v in RESULT_FR.items()], "ALL", "Résultat"),
        ctrl("f-rush", [{"label": "Tous", "value": "ALL"}] + [{"label": f"{r} rushers" + (" (blitz)" if r == "6+" else ""),
                                                               "value": r} for r in C.RUSH_ORDER], "ALL", "Rushers"),
        ctrl("f-pressure", [{"label": "Toutes", "value": "ALL"}, {"label": "Avec pression PFF", "value": "1"},
                            {"label": "Sans pression PFF", "value": "0"}], "ALL", "Pression"),
        ctrl("f-sort", [{"label": "Chronologique", "value": "chrono"},
                        {"label": "Pires poches d'abord (PIS ↑)", "value": "pis_asc"},
                        {"label": "Meilleures poches d'abord (PIS ↓)", "value": "pis_desc"},
                        {"label": "Pression la plus rapide", "value": "ttp"},
                        {"label": "Lancer le plus long", "value": "ttt"}], "chrono", "Trier par"),
    ], className="filters"),
        html.Div(id="n-actions", className="toolbar")),
    card(table("t-actions", DF.head(0), ACTION_COLUMNS, page_size=12, tips=ACTION_TIPS, cond=ACTION_COND),
         html.Div([
             html.Button("◀ Action précédente", id="btn-prev", n_clicks=0, className="btn secondary"),
             html.Button("Action suivante ▶", id="btn-next", n_clicks=0, className="btn secondary"),
             html.Div([html.Label("Vue du replay", style={"marginRight": 8, "fontWeight": 600}),
                       dcc.RadioItems(id="view", value="pocket", inline=True,
                                      options=[{"label": " Zoom sur la poche ", "value": "pocket"},
                                               {"label": " Terrain complet", "value": "field"}],
                                      inputStyle={"marginLeft": 10})],
                      style={"display": "flex", "alignItems": "center", "marginLeft": "auto"}),
         ], className="toolbar")),
    dcc.Store(id="sel-id"),
    card(html.Div(id="play-head"), html.Div(id="play-verdict"), html.Div(id="play-cards", className="metrics"),
         html.Div([
             html.Span([html.Span(className="dot", style={"background": "#f59e0b"}), "QB"]),
             html.Span([html.Span(className="dot", style={"background": "#2563eb"}), "Attaque (n° de maillot)"]),
             html.Span([html.Span(className="dot", style={"background": "#b91c1c"}), "Pass rushers"]),
             html.Span([html.Span(className="dot", style={"background": "#fecaca", "border": "1px solid #991b1b"}),
                        "Couverture"]),
             html.Span([html.Span(className="dot", style={"background": "rgba(59,130,246,.35)"}), "Poche (polygone des bloqueurs)"]),
             html.Span("Cercle jaune → rouge : rusher à moins de 2 yd · bleu = ligne de mêlée · jaune = 1er down"),
         ], className="legend-row"),
         dcc.Loading(graph("g-replay", style={"height": "720px"})),
         html.H3("La pression seconde par seconde", style={"marginTop": 8}),
         html.P("Ligne noire : distance entre le QB et le rusher le plus proche. Zone rouge : moins de 2 yards = pression. "
                "Pointillés bleus : aire de la poche. Zones grises : avant le snap / après le lancer.",
                style={"color": "#64748b", "margin": 0}),
         dcc.Loading(graph("g-timeline", style={"height": "300px"}))),
])

# =========================================================================== #
# Tab 2: pocket survival
# =========================================================================== #
KM_LABELS = {"rush_bin": "rushers", "box_bin": "défenseurs dans la boîte", "cover_txt": "couverture",
             "offenseFormation": "formation", "pa_txt": "play-action", "down": "down"}
KM_ORDERS = {"rush_bin": C.RUSH_ORDER, "box_bin": C.BOX_ORDER, "cover_txt": ["Man", "Zone", "Autre"],
             "pa_txt": ["non", "oui"]}
KM_COLUMNS = [col("group", "Groupe"), col("n_plays", "Jeux", "int"), col("pressure_events", "Pressions observées", "int"),
              col("censored_pct", "% où la poche a tenu", "dec1"),
              col("median_time_to_pressure_s", "Temps médian avant pression (s)", "dec1"),
              col("S(1.5s)", "Poches propres à 1,5 s", "pct"), col("S(2.5s)", "Poches propres à 2,5 s", "pct"),
              col("mean_time_to_throw_s", "Lancer moyen (s)", "dec2")]

tab_km = html.Div([
    explain("Comment lire une courbe de survie de la poche",
            html.Ul([
                html.Li("Axe horizontal : secondes après le snap. Axe vertical : part des actions où la poche est encore "
                        "propre (aucun rusher à moins du rayon choisi)."),
                html.Li("Toutes les courbes partent de 100 % et descendent à chaque pression. Une courbe plus basse = "
                        "la pression arrive plus vite."),
                html.Li("Là où une courbe croise la ligne pointillée de 50 %, on lit le temps médian avant pression "
                        "(indiqué dans la légende)."),
                html.Li("Les actions où le ballon part avant toute pression sont « censurées » : elles comptent jusqu'au "
                        "lancer puis sortent du calcul. Zone ombrée = marge d'incertitude à 95 %."),
            ]), open_=True),
    card(html.Div([
        ctrl("km-by", [{"label": "Nombre de rushers (PFF)", "value": "rush_bin"},
                       {"label": "Défenseurs dans la boîte", "value": "box_bin"},
                       {"label": "Couverture (man / zone)", "value": "cover_txt"},
                       {"label": "Formation offensive", "value": "offenseFormation"},
                       {"label": "Play-action", "value": "pa_txt"},
                       {"label": "Down", "value": "down"}], "rush_bin", "Comparer par"),
        ctrl("km-radius", [{"label": f"{fr(r)} yards" + (" (référence)" if r == 2 else ""), "value": r}
                           for r in C.RADII], 2.0, "Rayon de pression"),
        ctrl("km-team", [{"label": "Toute la ligue", "value": "ALL"}] + [{"label": t, "value": t} for t in ALL_TEAMS],
             "ALL", "Attaque"),
        ctrl("km-week", WEEK_OPTS, 0, "Semaine"),
    ], className="filters"),
        html.Div(id="km-takeaway", className="takeaway"),
        graph("g-km", style={"height": "540px"}),
        html.Div(id="km-table")),
    card(html.H2("Le blitz vu avant le snap : défenseurs dans la boîte"),
         html.P("Chaque panneau regroupe les actions selon le nombre de défenseurs près de la ligne avant le snap ; les "
                "couleurs donnent le nombre réel de rushers. Ce que la défense montre avant le snap (la boîte) prédit moins "
                "bien la pression que ce qu'elle fait vraiment (les rushers).", className="lead"),
         graph(FIG_SMALL)),
    card(html.H2("Le choix du rayon change-t-il les conclusions ?"),
         html.P("Temps médian avant pression avec un cercle de 1,5, 2 ou 3 yards. Les valeurs bougent, mais l'ordre reste "
                "le même : plus de rushers = pression plus rapide.", className="lead"),
         graph(FIG_RADIUS)),
])

# =========================================================================== #
# Tab 3: teams
# =========================================================================== #
TEAM_COLUMNS = [col("rank_PIS_adj", "Rang", "int"), col("possessionTeam", "Équipe"), col("n_plays", "Dropbacks", "int"),
                col("PIS_adj", "PIS ajusté", "dec1"), col("PIS", "PIS brut", "dec1"),
                col("PIS_adj_noPA_noPR", "PIS ajusté hors play-action", "dec1"),
                col("time_to_throw", "Lancer moyen (s)", "dec2"), col("pressure_rate_pff", "Pression PFF", "pct"),
                col("sack_rate", "Taux de sack", "pct"), col("beaten_rate", "Bloqueur battu", "pct"),
                col("median_time_to_pressure_s", "Médiane avant pression (s)", "dec1")]
TEAM_TIPS = {"PIS_adj": "PIS corrigé du temps de lancer (50 = moyenne) : sert au classement",
             "PIS": "PIS moyen sans correction", "PIS_adj_noPA_noPR": "Même calcul sans play-action ni pocket-roll",
             "pressure_rate_pff": "% des dropbacks avec hit, hurry ou sack (PFF)",
             "beaten_rate": "% des dropbacks où au moins un bloqueur est noté battu par PFF",
             "median_time_to_pressure_s": "Temps médian avant qu'un rusher arrive à 2 yd (Kaplan-Meier)"}
_best, _worst = TEAMS.iloc[0], TEAMS.iloc[-1]

tab_teams = html.Div([
    explain("Comment lire ce classement",
            html.Ul([
                html.Li("Le classement utilise le PIS ajusté : on retire l'effet du temps de lancer pour juger la ligne "
                        "et pas le style du QB. 50 = moyenne de la ligue."),
                html.Li(f"Meilleure protection : {_best.possessionTeam} ({fr(_best.PIS_adj)}). "
                        f"Moins bonne : {_worst.possessionTeam} ({fr(_worst.PIS_adj)})."),
                html.Li(f"Le PIS ajusté suit le taux de bloqueurs battus selon PFF (ρ = {fr(_rho_beaten, 2)}) plus que le "
                        f"taux de pression global (ρ = {fr(_rho, 2)}), qui inclut aussi les pressions dues au QB."),
                html.Li("Clique une équipe dans le tableau pour voir ses actions en attaque dans « Actions & replays »."),
            ]), open_=True),
    html.Div([
        card(html.H2("Classement (PIS ajusté)"),
             html.P("Vert = au-dessus de la moyenne, rouge = en dessous.", className="lead"), graph(FIG_BARS)),
        card(html.H2("Détail par équipe"),
             html.P("Survole un en-tête pour sa définition ; clique une ligne pour ouvrir les actions de l'équipe.",
                    className="lead"),
             table("t-teams", TEAMS, TEAM_COLUMNS, page_size=32, tips=TEAM_TIPS)),
    ], className="grid-2-1"),
    card(html.H2("Protection mesurée par le tracking vs pression notée par PFF"),
         html.P("Chaque point est une équipe. En haut = poche propre ; à droite = beaucoup de pressions PFF ; couleur = taux "
                "de sack. Les équipes en haut à droite ont une bonne ligne mais leur QB subit quand même des pressions "
                "(QB qui tient le ballon, schéma) ; en bas à gauche, peu de pressions notées mais une poche serrée.",
                className="lead"),
         graph(FIG_TEAMS)),
])

# =========================================================================== #
# Tab 4: validation
# =========================================================================== #
LABEL_FR = {"pressure_pff": "Pression PFF", "sack": "Sack", "beaten": "Bloqueur battu (PFF)"}
SCORE_FR = {"PIS (low = pressure)": "PIS", "PIS_adj (time-to-throw adjusted)": "PIS ajusté",
            "time_to_pressure (censored = clean)": "Temps avant pression", "time_to_throw (baseline)": "Temps de lancer seul (référence)"}
pv = PLAY_VAL.assign(label=PLAY_VAL.label.map(LABEL_FR), score=PLAY_VAL.score.map(SCORE_FR),
                     r=PLAY_VAL.point_biserial_r.abs())
tv = TEAM_VAL.assign(
    pair=TEAM_VAL.team_score.map({"PIS": "PIS brut", "PIS_adj": "PIS ajusté"}) + " vs "
         + TEAM_VAL.vs.map({"pressure_rate_pff": "taux de pression PFF", "sack_rate": "taux de sack",
                            "beaten_rate": "taux de bloqueur battu"}),
    lecture=np.where(TEAM_VAL.p_value < 0.05, "lien significatif", "pas de lien net"))
WEIGHT_FR = {"baseline": "base", "equal": "égaux", "no area": "sans aire", "cushion heavy": "surtout coussin",
             "clean heavy": "surtout temps propre", "area heavy": "surtout aire"}
sens = SENS.assign(weights=SENS.weights.map(lambda s: WEIGHT_FR.get(s.split(" (")[0], s) + " (" + s.split(" (")[1]))
SUBSET_FR = {"all kept plays": "Toutes les actions", "excluding play-action": "Sans play-action",
             "excluding pocket-roll (PR)": "Sans pocket-roll", "excluding play-action and PR": "Sans play-action ni pocket-roll",
             "play-action only": "Play-action uniquement"}
subs = SUBSETS.assign(subset=SUBSETS.subset.map(SUBSET_FR))

tab_val = html.Div([
    explain("Pourquoi cette page ?",
            html.P("Une métrique tracking n'a de valeur que si elle retrouve ce qu'un expert voit à la vidéo. On la compare "
                   "donc aux notes des scouts PFF (hit, hurry, sack, bloqueur battu), qui n'ont pas servi à la construire."),
            html.Ul([html.Li("AUC : chance qu'une action sous pression ait un score pire qu'une action propre prise au hasard. "
                             "0,5 = hasard, 0,7 = correct, 0,8 = bon, 0,9+ = excellent."),
                     html.Li("Toutes les corrélations au niveau des actions sont significatives (p < 0,001).")]),
            open_=True),
    html.Div([kpi(fr(_auc, 2), "AUC PIS → pression PFF"), kpi(fr(_auc_sack, 2), "AUC PIS → sack"),
              kpi(fr(_rho_beaten, 2), "ρ équipes : PIS ajusté vs bloqueur battu"),
              kpi(fr(_rho, 2), "ρ équipes : PIS ajusté vs pression PFF")], className="kpis"),
    card(html.H2("Le PIS sépare-t-il les poches propres des poches sous pression ?"),
         html.P("À gauche : PIS des actions sans pression PFF, avec pression et avec sack. À droite : quand un rusher arrive "
                "à 2 yards, combien de temps après le snap. Plus les boîtes sont séparées, mieux la métrique distingue.",
                className="lead"), graph(FIG_BOX)),
    card(html.H2("Le piège du temps de lancer"),
         html.P(f"Plus le QB garde le ballon, plus le PIS baisse (environ {fr(-SLOPE)} points de moins par seconde). Sans correction, les "
                "équipes à jeu rapide paraîtraient mieux protégées qu'elles ne le sont. D'où le PIS ajusté.",
                className="lead"), graph(FIG_TTT)),
    html.Div([
        card(html.H3("Validation action par action"),
             table("t-pval", pv, [col("label", "Ce qu'on veut repérer"), col("score", "Indicateur"),
                                  col("AUC", "AUC", "dec2"), col("r", "Corrélation |r|", "dec2")], page_size=12)),
        card(html.H3("Validation par équipe (Spearman)"),
             table("t-tval", tv, [col("pair", "Comparaison"), col("spearman_rho", "ρ", "dec2"),
                                  col("p_value", "p-value", "dec2"), col("lecture", "Lecture")], page_size=6)),
    ], className="grid-2"),
    html.Div([
        card(html.H3("Les poids du PIS changent-ils le classement ?"),
             html.P("ρ proche de 1 = classement des équipes quasi identique aux poids de base.", className="lead"),
             table("t-sens", sens, [col("weights", "Poids coussin / temps propre / aire"), col("AUC_pressure", "AUC", "dec2"),
                                    col("spearman_team_rank_vs_baseline", "Stabilité du classement (ρ)", "dec2"),
                                    col("top5_teams", "Top 5")], page_size=6)),
        card(html.H3("Avec ou sans play-action / pocket-roll"),
             table("t-sub", subs, [col("subset", "Actions"), col("n_plays", "Jeux", "int"), col("mean_PIS", "PIS moyen", "dec1"),
                                   col("mean_time_to_throw_s", "Lancer moyen (s)", "dec2"),
                                   col("pff_pressure_rate", "Pression PFF", "pct"),
                                   col("AUC_PIS_vs_pff_pressure", "AUC", "dec2"),
                                   col("KM_median_time_to_pressure_s", "Médiane avant pression (s)", "dec1")], page_size=6)),
    ], className="grid-2"),
])

# =========================================================================== #
# Tab 5: blockers
# =========================================================================== #
BLK_COLUMNS = [col("displayName", "Joueur"), col("team", "Équipe"), col("position", "Aligné à"),
               col("officialPosition", "Poste officiel"), col("n_blocks", "Blocs", "int"),
               col("defender_reached_QB_pct", "% défenseur à < 2 yd du QB", "dec1"),
               col("S(2.5s)", "Défenseur contenu à 2,5 s", "pct")]
BLK_TIPS = {"position": "Position alignée au snap (PFF) : LT/LG/C/RG/RT, TE, RB…",
            "defender_reached_QB_pct": "Part des blocs où le défenseur bloqué arrive à moins de 2 yards du QB avant le lancer",
            "S(2.5s)": "Probabilité (Kaplan-Meier) que son défenseur ne soit pas encore à 2 yd du QB 2,5 s après le snap"}
tab_blk = html.Div([
    explain("Comment c'est calculé",
            html.P("PFF indique pour chaque bloqueur le premier défenseur qu'il bloque. On regarde si ce défenseur arrive à "
                   "moins de 2 yards du QB avant le lancer. Plus le pourcentage est bas, mieux le bloqueur a tenu son duel. "
                   "Seuls les joueurs avec au moins 100 blocs sont classés."),
            html.P("Attention : un défenseur peut arriver au QB parce qu'un autre bloqueur l'a lâché (stunt, échange) ; c'est "
                   "un indicateur de duel, pas une note définitive."), open_=True),
    html.Div([
        card(html.H2("Les 15 meilleurs bloqueurs"), graph(FIG_BLK)),
        card(html.H2("Tous les bloqueurs"),
             html.P("Filtre par équipe ou poste dans la ligne grise (ex. KC, ou LT).", className="lead"),
             table("t-blk", BLOCKERS, BLK_COLUMNS, page_size=14, tips=BLK_TIPS)),
    ], className="grid-2"),
])

# =========================================================================== #
# App
# =========================================================================== #
app = Dash(__name__, title="Pocket Integrity · NFL 2021", suppress_callback_exceptions=True)
app.layout = html.Div([
    html.Div([html.H1("Protection de passe · Pocket Integrity Score & Blitz Time Pressure"),
              html.Div(f"NFL 2021, semaines 1-8 · {fr(len(DF), 0)} dropbacks · {DF.gameId.nunique()} matchs · "
                       "tracking Next Gen Stats + notes PFF", className="sub")], className="app-header"),
    dcc.Tabs(id="tabs", value="guide", className="tabs-bar", children=[
        dcc.Tab(label="Guide", value="guide", className="tab", selected_className="tab--selected",
                children=html.Div(tab_guide, className="container")),
        dcc.Tab(label="Actions & replays", value="actions", className="tab", selected_className="tab--selected",
                children=html.Div(tab_actions, className="container")),
        dcc.Tab(label="Survie de la poche (blitz)", value="km", className="tab", selected_className="tab--selected",
                children=html.Div(tab_km, className="container")),
        dcc.Tab(label="Classement équipes", value="teams", className="tab", selected_className="tab--selected",
                children=html.Div(tab_teams, className="container")),
        dcc.Tab(label="Validation PFF", value="val", className="tab", selected_className="tab--selected",
                children=html.Div(tab_val, className="container")),
        dcc.Tab(label="Bloqueurs", value="blk", className="tab", selected_className="tab--selected",
                children=html.Div(tab_blk, className="container")),
    ]),
])


# =========================================================================== #
# Callbacks: actions
# =========================================================================== #
@app.callback(Output("f-game", "options"), Output("f-game", "value"),
              Input("f-week", "value"), Input("f-team", "value"), State("f-game", "value"))
def game_options(week, team, current):
    g = GAMES
    if week:
        g = g[g.week.eq(week)]
    if team != "ALL":
        g = g[g.homeTeamAbbr.eq(team) | g.visitorTeamAbbr.eq(team)]
    opts = [{"label": f"Tous les matchs ({len(g)})", "value": "ALL"}] + \
           [{"label": m, "value": int(gid)} for gid, m in zip(g.gameId, g.match)]
    value = current if current in set(g.gameId) else "ALL"
    return opts, value


def filtered(week, team, side, game, result, rush, pressure):
    d = DF
    if week:
        d = d[d.week.eq(week)]
    if team != "ALL":
        if side == "off":
            d = d[d.possessionTeam.eq(team)]
        elif side == "def":
            d = d[d.defensiveTeam.eq(team)]
        else:
            d = d[d.possessionTeam.eq(team) | d.defensiveTeam.eq(team)]
    if game not in ("ALL", None):
        d = d[d.gameId.eq(game)]
    if result != "ALL":
        d = d[d.passResult.eq(result)]
    if rush != "ALL":
        d = d[d.rush_bin.astype(str).eq(rush)]
    if pressure != "ALL":
        d = d[d.pressure_pff.eq(int(pressure))]
    return d


SORTERS = {"chrono": lambda x: x,
           "pis_asc": lambda x: x.sort_values("PIS"),
           "pis_desc": lambda x: x.sort_values("PIS", ascending=False),
           "ttp": lambda x: x.sort_values(["pressure_event", "time_to_pressure"], ascending=[False, True]),
           "ttt": lambda x: x.sort_values("time_to_throw", ascending=False)}


@app.callback(Output("t-actions", "data"), Output("n-actions", "children"),
              Input("f-week", "value"), Input("f-team", "value"), Input("f-side", "value"), Input("f-game", "value"),
              Input("f-result", "value"), Input("f-rush", "value"), Input("f-pressure", "value"),
              Input("f-sort", "value"))
def filter_actions(week, team, side, game, result, rush, pressure, sort):
    d = SORTERS[sort](filtered(week, team, side, game, result, rush, pressure))
    n = len(d)
    if not n:
        chips = [html.Span("Aucune action pour ces filtres.", className="chip")]
    else:
        chips = [html.Span([html.B(fr(n, 0)), " actions"], className="chip"),
                 html.Span(["PIS moyen ", html.B(fr(d.PIS.mean()))], className="chip"),
                 html.Span(["Pression PFF ", html.B(f"{fr(100 * d.pressure_pff.mean(), 0)} %")], className="chip"),
                 html.Span(["Sacks ", html.B(fr(d.sack.sum(), 0))], className="chip"),
                 html.Span(["Lancer moyen ", html.B(f"{fr(d.time_to_throw.mean(), 2)} s")], className="chip")]
    cols = [c["id"] for c in ACTION_COLUMNS] + ["id"]
    return records(d[cols]), chips


@app.callback(Output("sel-id", "data"),
              Input("t-actions", "active_cell"), Input("btn-prev", "n_clicks"), Input("btn-next", "n_clicks"),
              Input("t-actions", "derived_virtual_row_ids"), State("sel-id", "data"))
def select_play(active, _p, _n, row_ids, current):
    row_ids = row_ids or []
    trig = ctx.triggered_id
    if trig == "t-actions" and ctx.triggered[0]["prop_id"].endswith("active_cell") and active:
        return active.get("row_id", current)
    if trig in ("btn-prev", "btn-next") and row_ids:
        i = row_ids.index(current) if current in row_ids else -1
        i = max(i - 1, 0) if trig == "btn-prev" else min(i + 1, len(row_ids) - 1)
        return row_ids[i]
    if current in row_ids:
        return no_update
    return row_ids[0] if row_ids else None


def verdict(r) -> html.Div:
    cat, color = pis_category(r.PIS)
    end = "sack" if r.sack else "lancer"
    parts = [f"PIS {fr(r.PIS, 0)} : meilleur que {fr(r.pis_pct, 0)} % des dropbacks de la ligue."]
    if r.pressure_event:
        parts.append(f"Un rusher est arrivé à moins de 2 yards du QB après {fr(r.time_to_pressure)} s, "
                     f"{end} à {fr(r.time_to_throw)} s.")
    else:
        parts.append(f"Aucun rusher à moins de 2 yards avant le {end} ({fr(r.time_to_throw)} s) : la poche a tenu.")
    parts.append("PFF a noté une pression (hit, hurry ou sack)." if r.pressure_pff else "PFF n'a noté aucune pression.")
    exp = "plus long" if r.time_to_throw > AVG["ttt"] else "plus court"
    parts.append(f"Lancer {exp} que la moyenne ({fr(AVG['ttt'])} s) ; une fois corrigé, PIS ajusté = {fr(r.PIS_adj, 0)}.")
    return html.Div([html.Div(cat, className="t"), html.Div(" ".join(parts))], className="verdict",
                    style={"background": color})


@app.callback(Output("g-replay", "figure"), Output("g-timeline", "figure"), Output("play-head", "children"),
              Output("play-verdict", "children"), Output("play-cards", "children"),
              Input("sel-id", "data"), Input("view", "value"))
def show_play(sel, view):
    if not sel:
        return {}, {}, html.Div("Aucune action sélectionnée."), None, []
    gid, pid = map(int, sel.split("-"))
    r = DF[DF.id.eq(sel)].iloc[0]
    _, pcol = pis_category(r.PIS)
    head = html.Div([
        html.Div([html.Div(f"{r.possessionTeam} en attaque contre {r.defensiveTeam} · {r.match}", className="title"),
                  html.Div(f"Q{r.quarter} · {r.gameClock} · {r.down}e & {r.yardsToGo} · {r.result_txt}"
                           + (" · play-action" if r.play_action else ""), style={"fontWeight": 600}),
                  html.Div(str(r.playDescription), className="desc")]),
        html.Div(f"gameId {gid} · playId {pid}", style={"color": "#94a3b8", "fontSize": 12}),
    ], className="play-head")
    ttp = (f"{fr(r.time_to_pressure)} s", "1er rusher à moins de 2 yd du QB", "#dc2626") if r.pressure_event else \
          ("aucune", "personne à moins de 2 yd avant le lancer", "#16a34a")
    cards = [
        metric("PIS", fr(r.PIS, 0), f"sur 100 · meilleur que {fr(r.pis_pct, 0)} % des jeux", pcol),
        metric("PIS ajusté", fr(r.PIS_adj, 0), f"50 = normal pour un lancer en {fr(r.time_to_throw)} s"),
        metric("Temps de lancer", f"{fr(r.time_to_throw)} s", f"snap → {'sack' if r.sack else 'lancer'} · moyenne {fr(AVG['ttt'])} s"),
        metric("1re pression", *ttp),
        metric("Coussin moyen", f"{fr(r.cushion)} yd", f"distance QB ↔ rusher le plus proche · moy. {fr(AVG['cushion'])}"),
        metric("Temps propre", f"{fr(100 * r.clean_time_pct, 0)} %", f"sans rusher à < 2 yd · moy. {fr(100 * AVG['clean'], 0)} %"),
        metric("Aire de la poche", f"{fr(r.pocket_area, 0)} yd²", f"moyenne des bloqueurs · ligue {fr(AVG['area'], 0)}"),
        metric("Rushers", str(r.n_rushers), ("blitz · " if r.n_rushers >= 6 else "") + f"couverture {r.cover_txt}",
               "#dc2626" if r.n_rushers >= 6 else "#0f172a"),
        metric("Pression PFF", r.pressure_txt, "hit, hurry ou sack noté à la vidéo",
               "#dc2626" if r.pressure_pff else "#16a34a"),
    ]
    try:
        rep, tl = replay_dict(gid, pid, view), timeline_dict(gid, pid)
    except Exception as e:  # a tracking issue on one play must not break the app
        return {}, {}, head, html.Div(f"Replay indisponible : {e}"), cards
    return rep, tl, head, verdict(r), cards


@app.callback(Output("tabs", "value"), Output("f-team", "value"), Output("f-week", "value"), Output("f-side", "value"),
              Input("t-teams", "active_cell"), State("t-teams", "derived_virtual_data"), prevent_initial_call=True)
def team_to_actions(active, data):
    if not active or not data:
        return no_update, no_update, no_update, no_update
    return "actions", data[active["row"]]["possessionTeam"], 0, "off"


# =========================================================================== #
# Callbacks: survival
# =========================================================================== #
@app.callback(Output("g-km", "figure"), Output("km-table", "children"), Output("km-takeaway", "children"),
              Input("km-by", "value"), Input("km-radius", "value"), Input("km-team", "value"), Input("km-week", "value"))
def km_tab(by, radius, team, week):
    d = DF
    if team != "ALL":
        d = d[d.possessionTeam.eq(team)]
    if week:
        d = d[d.week.eq(week)]
    min_n = 15 if (team != "ALL" or week) else 20
    fig = viz.km_generic(d, by, KM_ORDERS.get(by), radius, KM_LABELS[by], min_n=min_n)
    fig.update_layout(title=None, autosize=True, width=None, margin=dict(t=30))
    tag = str(float(radius)).replace(".", "_")
    tab = M.km_table(d, by, time_col=f"time_to_pressure_r{tag}", event_col=f"pressure_event_r{tag}",
                     order=KM_ORDERS.get(by))
    tab = tab[tab.n_plays >= min_n]
    if len(tab) >= 2:
        hi, lo = tab.loc[tab["S(2.5s)"].idxmax()], tab.loc[tab["S(2.5s)"].idxmin()]
        scope = "la ligue" if team == "ALL" else team
        msg = [html.B("À retenir : "),
               f"2,5 s après le snap ({scope}), {fr(100 * hi['S(2.5s)'], 0)} % des poches sont encore propres avec "
               f"{KM_LABELS[by]} = « {hi.group} », contre {fr(100 * lo['S(2.5s)'], 0)} % avec « {lo.group} »."]
    else:
        msg = "Pas assez d'actions pour comparer ces groupes."
    return fig, (table("t-km", tab, KM_COLUMNS, page_size=10) if len(tab) else html.Div()), msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"Ouvre http://127.0.0.1:{a.port} dans ton navigateur (Ctrl+C pour arrêter).")
    app.run(host="127.0.0.1", port=a.port, debug=a.debug)


if __name__ == "__main__":
    main()
