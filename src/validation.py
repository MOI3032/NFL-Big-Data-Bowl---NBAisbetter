"""Validation of PIS and time-to-pressure against PFF charting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score

from . import metrics as M


def _ttp_score(df: pd.DataFrame, col="time_to_pressure", ev="pressure_event") -> pd.Series:
    """Higher = more pressure. Censored plays (pocket held) rank as the cleanest."""
    return -np.where(df[ev].eq(1), df[col], 99.0)


def play_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["pressure_pff", "sack", "beaten"]:
        y = df[label]
        r, p = pointbiserialr(y, df.PIS)
        rows.append({"label": label, "score": "PIS (low = pressure)", "AUC": roc_auc_score(y, -df.PIS),
                     "point_biserial_r": r, "p_value": p})
        r, p = pointbiserialr(y, df.PIS_adj)
        rows.append({"label": label, "score": "PIS_adj (time-to-throw adjusted)",
                     "AUC": roc_auc_score(y, -df.PIS_adj), "point_biserial_r": r, "p_value": p})
        s = _ttp_score(df)
        r, p = pointbiserialr(y, s)
        rows.append({"label": label, "score": "time_to_pressure (censored = clean)",
                     "AUC": roc_auc_score(y, s), "point_biserial_r": r, "p_value": p})
        rows.append({"label": label, "score": "time_to_throw (baseline)",
                     "AUC": roc_auc_score(y, df.time_to_throw),
                     "point_biserial_r": pointbiserialr(y, df.time_to_throw)[0], "p_value": np.nan})
    return pd.DataFrame(rows).round(4)


def tracking_vs_pff(df: pd.DataFrame) -> pd.DataFrame:
    """Confusion table: tracking 2-yd pressure event vs PFF pressure label."""
    return pd.crosstab(df.pressure_event.map({0: "tracking: pocket held", 1: "tracking: rusher ≤2 yd"}),
                       df.pressure_pff.map({0: "PFF: clean", 1: "PFF: pressure"}), margins=True)


def weight_sensitivity(df: pd.DataFrame, min_plays: int) -> pd.DataFrame:
    """Recompute PIS under alternative weights: play AUC and team-rank stability."""
    base = M.team_table(M.add_pis(df), min_plays).set_index("possessionTeam")
    rows = []
    for name, w in M.WEIGHT_SETS.items():
        d = df.copy()
        d["PIS"] = M.pis_score(d, w)
        slope, icpt = np.polyfit(d.time_to_throw, d.PIS, 1)
        d["PIS_adj"] = 50 + d.PIS - (icpt + slope * d.time_to_throw)
        t = M.team_table(d, min_plays).set_index("possessionTeam").loc[base.index]
        rows.append({"weights": name, "AUC_pressure": roc_auc_score(d.pressure_pff, -d.PIS),
                     "spearman_team_rank_vs_baseline": spearmanr(t.PIS_adj, base.PIS_adj)[0],
                     "spearman_team_PISadj_vs_pressure_rate": spearmanr(t.PIS_adj, t.pressure_rate_pff)[0],
                     "top5_teams": ", ".join(t.PIS_adj.sort_values(ascending=False).index[:5])})
    return pd.DataFrame(rows).round(3)


def team_level(teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlations and teams where PIS disagrees with sack rate."""
    rows = []
    for score in ["PIS", "PIS_adj"]:
        for target in ["pressure_rate_pff", "sack_rate", "beaten_rate"]:
            rho, p = spearmanr(teams[score], teams[target])
            rows.append({"team_score": score, "vs": target, "spearman_rho": rho, "p_value": p})
    corr = pd.DataFrame(rows).round(4)

    t = teams.copy()
    t["sack_rank"] = t.sack_rate.rank()          # 1 = fewest sacks
    t["pis_rank"] = t.PIS_adj.rank(ascending=False)  # 1 = cleanest pocket
    n = len(t)
    third = n / 3
    t["profile"] = np.select(
        [(t.sack_rank <= third) & (t.pis_rank > 2 * third),
         (t.sack_rank > 2 * third) & (t.pis_rank <= third)],
        ["few sacks but poor pocket (sacks hide pressure)",
         "many sacks but solid pocket (sacks not on the line)"], "")
    disagree = t[t.profile.ne("")][["possessionTeam", "profile", "sack_rate", "PIS_adj",
                                    "pressure_rate_pff", "time_to_throw"]]
    return corr, disagree.round(3)


def subset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers with and without play-action / pocket-roll plays."""
    subsets = {
        "all kept plays": df,
        "excluding play-action": df[df.play_action.eq(0)],
        "excluding pocket-roll (PR)": df[df.pocket_roll.eq(0)],
        "excluding play-action and PR": df[df.play_action.eq(0) & df.pocket_roll.eq(0)],
        "play-action only": df[df.play_action.eq(1)],
    }
    rows = []
    for name, d in subsets.items():
        km = M.kaplan_meier(d.time_to_pressure, d.pressure_event)
        rows.append({"subset": name, "n_plays": len(d), "mean_PIS": d.PIS.mean(),
                     "mean_time_to_throw_s": d.time_to_throw.mean(),
                     "pff_pressure_rate": d.pressure_pff.mean(),
                     "AUC_PIS_vs_pff_pressure": roc_auc_score(d.pressure_pff, -d.PIS),
                     "KM_median_time_to_pressure_s": M.km_median(km)})
    return pd.DataFrame(rows).round(3)
