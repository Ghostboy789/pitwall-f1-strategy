"""Phase 3 - probability of completing a pass. The headline layer.

Everything the project exists to say comes out of this module, so it is built
to be checked rather than believed: the pass detector is validated against
independently known overtake counts, and the probability model is reported
with a reliability curve, not just a discrimination score.

Defining an overtake from lap data
----------------------------------
There is no "overtake" field. A pass is inferred from consecutive laps:

    at lap L    A is directly behind B
    at lap L+1  A is ahead of B
    and neither car pitted across that boundary

The pit exclusion is what separates a *racing* pass from a position swap
bought in the pit lane, and it is the single easiest way to get this wrong -
without it, every undercut is miscounted as an overtake and Monaco looks as
easy to pass at as Monza.

The denominator matters as much as the numerator
------------------------------------------------
A pass probability needs the attempts, not just the successes. An
*opportunity* is a lap where A ran directly behind B within striking distance
(``ATTACK_GAP_S``). Counting passes alone would measure how often cars were
close, not how often closeness converted.

What this yields
----------------
For each circuit, the per-lap probability that a car of a given pace advantage
converts a following position into a pass. The reciprocal of that probability
is how many laps a faster car spends stuck, and that - multiplied by the pace
it is being held to - is the value of track position in seconds. See
``pitwall.trackposition``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.circuits import CIRCUIT_REF

log = logging.getLogger("pitwall.models.overtaking")

# A car more than this far back is not attacking, it is merely next in line.
# Chosen before fitting: ~2.0 s is the usual "dirty air" range and 1.0 s is the
# DRS activation threshold, so 2.0 keeps genuine attacks without diluting the
# denominator with cars circulating in clear air.
ATTACK_GAP_S = 2.0

# DRS is enabled from lap 3 of a race and requires being within 1.0 s at the
# detection point. Approximated here by gap and lap number; the approximation
# is stated in the methodology rather than presented as a measurement.
DRS_GAP_S = 1.0
DRS_FROM_LAP = 3

MIN_OPPORTUNITIES_FOR_CIRCUIT = 100


def clean_air_pace(pace: pd.DataFrame) -> pd.DataFrame:
    """Each car's race pace when not held up, corrected for fuel.

    A car stuck behind another runs the *other* car's pace, so recent lap
    times are the one thing that must not be used to measure how much faster
    it really is. Only laps in clear air count, and they are corrected to a
    common fuel load so a stint early in the race is comparable with one late.
    """
    d = pace.copy()
    d["lap_time_s"] = pd.to_numeric(d["lap_time_s"], errors="coerce")
    clear = d[(d["gap_ahead_s"] > ATTACK_GAP_S) | ~np.isfinite(d["gap_ahead_s"])]

    # Fuel correction: bring every lap to mid-race fuel load using the effect
    # estimated from data (pitwall.models.pace), not a textbook constant.
    fuel_coef = -0.06
    mid = clear.groupby("race_id")["fuel_laps_burned"].transform("mean")
    adj = clear["lap_time_s"] - fuel_coef * (clear["fuel_laps_burned"] - mid)

    out = (
        clear.assign(pace_adj=adj)
        .groupby("car_id")
        .agg(
            clean_pace=("pace_adj", "median"),
            n_clean_laps=("pace_adj", "size"),
            circuit=("circuit", "first"),
            race_id=("race_id", "first"),
        )
        .reset_index()
    )
    return out[out["n_clean_laps"] >= 3]


def build_opportunities(laps: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    """One row per lap where a car ran directly behind another within reach.

    ``passed`` is the outcome: did the following car complete the move on the
    next lap, without either car pitting across the boundary.
    """
    d = laps[
        [
            "race_id",
            "circuit",
            "era",
            "year",
            "Driver",
            "Team",
            "LapNumber",
            "Position",
            "Time",
            "TyreLife",
            "Compound",
            "compound_rank",
            "is_inlap",
            "is_outlap",
            "is_green",
            "race_laps",
            "gap_ahead_s",
        ]
    ].copy()
    d = d.dropna(subset=["Position", "LapNumber", "Time"])
    d["Position"] = d["Position"].astype(int)
    d["LapNumber"] = d["LapNumber"].astype(int)
    d["car_id"] = d["race_id"] + "_" + d["Driver"].astype(str)

    cp = clean_air_pace(pace).set_index("car_id")["clean_pace"]

    rows = []
    for race_id, race in d.groupby("race_id", sort=False):
        by_lap = {ln: g.set_index("Position") for ln, g in race.groupby("LapNumber")}
        laps_sorted = sorted(by_lap)
        for ln, nxt in zip(laps_sorted, laps_sorted[1:]):
            if nxt != ln + 1:
                continue
            cur, fut = by_lap[ln], by_lap[nxt]
            for pos in cur.index:
                if pos - 1 not in cur.index:
                    continue  # no car ahead
                follower = cur.loc[pos]
                leader = cur.loc[pos - 1]
                if isinstance(follower, pd.DataFrame) or isinstance(leader, pd.DataFrame):
                    continue  # duplicated position in the feed; skip rather than guess
                gap = follower["Time"] - leader["Time"]
                if not np.isfinite(gap) or gap <= 0 or gap > ATTACK_GAP_S:
                    continue
                if not bool(follower["is_green"]) or not bool(leader["is_green"]):
                    continue

                f_id, l_id = follower["car_id"], leader["car_id"]
                f_next = fut[fut["car_id"] == f_id]
                l_next = fut[fut["car_id"] == l_id]
                if f_next.empty or l_next.empty:
                    continue  # someone did not complete the next lap
                f_next, l_next = f_next.iloc[0], l_next.iloc[0]

                # A position swap bought in the pit lane is not an overtake.
                if any(
                    bool(x)
                    for x in (
                        follower["is_inlap"],
                        leader["is_inlap"],
                        f_next["is_outlap"],
                        l_next["is_outlap"],
                        f_next["is_inlap"],
                        l_next["is_inlap"],
                    )
                ):
                    continue

                passed = int(f_next.name < l_next.name)
                fp, lp = cp.get(f_id, np.nan), cp.get(l_id, np.nan)
                rows.append(
                    {
                        "race_id": race_id,
                        "circuit": follower["circuit"],
                        "era": follower["era"],
                        "year": follower["year"],
                        "lap": ln,
                        "race_laps": follower["race_laps"],
                        "follower": f_id,
                        "leader": l_id,
                        "position": int(pos),
                        "gap_s": float(gap),
                        "pace_delta_s": float(fp - lp)
                        if np.isfinite(fp) and np.isfinite(lp)
                        else np.nan,
                        "tyre_age_delta": float(
                            (follower["TyreLife"] or np.nan) - (leader["TyreLife"] or np.nan)
                        ),
                        "compound_rank_delta": float(
                            (
                                follower["compound_rank"]
                                if pd.notna(follower["compound_rank"])
                                else np.nan
                            )
                            - (
                                leader["compound_rank"]
                                if pd.notna(leader["compound_rank"])
                                else np.nan
                            )
                        ),
                        "drs_available": int(gap <= DRS_GAP_S and ln >= DRS_FROM_LAP),
                        "passed": passed,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["lap_progress"] = out["lap"] / out["race_laps"]
    out["drs_zones"] = out["circuit"].map(
        lambda c: CIRCUIT_REF[c].drs_zones if c in CIRCUIT_REF else np.nan
    )
    out["street"] = out["circuit"].map(
        lambda c: int(CIRCUIT_REF[c].street) if c in CIRCUIT_REF else np.nan
    )
    log.info(
        "overtaking: %d opportunities across %d races, %d completed passes (%.1f%%)",
        len(out),
        out["race_id"].nunique(),
        int(out["passed"].sum()),
        100 * out["passed"].mean(),
    )
    return out


def detector_sanity(opps: pd.DataFrame) -> pd.DataFrame:
    """Passes per race by circuit, for comparison against known counts.

    This is the check that the detector measures overtaking rather than noise.
    Published so a reader can compare it with any public overtake tally: the
    ordering should put Monaco far below Monza, and a detector that does not
    is broken regardless of how good its model scores look.
    """
    per_race = opps.groupby(["circuit", "race_id"])["passed"].sum().reset_index()
    return (
        per_race.groupby("circuit")
        .agg(
            races=("race_id", "nunique"),
            mean_passes_per_race=("passed", "mean"),
            median_passes=("passed", "median"),
            min_passes=("passed", "min"),
            max_passes=("passed", "max"),
        )
        .reset_index()
        .sort_values("mean_passes_per_race")
    )


FEATURES = [
    "pace_delta_s",
    "gap_s",
    "drs_available",
    "tyre_age_delta",
    "compound_rank_delta",
    "lap_progress",
    "position",
    "drs_zones",
    "street",
]


def fit(opps: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Fit and calibrate the pass-probability model.

    Two models are fitted: a logistic regression, which is interpretable and
    naturally close to calibrated, and gradient boosting, which is not
    constrained to a linear log-odds surface. Both are reported. Whichever is
    used downstream, the reliability curve travels with it.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    d = opps.dropna(subset=["pace_delta_s"]).copy()
    x = d[FEATURES]
    y = d["passed"].to_numpy(int)
    groups = d["race_id"].to_numpy()

    # Split by RACE, never by row. Two opportunities from the same race share
    # weather, track state and the same two cars; a random split would leak
    # them across the fold boundary and flatter every score.
    cv = GroupKFold(n_splits=5)

    models = {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        ),
        "gbm": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.06,
                        max_iter=300,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    results = {}
    for name, pipe in models.items():
        oof = np.full(len(d), np.nan)
        for tr, te in cv.split(x, y, groups):
            m = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
            m.fit(x.iloc[tr], y[tr])
            oof[te] = m.predict_proba(x.iloc[te])[:, 1]
        results[name] = {
            "oof_pred": oof,
            "auc": float(roc_auc_score(y, oof)),
            "brier": float(brier_score_loss(y, oof)),
            "base_rate": float(y.mean()),
        }
        log.info(
            "%s: AUC %.3f  Brier %.4f  (base rate %.4f)",
            name,
            results[name]["auc"],
            results[name]["brier"],
            y.mean(),
        )

    # Refit the better-calibrated model on everything for downstream use.
    best = min(results, key=lambda k: results[k]["brier"])
    final = CalibratedClassifierCV(models[best], method="isotonic", cv=5)
    final.fit(x, y)

    return {
        "data": d,
        "y": y,
        "results": results,
        "best": best,
        "model": final,
        "features": FEATURES,
    }


MIN_OPPS_FOR_CIRCUIT_EFFECT = 80


def fit_hierarchical(opps: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Pass model with a partially-pooled per-circuit effect.

    Why this exists. The first version of this model had no circuit term at
    all: a circuit reached the prediction only through ``drs_zones`` and
    ``street``. Every circuit sharing that pair therefore received an
    identical probability - Monza, Spa, Silverstone, Catalunya, Interlagos and
    eight others all came out at exactly 0.083655. For a project whose entire
    output is a *per-circuit* number, the model could not express the quantity
    it existed to measure.

    Adding raw circuit dummies would fix that and break something else: a
    circuit with two races would get a confident effect estimated from two
    races. So it is done in two stages, the same way ``pitwall.models.pace``
    handles its circuit coefficients:

      1. a gradient-boosted model on the mechanics of passing - gap, pace
         advantage, DRS, tyre state - with no circuit identity at all, so it
         learns how passing works rather than where;
      2. a per-circuit intercept fitted on top of that model's log-odds as a
         GLM offset, then shrunk toward zero by empirical Bayes in proportion
         to its own precision.

    A circuit with eleven races keeps most of its estimated effect; one with
    two is pulled most of the way back to the average. Nothing is claimed for
    a circuit that has not earned it.
    """
    import statsmodels.api as sm
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    from pitwall.models.pace import pool_random_effects

    d = opps.dropna(subset=["pace_delta_s"]).copy()
    base_features = [f for f in FEATURES if f not in ("drs_zones", "street")]

    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.06,
                    max_iter=300,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    base = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    base.fit(d[base_features], d["passed"].to_numpy(int))

    p_base = np.clip(base.predict_proba(d[base_features])[:, 1], 1e-6, 1 - 1e-6)
    offset = np.log(p_base / (1 - p_base))

    counts = d.groupby("circuit")["race_id"].nunique()
    usable = [
        c
        for c in sorted(d["circuit"].unique())
        if (d["circuit"] == c).sum() >= MIN_OPPS_FOR_CIRCUIT_EFFECT
    ]
    dummies = pd.get_dummies(d["circuit"], dtype=float).reindex(columns=usable, fill_value=0.0)

    try:
        glm = sm.GLM(
            d["passed"].to_numpy(int),
            dummies.to_numpy(float),
            family=sm.families.Binomial(),
            offset=offset,
        ).fit_regularized(alpha=1e-4, L1_wt=0.0)
        raw = np.asarray(glm.params, dtype=float)
        # fit_regularized gives no standard errors; recover them from the
        # unregularised information matrix at the fitted point.
        unreg = sm.GLM(
            d["passed"].to_numpy(int),
            dummies.to_numpy(float),
            family=sm.families.Binomial(),
            offset=offset,
        ).fit(start_params=raw, maxiter=50)
        raw, se = np.asarray(unreg.params, float), np.asarray(unreg.bse, float)
    except Exception as exc:
        log.warning("circuit-effect GLM failed (%s); falling back to no effects", exc)
        raw = np.zeros(len(usable))
        se = np.full(len(usable), 1.0)

    mu, tau2, shrunk, shrunk_se = pool_random_effects(raw, se)
    effects = pd.DataFrame(
        {
            "circuit": usable,
            "n_races": [int(counts.get(c, 0)) for c in usable],
            "n_opps": [int((d["circuit"] == c).sum()) for c in usable],
            "logodds_raw": raw,
            "logodds_se": se,
            "logodds_shrunk": shrunk,
            "logodds_shrunk_se": shrunk_se,
        }
    ).sort_values("logodds_shrunk")

    log.info(
        "circuit effects: %d circuits, between-circuit SD %.3f log-odds, range %.2f to %.2f",
        len(effects),
        float(np.sqrt(tau2)),
        float(effects["logodds_shrunk"].min()),
        float(effects["logodds_shrunk"].max()),
    )

    return {
        "base_model": base,
        "base_features": base_features,
        "effects": effects,
        "global_mean_logodds": mu,
        "between_circuit_sd": float(np.sqrt(tau2)),
        "data": d,
    }


def predict_hierarchical(fitted: dict, frame: pd.DataFrame) -> np.ndarray:
    """Pass probability for rows carrying a ``circuit`` column."""
    p = np.clip(
        fitted["base_model"].predict_proba(frame[fitted["base_features"]])[:, 1],
        1e-6,
        1 - 1e-6,
    )
    lo = np.log(p / (1 - p))
    eff = fitted["effects"].set_index("circuit")["logodds_shrunk"]
    adj = frame["circuit"].map(eff).fillna(fitted["global_mean_logodds"]).to_numpy(float)
    return 1.0 / (1.0 + np.exp(-(lo + adj)))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve: when the model says 30%, does it happen 30% of the time?

    Validation V2. Published whatever it shows - miscalibration is a finding,
    not something to tune away after the fact.
    """
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
                "gap": float(p[m].mean() - y[m].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["ece"] = float((out["n"] / out["n"].sum() * out["gap"].abs()).sum())
    return out
