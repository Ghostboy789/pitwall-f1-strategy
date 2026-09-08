"""TRAP T2 - degradation under a censored, endogenous pit decision.

The problem, stated precisely
-----------------------------
Teams do not observe tyres to destruction. They pit when the tyre is finished,
which means the end of every degradation curve is missing - and missing
*non-randomly*. A stint that was degrading badly gets cut short; a stint that
was holding up runs long. So the observations at high tyre age are dominated
by the tyres that happened to be behaving.

Fit a naive regression of lap time on tyre age and the slope at high age is
estimated almost entirely from the well-behaved stints. Degradation comes out
understated - and in this dataset it comes out *negative* at several circuits,
which would mean tyres getting faster as they age.

This is selection on the covariate, not classical response censoring, and the
selection is informative: the thing that determines whether an observation
exists is the thing being measured.

What is done about it
---------------------
Four pieces, in order:

1. ``naive_fit``        the biased baseline, kept deliberately so the
                        correction can be compared against something.
2. ``selection_evidence`` a direct test that the mechanism is real: is a
                        stint's own early degradation predictive of how soon
                        it ends? If teams pit in response to degradation, the
                        relationship is negative.
3. ``fit_stint_survival`` an accelerated-failure-time model of stint length,
                        treating a stint that ended at the flag, in a
                        retirement or under a red flag as right-censored,
                        because those did not end on a tyre judgement.
4. ``ipcw_fit``         the degradation slope refitted with each lap weighted
                        by the inverse probability its stint survived to that
                        tyre age, so the rare old-tyre observations carry the
                        weight the missing ones would have.

The assumption this rests on, stated plainly
--------------------------------------------
Inverse-probability weighting requires that, conditional on the covariates in
the survival model, the pit decision is independent of the *residual*
degradation. That is an approximation. Teams see live tyre temperatures,
degradation trends and radio traffic that this model does not, so some
informative selection certainly survives conditioning. The correction should
therefore be read as recovering part of the bias, not all of it - and the
direction of what remains is known: still understated.

Weights are clipped (see ``MAX_WEIGHT``) because a survival probability near
zero would otherwise let a single lap dominate the fit.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from pitwall import config

log = logging.getLogger("pitwall.models.degradation")

MIN_STINT_LAPS = config.MIN_STINT_LAPS_FOR_DEG
MAX_WEIGHT = 10.0
MIN_STINTS_FOR_SURVIVAL = 60
EARLY_STINT_LAPS = 5


def with_compound_rank(stints: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    """Attach each stint's within-event compound rank, idempotently.

    Callers legitimately arrive with the column already present; merging again
    would produce ``_x``/``_y`` suffixes and a KeyError further down.
    """
    if "compound_rank_label" in stints.columns:
        return stints
    rank = pace[["stint_id", "compound_rank_label"]].drop_duplicates("stint_id")
    return stints.merge(rank, on="stint_id", how="left")


def _stint_frame(pace: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Laps joined to their stint's metadata, filtered to usable stints."""
    keep = stints[stints["stint_length"] >= MIN_STINT_LAPS]["stint_id"]
    d = pace[pace["stint_id"].isin(set(keep))].copy()
    d["tyre_age"] = pd.to_numeric(d["TyreLife"], errors="coerce")
    return d.dropna(subset=["tyre_age", "lap_time_s"])


def _within_car_centred(d: pd.DataFrame) -> pd.DataFrame:
    """Centre within each car-race, and build the fuel control.

    Why not within *stint* (TRAP T5, and a real bug this code once had).
    Inside a single stint, tyre age and laps completed advance one-for-one, so
    their deviations from the stint mean are the identical vector. Regressing
    on tyre age alone with stint-level centring therefore estimates
    (degradation - fuel burn), not degradation. It produced a negative slope
    in every circuit and compound cell in this dataset: tyres apparently
    getting faster as they aged.

    Centring within the **car-race** instead keeps what identifies them apart.
    Across a driver's two or three stints, tyre age resets at every stop while
    laps completed keeps climbing, so a lap at race-lap 40 might be on a
    5-lap-old tyre or a 25-lap-old one. That between-stint variation separates
    the two effects, while the car-race mean still absorbs car pace, driver
    skill and the day's conditions.
    """
    out = d.copy()
    g = out.groupby("car_id")
    out["lap_time_dev"] = out["lap_time_s"] - g["lap_time_s"].transform("mean")
    out["tyre_age_dev"] = out["tyre_age"] - g["tyre_age"].transform("mean")
    fuel = pd.to_numeric(out["fuel_laps_burned"], errors="coerce")
    out["fuel_dev"] = fuel - out.groupby("car_id")["fuel_laps_burned"].transform("mean")
    return out.dropna(subset=["lap_time_dev", "tyre_age_dev", "fuel_dev"])


def _wls_multi(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Weighted least squares through the origin on centred data.

    Returns ``(coefficients, standard_errors, n, vif_of_first_column)``.
    The VIF is reported so the fuel/tyre-age separation is demonstrated rather
    than assumed - it is the whole basis of the identification above.
    """
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(x).all(axis=1)
    x, y, w = x[ok], y[ok], w[ok]
    n = len(y)
    k = x.shape[1]
    if n < 20:
        return np.full(k, np.nan), np.full(k, np.nan), n, np.nan

    sw = np.sqrt(w)
    xw, yw = x * sw[:, None], y * sw
    xtx = xw.T @ xw
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), np.full(k, np.nan), n, np.inf
    beta = xtx_inv @ (xw.T @ yw)
    resid = yw - xw @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))

    # VIF of tyre age against the other regressors.
    vif = np.nan
    if k > 1:
        other = x[:, 1:]
        try:
            b = np.linalg.lstsq(other, x[:, 0], rcond=None)[0]
            r = x[:, 0] - other @ b
            ss_tot = float(((x[:, 0] - x[:, 0].mean()) ** 2).sum())
            r2 = 1.0 - float((r**2).sum()) / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / max(1.0 - r2, 1e-9)
        except np.linalg.LinAlgError:
            vif = np.inf
    return beta, se, n, float(vif)


def naive_fit(pace: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Degradation slope per circuit and compound rank, uncorrected.

    This is the number the project is trying to improve on. It is computed and
    published so the correction has something honest to be compared against.
    """
    d = _within_car_centred(_stint_frame(pace, stints))
    return _slopes_by_cell(d, weights=None, method="naive")


def _slopes_by_cell(d: pd.DataFrame, weights: pd.Series | None, method: str) -> pd.DataFrame:
    """Degradation slope per circuit x compound rank, controlling for fuel."""
    rows = []
    for (circuit, rank), grp in d.groupby(["circuit", "compound_rank_label"]):
        if len(grp) < 50 or grp["stint_id"].nunique() < 5:
            continue
        x = np.column_stack(
            [grp["tyre_age_dev"].to_numpy(float), grp["fuel_dev"].to_numpy(float)]
        )
        y = grp["lap_time_dev"].to_numpy(float)
        w = np.ones(len(grp)) if weights is None else weights.loc[grp.index].to_numpy(float)
        beta, se, n, vif = _wls_multi(x, y, w)
        row = {
            "circuit": circuit,
            "compound_rank_label": rank,
            "method": method,
            "slope_s_per_lap": float(beta[0]),
            "slope_se": float(se[0]),
            "fuel_s_per_lap": float(beta[1]),
            "fuel_se": float(se[1]),
            "tyre_age_vif": vif,
            "n_laps": int(n),
            "n_stints": int(grp["stint_id"].nunique()),
        }
        if weights is not None:
            row["mean_weight"] = float(w.mean())
            row["max_weight"] = float(w.max())
        rows.append(row)
    return pd.DataFrame(rows)


def selection_evidence(pace: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Direct test that the pit decision responds to degradation.

    For every stint long enough to measure it, fit the slope over its first
    few laps, then ask whether that early slope predicts how long the stint
    lasted. A negative relationship means stints that started degrading fast
    were ended sooner - i.e. the selection mechanism this module corrects for
    is real and not merely assumed.

    Restricted to stints that ended in a pit stop, since those are the only
    ones where a decision was actually taken.
    """
    from scipy import stats

    d = _stint_frame(pace, stints)
    ended = set(stints.loc[stints["event_observed"].astype(bool), "stint_id"])
    d = d[d["stint_id"].isin(ended)]

    rows = []
    for stint_id, grp in d.groupby("stint_id"):
        grp = grp.sort_values("tyre_age")
        early = grp.head(EARLY_STINT_LAPS)
        if len(early) < EARLY_STINT_LAPS or len(grp) < MIN_STINT_LAPS:
            continue
        x = early["tyre_age"].to_numpy(float)
        y = early["lap_time_s"].to_numpy(float)
        if np.ptp(x) == 0:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        rows.append(
            {
                "stint_id": stint_id,
                "early_slope": slope,
                "stint_length": int(len(grp)),
                "circuit": grp["circuit"].iloc[0],
            }
        )

    if len(rows) < 100:
        return {"status": "insufficient_stints", "n_stints": len(rows)}

    e = pd.DataFrame(rows)
    # Trim the wildest 1% of early slopes: an out-lap or traffic can produce a
    # meaningless slope over five laps. Symmetric, and fixed in advance.
    lo, hi = e["early_slope"].quantile([0.005, 0.995])
    e = e[e["early_slope"].between(lo, hi)]

    r, p = stats.pearsonr(e["early_slope"], e["stint_length"])
    rho, prho = stats.spearmanr(e["early_slope"], e["stint_length"])
    return {
        "status": "ok",
        "n_stints": int(len(e)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(prho),
        "interpretation": (
            "negative correlation = stints degrading faster were pitted sooner, "
            "which is the informative selection this module corrects"
        ),
    }


def fit_stint_survival(stints: pd.DataFrame):
    """Accelerated-failure-time model of stint length.

    A stint is an *observed* pit decision only if it ended in a pit stop.
    Stints that ran to the chequered flag, ended in a retirement, or were
    interrupted by a red flag are right-censored: they say the tyre lasted at
    least that long, not that anyone judged it finished. Treating them as
    completed decisions is the bias this whole module exists to remove.
    """
    from lifelines import WeibullAFTFitter

    d = stints.copy()
    d = d[d["stint_length"] >= MIN_STINT_LAPS]
    d = d.dropna(subset=["stint_length", "event_observed", "circuit", "compound_rank_label"])
    if len(d) < MIN_STINTS_FOR_SURVIVAL:
        log.warning("only %d stints; survival model not fitted", len(d))
        return None, d

    d["start_progress"] = d["start_lap"] / d["race_laps"].replace(0, np.nan)
    model_df = pd.DataFrame(
        {
            "duration": d["stint_length"].astype(float),
            "observed": d["event_observed"].astype(bool),
            "circuit": d["circuit"].astype(str),
            "rank_lab": d["compound_rank_label"].astype(str),
            "era": d["era"].astype(str),
            "start_progress": d["start_progress"].fillna(0.5).astype(float),
            "fresh": d["fresh_tyre"].fillna(True).astype(bool).astype(int),
        }
    )
    model_df = pd.get_dummies(
        model_df, columns=["circuit", "rank_lab", "era"], drop_first=True, dtype=float
    )

    aft = WeibullAFTFitter(penalizer=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft.fit(model_df, duration_col="duration", event_col="observed")
    log.info(
        "stint survival fitted: %d stints, %d uncensored (%.0f%%), concordance %.3f",
        len(model_df), int(model_df["observed"].sum()),
        100 * model_df["observed"].mean(), aft.concordance_index_,
    )
    return aft, d


def _survival_weights(aft, stint_meta: pd.DataFrame, laps: pd.DataFrame) -> pd.Series:
    """1 / P(stint survives to this tyre age), clipped.

    An old-tyre lap that only one stint in twenty ever reached stands in for
    the nineteen that were pitted before getting there, so it carries more
    weight. Clipping stops a near-zero survival probability from letting a
    single lap decide the answer.
    """
    idx = stint_meta.set_index("stint_id")
    covars = aft.params_.index.get_level_values(1).unique()

    weights = pd.Series(1.0, index=laps.index)
    for stint_id, grp in laps.groupby("stint_id"):
        if stint_id not in idx.index:
            continue
        row = idx.loc[[stint_id]]
        x = pd.DataFrame(index=row.index)
        for c in covars:
            if c in ("Intercept", "_intercept"):
                continue
            x[c] = row[c].values if c in row.columns else 0.0
        ages = grp["tyre_age"].to_numpy(float)
        try:
            sf = aft.predict_survival_function(x, times=np.unique(ages))
            s = sf.iloc[:, 0].reindex(np.unique(ages)).to_numpy(float)
            lookup = dict(zip(np.unique(ages), s))
            w = np.array([1.0 / max(lookup.get(a, 1.0), 1e-3) for a in ages])
        except Exception:  # noqa: BLE001
            w = np.ones_like(ages)
        weights.loc[grp.index] = np.clip(w, 1.0, MAX_WEIGHT)
    return weights


def ipcw_fit(pace: pd.DataFrame, stints: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Degradation slope corrected by inverse-probability-of-censoring weights."""
    aft, stint_meta = fit_stint_survival(with_compound_rank(stints, pace))
    d = _within_car_centred(_stint_frame(pace, stints))
    if aft is None:
        d["w"] = 1.0
    else:
        # Rebuild the dummy columns the fitted model expects.
        meta = stint_meta.copy()
        meta["start_progress"] = meta["start_lap"] / meta["race_laps"].replace(0, np.nan)
        meta["fresh"] = meta["fresh_tyre"].fillna(True).astype(bool).astype(int)
        meta = pd.concat(
            [
                meta[["stint_id", "start_progress", "fresh"]].reset_index(drop=True),
                pd.get_dummies(
                    meta[["circuit", "compound_rank_label", "era"]].rename(
                        columns={"compound_rank_label": "rank_lab"}
                    ),
                    columns=["circuit", "rank_lab", "era"],
                    drop_first=True, dtype=float,
                ).reset_index(drop=True),
            ],
            axis=1,
        )
        meta["start_progress"] = meta["start_progress"].fillna(0.5)
        d["w"] = _survival_weights(aft, meta, d)

    return _slopes_by_cell(d, weights=d["w"], method="ipcw"), aft


def compare(pace: pd.DataFrame, stints: pd.DataFrame, save: bool = True) -> dict:
    """The committed deliverable: naive vs censoring-corrected, side by side.

    Published whichever direction it goes, including if the correction turns
    out to change little.
    """
    naive = naive_fit(pace, stints)
    corrected, aft = ipcw_fit(pace, stints)
    evidence = selection_evidence(pace, stints)

    merged = naive.merge(
        corrected,
        on=["circuit", "compound_rank_label"],
        suffixes=("_naive", "_ipcw"),
        how="inner",
    )
    merged["delta_s_per_lap"] = (
        merged["slope_s_per_lap_ipcw"] - merged["slope_s_per_lap_naive"]
    )
    merged["pct_change"] = 100 * merged["delta_s_per_lap"] / merged["slope_s_per_lap_naive"].abs()

    summary = {
        "n_cells": int(len(merged)),
        "median_naive_slope": float(merged["slope_s_per_lap_naive"].median()),
        "median_ipcw_slope": float(merged["slope_s_per_lap_ipcw"].median()),
        "median_delta": float(merged["delta_s_per_lap"].median()),
        "share_increased": float((merged["delta_s_per_lap"] > 0).mean()),
        "n_naive_negative": int((merged["slope_s_per_lap_naive"] < 0).sum()),
        "n_ipcw_negative": int((merged["slope_s_per_lap_ipcw"] < 0).sum()),
    }

    if save:
        merged.to_parquet(config.MODELS_OUT / "degradation_naive_vs_ipcw.parquet", index=False)
        pd.DataFrame([summary]).to_csv(config.MODELS_OUT / "degradation_summary.csv", index=False)
        pd.DataFrame([evidence]).to_csv(config.MODELS_OUT / "selection_evidence.csv", index=False)

    return {
        "comparison": merged,
        "summary": summary,
        "selection_evidence": evidence,
        "aft": aft,
    }
