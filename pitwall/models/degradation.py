"""TRAP T2 - degradation under a censored, endogenous pit decision.

The problem, stated precisely
-----------------------------
Teams do not observe tyres to destruction. They pit when the tyre is finished,
so the end of every degradation curve is missing, and missing *non-randomly*:
a stint that was degrading badly gets cut short, a stint that was holding up
runs long. Observations at high tyre age are therefore dominated by the tyres
that happened to be behaving.

There is a second, larger selection on top of it, which cost this module two
rewrites to find. Teams do not merely choose *when* to pit, they choose *which
compound* to run in which conditions - the harder tyre goes on the long, hot,
high-fuel stint precisely because that is where degradation will be worst. So
the compounds are never observed under equal conditions, and a naive
comparison says harder tyres degrade faster than softer ones, which is
backwards.

Both are the same disease: what determines whether an observation exists is
the thing being measured.

The three estimators, and what each fixes
-----------------------------------------
Reported side by side, because the progression *is* the result:

1. ``naive_fit``      centres within the car-race and regresses on tyre age,
                      controlling for fuel. This is the biased baseline. On
                      this dataset it puts 22 of 91 circuit-compound cells at
                      *negative* degradation - tyres getting faster as they
                      age - and inverts the compound ordering.

2. ``twoway_fit``     two-way within transformation, removing both a car-race
                      mean and a **race-moment** mean (the race, bucketed into
                      five-lap windows). Every comparison is then between cars
                      running at the same point of the same race, which is the
                      only place the conditions are actually equal.

                      This also disposes of TRAP T5 without needing to solve
                      it: cars in the same race-moment carry the same fuel
                      load and the same track evolution, so both difference
                      out rather than having to be separately identified -
                      which, as ``models.pace`` documents, they cannot be.

                      Negative cells fall from 22 to 4 of 91, and the compound
                      ordering rights itself.

3. ``ipcw_fit``       the two-way estimator reweighted by the inverse
                      probability that each stint survived to that tyre age,
                      from an AFT survival model. This is the correction for
                      the censoring in (1).

The assumption this rests on, stated plainly
--------------------------------------------
Inverse-probability weighting requires that, conditional on the covariates in
the survival model, the pit decision is independent of the *residual*
degradation. That is an approximation: teams see live tyre temperatures,
degradation trends and radio traffic this model does not, so some informative
selection certainly survives conditioning. The correction recovers part of the
bias, not all of it, and the direction of what remains is known - still
understated.
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

# Width of the race-moment bucket, in laps. Small enough that fuel load and
# track state are near-constant inside one, wide enough that several cars on
# different compounds fall in the same bucket.
LAP_BUCKET = 5
DEMEAN_ITERS = 12

MIN_LAPS_PER_CELL = 50
MIN_STINTS_PER_CELL = 5


def with_compound_rank(stints: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    """Attach each stint's within-event compound rank, idempotently."""
    if "compound_rank_label" in stints.columns:
        return stints
    rank = pace[["stint_id", "compound_rank_label"]].drop_duplicates("stint_id")
    return stints.merge(rank, on="stint_id", how="left")


def _stint_frame(pace: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Laps joined to their stint's metadata, filtered to usable stints."""
    keep = stints[stints["stint_length"] >= MIN_STINT_LAPS]["stint_id"]
    d = pace[pace["stint_id"].isin(set(keep))].copy()
    d["tyre_age"] = pd.to_numeric(d["TyreLife"], errors="coerce")
    d["lap_bucket"] = (
        d["race_id"]
        + "_b"
        + (pd.to_numeric(d["LapNumber"]) // LAP_BUCKET).astype("Int64").astype(str)
    )
    return d.dropna(subset=["tyre_age", "lap_time_s"])


def _within_car_centred(d: pd.DataFrame) -> pd.DataFrame:
    """Baseline transformation: centre within the car-race, keep a fuel term.

    Kept because it is the estimator the corrected one is compared against.
    Note that centring within a *stint* would be worse still: inside one stint
    tyre age and laps completed are the identical vector, so the slope would
    be (degradation - fuel) rather than degradation.
    """
    out = d.copy()
    g = out.groupby("car_id")
    out["lap_time_dev"] = out["lap_time_s"] - g["lap_time_s"].transform("mean")
    out["tyre_age_dev"] = out["tyre_age"] - g["tyre_age"].transform("mean")
    fuel = pd.to_numeric(out["fuel_laps_burned"], errors="coerce")
    out["fuel_dev"] = fuel - out.groupby("car_id")["fuel_laps_burned"].transform("mean")
    return out.dropna(subset=["lap_time_dev", "tyre_age_dev", "fuel_dev"])


def _twoway_demean(d: pd.DataFrame, n_iter: int = DEMEAN_ITERS) -> pd.DataFrame:
    """Remove car-race and race-moment means by alternating projections.

    Two-way fixed effects with no dummy matrix: alternately subtracting the
    group means converges to the same within transformation, and costs two
    groupbys per iteration instead of tens of thousands of columns.
    """
    out = d.copy()
    out["lap_time_dev"] = out["lap_time_s"].astype(float)
    out["tyre_age_dev"] = out["tyre_age"].astype(float)
    for _ in range(n_iter):
        for key in ("car_id", "lap_bucket"):
            g = out.groupby(key, sort=False)
            out["lap_time_dev"] -= g["lap_time_dev"].transform("mean")
            out["tyre_age_dev"] -= g["tyre_age_dev"].transform("mean")
    return out.dropna(subset=["lap_time_dev", "tyre_age_dev"])


def _wls_multi(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Weighted least squares through the origin on centred data.

    Returns ``(coefficients, standard_errors, n, vif_of_first_column)``.
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
    sigma2 = float(resid @ resid) / max(n - k, 1)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))

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


def _cluster_se_first(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, beta: np.ndarray, groups: np.ndarray
) -> float:
    """Standard error of the first coefficient, clustered by ``groups``.

    Laps from the same race share weather, track state and the same field, so
    they are not independent draws. The ordinary SE treats 2,000 laps from
    eight races as 2,000 observations and comes out several times too small;
    this sandwich estimator treats them as eight.
    """
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(x).all(axis=1)
    x, y, w, groups = x[ok], y[ok], w[ok], groups[ok]
    if not np.isfinite(beta).all():
        return float("nan")
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return float("nan")
    try:
        bread = np.linalg.inv((x * w[:, None]).T @ x)
    except np.linalg.LinAlgError:
        return float("nan")
    score = x * (w * (y - x @ beta))[:, None]
    by_group = pd.DataFrame(score).groupby(groups).sum().to_numpy()
    meat = by_group.T @ by_group
    v = bread @ meat @ bread * n_groups / (n_groups - 1)
    return float(np.sqrt(max(v[0, 0], 0.0)))


def _slopes_by_cell(
    d: pd.DataFrame,
    weights: pd.Series | None,
    method: str,
    with_fuel: bool,
) -> pd.DataFrame:
    """Degradation slope per circuit x compound rank."""
    rows = []
    for (circuit, rank), grp in d.groupby(["circuit", "compound_rank_label"]):
        if len(grp) < MIN_LAPS_PER_CELL or grp["stint_id"].nunique() < MIN_STINTS_PER_CELL:
            continue
        cols = [grp["tyre_age_dev"].to_numpy(float)]
        if with_fuel:
            cols.append(grp["fuel_dev"].to_numpy(float))
        x = np.column_stack(cols)
        y = grp["lap_time_dev"].to_numpy(float)
        w = np.ones(len(grp)) if weights is None else weights.loc[grp.index].to_numpy(float)
        beta, se, n, vif = _wls_multi(x, y, w)
        row = {
            "circuit": circuit,
            "compound_rank_label": rank,
            "method": method,
            "slope_s_per_lap": float(beta[0]),
            "slope_se": float(se[0]),
            "slope_se_race_clustered": _cluster_se_first(x, y, w, beta, grp["race_id"].to_numpy()),
            "n_races": int(grp["race_id"].nunique()),
            "tyre_age_vif": vif,
            "n_laps": int(n),
            "n_stints": int(grp["stint_id"].nunique()),
        }
        if with_fuel:
            row["fuel_s_per_lap"] = float(beta[1])
        if weights is not None:
            row["mean_weight"] = float(w.mean())
            row["max_weight"] = float(w.max())
        rows.append(row)
    return pd.DataFrame(rows)


def naive_fit(pace: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Estimator 1: car-race centring with a fuel control. The biased baseline."""
    d = _within_car_centred(_stint_frame(pace, stints))
    return _slopes_by_cell(d, None, "naive", with_fuel=True)


def twoway_fit(pace: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Estimator 2: two-way within transformation. Fixes the conditions confound."""
    d = _twoway_demean(_stint_frame(pace, stints))
    return _slopes_by_cell(d, None, "twoway", with_fuel=False)


def selection_evidence(pace: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Direct test that the pit decision responds to degradation.

    For every stint long enough to measure it, fit the slope over its first
    few laps, then ask whether that early slope predicts how long the stint
    lasted. A negative relationship means stints that started degrading fast
    were ended sooner - the selection this module corrects is real, not
    merely assumed.
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
        rows.append(
            {
                "stint_id": stint_id,
                "early_slope": float(np.polyfit(x, y, 1)[0]),
                "stint_length": len(grp),
            }
        )

    if len(rows) < 100:
        return {"status": "insufficient_stints", "n_stints": len(rows)}

    e = pd.DataFrame(rows)
    lo, hi = e["early_slope"].quantile([0.005, 0.995])
    e = e[e["early_slope"].between(lo, hi)]

    r, p = stats.pearsonr(e["early_slope"], e["stint_length"])
    rho, prho = stats.spearmanr(e["early_slope"], e["stint_length"])
    return {
        "status": "ok",
        "n_stints": len(e),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(prho),
    }


def fit_stint_survival(stints: pd.DataFrame):
    """Accelerated-failure-time model of stint length.

    A stint is an *observed* pit decision only if it ended in a pit stop.
    Stints that ran to the flag, ended in a retirement, or were interrupted by
    a red flag are right-censored: they say the tyre lasted at least that
    long, not that anyone judged it finished.
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
        "stint survival: %d stints, %d uncensored (%.0f%%), concordance %.3f",
        len(model_df),
        int(model_df["observed"].sum()),
        100 * model_df["observed"].mean(),
        aft.concordance_index_,
    )
    return aft, d


def _survival_weights(aft, stint_meta: pd.DataFrame, laps: pd.DataFrame) -> pd.Series:
    """1 / P(stint survives to this tyre age), clipped.

    An old-tyre lap that only one stint in twenty ever reached stands in for
    the nineteen pitted before getting there, so it carries more weight.
    Clipping stops a near-zero survival probability letting one lap decide.
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
            uniq = np.unique(ages)
            sf = aft.predict_survival_function(x, times=uniq)
            lookup = dict(zip(uniq, sf.iloc[:, 0].reindex(uniq).to_numpy(float)))
            w = np.array([1.0 / max(lookup.get(a, 1.0), 1e-3) for a in ages])
        except Exception:
            w = np.ones_like(ages)
        weights.loc[grp.index] = np.clip(w, 1.0, MAX_WEIGHT)
    return weights


def ipcw_fit(pace: pd.DataFrame, stints: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Estimator 3: two-way transformation plus censoring weights."""
    aft, stint_meta = fit_stint_survival(with_compound_rank(stints, pace))
    d = _twoway_demean(_stint_frame(pace, stints))
    if aft is None:
        d["w"] = 1.0
    else:
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
                    drop_first=True,
                    dtype=float,
                ).reset_index(drop=True),
            ],
            axis=1,
        )
        meta["start_progress"] = meta["start_progress"].fillna(0.5)
        d["w"] = _survival_weights(aft, meta, d)

    return _slopes_by_cell(d, d["w"], "ipcw", with_fuel=False), aft


def shrink_slopes(
    table: pd.DataFrame,
    slope_col: str,
    se_col: str,
    prior: str = "rank",
) -> pd.DataFrame:
    """Empirical-Bayes partial pooling of cell slopes.

    A cell estimated from a few races can land at 0.0004 s/lap - a tyre that
    never wears - purely by noise, and an optimiser will find every such cell
    and put every car on that compound. Each cell is pulled towards a prior in
    proportion to how noisy it is: B = tau^2 / (tau^2 + se^2), with tau^2, the
    real between-cell variance, estimated by method of moments.

    ``prior="rank"``    the mean slope of that compound rank across circuits.
    ``prior="circuit"`` the same circuit's other compounds (leave-one-out),
                        plus the average gap between this rank and theirs -
                        so Bahrain stays a high-wear circuit.
    """
    t = table.copy()
    b = t[slope_col].astype(float)
    se2 = t[se_col].astype(float) ** 2
    ok = b.notna() & se2.notna()

    if prior == "rank":
        prec = (1.0 / se2).where(ok)
        m = (b * prec).groupby(t["compound_rank_label"]).transform("sum") / prec.groupby(
            t["compound_rank_label"]
        ).transform("sum")
    elif prior == "circuit":
        rank_mean = b.where(ok).groupby(t["compound_rank_label"]).transform("mean")
        resid = (b - rank_mean).where(ok)
        s = resid.groupby(t["circuit"]).transform("sum")
        c = resid.notna().groupby(t["circuit"]).transform("sum")
        loo = (s - resid.fillna(0.0)) / (c - resid.notna().astype(int))
        m = rank_mean + loo.where(np.isfinite(loo), 0.0)
    else:
        raise ValueError(f"unknown prior {prior!r}")

    tau2 = max(float(((b - m) ** 2)[ok].mean() - se2[ok].mean()), 1e-8)
    shrink = tau2 / (tau2 + se2)
    t[f"{slope_col}_prior"] = m
    t[f"{slope_col}_shrinkage"] = 1.0 - shrink
    t[f"{slope_col}_shrunk"] = (m + shrink * (b - m)).where(ok, b)
    t.attrs["tau2"] = tau2
    return t


def ordering_check(table: pd.DataFrame, slope_col: str) -> dict:
    """Does the estimator respect the physics it never saw?

    Softer compounds degrade faster. Nothing in any of these estimators is
    told that, so whether it comes out is an independent check on the
    identification rather than a fitted result.
    """
    p = (
        table.pivot_table(index="circuit", columns="compound_rank_label", values=slope_col)
        .reindex(columns=["SOFTEST", "MIDDLE", "HARDEST"])
        .dropna()
    )
    if p.empty:
        return {"status": "no_complete_circuits"}
    return {
        "n_circuits": len(p),
        "n_softest_faster_than_hardest": int((p["SOFTEST"] > p["HARDEST"]).sum()),
        "share_correct_ordering": float((p["SOFTEST"] > p["HARDEST"]).mean()),
        "n_monotone": int(((p["SOFTEST"] > p["MIDDLE"]) & (p["MIDDLE"] > p["HARDEST"])).sum()),
        "median_softest": float(p["SOFTEST"].median()),
        "median_middle": float(p["MIDDLE"].median()),
        "median_hardest": float(p["HARDEST"].median()),
        "n_negative_cells": int((table[slope_col] < 0).sum()),
        "n_cells": int(table[slope_col].notna().sum()),
    }


def compare(pace: pd.DataFrame, stints: pd.DataFrame, save: bool = True) -> dict:
    """The committed deliverable: all three estimators, side by side.

    Published whichever direction it goes, including if the corrections turn
    out to change little.
    """
    naive = naive_fit(pace, stints)
    twoway = twoway_fit(pace, stints)
    corrected, aft = ipcw_fit(pace, stints)
    evidence = selection_evidence(pace, stints)

    key = ["circuit", "compound_rank_label"]
    merged = (
        naive[key + ["slope_s_per_lap", "slope_se", "n_laps", "n_stints"]]
        .rename(columns={"slope_s_per_lap": "slope_naive", "slope_se": "se_naive"})
        .merge(
            twoway[key + ["slope_s_per_lap", "slope_se"]].rename(
                columns={"slope_s_per_lap": "slope_twoway", "slope_se": "se_twoway"}
            ),
            on=key,
            how="inner",
        )
        .merge(
            corrected[
                key
                + [
                    "slope_s_per_lap",
                    "slope_se",
                    "slope_se_race_clustered",
                    "n_races",
                    "mean_weight",
                ]
            ].rename(
                columns={
                    "slope_s_per_lap": "slope_ipcw",
                    "slope_se": "se_ipcw",
                    "slope_se_race_clustered": "se_ipcw_race_clustered",
                }
            ),
            on=key,
            how="inner",
        )
    )
    merged["delta_twoway_vs_naive"] = merged["slope_twoway"] - merged["slope_naive"]
    merged["delta_ipcw_vs_twoway"] = merged["slope_ipcw"] - merged["slope_twoway"]
    merged["slope_s_per_lap_ipcw"] = merged["slope_ipcw"]

    # The column the simulator consumes: the censoring-corrected slope,
    # partially pooled towards its compound's cross-circuit mean. Chosen by a
    # rule fixed before the result: adopt only if it lowers held-out lap-time
    # error under 5-fold cross-validation by race. It did (MSE -0.0035 s^2,
    # 95% CI -0.0059 to -0.0014; cell slope RMSE 0.0306 -> 0.0283 s/lap).
    pooled = shrink_slopes(merged, "slope_ipcw", "se_ipcw_race_clustered", prior="rank")
    merged["slope_ipcw_shrinkage"] = pooled["slope_ipcw_shrinkage"]
    merged["slope_s_per_lap_sim"] = pooled["slope_ipcw_shrunk"]

    summary = {
        "n_cells": len(merged),
        "naive": ordering_check(naive.rename(columns={"slope_s_per_lap": "s"}), "s"),
        "twoway": ordering_check(twoway.rename(columns={"slope_s_per_lap": "s"}), "s"),
        "ipcw": ordering_check(corrected.rename(columns={"slope_s_per_lap": "s"}), "s"),
        "median_delta_ipcw_vs_twoway": float(merged["delta_ipcw_vs_twoway"].median()),
        "share_ipcw_increased": float((merged["delta_ipcw_vs_twoway"] > 0).mean()),
    }

    if save:
        merged.to_parquet(config.MODELS_OUT / "degradation_naive_vs_ipcw.parquet", index=False)
        pd.DataFrame([{k: str(v) for k, v in summary.items()}]).to_csv(
            config.MODELS_OUT / "degradation_summary.csv", index=False
        )
        pd.DataFrame([evidence]).to_csv(config.MODELS_OUT / "selection_evidence.csv", index=False)

    return {
        "comparison": merged,
        "summary": summary,
        "selection_evidence": evidence,
        "aft": aft,
    }
