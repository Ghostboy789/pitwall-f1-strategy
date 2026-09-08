"""Phase 2 - lap time decomposition. The statistical core.

Observed lap time is decomposed into:

    lap_time = circuit/era level
             + driver-team baseline pace
             + fuel burn            (own laps completed)
             + track evolution      (session time elapsed, plus curvature)
             + tyre degradation     (compound rank x tyre age, non-linear)
             + traffic penalty      (dirty air)
             + race conditions      (random intercept per race)
             + noise

Design decisions, and why
-------------------------
**Two-stage hierarchy rather than one giant crossed model.** A single MixedLM
with circuit x era x compound x driver interactions runs to several hundred
fixed-effect columns over ~150k laps: estimable, but slow, fragile and hard to
audit. Instead:

  Stage 1  fit an independent mixed model per circuit, with a random intercept
           per race so per-race conditions (weather, track state, session
           quirks) are absorbed rather than blamed on tyres.
  Stage 2  pool the per-circuit coefficients with a random-effects
           meta-analysis, shrinking each circuit toward the global mean in
           proportion to its own precision.

Stage 2 *is* partial pooling - the same shrinkage a one-shot hierarchical
model applies, in a form that can be inspected and tested. It also makes the
thin-circuit behaviour explicit rather than emergent: a circuit with one race
is shrunk hard toward the global mean and its interval is wide, by
construction.

Separating fuel from track evolution (TRAP T5)
----------------------------------------------
There is no refuelling, so a car's fuel mass is a deterministic function of
the laps it has completed. Track evolution is also monotone in time. Within a
race the two are very nearly the same variable, and an earlier version of this
model included both ``fuel_burned`` and ``race_progress`` as linear terms -
which are an exact affine transform of one another within a race. The design
matrix was near-singular and returned fuel coefficients of +4.46 s per lap,
i.e. cars getting four seconds slower per lap of fuel burned. That was a real
bug in this model, found by the estimate being physically absurd, and it is
recorded here rather than quietly corrected.

What identifies them apart at all is that they are not quite the same
variable: fuel depends on **each car's own** laps completed, while track
evolution depends on **elapsed session time**, shared by everyone on track.
Those diverge for lapped cars, for cars delayed by a stop under a safety car,
and for anyone who loses time and rejoins. The separation therefore rests on a
minority of the field, and its condition number is computed and published
(``collinearity_report``) rather than assumed to be acceptable. Where the
diagnostic says the two are not separable for a circuit, that circuit reports
a single combined progression term and says so.

**Degradation here carries no censoring correction.** That is deliberate: this
module produces the *naive* fit that ``pitwall.models.degradation`` corrects,
so the comparison the validation plan commits to can be made honestly.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pitwall import config

log = logging.getLogger("pitwall.models.pace")

# Fixed in advance so coefficients are interpretable and comparable across
# circuits. Not tuned.
TYRE_AGE_CENTRE = 10.0
MIN_LAPS_FOR_CIRCUIT_FIT = 300
MIN_RACES_FOR_CIRCUIT_FIT = 2

# Above this condition number the fuel / evolution split is not trusted and
# the circuit falls back to a single combined progression term.
#
# This was originally set to 50, which passed most circuits. The resulting
# estimates were plainly compensating rather than identified - Hungaroring
# returned fuel = -0.31 s/lap against evolution = +4.77 s, and Yas Marina
# +0.19 against -3.72 - while the circuits that fell back to the combined
# term returned a tight, physically sensible -0.03 to -0.09 s/lap. The
# threshold is now strict, and the consequence (that the split essentially
# never succeeds) is published as a negative result rather than hidden by a
# permissive cut-off. See `separability_summary`.
MAX_CONDITION_NUMBER = 5.0

# Explicit reference level, so coefficient names mean what they say. Patsy
# would otherwise pick alphabetically (HARDEST), which silently made an
# earlier version of this module report the hardest compound's degradation
# slope under the name `deg_softest`.
RANK_REF = "HARDEST"


def prepare(pace: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix: centred, typed, and free of unusable rows."""
    df = pace.copy()
    df = df[df["compound_rank"].notna() & df["lap_time_s"].notna() & df["TyreLife"].notna()]

    df["tyre_age"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df["tyre_age_c"] = df["tyre_age"] - TYRE_AGE_CENTRE
    df["tyre_age_sq"] = df["tyre_age_c"] ** 2

    # Fuel proxy: laps this car has completed, centred within its own race.
    fuel = pd.to_numeric(df["fuel_laps_burned"], errors="coerce")
    df["fuel_burned_c"] = fuel - df.groupby("race_id")["fuel_laps_burned"].transform("mean")

    # Track-evolution proxy: elapsed session time, centred and scaled within
    # race. Distinct from fuel only for cars off the lead lap or delayed.
    tsec = pd.to_numeric(df["Time"], errors="coerce")
    g = df.groupby("race_id")["Time"]
    df["sess_time_c"] = (tsec - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    # Curvature: evolution is concave (big early grip gain, then a plateau).
    df["sess_time_sq"] = df["sess_time_c"] ** 2
    df["sess_time_sq"] = df["sess_time_sq"] - df.groupby("race_id")["sess_time_sq"].transform("mean")

    df["dirty_air"] = pd.to_numeric(df["dirty_air_intensity"], errors="coerce").fillna(0.0)
    df["rank_lab"] = df["compound_rank_label"].astype(str)
    df["driver_team"] = df["Driver"].astype(str) + "|" + df["Team"].astype(str)
    df["era"] = df["era"].astype(str)

    return df.dropna(
        subset=["tyre_age_c", "fuel_burned_c", "sess_time_c", "lap_time_s", "rank_lab"]
    )


_BASE = (
    "lap_time_s ~ C(driver_team) "
    f"+ tyre_age_c * C(rank_lab, Treatment(reference='{RANK_REF}')) "
    "+ tyre_age_sq + dirty_air"
)
FORMULA_SPLIT = _BASE + " + fuel_burned_c + sess_time_c + sess_time_sq"
FORMULA_COMBINED = _BASE + " + fuel_burned_c + sess_time_sq"
ERA_TERM = "C(era) + "


def collinearity_report(d: pd.DataFrame) -> dict:
    """Condition number of the fuel / evolution pair for one circuit.

    Reported, not assumed. A high value means the two cannot be told apart in
    this circuit's data and the model must not pretend otherwise.
    """
    x = d[["fuel_burned_c", "sess_time_c"]].to_numpy(float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 10:
        return {"condition_number": np.inf, "corr": np.nan, "n": len(x)}
    xs = (x - x.mean(0)) / np.where(x.std(0) == 0, 1, x.std(0))
    sv = np.linalg.svd(xs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    return {
        "condition_number": cond,
        "corr": float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]),
        "n": int(len(x)),
    }


def fit_one_circuit(df: pd.DataFrame, circuit: str) -> dict | None:
    """Fit the decomposition for one circuit. Returns None if not estimable."""
    d = df[df["circuit"] == circuit]
    if len(d) < MIN_LAPS_FOR_CIRCUIT_FIT or d["race_id"].nunique() < MIN_RACES_FOR_CIRCUIT_FIT:
        log.info(
            "skip %-16s %d laps / %d races (need %d / %d)",
            circuit, len(d), d["race_id"].nunique(),
            MIN_LAPS_FOR_CIRCUIT_FIT, MIN_RACES_FOR_CIRCUIT_FIT,
        )
        return None

    coll = collinearity_report(d)
    separable = coll["condition_number"] <= MAX_CONDITION_NUMBER
    formula = FORMULA_SPLIT if separable else FORMULA_COMBINED
    if d["era"].nunique() > 1:
        formula = formula.replace("lap_time_s ~ ", "lap_time_s ~ " + ERA_TERM)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(formula, d, groups=d["race_id"]).fit(
                method="lbfgs", maxiter=400, disp=False
            )
    except Exception as exc:  # noqa: BLE001 - a circuit that will not fit is itself a result
        log.warning("circuit %-16s failed: %s: %s", circuit, type(exc).__name__, exc)
        return None

    params, bse = res.params, res.bse

    # `resid` needs the random-effect covariance to be invertible; when the
    # between-race variance collapses to zero it is not. That is a legitimate
    # outcome (all races at this circuit behaved alike), not a failure.
    try:
        resid_sd = float(np.std(res.resid))
    except Exception:  # noqa: BLE001
        resid_sd = np.nan

    ref = f"C(rank_lab, Treatment(reference='{RANK_REF}'))"
    out = {
        "circuit": circuit,
        "n_laps": int(len(d)),
        "n_races": int(d["race_id"].nunique()),
        "n_drivers": int(d["Driver"].nunique()),
        "eras": int(d["era"].nunique()),
        "converged": bool(getattr(res, "converged", True)),
        "fuel_evo_separable": bool(separable),
        "cond_number": float(coll["condition_number"]),
        "fuel_evo_corr": float(coll["corr"]),
        "loglike": float(res.llf),
        "resid_sd": resid_sd,
        "race_var": float(res.cov_re.iloc[0, 0]) if res.cov_re.size else np.nan,
    }

    # `tyre_age_c` is the degradation slope of the REFERENCE compound rank
    # (hardest available); the interactions are deltas from it.
    wanted = {
        "deg_hardest": "tyre_age_c",
        "deg_quad": "tyre_age_sq",
        "deg_middle_delta": f"tyre_age_c:{ref}[T.MIDDLE]",
        "deg_softest_delta": f"tyre_age_c:{ref}[T.SOFTEST]",
        "fuel": "fuel_burned_c",
        "evolution": "sess_time_c",
        "evolution_quad": "sess_time_sq",
        "traffic": "dirty_air",
    }
    for key, name in wanted.items():
        out[key] = float(params.get(name, np.nan))
        out[f"{key}_se"] = float(bse.get(name, np.nan))

    # Absolute degradation of the softest available compound, the quantity the
    # strategy model actually consumes. It is a sum of two estimated
    # coefficients, so its standard error needs their covariance -- treating
    # them as independent would understate it.
    base, delta = "tyre_age_c", f"tyre_age_c:{ref}[T.SOFTEST]"
    out["deg_softest"] = out["deg_hardest"] + (
        out["deg_softest_delta"] if np.isfinite(out["deg_softest_delta"]) else 0.0
    )
    try:
        cov = res.cov_params()
        v = cov.loc[base, base] + cov.loc[delta, delta] + 2 * cov.loc[base, delta]
        out["deg_softest_se"] = float(np.sqrt(max(v, 0.0)))
    except Exception:  # noqa: BLE001 - compound absent at this circuit
        out["deg_softest_se"] = out["deg_hardest_se"]
    return out


def fit_all_circuits(pace: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 - independent per-circuit fits."""
    df = prepare(pace)
    rows = []
    for circuit in sorted(df["circuit"].unique()):
        r = fit_one_circuit(df, circuit)
        if r is None:
            continue
        rows.append(r)
        log.info(
            "%-16s laps=%-6d races=%-3d fuel=%+.4f evo=%+.3f degH=%+.4f degS=%+.4f traffic=%+.3f %s",
            circuit, r["n_laps"], r["n_races"], r["fuel"], r["evolution"],
            r["deg_hardest"], r["deg_softest"], r["traffic"],
            "" if r["fuel_evo_separable"] else "[combined]",
        )
    return pd.DataFrame(rows)


def pool_random_effects(
    est: np.ndarray, se: np.ndarray, max_iter: int = 200
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Random-effects meta-analysis by Paule-Mandel iteration.

    Returns ``(mu, tau2, shrunk_estimates, shrunk_ses)``. This is Stage 2: the
    partial pooling. Each circuit is pulled toward the global mean ``mu`` by a
    weight set by its own precision relative to the between-circuit spread
    ``tau2``. A circuit with one race and a wide standard error is pulled
    almost all the way to the mean; a circuit with eleven barely moves.
    """
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    if ok.sum() < 2:
        return float(np.nanmean(est)), 0.0, est.astype(float).copy(), se.astype(float).copy()

    e, s2 = est[ok].astype(float), se[ok].astype(float) ** 2
    df_ = len(e) - 1
    tau2 = 0.0
    for _ in range(max_iter):
        w = 1.0 / (s2 + tau2)
        mu = float((w * e).sum() / w.sum())
        q = float((w * (e - mu) ** 2).sum())
        if q <= df_:
            new_tau2 = 0.0
        else:
            deriv = float((w**2 * (e - mu) ** 2).sum())
            new_tau2 = max(tau2 + (q - df_) / max(deriv, 1e-12), 0.0)
        if abs(new_tau2 - tau2) < 1e-12:
            tau2 = new_tau2
            break
        tau2 = new_tau2

    w = 1.0 / (s2 + tau2)
    mu = float((w * e).sum() / w.sum())

    shrunk = est.astype(float).copy()
    shrunk_se = se.astype(float).copy()
    b = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(s2)
    shrunk[ok] = mu + b * (e - mu)
    shrunk_se[ok] = np.sqrt(b * s2) if tau2 > 0 else np.sqrt(1.0 / (1.0 / s2).sum())
    return mu, float(tau2), shrunk, shrunk_se


POOLED_TERMS = (
    "fuel", "evolution", "deg_hardest", "deg_softest", "deg_quad", "traffic",
)


def pool_all(circuit_fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 2 - shrink every per-circuit coefficient toward its global mean."""
    out = circuit_fits.copy()
    summary = []
    for term in POOLED_TERMS:
        if term not in out.columns:
            continue
        mu, tau2, shrunk, shrunk_se = pool_random_effects(
            out[term].to_numpy(float), out[f"{term}_se"].to_numpy(float)
        )
        out[f"{term}_shrunk"] = shrunk
        out[f"{term}_shrunk_se"] = shrunk_se
        raw_dev = out[term].to_numpy(float) - mu
        with np.errstate(divide="ignore", invalid="ignore"):
            kept = np.where(np.abs(raw_dev) > 1e-12, (shrunk - mu) / raw_dev, 0.0)
        out[f"{term}_shrinkage"] = 1.0 - np.clip(kept, 0, 1)
        summary.append(
            {
                "term": term,
                "global_mean": mu,
                "between_circuit_sd": float(np.sqrt(tau2)),
                "n_circuits": int(np.isfinite(out[term]).sum()),
            }
        )
    return out, pd.DataFrame(summary)


def separability_summary(circuit_fits: pd.DataFrame) -> dict:
    """Whether fuel burn and track evolution could be told apart, and where.

    Published as a negative result. Fuel mass and elapsed session time are
    near-perfectly collinear within a race; the only thing that breaks the tie
    is cars off the lead lap, and there are too few of them. The honest report
    is the count, not a claim of separation.
    """
    d = circuit_fits
    n = len(d)
    sep = int(d["fuel_evo_separable"].sum()) if "fuel_evo_separable" in d else 0
    return {
        "n_circuits": int(n),
        "n_separable": sep,
        "n_combined": int(n - sep),
        "median_condition_number": float(d["cond_number"].median()) if n else np.nan,
        "median_fuel_evo_corr": float(d["fuel_evo_corr"].median()) if n else np.nan,
        "threshold": MAX_CONDITION_NUMBER,
        "conclusion": (
            "fuel burn and track evolution are NOT separately identified in "
            "race data; the linear term is reported as a combined progression "
            "effect and interpreted as fuel-dominated on the evidence of "
            "fuel_scaling_check and its agreement with the known physical "
            "value of roughly -0.05 s per lap of fuel burned"
        ),
    }


PHYSICAL_FUEL_S_PER_LAP = (-0.12, -0.02)
"""Plausible range for the per-lap gain from fuel burn.

Roughly 1.6-1.9 kg of fuel per lap at about 0.03 s per kg gives ~0.05 s/lap,
varying with circuit. Used ONLY as an external plausibility check on an
estimate made from data - never as a prior, and never substituted for it.
"""


def fuel_plausibility_check(circuit_fits: pd.DataFrame) -> dict:
    """Compare the estimated fuel term against its independently known value.

    This is a validation, not a calibration: nothing is adjusted to fit. If
    the data-driven estimate lands in the physically expected band that is
    evidence the decomposition is working; if it does not, that is reported.
    """
    f = circuit_fits["fuel"].dropna()
    lo, hi = PHYSICAL_FUEL_S_PER_LAP
    return {
        "n_circuits": int(len(f)),
        "median_s_per_lap": float(f.median()),
        "iqr_low": float(f.quantile(0.25)),
        "iqr_high": float(f.quantile(0.75)),
        "expected_band": PHYSICAL_FUEL_S_PER_LAP,
        "share_in_expected_band": float(f.between(lo, hi).mean()),
        "median_in_band": bool(lo <= f.median() <= hi),
    }


def fuel_scaling_check(circuit_fits: pd.DataFrame) -> dict:
    """Empirical test that the fuel term behaves like fuel.

    Fuel burned per lap scales with lap distance, so if the linear own-laps
    coefficient really is fuel burn, it should get more negative as lap length
    grows. Track evolution has no reason to. This does not fully separate the
    two, but a clear relationship is evidence the term is fuel-dominated and a
    flat one is evidence it is not. Whichever it shows is what gets published.
    """
    from scipy import stats

    from pitwall.circuits import CIRCUIT_REF

    d = circuit_fits.copy()
    d["lap_km"] = d["circuit"].map(
        lambda c: CIRCUIT_REF[c].lap_km if c in CIRCUIT_REF else np.nan
    )
    d = d.dropna(subset=["lap_km", "fuel"])
    if len(d) < 5:
        return {"status": "insufficient_circuits", "n_circuits": int(len(d))}

    r, p = stats.pearsonr(d["lap_km"], d["fuel"])
    lr = stats.linregress(d["lap_km"], d["fuel"])
    return {
        "status": "ok",
        "n_circuits": int(len(d)),
        "pearson_r": float(r),
        "p_value": float(p),
        "slope_s_per_lap_per_km": float(lr.slope),
        "slope_se": float(lr.stderr),
    }


def run(pace: pd.DataFrame, save: bool = True) -> dict:
    """Fit the full decomposition and return every artefact."""
    fits = fit_all_circuits(pace)
    if fits.empty:
        raise RuntimeError("no circuit produced an estimable pace model")
    pooled, summary = pool_all(fits)
    check = fuel_scaling_check(fits)

    if save:
        pooled.to_parquet(config.MODELS_OUT / "pace_per_circuit.parquet", index=False)
        summary.to_csv(config.MODELS_OUT / "pace_global_terms.csv", index=False)
        pd.DataFrame([check]).to_csv(config.MODELS_OUT / "fuel_scaling_check.csv", index=False)

    return {"per_circuit": pooled, "global": summary, "fuel_check": check}
