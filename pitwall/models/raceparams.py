"""Phase 4 - per-circuit race parameters the simulator needs.

Two quantities, both estimated from data and both partially pooled across
circuits with the same empirical-Bayes shrinkage used elsewhere in the
project:

* **Pit-lane loss** - the time a stop actually costs. Published figures vary
  between sources and change whenever a pit lane is resurfaced, a speed limit
  moves, or an entry is redrawn, so this is measured from the in-lap and
  out-lap penalty rather than looked up. The published numbers are used only
  as a sanity check on the result.

* **Caution hazard** - the per-lap probability that a safety car, virtual
  safety car or red flag begins. This is sparse: some circuits have a handful
  of races and a couple of cautions between them, so an unpooled rate would
  be confidently wrong. Shrinkage is doing real work here.

A deliberate asymmetry, carried over from VALIDATION_PLAN.md: races excluded
from strategy modelling by rules E3 and E5 are still used for the caution
hazard. Dropping the chaotic races from a model *of* chaos would bias it
badly. The counts behind each estimate are reported so the reader can see
which circuits are thin.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pitwall import config
from pitwall.models.pace import pool_random_effects

log = logging.getLogger("pitwall.models.raceparams")

SC_CODES = ("4", "6")   # safety car, virtual safety car
RED_CODE = "5"

MIN_STOPS_FOR_CIRCUIT = 8


def estimate_pit_loss(laps: pd.DataFrame) -> pd.DataFrame:
    """Time lost to a pit stop, per circuit, measured from in/out laps.

    For each stop, the loss is how much slower the in-lap and out-lap were
    than that driver's own normal green-flag lap in the same race:

        loss = (in-lap - reference) + (out-lap - reference)

    Using the driver's own reference removes car pace, fuel load and the day's
    conditions in one step. Stops taken under a safety car are excluded: the
    field is slow anyway, so the stop is much cheaper and averaging the two
    together would understate the real cost of a green-flag stop, which is the
    number a strategy model needs.
    """
    d = laps.copy()
    d["lap_time_s"] = pd.to_numeric(d["LapTime"], errors="coerce")

    # Dry running only. A green flag is not enough: an out-lap on
    # intermediates is 30-50 s slower than a dry reference lap and none of
    # that is pit-lane time. Left in, wet races put Imola at 45.9 s against a
    # published figure near 28, driven entirely by the wet 2021 and 2022
    # races - the dry 2020 race on its own gives 29.6 s.
    dry = ~d["is_wet_tyre"].fillna(False)

    green_ref = (
        d[d["is_green"] & dry & ~d["is_inlap"] & ~d["is_outlap"]]
        .groupby("car_id")["lap_time_s"]
        .median()
    )
    d["ref_lap_s"] = d["car_id"].map(green_ref)

    inlaps = d[d["is_inlap"] & d["is_green"] & dry][
        ["car_id", "circuit", "year", "race_id", "LapNumber", "lap_time_s", "ref_lap_s"]
    ].rename(columns={"lap_time_s": "inlap_s"})
    outlaps = d[d["is_outlap"] & d["is_green"] & dry][
        ["car_id", "LapNumber", "lap_time_s"]
    ].rename(columns={"lap_time_s": "outlap_s"})
    outlaps["LapNumber"] = outlaps["LapNumber"] - 1  # pair each out-lap with its in-lap

    stops = inlaps.merge(outlaps, on=["car_id", "LapNumber"], how="inner").dropna(
        subset=["inlap_s", "outlap_s", "ref_lap_s"]
    )
    stops["pit_loss_s"] = (
        (stops["inlap_s"] - stops["ref_lap_s"]) + (stops["outlap_s"] - stops["ref_lap_s"])
    )

    # A stop cannot plausibly cost under 8 s or over 60 s under green; outside
    # that band the pairing has gone wrong (a missed lap, a red flag) and the
    # row is noise rather than a stop.
    stops = stops[stops["pit_loss_s"].between(8.0, 60.0)]

    per_circuit = (
        stops.groupby("circuit")
        .agg(
            n_stops=("pit_loss_s", "size"),
            n_races=("race_id", "nunique"),
            pit_loss_s=("pit_loss_s", "median"),
            pit_loss_mean=("pit_loss_s", "mean"),
            pit_loss_sd=("pit_loss_s", "std"),
            q25=("pit_loss_s", lambda s: s.quantile(0.25)),
            q75=("pit_loss_s", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    per_circuit["pit_loss_se"] = per_circuit["pit_loss_sd"] / np.sqrt(per_circuit["n_stops"])

    usable = per_circuit["n_stops"] >= MIN_STOPS_FOR_CIRCUIT
    mu, tau2, shrunk, shrunk_se = pool_random_effects(
        per_circuit.loc[usable, "pit_loss_s"].to_numpy(float),
        per_circuit.loc[usable, "pit_loss_se"].to_numpy(float),
    )
    per_circuit["pit_loss_shrunk"] = per_circuit["pit_loss_s"]
    per_circuit.loc[usable, "pit_loss_shrunk"] = shrunk
    per_circuit.loc[~usable, "pit_loss_shrunk"] = mu
    per_circuit["pit_loss_shrunk_se"] = np.nan
    per_circuit.loc[usable, "pit_loss_shrunk_se"] = shrunk_se

    log.info(
        "pit loss: %d stops across %d circuits, global median %.1fs (between-circuit SD %.1fs)",
        int(per_circuit["n_stops"].sum()), len(per_circuit), mu, float(np.sqrt(tau2)),
    )
    return per_circuit.sort_values("pit_loss_shrunk")


def caution_events(track_status: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """One row per caution deployment, with the lap it began on.

    Track status arrives as a timestamped stream, so each deployment is
    matched to a lap by finding the leader's lap in progress at that moment.
    """
    ts = track_status.copy()
    ts["race_id"] = ts["year"].astype(str) + "_" + ts["round"].astype(str).str.zfill(2)
    ts["Status"] = ts["Status"].astype(str)
    ts["Time"] = pd.to_numeric(ts["Time"], errors="coerce")

    # Leader's lap boundaries, used to place a timestamp on a lap number.
    lead = (
        laps.groupby(["race_id", "LapNumber"])["Time"]
        .min()
        .reset_index()
        .rename(columns={"Time": "lap_end_time"})
        .sort_values(["race_id", "lap_end_time"])
    )

    rows = []
    for race_id, g in ts.groupby("race_id"):
        g = g.sort_values("Time")
        boundaries = lead[lead["race_id"] == race_id]
        if boundaries.empty:
            continue
        prev = "1"
        for _, r in g.iterrows():
            code = r["Status"]
            started = (code in SC_CODES and prev not in SC_CODES) or (
                code == RED_CODE and prev != RED_CODE
            )
            if started and np.isfinite(r["Time"]):
                lap = boundaries.loc[
                    boundaries["lap_end_time"] >= r["Time"], "LapNumber"
                ]
                rows.append(
                    {
                        "race_id": race_id,
                        "kind": "red" if code == RED_CODE else "sc_or_vsc",
                        "time_s": float(r["Time"]),
                        "lap": int(lap.iloc[0]) if len(lap) else np.nan,
                    }
                )
            prev = code
    return pd.DataFrame(rows)


def caution_hazard(laps: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per-lap probability that a caution begins, by circuit.

    The denominator is racing laps at risk - the number of green laps run by
    the leader across all that circuit's races - so a long race does not look
    more dangerous per lap simply for being long.
    """
    race_laps = laps.groupby("race_id")["LapNumber"].max().rename("race_laps")
    race_circuit = laps.groupby("race_id")["circuit"].first().rename("circuit")
    per_race = pd.concat([race_circuit, race_laps], axis=1).reset_index()

    ev = events[events["kind"] == "sc_or_vsc"].groupby("race_id").size().rename("n_cautions")
    per_race["n_cautions"] = per_race["race_id"].map(ev).fillna(0).astype(int)

    # Cautions CLUSTER WITHIN RACES: a wet, chaotic afternoon produces several
    # and a clean one produces none, so laps are not independent trials. A
    # binomial standard error over laps-at-risk treats 300 correlated laps as
    # 300 independent ones and reports a precision the data does not have.
    # The unit of replication is the race, so the spread is measured across
    # races and a circuit seen twice is honestly uncertain.
    out = (
        per_race.groupby("circuit")
        .agg(
            n_races=("race_id", "nunique"),
            laps_at_risk=("race_laps", "sum"),
            mean_race_laps=("race_laps", "mean"),
            n_cautions=("n_cautions", "sum"),
            cautions_per_race=("n_cautions", "mean"),
            cautions_sd=("n_cautions", "std"),
        )
        .reset_index()
    )
    # Between-race SEM, with a Poisson fallback where one race gives no spread.
    poisson_sd = np.sqrt(np.maximum(out["cautions_per_race"], 0.5))
    sd = out["cautions_sd"].fillna(poisson_sd).replace(0, np.nan).fillna(poisson_sd)
    out["cautions_per_race_se"] = sd / np.sqrt(out["n_races"])

    out["hazard_per_lap"] = out["cautions_per_race"] / out["mean_race_laps"]
    out["hazard_se"] = out["cautions_per_race_se"] / out["mean_race_laps"]

    mu, tau2, shrunk, shrunk_se = pool_random_effects(
        out["hazard_per_lap"].to_numpy(float), out["hazard_se"].to_numpy(float)
    )
    out["hazard_shrunk"] = np.clip(shrunk, 0.0, None)
    out["hazard_shrunk_se"] = shrunk_se
    out["expected_cautions_per_race"] = out["hazard_shrunk"] * (
        out["laps_at_risk"] / out["n_races"]
    )

    log.info(
        "caution hazard: %d cautions over %d circuits, global %.4f/lap (between-circuit SD %.4f)",
        int(out["n_cautions"].sum()), len(out), mu, float(np.sqrt(tau2)),
    )
    return out.sort_values("hazard_shrunk", ascending=False)


def run(laps: pd.DataFrame, track_status: pd.DataFrame, save: bool = True) -> dict:
    """Estimate both parameter sets and optionally persist them."""
    pit = estimate_pit_loss(laps)
    events = caution_events(track_status, laps)
    hazard = caution_hazard(laps, events)

    if save:
        pit.to_parquet(config.MODELS_OUT / "pit_loss.parquet", index=False)
        hazard.to_parquet(config.MODELS_OUT / "caution_hazard.parquet", index=False)
        events.to_parquet(config.MODELS_OUT / "caution_events.parquet", index=False)

    return {"pit_loss": pit, "hazard": hazard, "events": events}
