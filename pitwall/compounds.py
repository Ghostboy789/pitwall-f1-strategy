"""Compound handling. This is where TRAP T1 is resolved.

The problem
-----------
FastF1 gives compound *labels*, not compounds. From 2019 Pirelli brings three
tyres from the C1-C5 (later C0-C6) range to each event and relabels them
SOFT / MEDIUM / HARD locally. A HARD at Monza is different rubber from a HARD
at Monaco. Pooling degradation by label across events is a silent error, and
it is the single easiest way to get every number in this project wrong.

2018 is a second, different problem: it used a seven-name scheme
(HYPERSOFT ... SUPERHARD) whose names do not map onto the later relative
labels at all. Confirmed from the data - 2018 Monaco ran
HYPERSOFT/ULTRASOFT/SUPERSOFT, 2018 Spa ran SUPERSOFT/SOFT/MEDIUM.

What was tried and rejected
---------------------------
Recovering the underlying C-number per event. No authoritative structured
dataset of per-event allocations exists publicly (see SOURCES.md); the one
project attempting it says so explicitly and has compiled only a handful of
races by hand. Adopting a partial table of unverified provenance would import
someone else's uncertainty while looking authoritative.

What is done instead
--------------------
Convert each label to its **within-event relative hardness rank**: 0 for the
softest compound available at that event, 1 for the middle, 2 for the hardest.

This is not a workaround, it is what the label actually means. Pirelli's
labels are *defined* relatively, so the rank recovers the label's true
semantics and is directly comparable across events - including across the
2018/2019 naming change, which the rank absorbs without special-casing.

What is deliberately not claimed: the rank does NOT make the physical rubber
comparable. A rank-0 tyre at Monaco is softer in absolute terms than a rank-0
at Monza. Absolute hardness is therefore modelled as a *circuit-level* effect
that the rank sits inside, never as a pooled constant. This limitation is
stated in the methodology report rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordinal hardness, soft -> hard, spanning both naming schemes. Used only to
# ORDER the compounds present at one event; the absolute positions carry no
# cross-event meaning and are never used as if they did.
_HARDNESS_ORDER: dict[str, int] = {
    # 2018 seven-name scheme
    "HYPERSOFT": 0,
    "ULTRASOFT": 1,
    "SUPERSOFT": 2,
    # shared names -- note SOFT/MEDIUM/HARD mean different things pre/post 2019,
    # which is exactly why only the within-event ordering is ever used.
    "SOFT": 3,
    "MEDIUM": 4,
    "HARD": 5,
    "SUPERHARD": 6,
}

WET = ("INTERMEDIATE", "WET")


def hardness_order(compound: str) -> float:
    """Ordinal position of a compound label, soft to hard. NaN if unknown."""
    return _HARDNESS_ORDER.get(str(compound).upper(), np.nan)


def add_relative_hardness(laps: pd.DataFrame) -> pd.DataFrame:
    """Add the within-event relative hardness rank of each lap's compound.

    ``compound_rank``      0 = softest available at that event, 1, 2, ...
    ``compound_rank_label`` human-readable: SOFTEST / MIDDLE / HARDEST
    ``n_compounds_event``  how many dry compounds actually appeared that race

    Ranking is done per race, over the dry compounds that were actually used.
    A race where only two dry compounds ever appeared gets ranks 0 and 1 - the
    rank is relative to what was *run*, which is what the data can support.
    """
    out = laps.copy()
    comp = out["Compound"].astype(str).str.upper()
    out["Compound"] = comp
    out["is_wet_tyre"] = comp.isin(WET)
    out["hardness_ord"] = comp.map(_HARDNESS_ORDER).astype("float")

    dry = out.loc[~out["is_wet_tyre"] & out["hardness_ord"].notna()]

    ranks: dict[tuple[str, str], int] = {}
    n_comp: dict[str, int] = {}
    for race_id, grp in dry.groupby("race_id", sort=False):
        present = (
            grp[["Compound", "hardness_ord"]]
            .drop_duplicates()
            .sort_values("hardness_ord")["Compound"]
            .tolist()
        )
        n_comp[race_id] = len(present)
        for i, c in enumerate(present):
            ranks[(race_id, c)] = i

    key = list(zip(out["race_id"], out["Compound"]))
    out["compound_rank"] = pd.Series([ranks.get(k, np.nan) for k in key], index=out.index)
    out["n_compounds_event"] = out["race_id"].map(n_comp).astype("float")

    label = {0: "SOFTEST", 1: "MIDDLE", 2: "HARDEST"}
    out["compound_rank_label"] = out["compound_rank"].map(label)

    # An event-specific compound identity, for models that want the compound
    # to be fully local rather than pooled by rank.
    out["compound_event_id"] = out["race_id"] + ":" + out["Compound"]

    return out


def coverage_report(laps: pd.DataFrame) -> pd.DataFrame:
    """How many dry compounds each race actually ran, by season.

    Published in the methodology report: races where only one dry compound
    appeared carry no within-event contrast and cannot inform the compound
    effect at all.
    """
    per_race = (
        laps.loc[~laps["is_wet_tyre"]]
        .groupby(["year", "race_id"], as_index=False)["Compound"]
        .nunique()
        .rename(columns={"Compound": "n_dry_compounds"})
    )
    return (
        per_race.groupby(["year", "n_dry_compounds"])
        .size()
        .rename("n_races")
        .reset_index()
        .sort_values(["year", "n_dry_compounds"])
    )


def naming_scheme(year: int) -> str:
    """Which Pirelli naming scheme a season used."""
    return "2018_seven_name" if year <= 2018 else "relative_smh"
