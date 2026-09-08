"""Project-wide configuration: paths, scope, eras, and compound handling.

Everything that another module might otherwise hard-code lives here, so the
scope of a run is described in exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_OUT = ROOT / "models_out"

# FastF1's cache is large (GBs) and regenerable, so it lives outside the repo.
# Override with PITWALL_CACHE if you want it somewhere else.
CACHE = Path(
    os.environ.get(
        "PITWALL_CACHE",
        Path(os.environ.get("LOCALAPPDATA", ROOT)) / "pitwall_fastf1_cache",
    )
)

for _d in (RAW, INTERIM, PROCESSED, REPORTS, FIGURES, MODELS_OUT, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

# FastF1's timing archive begins in 2018. 2026 is the current season.
SEASONS = tuple(range(2018, 2027))

# The four circuits the headline finding is framed around. They are a
# deliberate spread along one axis: how much track position is worth relative
# to tyre life. The model covers the whole calendar; these are the ones the
# narrative features.
FOCUS_CIRCUITS = ("Monaco", "Zandvoort", "Spa-Francorchamps", "Monza")


# --------------------------------------------------------------------------
# Regulation eras
# --------------------------------------------------------------------------
# Aerodynamic regulation changes alter dirty-air behaviour and therefore
# overtaking, so seasons are not interchangeable. Era enters the models as a
# factor; it is never pooled away silently.

ERAS: dict[int, str] = {
    2018: "2017-2018_wide_aero",
    2019: "2019-2021_simplified_front_wing",
    2020: "2019-2021_simplified_front_wing",
    2021: "2019-2021_simplified_front_wing",
    2022: "2022-2025_ground_effect",
    2023: "2022-2025_ground_effect",
    2024: "2022-2025_ground_effect",
    2025: "2022-2025_ground_effect",
    2026: "2026_new_regs",
}

# 2021 cut the floor area and cost ~1s/lap, but kept the 2019 aero philosophy;
# it is grouped with 2019-2020 and flagged separately for sensitivity checks.
FLOOR_CUT_SEASONS = (2021,)


# --------------------------------------------------------------------------
# Tyre compounds
# --------------------------------------------------------------------------
# TRAP #1. Compound labels are RELATIVE, not absolute.
#
# From 2019 onwards Pirelli brings three compounds from the C1-C5 (later C0-C6)
# range to each event and relabels them SOFT / MEDIUM / HARD locally. A "HARD"
# at Monza and a "HARD" at Monaco are different rubber. Pooling by label across
# events is a silent error.
#
# 2018 is worse than that: it used a seven-name absolute-ish scheme
# (SUPERHARD ... HYPERSOFT) which does not map onto the later relative labels
# at all. It is handled by mapping both schemes onto a common hardness index.
#
# The per-event C-allocation is the only thing that makes cross-event pooling
# defensible; see pitwall/compounds.py for how it is resolved and what happens
# when it is unknown.

RELATIVE_LABELS = ("SOFT", "MEDIUM", "HARD")
WET_LABELS = ("INTERMEDIATE", "WET")

# 2018 names, ordered soft -> hard. Pirelli's 2018 range was a superset of the
# later C-range; these indices are ordinal, not physical compound numbers.
LEGACY_2018_ORDER = (
    "HYPERSOFT",
    "ULTRASOFT",
    "SUPERSOFT",
    "SOFT",
    "MEDIUM",
    "HARD",
    "SUPERHARD",
)

# Values FastF1 emits that are not real compounds and must be dropped.
INVALID_COMPOUNDS = ("nan", "NAN", "UNKNOWN", "TEST_UNKNOWN", "", None)


# --------------------------------------------------------------------------
# Data-quality thresholds
# --------------------------------------------------------------------------
# Applied mechanically in pitwall/quality.py. Chosen before looking at results.

MIN_LAP_TIME_S = 50.0  # no F1 circuit has a sub-50s racing lap
MAX_LAP_TIME_S = 400.0  # beyond this it is a red-flag crawl, not a racing lap
MIN_GREEN_LAPS_FOR_RACE = 20  # see EXCLUSION RULE in VALIDATION_PLAN.md
MIN_STINT_LAPS_FOR_DEG = 4  # below this a degradation slope is not identified

# Race classified as strategically destroyed if a red flag falls this early,
# because the free tyre change removes the strategic premise entirely.
EARLY_RED_FLAG_RACE_FRACTION = 0.25


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED = 20260908
N_SIM_DEFAULT = 2000
