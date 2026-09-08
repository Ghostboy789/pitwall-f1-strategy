"""Canonical circuit identity, and per-circuit reference facts.

TRAP T6. FastF1's ``Location`` field is not a stable circuit key. Across
2018-2026 it names the same physical track differently from season to season:

    Monaco / Monte Carlo          -> the same street circuit
    Singapore / Marina Bay        -> the same street circuit
    Miami / Miami Gardens         -> the same circuit
    Yas Island / Yas Marina       -> the same circuit
    Silverstone (GB / UK)         -> same track, country string changed

Grouping by ``Location`` would split single circuits into two thin samples,
which is fatal for a project whose entire output is a *per-circuit* number.

The inverse error is just as bad and less obvious: ``Sakhir`` covers **two
different layouts**. The 2020 Sakhir Grand Prix ran the Bahrain outer loop, a
3.5 km configuration with a lap time around 54 s, against the 5.4 km Grand
Prix layout used every other year. Merging them by location would blend two
unrelated tracks. Canonicalisation therefore keys on ``(location,
event_name)``, not location alone.

Everything here is either verifiable from the source data or a published
reference fact with a citation in SOURCES.md. Pit-lane loss is deliberately
NOT hard-coded: it is estimated empirically per circuit-season in
``pitwall.features``, because published figures vary and the data can answer
it directly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd


def fold(text: str) -> str:
    """ASCII-fold and normalise a name for use as a lookup key.

    The source data contains accented circuit names (Montreal, Sao Paulo,
    Nurburgring, Portimao). Folding avoids embedding accented literals in the
    lookup tables and survives any encoding wobble in the upstream feed.
    """
    if text is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


# Location alias -> canonical circuit key.
# Keys are folded. Every alias observed in 2018-2026 is listed explicitly so a
# new, unrecognised location fails loudly rather than being silently mangled.
_LOCATION_TO_CIRCUIT: dict[str, str] = {
    "austin": "cota",
    "baku": "baku",
    "barcelona": "catalunya",
    "budapest": "hungaroring",
    "hockenheim": "hockenheim",
    "imola": "imola",
    "istanbul": "istanbul",
    "jeddah": "jeddah",
    "las_vegas": "las_vegas",
    "le_castellet": "paul_ricard",
    "lusail": "lusail",
    "marina_bay": "singapore",
    "singapore": "singapore",
    "melbourne": "albert_park",
    "mexico_city": "rodriguez",
    "miami": "miami",
    "miami_gardens": "miami",
    "monaco": "monaco",
    "monte_carlo": "monaco",
    "montreal": "villeneuve",
    "monza": "monza",
    "mugello": "mugello",
    "nurburgring": "nurburgring",
    "portimao": "portimao",
    "sakhir": "bahrain",  # refined below by event name (outer loop)
    "shanghai": "shanghai",
    "silverstone": "silverstone",
    "sochi": "sochi",
    "spa_francorchamps": "spa",
    "spielberg": "red_bull_ring",
    "suzuka": "suzuka",
    "sao_paulo": "interlagos",
    "yas_island": "yas_marina",
    "yas_marina": "yas_marina",
    "zandvoort": "zandvoort",
}

# (folded location, folded event name) -> canonical key, for the cases where
# one location hosts more than one layout.
_LAYOUT_OVERRIDES: dict[tuple[str, str], str] = {
    ("sakhir", "sakhir_grand_prix"): "bahrain_outer",
}


def canonical_circuit(location: str, event_name: str = "") -> str:
    """Return the canonical circuit key for a race.

    Raises ``KeyError`` for an unrecognised location. That is deliberate: a
    silent fallback would let a new circuit quietly corrupt a per-circuit
    estimate, which is exactly the failure mode T6 exists to prevent.
    """
    loc = fold(location)
    evt = fold(event_name)
    override = _LAYOUT_OVERRIDES.get((loc, evt))
    if override:
        return override
    if loc not in _LOCATION_TO_CIRCUIT:
        raise KeyError(
            f"unknown circuit location {location!r} (folded {loc!r}). "
            "Add it to _LOCATION_TO_CIRCUIT rather than letting it fall through."
        )
    return _LOCATION_TO_CIRCUIT[loc]


def add_circuit_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``circuit`` column to a table carrying location and event_name."""
    out = df.copy()
    out["circuit"] = [
        canonical_circuit(loc, evt)
        for loc, evt in zip(out["location"], out.get("event_name", [""] * len(out)))
    ]
    return out


@dataclass(frozen=True)
class CircuitRef:
    """Published reference facts for a circuit.

    ``lap_km`` and ``drs_zones`` are reference values (see SOURCES.md), used
    only as sanity checks and as model covariates. Pit-lane loss is NOT here
    on purpose - it is estimated from the data per circuit-season.
    """

    key: str
    name: str
    lap_km: float
    drs_zones: int
    street: bool


# Reference values. Sources and access dates recorded in SOURCES.md.
# `drs_zones` is the count in the most recent season the circuit was used;
# where it changed across the period the model uses a per-season override
# resolved in pitwall.features, not this table.
CIRCUIT_REF: dict[str, CircuitRef] = {
    "albert_park": CircuitRef("albert_park", "Albert Park", 5.278, 4, True),
    "bahrain": CircuitRef("bahrain", "Bahrain International", 5.412, 3, False),
    "bahrain_outer": CircuitRef("bahrain_outer", "Bahrain Outer Loop", 3.543, 2, False),
    "baku": CircuitRef("baku", "Baku City", 6.003, 2, True),
    "catalunya": CircuitRef("catalunya", "Barcelona-Catalunya", 4.657, 2, False),
    "cota": CircuitRef("cota", "Circuit of the Americas", 5.513, 2, False),
    "hockenheim": CircuitRef("hockenheim", "Hockenheimring", 4.574, 3, False),
    "hungaroring": CircuitRef("hungaroring", "Hungaroring", 4.381, 2, False),
    "imola": CircuitRef("imola", "Imola", 4.909, 1, False),
    "interlagos": CircuitRef("interlagos", "Interlagos", 4.309, 2, False),
    "istanbul": CircuitRef("istanbul", "Istanbul Park", 5.338, 2, False),
    "jeddah": CircuitRef("jeddah", "Jeddah Corniche", 6.174, 3, True),
    "las_vegas": CircuitRef("las_vegas", "Las Vegas Strip", 6.201, 2, True),
    "lusail": CircuitRef("lusail", "Lusail International", 5.419, 1, False),
    "miami": CircuitRef("miami", "Miami International", 5.412, 3, True),
    "monaco": CircuitRef("monaco", "Monte Carlo", 3.337, 1, True),
    "monza": CircuitRef("monza", "Monza", 5.793, 2, False),
    "mugello": CircuitRef("mugello", "Mugello", 5.245, 1, False),
    "nurburgring": CircuitRef("nurburgring", "Nurburgring GP", 5.148, 2, False),
    "paul_ricard": CircuitRef("paul_ricard", "Paul Ricard", 5.842, 2, False),
    "portimao": CircuitRef("portimao", "Algarve", 4.653, 2, False),
    "red_bull_ring": CircuitRef("red_bull_ring", "Red Bull Ring", 4.318, 3, False),
    "rodriguez": CircuitRef("rodriguez", "Hermanos Rodriguez", 4.304, 3, False),
    "shanghai": CircuitRef("shanghai", "Shanghai International", 5.451, 2, False),
    "silverstone": CircuitRef("silverstone", "Silverstone", 5.891, 2, False),
    "singapore": CircuitRef("singapore", "Marina Bay", 4.940, 3, True),
    "sochi": CircuitRef("sochi", "Sochi Autodrom", 5.848, 2, True),
    "spa": CircuitRef("spa", "Spa-Francorchamps", 7.004, 2, False),
    "suzuka": CircuitRef("suzuka", "Suzuka", 5.807, 1, False),
    "villeneuve": CircuitRef("villeneuve", "Gilles Villeneuve", 4.361, 2, False),
    "yas_marina": CircuitRef("yas_marina", "Yas Marina", 5.281, 2, False),
    "zandvoort": CircuitRef("zandvoort", "Zandvoort", 4.259, 2, False),
}

# The four circuits the headline narrative is framed around, in order of
# increasing overtaking difficulty. Chosen for contrast, not convenience.
FOCUS = ("monza", "spa", "zandvoort", "monaco")


def known_circuits() -> set[str]:
    """Every canonical key the location map can produce."""
    return set(_LOCATION_TO_CIRCUIT.values()) | set(_LAYOUT_OVERRIDES.values())
