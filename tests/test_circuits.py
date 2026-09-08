"""Circuit identity is trap T6. These tests are the guard on it."""

from __future__ import annotations

import pytest

from pitwall.circuits import CIRCUIT_REF, canonical_circuit, fold, known_circuits


def test_aliases_collapse_to_one_circuit():
    """The same track under different Location strings must be one key."""
    assert canonical_circuit("Monaco") == canonical_circuit("Monte Carlo")
    assert canonical_circuit("Singapore") == canonical_circuit("Marina Bay")
    assert canonical_circuit("Miami") == canonical_circuit("Miami Gardens")
    assert canonical_circuit("Yas Island") == canonical_circuit("Yas Marina")


def test_sakhir_layouts_stay_separate():
    """One Location, two layouts. Merging them would blend different tracks.

    The 2020 Sakhir Grand Prix ran Bahrain's 3.5 km outer loop; every other
    Bahrain race used the 5.4 km Grand Prix circuit.
    """
    gp = canonical_circuit("Sakhir", "Bahrain Grand Prix")
    outer = canonical_circuit("Sakhir", "Sakhir Grand Prix")
    assert gp != outer
    assert CIRCUIT_REF[gp].lap_km > CIRCUIT_REF[outer].lap_km


def test_accented_names_fold():
    """Source data carries accents; the key must not depend on encoding."""
    assert canonical_circuit("Montréal") == canonical_circuit("Montreal")
    assert canonical_circuit("São Paulo") == canonical_circuit("Sao Paulo")
    assert fold("Nürburgring") == "nurburgring"


def test_unknown_location_raises():
    """A new circuit must fail loudly, not fall through to a wrong key."""
    with pytest.raises(KeyError):
        canonical_circuit("Nowhere Special")


def test_every_known_circuit_has_reference_data():
    assert known_circuits() == set(CIRCUIT_REF)


def test_reference_lap_distances_are_plausible():
    for key, ref in CIRCUIT_REF.items():
        assert 3.0 < ref.lap_km < 8.0, f"{key} lap distance implausible"
        assert 0 <= ref.drs_zones <= 4, f"{key} DRS zone count implausible"
