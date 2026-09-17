"""Team themes: true to each team's brand, readable in both modes, and distinct."""

from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest

from pitwall import config
from pitwall import team_colours as tc

# Values read from the official sources named in tc.BRANDS, checked 2026-09-17.
SOURCE_VALUES = {
    "Ferrari": "#da291c",
    "Red Bull Racing": "#001e3c",
    "Mercedes": "#00a19c",
    "McLaren": "#ff8000",
    "Aston Martin": "#00665e",
    "Alpine": "#2173b8",
    "Williams": "#0230be",
    "Racing Bulls": "#1434cb",
    "Haas": "#e6002d",
    "Audi": "#ff2d00",
    "Cadillac": "#111111",
}


def _results(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"year": 2026, "TeamName": teams, "TeamColor": "FFFFFF"})


def test_delta_e_behaves_like_ciede2000() -> None:
    assert tc.delta_e("#ffffff", "#ffffff") == pytest.approx(0.0)
    assert tc.delta_e("#000000", "#ffffff") == pytest.approx(100.0, abs=0.01)
    assert 1 < tc.delta_e("#da291c", "#dc0000") < 10


def test_contrast_matches_wcag_reference_values() -> None:
    assert tc.contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert tc.contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


def test_brand_colours_match_their_sources() -> None:
    for team, value in SOURCE_VALUES.items():
        brand = tc.BRANDS[team]
        assert min(tc.delta_e(brand[k], value) for k in ("band", "stripe", "accent")) <= 2, team
        assert brand["source"]


def test_no_two_themes_look_alike() -> None:
    for a, b in combinations(tc.BRANDS, 2):
        parts = [tc.delta_e(tc.BRANDS[a][k], tc.BRANDS[b][k]) for k in ("band", "stripe", "accent")]
        assert max(parts) >= 10, f"{a} and {b} are too similar"


def test_every_theme_clears_its_contrast_floor() -> None:
    themes = tc.team_themes(_results(list(tc.BRANDS)))
    for t in themes.itertuples():
        assert tc.contrast(t.light_accent, tc.LIGHT_PANE) >= tc.GRAPHIC_CONTRAST
        assert tc.contrast(t.light_ink, tc.LIGHT_GROUND) >= tc.TEXT_CONTRAST
        assert tc.contrast(t.dark_accent, tc.DARK_PANE) >= tc.GRAPHIC_CONTRAST
        assert tc.contrast(t.dark_ink, tc.DARK_GROUND) >= tc.TEXT_CONTRAST


def test_accent_is_only_adjusted_when_it_must_be() -> None:
    themes = tc.team_themes(_results(list(tc.BRANDS))).set_index("team")
    assert themes.loc["Ferrari", "light_accent"] == tc.BRANDS["Ferrari"]["accent"]
    assert themes.loc["Red Bull Racing", "light_accent"] == "#001e3c"
    # Navy is invisible on a dark page, so dark mode lightens it, keeping its hue.
    rb_dark = themes.loc["Red Bull Racing", "dark_accent"]
    assert rb_dark != "#001e3c"
    assert abs(tc._hue(rb_dark) - tc._hue("#001e3c")) < 6


def test_a_team_without_verified_colours_is_refused() -> None:
    with pytest.raises(KeyError):
        tc.team_themes(_results(["Ferrari", "Some New Team"]))


def test_committed_themes_cover_the_latest_grid() -> None:
    path = config.MODELS_OUT / "team_themes.csv"
    if not path.exists():
        pytest.skip("team themes not exported")
    themes = pd.read_csv(path)
    assert len(themes) >= 10
    assert themes["key"].is_unique
    assert set(themes["team"]) <= set(tc.BRANDS)
