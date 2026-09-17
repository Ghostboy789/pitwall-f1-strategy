"""Team colour themes for the dashboards, from each team's own brand colours.

The timing feed's ``TeamColor`` is a broadcast-graphics tint (Ferrari #ED1131,
Red Bull #4781D7), not the colour a team actually wears, so it is used only to
know which teams are on the latest grid. The colours come from ``BRANDS``: each
value was read from the team's official website (its stylesheet, CSS variables
or theme-colour tag) or its official logo artwork, checked 2026-09-17, and
``tests/test_team_colours.py`` holds every value to its source within a
just-noticeable colour difference (CIEDE2000 ΔE ≤ 2).

A theme wears the brand the way the team does: a ``band`` for headers, a
``stripe`` beneath it, and an ``accent`` for charts. Only the accent is ever
adjusted, and only in lightness, when it would be unreadable on the page it sits
on:

- ``accent`` (lines, bars, fills) clears 3:1 against the pane, the WCAG floor
  for graphical objects;
- ``ink`` (accent-coloured text) clears 4.5:1 against the page ground.

The simulator series is blue. Teams whose accent is blue get a violet simulator
colour so comparison charts never show two near-identical blues.
"""

from __future__ import annotations

import colorsys
import math
import re

import pandas as pd

LIGHT_PANE, LIGHT_GROUND = "#ffffff", "#f4f4ef"
DARK_PANE, DARK_GROUND = "#15181b", "#0e1012"
SIM_LIGHT, SIM_DARK = "#2d5bd2", "#7d9ef6"
ALT_SIM_LIGHT, ALT_SIM_DARK = "#7b3fc4", "#b794f6"
SIM_HUE = 223  # degrees, the hue of the simulator blue

GRAPHIC_CONTRAST = 3.0
TEXT_CONTRAST = 4.5

# band: header colour · stripe: the line under it · accent: charts.
# source: where each value was read (official site CSS, theme-color, or logo artwork).
BRANDS: dict[str, dict[str, str]] = {
    "Alpine": {
        "band": "#2173b8",
        "stripe": "#ff87bc",
        "accent": "#2173b8",
        "source": "Alpine F1 Team logo artwork (#2173B8); BWT pink of the team livery",
    },
    "Aston Martin": {
        "band": "#00665e",
        "stripe": "#cedc00",
        "accent": "#00665e",
        "source": "astonmartinf1.com stylesheet: racing green #00665E, lime #CEDC00",
    },
    "Audi": {
        "band": "#1a1a1a",
        "stripe": "#ff2d00",
        "accent": "#ff2d00",
        "source": "audif1.com CSS variable --color-af1-lava-red #FF2D00, carbon black",
    },
    "Cadillac": {
        "band": "#111111",
        "stripe": "#c8ccd0",
        "accent": "#111111",
        "source": "cadillacf1team.com: black #111111 and white livery; silver stripe",
    },
    "Ferrari": {
        "band": "#da291c",
        "stripe": "#ffcc00",
        "accent": "#da291c",
        "source": "ferrari.com stylesheet: Ferrari red #DA291C (Pantone 485 C); shield yellow",
    },
    "Haas": {
        "band": "#111111",
        "stripe": "#e6002d",
        "accent": "#e6002d",
        "source": "haasf1team.com: red #E6002D on black and white",
    },
    "McLaren": {
        "band": "#1e1e1e",
        "stripe": "#ff8000",
        "accent": "#ff8000",
        "source": "mclaren.com stylesheet: papaya #FF8000; anthracite",
    },
    "Mercedes": {
        "band": "#000000",
        "stripe": "#00a19c",
        "accent": "#00a19c",
        "source": "mercedesamgf1.com: theme-color #000000, PETRONAS teal #00A19C",
    },
    "Racing Bulls": {
        "band": "#1434cb",
        "stripe": "#db0a40",
        "accent": "#1434cb",
        "source": "visacashapprb.com stylesheet: blue #1434CB, red #DB0A40",
    },
    "Red Bull Racing": {
        "band": "#001e3c",
        "stripe": "#db0a40",
        "accent": "#001e3c",
        "source": "redbullracing.com: theme-color navy #001E3C, red #DB0A40, yellow #FCD700",
    },
    "Williams": {
        "band": "#0230be",
        "stripe": "#b2c7ff",
        "accent": "#0056ff",
        "source": "williamsf1.com CSS variables: brand primary #0230BE, highlight #0056FF, #B2C7FF",
    },
}


def _rgb(hex_: str) -> tuple[float, float, float]:
    h = hex_.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02x}" for c in rgb)


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_: str) -> float:
    """WCAG relative luminance."""
    r, g, b = (_lin(c) for c in _rgb(hex_))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _lab(hex_: str) -> tuple[float, float, float]:
    r, g, b = (_lin(c) for c in _rgb(hex_))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    """CIEDE2000 colour difference. About 2 is just noticeable; above 10 reads as different."""
    l1, a1, b1 = _lab(a)
    l2, a2, b2 = _lab(b)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cm = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cm**7 / (cm**7 + 25**7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dlp, dcp = l2 - l1, c2p - c1p
    dhp = 0.0 if c1p * c2p == 0 else (h2p - h1p + 180) % 360 - 180
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp / 2))
    lpm, cpm = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(hpm - 30))
        + 0.24 * math.cos(math.radians(2 * hpm))
        + 0.32 * math.cos(math.radians(3 * hpm + 6))
        - 0.20 * math.cos(math.radians(4 * hpm - 63))
    )
    sl = 1 + 0.015 * (lpm - 50) ** 2 / math.sqrt(20 + (lpm - 50) ** 2)
    sc, sh = 1 + 0.045 * cpm, 1 + 0.015 * cpm * t
    rt = (
        -2
        * math.sqrt(cpm**7 / (cpm**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((hpm - 275) / 25) ** 2))))
    )
    return math.sqrt(
        (dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2 + rt * (dcp / sc) * (dHp / sh)
    )


def meet_contrast(hex_: str, against: str, target: float) -> str:
    """Shift lightness, keeping hue and saturation, until ``target`` is met.

    Darkens against a light background and lightens against a dark one. Returns
    the colour unchanged if it already passes.
    """
    if contrast(hex_, against) >= target:
        return hex_
    h, lum, s = colorsys.rgb_to_hls(*_rgb(hex_))
    darken = luminance(against) > 0.5
    candidate = hex_
    for _ in range(200):
        lum = lum - 0.005 if darken else lum + 0.005
        candidate = _hex(colorsys.hls_to_rgb(h, max(0.0, min(1.0, lum)), s))
        if contrast(candidate, against) >= target or lum <= 0 or lum >= 1:
            break
    return candidate


def _hue(hex_: str) -> float:
    return colorsys.rgb_to_hls(*_rgb(hex_))[0] * 360


def _near_sim_blue(hex_: str) -> bool:
    h, _, s = colorsys.rgb_to_hls(*_rgb(hex_))
    gap = abs(h * 360 - SIM_HUE)
    return s > 0.25 and min(gap, 360 - gap) < 35


def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")


def display_name(team: str) -> str:
    return team.replace(" F1 Team", "")


def team_themes(results: pd.DataFrame) -> pd.DataFrame:
    """One row per team on the latest season's grid, with its brand and mode-safe colours."""
    latest = results[results["year"] == results["year"].max()]
    teams = sorted({display_name(t) for t in latest["TeamName"].dropna()})
    missing = [t for t in teams if t not in BRANDS]
    if missing:
        raise KeyError(f"no verified brand colours for: {missing}")
    rows = []
    for team in teams:
        b = BRANDS[team]
        accent = b["accent"]
        near_blue = _near_sim_blue(accent)
        rows.append(
            {
                "key": slug(team),
                "team": team,
                "season": int(latest["year"].iloc[0]),
                "band": b["band"],
                "stripe": b["stripe"],
                "accent": accent,
                "light_accent": meet_contrast(accent, LIGHT_PANE, GRAPHIC_CONTRAST),
                "light_ink": meet_contrast(accent, LIGHT_GROUND, TEXT_CONTRAST),
                "dark_accent": meet_contrast(accent, DARK_PANE, GRAPHIC_CONTRAST),
                "dark_ink": meet_contrast(accent, DARK_GROUND, TEXT_CONTRAST),
                "sim_light": ALT_SIM_LIGHT if near_blue else SIM_LIGHT,
                "sim_dark": ALT_SIM_DARK if near_blue else SIM_DARK,
                "source": b["source"],
            }
        )
    return pd.DataFrame(rows)


def rgba(hex_: str, alpha: float) -> str:
    r, g, b = (round(c * 255) for c in _rgb(hex_))
    return f"rgba({r}, {g}, {b}, {alpha})"
