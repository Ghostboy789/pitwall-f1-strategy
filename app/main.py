"""Pit Wall - FastAPI app serving the dashboard and the live simulator.

Artefacts are loaded once at startup and held in memory: every model is
already fitted by ``python -m pitwall.pipeline``, so a request never refits
anything. The one thing computed per request is a Monte Carlo run, which is
fast enough to be interactive.

Run locally:
    uvicorn app.main:app --reload

Deployment reads ``$PORT`` (see Dockerfile), so the same image runs on Render,
Fly, Railway, Koyeb or Hugging Face Spaces without modification.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from pitwall import config
from pitwall.circuits import CIRCUIT_REF, FOCUS
from pitwall.sim import Car, Strategy, simulate

log = logging.getLogger("pitwall.app")

APP_DIR = Path(__file__).resolve().parent

AUTHOR = {
    "name": "Medhansh Shekhawat",
    "github": "https://github.com/Ghostboy789",
    "repo": "https://github.com/Ghostboy789/pitwall-f1-strategy",
    "linkedin": "https://www.linkedin.com/in/medhansh-shekhawat",
    "email": "medhanshshekhawat@gmail.com",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_state()
    yield


app = FastAPI(title="Pit Wall", docs_url="/api/docs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
# Content hash on asset URLs, so a redeploy is never hidden behind a cached
# stylesheet or script from the previous version.
templates.env.globals["asset_version"] = hashlib.sha256(
    b"".join((APP_DIR / "static" / n).read_bytes() for n in ("pitwall.css", "pitwall.js"))
).hexdigest()[:10]

STATE: dict[str, Any] = {}

# A circuit needs this many races before it is allowed to anchor the headline
# claim. Without it the strapline names whichever circuit happens to sit at the
# bottom of the table, which is usually the one with the least evidence.
HEADLINE_MIN_RACES = 5

GATE_LABELS = {
    "non_empty": "The lap table has rows",
    "lap_number_monotonic": "Lap numbers increase within every car's race",
    "stint_non_decreasing": "Stint numbers never go backwards",
    "tyre_age_increments": "Tyre age advances by one lap within a stint",
    "fresh_tyre_starts_at_age_1": "A new set of tyres starts at age 1",
    "lap_time_in_bounds": "Every lap time is between 50 s and 400 s",
    "circuit_keys_known": "Every circuit maps to a known track layout",
    "no_duplicate_car_laps": "One row per car per lap",
}


def _pretty(circuit: str) -> str:
    ref = CIRCUIT_REF.get(circuit)
    return ref.name if ref else circuit.replace("_", " ").title()


def load_state() -> None:
    """Read every fitted artefact into memory once."""
    from pitwall import pipeline

    try:
        art = pipeline.load_artifacts()
    except FileNotFoundError as exc:
        log.error("artefacts missing: %s. Run `python -m pitwall.pipeline` first.", exc)
        STATE["ready"] = False
        return

    STATE["artifacts"] = art
    STATE["metrics"] = art["metrics"]
    STATE["ready"] = True

    tp = art["trackposition"].copy()
    tp["name"] = tp["circuit"].map(_pretty)
    tp["is_focus"] = tp["circuit"].isin(FOCUS)
    STATE["trackposition"] = tp

    deg = art["degradation_table"].copy()
    deg["name"] = deg["circuit"].map(_pretty)
    STATE["degradation"] = deg

    import json

    ord_path = config.MODELS_OUT / "degradation_ordering.json"
    STATE["deg_ordering"] = json.loads(ord_path.read_text()) if ord_path.exists() else None

    # The headline comparison is derived, never typed: the extremes of the
    # fitted table decide which circuits the strapline names.
    #
    # Restricted to circuits with a real sample. Taking the outright minimum
    # named Hockenheimring, which rests on two races -- a bigger ratio bought
    # from the thinnest evidence on the page. Requiring HEADLINE_MIN_RACES
    # gives a slightly smaller claim standing on much firmer ground, and the
    # pair it picks is one whose intervals actually separate.
    eligible = tp[tp["n_races"] >= HEADLINE_MIN_RACES]
    tp_sorted = (eligible if len(eligible) >= 2 else tp).sort_values("value_s", ascending=False)
    if len(tp_sorted) >= 2:
        top, bottom = tp_sorted.iloc[0], tp_sorted.iloc[-1]
        STATE["headline"] = {
            "top_name": top["name"],
            "top_value": float(top["value_s"]),
            "bottom_name": bottom["name"],
            "bottom_value": float(bottom["value_s"]),
            "ratio": float(top["value_s"] / bottom["value_s"]) if bottom["value_s"] else None,
            "top_races": int(top["n_races"]),
            "bottom_races": int(bottom["n_races"]),
            # Only claim a difference the intervals actually support.
            "distinguishable": bool(
                top["value_lo"] > bottom["value_hi"] or bottom["value_lo"] > top["value_hi"]
            ),
        }
    else:
        STATE["headline"] = None

    rel_path = config.MODELS_OUT / "overtaking_reliability.csv"
    STATE["reliability"] = _records(pd.read_csv(rel_path)) if rel_path.exists() else []

    sel_path = config.MODELS_OUT / "selection_evidence.csv"
    STATE["selection"] = _records(pd.read_csv(sel_path))[0] if sel_path.exists() else None

    # The sanity gate is optional: the dashboard is honest about not having run
    # it rather than showing a blank where a verdict should be.
    gate_path = config.MODELS_OUT / "sanity_gate.json"
    STATE["gate"] = None
    if gate_path.exists():
        import json

        g = json.loads(gate_path.read_text())
        STATE["gate"] = g if g.get("status") == "ok" else None

    # The gate's evidence, as distributions only. Per-team and per-driver
    # numbers stay withheld: the audit failed its own check.
    audit_path = config.MODELS_OUT / "audit.parquet"
    STATE["gate_evidence"] = None
    if STATE["gate"] and audit_path.exists():
        a = pd.read_parquet(audit_path)
        best_stops = a["best_strategy"].astype(str).str.count(r"L\d+")
        extra = (a["actual_n_stops"] - best_stops).clip(lower=0, upper=3)
        edges = list(range(0, 105, 5))
        counts, _ = np.histogram(a["gain_s"].clip(upper=edges[-1] - 1e-9), bins=edges)
        STATE["gate_evidence"] = {
            "bin_edges": edges,
            "counts": counts.tolist(),
            "by_extra_stops": [
                {
                    "extra": int(k),
                    "n": len(g),
                    "mean": float(g.mean()),
                    "median": float(g.median()),
                }
                for k, g in a["gain_s"].groupby(extra)
            ],
        }

    sv_path = config.MODELS_OUT / "strategy_validation.json"
    STATE["strategy_validation"] = json.loads(sv_path.read_text()) if sv_path.exists() else None

    STATE["circuits"] = sorted(set(tp["circuit"]) & set(art["circuit_reference"]["circuit"]))
    STATE["circuit_profiles"] = _circuit_profiles(art, tp, deg)
    STATE["quality_gates"] = _read_csv("data_quality_gates.csv")
    STATE["exclusions"] = _read_csv("race_exclusions.csv")
    STATE["detector"] = _read_csv("detector_sanity.csv")
    fuel = _read_csv("fuel_scaling_check.csv")
    STATE["fuel_check"] = fuel[0] if fuel else None
    log.info(
        "loaded artefacts: %d circuits, reliability=%d bins, gate=%s",
        len(STATE["circuits"]),
        len(STATE["reliability"]),
        "yes" if STATE["gate"] else "no",
    )


def _records(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN as None, so they serialise as valid JSON."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _read_csv(name: str) -> list[dict]:
    path = config.MODELS_OUT / name
    return _records(pd.read_csv(path)) if path.exists() else []


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _circuit_profiles(art: dict, tp: pd.DataFrame, deg: pd.DataFrame) -> list[dict]:
    """Everything the model knows about each circuit, with its uncertainty.

    Assembled once at startup so the circuit dashboard can switch and compare
    without a round trip. Only fitted artefacts are read.
    """
    ref = art["circuit_reference"].set_index("circuit")
    pit = art["pit_loss"].set_index("circuit")
    haz = art["hazard"].set_index("circuit")
    tpi = tp.set_index("circuit")
    detector = pd.DataFrame(_read_csv("detector_sanity.csv"))
    det = detector.set_index("circuit") if len(detector) else pd.DataFrame()
    limits = art.get("stint_limits")

    out = []
    for c in STATE["circuits"]:
        t = tpi.loc[c]
        compounds = []
        for label in ("SOFTEST", "MIDDLE", "HARDEST"):
            row = deg[(deg["circuit"] == c) & (deg["compound_rank_label"] == label)]
            lim = (
                limits[(limits["circuit"] == c) & (limits["compound_rank_label"] == label)]
                if limits is not None
                else pd.DataFrame()
            )
            if not len(row):
                compounds.append({"rank": label, "estimated": False})
                continue
            r = row.iloc[0]
            compounds.append(
                {
                    "rank": label,
                    "estimated": True,
                    "sim": _num(r["slope_s_per_lap_sim"]),
                    "ipcw": _num(r["slope_ipcw"]),
                    "se": _num(r["se_ipcw_race_clustered"]),
                    "naive": _num(r["slope_naive"]),
                    "n_stints": int(r["n_stints"]),
                    "max_stint": int(lim["max_stint"].iloc[0]) if len(lim) else None,
                }
            )
        out.append(
            {
                "key": c,
                "name": _pretty(c),
                "races": int(t["n_races"]),
                "race_laps": _num(ref["race_laps"].get(c)),
                "median_lap_s": _num(ref["median_lap_s"].get(c)),
                "value_s": _num(t["value_s"]),
                "value_lo": _num(t["value_lo"]),
                "value_hi": _num(t["value_hi"]),
                "pass_per_lap": _num(t["p_pass_per_lap"]),
                "focus": bool(t["is_focus"]),
                "pit_loss_s": _num(pit["pit_loss_shrunk"].get(c)),
                "pit_loss_se": _num(pit["pit_loss_shrunk_se"].get(c)),
                "cautions_per_race": _num(haz["expected_cautions_per_race"].get(c)),
                "hazard": _num(haz["hazard_shrunk"].get(c)),
                "hazard_se": _num(haz["hazard_shrunk_se"].get(c)),
                "passes_per_race": _num(det["mean_passes_per_race"].get(c)) if len(det) else None,
                "compounds": compounds,
            }
        )
    return out


def _page(request: Request, name: str, active: str, **extra):
    """Render a page with the context every page shares."""
    if not STATE.get("ready"):
        return templates.TemplateResponse(request, "not_ready.html", {}, status_code=503)
    ctx = {
        "active": active,
        "author": AUTHOR,
        "metrics": STATE["metrics"],
        "gate": STATE["gate"],
    }
    ctx.update(extra)
    return templates.TemplateResponse(request, name, ctx)


def _require_ready() -> None:
    if not STATE.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Model artefacts not built. Run `python -m pitwall.pipeline`.",
        )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "ready": bool(STATE.get("ready"))}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not STATE.get("ready"):
        return _page(request, "", "")
    return _page(
        request,
        "index.html",
        "overview",
        trackposition=_records(STATE["trackposition"]),
        degradation=_records(STATE["degradation"]),
        reliability=STATE["reliability"],
        headline=STATE["headline"],
        deg_ordering=STATE["deg_ordering"],
        selection=STATE["selection"],
        strategy_validation=STATE["strategy_validation"],
        gate_evidence=STATE["gate_evidence"],
    )


@app.get("/circuits", response_class=HTMLResponse)
def circuits_page(request: Request):
    if not STATE.get("ready"):
        return _page(request, "", "")
    return _page(request, "circuits.html", "circuits", profiles=STATE["circuit_profiles"])


@app.get("/tyres", response_class=HTMLResponse)
def tyres_page(request: Request):
    if not STATE.get("ready"):
        return _page(request, "", "")
    return _page(
        request,
        "tyres.html",
        "tyres",
        degradation=_records(STATE["degradation"]),
        deg_ordering=STATE["deg_ordering"],
        selection=STATE["selection"],
    )


@app.get("/validation", response_class=HTMLResponse)
def validation_page(request: Request):
    if not STATE.get("ready"):
        return _page(request, "", "")
    return _page(
        request,
        "validation.html",
        "validation",
        reliability=STATE["reliability"],
        strategy_validation=STATE["strategy_validation"],
        gate_evidence=STATE["gate_evidence"],
        quality_gates=STATE["quality_gates"],
        gate_labels=GATE_LABELS,
        exclusions=STATE["exclusions"],
        detector=STATE["detector"],
        fuel_check=STATE["fuel_check"],
        selection=STATE["selection"],
    )


@app.get("/simulator", response_class=HTMLResponse)
def simulator_page(request: Request):
    if not STATE.get("ready"):
        return _page(request, "", "")
    return _page(
        request,
        "simulator.html",
        "simulator",
        circuits=[{"key": c, "name": _pretty(c)} for c in STATE["circuits"]],
    )


@app.get("/api/track-position")
def api_track_position() -> JSONResponse:
    _require_ready()
    return JSONResponse(_records(STATE["trackposition"]))


@app.get("/api/degradation")
def api_degradation() -> JSONResponse:
    _require_ready()
    return JSONResponse(_records(STATE["degradation"]))


@app.get("/api/circuit/{circuit}")
def api_circuit(circuit: str) -> JSONResponse:
    """Everything known about one circuit, for the simulator panel."""
    _require_ready()
    from pitwall import pipeline

    if circuit not in STATE["circuits"]:
        raise HTTPException(404, f"unknown circuit {circuit}")
    p = pipeline.circuit_params(circuit, STATE["artifacts"])
    tp = STATE["trackposition"]
    row = tp[tp["circuit"] == circuit]
    return JSONResponse(
        {
            "circuit": circuit,
            "name": _pretty(circuit),
            "race_laps": p.race_laps,
            "base_lap_s": round(p.base_lap_s, 3),
            "pit_loss_s": round(p.pit_loss_s, 2),
            "fuel_s_per_lap": round(p.fuel_s_per_lap, 4),
            "caution_hazard_per_lap": round(p.caution_hazard_per_lap, 5),
            "traffic_s": round(p.traffic_s, 3),
            "deg_by_rank": {str(k): round(v, 5) for k, v in p.deg_by_rank.items()},
            "pass_p_base": round(p.pass_p_base, 5),
            "n_races": int(row["n_races"].iloc[0]) if len(row) else None,
            "value_s": float(row["value_s"].iloc[0]) if len(row) else None,
            "value_lo": float(row["value_lo"].iloc[0]) if len(row) else None,
            "value_hi": float(row["value_hi"].iloc[0]) if len(row) else None,
        }
    )


class SimRequest(BaseModel):
    """A live what-if: one car's stops against a representative field."""

    circuit: str
    stops: list[tuple[int, int]] = Field(default_factory=list)
    start_rank: int = 0
    grid_position: int = 5
    n_sims: int = 600
    field_size: int = 12


@app.post("/api/simulate")
def api_simulate(req: SimRequest) -> JSONResponse:
    """Re-simulate with a user-chosen strategy and return the distribution."""
    _require_ready()
    from pitwall import pipeline

    if req.circuit not in STATE["circuits"]:
        raise HTTPException(404, f"unknown circuit {req.circuit}")
    if not 1 <= req.field_size <= 20:
        raise HTTPException(400, "field_size must be 1-20")
    if not 100 <= req.n_sims <= 3000:
        raise HTTPException(400, "n_sims must be 100-3000")

    p = pipeline.circuit_params(req.circuit, STATE["artifacts"])
    for lap, rank in req.stops:
        if not 1 <= lap < p.race_laps:
            raise HTTPException(400, f"stop lap {lap} outside 1-{p.race_laps - 1}")
        if rank not in (0, 1, 2):
            raise HTTPException(400, "compound rank must be 0, 1 or 2")

    # A representative field: evenly spread pace, each on a sensible one-stop,
    # so the user's choice is judged against a plausible race rather than a
    # field of clones.
    mid = p.race_laps // 2
    field = [
        Car(
            driver=f"CAR{i + 1}",
            pace_offset_s=0.12 * i,
            grid_position=i + 1,
            strategy=Strategy(stops=((mid + (i % 5) - 2, 2),), start_rank=1),
        )
        for i in range(req.field_size)
    ]
    focus = min(req.grid_position, req.field_size) - 1
    field[focus] = Car(
        driver="YOU",
        pace_offset_s=0.12 * focus,
        grid_position=focus + 1,
        strategy=Strategy(stops=tuple(req.stops), start_rank=req.start_rank),
    )

    res = simulate(p, field, n_sims=req.n_sims)
    pos = res.finish_positions[:, focus]
    gap = res.gaps_to_winner[:, focus]
    hist = np.bincount(pos, minlength=req.field_size + 1)[1:].tolist()

    return JSONResponse(
        {
            "circuit": req.circuit,
            "name": _pretty(req.circuit),
            "race_laps": p.race_laps,
            "n_sims": req.n_sims,
            "grid_position": focus + 1,
            "mean_position": float(pos.mean()),
            "median_position": float(np.median(pos)),
            "p05_position": float(np.percentile(pos, 5)),
            "p95_position": float(np.percentile(pos, 95)),
            "win_rate": float((pos == 1).mean()),
            "podium_rate": float((pos <= 3).mean()),
            "mean_gap_s": float(gap.mean()),
            "gap_lo": float(np.percentile(gap, 5)),
            "gap_hi": float(np.percentile(gap, 95)),
            "position_histogram": hist,
            "mean_cautions": float(res.n_cautions.mean()),
        }
    )


@app.post("/api/optimise")
def api_optimise(req: SimRequest) -> JSONResponse:
    """Best strategies for a circuit, closed form then simulated."""
    _require_ready()
    from pitwall import pipeline
    from pitwall.optimize import diverse_candidates, enumerate_strategies

    if req.circuit not in STATE["circuits"]:
        raise HTTPException(404, f"unknown circuit {req.circuit}")
    p = pipeline.circuit_params(req.circuit, STATE["artifacts"])
    cands = diverse_candidates(enumerate_strategies(p, max_stops=2), 8)

    mid = p.race_laps // 2
    field = [
        Car(
            driver=f"CAR{i + 1}",
            pace_offset_s=0.12 * i,
            grid_position=i + 1,
            strategy=Strategy(stops=((mid + (i % 5) - 2, 2),), start_rank=1),
        )
        for i in range(req.field_size)
    ]
    focus = min(req.grid_position, req.field_size) - 1

    rows = []
    for c in cands:
        trial = list(field)
        trial[focus] = Car(
            driver="YOU",
            pace_offset_s=0.12 * focus,
            grid_position=focus + 1,
            strategy=c.strategy,
        )
        r = simulate(p, trial, n_sims=400)
        pos = r.finish_positions[:, focus]
        rows.append(
            {
                "strategy": str(c.strategy),
                "stops": [list(s) for s in c.strategy.stops],
                "start_rank": c.strategy.start_rank,
                "n_stops": c.n_stops,
                "deterministic_cost_s": round(c.deterministic_cost_s, 2),
                "mean_position": round(float(pos.mean()), 3),
            }
        )
    rows.sort(key=lambda r: r["mean_position"])
    return JSONResponse({"circuit": req.circuit, "candidates": rows})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
