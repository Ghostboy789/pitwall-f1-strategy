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

import logging
import os
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
app = FastAPI(title="Pit Wall", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

STATE: dict[str, Any] = {}

# A circuit needs this many races before it is allowed to anchor the headline
# claim. Without it the strapline names whichever circuit happens to sit at the
# bottom of the table, which is usually the one with the least evidence.
HEADLINE_MIN_RACES = 5


def _pretty(circuit: str) -> str:
    ref = CIRCUIT_REF.get(circuit)
    return ref.name if ref else circuit.replace("_", " ").title()


@app.on_event("startup")
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
    STATE["reliability"] = pd.read_csv(rel_path).to_dict("records") if rel_path.exists() else []

    sel_path = config.MODELS_OUT / "selection_evidence.csv"
    STATE["selection"] = pd.read_csv(sel_path).to_dict("records")[0] if sel_path.exists() else None

    # The sanity gate is optional: the dashboard is honest about not having run
    # it rather than showing a blank where a verdict should be.
    gate_path = config.MODELS_OUT / "sanity_gate.json"
    STATE["gate"] = None
    if gate_path.exists():
        import json

        g = json.loads(gate_path.read_text())
        STATE["gate"] = g if g.get("status") == "ok" else None

    STATE["circuits"] = sorted(set(tp["circuit"]) & set(art["laps"]["circuit"].unique()))
    log.info(
        "loaded artefacts: %d circuits, reliability=%d bins, gate=%s",
        len(STATE["circuits"]),
        len(STATE["reliability"]),
        "yes" if STATE["gate"] else "no",
    )


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
        return templates.TemplateResponse(request, "not_ready.html", {}, status_code=503)
    tp = STATE["trackposition"]
    focus = tp[tp["is_focus"]].sort_values("value_s", ascending=False)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "metrics": STATE["metrics"],
            "trackposition": tp.to_dict("records"),
            "focus": focus.to_dict("records"),
            "degradation": STATE["degradation"].to_dict("records"),
            "reliability": STATE["reliability"],
            "gate": STATE["gate"],
            "headline": STATE["headline"],
            "deg_ordering": STATE["deg_ordering"],
            "selection": STATE["selection"],
            "circuits": [{"key": c, "name": _pretty(c)} for c in STATE["circuits"]],
        },
    )


@app.get("/api/track-position")
def api_track_position() -> JSONResponse:
    _require_ready()
    return JSONResponse(STATE["trackposition"].to_dict("records"))


@app.get("/api/degradation")
def api_degradation() -> JSONResponse:
    _require_ready()
    return JSONResponse(STATE["degradation"].to_dict("records"))


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
