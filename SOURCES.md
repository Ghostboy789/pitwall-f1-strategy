# Sources

Every external source consulted, with URL and access date. Research was done
before and during the build, not reconstructed afterwards.

All access dates are **2026-09-08** unless stated otherwise.

---

## Data

### FastF1 (primary data source)
- **Docs:** <https://docs.fastf1.dev/> — accessed 2026-09-08, version **3.8.3**
- **What it provides:** lap timing, stint and compound labels, tyre age,
  track status, weather, race control messages, session results, event
  schedule, for 2018 onwards.
- **Verified directly rather than assumed.** The API surface described in this
  project's brief was checked against the installed 3.8.3 before anything was
  designed around it.

### The Ergast wind-down — verified, not assumed
The Ergast API that FastF1 historically depended on for historical results has
been wound down. This was checked directly rather than taken on trust:

| Endpoint | Response | Note |
|---|---|---|
| `https://ergast.com/api/f1/2024/1/results.json` | **404** | Dead, as expected |
| `https://api.jolpi.ca/ergast/f1/2024/1/results.json` | **200** | Live |
| `https://livetiming.formula1.com/static/2024/Index.json` | **200** | FastF1's timing archive, live |

**Finding:** FastF1 3.8.3 has already migrated. `fastf1.ergast.interface.BASE_URL`
resolves to `https://api.jolpi.ca/ergast/f1` — the community-run Jolpica mirror
of the Ergast schema. A live call (`Ergast().get_race_results(season=2025,
round=16)`) returned a valid `ErgastMultiResponse` with content shape (20, 24).

- **Jolpica-F1 (Ergast successor):** <https://api.jolpi.ca/ergast/f1> — accessed 2026-09-08
- **Jolpica project:** <https://github.com/jolpica/jolpica-f1>

**Consequence for this project:** results data is available across the full
2018-2026 range, so no redesign was needed. FastF1 still calls this mirror for
first-lap times; a 1 s inter-session delay and 3-attempt backoff were added to
the ingestion loop as protection against rate-limiting across 186 sessions.

### Known hole in the source archive
2018 Italian Grand Prix (Monza, round 14) has **no usable lap data**. FastF1
logs `Failed to load timing data!` and `Session.laps` then raises
`DataNotLoadedError`. This is a defect in the F1 livetiming archive itself, not
in FastF1 or in this project. It is recorded as `failed` in the ingestion
manifest and named in the methodology report.

---

## Pirelli compound allocations (trap T1)

The central difficulty: FastF1 exposes only the *relative* labels
SOFT / MEDIUM / HARD. From 2019 Pirelli brings three compounds from the C1-C5
(later C0-C6) range to each event and relabels them locally, so a "HARD" at
Monza is different rubber from a "HARD" at Monaco.

**Research conclusion: no authoritative structured dataset of per-event
allocations exists publicly.**

- **f1-tyre-compound-predictor** — <https://github.com/Sonic815/f1-tyre-compound-predictor>
  — accessed 2026-09-08. The one public project attacking this problem
  directly. Its README states the allocation mapping "doesn't exist anywhere
  as structured data", that compiling it is "a core contribution" of that
  project, and that its own table "currently covers only a handful of manually
  researched races", with expanding it named as the main bottleneck.
  It references **FastF1 Discussion #517**, where a user raised the same gap.

**Decision taken, and why.** Rather than adopt a partial hand-compiled table of
unverified provenance, compound effects are estimated **per event** and pooled
hierarchically across circuits and eras. This is not a workaround — for this
project's purpose it is the better estimator. What the strategy model actually
needs is the pace and degradation *delta between the compounds available at a
given event*, both directly identifiable from race data. The C-number would
only be a device for pooling across events, and partial pooling achieves that
without depending on a dataset that does not exist.

The 2018 season is handled separately: it used a seven-name absolute-ish scheme
(HYPERSOFT through SUPERHARD) that does not map onto the later relative labels
at all. Confirmed directly from the data — 2018 Monaco returns
`HYPERSOFT, ULTRASOFT, SUPERSOFT`, 2018 Spa returns `SUPERSOFT, SOFT, MEDIUM`.

---

## Prior art

Searched deliberately so the README can state honestly what is and is not
novel here.

- **Heilmeier, Graf, Lienkamp — "A Race Simulation for Strategy Decisions in
  Circuit Motorsports"** (Technical University of Munich). The serious
  academic reference for lap-time-decomposition race simulation. Establishes
  the decomposition approach this project's Phase 2 follows.
- **F1 Undercut Monte Carlo Simulation** —
  <https://github.com/diegor117/f1-undercut-monte-carlo> — accessed 2026-09-08.
  Self-describes as "a focused pit-window simulation, not a full
  race-strategy engine"; explicitly does not model traffic, safety cars,
  weather, or alternative strategies.
- Several further public Monte Carlo pit-stop optimisers exist
  (`anushka-srivastavas/Undercut`, `panagiotagrosdouli/formula1-race-simulation`,
  assorted "F1 Strategy AI" repositories), accessed 2026-09-08.

**Honest positioning.** Monte Carlo race simulation is well-trodden and this
project does not claim to have invented it. What is not common in the public
work surveyed:

1. **Correcting the censoring in degradation data.** Teams pit when the tyre
   is finished, so the worst of every degradation curve is unobserved. The
   surveyed projects fit naive slopes.
2. **Calibrating the overtaking model** and publishing a reliability curve,
   rather than reporting discrimination only.
3. **Comparing the value of track position across circuits** as the headline
   output, rather than simulating one race.
4. **A pre-registered sanity gate** that treats an implausibly good optimiser
   as evidence of a broken model.

---

## Circuit reference data

Lap distances and DRS zone counts in `pitwall/circuits.py` are published
reference figures, used only as covariates and sanity checks.

**Pit-lane loss time is deliberately not sourced from publication.** It is
estimated empirically per circuit-season from the data, as the brief requires:
published figures vary between sources and change with pit-lane speed-limit
and layout revisions, whereas the observed in-lap/out-lap time penalty is
directly measurable.

---

## Tooling

- **Jolpica-F1 API** — <https://github.com/jolpica/jolpica-f1>
- **lifelines** (survival analysis, for the censoring correction) —
  <https://lifelines.readthedocs.io/>
- **statsmodels MixedLM** (hierarchical lap-time decomposition) —
  <https://www.statsmodels.org/stable/mixed_linear.html>

---

*Anything added later is appended here with its access date, not backfilled.*
