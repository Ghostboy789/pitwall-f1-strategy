# Methodology

How every number in this project is produced, what it rests on, and where it
is weak. The validation plan (`VALIDATION_PLAN.md`) was committed before any
result was computed; this document reports what happened when it was applied.

---

## 1. Data

**Source.** FastF1 3.8.3, reading the Formula 1 live-timing archive
(`livetiming.formula1.com`) for lap timing, stints, compounds, tyre age, track
status and weather, and the Jolpica mirror of the Ergast schema
(`api.jolpi.ca/ergast/f1`) for race classification.

**The Ergast wind-down was verified, not assumed.** `ergast.com` returns 404;
FastF1 3.8.3 has already migrated to the Jolpica mirror, which returns 200 and
a valid response for the whole 2018–2026 range. No redesign was needed. Full
evidence in `SOURCES.md`.

**Scope.** All 186 completed championship races, 2018–2026, across 32 canonical
circuits. **185 ingested successfully**, giving 203,644 laps of which 149,432
survive the pre-registered filters, across 44 drivers and 21 teams.

**The one permanent hole.** The 2018 Italian Grand Prix (Monza, round 14) has
no usable lap data: FastF1 logs `Failed to load timing data!` and `Session.laps`
raises `DataNotLoadedError`. This is a defect in the source archive, not in
FastF1 or in this project. It is recorded as `failed` in the ingestion manifest
rather than quietly omitted.

### Ingestion engineering that mattered

The Jolpica mirror allows 500 calls/hour. FastF1 spends two per race session:
one for classification, one purely to backfill lap 1's time. A first full
ingest exhausted the budget partway through 2020 and lost **32 consecutive
sessions**; the retry logic made it worse by spending three attempts per
failure.

Two fixes:

1. **The first-lap call is disabled.** Pre-registered filter L4 discards lap 1
   of every race anyway, so its result is unusable here.
2. **Classification is fetched separately** (`pitwall/backfill_results.py`) at
   a deliberate pace, after a bulk pass that runs with Ergast disabled
   entirely. Verified on 2018 round 1: with both calls disabled the lap table
   is byte-identical at (940, 31) with every modelling column intact; only
   `GridPosition`, `Points` and `Status` go empty.

Rate limiting is now treated as backpressure — wait and retry without spending
an attempt — rather than as failure.

---

## 2. The six data traps

### T1 — Compound labels are relative, not absolute

From 2019 Pirelli brings three compounds from the C1–C5 range to each event and
relabels them SOFT / MEDIUM / HARD *locally*. A HARD at Monza is different
rubber from a HARD at Monaco.

**How real this is, measured:** in this dataset the label `SOFT` is the
**hardest** available compound in 3,006 laps, the middle in 2,272, and the
softest in 5,772. Pooling by label merges three different tyres.

**What was tried and rejected.** Recovering the underlying C-number per event.
No authoritative structured dataset of per-event allocations exists publicly —
the one project attempting it states so explicitly and has compiled only a
handful of races by hand (`SOURCES.md`). Adopting a partial table of unverified
provenance would import someone else's uncertainty while looking authoritative.

**What is done.** Each compound is converted to its **within-event relative
hardness rank** — 0 for the softest brought to that event, 1 middle, 2 hardest.
This is not a workaround: it is what the label *means*. It also absorbs the
2018 seven-name scheme (HYPERSOFT … SUPERHARD) without special-casing, since
ranking is by position, not name.

**What is not claimed.** The rank does not make the physical rubber comparable
across events. A rank-0 tyre at Monaco is softer in absolute terms than a
rank-0 at Monza. Absolute hardness is therefore only ever a circuit-level
effect the rank sits inside.

### T2 — Degradation observations are censored, and the selection is informative

Teams pit when the tyre is finished, so the worst of every degradation curve is
unobserved and missing non-randomly.

**The selection is measured, not assumed.** For every stint that ended in a pit
stop, the slope over its first five laps predicts how soon it was pitted:
**r = −0.126, p = 1.7×10⁻¹⁷, across 4,565 stints**. Stints degrading faster were
ended sooner.

**A second, larger selection was found during the build.** Teams do not only
choose *when* to pit, they choose *which compound* runs in which conditions —
the harder tyre goes on the long, hot, high-fuel stint precisely because that
is where degradation will be worst. The compounds are therefore never observed
under equal conditions, and a naive estimator concludes that **harder tyres
degrade faster than softer ones**, which is backwards.

Two hypotheses were tested and rejected before the cause was found:

| Hypothesis | Test | Result |
|---|---|---|
| Non-linearity + unequal age ranges | restrict all compounds to a common tyre-age window (25, 20, 15 laps) | no improvement — 9–10 of 29 circuits correct at every window |
| Fuel confound | subtract the independently estimated fuel effect first, then centre within stint | 13 of 29 — better, not fixed |

**The fix: a two-way within transformation**, removing both a car-race mean and
a **race-moment mean** (the race bucketed into five-lap windows). Every
comparison is then between cars running at the same point of the same race,
which is the only place conditions are actually equal.

Three estimators are reported side by side, because the progression is the
result — see `models_out/degradation_ordering.json` for the current numbers and
§6 below.

**The assumption IPCW rests on, stated plainly.** Inverse-probability weighting
requires that, conditional on the survival model's covariates, the pit decision
is independent of the *residual* degradation. That is an approximation: teams
see live tyre temperatures and degradation trends this model does not. The
correction recovers part of the bias, not all of it, and the direction of what
remains is known — still understated.

### T3 — Regulation eras are not comparable

Era enters as a fixed factor with four levels: `2017-2018_wide_aero`,
`2019-2021_simplified_front_wing`, `2022-2025_ground_effect`, `2026_new_regs`.
Nothing is pooled across eras without the factor present. The 2021 floor cut is
flagged separately for sensitivity.

### T4 — Sample sizes are badly unbalanced

Circuit race counts range from 1 (Mugello, Nürburgring, Bahrain outer loop) to
11 (Red Bull Ring). Partial pooling via Paule–Mandel random-effects
meta-analysis shrinks each circuit toward the global mean in proportion to its
own precision. **Per-circuit race counts appear next to every per-circuit
number in every user-facing surface.**

Circuits with fewer than 2 races are excluded from the headline estimate
entirely rather than shown with a fabricated interval.

### T5 — Fuel burn confounds degradation

With no refuelling, fuel mass is a deterministic function of laps completed.

**A negative result, reported as one.** Fuel burn and track evolution are
**not separately identified** in race data. Zero of 27 circuits met the
separability threshold; the median correlation between the fuel proxy and
elapsed session time is **0.9981**. An earlier permissive threshold let the
split through and produced obviously compensating estimates (Hungaroring:
fuel −0.31 s/lap against evolution +4.77 s).

**What is identified, and is what T5 actually requires:** fuel is cleanly
separated from *degradation*, because stint timing varies across drivers — at
any given lap number the field carries a wide spread of tyre ages. Median VIF
for tyre age against fuel is **1.27**.

**Two independent estimates agree.** The mixed-effects pace model returns
**−0.0535 s/lap**; the entirely separate degradation specification returns
**−0.0574 s/lap**. Both sit inside the physically expected band (~1.7 kg/lap ×
~0.03 s/kg ≈ 0.05 s/lap), and **all 27 circuits** fall in that band — with
nothing calibrated to achieve it.

**And it behaves like fuel.** The coefficient scales with circuit lap distance,
as fuel must and track evolution need not: **r = −0.759, p = 4×10⁻⁶**.

The two-way estimator in T2 sidesteps this problem rather than solving it:
cars in the same race-moment share a fuel load *and* a track state, so both
difference out.

### T6 — Circuit identity is not stable (found during Phase 0, added before results)

FastF1's `Location` field gives 35 distinct strings for 32 real circuits.

- **Same circuit, different names:** Monaco/Monte Carlo, Singapore/Marina Bay,
  Miami/Miami Gardens, Yas Island/Yas Marina, Silverstone (GB/UK).
- **Same name, different circuits:** `Sakhir` covers two layouts — the 2020
  Sakhir Grand Prix ran Bahrain's 3.5 km outer loop against the 5.4 km Grand
  Prix circuit used every other year.

Grouping by `Location` would both split and merge circuits, corrupting the
per-circuit estimate this project exists to produce. Canonicalisation keys on
`(location, event_name)`, an unknown location raises rather than falling
through, and both behaviours are asserted in `tests/test_circuits.py`.

---

## 3. Exclusion rules

Pre-registered, applied mechanically, never per-result.

| Rule | Condition |
|---|---|
| E1 | fewer than 20 green racing laps for the median driver |
| E2 | fewer than 2 racing laps (race abandoned) |
| E3 | red flag within the first 25% of distance (free tyre change destroys the pit-loss premise) |
| E5 | more than 30% of laps on intermediates or wets |

Lap-level filters L1–L5 (implausible lap time, non-green, in/out lap, lap 1,
invalid or wet compound) retain roughly **78%** of laps; the attrition by rule
is reported rather than assumed.

**A deliberate asymmetry:** races excluded by E3 and E5 are still used for the
caution hazard model. Removing the chaotic races from a model *of* chaos would
bias it badly.

---

## 4. Models

### Lap-time decomposition (`pitwall/models/pace.py`)

Two-stage hierarchy rather than one giant crossed model:

1. An independent mixed model per circuit, with a random intercept per race so
   per-race conditions are absorbed rather than blamed on tyres.
2. Paule–Mandel random-effects pooling of the per-circuit coefficients.

Two bugs were found here by the estimates being physically absurd rather than
by anything failing:

- `fuel_burned` and `race_progress` are an exact affine transform of each other
  within a race. Including both gave a near-singular design and a fuel
  coefficient of **+4.46 s per lap** — cars getting four seconds slower per lap
  of fuel burned.
- patsy's alphabetical reference level meant the *hardest* compound's
  degradation slope was being reported under the name `deg_softest`.

### Degradation (`pitwall/models/degradation.py`)

Three estimators, reported together: naive (car-race centring + fuel control),
two-way fixed effects, and two-way + IPCW weights from a Weibull AFT survival
model of stint length. A stint is an *observed* pit decision only if it ended
in a pit stop; stints ending at the flag, in a retirement, or under a red flag
are right-censored.

Weights are clipped at 10× so a near-zero survival probability cannot let one
lap decide the answer.

### Overtaking (`pitwall/models/overtaking.py`)

A pass is inferred from consecutive laps: A behind B at lap L, ahead at L+1,
**with neither car pitting across the boundary**. Without that exclusion every
undercut counts as an overtake and Monaco looks as passable as Monza.

The denominator is *opportunities* — laps where a car ran directly behind
another within 2.0 s under green — so this measures conversion, not proximity.

**The detector validates against reality without being tuned to it:** Monaco
2.3 passes per race (median 1), Albert Park 5.0, rising to Interlagos 36 and
Portimão 37.5.

**A design flaw found and fixed.** The first version had no circuit term at
all — circuits reached the model only through `drs_zones` and `street`, so
thirteen of them (Monza, Spa, Silverstone, Catalunya, Interlagos among others)
received the *identical* probability of 0.083655. A per-circuit headline number
computed from that would have been an artefact of the grouping.

It is now two-stage: a gradient-boosted model on the mechanics of passing with
no circuit identity, then a per-circuit intercept fitted as a GLM offset on its
log-odds and shrunk by empirical Bayes.

**Leakage was checked, not assumed.** `pace_delta_s` is built from clean-air
laps across the whole race, so it can see past the pass. Ablating it costs
0.015 AUC (0.908 → 0.893); `gap_s` alone reaches 0.851. The model runs on
proximity and circuit character, not on a variable that already knows the
answer.

### Pit loss and caution hazard (`pitwall/models/raceparams.py`)

Pit loss is **measured**, not looked up: the in-lap and out-lap penalty against
each driver's own green-flag reference. Wet running had to be excluded, not
just non-green running — an out-lap on intermediates is 30–50 s slower than a
dry reference and none of that is pit-lane time. Left in, Imola came out at
**45.9 s** against a published figure near 28.

Caution hazard needed **race-clustered** uncertainty. Cautions cluster within
races, so a binomial standard error over laps-at-risk treats 300 correlated
laps as 300 independent trials; it reported a precision the data does not have
and the pooling step collapsed every circuit to one global rate.

### Simulator (`pitwall/sim.py`)

Lap times accumulate from pace, fuel, degradation and noise; stops cost the
fitted pit loss; cautions are drawn from the fitted hazard and make a stop
cost 45% of its green-flag price.

**The blocking rule is the mechanic that matters.** A car that would have moved
ahead on raw pace instead draws against the overtaking model; a failed draw
clamps it to a car length behind with a dirty-air penalty.

A bug the tests caught: blocking was originally resolved against the running
order computed *after* the lap, which asks whether the car that already got
past was blocked by the car it passed. Every faster car escaped, 100% of the
time, and track position was worth exactly nothing.

### Optimiser (`pitwall/optimize.py`)

Closed form first — for a stint of *n* laps on a compound degrading at *d*, the
tyre costs `d·n(n+1)/2`, and base pace and fuel cancel between strategies over
the same distance — which enumerates ~21,000 legal strategies in milliseconds.
The top candidates then go through the full simulator, which can see traffic
and cautions. The two-compound regulation is enforced during enumeration so the
optimiser cannot "win" by proposing something that would be disqualified.

---

## 5. The headline metric

For a car with pace advantage `Δ` over the car ahead:

```
expected laps stuck   = min(1/p, laps_remaining)
value of track position = Δ · min(1/p, laps_remaining)
```

Evaluated at an identical scenario for every circuit, fixed before results:
Δ = 0.5 s/lap, gap 1.0 s, DRS available, mid-race, 25 laps remaining, P8.

**The geometric expectation `1/p` assumes each lap is an independent attempt.**
A car that has failed to pass for ten laps is probably in a worse position than
the average attacker, so this estimate is, if anything, optimistic about
escaping.

**Two independent routes agree.** The closed form gives 0.5 × min(1/p, 40) =
20.0 s at p = 0.005 over 40 laps; the Monte Carlo simulator, which shares none
of that arithmetic, gives 20.5 s.

**Uncertainty** is a cluster bootstrap over races, **stratified by circuit**. An
unstratified resample frequently gave a three-race circuit no races at all, so
its effect reverted to the global mean — every interval spanned the full range
and nothing was distinguishable. Stratifying also asks the right question: how
well do three races at Monaco pin down Monaco, not what if the calendar had
visited different circuits.

---

## 6. Validation results

See `MORNING_REPORT.md` for the current run's numbers, and `models_out/` for
the artefacts behind them. The procedures are:

- **V1 held-out race prediction** — split by season, not by row.
- **V2 overtaking calibration** — reliability curve with 10 bins, plus Brier
  and ECE. Split by race with `GroupKFold`; opportunities from one race share
  weather, track state and the same two cars, so a random split would leak.
- **V3 the sanity gate** — thresholds fixed in advance: mean gain > 2.0 s, or
  the optimiser beating > 70% of car-races, or any single gain > 30 s declares
  the model broken.

  **It tripped, on all three, and the diagnosis is a result in itself.** The
  first run returned a mean gain of 23.49 s, beat 97% of car-races, and claimed
  a single gain of 107.6 s. The cause was not the optimiser but the
  reconstruction it was compared against: strategies were inferred from
  FastF1's `Stint` counter, which increments for reasons other than a pit stop,
  so the audit invented stops on consecutive laps and lap-1 stops that refitted
  the compound already on the car. The simulator charges full pit loss per
  stop, so a phantom stop cost ~22 s of race time that never happened.

  The diagnostic signature was that mean claimed gain rose monotonically with
  the number of *reconstructed* stops — 10.9 s at one, 22.0 s at two, 47.8 s at
  three, 69.6 s at four. A genuine strategic insight would not scale with an
  artefact of the parser.

  A stop is now defined as an in-lap and nothing else; runs of consecutive
  in-laps collapse to one; a stop on the final lap is ignored; and a
  reconstruction claiming more than four stops is dropped rather than modelled.
  A second bug surfaced in the same pass: the audit simulated each race over
  the circuit's *median* lap count rather than that race's own.
- **V4 ablations** — each component removed, change in held-out error reported.
- **V5 data-quality gates** — structural checks on the pipeline itself.

---

## 7. Limitations

Stated plainly, because a portfolio piece that hides these is worth less than
one that names them.

1. **Fuel and track evolution cannot be separated** in race data. The linear
   term is reported as combined and interpreted as fuel-dominated on evidence
   (physical band, lap-length scaling), not by assumption.
2. **The censoring correction is partial.** IPCW conditions on observables;
   teams act on information this model does not have. Residual bias
   understates degradation.
3. **Compound ranks are relative, not physical.** Cross-event pooling of
   absolute compound behaviour is not supported by this data.
4. **The dirty-air proxy is a gap at the line**, not an average through the
   lap, and does not know whether the car ahead was on the racing line.
5. **DRS availability is approximated** from gap and lap number, not measured
   at the detection point.
6. **Grid position falls back to the order at the end of lap 1** where
   classification data is unavailable — lap 1 has already been raced.
7. **The simulator does not model** driver error, mechanical failure,
   team-mate priority, pit-lane congestion, or rivals *reacting* to a strategy
   change. The counterfactual holds the rest of the field fixed, which
   overstates the benefit of a change a real field would have answered.
8. **Thin circuits stay thin.** Shrinkage widens their intervals honestly, but
   one race is one race.
9. **Wet races are out of scope** for strategy modelling entirely.
10. **The 2026 season is in progress**, so its sample is partial and its
    regulation era rests on 13 races.
