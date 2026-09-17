# Validation Plan (pre-registered)

**Written and committed before any model was fitted or any result computed.**
Commit history is the evidence: this file's first commit precedes every commit
that produces a number.

The point of writing it first is to remove the freedom to tune into a
flattering answer. Where this plan turns out to be wrong or unworkable, the
deviation is recorded in `RUN_LOG.md` with the reason, rather than the plan
being quietly edited.

---

## 1. Exclusion rules (mechanical, applied before looking at results)

A race is **excluded from strategy modelling** if any of the following hold.
Each is checkable from data alone, with no judgement call at the point of use.

| # | Rule | Threshold | Why |
|---|---|---|---|
| E1 | Too few green-flag racing laps | fewer than **20** laps under track status `1` across the field's median driver | With no green running there is no strategy to model |
| E2 | Race abandoned / not classified as a full race | official classification shows fewer than **2 racing laps** (e.g. Spa 2021) | No lap data of any strategic meaning |
| E3 | Early red flag hands the field a free tyre change | a red flag (`track_status == 5`) occurring within the first **25%** of scheduled distance | The stationary tyre change removes the pit-loss premise the whole model rests on |
| E4 | Lap table missing or unparseable | FastF1 raises `DataNotLoadedError`, or lap count is 0 | Source archive hole, not a modelling choice |
| E5 | Wet-declared race | more than **30%** of classified laps run on `INTERMEDIATE` or `WET` | Dry-tyre degradation model does not apply; wet strategy is a different problem, out of scope |

Excluded races are **listed by name in the methodology report**, with which
rule caught them. They are not silently dropped.

E3 and E5 exclude races from *degradation and strategy* fitting but their
**safety-car and red-flag events are still used** for the caution hazard model
(Phase 4) — removing the chaotic races from a model *of* chaos would bias it
badly. This split is deliberate and is stated wherever hazard results appear.

### Lap-level filters (applied within retained races)

| # | Rule | Threshold |
|---|---|---|
| L1 | Implausible lap time | `< 50 s` or `> 400 s` |
| L2 | Non-green lap | `TrackStatus` contains anything other than `1` |
| L3 | In-lap or out-lap | `PitInTime` or `PitOutTime` is not null |
| L4 | First lap of race | `LapNumber == 1` (standing start is not representative pace) |
| L5 | Invalid compound label | compound in `{nan, UNKNOWN, ''}` |
| L6 | Stint too short to identify a slope | stint length `< 4` laps (degradation fit only) |

L1-L5 apply to all pace modelling. L6 applies only to degradation.

---

## 2. The five documented data traps and how each is handled

These are stated here as commitments, before the results exist.

**T1 - Compound labels are relative, not absolute.**
Never pool by the label `SOFT`/`MEDIUM`/`HARD` across events. Each event's
three compounds are mapped to the underlying Pirelli C-range where the
allocation is known, and modelled as a hardness index. 2018 used a different,
seven-name scheme entirely and is mapped separately. Where an allocation
cannot be established, the event's compounds are modelled as
event-specific random effects rather than assumed. The share of events with
an unresolved allocation is reported.

**T2 - Degradation observations are censored.**
Teams pit when the tyre is finished, so the worst part of every degradation
curve is systematically unobserved. A naive OLS slope on stint data is
therefore biased toward optimism.
*Chosen approach:* an accelerated-failure-time / survival model of stint
length (`lifelines`) to characterise the pitting decision, combined with a
censored (Tobit-style) regression for the degradation slope itself.
*Committed deliverable:* a direct side-by-side of the naive fit against the
censoring-corrected fit, per compound and circuit. **That comparison is
published whichever direction it goes**, including if the correction turns out
to change little.

**T3 - Regulation eras are not comparable.**
Era enters as a fixed factor with levels `2017-2018_wide_aero`,
`2019-2021_simplified_front_wing`, `2022-2025_ground_effect`,
`2026_new_regs`. Nothing is pooled across eras without the factor present. The
2021 floor cut is flagged for a sensitivity check.

**T4 - Sample sizes are badly unbalanced.**
Partial pooling (hierarchical random effects) so thin circuits borrow strength
from the calendar. **Per-circuit race counts are printed next to every
per-circuit number in every user-facing surface.** Intervals widen where data
is thin, and that is by construction, not decoration.

**T5 - Fuel burn confounds degradation.**
Cars get faster through a stint as fuel burns off, partially masking tyre
wear. The fuel effect is **estimated from data** as a per-circuit
seconds-per-lap-of-fuel-burned term, identified from laps early in a stint
across differing fuel loads, not assumed from a textbook figure. Degradation
is only interpreted after the fuel term is in the model.

**T6 (found during Phase 0, added here before results) - Circuit identity is
not stable in the source data.**
FastF1's `Location` field names the same circuit differently across seasons:
`Monaco`/`Monte Carlo`, `Singapore`/`Marina Bay`, `Miami`/`Miami Gardens`,
`Yas Island`/`Yas Marina`. Naive grouping by `Location` would split one
circuit into two thin samples and corrupt every per-circuit estimate. A
canonical circuit key is defined in `pitwall/circuits.py` and asserted in
tests.

---

## 3. Validation procedures

### V1 - Held-out race prediction
Fit on a training subset, simulate **entirely unseen races**, compare
predicted finishing order and gaps to actual.
- Split: by season, not randomly, to avoid leaking within-race information.
  Hold out the most recent complete season and a random 15% of earlier races.
- Metrics: Spearman rank correlation of predicted vs actual finishing order;
  mean absolute error of predicted gap to winner (seconds); share of podium
  correctly identified.
- **Reported honestly whatever the numbers are.** A weak result is published
  as a weak result.

### V2 - Overtaking calibration
The overtaking model outputs probabilities, so it must be calibrated, not just
discriminative.
- Reliability curve with 10 bins, plus Brier score and ECE.
- Discrimination (ROC-AUC) reported alongside, never instead.
- **Acceptance:** the reliability curve is published regardless. Miscalibration
  is a finding, not a failure to hide.

### V3 - THE SANITY GATE (the most important check in the project)
Real F1 strategists are excellent and have far more information than this
model - live tyre temperatures, radio, competitor telemetry, and a race
engineer's judgement.

**Pre-registered expectation:** the optimiser's mean gain over teams' actual
strategy should be **small and frequently indistinguishable from zero**.

**Failure criteria - if any of these hold, the model is declared broken and
the cause hunted down before any headline number is published:**
- Mean claimed gain across all cars and races exceeds **2.0 s** of race time, or
- The optimiser claims a gain for more than **70%** of car-races, or
- Any single claimed gain exceeds **30 s** without an identifiable, explainable
  cause (e.g. a car that actually retired).

**Committed in advance:** if the gate trips, the diagnosis and the fix are
written up and featured **prominently in the README as a headline result**,
not buried. A model that had to be caught lying is a better portfolio artefact
than one that was never tested.

### V4 - Ablations
Refit with each component removed and report the change in held-out error:
- no traffic / dirty-air term
- no fuel-burn term
- no censoring correction (naive degradation)
- no era factor
- no partial pooling (per-circuit independent fits)

This quantifies which components earn their place. A component that does not
improve held-out error is **reported as not earning its place**, and said so
plainly, rather than kept for appearances.

### V5 - Data-quality gates (Phase 1, must pass before modelling)
- Row counts per session non-zero for retained races
- Lap numbers monotonic within `(race, driver)`
- Stint numbers non-decreasing within `(race, driver)`
- `TyreLife` resets at each stint boundary
- No lap time outside the L1 bounds survives filtering
- Every retained lap has a valid compound
- Canonical circuit key is 1:1 with a known circuit

Failures are counted and reported, not silently dropped.

---

## 4. The headline number and how it is allowed to be claimed

**Target:** the value of track position, in seconds, per circuit - defined as
the expected race-time cost of losing one net position on track, given the
circuit's overtaking difficulty.

**Rules for stating it:**
1. It carries an interval, always, everywhere. No point estimate appears
   anywhere without one.
2. The interval reflects both parameter uncertainty and simulation noise.
3. Per-circuit sample size appears next to it.
4. If the estimate cannot be identified from the data for a given circuit,
   the surface says so - it does not fall back to a pooled number dressed up
   as a circuit-specific one.
5. If the whole finding fails to establish, `MORNING_REPORT.md` says it failed
   to establish, and why.

---

## 5. What would falsify the project's central claim

Stated in advance so it is not rationalised away later.

The central claim is that optimal strategy **inverts** across circuits
according to the value of track position. It is falsified if:
- The per-circuit track-position values are not statistically distinguishable
  from one another, **or**
- The optimiser's preferred strategy does not actually change direction
  between the high-overtaking and low-overtaking ends of the spread.

If either holds, that is the reported finding.

---

*Committed before results. Deviations logged in `RUN_LOG.md`.*

---

## Addendum, 2026-09-17 (added after results; the plan above is unchanged)

- Deviations are recorded in `METHODOLOGY.md` §8 rather than in a separate run
  log, which is no longer part of the repository.
- V1 was not run; V4 was run only for the leakage ablation.
- V3 failed. V6 (strategy costs, real vs simulated) was added afterwards as a
  diagnostic and is post hoc.

