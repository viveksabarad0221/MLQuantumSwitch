# Calibration log: waveplate zeros and mirror phase

This file records every change to the four tomography calibration parameters, and nothing else:

| parameter | constant in `tomography.py` |
|---|---|
| HWP1 zero | `CAL_HWP1_DEG` |
| QWP0 zero | `CAL_QWP0_DEG` |
| HWP0 zero | `CAL_HWP0_DEG` |
| mirror phase | `MIRROR_PHASE_DEG` |

`tomography_single_photon.py` and `debug_motor_moves.py` read the same constants.

Add an entry every time one of them changes. Include the date, old → new values, where the new value came from, and which data sets used it. Newest entries go at the bottom.

**Conventions**
- Commanded motor angle = target angle + zero.
- A Jones fit gives offsets d such that the model angle = commanded − true zero. So **true zero = zero used − d**.
- The mirror is modelled as M = diag(1, e^{iφ}), with φ the mirror phase.

## Current values (since 2026-11-13)

| HWP1 | QWP0 | HWP0 | mirror phase |
|---|---|---|---|
| 132.42 | **119.27** | **58.19** | 210.26 |

QWP0 has been the new motor since 2026-11-13: ELL14 s/n 11400796, bus address 1.

## History

Entries 1-4 predate this log. They are reconstructed from the `calibration_deg` and `mirror_phase_deg` fields that each run saves in its JSON. The dates are the files' modification times. All of entries 1-8 used the old QWP0 motor; that data is in `Data/Tomography/old_QWP0/`.

### 1. 2026-09-04: earliest recorded values (source unknown)
- HWP1 132.42, QWP0 90.01, HWP0 75.3, no mirror correction.
- Used by: `old_QWP0/archive/preliminary/test0`-`test3`.
- Later the same day, QWP0 → 42.96 and HWP0 → 62.42, used by `old_QWP0/archive/with_mirror_phase/test0`-`test3-hwp1-22.5`.

### 2. 2026-09-11: characterisation-plot values
- QWP0 128.91, HWP0 147.77 (HWP1 132.42), no mirror correction.
- These equal 38.91 + 90 and 57.77 + 90, the "from characterization plot" values. 128.91 is the QWP0 slow axis.
- Used by: `old_QWP0/archive/with_mirror_phase/test4`-`test15`.

### 3. 2026-10-02: polarimeter values; mirror correction introduced
- QWP0 37.85, HWP0 57.35 (HWP1 132.42), from a polarimeter.
- Used without mirror correction by test18 (2026-10-02).
- Mirror phase 212.05 was then introduced, fitted to test6. Used by `old_QWP0/test0-hwp1-0/22.5/45` (2026-10-06).

### 4. 2026-10-06: joint fit to laser test0
| parameter | old → new |
|---|---|
| QWP0 | 37.85 → **39.40** |
| HWP0 | 57.35 → **57.24** |
| mirror | 212.05 → **209.47** |
- HWP1 stayed at 132.42.
- Source: joint fit of the mirror phase and the QWP0/HWP0 zeros to `test0-hwp1-0/22.5/45`.
- Used by: `old_QWP0/test1-hwp1-*`, and single-photon `sp_test_*`, `sp_rep1`-`4`, `sp_sweep1`.

### 5. 2026-11-12 ~18:25: per-session fit of sp_sweep1, applied with a SIGN ERROR (reverted)
| parameter | old → new |
|---|---|
| HWP1 | 132.42 → 132.22 |
| QWP0 | 39.40 → 40.97 |
| HWP0 | 57.24 → 56.53 |
| mirror | 209.47 → 211.66 |
- The fitted offsets were added instead of subtracted. The correct values would have been HWP1 132.62, QWP0 37.83, HWP0 57.95.
- Used by: `old_QWP0/SinglePhoton/sp_sweep2_newcal` (worse, as expected).

### 6. 2026-11-12 ~18:35: back to the laser zeros; mirror-only fit
| parameter | old → new |
|---|---|
| HWP1 | → 132.42 |
| QWP0 | → 39.40 |
| HWP0 | → 57.24 |
| mirror | → 216.55 |
- The zeros returned to the entry 4 values at the user's request.
- The mirror phase came from a mirror-only fit with the zeros fixed. It was biased, because φ absorbed the QWP0 zero error.
- Used by: `old_QWP0/SinglePhoton/sp_sweep3_mirror`.

### 7. 2026-11-12 ~19:30: mirror phase from per-session joint fits
- Mirror 216.55 → **210.26 ± 1.4**: the mean of the joint fits to sp_sweep1/2/3, which gave 211.66, 210.21 and 208.90.
- Zeros unchanged (132.42, 39.40, 57.24).
- Used by: `old_QWP0/SinglePhoton/sp_repA_t10`, `sp_repB_t10`.

### 8. Not applied: per-session single-photon fits (2026-11-12)
| session | HWP1 | QWP0 | HWP0 | mirror |
|---|---|---|---|---|
| sp_sweep1 | 132.62 | 37.83 | 57.95 | 211.66 |
| sp_sweep2 | 132.50 | 36.38 | 58.03 | 210.21 |
| sp_sweep3 | 132.39 | 44.29 | 58.49 | 208.90 |
- Uncertainties are about ±0.25 (HWP1), 0.6 (QWP0) and 0.4 (HWP0).
- The QWP0 zero jumped between start-ups. A laser-free dark-point check (2026-11-13) traced this to the old QWP0 motor: its axis moved by 4-7 deg between start-ups, even after cleaning and tightening.

### 9. 2026-11-13: QWP0 motor replaced
- QWP0 39.40 → **118.27**. HWP1, HWP0 and the mirror unchanged (132.42, 57.24, 210.26).
- Source: laser + PM100D crossed/parallel scan of QWP0 with H input, HWP0 crossed at its zero, 4 full start-ups.
  - Dark point (slow axis) at 28.24 / 28.26 / 28.27 / 28.28, a spread of 0.03 deg.
  - Fast axis at 118.27 ± 0.02, decided by a D-input test: r = 0.736 measured, 0.752 predicted.
  - Retardance 90.3 deg, extinction 0.13%.
- Caveat: the dark point is measured against HWP0, so it carries any HWP0 zero error. This is what entry 10 corrected.
- Used by: `Data/Tomography/test0-hwp1-0/22.5/45/67.5/90` (laser).
  - Bloch-angle errors 1.9 / 1.4 / 3.0 / 2.5 / 2.7 deg; fidelity ≥ 0.998.

### 10. 2026-11-13: joint fit to the new laser test0
| parameter | old → new |
|---|---|
| QWP0 | 118.27 → **119.27 ± 0.28** |
| HWP0 | 57.24 → **58.19 ± 0.26** |
- HWP1 (132.42) and the mirror (210.26) were kept: their fitted values are consistent with the current ones.
- Source: Jones fit to the 20 powers of `test0-hwp1-0..90`. Free parameters: all three zero offsets, the mirror phase, the visibility and the laser power.
  - Fitted HWP1 zero 132.57 ± 0.13, mirror 209.1 ± 0.9, visibility 0.997.
  - rms residual 1.07% → 0.34% of max power.
  - Predicted Bloch-angle error ~1.0 deg (leave-one-run-out; 0.6 / 0.9 / 0.8 / 1.7 / 0.6 deg), down from 2.3 deg.
- The HWP0 correction matches the single-photon fits in entry 8 (57.95-58.49). The QWP0 change follows from it, because the entry 9 dark point is shifted by the HWP0 error.
- Fit script: `fit_laser_new.py` in the session scratchpad (not in the repo).
- Verified 2026-11-13 with laser `Data/Tomography/test1-hwp1-0/22.5/45/67.5/90`:
  - Bloch-angle errors 0.25 / 0.92 / 1.04 / 1.03 / 1.03 deg (mean 0.85, was 2.29 with entry 9); fidelity ≈ 0.999.
  - The H inputs' Sx went from −0.031 / −0.045 to +0.004 / −0.018. The A input's |S| of 1.006 (normalisation) remains.
- Used by: test1 (laser) and the following single-photon runs.
