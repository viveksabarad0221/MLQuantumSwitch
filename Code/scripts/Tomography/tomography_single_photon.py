#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Heralded single-photon polarization tomography (single detector per arm).

Same optics and reconstruction as tomography.py (laser + powermeter), with
the powermeter replaced by a single-photon detector and the photon pair
source in place of the laser:

    signal: collimator -> PBS -> HWP1 -> mirror -> QWP0 -> HWP0 -> PBS -> det (ch 8)
    idler : ---------------------------------------------------------> det (ch 16, herald)

For each analysis setting (QWP0, HWP0) we count signal-idler coincidences
on the Swabian Time Tagger. The coincidence count plays the role of the
power reading in tomography.py: n_k ~ <t_k|rho|t_k>. The bases (incl. the
mirror correction), the James et al. Eq. (2.1)-(2.3) reconstruction, the
waveplate zeros and the Bloch-sphere plot are all imported from
tomography.py, so calibration changes there apply here too.

Time Tagger handling (network connection, client-side delays, Coincidences
+ single-bin Counter readout) follows two_qubit_tomo.py from the lab.

Accidentals: the Swabian Coincidences virtual channel counts pairs with
|t_s - t_i| <= window, so the expected accidental count is
2 * window * R_s * R_i * T. It is always recorded; --subtract-accidentals
removes it before reconstruction.

Error bars come from Poisson-resampling the coincidence counts.

First run with new cables/detectors: use --find-delay to locate the
signal-idler correlation peak and set the input delay so it sits at 0.
"""

import argparse
import csv
import json
import os
import time

import numpy
import TimeTagger

from tomography import (RAW_BASES, MIRROR_PHASE_DEG, CAL_HWP1_DEG, CAL_QWP0_DEG, CAL_HWP0_DEG, DATA_DIR,
                        build_bases, move_and_settle, stokes_and_density_matrix, plot_poincare)
import elliptec  # importable after tomography.py has set up sys.path

SP_DATA_DIR = os.path.join(DATA_DIR, 'SinglePhoton')
PS = 1e12  # ps per second (Time Tagger units are ps)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Heralded single-photon polarization tomography: fix HWP1, sweep QWP0/HWP0, '
                    'count signal-idler coincidences on the Swabian Time Tagger.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--name', type=str, default='', help='Name of this run (output filenames)')
    parser.add_argument('-d', '--motor-device', type=str, required=True, help='Serial port of the elliptec bus, e.g. COM5')
    parser.add_argument('-s', '--hwp1-angle', type=float, nargs='+', required=True,
                        help='HWP1 angle(s) [deg] (state prep). Several angles are measured back-to-back in one '
                             'session (motors homed once), e.g. -s 0 22.5 45 67.5 90 0 (repeat 0 as a reference check)')
    parser.add_argument('--tagger-address', type=str, default='131.130.102.199:41101',
                        help='Time Tagger network address host:port (41101 is the Time Tagger Server default)')
    parser.add_argument('--ch-signal', type=int, default=8, help='Time Tagger channel of the signal detector')
    parser.add_argument('--ch-idler', type=int, default=16, help='Time Tagger channel of the idler (herald) detector')
    parser.add_argument('--trigger-level', type=float, default=0.05,
                        help='Trigger level [V] for both channels (0.05 as in PythonQST Settings)')
    parser.add_argument('--delay-signal', type=float, default=32750.0, help='Delay [ps] on the signal channel (32.75 ns)')
    parser.add_argument('--delay-idler', type=float, default=0.0, help='Delay [ps] on the idler channel')
    parser.add_argument('--find-delay', action='store_true',
                        help='Measure the signal-idler correlation first and set the idler delay to centre the peak')
    parser.add_argument('--window', type=float, default=800.0, help='Coincidence window [ps] (|dt| <= window; 800 as in two_qubit_tomo.py)')
    parser.add_argument('-t', '--exposure', type=float, default=5.0, help='Integration time [s] per basis setting')
    parser.add_argument('--subtract-accidentals', action='store_true', help='Subtract estimated accidentals before reconstruction')
    parser.add_argument('--addr-hwp1', type=str, default='0', help='Elliptec address of HWP1 (state prep)')
    parser.add_argument('--addr-qwp0', type=str, default='1', help='Elliptec address of QWP0 (analysis)')
    parser.add_argument('--addr-hwp0', type=str, default='2', help='Elliptec address of HWP0 (analysis)')
    parser.add_argument('--cal-hwp1', type=float, default=CAL_HWP1_DEG, help='HWP1 zero [deg]')
    parser.add_argument('--cal-qwp0', type=float, default=CAL_QWP0_DEG, help='QWP0 zero [deg]')
    parser.add_argument('--cal-hwp0', type=float, default=CAL_HWP0_DEG, help='HWP0 zero [deg]')
    parser.add_argument('--settle', type=float, default=0.3, help='Settle time [s] after each motor move')
    parser.add_argument('--pos-tol', type=float, default=0.5,
                        help='Warn when a read-back waveplate angle is further than this [deg] from the commanded one')
    parser.add_argument('--move-timeout', type=float, default=6.0, help='Elliptec serial timeout [s] (see tomography.py)')
    parser.add_argument('--mirror-phase', type=float, default=MIRROR_PHASE_DEG,
                        help='Mirror V-vs-H phase [deg] compensated by the A/R settings (0 = no correction)')
    parser.add_argument('--raw', action='store_true', help='Also measure the uncorrected A/R settings (state after the mirror)')
    parser.add_argument('--n-boot', type=int, default=2000, help='Poisson resamples for error bars')
    parser.add_argument('--no-plot', action='store_true', help='Skip the Bloch-sphere plot')
    return parser.parse_args()


def set_delay(tagger, channel, delay_ps):
    """Channel delay [ps]. A network tagger (has isConnected) takes a client-side
    delay; a local one takes the input delay. Same as two_qubit_tomo.py."""
    delay_ps = int(round(delay_ps))
    if hasattr(tagger, 'isConnected'):
        tagger.setDelayClient(channel=channel, time=delay_ps)
    else:
        tagger.setInputDelay(channel=channel, delay=delay_ps)


def correlation_peak(tagger, ch_a, ch_b, seconds, binwidth_ps=100, n_bins=2000):
    """Time of the ch_a/ch_b correlation peak [ps] (histogram spans +-100 ns)."""
    corr = TimeTagger.Correlation(tagger, ch_a, ch_b, binwidth=binwidth_ps, n_bins=n_bins)
    try:
        corr.start()
        time.sleep(seconds)
        corr.stop()
        data, index = corr.getData(), corr.getIndex()
    finally:
        corr.stop()
    i = int(numpy.argmax(data))
    print(f'  correlation peak: {data[i]} counts at {index[i]} ps (median bin {numpy.median(data):.0f})')
    return float(index[i])


def find_and_set_delay(tagger, ch_s, ch_i, base_delay_idler, seconds=5.0, binwidth_ps=100):
    """Shift the idler delay so the signal-idler correlation peak sits at 0.

    Tries one sign of the correction and checks it; flips if the peak moved
    the wrong way, so we don't depend on the histogram's sign convention.
    """
    print('Finding signal-idler delay ...')
    tau = correlation_peak(tagger, ch_s, ch_i, seconds, binwidth_ps)
    if abs(tau) <= 2 * binwidth_ps:
        return base_delay_idler
    for delay in (base_delay_idler - tau, base_delay_idler + tau):
        set_delay(tagger, ch_i, delay)
        print(f'  trying idler delay {delay:.0f} ps')
        if abs(correlation_peak(tagger, ch_s, ch_i, seconds, binwidth_ps)) <= 2 * binwidth_ps:
            return delay
    set_delay(tagger, ch_i, base_delay_idler)
    print('WARNING: could not centre the correlation peak; check the peak/background above.')
    return base_delay_idler


def measure_counts(tagger, ch_s, ch_i, window_ps, exposure_s):
    """(signal singles, idler singles, coincidences) over exposure_s seconds.

    Same pattern as two_qubit_tomo.py (Coincidences virtual channel + Counters;
    start, sleep, stop), but with 100 ms bins and a buffer longer than the
    exposure, summed at the end. Unfilled bins are zero, so the sum is the
    total over the run whether or not the last, partial bin is returned.
    """
    binwidth = int(0.1 * PS)
    n_values = int(numpy.ceil(exposure_s / 0.1)) + 10
    coinc = TimeTagger.Coincidences(tagger, [[ch_s, ch_i]], int(window_ps))
    singles_counter = TimeTagger.Counter(tagger, [ch_s, ch_i], binwidth=binwidth, n_values=n_values)
    coinc_counter = TimeTagger.Counter(tagger, list(coinc.getChannels()), binwidth=binwidth, n_values=n_values)
    try:
        coinc.start()
        coinc_counter.start()
        singles_counter.start()
        time.sleep(exposure_s)
        singles_counter.stop()
        coinc_counter.stop()
        coinc.stop()
        singles = numpy.asarray(singles_counter.getData()).reshape(2, -1).sum(axis=1)
        coincidences = numpy.asarray(coinc_counter.getData()).reshape(1, -1).sum(axis=1)
    finally:
        for m in (singles_counter, coinc_counter, coinc):
            try:
                m.stop()
            except Exception:
                pass
    return int(singles[0]), int(singles[1]), int(coincidences[0])


def read_angle(dev, addr):
    """Actual angle [deg, 0-360) reported by the motor's encoder, or None on a bad reply."""
    ret = dev.pos(addr)
    if dev.ispos(ret) and ret[0] == addr:
        return dev.pos2deg(addr, ret) % 360
    print(f'  WARNING: position query of addr {addr} got unexpected reply {ret!r}')
    return None


def angle_dev(actual, commanded):
    """Signed deviation actual - commanded [deg], wrapped to [-180, 180)."""
    return None if actual is None else (actual - commanded % 360 + 180) % 360 - 180


def read_back(dev, targets, tol, when):
    """Read every waveplate's actual angle, warn if any is more than tol deg off.

    targets: {name: (addr, commanded absolute angle incl. calibration)}.
    Returns {name_read_deg: actual, name_dev_deg: actual - commanded}.
    """
    out = {}
    for name, (addr, cmd) in targets.items():
        actual = read_angle(dev, addr)
        dev_deg = angle_dev(actual, cmd)
        out[f'{name}_read_deg'] = actual
        out[f'{name}_dev_deg'] = dev_deg
        if dev_deg is None or abs(dev_deg) > tol:
            print(f'  WARNING ({when}): {name} at {actual} deg, commanded {cmd % 360:.3f} deg (dev {dev_deg})')
    return out


def poisson_errors(n, a_key, r_key, n_boot, rng):
    """Std of (sx, sy, sz, purity) from Poisson-resampling the counts in n."""
    keys = ['H', 'V', a_key, r_key]
    samples = rng.poisson(numpy.maximum([n[k] for k in keys], 0), size=(n_boot, len(keys)))
    out = []
    for row in samples:
        if row[0] + row[1] == 0:
            continue
        r = stokes_and_density_matrix(dict(zip(keys, row.astype(float))), a_key, r_key)
        out.append((r['sx'], r['sy'], r['sz'], r['purity']))
    return numpy.std(out, axis=0)


PLATES = ('hwp1', 'qwp0', 'hwp0')
CSV_FIELDS = (['basis', 'qwp0_deg', 'hwp0_deg', 'singles_signal', 'singles_idler', 'coincidences', 'accidentals']
              + [f'{p}_{k}' for k in ('read_deg', 'dev_deg') for p in PLATES]
              + [f'{p}_dev_after_deg' for p in PLATES] + ['t_start_s'])


def measure_tomography(args, dev, tagger, bases, hwp1_angle, csv_path, t0):
    """Count every analysis setting for one (already set) HWP1 angle; rows are also written to csv_path."""
    ch_s, ch_i = args.ch_signal, args.ch_idler
    rows = []
    with open(csv_path, 'w', newline='') as f:  # written row by row, so a crash keeps what was measured
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for label, q, h in bases:
            print(f'Basis {label}: QWP0 -> {q} deg, HWP0 -> {h} deg')
            move_and_settle(dev, args.addr_qwp0, q + args.cal_qwp0, args.settle)
            move_and_settle(dev, args.addr_hwp0, h + args.cal_hwp0, args.settle)
            targets = {'hwp1': (args.addr_hwp1, hwp1_angle + args.cal_hwp1),
                       'qwp0': (args.addr_qwp0, q + args.cal_qwp0),
                       'hwp0': (args.addr_hwp0, h + args.cal_hwp0)}
            pos = read_back(dev, targets, args.pos_tol, 'after move')
            print('  read-back dev [deg]: ' + '  '.join(f'{p}={pos[p + "_dev_deg"]:+.3f}' for p in PLATES
                                                     if pos[p + '_dev_deg'] is not None))
            t_start = time.time() - t0
            s, i, c = measure_counts(tagger, ch_s, ch_i, args.window, args.exposure)
            after = read_back(dev, targets, args.pos_tol, 'after exposure')  # catches drift / bumps during counting
            acc = 2 * args.window / PS * s * i / args.exposure
            print(f'  singles s={s} ({s / args.exposure:.0f}/s)  i={i} ({i / args.exposure:.0f}/s)   '
                  f'coincidences={c} ({c / args.exposure:.0f}/s)   accidentals~{acc:.1f}')
            row = {'basis': label, 'qwp0_deg': q, 'hwp0_deg': h, 'singles_signal': s,
                   'singles_idler': i, 'coincidences': c, 'accidentals': acc, **pos,
                   **{f'{p}_dev_after_deg': after[f'{p}_dev_deg'] for p in PLATES},
                   't_start_s': round(t_start, 2)}
            rows.append(row)
            w.writerow(row)
            f.flush()
    return rows


def analyse_and_save(args, name, base, hwp1_angle, rows, delay_idler, rng):
    """Reconstruct the state(s) from rows, print, write base.json; returns {key: (res, err)}."""
    n = {r['basis']: r['coincidences'] - (r['accidentals'] if args.subtract_accidentals else 0) for r in rows}
    results = {}
    for key, title, a_key, r_key in (('corrected', 'Mirror-corrected (state before the mirror)', 'A', 'R'),
                                     ('raw', 'Raw (state after the mirror, no correction)', 'A_raw', 'R_raw')):
        if key == 'raw' and not args.raw:
            continue
        res = stokes_and_density_matrix(n, a_key, r_key)
        err = poisson_errors(n, a_key, r_key, args.n_boot, rng)
        results[key] = (res, err)
        print(f'\n--- HWP1 = {hwp1_angle} deg: {title} ---')
        print(f'n_H={n["H"]:.1f}  n_V={n["V"]:.1f}  n_A={n[a_key]:.1f}  n_R={n[r_key]:.1f}  coincidences')
        print(f'Sx={res["sx"]:.4f}+/-{err[0]:.4f}  Sy={res["sy"]:.4f}+/-{err[1]:.4f}  Sz={res["sz"]:.4f}+/-{err[2]:.4f}')
        print(f'Purity Tr(rho^2) = {res["purity"]:.4f} +/- {err[3]:.4f}')
        print(f'rho eigenvalues: {res["eigvals"]}  (a negative one = unphysical, from noise)')
        print('rho =')
        print(res['rho'])

    summary = {
        'name': name,
        'hwp1_angle_deg': hwp1_angle,
        'calibration_deg': {'hwp1': args.cal_hwp1, 'qwp0': args.cal_qwp0, 'hwp0': args.cal_hwp0},
        'mirror_phase_deg': args.mirror_phase,
        'tagger': {'address': args.tagger_address, 'ch_signal': args.ch_signal, 'ch_idler': args.ch_idler,
                   'trigger_level_V': args.trigger_level, 'delay_signal_ps': args.delay_signal,
                   'delay_idler_ps': delay_idler, 'window_ps': args.window, 'exposure_s': args.exposure},
        'session_hwp1_angles_deg': args.hwp1_angle,
        'accidentals_subtracted': args.subtract_accidentals,
        'measurements': rows,
    }
    for key, (res, err) in results.items():
        summary[key] = {
            'stokes': {'sx': res['sx'], 'sy': res['sy'], 'sz': res['sz']},
            'stokes_err': {'sx': err[0], 'sy': err[1], 'sz': err[2]},
            'purity': res['purity'], 'purity_err': err[3],
            'rho_real': res['rho'].real.tolist(), 'rho_imag': res['rho'].imag.tolist(),
            'rho_eigvals': res['eigvals'].tolist(),
        }
    with open(base + '.json', 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f'\nSaved counts to {base}.csv and summary to {base}.json')
    return results


def main():
    args = parse_args()

    name = (args.name or input('Please enter a name for this run: ').strip() or 'tomography_sp').replace(' ', '_')
    os.makedirs(SP_DATA_DIR, exist_ok=True)
    angles = args.hwp1_angle
    # one angle: <name>.csv as before; several: <name>_<k>_hwp1_<angle>.csv (k keeps a repeated angle apart)
    run_bases = ([os.path.join(SP_DATA_DIR, name)] if len(angles) == 1 else
                 [os.path.join(SP_DATA_DIR, f'{name}_{k:02d}_hwp1_{a:g}') for k, a in enumerate(angles)])

    bases = build_bases(args.mirror_phase)
    if args.raw:
        bases += [(label + '_raw', q, h) for label, q, h in RAW_BASES if label in ('A', 'R')]
    print(f'Mirror phase {args.mirror_phase} deg; analysis settings: {bases}')

    print(f'Connecting to Time Tagger at {args.tagger_address} ...')
    tagger = TimeTagger.createTimeTaggerNetwork(args.tagger_address)
    try:
        print(f'Connected! Available channels: {sorted(ch for ch in tagger.getChannelList() if ch >= 0)}')
        ch_s, ch_i = args.ch_signal, args.ch_idler
        for ch, delay in ((ch_s, args.delay_signal), (ch_i, args.delay_idler)):
            tagger.setTriggerLevel(channel=ch, voltage=args.trigger_level)
            set_delay(tagger, ch, delay)
        delay_idler = args.delay_idler
        if args.find_delay:
            delay_idler = find_and_set_delay(tagger, ch_s, ch_i, args.delay_idler)
        print(f'Delays: signal {args.delay_signal:.0f} ps, idler {delay_idler:.0f} ps')

        addrs = [args.addr_hwp1, args.addr_qwp0, args.addr_hwp0]
        print(f'Connecting to elliptec bus on {args.motor_device}, addresses {addrs} ...')
        dev = elliptec.Elliptec(dev=args.motor_device, addrs=addrs, home=True, freq=True)
        dev.ser.timeout = args.move_timeout

        runs = []  # (hwp1 angle, output base, rows)
        t0 = time.time()
        for angle, base in zip(angles, run_bases):
            print(f'\n=== Setting HWP1 to {angle} deg (+ {args.cal_hwp1} deg calibration) ===')
            move_and_settle(dev, args.addr_hwp1, angle + args.cal_hwp1, args.settle)
            runs.append((angle, base, measure_tomography(args, dev, tagger, bases, angle, base + '.csv', t0)))
    finally:
        TimeTagger.freeTimeTagger(tagger)

    rng = numpy.random.default_rng()
    all_results = [(angle, base, analyse_and_save(args, name, base, angle, rows, delay_idler, rng))
                   for angle, base, rows in runs]

    if len(runs) > 1:
        # linear-polarisation angle on the sphere: psi = atan2(sx, sz)/2 (sx>0 = D); an ideal HWP1 at a gives |psi| = 2a
        print('\n=== Session summary (mirror-corrected) ===')
        print(f'{"HWP1":>6} {"Sx":>15} {"Sy":>15} {"Sz":>15} {"purity":>7} {"psi=atan2(Sx,Sz)/2":>19}')
        for angle, _, results in all_results:
            res, err = results['corrected']
            psi = numpy.degrees(numpy.arctan2(res['sx'], res['sz'])) / 2
            print(f'{angle:6g} {res["sx"]:+.3f}+/-{err[0]:.3f} {res["sy"]:+.3f}+/-{err[1]:.3f} '
                  f'{res["sz"]:+.3f}+/-{err[2]:.3f} {res["purity"]:7.3f} {psi:+19.2f}')

    if not args.no_plot:  # at the end: plt.show() in plot_poincare blocks in interactive backends
        for _, base, results in all_results:
            plot_poincare([(r['sx'], r['sy'], r['sz']) for r, _ in results.values()], base)


if __name__ == '__main__':
    main()
