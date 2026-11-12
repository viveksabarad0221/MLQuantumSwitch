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
    parser.add_argument('-s', '--hwp1-angle', type=float, required=True, help='Fixed HWP1 angle [deg] (state prep)')
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


def main():
    args = parse_args()

    name = (args.name or input('Please enter a name for this run: ').strip() or 'tomography_sp').replace(' ', '_')
    os.makedirs(SP_DATA_DIR, exist_ok=True)
    base = os.path.join(SP_DATA_DIR, name)

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

        print(f'Setting HWP1 to {args.hwp1_angle} deg (+ {args.cal_hwp1} deg calibration) ...')
        move_and_settle(dev, args.addr_hwp1, args.hwp1_angle + args.cal_hwp1, args.settle)

        rows = []
        fields = ['basis', 'qwp0_deg', 'hwp0_deg', 'singles_signal', 'singles_idler', 'coincidences', 'accidentals']
        with open(base + '.csv', 'w', newline='') as f:  # written row by row, so a crash keeps what was measured
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for label, q, h in bases:
                print(f'Basis {label}: QWP0 -> {q} deg, HWP0 -> {h} deg')
                move_and_settle(dev, args.addr_qwp0, q + args.cal_qwp0, args.settle)
                move_and_settle(dev, args.addr_hwp0, h + args.cal_hwp0, args.settle)
                s, i, c = measure_counts(tagger, ch_s, ch_i, args.window, args.exposure)
                acc = 2 * args.window / PS * s * i / args.exposure
                print(f'  singles s={s} ({s / args.exposure:.0f}/s)  i={i} ({i / args.exposure:.0f}/s)   '
                      f'coincidences={c} ({c / args.exposure:.0f}/s)   accidentals~{acc:.1f}')
                row = {'basis': label, 'qwp0_deg': q, 'hwp0_deg': h, 'singles_signal': s,
                       'singles_idler': i, 'coincidences': c, 'accidentals': acc}
                rows.append(row)
                w.writerow(row)
                f.flush()
    finally:
        TimeTagger.freeTimeTagger(tagger)

    n = {r['basis']: r['coincidences'] - (r['accidentals'] if args.subtract_accidentals else 0) for r in rows}
    rng = numpy.random.default_rng()
    results = {}
    for key, title, a_key, r_key in (('corrected', 'Mirror-corrected (state before the mirror)', 'A', 'R'),
                                     ('raw', 'Raw (state after the mirror, no correction)', 'A_raw', 'R_raw')):
        if key == 'raw' and not args.raw:
            continue
        res = stokes_and_density_matrix(n, a_key, r_key)
        err = poisson_errors(n, a_key, r_key, args.n_boot, rng)
        results[key] = (res, err)
        print(f'\n--- {title} ---')
        print(f'n_H={n["H"]:.1f}  n_V={n["V"]:.1f}  n_A={n[a_key]:.1f}  n_R={n[r_key]:.1f}  coincidences')
        print(f'Sx={res["sx"]:.4f}+/-{err[0]:.4f}  Sy={res["sy"]:.4f}+/-{err[1]:.4f}  Sz={res["sz"]:.4f}+/-{err[2]:.4f}')
        print(f'Purity Tr(rho^2) = {res["purity"]:.4f} +/- {err[3]:.4f}')
        print(f'rho eigenvalues: {res["eigvals"]}  (a negative one = unphysical, from noise)')
        print('rho =')
        print(res['rho'])

    summary = {
        'name': name,
        'hwp1_angle_deg': args.hwp1_angle,
        'calibration_deg': {'hwp1': args.cal_hwp1, 'qwp0': args.cal_qwp0, 'hwp0': args.cal_hwp0},
        'mirror_phase_deg': args.mirror_phase,
        'tagger': {'address': args.tagger_address, 'ch_signal': args.ch_signal, 'ch_idler': args.ch_idler,
                   'trigger_level_V': args.trigger_level, 'delay_signal_ps': args.delay_signal,
                   'delay_idler_ps': delay_idler, 'window_ps': args.window, 'exposure_s': args.exposure},
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

    if not args.no_plot:
        plot_poincare([(r['sx'], r['sy'], r['sz']) for r, _ in results.values()], base)


if __name__ == '__main__':
    main()
