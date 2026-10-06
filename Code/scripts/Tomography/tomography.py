#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-qubit polarization state tomography with a single detector.

Follows James, Kwiat, Munro & White, "Measurement of qubits",
Phys. Rev. A 64, 052312 (2001), Sec. II.A, Eq. (2.1)-(2.3).

Setup (order along the beam path):

    collimator -> PBS -> HWP1 -> mirror -> QWP0 -> HWP0 -> PBS -> powermeter
                          ^^^^               ^^^^^^^^^^^^
                       state prep         analysis / tomography

HWP1 is held at a fixed, user-specified angle for the whole run: it just
sets the input polarization once. QWP0 and HWP0 are then stepped through
four settings while the powermeter reads the PBS transmission port. With
the analysis waveplates in front of a polarizer that always transmits
|H>, the measured power for a given (q, h) setting of (QWP0, HWP0) is

    P(q, h)  ~  |<H| HWP0(h) QWP0(q) M |psi>|^2

where M is the mirror (see the note below). This is a projective
measurement onto some target state |t> (of the state *before* the
mirror), provided HWP0(h) QWP0(q) M |t> = |H> (up to a global phase).
Working that out with Jones calculus for the paper's four target states
(H, V, the 45 deg state (|H>-|V>)/sqrt(2), and R) gives the settings
built by build_bases() below.

Note on the mirror
------------------
There is a mirror between the preparation stage (HWP1) and the
tomography stage (QWP0, HWP0). Its s and p axes are along H/V, so it
leaves |H> and |V> alone but adds a relative phase to any H-V
superposition (checked with a polarimeter placed after the mirror):

    M = diag(1, exp(i*phi))

phi includes the ~180 deg from the reflection itself (which takes D to
A) plus ~30 deg of retardance from the coating. Without correction, a D
input (HWP1 at 22.5 deg) was reconstructed as an elliptical state near A
(tests 4-18, see Data/Tomography/indexing.txt).

We do not undo M on the reconstructed rho. Instead the analysis
waveplates absorb it: for each target |t> we pick (q, h) such that
HWP0(h) QWP0(q) M |t> = |H>. Since M is diagonal, H and V need no change.
M|A> and M|R> are still on the equator of the Bloch sphere (M is a
rotation about the H/V axis), so a QWP at +-45 deg makes them linear and
HWP0 then rotates them onto H. The reconstructed state is therefore the
one leaving HWP1, i.e. before the mirror.

phi = 212.05 deg (MIRROR_PHASE_DEG) was fitted to test6 by modelling each
run with the analysis angles it actually used. That phi then predicts
tests 9, 12, 15, 18 (including the sign flip of Sy in tests 12/15, where R
was measured with QWP0 at 135 deg). Fitting those runs individually
gives 208.7, 209.6, 208.6, 206.5 deg, so phi is known to roughly +-3 deg.
The spread is about what you'd expect from the n0 = (n_H + n_V)/2
normalisation (purity > 1 in some runs). Override with --mirror-phase if
the mirror is realigned. --raw also measures the uncorrected A/R
settings (RAW_BASES) and reports the as-measured state after the mirror.

The paper's n0 is obtained with a 50%-transmission, polarization-
independent filter: n0 = (N/2)(<H|rho|H> + <V|rho|V>). We don't have
such a filter, so per Eq. (2.1) we get the same quantity by measuring H
and V separately and averaging: n0 = (n_H + n_V) / 2. That is the only
difference from the paper.

Convention used throughout:
    |H> = (1, 0),  |V> = (0, 1)
    |D> = (|H> + |V>) / sqrt(2),   |A> = (|H> - |V>) / sqrt(2)
    |R> = (|H> - i|V>) / sqrt(2),  |L> = (|H> + i|V>) / sqrt(2)
The paper's 45 deg state |Dbar> = (|H> - |V>)/sqrt(2) is our |A>.

Only the analysis stage (QWP0, HWP0) is swept here. HWP1 is moved once
at the start and left alone -- this file is meant to validate the
tomography stage itself before the birefringent (dephasing) crystal is
inserted between HWP1 and QWP0. The time-tagger / single-photon side of
that experiment is out of scope for this script.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, 'C:\\Users\\Kalvarienberg\\OneDrive\\Desktop\\MLQS\\MLQuantumSwitch\\Code\\scripts\\Characterization\\')

import numpy
import elliptec
from elliptec import ReportedError
from ThorlabsPM100 import ThorlabsPM100, USBTMC

# --------------------------------------------------------------------------
# Uncorrected analysis settings: (label, QWP0 angle q [deg], HWP0 angle h [deg])
# Derived by requiring HWP0(h) @ QWP0(q) |target> = |H> (up to phase), i.e.
# ignoring the mirror. These are the four projections of paper Eq. (2.1):
# H, V, Dbar (45 deg, our A), and R. Measuring with these gives the state
# after the mirror. Only used directly with --raw; build_bases() derives
# the mirror-corrected settings from them.
# --------------------------------------------------------------------------
RAW_BASES = [
    ("H", 0.0, 0.0),
    ("V", 0.0, 45.0),
    ("A", -45.0, 67.5),
    ("R", -45.0, 0.0),
]

# Relative phase [deg] the mirror adds to |V> relative to |H>,
# M = diag(1, exp(i*phi)). Fitted from test6, see the note in the docstring.
MIRROR_PHASE_DEG = 212.05

PAULI_X = numpy.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = numpy.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = numpy.array([[1, 0], [0, -1]], dtype=complex)
IDENT = numpy.eye(2, dtype=complex)

KET_H = numpy.array([1, 0], dtype=complex)
TARGETS = {
    "H": KET_H,
    "V": numpy.array([0, 1], dtype=complex),
    "A": numpy.array([1, -1], dtype=complex) / numpy.sqrt(2),
    "R": numpy.array([1, -1j], dtype=complex) / numpy.sqrt(2),
}


def waveplate(angle_deg, retardance):
    # Jones matrix of a waveplate with fast axis at angle_deg from H. Same
    # convention RAW_BASES was derived in: QWP(0) = diag(1, i), HWP(0) = diag(1, -1).
    t = numpy.deg2rad(angle_deg)
    rot = numpy.array([[numpy.cos(t), -numpy.sin(t)], [numpy.sin(t), numpy.cos(t)]])
    return rot @ numpy.diag([1, numpy.exp(1j * retardance)]) @ rot.T


def qwp(angle_deg):
    return waveplate(angle_deg, numpy.pi / 2)


def hwp(angle_deg):
    return waveplate(angle_deg, numpy.pi)


def mirror(phase_deg):
    return numpy.diag([1, numpy.exp(1j * numpy.deg2rad(phase_deg))])


def analysis_setting(state):
    # (q, h) with HWP0(h) QWP0(q) |state> = |H> up to phase. QWP0 goes along
    # the polarization ellipse's major axis (which makes the state linear),
    # then HWP0 at half the resulting linear angle rotates it onto H.
    a, b = state
    s1 = abs(a)**2 - abs(b)**2
    s2 = 2 * numpy.real(numpy.conj(a) * b)
    q = numpy.degrees(0.5 * numpy.arctan2(s2, s1))
    v = qwp(q) @ state
    v = v * numpy.exp(-1j * numpy.angle(v[numpy.argmax(abs(v))]))
    h = numpy.degrees(numpy.arctan2(v[1].real, v[0].real)) / 2
    return float(q), float(h)


def build_bases(mirror_phase_deg):
    # Mirror-corrected settings: project onto M|t> so the measurement is
    # onto |t> for the state before the mirror. M is diagonal, so H and V
    # keep their RAW_BASES settings; only A and R are re-solved.
    m = mirror(mirror_phase_deg)
    bases = []
    for label, q, h in RAW_BASES:
        if label in ("A", "R"):
            q, h = analysis_setting(m @ TARGETS[label])
        fidelity = abs(KET_H.conj() @ hwp(h) @ qwp(q) @ m @ TARGETS[label])**2
        assert fidelity > 1 - 1e-9, f'basis {label}: fidelity {fidelity} with mirror phase {mirror_phase_deg}'
        bases.append((label, round(q, 3), round(h, 3)))
    return bases

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'Tomography'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single-qubit polarization tomography: fix HWP1 (state prep), '
                    'sweep QWP0/HWP0 (analysis) in front of a single-port PBS + powermeter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--name', type=str, default='',
                        help='Name to identify this run (used for output filenames)')
    parser.add_argument('-p', '--powermeter-device', type=str, default='',
                        help='path to PM100D usbtmc device (Linux only)')
    parser.add_argument('-n', '--serial-number', type=str, default='',
                        help='Serial number of powermeter (Windows only)')
    parser.add_argument('-d', '--motor-device', type=str, default='',
                        help='Serial device path for the elliptec bus (HWP1, QWP0, HWP0 daisy-chained on it)')
    parser.add_argument('-s', '--hwp1-angle', type=float, default=None, required=True,
                        help='Fixed HWP1 angle [deg] setting the input polarization for this run')
    parser.add_argument('--addr-hwp1', type=str, default='0', help='Elliptec address of HWP1 (state prep)')
    parser.add_argument('--addr-qwp0', type=str, default='1', help='Elliptec address of QWP0 (analysis)')
    parser.add_argument('--addr-hwp0', type=str, default='2', help='Elliptec address of HWP0 (analysis)')
    parser.add_argument('--cal-hwp1', type=float, default=132.42,
                        help='Calibration offset [deg] added to HWP1 angle (fast-axis zero from prior characterization)')
    parser.add_argument('--cal-qwp0', type=float, default=37.85, #default=38.93, #37.8 (from polarimeter), 38.93 (from characterization plot)
                        help='Calibration offset [deg] added to QWP0 angles')
    parser.add_argument('--cal-hwp0', type=float, default=57.35, #default=57.77, #57.35 (from polarimeter), 57.77 (from characterization plot)
                        help='Calibration offset [deg] added to HWP0 angles')
    parser.add_argument('--repeats', type=int, default=5, help='Number of powermeter readings averaged per basis setting')
    parser.add_argument('--settle', type=float, default=0.3, help='Settle time [s] after each motor move, before reading')
    parser.add_argument('--move-timeout', type=float, default=6.0,
                        help='Serial read timeout [s] while waiting for a move-complete reply from the elliptec bus. '
                             'Must comfortably exceed the slowest single-axis rotation, or moveabsolute() will '
                             'time out mid-move and resend the command, desyncing the reply stream.')
    parser.add_argument('--pm-average-count', type=int, default=100, help='Powermeter hardware averaging count')
    parser.add_argument('--dark', type=float, default=0.0, help='Dark/background power [mW] subtracted from each reading')
    parser.add_argument('--mirror-phase', type=float, default=MIRROR_PHASE_DEG,
                        help='Relative V-vs-H phase [deg] added by the mirror between HWP1 and QWP0; '
                             'the A/R analysis settings are chosen to compensate it (0 = no mirror correction)')
    parser.add_argument('--raw', action='store_true',
                        help='Also measure the uncorrected A/R settings and report the raw (after-mirror) state')
    parser.add_argument('--no-plot', action='store_true', help='Skip the Poincare-sphere plot')
    return parser.parse_args()


def move_and_settle(dev, addr, angle, settle):
    angle = angle % 360
    try:
        dev.moveabsolute(addr, angle)
    except ReportedError as e:
        print(f"Caught sensor error moving addr {addr}: {e}; re-homing and retrying")
        dev.home(addr)
        dev.moveabsolute(addr, angle)
    time.sleep(settle)


def read_power_mw(pm, repeats, dark):
    readings = []
    for _ in range(repeats):
        readings.append(pm.read * 1000 - dark)
    readings = numpy.asarray(readings)
    return float(readings.mean()), float(readings.std())


def stokes_and_density_matrix(power, a_key="A", r_key="R"):
    # power: dict label -> mean power [mW]. Paper Eq. (2.1), with n0 (normally a 50% filter reading) replaced by (n_H + n_V) / 2.
    n0 = (power["H"] + power["V"]) / 2.0
    n1 = power["H"]
    n2 = power[a_key]  # paper's Dbar = (|H>-|V>)/sqrt(2)
    n3 = power[r_key]

    # Paper Eq. (2.2)
    s0 = 2 * n0
    s1 = 2 * (n1 - n0)
    s2 = 2 * (n2 - n0)
    s3 = 2 * (n3 - n0)

    sz = s1 / s0 
    sx = -s2 / s0
    sy = -s3 / s0

    # Paper Eq. (2.3): rho = 1/2 sum_i (S_i/S0) sigma_i (in the |R> / |L> basis), expressed here in the |H>,|V> basis (sigma_1->PAULI_Z, sigma_2->-PAULI_X, sigma_3->-PAULI_Y).
    rho = 0.5 * (IDENT + sx * PAULI_X + sy * PAULI_Y + sz * PAULI_Z)
    purity = 0.5 * (1 + sx**2 + sy**2 + sz**2)
    eigvals = numpy.linalg.eigvalsh(rho)
    return {
        "n0": n0, "n1": n1, "n2": n2, "n3": n3,
        "s0": s0, "s1": s1, "s2": s2, "s3": s3,
        "sx": sx, "sy": sy, "sz": sz,
        "rho": rho, "purity": purity, "eigvals": eigvals,
    }


def plot_poincare(vectors, savepath_base):
    # vectors: list of (sx, sy, sz); first is drawn red, a second (raw) one blue.
    # (sx, sy, sz) poles: +x=D/-x=A, +y=L/-y=R, +z=H/-z=V (see
    # stokes_and_density_matrix). qutip.Bloch draws the sphere and places
    # pole labels itself, so there's no hand-coded text position to get
    # wrong (which is what caused the earlier D/A-vs-L/R-vs-H/V mixup).
    import matplotlib.pyplot as plt
    from qutip import Bloch

    b = Bloch()
    b.xlabel = ['D', 'A']
    b.ylabel = ['L', 'R']
    b.zlabel = ['H', 'V']
    b.vector_color = ['r', 'b']
    for v in vectors:
        b.add_vectors(list(v))
    b.render()
    b.fig.savefig(savepath_base + '.pdf', bbox_inches='tight', pad_inches=0.1)
    b.fig.savefig(savepath_base + '.png', bbox_inches='tight', pad_inches=0.1)
    # b.show() alone doesn't block, so a plain `python script.py` run would
    # exit (and close the window) before you get a chance to drag-rotate it.
    plt.show()


def main():
    args = parse_args()

    if sys.platform != 'linux':
        windows = True
        if args.serial_number == '':
            print('Please provide a powermeter serial number (-n)')
            sys.exit(1)
    else:
        windows = False
        if args.powermeter_device == '':
            print('Please provide a powermeter device path (-p)')
            sys.exit(1)

    if args.motor_device == '':
        print('Please specify the elliptec bus device path (-d)')
        sys.exit(1)

    name = args.name
    if name == '':
        name = input('Please enter a name for this run: ').strip()
        if name == '':
            name = 'tomography'
    name = name.replace(' ', '_')

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, name + '.csv')
    json_path = os.path.join(DATA_DIR, name + '.json')
    plot_path_base = os.path.join(DATA_DIR, name)

    addrs = [args.addr_hwp1, args.addr_qwp0, args.addr_hwp0]
    print(f'Connecting to elliptec bus on {args.motor_device}, addresses {addrs} ...')
    dev = elliptec.Elliptec(dev=args.motor_device, addrs=addrs, home=True, freq=True)
    # elliptec.py hardcodes this to 2s after init, which can be shorter than a
    # real move -- moveabsolute() would then time out mid-rotation and resend
    # the command, desyncing the reply stream. Widen it for the whole sweep.
    dev.ser.timeout = args.move_timeout

    print(f'Setting HWP1 (state prep) to {args.hwp1_angle} deg (+ {args.cal_hwp1} deg calibration) ...')
    move_and_settle(dev, args.addr_hwp1, args.hwp1_angle + args.cal_hwp1, args.settle)

    print('Connecting to powermeter ...')
    if windows:
        import pyvisa as visa
        rm = visa.ResourceManager()
        inst = rm.open_resource('USB0::0x1313::0x8078::' + args.serial_number + '::INSTR')
        inst.read_termination = '\n'
        inst.write_termination = '\n'
        inst.timeout = 1000
    else:
        rm = None
        inst = USBTMC(device=args.powermeter_device)
    pm = ThorlabsPM100(inst=inst)
    pm.sense.average.count = args.pm_average_count

    # Mirror-corrected settings, plus (with --raw) the uncorrected A/R ones
    # under 'A_raw'/'R_raw'. H and V are shared between the two.
    bases = build_bases(args.mirror_phase)
    if args.raw:
        bases += [(label + '_raw', q, h) for label, q, h in RAW_BASES if label in ('A', 'R')]
    print(f'Mirror phase {args.mirror_phase} deg; analysis settings: {bases}')

    power = {}
    power_std = {}

    try:
        with open(csv_path, mode='w', newline='\n') as f:
            fwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fwriter.writerow(['basis', 'qwp0_deg', 'hwp0_deg', 'power_mW', 'power_std_mW'])

            for label, q, h in bases:
                print(f'Basis {label}: QWP0 -> {q} deg, HWP0 -> {h} deg')
                move_and_settle(dev, args.addr_qwp0, q + args.cal_qwp0, args.settle)
                move_and_settle(dev, args.addr_hwp0, h + args.cal_hwp0, args.settle)

                mean_pw, std_pw = read_power_mw(pm, args.repeats, args.dark)
                power[label] = mean_pw
                power_std[label] = std_pw
                print(f'  power = {mean_pw:.6f} +/- {std_pw:.6f} mW')
                fwriter.writerow([label, q, h, mean_pw, std_pw])
    finally:
        if windows and rm is not None:
            rm.close()

    result = stokes_and_density_matrix(power)
    raw = stokes_and_density_matrix(power, 'A_raw', 'R_raw') if args.raw else None

    for title, res in (('Mirror-corrected (state before the mirror)', result),
                       ('Raw (state after the mirror, no correction)', raw)):
        if res is None:
            continue
        print()
        print(f'--- {title} ---')
        print(f'n0={res["n0"]:.4f}  n1={res["n1"]:.4f}  n2={res["n2"]:.4f}  n3={res["n3"]:.4f} mW')
        print(f'Stokes vector: S0={res["s0"]:.4f}  S1={res["s1"]:.4f}  S2={res["s2"]:.4f}  S3={res["s3"]:.4f}')
        print(f'Normalized: Sx={res["sx"]:.4f}  Sy={res["sy"]:.4f}  Sz={res["sz"]:.4f}')
        print(f'Purity Tr(rho^2) = {res["purity"]:.4f}  (1.0 = pure state, 0.5 = maximally mixed)')
        print(f'rho eigenvalues: {res["eigvals"]}')
        print('rho =')
        print(res['rho'])

    summary = {
        'name': name,
        'hwp1_angle_deg': args.hwp1_angle,
        'calibration_deg': {'hwp1': args.cal_hwp1, 'qwp0': args.cal_qwp0, 'hwp0': args.cal_hwp0},
        'mirror_phase_deg': args.mirror_phase,
        'bases': [{'basis': label, 'qwp0_deg': q, 'hwp0_deg': h} for label, q, h in bases],
        'power_mW': power,
        'power_std_mW': power_std,
        'n_mW': {'n0': result['n0'], 'n1': result['n1'], 'n2': result['n2'], 'n3': result['n3']},
        'stokes_unnormalized': {'s0': result['s0'], 's1': result['s1'], 's2': result['s2'], 's3': result['s3']},
        'stokes': {'sx': result['sx'], 'sy': result['sy'], 'sz': result['sz']},
        'purity': result['purity'],
        'rho_real': result['rho'].real.tolist(),
        'rho_imag': result['rho'].imag.tolist(),
        'rho_eigvals': result['eigvals'].tolist(),
    }
    if raw is not None:
        summary['raw'] = {
            'stokes': {'sx': raw['sx'], 'sy': raw['sy'], 'sz': raw['sz']},
            'purity': raw['purity'],
            'rho_real': raw['rho'].real.tolist(),
            'rho_imag': raw['rho'].imag.tolist(),
        }
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f'\nSaved raw data to {csv_path}')
    print(f'Saved summary to {json_path}')

    if not args.no_plot:
        vectors = [(result['sx'], result['sy'], result['sz'])]
        if raw is not None:
            vectors.append((raw['sx'], raw['sy'], raw['sz']))
        plot_poincare(vectors, plot_path_base)


if __name__ == '__main__':
    main()
