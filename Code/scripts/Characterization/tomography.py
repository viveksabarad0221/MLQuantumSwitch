#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-qubit polarization state tomography with a single detector.

Setup (order along the beam path):

    collimator -> PBS -> HWP1 -> QWP0 -> HWP0 -> PBS -> powermeter
                          ^^^^     ^^^^^^^^^^^^
                       state prep   analysis / tomography

HWP1 is held at a fixed, user-specified angle for the whole run: it just
sets the input polarization once. QWP0 and HWP0 are then stepped through
six settings while the powermeter reads the PBS transmission port. The
reflection port is not monitored (only one detector is available), which
is why this is "slightly different" from a two-detector scheme -- but it
turns out not to matter: with the analysis waveplates in front of a
polarizer that always transmits |H>, the measured power for a given
(q, h) setting of (QWP0, HWP0) is

    P(q, h)  ~  |<H| HWP0(h) QWP0(q) |psi>|^2

which is exactly a projective measurement onto some target state |t>,
provided HWP0(h) QWP0(q) |t> = |H> (up to a global phase). Working that
out with Jones calculus for six target states gives the settings in
BASES below. Because opposite bases are complementary
(I_H + I_V = I_D + I_A = I_L + I_R = total flux, up to non-idealities),
the Stokes vector can be recovered from these six single-port readings
without ever needing the reflection port.

Convention used throughout:
    |H> = (1, 0),  |V> = (0, 1)
    |D> = (|H> + |V>) / sqrt(2),   |A> = (|H> - |V>) / sqrt(2)
    |R> = (|H> - i|V>) / sqrt(2),  |L> = (|H> + i|V>) / sqrt(2)

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

import numpy
import elliptec
from elliptec import ReportedError
from ThorlabsPM100 import ThorlabsPM100, USBTMC

# --------------------------------------------------------------------------
# Analysis settings: (label, QWP0 angle q [deg], HWP0 angle h [deg])
# Derived by requiring HWP0(h) @ QWP0(q) |target> = |H> (up to phase).
# --------------------------------------------------------------------------
BASES = [
    ("H", 0.0, 0.0),
    ("V", 0.0, 45.0),
    ("D", 45.0, 22.5),
    ("A", 45.0, 67.5),
    ("L", 45.0, 0.0),
    ("R", 45.0, 45.0),
]

PAULI_X = numpy.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = numpy.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = numpy.array([[1, 0], [0, -1]], dtype=complex)
IDENT = numpy.eye(2, dtype=complex)

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
    parser.add_argument('--addr-hwp1', type=str, default='0', help='Elliptec address of HWP1 (state prep)')
    parser.add_argument('--addr-qwp0', type=str, default='1', help='Elliptec address of QWP0 (analysis)')
    parser.add_argument('--addr-hwp0', type=str, default='2', help='Elliptec address of HWP0 (analysis)')
    parser.add_argument('--cal-hwp1', type=float, default=132.42,
                        help='Calibration offset [deg] added to HWP1 angle (fast-axis zero from prior characterization)')
    parser.add_argument('--cal-qwp0', type=float, default=90.01,
                        help='Calibration offset [deg] added to QWP0 angles')
    parser.add_argument('--cal-hwp0', type=float, default=75.30,
                        help='Calibration offset [deg] added to HWP0 angles')
    parser.add_argument('-s', '--hwp1-angle', type=float, default=None, required=True,
                        help='Fixed HWP1 angle [deg] setting the input polarization for this run')
    parser.add_argument('--repeats', type=int, default=5, help='Number of powermeter readings averaged per basis setting')
    parser.add_argument('--settle', type=float, default=0.3, help='Settle time [s] after each motor move, before reading')
    parser.add_argument('--move-timeout', type=float, default=6.0,
                        help='Serial read timeout [s] while waiting for a move-complete reply from the elliptec bus. '
                             'Must comfortably exceed the slowest single-axis rotation, or moveabsolute() will '
                             'time out mid-move and resend the command, desyncing the reply stream.')
    parser.add_argument('--pm-average-count', type=int, default=100, help='Powermeter hardware averaging count')
    parser.add_argument('--dark', type=float, default=0.0, help='Dark/background power [mW] subtracted from each reading')
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


def stokes_and_density_matrix(power):
    # power: dict label -> mean power [mW]
    s0_hv = power["H"] + power["V"]
    s0_da = power["D"] + power["A"]
    s0_lr = power["L"] + power["R"]
    s0 = (s0_hv + s0_da + s0_lr) / 3.0

    sz = (power["H"] - power["V"]) / s0
    sx = (power["D"] - power["A"]) / s0
    sy = (power["L"] - power["R"]) / s0

    rho = 0.5 * (IDENT + sx * PAULI_X + sy * PAULI_Y + sz * PAULI_Z)
    purity = 0.5 * (1 + sx**2 + sy**2 + sz**2)
    eigvals = numpy.linalg.eigvalsh(rho)
    return {
        "s0_hv": s0_hv, "s0_da": s0_da, "s0_lr": s0_lr, "s0": s0,
        "sx": sx, "sy": sy, "sz": sz,
        "rho": rho, "purity": purity, "eigvals": eigvals,
    }


def plot_poincare(sx, sy, sz, savepath_base):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u, v = numpy.mgrid[0:2 * numpy.pi:40j, 0:numpy.pi:20j]
    xs = numpy.cos(u) * numpy.sin(v)
    ys = numpy.sin(u) * numpy.sin(v)
    zs = numpy.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.5, alpha=0.5)

    ax.scatter([sx], [sy], [sz], color='red', s=60, depthshade=False)
    ax.plot([0, sx], [0, sy], [0, sz], color='red', linewidth=1.5)

    ax.text(0, 0, 1.15, 'L', ha='center')
    ax.text(0, 0, -1.15, 'R', ha='center')
    ax.text(1.15, 0, 0, 'D', ha='center')
    ax.text(-1.15, 0, 0, 'A', ha='center')
    ax.text(0, 1.15, 0, 'H', ha='center')
    ax.text(0, -1.15, 0, 'V', ha='center')

    ax.set_xlabel('Sx (D/A)')
    ax.set_ylabel('Sy (L/R)')
    ax.set_zlabel('Sz (H/V)')
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    plt.savefig(savepath_base + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(savepath_base + '.png', format='png', bbox_inches='tight', pad_inches=0)
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

    power = {}
    power_std = {}

    try:
        with open(csv_path, mode='w', newline='\n') as f:
            fwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fwriter.writerow(['basis', 'qwp0_deg', 'hwp0_deg', 'power_mW', 'power_std_mW'])

            for label, q, h in BASES:
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

    print()
    print(f'S0 (H+V / D+A / L+R): {result["s0_hv"]:.4f} / {result["s0_da"]:.4f} / {result["s0_lr"]:.4f} mW '
          f'(consistency check -- should roughly agree)')
    print(f'Stokes vector: Sx={result["sx"]:.4f}  Sy={result["sy"]:.4f}  Sz={result["sz"]:.4f}')
    print(f'Purity Tr(rho^2) = {result["purity"]:.4f}  (1.0 = pure state, 0.5 = maximally mixed)')
    print(f'rho eigenvalues: {result["eigvals"]}')
    print('rho =')
    print(result['rho'])

    summary = {
        'name': name,
        'hwp1_angle_deg': args.hwp1_angle,
        'calibration_deg': {'hwp1': args.cal_hwp1, 'qwp0': args.cal_qwp0, 'hwp0': args.cal_hwp0},
        'power_mW': power,
        'power_std_mW': power_std,
        'stokes': {'sx': result['sx'], 'sy': result['sy'], 'sz': result['sz']},
        's0_check_mW': {'hv': result['s0_hv'], 'da': result['s0_da'], 'lr': result['s0_lr']},
        'purity': result['purity'],
        'rho_real': result['rho'].real.tolist(),
        'rho_imag': result['rho'].imag.tolist(),
        'rho_eigvals': result['eigvals'].tolist(),
    }
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f'\nSaved raw data to {csv_path}')
    print(f'Saved summary to {json_path}')

    if not args.no_plot:
        plot_poincare(result['sx'], result['sy'], result['sz'], plot_path_base)


if __name__ == '__main__':
    main()
