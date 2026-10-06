#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TEMPORARY: find out why HWP1 moves during a tomography sweep.

Repeats exactly the motor sequence of tomography.py (same driver, same
addresses, calibrations and analysis settings), but without the
powermeter, and logs:

  * which elliptec.py / tomography.py files were actually imported,
  * model + serial number of the mount at every address,
  * every byte written to / read from the elliptec bus, with a timestamp,
    flagging replies that come from another address than the one just
    commanded and bytes left over in the input buffer (both signs of
    the reply stream getting out of sync),
  * the position of all three mounts after every single move, flagging
    any move of HWP1 away from its set angle.

With --identify it first wiggles each address (3 x +-45 deg, after you press
Enter and a countdown) and asks which mount moved (checks the address mapping).

Usage (same -d / --addr-* / --cal-* / --mirror-phase / --raw options as
tomography.py):

    python debug_motor_moves.py -d COM5 -s 22.5 --identify

Then send back the log file it writes (path printed at the end).
"""

import argparse
import datetime
import os
import sys
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tomography  # also puts the Characterization dir (elliptec.py) on sys.path
import elliptec

T0 = time.time()
STATE = {'last_addr': None}

# Written line by line, so an interrupted run (Ctrl+C) still leaves a log.
_LOG_DIR = os.path.join(tomography.DATA_DIR, 'debug')
os.makedirs(_LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(_LOG_DIR, f'motor_debug_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
_LOG_FILE = open(LOG_PATH, 'w')


def log(line):
    line = f'[{time.time() - T0:8.3f}s] {line}'
    print(line)
    _LOG_FILE.write(line + '\n')
    _LOG_FILE.flush()


class LoggingSerial(serial.Serial):
    """pyserial port that logs all traffic and flags out-of-sync replies."""

    def write(self, data):
        waiting = self.in_waiting
        if waiting:
            log(f'  !! {waiting} unread byte(s) still in input buffer before this write (stale reply?)')
        text = data.decode(errors='replace') if isinstance(data, bytes) else str(data)
        STATE['last_addr'] = text[:1]
        log(f'  TX {text!r}')
        return super().write(data)

    def readline(self, *args, **kwargs):
        start = time.time()
        data = super().readline(*args, **kwargs)
        text = data.decode(errors='replace')
        note = ''
        if text == '':
            note = f'  !! TIMEOUT after {time.time() - start:.2f}s (driver may resend the command)'
        elif text[:1] != STATE['last_addr']:
            note = f'  !! reply from address {text[:1]!r}, but last command went to {STATE["last_addr"]!r}'
        log(f'  RX {text!r}{note}')
        return data


def read_positions(dev, names):
    pos = {}
    for addr, name in names.items():
        ret = dev.pos(addr)
        if dev.ispos(ret) and ret[0] == addr:
            pos[name] = dev.pos2deg(addr, ret) % 360
        else:
            pos[name] = None
            log(f'  !! position query of {name} (addr {addr}) got unexpected reply {ret!r}')
    log('  POSITIONS: ' + '  '.join(f'{n}={p:.2f}' if p is not None else f'{n}=??' for n, p in pos.items()))
    return pos


def check_hwp1(pos, expected, step):
    p = pos['HWP1']
    if p is None:
        return
    dev_deg = (p - expected + 180) % 360 - 180
    if abs(dev_deg) > 0.2:
        log(f'  !!!!!! HWP1 is at {p:.2f} deg, expected {expected:.2f} deg (off by {dev_deg:+.2f}) after: {step}')


def main():
    parser = argparse.ArgumentParser(description='Debug unexpected HWP1 motion (temporary).')
    parser.add_argument('-d', '--motor-device', type=str, required=True)
    parser.add_argument('-s', '--hwp1-angle', type=float, required=True)
    parser.add_argument('--addr-hwp1', type=str, default='0')
    parser.add_argument('--addr-qwp0', type=str, default='1')
    parser.add_argument('--addr-hwp0', type=str, default='2')
    parser.add_argument('--cal-hwp1', type=float, default=132.42)
    parser.add_argument('--cal-qwp0', type=float, default=37.85)
    parser.add_argument('--cal-hwp0', type=float, default=57.35)
    parser.add_argument('--mirror-phase', type=float, default=tomography.MIRROR_PHASE_DEG)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--settle', type=float, default=0.3)
    parser.add_argument('--move-timeout', type=float, default=6.0)
    parser.add_argument('--sweeps', type=int, default=2, help='How many times to repeat the basis sweep')
    parser.add_argument('--identify', action='store_true',
                        help='Nudge each address by +20 deg and ask which mount moved')
    args = parser.parse_args()

    names = {args.addr_hwp1: 'HWP1', args.addr_qwp0: 'QWP0', args.addr_hwp0: 'HWP0'}
    if len(names) != 3:
        log(f'!! addresses are not distinct: hwp1={args.addr_hwp1} qwp0={args.addr_qwp0} hwp0={args.addr_hwp0}')

    log(f'python {sys.version.split()[0]} on {sys.platform}; argv = {sys.argv}')
    log(f'elliptec imported from:   {elliptec.__file__}')
    log(f'tomography imported from: {tomography.__file__}')

    # Make the driver open the port through the logging class, so the
    # init sequence (info, frequency search, homing) is logged as well.
    elliptec.serial.Serial = LoggingSerial

    log('=== init: elliptec.Elliptec(home=True, freq=True), as in tomography.py ===')
    dev = elliptec.Elliptec(dev=args.motor_device, addrs=list(names), home=True, freq=True)
    dev.ser.timeout = args.move_timeout
    for addr, name in names.items():
        info = dev.info[addr]
        log(f'  {name}: addr {addr}  ELL{info["partnumber"]}  serial {info["serialnumber"]}')
    read_positions(dev, names)

    if args.identify:
        log('=== identify: wiggling each address in turn ===')
        print('\nFor each address, one mount will swing +45 deg / back, 3 times, after a countdown.')
        print('Watch all three mounts (HWP1 = state prep, QWP0 and HWP0 = after the mirror).')
        for addr, name in names.items():
            before = read_positions(dev, names)
            input(f'\n  >>> Press Enter, then watch the mounts: address {addr} (expected to be {name}) will wiggle ...')
            for n in (3, 2, 1):
                print(f'      {n} ...')
                time.sleep(1.0)
            log(f'  WIGGLING ONLY addr {addr} ({name}): 3 x (+45 deg, back)')
            print('      >>> MOVING NOW <<<')
            for _ in range(3):
                dev.moverelative(addr, 45)
                time.sleep(0.5)
                dev.moverelative(addr, -45)
                time.sleep(0.5)
            print('      >>> DONE <<<')
            read_positions(dev, names)
            answer = input('  Which mount wiggled?  1 = HWP1 (prep)   2 = QWP0   3 = HWP0   '
                           '4 = more than one   5 = none / did not see  : ').strip()
            label = {'1': 'HWP1', '2': 'QWP0', '3': 'HWP0', '4': 'several', '5': 'none/unseen'}.get(answer, answer)
            log(f'  USER: addr {addr} (expected {name}) -> physically wiggled: {label}')
            if before[name] is not None:
                dev.moveabsolute(addr, before[name])
            time.sleep(0.5)

    hwp1_target = (args.hwp1_angle + args.cal_hwp1) % 360
    log(f'=== set HWP1 to {args.hwp1_angle} + {args.cal_hwp1} = {hwp1_target:.2f} deg ===')
    tomography.move_and_settle(dev, args.addr_hwp1, args.hwp1_angle + args.cal_hwp1, args.settle)
    check_hwp1(read_positions(dev, names), hwp1_target, 'setting HWP1')

    bases = tomography.build_bases(args.mirror_phase)
    if args.raw:
        bases += [(label + '_raw', q, h) for label, q, h in tomography.RAW_BASES if label in ('A', 'R')]
    log(f'analysis settings: {bases}')

    for sweep in range(args.sweeps):
        for label, q, h in bases:
            log(f'=== sweep {sweep + 1}, basis {label}: QWP0 -> {q} (+{args.cal_qwp0}), HWP0 -> {h} (+{args.cal_hwp0}) ===')
            tomography.move_and_settle(dev, args.addr_qwp0, q + args.cal_qwp0, args.settle)
            check_hwp1(read_positions(dev, names), hwp1_target, f'basis {label} QWP0 move')
            tomography.move_and_settle(dev, args.addr_hwp0, h + args.cal_hwp0, args.settle)
            check_hwp1(read_positions(dev, names), hwp1_target, f'basis {label} HWP0 move')
            time.sleep(1.0)  # roughly where the powermeter would read
            check_hwp1(read_positions(dev, names), hwp1_target, f'basis {label} idle 1 s')

    dev.close()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        log('=== interrupted by user ===')
    finally:
        _LOG_FILE.close()
        print(f'\nLog written to {LOG_PATH} -- please send it back.')
