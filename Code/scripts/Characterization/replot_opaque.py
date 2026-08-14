#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Re-plot existing Characterization CSVs (same fit as motorized.py) but save
non-transparent PNG/PDF, overwriting the transparent versions in
Data/Characterization."""

import csv
import glob
import os

import numpy
import scipy.optimize
from scipy.signal import argrelmax
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'Characterization')
DATA_DIR = os.path.normpath(DATA_DIR)

OPTIC_LABELS = {'hwp': 'Half-wave plate', 'qwp': 'Quarter-wave plate', 'polarizer': 'Polarizer'}

pi = numpy.pi


def sinfunc(param, x):
    return param[0] + param[1] * numpy.sin(param[2] * x + param[3])


def sinfuncerr(param, x, y):
    return sinfunc(param, x) - y


def optic_type_for(name):
    lname = name.lower()
    if lname.startswith('hwp'):
        return 'hwp'
    if lname.startswith('qwp'):
        return 'qwp'
    if lname.startswith('polarizer'):
        return 'polarizer'
    raise ValueError(f"Can't infer optic type from name '{name}'")


for csv_path in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
    wpname = os.path.splitext(os.path.basename(csv_path))[0]
    optic_type = optic_type_for(wpname)

    with open(csv_path, 'r', newline='\n') as infile:
        indat = list(csv.reader(infile, delimiter=','))

    header = indat[0][1:]
    angles = numpy.asarray([float(row[0]) for row in indat[1:]])
    wp = numpy.array([[float(v) for v in row[1:]] for row in indat[1:]]).T.tolist()

    plotrange = numpy.arange(0, 320, 0.001)
    freq_guess = 1 / (8 * pi) if optic_type == 'polarizer' else 1 / (4 * pi)

    for i in range(len(wp)):
        parest = [1., max(wp[i]), freq_guess, 0.]
        par, success = scipy.optimize.leastsq(sinfuncerr, parest[:], args=(angles, wp[i]))

        fig, ax1 = plt.subplots()
        ax1.plot(angles, wp[i], linestyle='none', marker='o', ms=3)
        ax1.plot(plotrange, sinfunc(par, plotrange))
        plt.title(OPTIC_LABELS[optic_type] + ' ' + header[i], fontsize=14, fontweight='bold')
        ax1.set_xlabel('Rotation angle [°]', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Power, H polarized (a.u.)', fontsize=12, fontweight='bold')
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

        maximaindices = argrelmax(sinfunc(par, plotrange), order=20)
        maxima = [plotrange[j] for j in maximaindices[0]]
        maxtext = 'Maxima at ' + ', '.join('{0:.2f}°'.format(m) for m in maxima)
        fig.tight_layout()
        ax1.text(0.01, 0.01, maxtext, fontsize='8', color='black', transform=ax1.transAxes)

        pdf_path = os.path.join(DATA_DIR, wpname + '.pdf')
        png_path = os.path.join(DATA_DIR, wpname + '.png')
        print('saving', png_path)
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, transparent=False, facecolor='white', dpi=300)
        plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0, transparent=False, facecolor='white', dpi=300)
        plt.close(fig)
