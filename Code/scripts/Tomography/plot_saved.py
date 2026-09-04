#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the Poincare sphere for a tomography run already saved by tomography.py.

Standalone (no elliptec/ThorlabsPM100 needed) -- reads sx, sy, sz straight out
of the saved .json summary.

Usage:
    python plot_saved.py test1-hwp1-0
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from qutip import Bloch

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'Tomography'))


def plot_poincare(sx, sy, sz, savepath_base):
    # +x=D/-x=A, +y=L/-y=R, +z=H/-z=V (see tomography.stokes_and_density_matrix)
    b = Bloch()
    b.xlabel = ['D', 'A']
    b.ylabel = ['L', 'R']
    b.zlabel = ['H', 'V']
    b.vector_color = ['r']
    b.add_vectors([sx, sy, sz])
    b.render()
    b.fig.savefig(savepath_base + '.pdf', bbox_inches='tight', pad_inches=0.1)
    b.fig.savefig(savepath_base + '.png', bbox_inches='tight', pad_inches=0.1)
    # b.show() alone doesn't block, so a plain `python script.py` run would
    # exit (and close the window) before you get a chance to drag-rotate it.
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='Run name (basename of the .json in Data/Tomography)')
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, args.name + '.json')) as f:
        summary = json.load(f)

    stokes = summary['stokes']
    print(f"sx={stokes['sx']:.4f}  sy={stokes['sy']:.4f}  sz={stokes['sz']:.4f}  "
          f"purity={summary['purity']:.4f}")

    plot_poincare(stokes['sx'], stokes['sy'], stokes['sz'],
                  os.path.join(DATA_DIR, args.name))


if __name__ == '__main__':
    main()
