
# -*- coding: utf-8 -*-

"""
Build a clean optical-circuit diagram matching the hand-drawn sketch.

This script uses the PyOpticalTable module (pyopticaltable.py) to place standard
optical elements and then uses a few small matplotlib helpers to draw the
curved fibre routes and fibre-coil symbols.

Outputs:
  - circuit_diagram.png
  - circuit_diagram.svg

How to run:
  python build_circuit_diagram.py
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

# Ensure local import works if pyopticaltable.py is in the same folder.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pyopticaltable import OpticalTable  # noqa: E402


Point = Tuple[float, float]


# -----------------------------
# Drawing helpers (matplotlib)
# -----------------------------
def bezier(ax, p0: Point, p1: Point, p2: Point, p3: Point, *, color="tab:blue", lw=1.6, ls="-", zorder=1):
    """Cubic Bezier curve between p0 and p3 with control points p1, p2."""
    verts = [p0, p1, p2, p3]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    patch = mpatches.PathPatch(Path(verts, codes), facecolor="none", edgecolor=color, lw=lw, ls=ls, zorder=zorder)
    ax.add_patch(patch)


def polyline(ax, pts: Sequence[Point], *, color="tab:blue", lw=1.6, ls="-", zorder=1):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=zorder)


def fibre_coil(ax, center: Point, *, n=4, radius=0.22, pitch=0.28, angle_deg=0, color="tab:blue", lw=1.4, zorder=1):
    """Draw a little 'coil' symbol (n small loops) roughly along angle_deg."""
    cx, cy = center
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta) * pitch, math.sin(theta) * pitch
    # Start so that the coil is centered about 'center'
    start_x = cx - dx * (n - 1) / 2
    start_y = cy - dy * (n - 1) / 2
    for i in range(n):
        x = start_x + i * dx
        y = start_y + i * dy
        circ = mpatches.Circle((x, y), radius=radius, fill=False, edgecolor=color, lw=lw, zorder=zorder)
        ax.add_patch(circ)


def label(ax, x: float, y: float, s: str, *, size=8, color="teal", ha="center", va="center", rot=0, zorder=20):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, rotation=rot, zorder=zorder)


# -----------------------------
# Semantic helpers (PyOpticalTable)
# -----------------------------
def add_fc(table: OpticalTable, x: float, y: float, *, text: str = "FC", r: float = 0.18):
    """Fibre connector / coupler symbol (small open circle with label)."""
    table.generic_circle(x, y, r, colour="tab:blue", fill=False, label=text, label_pos="top", labelpad=0.18, textcolour="tab:blue", fontsize=7)


def add_wp(table: OpticalTable, x: float, y: float, *, lab: str, angle: float = 90):
    """Generic waveplate / compensator drawn as a transmissive plate."""
    table.transmissive_plate(x, y, size=0.55, angle=angle, colour="k",
                             label=lab, label_pos="top", labelpad=0.15, textcolour="k", fontsize=7, zorder=5)


def add_pbs(table: OpticalTable, x: float, y: float, *, lab: str = "PBS", angle: float = 0, direction: str = "R"):
    """PBS/BS cube."""
    table.beamsplitter_cube(x, y, size=0.7, angle=angle, direction=direction, colour="k",
                            label=lab, label_pos="top", labelpad=0.2, textcolour="k", fontsize=7)


def add_switch(table: OpticalTable, x: float, y: float, *, lab: str = "opt. switch", w: float = 1.2, h: float = 0.7):
    """Optical switch box with a cross."""
    # base box
    table.box(x, y, w, h, angle=0, colour="k", standalone=True, label=lab, label_pos="top", labelpad=0.25, textcolour="k", fontsize=7)
    # cross
    table.ax.plot([x - w/2, x + w/2], [y - h/2, y + h/2], color="k", lw=1.0, zorder=6)
    table.ax.plot([x - w/2, x + w/2], [y + h/2, y - h/2], color="k", lw=1.0, zorder=6)


def add_circulator(table: OpticalTable, x: float, y: float, *, lab: str = "circulator"):
    table.generic_circle(x, y, size=0.32, colour="k", fill=False, label=lab, label_pos="top", labelpad=0.2, textcolour="k", fontsize=7)


def add_detector(table: OpticalTable, x: float, y: float, *, lab: str):
    # draw as a beam dump so fibres appear to terminate into it
    table.beam_dump(x, y, size=0.5, angle=90, colour="k", fillcolour="k", label=lab, label_pos="right", labelpad=0.2, textcolour="k", fontsize=7)


# -----------------------------
# Layout
# -----------------------------
def build():
    # Table coordinates are arbitrary "diagram units".
    # Tune LENGTH/WIDTH if you want more whitespace.
    table = OpticalTable(length=28, width=16, size_factor=8.0, show_edge=False, show_grid=False)
    ax = table.ax

    fibre_color = "tab:blue"
    fibre_lw = 1.8

    # =========================
    # LEFT: SPDC source + split
    # =========================
    spdc = table.box_source(x=-12.8, y=0.0, size_x=1.6, size_y=0.9, angle=0, output_side="right",
                            colour="k", label="SPDC\n|Bell⟩", label_pos="bottom", labelpad=0.25, textcolour="k", fontsize=8)

    # Fibre out of SPDC (split to upper network and Bob branch)
    # Split node (invisible)
    split_x, split_y = -10.8, 0.0
    add_fc(table, split_x, split_y, text="FC")

    # Connection from SPDC to split FC
    polyline(ax, [(spdc.x, spdc.y), (split_x, split_y)], color=fibre_color, lw=fibre_lw)

    # =========================
    # LEFT-TOP: Charlie 1 block
    # =========================
    label(ax, -7.6, 4.1, "Charlie 1", size=10, color="teal", ha="left")

    add_circulator(table, -10.4, 3.8, lab="circulator")
    fibre_coil(ax, (-9.4, 3.2), n=4, angle_deg=75, color=fibre_color)

    # State-prep plates leading to PBS
    add_wp(table, -8.4, 3.8, lab="QWP")
    add_wp(table, -7.7, 3.8, lab="HWP")
    add_wp(table, -7.0, 3.8, lab="Comp")

    add_pbs(table, -5.9, 3.8, lab="PBS", direction="R")

    # Fibre routing: split -> circulator -> state-prep -> PBS
    bezier(ax, (split_x, split_y), (-10.2, 0.8), (-11.2, 2.4), (-10.4, 3.8), color=fibre_color, lw=fibre_lw)
    polyline(ax, [(-10.4, 3.8), (-8.4, 3.8), (-7.0, 3.8), (-5.9, 3.8)], color=fibre_color, lw=fibre_lw)

    # PBS outputs labelled 1 and 2 (these go to the central/right network)
    out1 = (-4.4, 5.1)
    out2 = (-4.4, 2.5)
    polyline(ax, [(-5.9, 3.8), out1], color=fibre_color, lw=fibre_lw)
    polyline(ax, [(-5.9, 3.8), out2], color=fibre_color, lw=fibre_lw)
    label(ax, out1[0] + 0.25, out1[1] + 0.2, "1", size=9, color="k", ha="left")
    label(ax, out2[0] + 0.25, out2[1] + 0.2, "2", size=9, color="k", ha="left")

    # =================================
    # LEFT-BOTTOM: Bob state-prep + PBS
    # =================================
    label(ax, -7.6, -5.3, "Bob", size=10, color="teal", ha="left")

    # Branch from split FC to Bob
    bezier(ax, (split_x, split_y), (-10.5, -0.8), (-10.2, -2.8), (-9.2, -4.4), color=fibre_color, lw=fibre_lw)
    fibre_coil(ax, (-10.6, -2.0), n=3, angle_deg=-65, color=fibre_color)

    # Bob waveplates and PBS
    add_wp(table, -8.4, -4.4, lab="QWP")
    add_wp(table, -7.7, -4.4, lab="HWP")
    add_wp(table, -7.0, -4.4, lab="tomo")
    add_pbs(table, -5.9, -4.4, lab="PBS", direction="R")

    polyline(ax, [(-9.2, -4.4), (-8.4, -4.4), (-7.0, -4.4), (-5.9, -4.4)], color=fibre_color, lw=fibre_lw)

    # Bob outputs to two detectors
    det_b1 = (-4.2, -3.4)
    det_b2 = (-4.2, -5.4)
    polyline(ax, [(-5.9, -4.4), det_b1], color=fibre_color, lw=fibre_lw)
    polyline(ax, [(-5.9, -4.4), det_b2], color=fibre_color, lw=fibre_lw)
    add_detector(table, det_b1[0], det_b1[1], lab="Db1")
    add_detector(table, det_b2[0], det_b2[1], lab="Db2")

    # ==================================
    # CENTER-BOTTOM: Charlie 0 analyser
    # ==================================
    label(ax, 0.4, -5.2, "Charlie 0", size=10, color="teal", ha="left")

    # An incoming fibre (from right network) into tomography plate then PBS
    # We'll draw that incoming line later from the right-side "main return".
    add_wp(table, -0.7, -4.4, lab="tomo")
    add_pbs(table, 0.6, -4.4, lab="PBS", direction="R")

    # Outputs to two detectors
    det_c01 = (2.4, -3.4)
    det_c02 = (2.4, -5.4)
    polyline(ax, [(0.6, -4.4), det_c01], color=fibre_color, lw=fibre_lw)
    polyline(ax, [(0.6, -4.4), det_c02], color=fibre_color, lw=fibre_lw)
    add_detector(table, det_c01[0], det_c01[1], lab="Dc01")
    add_detector(table, det_c02[0], det_c02[1], lab="Dc02")

    # ==========================================
    # TOP-RIGHT: Control qubit preparation block
    # ==========================================
    label(ax, 5.6, 7.0, "Preparation of control qubit", size=9, color="teal", ha="left")

    # A delay fibre (3 m fibre / ~150 ns) feeding an optical switch
    fibre_coil(ax, (3.8, 7.0), n=5, angle_deg=0, color=fibre_color)
    add_switch(table, 6.3, 6.8, lab="opt. switch")

    # After switch: FC -> PBS -> (PC / phase) -> FC (output)
    add_fc(table, 7.6, 6.8, text="FC")
    add_pbs(table, 8.7, 6.8, lab="PBS", direction="R")
    add_wp(table, 9.9, 6.8, lab="PC/φ")  # stand-in for Pockels cell / phase mod
    add_fc(table, 11.1, 6.8, text="FC")

    polyline(ax, [(2.6, 6.8), (5.1, 6.8)], color=fibre_color, lw=fibre_lw)  # placeholder feed-in
    polyline(ax, [(5.1, 6.8), (6.3, 6.8), (7.6, 6.8), (8.7, 6.8), (9.9, 6.8), (11.1, 6.8)],
             color=fibre_color, lw=fibre_lw)

    # ==========================================
    # RIGHT: Alice module (two internal branches)
    # ==========================================
    # Boundary box (dashed) to mimic sketch
    table.box(7.2, 1.7, size_x=13.0, size_y=7.2, angle=0, colour="k", linestyle="--",
              standalone=True, label=None)
    label(ax, 1.4, 1.7, "", size=1)  # no-op (keeps consistent)

    label(ax, 2.0, 3.0, "Alice 1", size=10, color="teal", ha="left")
    label(ax, 2.0, 1.0, "Alice 2", size=10, color="teal", ha="left")

    # Inputs into Alice from Charlie1 PBS outputs (1 and 2)
    alice_in1 = (1.0, 3.0)
    alice_in2 = (1.0, 1.0)
    bezier(ax, out1, (-1.0, 6.0), (0.0, 4.0), alice_in1, color=fibre_color, lw=fibre_lw)
    bezier(ax, out2, (-1.0, 0.0), (0.0, 0.8), alice_in2, color=fibre_color, lw=fibre_lw)

    # Alice 1 optics train
    add_fc(table, alice_in1[0], alice_in1[1], text="FC")
    add_wp(table, 1.8, 3.0, lab="QWP")
    add_wp(table, 2.5, 3.0, lab="HWP")
    add_wp(table, 3.2, 3.0, lab="Comp")
    add_wp(table, 3.9, 3.0, lab="QWP")
    polyline(ax, [(alice_in1[0], alice_in1[1]), (3.9, 3.0)], color=fibre_color, lw=fibre_lw)

    # Alice 2 optics train
    add_fc(table, alice_in2[0], alice_in2[1], text="FC")
    add_wp(table, 1.8, 1.0, lab="QWP")
    add_wp(table, 2.5, 1.0, lab="HWP")
    add_wp(table, 3.2, 1.0, lab="Comp")
    add_wp(table, 3.9, 1.0, lab="QWP")
    polyline(ax, [(alice_in2[0], alice_in2[1]), (3.9, 1.0)], color=fibre_color, lw=fibre_lw)

    # Combine at a PPBS / coupler stage + fibre loops, then an optical switch (as in sketch)
    add_pbs(table, 5.0, 2.0, lab="PPBS", direction="R")
    # Route both paths into PPBS
    bezier(ax, (3.9, 3.0), (4.2, 3.0), (4.6, 2.6), (5.0, 2.0), color=fibre_color, lw=fibre_lw)
    bezier(ax, (3.9, 1.0), (4.2, 1.0), (4.6, 1.4), (5.0, 2.0), color=fibre_color, lw=fibre_lw)

    fibre_coil(ax, (6.3, 2.8), n=3, angle_deg=35, color=fibre_color)
    fibre_coil(ax, (6.3, 1.2), n=3, angle_deg=-35, color=fibre_color)

    add_switch(table, 8.4, 2.0, lab="opt. switch")
    polyline(ax, [(5.0, 2.0), (8.4, 2.0)], color=fibre_color, lw=fibre_lw)

    # A small "opt. scaling" block at the right edge (sketch has this)
    table.box(10.7, 2.0, 1.6, 0.7, angle=0, colour="k", standalone=True, label="opt. scaling",
              label_pos="top", labelpad=0.2, textcolour="k", fontsize=7)
    polyline(ax, [(9.0, 2.0), (9.9, 2.0), (10.7, 2.0), (11.5, 2.0)], color=fibre_color, lw=fibre_lw)

    # ==========================================
    # OUTER RETURN FIBRE (right -> Charlie0)
    # ==========================================
    # This is the big loop drawn in the sketch. We keep it as a clean loop.
    main_return_top = (11.5, 2.0)
    main_return_mid = (12.6, -0.5)
    main_return_bot = (3.2, -4.4)   # incoming to Charlie0 tomo plate
    bezier(ax, main_return_top, (13.5, 2.0), (13.5, -2.5), main_return_mid, color=fibre_color, lw=fibre_lw)
    bezier(ax, main_return_mid, (12.0, -3.8), (7.5, -4.8), main_return_bot, color=fibre_color, lw=fibre_lw)
    polyline(ax, [(main_return_bot[0], main_return_bot[1]), (-0.7, -4.4)], color=fibre_color, lw=fibre_lw)

    # =========================
    # Cosmetics and export
    # =========================
    ax.set_xlim(-14, 14)
    ax.set_ylim(-8, 8)

    # Title for convenience (remove if you want a figure-ready output)
    label(ax, 0.0, 7.75, "Photonic circuit layout (clean redraw)", size=10, color="k")

    out_png = HERE / "circuit_diagram.png"
    out_svg = HERE / "circuit_diagram.svg"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    build()
