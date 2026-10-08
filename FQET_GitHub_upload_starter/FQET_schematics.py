#!/usr/bin/env python3
"""
Publication-ready, standalone FQET schematic figures.

Generates (in --output-dir):
    FQET_Bose_Hubbard_schematic.pdf  (vector graphics)
    FQET_Bose_Hubbard_schematic.png  (600-dpi raster preview)
    FQET_Ising_chain_schematic.pdf   (vector graphics)
    FQET_Ising_chain_schematic.png   (600-dpi raster preview)

Dependencies: numpy, matplotlib.  No LaTeX installation required.
Run: python FQET_schematics.py --output-dir Figs

Both drawings are pedagogical schematics, NOT snapshots of numerical
many-body wave functions.  The first is a 1D chain (drawn in 3D), not
a 2D optical lattice.  The second is not a phase diagram.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

# -------------------------- Shared graphic style -----------------------------
INK = "#1E3445"
MUTED = "#607480"
BLUE = "#2458CA"
ORANGE = "#EB7129"
TEAL = "#087F77"
PALE = "#EAF3F4"
PAPER = "#FFFFFF"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Tinos", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,  # Embed searchable TrueType text in the PDF.
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
})


def save_figure(fig: plt.Figure, basename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / (basename + ".pdf")
    png = output_dir / (basename + ".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.10)
    print("Wrote", pdf)
    print("Wrote", png)
    # Keep the figure open so it can be displayed in Jupyter or a GUI.


class Arrow3D(FancyArrowPatch):
    """Project a 3D line segment to an ordinary Matplotlib vector arrow."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        x2, y2, z2 = proj3d.proj_transform(xs, ys, zs, self.axes.get_proj())
        self.set_positions((x2[0], y2[0]), (x2[1], y2[1]))
        return float(np.min(z2))

    def draw(self, renderer):
        self.do_3d_projection(renderer)
        super().draw(renderer)


def potential(x, y, offset=0.40):
    """Stylized four-well potential for illustration only (not the BH Hamiltonian)."""
    barrier = 1.16 * np.cos(np.pi * x) ** 2
    confinement = 0.34 * (y / 0.56) ** 2
    right_half = 0.5 * (1.0 + np.tanh(x / 0.10))
    return barrier + confinement + offset * right_half


def boson_sphere(ax, x, y, z, radius=0.115):
    """Draw one shaded 3D boson as vector surface polygons."""
    theta = np.linspace(0, 2 * np.pi, 22)
    phi = np.linspace(0.04, np.pi - 0.04, 13)
    tt, pp = np.meshgrid(theta, phi)
    xx = x + radius * np.cos(tt) * np.sin(pp)
    yy = y + radius * np.sin(tt) * np.sin(pp)
    zz = z + radius * np.cos(pp)
    light = (np.cos(tt - 0.9) * np.sin(pp) + np.cos(pp) + 2) / 4
    color_map = colors.LinearSegmentedColormap.from_list(
        "boson", ["#AC3718", "#E9601E", "#FFA72D", "#FFF3A8"]
    )
    facecolors = color_map(np.clip(light, 0, 1))
    ax.plot_surface(xx, yy, zz, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True,
                    shade=False, zorder=20)


def make_bose_hubbard(output_dir: Path):
    fig = plt.figure(figsize=(8.0, 4.5), facecolor=PAPER)
    ax = fig.add_axes([0.0, 0.09, 1.0, 0.84], projection="3d", computed_zorder=False)
    ax.set_axis_off()
    ax.set_proj_type("ortho")

    x = np.linspace(-1.94, 1.94, 185)
    y = np.linspace(-0.57, 0.57, 36)
    xx, yy = np.meshgrid(x, y)
    zz = potential(xx, yy)
    cmap = colors.LinearSegmentedColormap.from_list(
        "wells", ["#146F65", "#37A580", "#90D7A2", "#E5F3BA"]
    )
    scaled = (zz - zz.min()) / (zz.max() - zz.min())
    ax.plot_surface(xx, yy, zz, facecolors=cmap(scaled),
                    rstride=1, cstride=1, linewidth=0, shade=False,
                    antialiased=True, alpha=0.99, zorder=2)

    # Four sites, four bosons. Configuration |2,0,1,1> illustrates on-site U.
    positions = [(-1.50, -0.145), (-1.50, 0.145), (0.50, 0.0), (1.50, 0.0)]
    for sx, sy in positions:
        boson_sphere(ax, sx, sy, float(potential(sx, sy)) + 0.17)

    # Coherent nearest-neighbor hopping J, represented by three blue arcs.
    # Arcs float above the potential so none is confused with a trajectory.
    centers = [-1.5, -0.5, 0.5, 1.5]
    for left, right in zip(centers[:-1], centers[1:]):
        tt = np.linspace(0.0, 1.0, 80)
        xx_arc = left + (right - left) * tt
        yy_arc = np.full_like(tt, -0.34)
        zz_arc = 1.79 + 0.20 * np.sin(np.pi * tt)
        ax.plot(xx_arc, yy_arc, zz_arc, color=BLUE, lw=2.75,
                solid_capstyle="round", zorder=30)
        arrow = Arrow3D([xx_arc[-7], xx_arc[-1]],
                        [yy_arc[-7], yy_arc[-1]],
                        [zz_arc[-7], zz_arc[-1]],
                        arrowstyle="-|>", mutation_scale=18,
                        lw=0.5, color=BLUE, zorder=31)
        ax.add_artist(arrow)

    # Separation between the two halves (a visual divider, not a barrier).
    ax.plot([0, 0], [0.67, 0.67], [-0.08, 1.60],
            color=TEAL, ls=(0, (4, 3)), lw=1.1, alpha=0.85)

    ax.set_xlim(-1.94, 1.94)
    ax.set_ylim(-0.67, 0.67)
    ax.set_zlim(-0.17, 2.22)
    ax.set_box_aspect((4.4, 1.48, 1.65), zoom=1.17)
    ax.view_init(elev=55, azim=-83)

    # Text is outside the 3D axes, so it stays readable in a small paper column.
    fig.text(0.040, 0.955, "BOSE-HUBBARD WORKING MEDIUM",
             color=INK, fontsize=14.2, weight="bold", ha="left", va="top")
    fig.text(0.040, 0.897, "Four sites, four bosons; right-half confinement is controlled",
             color=MUTED, fontsize=10.6, ha="left")
    fig.text(0.066, 0.075, r"$J$  nearest-neighbor hopping",
             color=BLUE, fontsize=10.3, weight="bold")
    fig.text(0.355, 0.075, r"$U$  on-site interaction",
             color=ORANGE, fontsize=10.3, weight="bold")
    fig.text(0.631, 0.075, r"$\Lambda(t)$  right-half offset",
             color=TEAL, fontsize=10.3, weight="bold")
    fig.text(0.50, 0.023,
             r"Illustrative Fock configuration $|2,0,1,1\rangle$; potential not to scale",
             color=MUTED, fontsize=9.4, ha="center")

    save_figure(fig, "FQET_Bose_Hubbard_schematic", output_dir)


def draw_spin(ax, x: float, y: float, up: bool):
    """One spin site plus an illustrative Z-basis up/down arrow."""
    ax.add_patch(Circle((x, y), 0.44, facecolor="#F2F6FA",
                        edgecolor="#BECAD4", lw=1.4, zorder=3))
    dy = 0.28 if up else -0.28
    ax.add_patch(FancyArrowPatch((x, y - 0.83 * dy),
                                 (x, y + dy),
                                 arrowstyle="-|>", mutation_scale=21,
                                 lw=2.8, color=BLUE if up else ORANGE,
                                 zorder=5))
    # A small orange/blue arrow on the same site is only a basis illustration,
    # not the expectation value of spin Z in the driven quantum state.


def make_ising(output_dir: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.20), facecolor=PAPER)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.96)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(0.12, 4.74, "DRIVEN TRANSVERSE-FIELD ISING CHAIN",
            color=INK, fontsize=14.2, weight="bold", ha="left", va="top")
    ax.text(0.12, 4.15,
            r"$H(t)=-J\sum_{j=1}^{3}Z_jZ_{j+1}-g\sum_{j=1}^{4}X_j"
            r"+\Lambda(t)\sum_{j=1}^{4}\frac{\mathbb{I}+Z_j}{2}$",
            fontsize=16.1, color=INK, va="center", ha="left")

    sites = np.array([1.2, 3.32, 5.44, 7.56])
    cy = 2.62
    for i, x in enumerate(sites):
        if i < len(sites) - 1:
            ax.plot([x + 0.44, sites[i + 1] - 0.44], [cy, cy],
                    color=TEAL, lw=4.1, zorder=1, solid_capstyle="round")
            ax.text((x + sites[i + 1]) / 2, cy + 0.35, r"$J$",
                    ha="center", va="bottom", color=TEAL, fontsize=13.5)
    for i, x in enumerate(sites):
        draw_spin(ax, x, cy, up=(i in (0, 1)))
        ax.text(x, 1.95, rf"site ${i+1}$", ha="center", color=MUTED,
                fontsize=10.3)

    # All spins feel the transverse field g, directed along local X.
    for x in sites:
        ax.add_patch(FancyArrowPatch((x - 0.34, 1.70), (x + 0.34, 1.70),
                                     arrowstyle="-|>", mutation_scale=15,
                                     lw=2.0, color="#9B5BB3"))
    ax.text(8.25, 1.70, r"$gX_j$", fontsize=12.3, color="#7F3EA0",
            ha="left", va="center")
    ax.text(8.20, 2.77, r"$Z_jZ_{j+1}$", color=TEAL, fontsize=11.5,
            va="center", ha="left")
    ax.text(8.15, 2.35, "Ising coupling", color=MUTED, fontsize=9.5,
            ha="left", va="center")

    # A controlled longitudinal bias, not a quantum critical phase diagram.
    box = FancyBboxPatch((0.16, 0.32), 9.62, 0.93,
                         boxstyle="round,pad=0.075,rounding_size=0.17",
                         facecolor=PALE, edgecolor="#BFD8DC", lw=1)
    ax.add_patch(box)
    ax.text(0.40, 0.92, "Controlled longitudinal offset",
            color=INK, ha="left", va="center", fontsize=11.3)
    ax.add_patch(FancyArrowPatch((4.35, 0.78), (7.72, 0.78),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=2.6, color=TEAL))
    ax.text(4.30, 0.97, r"$\Lambda(0)=+2J$", ha="left", va="bottom",
            fontsize=11.0, color=TEAL)
    ax.text(7.75, 0.97, r"$\Lambda(\tau_{\rm drv})=-2J$", ha="right",
            va="bottom", fontsize=11.0, color=TEAL)
    ax.text(0.40, 0.48,
            r"$R(t)=\{p_0(t),\ldots,p_4(t)\}$: probability of $r$ up spins",
            fontsize=10.4, color=MUTED)
    ax.text(9.81, 0.10,
            "Four-spin schematic; arrows show a basis configuration, not a phase transition",
            ha="right", color=MUTED, fontsize=9.0)

    save_figure(fig, "FQET_Ising_chain_schematic", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="Figs",
                        help="Directory for the four generated figures (default: Figs)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without displaying them (e.g., for batch runs)")
    args = parser.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    make_bose_hubbard(out)
    make_ising(out)
    print("Done. PDF artwork is vector-based; PNG files are 600-dpi previews.")
    if args.no_show:
        plt.close("all")
    else:
        # Display BOTH figures. In a notebook use: %matplotlib inline
        # For separate interactive windows use: %matplotlib qt
        # Do not close the figures here: Qt backends may return immediately.
        plt.show()


if __name__ == "__main__":
    main()
