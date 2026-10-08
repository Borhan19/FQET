#!/usr/bin/env python3
"""
Finite Quantum Engineering Thermodynamics (FQET)
Publication-plot generator for the main manuscript.

Outputs vector PDF figures using a common publication style.
The script is intentionally standalone: it only needs numpy, pandas,
and matplotlib, plus the CSV files in ./data.

Usage
-----
1. Put this script next to the data/ folder supplied with the package.
2. Run in Spyder/Jupyter/VS Code by clicking Run, or from a terminal:
       python FQET_manuscript_all_plots.py
3. PDFs are written to ./output.

Plotting conventions follow the project preferences:
- Times New Roman / Times-like serif font
- panel labels inside the axes, upper-left
- high-contrast, color-blind-friendly palette
- thin publication lines (lw ~ 1), emphasized lines at 1.5
- compact markers (ms ~ 4)
- vector PDF output with editable text
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from matplotlib import font_manager

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon
from matplotlib.lines import Line2D


# =============================================================================
# Paths and global style
# =============================================================================

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_OUT = HERE / "output"

# High-contrast, color-blind-friendly palette (Okabe-Ito inspired)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#E69F00"
BLACK = "#111111"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT = "#F3F3F3"

LW = 1.0
LW_EMPH = 1.5
MS = 4.0
PANEL_FS = 12
LABEL_FS = 12
TICK_FS = 12
LEGEND_FS = 12
ANNOT_FS = 12


def set_publication_style() -> None:
    """Apply the manuscript plotting style globally."""
    mpl.rcParams.update({
        "text.usetex": True,

        "text.latex.preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}
""",

        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Tinos",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],

        "font.size": 12.0,
        "axes.labelsize": LABEL_FS,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": LEGEND_FS,
        "legend.frameon": False,
        "lines.linewidth": LW,
        "lines.markersize": MS,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
    })

def panel_label(ax, label: str) -> None:
    """Place panel label inside the upper-left corner of an axis."""
    ax.text(
        0.025, 0.975, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_FS,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.5),
        zorder=50,
    )


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=LIGHT_GRAY, linewidth=0.45, alpha=0.55, zorder=0)


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")

    # Show the figure in Jupyter as well as saving it to disk.
    # plt.show() must come before plt.close(fig), otherwise the inline
    # notebook backend has nothing left to display.
    plt.show()
    plt.close(fig)


def clean_axis_for_diagram(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def rounded_box(ax, xy, w, h, text, *, edge=BLACK, face="white", fontsize=8.2,
                lw=0.9, radius=0.02, zorder=2, text_kwargs=None):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=zorder,
    )
    ax.add_patch(patch)
    kw = dict(ha="center", va="center", fontsize=fontsize, color=BLACK, zorder=zorder + 1)
    if text_kwargs:
        kw.update(text_kwargs)
    ax.text(x + w / 2, y + h / 2, text, **kw)
    return patch


def arrow(ax, start, end, *, color=BLACK, lw=1.0, style="-|>", mutation=9, zorder=3):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=mutation,
                        shrinkA=2, shrinkB=2),
        zorder=zorder,
    )


# =============================================================================
# Data loaders
# =============================================================================


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = {
        "bh_task": "bh_taskaware_family_comparison.csv",
        "bh_geometry": "bh_history_geometry.csv",
        "bh_blind": "bh_blind_parameter_points.csv",
        "bh_blind_summary": "bh_blind_summary.csv",
        "cross": "crossplatform_geometry.csv",
        "ising_phase": "ising_taskaware_phase_map.csv",
        "ising_h3": "ising_h3_branch_curve.csv",
        "ising_audit": "ising_solverB_spectral_audit.csv",
        "ising_h3_geom": "ising_h3_geometry_crossing.csv",
        "ising_h3_cost": "ising_h3_cost_crossing.csv",
    }
    out = {}
    missing = []
    for key, filename in names.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(str(path))
        else:
            out[key] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Missing required data file(s):\n" + "\n".join(missing))
    return out


# =============================================================================
# Quantitative panel functions
# =============================================================================


def plot_bh_taskaware_shots(ax, df: pd.DataFrame) -> None:
    d = df.copy()
    d["horizon"] = pd.to_numeric(d["horizon"], errors="coerce")
    d = d[d["horizon"].notna()].sort_values("horizon")

    ax.plot(d["horizon"], d["history_shots"], "o-", color=BLUE,
            lw=LW_EMPH, ms=MS, label="Best history")
    ax.plot(d["horizon"], d["derivative_shots"], "s-", color=ORANGE,
            lw=LW_EMPH, ms=MS, label="Best derivatives")
    ax.set_yscale("log")
    ax.set_xlabel(r"Forecast horizon $J\tau$")
    ax.set_ylabel("Shared repetitions")
    light_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))
    ax.margins(x=0.03)

    # Emphasize the final finite-memory point.
    row = d.iloc[-1]
    ratio = row["derivative_shots"] / row["history_shots"]
    ax.annotate(
        rf"$\times {ratio:.1f}$",
        xy=(row["horizon"], row["derivative_shots"]),
        xytext=(-28, -15), textcoords="offset points",
        fontsize=ANNOT_FS, color=ORANGE,
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8),
    )


def plot_bh_blind_task_margin(ax, points: pd.DataFrame, summary: pd.DataFrame) -> None:
    d = points.copy().sort_values("point")
    target = float(summary.iloc[0]["task_target"])
    d["budget_percent"] = 100.0 * d["max_abs_event_error"] / target

    sob = d[d["source"].str.lower() == "sobol"]
    stress = d[d["source"].str.lower() != "sobol"]

    ax.scatter(sob["point"], sob["budget_percent"], s=15, marker="o",
               facecolor=BLUE, edgecolor=BLUE, linewidth=0.4, label="Sobol")
    ax.scatter(stress["point"], stress["budget_percent"], s=18, marker="s",
               facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.4, label="Stress")
    ax.axhline(100.0, color=BLACK, ls="--", lw=LW_EMPH, label="1% task limit")
    ax.set_xlabel("Blind parameter point")
    ax.set_ylabel(r"Task budget used $(\%)$")
    ax.set_ylim(0, 105)
    light_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), ncol=1)

    idx = d["budget_percent"].idxmax()
    r = d.loc[idx]
    ax.annotate(
        f"max = {r['budget_percent']:.2f}%",
        xy=(r["point"], r["budget_percent"]),
        xytext=(-55, 18), textcoords="offset points",
        fontsize=ANNOT_FS,
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8),
    )


def plot_bh_blind_error_vs_lambda(ax, points: pd.DataFrame) -> None:
    d = points.copy()
    x = d["Lambda_scale"].to_numpy(float)
    y = 100.0 * d["max_abs_event_error"].to_numpy(float)
    ax.scatter(x, y, s=15, color=BLUE, marker="o", alpha=0.9)
    if len(x) >= 2:
        p = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, np.polyval(p, xx), color=ORANGE, lw=LW_EMPH, label="Linear guide")
    ax.set_xlabel(r"Confinement scale $\Lambda_0/\Lambda_{0,0}$")
    ax.set_ylabel("Worst blind error (%)")
    light_grid(ax)
    ax.legend(loc="upper right")


def plot_bh_principal_angles(ax, geom: pd.DataFrame) -> None:
    d = geom.sort_values(["s1", "s2"]).copy()
    x = np.arange(len(d))
    ax.plot(x, d["J_mean_angle"], "o-", color=BLUE, lw=LW_EMPH, ms=MS,
            label=r"First history layer vs $\mathcal{J}$")
    ax.plot(x, d["K2_mean_angle"], "s-", color=ORANGE, lw=LW_EMPH, ms=MS,
            label=r"Second layer vs $K^{(2)}$")
    labels = [rf"({a:.3f},{b:.3f})" for a, b in zip(d["s1"], d["s2"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Mean principal angle (deg)")
    ax.set_xlabel(r"History lags $(Js_1,Js_2)$")
    light_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))

    # Practical pair highlight.
    mask = np.isclose(d["s1"], 0.10) & np.isclose(d["s2"], 0.20)
    if mask.any():
        i = int(np.where(mask)[0][0])
        ax.scatter([i], [d.iloc[i]["J_mean_angle"]], s=42, facecolors="none",
                   edgecolors=GREEN, linewidths=LW_EMPH, zorder=8)
        ax.scatter([i], [d.iloc[i]["K2_mean_angle"]], s=42, facecolors="none",
                   edgecolors=GREEN, linewidths=LW_EMPH, zorder=8)
        ax.annotate("practical", xy=(i, d.iloc[i]["K2_mean_angle"]),
                    xytext=(8, -18), textcoords="offset points", fontsize=ANNOT_FS,
                    color=GREEN)


def plot_bh_conditioning(ax, geom: pd.DataFrame) -> None:
    d = geom.sort_values("s1").copy()
    ax.plot(d["s1"], d["gram_H3"], "o-", color=BLUE, lw=LW_EMPH, ms=MS)
    ax.set_yscale("log")
    ax.set_xlabel(r"First lag $Js_1$")
    ax.set_ylabel("Normalized Gram condition number")
    light_grid(ax)

    for s1, s2, text, color in [
        (0.025, 0.050, "local jet", ORANGE),
        (0.100, 0.200, "practical memory", GREEN),
    ]:
        m = np.isclose(d["s1"], s1) & np.isclose(d["s2"], s2)
        if m.any():
            r = d[m].iloc[0]
            ax.scatter([r["s1"]], [r["gram_H3"]], s=38, facecolors="none",
                       edgecolors=color, linewidths=LW_EMPH, zorder=8)
            ax.annotate(text, xy=(r["s1"], r["gram_H3"]),
                        xytext=(7, 8 if text == "practical memory" else -14),
                        textcoords="offset points", fontsize=ANNOT_FS, color=color)


def plot_ising_phase_boundaries(ax, df: pd.DataFrame) -> None:
    d = df.sort_values("g_over_J").copy()
    x = d["g_over_J"]
    ax.plot(x, d["tau_family_shot_equal_est"], "o-", color=BLUE,
            lw=LW_EMPH, ms=MS, label=r"$H_2/ER\mathcal{J}$ equality")
    ax.plot(x, d["tau_H3_activation_est"], "s-", color=ORANGE,
            lw=LW_EMPH, ms=MS, label=r"$H_2\to H_3$")
    ax.plot(x, d["tau_K2_activation_mid"], "^-", color=GREEN,
            lw=LW_EMPH, ms=MS + 0.4, label=r"$ER\mathcal{J}\to ER\mathcal{J}K^{(2)}$")
    ax.fill_between(x, d["K2_activation_low"], d["K2_activation_high"],
                    color=GREEN, alpha=0.12, linewidth=0)
    ax.set_xlabel(r"Transverse field $g/J$")
    ax.set_ylabel(r"Boundary $J\tau$")
    light_grid(ax)
    ax.legend(loc="upper right")


def plot_ising_h3_branch_crossing(ax, df: pd.DataFrame) -> None:
    for branch, color, marker in [("local", BLUE, "o"), ("long", ORANGE, "s")]:
        d = df[df["branch"] == branch].sort_values("horizon_Jtau")
        ax.plot(d["horizon_Jtau"], d["shared_shots"] / 1e6,
                marker=marker, color=color, lw=LW_EMPH, ms=MS,
                label=branch.capitalize())
    cross = 0.0584
    ax.axvline(cross, color=BLACK, lw=LW, ls="--")
    ax.text(cross - 0.00025, 0.97, r"$J\tau\simeq0.0584$",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=ANNOT_FS)
    ax.set_xlabel(r"Forecast horizon $J\tau$")
    ax.set_ylabel(r"Shared repetitions ($10^6$)")
    light_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))


def plot_ising_solver_audit(ax, df: pd.DataFrame) -> None:
    d = df.copy().reset_index(drop=True)
    x = np.arange(len(d))
    diff = np.abs(d["upper_distance_difference"].to_numpy(float))
    gap = np.maximum(d["solverA_gap"].to_numpy(float), d["solverB_gap"].to_numpy(float))

    ax.plot(x, diff, "o-", color=BLUE, lw=LW, ms=MS,
            label=r"$|d_B-d_A|$")
    ax.plot(x, gap, "s--", color=ORANGE, lw=LW, ms=MS,
            label="Larger primal-lower gap")
    ax.set_yscale("log")
    ax.set_ylabel("Spectral-distance audit scale")
    ax.set_xlabel("Selected transition checks")
    light_grid(ax)

    # Compact labels; shaded groups identify the three transition gates.
    labels = []
    for _, r in d.iterrows():
        arch = str(r["architecture"]).replace("ERJK2", "K2")
        labels.append(f"{arch}\n{int(r['event_mask'])}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=6.6)
    ax.set_xlim(-0.5, len(d)-0.5)
    groups = [(0, 1, "family"), (2, 5, r"$H_3$ act."), (6, 9, r"$K^{(2)}$ act.")]
    for a, b, txt in groups:
        ax.axvspan(a-0.5, b+0.5, color=LIGHT_GRAY, alpha=0.15, zorder=-5)
        ax.text((a+b)/2, 0.90, txt, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.0, color=GRAY)
    ax.set_xlabel("Architecture / event mask")
    ax.legend(loc="lower left")


def _extract_horizon(case: str) -> float:
    m = re.search(r"Jtau=([0-9.]+)", str(case))
    return float(m.group(1)) if m else np.nan


def plot_ising_history_collapse(ax, cross: pd.DataFrame) -> None:
    d = cross[cross["platform"] == "Ising"].copy()
    d["horizon"] = d["case"].map(_extract_horizon)
    d = d.sort_values("horizon")

    l1, = ax.plot(d["horizon"], d["first_layer_J_overlap"], "o-",
                  color=BLUE, lw=LW_EMPH, ms=MS, label=r"Overlap with $\mathcal{J}$")
    l2, = ax.plot(d["horizon"], d["second_layer_K2_overlap"], "s-",
                  color=ORANGE, lw=LW_EMPH, ms=MS, label=r"Overlap with $K^{(2)}$")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel(r"Forecast horizon $J\tau$")
    ax.set_ylabel("Subspace overlap")
    light_grid(ax)

    ax2 = ax.twinx()
    l3, = ax2.plot(d["horizon"], d["normalized_gram_cond"], "^-",
                   color=GREEN, lw=LW, ms=MS, label="Gram condition")
    ax2.set_yscale("log")
    ax2.set_ylabel("Gram condition number", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.spines["right"].set_color(GREEN)

    ax.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
              loc="center left")


def plot_crossplatform_conditioning(ax, cross: pd.DataFrame) -> None:
    rows = []
    for _, r in cross.iterrows():
        if r["platform"] == "Bose-Hubbard":
            label = "BH\nlocal" if "local" in str(r["case"]) else "BH\npractical"
        else:
            h = _extract_horizon(r["case"])
            label = f"Ising\n$J\\tau={h:.2f}$"
        rows.append((label, float(r["normalized_gram_cond"])))
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    x = np.arange(len(vals))
    colors = [ORANGE, GREEN, BLUE, BLUE, BLUE]
    ax.bar(x, vals, width=0.62, color=colors, edgecolor=BLACK, linewidth=0.45)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized Gram condition number")
    light_grid(ax)


# =============================================================================
# Conceptual / diagram panels
# =============================================================================


def draw_framework_boundary(ax) -> None:
    clean_axis_for_diagram(ax)
    # Physical system boundary
    rounded_box(ax, (0.24, 0.23), 0.52, 0.54, "Finite quantum\nworking medium",
                edge=BLUE, face="#EAF4FB", fontsize=9.0, lw=LW_EMPH)
    ax.text(0.50, 0.69, r"exact state $\rho(t)$", ha="center", va="center",
            fontsize=12, color=GRAY)
    # Microscopic dots
    for xx, yy in [(0.37,0.40),(0.46,0.37),(0.56,0.42),(0.63,0.36),(0.50,0.47)]:
        ax.add_patch(Circle((xx,yy), 0.012, color=GRAY, alpha=0.7))
    # Controls in
    rounded_box(ax, (0.02, 0.60), 0.17, 0.14, "controls\n" + r"$\lambda(t)$", edge=ORANGE,
                face="#FCEFEA", fontsize=12)
    arrow(ax, (0.19,0.67), (0.24,0.67), color=ORANGE, lw=LW_EMPH)
    # Measurement out
    rounded_box(ax, (0.80, 0.54), 0.18, 0.22, "retained\nrecord\n" + r"$E,R,\ldots$", edge=GREEN,
                face="#EAF7F2", fontsize=12)
    arrow(ax, (0.76,0.64), (0.80,0.64), color=GREEN, lw=LW_EMPH)
    ax.text(0.50, 0.13, "Boundary first; no online tomography", ha="center", va="center",
            fontsize=12)


def draw_framework_compatibility(ax) -> None:
    clean_axis_for_diagram(ax)
    ell = Ellipse((0.48,0.52), 0.72, 0.60, facecolor="#F7F7F7", edgecolor=BLACK, lw=0.9)
    ax.add_patch(ell)
    pts = [(0.29,0.58),(0.36,0.40),(0.43,0.68),(0.51,0.48),(0.60,0.62),(0.67,0.39),(0.53,0.30)]
    for i,(x,y) in enumerate(pts):
        ax.add_patch(Circle((x,y),0.016,facecolor=BLUE if i in (1,4) else GRAY,edgecolor="none",alpha=0.9))
    ax.text(0.48,0.84,r"same measured record $\mathbf{m}$",ha="center",fontsize=8.5)
    ax.text(0.48,0.18,r"compatibility set $\mathcal{A}_{\mathcal{R},\mathfrak{P}}(\mathbf{m})$",
            ha="center",fontsize=8.3)
    # target observable spread
    ax.plot([0.79,0.79],[0.34,0.69],color=ORANGE,lw=LW_EMPH)
    ax.plot([0.765,0.815],[0.34,0.34],color=ORANGE,lw=LW_EMPH)
    ax.plot([0.765,0.815],[0.69,0.69],color=ORANGE,lw=LW_EMPH)
    ax.text(0.84,0.515,"ambiguity\n" + r"$2d_{\mathcal{R}}(X)$",ha="left",va="center",fontsize=8.0,color=ORANGE)


def draw_framework_workflow(ax) -> None:
    clean_axis_for_diagram(ax)
    rounded_box(ax,(0.03,0.58),0.24,0.20,"Microscopic theory\nor calibration",edge=BLUE,face="#EAF4FB",fontsize=8.2)
    rounded_box(ax,(0.38,0.58),0.24,0.20,"Constitutive\ndata sheet",edge=ORANGE,face="#FCEFEA",fontsize=8.2)
    rounded_box(ax,(0.73,0.58),0.24,0.20,"Certified\noperating domain",edge=GREEN,face="#EAF7F2",fontsize=8.2)
    arrow(ax,(0.27,0.68),(0.38,0.68),color=BLACK,lw=LW)
    arrow(ax,(0.62,0.68),(0.73,0.68),color=BLACK,lw=LW)
    ax.text(0.50,0.86,"OFFLINE",ha="center",fontsize=8.5,fontweight="bold")
    ax.plot([0.04,0.96],[0.46,0.46],color=LIGHT_GRAY,lw=0.8)
    ax.text(0.50,0.39,"ONLINE",ha="center",fontsize=8.5,fontweight="bold")
    rounded_box(ax,(0.08,0.10),0.29,0.18,"Measured\nthermodynamic state",edge=GREEN,face="#EAF7F2",fontsize=8.2)
    rounded_box(ax,(0.63,0.10),0.29,0.18,"Future macroscopic\nprediction",edge=BLUE,face="#EAF4FB",fontsize=8.2)
    arrow(ax,(0.37,0.19),(0.63,0.19),color=BLACK,lw=LW_EMPH)
    ax.text(0.50,0.25,"frozen response",ha="center",fontsize=7.8,color=GRAY)


def draw_framework_selection(ax) -> None:
    clean_axis_for_diagram(ax)
    rounded_box(ax,(0.03,0.61),0.39,0.18,"Instantaneous state\n" + r"$ER\mathcal{J}\to ER\mathcal{J}K^{(2)}$",
                edge=BLUE,face="#EAF4FB",fontsize=12)
    rounded_box(ax,(0.58,0.61),0.39,0.18,"Finite history\n" + r"$H_2\to H_3$",
                edge=ORANGE,face="#FCEFEA",fontsize=12)
    rounded_box(ax,(0.23,0.20),0.54,0.22,"Task certificate\nconstitutive error + metrology\n" + r"$\leq\,\epsilon_{\mathrm{task}}$",
                edge=GREEN,face="#EAF7F2",fontsize=12,lw=LW_EMPH)
    arrow(ax,(0.225,0.61),(0.40,0.42),color=BLUE,lw=LW_EMPH)
    arrow(ax,(0.775,0.61),(0.60,0.42),color=ORANGE,lw=LW_EMPH)
    ax.text(0.50,0.08,"Choose the cheapest sufficient representation",ha="center",fontsize=12)


def draw_bh_timeline(ax) -> None:
    clean_axis_for_diagram(ax)
    y=0.48
    ax.plot([0.08,0.93],[y,y],color=BLACK,lw=LW_EMPH)
    ticks=[(0.12,r"$t_c-0.2/J$"),(0.34,r"$t_c-0.1/J$"),(0.56,r"$t_c$"),(0.88,r"$t_c+\tau$")]
    for x,lab in ticks:
        ax.plot([x,x],[y-0.04,y+0.04],color=BLACK,lw=0.8)
        ax.text(x,y-0.10,lab,ha="center",va="top",fontsize=12)
    # measurement nodes
    for x in [0.12,0.34,0.56]:
        ax.add_patch(Circle((x,y),0.025,facecolor=BLUE,edgecolor="white",lw=0.6,zorder=4))
    ax.add_patch(Circle((0.56,y+0.18),0.025,facecolor=GREEN,edgecolor="white",lw=0.6,zorder=4))
    ax.text(0.56,y+0.25,r"$E(t_c)$",ha="center",fontsize=12,color=GREEN)
    ax.text(0.34,0.80,"retained past records",ha="center",fontsize=12,color=BLUE)
    ax.text(0.78,0.67,r"predict $R(t_c+\tau)$",ha="center",fontsize=12,color=ORANGE)
    arrow(ax,(0.58,0.58),(0.86,0.58),color=ORANGE,lw=LW_EMPH)
    ax.text(0.50,0.08,r"$Jt_c=0.2$,  $0\leq J\tau\leq0.2$",ha="center",fontsize=12)


def draw_bh_parameter_cube(ax) -> None:
    clean_axis_for_diagram(ax)
    # Isometric cube in 2D
    A=np.array([0.22,0.24]); B=np.array([0.68,0.24]); C=np.array([0.78,0.38]); D=np.array([0.32,0.38])
    shift=np.array([0.0,0.38]); A2=A+shift; B2=B+shift; C2=C+shift; D2=D+shift
    edges=[(A,B),(B,C),(C,D),(D,A),(A,A2),(B,B2),(C,C2),(D,D2),(A2,B2),(B2,C2),(C2,D2),(D2,A2)]
    for p,q in edges: ax.plot([p[0],q[0]],[p[1],q[1]],color=BLACK,lw=0.85)
    for p in [A,B,C,D,A2,B2,C2,D2]: ax.add_patch(Circle(tuple(p),0.015,facecolor=ORANGE,edgecolor=BLACK,lw=0.4,zorder=4))
    ax.text(0.50,0.51,"8 response\ncorner tables",ha="center",va="center",fontsize=12,color=ORANGE)
    ax.text(0.52,0.10,r"$U/U_0\in[0.9,1.1]$",ha="center",fontsize=12)
    ax.text(0.12,0.52,r"$\Lambda_0/\Lambda_{0,0}$",ha="center",rotation=90,fontsize=12)
    ax.text(0.80,0.74,r"$\tau_{BC}/\tau_{BC,0}$",ha="center",fontsize=12)
    ax.text(0.50,0.88,r"trilinear response + $5^3$ offline validation",ha="center",fontsize=12)


def draw_ising_families(ax) -> None:
    clean_axis_for_diagram(ax)
    ax.text(0.22,0.86,"History family",ha="center",fontsize=12,fontweight="bold",color=ORANGE)
    ax.text(0.78,0.86,"Instantaneous family",ha="center",fontsize=8.5,fontweight="bold",color=BLUE)
    rounded_box(ax,(0.05,0.58),0.34,0.15,r"$H_2$" + "\n" + r"$E,R,R(t_c-s_1)$",edge=ORANGE,face="#FCEFEA",fontsize=12)
    rounded_box(ax,(0.05,0.25),0.34,0.15,r"$H_3$" + "\n" + r"$+R(t_c-s_2)$",edge=ORANGE,face="#FCEFEA",fontsize=12)
    arrow(ax,(0.22,0.58),(0.22,0.40),color=ORANGE,lw=LW_EMPH)
    rounded_box(ax,(0.61,0.58),0.34,0.15,r"$ER\mathcal{J}$" + "\ncurrent layer",edge=BLUE,face="#EAF4FB",fontsize=12)
    rounded_box(ax,(0.61,0.25),0.34,0.15,r"$ER\mathcal{J}K^{(2)}$" + "\nsecond derivative",edge=BLUE,face="#EAF4FB",fontsize=12)
    arrow(ax,(0.78,0.58),(0.78,0.40),color=BLUE,lw=LW_EMPH)
    ax.text(0.50,0.08,"Family selection and layer activation are different decisions",ha="center",fontsize=12)


def draw_jet_vs_memory(ax) -> None:
    clean_axis_for_diagram(ax)
    # Left: derivative jet
    ax.text(0.25,0.88,"Local derivative jet",ha="center",fontsize=12,fontweight="bold",color=BLUE)
    curve_x=np.linspace(0.06,0.44,100)
    curve_y=0.58+0.13*np.sin((curve_x-0.06)*5.8)
    ax.plot(curve_x,curve_y,color=BLUE,lw=LW_EMPH)
    tc_x=0.36
    tc_y=np.interp(tc_x,curve_x,curve_y)
    ax.add_patch(Circle((tc_x,tc_y),0.014,facecolor=BLACK,edgecolor="none"))
    # close past points
    for dx in [0.04,0.08]:
        x=tc_x-dx; y=np.interp(x,curve_x,curve_y)
        ax.add_patch(Circle((x,y),0.014,facecolor=ORANGE,edgecolor="none"))
    ax.text(0.25,0.31,"nearby history " + r"$\approx$" + "\n" + r"$R,\dot R,\ddot R$",ha="center",fontsize=12)
    # Right: nonlocal memory
    ax.text(0.75,0.88,"Genuine finite memory",ha="center",fontsize=2,fontweight="bold",color=GREEN)
    x2=np.linspace(0.57,0.95,100)
    y2=0.55+0.18*np.sin((x2-0.57)*7.0+0.4)
    ax.plot(x2,y2,color=GREEN,lw=LW_EMPH)
    tc2=0.91; yy=np.interp(tc2,x2,y2)
    ax.add_patch(Circle((tc2,yy),0.014,facecolor=BLACK,edgecolor="none"))
    for x in [0.60,0.74]:
        y=np.interp(x,x2,y2)
        ax.add_patch(Circle((x,y),0.014,facecolor=PURPLE,edgecolor="none"))
    ax.text(0.75,0.31,"past points retain\nnew trajectory directions",ha="center",fontsize=8.1)
    ax.plot([0.50,0.50],[0.18,0.88],color=LIGHT_GRAY,lw=0.8)


# =============================================================================
# Composite manuscript figures
# =============================================================================


def make_fig1_framework(out_dir: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.2))
    draw_framework_boundary(axs[0,0]); panel_label(axs[0,0], "(a)")
    draw_framework_compatibility(axs[0,1]); panel_label(axs[0,1], "(b)")
    draw_framework_workflow(axs[1,0]); panel_label(axs[1,0], "(c)")
    draw_framework_selection(axs[1,1]); panel_label(axs[1,1], "(d)")
    fig.subplots_adjust(wspace=0.12, hspace=0.16)
    save_pdf(fig, out_dir / "Fig1_framework.pdf")


def make_fig2_bose_hubbard(out_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.4))
    draw_bh_timeline(axs[0,0]); panel_label(axs[0,0], "(a)")
    draw_bh_parameter_cube(axs[0,1]); panel_label(axs[0,1], "(b)")
    plot_bh_taskaware_shots(axs[1,0], data["bh_task"]); panel_label(axs[1,0], "(c)")
    plot_bh_blind_task_margin(axs[1,1], data["bh_blind"], data["bh_blind_summary"]); panel_label(axs[1,1], "(d)")
    fig.subplots_adjust(wspace=0.28, hspace=0.28)
    save_pdf(fig, out_dir / "Fig2_bose_hubbard_datasheet.pdf")


def make_fig3_ising(out_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.6))
    draw_ising_families(axs[0,0]); panel_label(axs[0,0], "(a)")
    plot_ising_phase_boundaries(axs[0,1], data["ising_phase"]); panel_label(axs[0,1], "(b)")
    plot_ising_h3_branch_crossing(axs[1,0], data["ising_h3"]); panel_label(axs[1,0], "(c)")
    plot_ising_solver_audit(axs[1,1], data["ising_audit"]); panel_label(axs[1,1], "(d)")
    fig.subplots_adjust(wspace=0.30, hspace=0.32)
    save_pdf(fig, out_dir / "Fig3_ising_representation_selection.pdf")


def make_fig4_mechanism(out_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.6))
    draw_jet_vs_memory(axs[0,0]); panel_label(axs[0,0], "(a)")
    plot_bh_principal_angles(axs[0,1], data["bh_geometry"]); panel_label(axs[0,1], "(b)")
    plot_crossplatform_conditioning(axs[1,0], data["cross"]); panel_label(axs[1,0], "(c)")
    plot_ising_history_collapse(axs[1,1], data["cross"]); panel_label(axs[1,1], "(d)")
    fig.subplots_adjust(wspace=0.34, hspace=0.34)
    save_pdf(fig, out_dir / "Fig4_crossplatform_mechanism.pdf")


# =============================================================================
# Standalone quantitative plots (useful for re-arranging manuscript panels)
# =============================================================================


def one_panel(out_dir: Path, filename: str, plotter, *args, figsize=(3.45, 2.75)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    plotter(ax, *args)
    fig.tight_layout(pad=0.3)
    save_pdf(fig, out_dir / filename)


def make_all_standalone(out_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    one_panel(out_dir, "bh_taskaware_shots.pdf", plot_bh_taskaware_shots, data["bh_task"])
    one_panel(out_dir, "bh_blind_task_margin.pdf", plot_bh_blind_task_margin,
              data["bh_blind"], data["bh_blind_summary"])
    one_panel(out_dir, "bh_blind_error_vs_lambda.pdf", plot_bh_blind_error_vs_lambda, data["bh_blind"])
    one_panel(out_dir, "bh_principal_angles.pdf", plot_bh_principal_angles, data["bh_geometry"], figsize=(4.2, 3.0))
    one_panel(out_dir, "bh_history_conditioning.pdf", plot_bh_conditioning, data["bh_geometry"])
    one_panel(out_dir, "ising_phase_boundaries.pdf", plot_ising_phase_boundaries, data["ising_phase"])
    one_panel(out_dir, "ising_h3_branch_crossing.pdf", plot_ising_h3_branch_crossing, data["ising_h3"])
    one_panel(out_dir, "ising_solverB_audit.pdf", plot_ising_solver_audit, data["ising_audit"], figsize=(4.4, 3.1))
    one_panel(out_dir, "ising_history_derivative_collapse.pdf", plot_ising_history_collapse, data["cross"])
    one_panel(out_dir, "crossplatform_conditioning.pdf", plot_crossplatform_conditioning, data["cross"])


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all FQET manuscript figures as vector PDFs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                        help="Directory containing the supplied CSV data files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="Directory for PDF output.")
    args = parser.parse_args()

    set_publication_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir)

    make_fig1_framework(args.out_dir)
    make_fig2_bose_hubbard(args.out_dir, data)
    make_fig3_ising(args.out_dir, data)
    make_fig4_mechanism(args.out_dir, data)
    make_all_standalone(args.out_dir, data)

    generated = sorted(args.out_dir.glob("*.pdf"))
    print(f"Generated {len(generated)} PDF figures in: {args.out_dir}")
    for p in generated:
        print("  ", p.name)


if __name__ == "__main__":
    main()
