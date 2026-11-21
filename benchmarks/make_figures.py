"""Turn results/benchmarks.json into the PNG figures used in the report.

    python3 benchmarks/make_figures.py [results/benchmarks.json]

Needs matplotlib (only this script does; the package itself has no
third-party dependencies). Figures go to figures/.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD, NEW, THIRD = "#eb6834", "#2a78d6", "#1baf7a"     # validated categorical slots
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 4.5, "axes.titlelocation": "left",
})


def line(ax, x, y, color, label, marker="o", end_label=True):
    ax.plot(x, y, color=color, marker=marker, label=label, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)
    if end_label:
        ax.annotate(label, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)


def speedups(ax, x, old, new, every=1):
    for i, (xi, a, b) in enumerate(zip(x, old, new)):
        if i % every == 0 or i == len(x) - 1:
            # above the Phase 2 point, where there is always free space
            ax.annotate(f"{a / b:.0f}x" if a / b >= 10 else f"{a / b:.1f}x",
                        (xi, a), xytext=(-2, 7), textcoords="offset points",
                        ha="right" if i == len(x) - 1 else "center", fontsize=6.8, color=MUTED)


def us(value, _):
    return f"{value:g}"


def fig_structures(r, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    (a, b), (c, d) = axes

    g = r["grid"]
    x = [row["n"] for row in g]
    old, new = [row["phase2"] * 1e6 for row in g], [row["phase3"] * 1e6 for row in g]
    line(a, x, old, OLD, "Phase 2")
    line(a, x, new, NEW, "Phase 3", marker="s")
    speedups(a, x, old, new, every=2)
    a.set(xscale="log", yscale="log", title="(a) Grid nearest-5 query",
          xlabel="free drivers (constant density)", ylabel="microseconds per query")

    ag = r["aging"]
    x = [row["n"] for row in ag]
    old, new = [row["phase2"] * 1e3 for row in ag], [row["phase3"] * 1e3 for row in ag]
    line(b, x, old, OLD, "Phase 2")
    line(b, x, new, NEW, "Phase 3", marker="s")
    speedups(b, x, old, new, every=2)
    b.set(xscale="log", yscale="log", title="(b) age_requests, one round",
          xlabel="waiting requests", ylabel="milliseconds per round")

    s = r["search"]
    x = [row["side"] for row in s]
    old, new = [row["phase2"] * 1e6 for row in s], [row["phase3"] * 1e6 for row in s]
    line(c, x, old, OLD, "Phase 2")
    line(c, x, new, NEW, "Phase 3", marker="s")
    speedups(c, x, old, new)
    c.set(title="(c) pick_driver, river city", xlabel="city side (intersections)",
          ylabel="microseconds per request", ylim=(0, max(old) * 1.2))

    rt = r["routes"]
    x = [row["side"] for row in rt]
    line(d, x, [row["dijkstra"] * 1e3 for row in rt], OLD, "Dijkstra (P2)")
    line(d, x, [row["astar"] * 1e3 for row in rt], THIRD, "A*", marker="^")
    line(d, x, [row["alt8"] * 1e3 for row in rt], NEW, "A* + 8 landmarks", marker="s")
    d.set(title="(d) Trip-duration route query", xlabel="city side (intersections)",
          ylabel="milliseconds per route")
    for ax in (a, b, c, d):
        ax.margins(x=0.08)
        ax.yaxis.set_major_formatter(FuncFormatter(us))
    c.set_xlim(right=c.get_xlim()[1] * 1.08)
    d.set_xlim(right=d.get_xlim()[1] * 1.25)
    a.set_xlim(right=a.get_xlim()[1] * 2.2)
    b.set_xlim(right=b.get_xlim()[1] * 2.2)
    for ax in (a, b):
        low, high = ax.get_ylim()
        ax.set_ylim(low, high * 3)      # room for the speed-up labels
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    fig.savefig(out)
    plt.close(fig)


def fig_simulation(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, key, title in ((axes[0], "sim", "(a) Normal demand"),
                           (axes[1], "sim_busy", "(b) Hotspots + incidents + shift changes")):
        rows = r[key]
        x = [row["side"] for row in rows]
        old = [row["phase2"] for row in rows]
        new = [row["phase3"] + row["setup"] for row in rows]
        line(ax, x, old, OLD, "Phase 2")
        line(ax, x, new, NEW, "Phase 3", marker="s")
        speedups(ax, x, old, new)
        ax.set(title=title, xlabel="city side (drivers = 0.15 per corner)",
               ylabel="seconds per simulated hour", ylim=(0, max(old) * 1.15))
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 8, x[-1] + 22)
    fig.tight_layout(w_pad=2)
    fig.savefig(out)
    plt.close(fig)


def fig_memory_tradeoffs(r, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    a, b, c = axes

    ls = r["long_shift"]
    x = [row["hours"] for row in ls]
    line(a, x, [row["phase2"] / 2**20 for row in ls], OLD, "Phase 2")
    line(a, x, [row["phase3"] / 2**20 for row in ls], NEW, "Phase 3", marker="s")
    a.set(title="(a) Peak memory", xlabel="shift length (hours)", ylabel="MiB", ylim=(0, None),
          xticks=x, xlim=(0.5, 11))

    cs = r["cell_size"]
    b.plot([row["cell"] for row in cs], [row["phase3"] * 1e6 for row in cs], color=NEW,
           marker="s", markeredgecolor="white", markeredgewidth=0.8)
    b.set(xscale="log", yscale="log", title="(b) Grid cell size", xlabel="cell size (m)",
          ylabel="microseconds per query")
    b.yaxis.set_major_formatter(FuncFormatter(us))
    b.xaxis.set_major_formatter(FuncFormatter(us))

    tr = r["tradeoff"]
    fixed = [row for row in tr if row["mode"] == "fixed"]
    exact = [row for row in tr if row["mode"] == "exact"][0]
    ks = [row["k"] for row in fixed]
    c.plot(ks, [row["time"] * 1e6 for row in fixed], color=OLD, marker="o",
           markeredgecolor="white", markeredgewidth=0.8, label="fixed list of k")
    c.axhline(exact["time"] * 1e6, color=NEW, linewidth=2, label="exact (Phase 3)")
    for row in fixed:
        if row["missed"] > 0:
            # share of requests that got a slower pickup than the exact search
            first = row["k"] == 1
            c.annotate(f"{row['missed']:.1%}", (row["k"], row["time"] * 1e6),
                       xytext=(-6, 3) if first else (5, -3), textcoords="offset points",
                       fontsize=6.8, color=MUTED, ha="right" if first else "left",
                       va="bottom" if first else "top")
    c.set(xscale="log", yscale="log", title="(c) Fixed list vs exact",
          xlabel="k (candidates checked)", ylabel="microseconds per request", ylim=(6, 600),
          xlim=(0.5, 22))
    c.yaxis.set_major_formatter(FuncFormatter(us))
    c.xaxis.set_major_formatter(FuncFormatter(us))
    c.legend(loc="upper left")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else os.path.join(ROOT, "results", "benchmarks.json")
    with open(path) as fh:
        r = json.load(fh)
    out_dir = os.path.join(ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_structures(r, os.path.join(out_dir, "fig1_structures.png"))
    fig_simulation(r, os.path.join(out_dir, "fig2_simulation.png"))
    fig_memory_tradeoffs(r, os.path.join(out_dir, "fig3_memory_tradeoffs.png"))
    print("figures written to figures/")


if __name__ == "__main__":
    main()
