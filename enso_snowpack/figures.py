"""Matplotlib figures. Palette: fixed categorical order per state, a
blue/red diverging scale with a neutral grey midpoint for signed
correlations, and one measure per axis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import STATE_NAMES  # noqa: E402

STATE_COLORS = {"MT": "#2a78d6", "ID": "#eb6834", "WY": "#e87ba4", "CO": "#1baf7a", "UT": "#eda100"}
PHASE_COLORS = {"La Nina": "#2a78d6", "Neutral": "#9a9892", "El Nino": "#e34948"}
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#0d366b", "#2a78d6", "#f0efec",
                                                          "#e34948", "#7a1f1e"])
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 130,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_oni_timeseries(enso: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    y = enso["oni_djf"].to_numpy()
    x = enso["water_year"].to_numpy()
    ax.bar(x, np.clip(y, 0.5, None) - 0.5, bottom=0.5, color=PHASE_COLORS["El Nino"], width=0.8,
           label="El Niño (DJF ONI ≥ +0.5)")
    ax.bar(x, np.clip(y, None, -0.5) + 0.5, bottom=-0.5, color=PHASE_COLORS["La Nina"], width=0.8,
           label="La Niña (DJF ONI ≤ −0.5)")
    ax.plot(x, y, color=TEXT, lw=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    for lvl in (0.5, -0.5):
        ax.axhline(lvl, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("Water year (DJF season ending in that year)")
    ax.set_ylabel("DJF ONI (°C)")
    ax.set_title("Oceanic Niño Index, December–February of each water year")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    return _save(fig, out, "fig1_oni_timeseries")


def fig_scatter_by_state(regional: pd.DataFrame, enso: pd.DataFrame, out: Path,
                         metric_label: str, name: str,
                         index_col: str = "oni_djf") -> Path:
    states = [s for s in STATE_COLORS if s in set(regional["region"])]
    fig, axes = plt.subplots(1, len(states), figsize=(3.4 * len(states), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, states):
        d = regional[regional["region"] == st].merge(enso[["water_year", index_col, "phase"]],
                                                     on="water_year")
        x, y = d[index_col].to_numpy(float), d["value"].to_numpy(float)
        ax.scatter(x, y, s=28, color=STATE_COLORS[st], edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        if len(d) >= 3:
            b, a = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, b * xx + a, color=TEXT, lw=1.5)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.03, 0.95, f"r = {r:+.2f}, n = {len(d)}", transform=ax.transAxes,
                    va="top", fontsize=9, color=TEXT)
        # label the extremes so the reader can find the famous winters
        for _, row in d.iloc[np.argsort(np.abs(x))[-4:]].iterrows():
            ax.annotate(str(int(row["water_year"])), (row[index_col], row["value"]),
                        fontsize=7.5, color=MUTED, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(STATE_NAMES.get(st, st))
        ax.set_xlabel("DJF ONI (°C)")
    axes[0].set_ylabel(metric_label)
    fig.suptitle("Snowpack anomaly vs. ENSO index, one point per water year", y=1.02,
                 fontsize=11, fontweight="bold")
    return _save(fig, out, name)


def fig_station_map(sc: pd.DataFrame, out: Path, title: str, name: str) -> Path:
    d = sc.dropna(subset=["latitude", "longitude", "r"])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)
    sig = d["p"] < 0.05
    ax.scatter(d.loc[~sig, "longitude"], d.loc[~sig, "latitude"], c=d.loc[~sig, "r"], cmap=DIVERGING,
               norm=norm, s=22, edgecolor="#bcbab4", linewidth=0.5, label="p ≥ 0.05")
    s = ax.scatter(d.loc[sig, "longitude"], d.loc[sig, "latitude"], c=d.loc[sig, "r"], cmap=DIVERGING,
                   norm=norm, s=46, edgecolor=TEXT, linewidth=0.8, label="p < 0.05")
    cb = fig.colorbar(s, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Pearson r (April-1 SWE anomaly vs DJF ONI)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.deg2rad(d["latitude"].mean())))
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left", title="Station significance")
    for st, (lon, lat) in {"MT": (-110.0, 46.9), "ID": (-114.6, 44.3), "WY": (-107.5, 43.0),
                           "CO": (-105.5, 39.0), "UT": (-111.6, 39.3)}.items():
        ax.text(lon, lat, st, fontsize=12, color=MUTED, ha="center", alpha=0.7, fontweight="bold")
    return _save(fig, out, name)


def fig_phase_boxes(regional: pd.DataFrame, enso: pd.DataFrame, out: Path, metric_label: str,
                    name: str) -> Path:
    states = [s for s in STATE_COLORS if s in set(regional["region"])]
    phases = ["La Nina", "Neutral", "El Nino"]
    fig, axes = plt.subplots(1, len(states), figsize=(3.2 * len(states), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, states):
        d = regional[regional["region"] == st].merge(enso[["water_year", "phase"]], on="water_year")
        groups = [d[d["phase"] == p]["value"].to_numpy() for p in phases]
        bp = ax.boxplot(groups, tick_labels=[f"{p}\n(n={len(g)})" for p, g in zip(phases, groups)],
                        widths=0.55, patch_artist=True, showfliers=False)
        for patch, p in zip(bp["boxes"], phases):
            patch.set_facecolor(PHASE_COLORS[p]); patch.set_alpha(0.55); patch.set_edgecolor(TEXT)
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set_color(TEXT)
        for i, g in enumerate(groups, start=1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(g)), g, s=10, color=TEXT,
                       alpha=0.5, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(STATE_NAMES.get(st, st))
    axes[0].set_ylabel(metric_label)
    fig.suptitle("Snowpack anomaly by ENSO phase (DJF ONI thresholds ±0.5)", y=1.02,
                 fontsize=11, fontweight="bold")
    return _save(fig, out, name)


def fig_nino_strength(regional: pd.DataFrame, enso: pd.DataFrame, out: Path, metric_label: str,
                      name: str) -> Path:
    """El Niño years only: does a stronger event mean more (or less) snow?"""
    states = [s for s in STATE_COLORS if s in set(regional["region"])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    e = enso[enso["phase"] == "El Nino"]
    for st in states:
        d = regional[regional["region"] == st].merge(e[["water_year", "oni_peak"]], on="water_year")
        if d.empty:
            continue
        x, y = d["oni_peak"].to_numpy(float), d["value"].to_numpy(float)
        ax.scatter(x, y, s=32, color=STATE_COLORS[st], edgecolor="#fcfcfb", linewidth=0.8,
                   label=f"{STATE_NAMES[st]} (n={len(d)})", zorder=3)
        if len(d) >= 4:
            b, a = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, b * xx + a, color=STATE_COLORS[st], lw=1.5)
    for thr, lab in [(1.0, "moderate"), (1.5, "strong"), (2.0, "very strong")]:
        ax.axvline(thr, color=MUTED, lw=0.6, ls=":")
        ax.text(thr, ax.get_ylim()[1], lab, fontsize=7.5, color=MUTED, ha="left", va="top", rotation=90)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Event strength: peak ONI of the winter (°C)")
    ax.set_ylabel(metric_label)
    ax.set_title("El Niño winters only — snowpack anomaly vs. event strength")
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, out, name)


def fig_regional_timeseries(regional: pd.DataFrame, enso: pd.DataFrame, out: Path, metric_label: str,
                            name: str) -> Path:
    states = [s for s in STATE_COLORS if s in set(regional["region"])]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    e = enso.set_index("water_year")
    for wy, row in e.iterrows():
        if row["phase"] == "El Nino":
            ax.axvspan(wy - 0.5, wy + 0.5, color=PHASE_COLORS["El Nino"], alpha=0.10, lw=0)
        elif row["phase"] == "La Nina":
            ax.axvspan(wy - 0.5, wy + 0.5, color=PHASE_COLORS["La Nina"], alpha=0.10, lw=0)
    for st in states:
        d = regional[regional["region"] == st].sort_values("water_year")
        ax.plot(d["water_year"], d["value"], color=STATE_COLORS[st], lw=1.8, label=STATE_NAMES[st])
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(metric_label); ax.set_xlabel("Water year")
    ax.set_title("Statewide snowpack anomaly; red bands = El Niño winters, blue = La Niña")
    ax.legend(frameon=False, ncol=5, loc="upper left")
    return _save(fig, out, name)
