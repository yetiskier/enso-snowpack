"""Figures about the snowpack as a skier experiences it.

Every quantity here is in days or inches, not z-scores or correlations, and
every figure follows one convention: **red means less snow, blue means more.**
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .figures import TEXT, MUTED, GRID, SNOW_DIVERGING, LESS_SNOW, MORE_SNOW, _save  # noqa: E402
from .ski import POWDER_IN  # noqa: E402

NEUTRAL_GREY = "#9a9892"
CLIMATE_ORDER = ["maritime", "transitional", "intermountain", "continental"]


def _regions_by_latitude(df: pd.DataFrame, meta: dict) -> list:
    regs = [r for r in df["region"].unique() if r in meta]
    return sorted(regs, key=lambda r: -meta[r].lat)


def fig_powder_days(comp: pd.DataFrame, meta: dict, out: Path, window: str = "Midwinter",
                    metric: str = "big_storm_days_per_window",
                    name: str = "fig_powder_days") -> Path:
    """Powder days in a normal winter against an El Niño winter, in days.

    One row per ski region, ordered north to south. The grey dot is the
    long-run average, the coloured dot is the El Niño average, and the bar
    between them is what El Niño costs or adds — read directly in days.
    """
    d = comp[(comp["window"] == window) & (comp["metric"] == metric)].dropna(subset=["mean_nino"])
    if d.empty:
        return out / f"{name}.png"
    # Sorted by effect, worst-hit first: the reader's question is "where does
    # El Nino hurt", and latitude is already the story of the map figure.
    d = d.sort_values("pct_change_nino").set_index("region").dropna(subset=["all_winters"])
    regions = list(d.index)
    y = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(regions) + 2.2))
    for i, r in enumerate(d.itertuples()):
        colour = LESS_SNOW if r.nino_minus_all < 0 else MORE_SNOW
        ax.plot([r.all_winters, r.mean_nino], [i, i], color=colour, lw=3.0,
                solid_capstyle="round", zorder=2, alpha=0.85)
        ax.scatter(r.all_winters, i, s=52, color="#fcfcfb", edgecolor=NEUTRAL_GREY,
                   linewidth=2.0, zorder=3)
        ax.scatter(r.mean_nino, i, s=62, color=colour, edgecolor="#fcfcfb",
                   linewidth=1.2, zorder=4)
        ax.text(max(r.all_winters, r.mean_nino) + 0.35, i,
                f"{r.nino_minus_all:+.1f} d ({r.pct_change_nino:+.0f} %)",
                va="center", fontsize=8.5, color=colour if abs(r.pct_change_nino) > 8 else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}" for r in regions], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"Powder days in the {window.lower()} window "
                  f"(days with at least {POWDER_IN:.0f} inches of new snow)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=52, color="#fcfcfb", edgecolor=NEUTRAL_GREY, linewidth=2.0,
               label="a normal winter")
    ax.scatter([], [], s=62, color=LESS_SNOW, label="an El Niño winter")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title(f"What El Niño costs in powder days — {window.lower()}", pad=12)
    return _save(fig, out, name)


def fig_season_shape(cube_means: dict, out: Path, name: str = "fig_season_shape",
                     unit: str = "in") -> Path:
    """The base through the winter, El Niño against La Niña, in inches.

    ``cube_means`` maps region -> {phase: array over day-of-water-year}. This
    shows both how much and *when*: a curve that separates in January and
    closes by April is a different problem from one that never closes.
    """
    regions = list(cube_means)
    ncol = 2
    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 2.9 * nrow), squeeze=False,
                             sharex=True)
    for ax, region in zip(axes.ravel(), regions):
        cm = cube_means[region]
        for phase, colour, lw in (("La Nina", MORE_SNOW, 2.0), ("Neutral", NEUTRAL_GREY, 1.6),
                                  ("El Nino", LESS_SNOW, 2.0)):
            v = cm.get(phase)
            if v is None:
                continue
            ax.plot(np.arange(len(v)), v, color=colour, lw=lw, label=phase.replace("Nino", "Niño")
                    .replace("Nina", "Niña"))
        nino, nina = cm.get("El Nino"), cm.get("La Nina")
        if nino is not None and nina is not None:
            ax.fill_between(np.arange(len(nino)), nino, nina,
                            where=np.isfinite(nino) & np.isfinite(nina),
                            color=LESS_SNOW, alpha=0.10, lw=0)
        ax.set_title(region, fontsize=9.5, loc="left")
        ax.set_ylabel(f"Base ({unit})", fontsize=9)
        ax.grid(color=GRID, lw=0.6)
    ticks = [(pd.Timestamp(2001 if m >= 10 else 2002, m, 1) - pd.Timestamp(2001, 10, 1)).days
             for m in (11, 12, 1, 2, 3, 4, 5)]
    for ax in axes.ravel():
        ax.set_xticks(ticks)
        ax.set_xticklabels(["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"])
        ax.set_xlim(ticks[0] - 10, ticks[-1] + 10)
    for ax in axes.ravel()[len(regions):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("How the base builds through the winter, by ENSO phase", y=1.002,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


def fig_strength(strength: pd.DataFrame, out: Path, regions: list | None = None,
                 name: str = "fig_strength_powder") -> Path:
    """Powder days by ENSO strength bin, in days.

    The question is whether a stronger event costs more. If the bars do not
    descend from La Niña through to very strong El Niño, it does not.
    """
    order = [("La Nina", "very strong"), ("La Nina", "strong"), ("La Nina", "moderate"),
             ("La Nina", "weak"), ("Neutral", "neutral"), ("El Nino", "weak"),
             ("El Nino", "moderate"), ("El Nino", "strong"), ("El Nino", "very strong")]
    d = strength.copy()
    if regions:
        d = d[d["region"].isin(regions)]
    if d.empty:
        return out / f"{name}.png"
    agg = (d.groupby(["phase", "strength"])
             .agg(mean=("mean", "mean"), allm=("all_winters", "mean"),
                  n=("n_winters", "max"), regions=("region", "nunique")).reset_index())
    keys = [k for k in order if ((agg["phase"] == k[0]) & (agg["strength"] == k[1])).any()]
    vals, ns, labels, colours = [], [], [], []
    base = float(agg["allm"].mean())
    for ph, stg in keys:
        row = agg[(agg["phase"] == ph) & (agg["strength"] == stg)].iloc[0]
        vals.append(row["mean"]); ns.append(int(row["n"]))
        labels.append(f"{ph.replace('Nino','Niño').replace('Nina','Niña')}\n{stg}")
        colours.append(MORE_SNOW if ph == "La Nina" else NEUTRAL_GREY if ph == "Neutral" else LESS_SNOW)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colours, width=0.68, edgecolor="#fcfcfb", linewidth=2)
    ax.axhline(base, color=TEXT, lw=1.2, ls="--")
    ax.text(len(vals) - 0.4, base, f" average winter: {base:.1f} days", va="bottom",
            ha="right", fontsize=8.5, color=TEXT)
    for xi, (v, n) in enumerate(zip(vals, ns)):
        ax.text(xi, v + 0.03 * max(vals), f"{v:.1f}", ha="center", fontsize=9,
                fontweight="bold", color=TEXT)
        ax.text(xi, 0.06 * max(vals), f"{n} winters", ha="center", fontsize=7.5, color="#fcfcfb")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(f"Powder days (at least {POWDER_IN:.0f} in of new snow)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Does a stronger El Niño cost more powder days?  No — the bars do not descend",
                 pad=12)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.grid(axis="x", visible=False)
    return _save(fig, out, name)


def fig_region_map(comp: pd.DataFrame, grid: pd.DataFrame, meta: dict, out: Path,
                   window: str = "Midwinter", metric: str = "big_storm_days_per_window",
                   name: str = "fig_region_map", max_lat: float = 52.0) -> Path:
    """Where El Niño costs powder days, as a percentage change.

    Alaska is drawn in its own inset rather than in the main frame: at 61 °N it
    would stretch the latitude axis so far that the contiguous ranges, which
    are the whole point, collapse into a smudge.
    """
    d = comp[(comp["window"] == window) & (comp["metric"] == metric)].dropna(
        subset=["pct_change_nino"])
    if d.empty:
        return out / f"{name}.png"
    # significance must come from the SAME window and metric, not any test
    sig_rows = grid[(grid.get("fdr_significant", False)) & (grid["window"] == window)
                    & (grid["metric"] == metric.replace("_per_window", ""))]
    sig = set(sig_rows["region"])

    d = d[d["region"].isin(meta)].copy()
    d["lat"] = [meta[r].lat for r in d["region"]]
    d["lon"] = [meta[r].lon for r in d["region"]]
    main, far = d[d["lat"] <= max_lat], d[d["lat"] > max_lat]

    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)

    def _draw(axis, rows, label=True):
        for k, r in enumerate(rows.sort_values("lon").itertuples()):
            marked = r.region in sig
            axis.scatter(r.lon, r.lat, s=230 if marked else 120, c=[r.pct_change_nino],
                         cmap=SNOW_DIVERGING, norm=norm,
                         edgecolor=TEXT if marked else "#bcbab4",
                         linewidth=2.2 if marked else 0.8, zorder=3)
            if not label:
                continue
            short = r.region.split(" (")[0]
            dy = 16 if k % 2 == 0 else -22          # alternate to reduce collisions
            axis.annotate(f"{short}\n{r.pct_change_nino:+.0f}%", (r.lon, r.lat),
                          fontsize=7.2, xytext=(0, dy), textcoords="offset points",
                          ha="center", va="bottom" if dy > 0 else "top", color=TEXT,
                          linespacing=1.15,
                          bbox=dict(boxstyle="round,pad=0.12", fc="#fcfcfb", ec="none",
                                    alpha=0.78))
    _draw(ax, main)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    lat_mid = float(main["lat"].mean())
    ax.set_aspect(1 / np.cos(np.deg2rad(lat_mid)))
    ax.set_ylim(main["lat"].min() - 1.4, main["lat"].max() + 2.6)
    ax.set_xlim(main["lon"].min() - 2.0, main["lon"].max() + 2.0)
    ax.grid(color=GRID, lw=0.6)
    if not far.empty:
        # Alaska sits 10 degrees north of everything else; naming it in a note
        # keeps the contiguous ranges legible instead of squashing them.
        note = "   ".join(f"{r.region.split(' (')[0]}: {r.pct_change_nino:+.0f}%"
                          for r in far.itertuples())
        ax.text(0.015, 0.975, f"Outside the frame — {note}", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec=GRID))
    sm = plt.cm.ScalarMappable(cmap=SNOW_DIVERGING, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Change in powder days in an El Niño winter (%)\nred = fewer, blue = more")
    ax.scatter([], [], s=230, facecolor="none", edgecolor=TEXT, linewidth=2.2,
               label="statistically significant")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title(f"Where El Niño costs powder days — {window.lower()}", pad=10)
    return _save(fig, out, name)


def fig_window_bars(comp: pd.DataFrame, out: Path, meta: dict | None = None,
                    metric: str = "big_storm_days_per_window",
                    name: str = "fig_window_change") -> Path:
    """Change in powder days by window of winter, split by latitude band.

    Averaging every region together would cancel the signal: the northern
    ranges lose powder days in an El Nino and the southern ones gain, so the
    mean is near zero and says nothing true about anywhere. Splitting at the
    ENSO node makes both halves visible.
    """
    from .ski import SKI_WINDOWS, REGION_META
    meta = meta or REGION_META
    d = comp[comp["metric"] == metric].dropna(subset=["pct_change_nino"]).copy()
    d = d[d["region"].isin(meta)]
    if d.empty:
        return out / f"{name}.png"
    d["lat"] = [meta[r].lat for r in d["region"]]
    bands = [("Northern ranges (north of 44° N)", d["lat"] >= 44),
             ("Central ranges (40–44° N)", (d["lat"] >= 40) & (d["lat"] < 44)),
             ("Southern ranges (south of 40° N)", d["lat"] < 40)]
    windows = [w for w, *_ in SKI_WINDOWS]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    width = 0.26
    x = np.arange(len(windows))
    for bi, (label, mask) in enumerate(bands):
        sub = d[mask]
        vals, bases = [], []
        for w in windows:
            row = sub[sub["window"] == w]
            vals.append(float(row["pct_change_nino"].mean()) if len(row) else np.nan)
            bases.append(float(row["all_winters"].mean()) if len(row) else np.nan)
        off = (bi - 1) * width
        colours = [LESS_SNOW if v < 0 else MORE_SNOW for v in vals]
        alpha = [1.0, 0.72, 0.5][bi]
        bars = ax.bar(x + off, vals, width=width * 0.92, color=colours, alpha=alpha,
                      edgecolor="#fcfcfb", linewidth=1.5)
        for xi, (v, b) in enumerate(zip(vals, bases)):
            if not np.isfinite(v):
                continue
            ax.text(xi + off, v + (1.2 if v >= 0 else -1.2), f"{v:+.0f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                    color=TEXT, fontweight="bold")
        ax.bar(np.nan, np.nan, color=MUTED, alpha=alpha, label=label)
    ax.axhline(0, color=TEXT, lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(windows, fontsize=10)
    ax.set_ylabel("Change in powder days in an El Niño winter (%)")
    ax.set_ylim(min(-32, ax.get_ylim()[0]), max(32, ax.get_ylim()[1]))
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=1)
    ax.set_title("El Niño's effect on powder days, by window of winter and latitude", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.grid(axis="x", visible=False)
    return _save(fig, out, name)
