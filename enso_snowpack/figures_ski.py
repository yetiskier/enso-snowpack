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
                     unit: str = "in of snow depth") -> Path:
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
        ax.set_ylabel(f"Base\n({unit})", fontsize=9)
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
    fig.suptitle("How the base builds through the winter, by ENSO phase\n"
                 "worst-hit regions at the top, a region that gains at the bottom right",
                 y=1.004,
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


def fig_significant_inches(sig: pd.DataFrame, out: Path,
                           name: str = "fig_significant_inches") -> Path:
    """Only what survives false-discovery control, measured in inches.

    Two panels, because inches of falling snow and inches of standing base are
    different quantities and must not share an axis. Each row is one
    region-window that cleared the whole 800-test grid; the bar runs from a
    normal winter to an El Niño one, so its length IS the loss or gain.
    """
    # Three separate axes. Inches of falling snow, inches of standing snow and
    # inches of water in that snow are different quantities; sharing one axis
    # would invite a 77-inch depth and a 27-inch water equivalent to be read
    # as comparable, which they are not.
    panels = [
        ("Total new snow that fell in the window", sig[sig["metric"] == "new_snow_in_total"],
         "new snow (in)"),
        ("Standing base — measured snow depth", sig[sig["metric"] == "mean_depth_in"],
         "snow depth (in)"),
        ("Standing base — snow water equivalent", sig[sig["metric"] == "mean_swe_in"],
         "water equivalent (in)"),
    ]
    panels = [(t, d, u) for t, d, u in panels if not d.empty]
    if not panels:
        return out / f"{name}.png"
    panels = [(t, d, u) for t, d, u in panels if not d.empty]
    heights = [max(1, len(d)) for _, d, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.0, 0.52 * sum(heights) + 3.0),
                             gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    for ax, (title, d, unit) in zip(axes, panels):
        d = d.sort_values("nino_minus_all")
        labels = []
        for i, r in enumerate(d.itertuples()):
            colour = LESS_SNOW if r.nino_minus_all < 0 else MORE_SNOW
            ax.plot([r.all_winters, r.mean_nino], [i, i], color=colour, lw=3.4,
                    solid_capstyle="round", zorder=2, alpha=0.9)
            ax.scatter(r.all_winters, i, s=54, color="#fcfcfb", edgecolor=NEUTRAL_GREY,
                       linewidth=2.0, zorder=3)
            ax.scatter(r.mean_nino, i, s=64, color=colour, edgecolor="#fcfcfb",
                       linewidth=1.2, zorder=4)
            ax.text(max(r.all_winters, r.mean_nino) + 0.02 * d["all_winters"].max(), i,
                    f"{r.nino_minus_all:+.1f} in  ({r.pct_change_nino:+.0f} %)   r² {r.r2:.0%}",
                    va="center", fontsize=8.5, color=colour)
            labels.append(f"{r.region.split(' (')[0]} — {r.window}")
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(left=0, right=d[["all_winters", "mean_nino"]].max().max() * 1.42)
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10, loc="left", pad=6)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
    axes[0].scatter([], [], s=54, color="#fcfcfb", edgecolor=NEUTRAL_GREY, linewidth=2.0,
                    label="a normal winter")
    axes[0].scatter([], [], s=64, color=LESS_SNOW, label="an El Niño winter")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("What El Niño changes, in inches — only results that survive "
                 "false-discovery control", y=1.005, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


def fig_significant_days(sig: pd.DataFrame, out: Path,
                         name: str = "fig_significant_days") -> Path:
    """The same surviving results counted in days rather than inches."""
    d = sig[sig["unit"] == "days"].copy()
    if d.empty:
        return out / f"{name}.png"
    label = {"big_storm_days_per_window": "powder days (≥6 in)",
             "storm_days_per_window": "storm days (≥2 in)",
             "days_with_base_per_window": "days with a skiable base"}
    d["what"] = d["metric"].map(label).fillna(d["metric"])
    d = d.sort_values("pct_change_nino")
    fig, ax = plt.subplots(figsize=(11.0, 0.52 * len(d) + 2.2))
    for i, r in enumerate(d.itertuples()):
        colour = LESS_SNOW if r.nino_minus_all < 0 else MORE_SNOW
        ax.plot([r.all_winters, r.mean_nino], [i, i], color=colour, lw=3.4,
                solid_capstyle="round", zorder=2, alpha=0.9)
        ax.scatter(r.all_winters, i, s=54, color="#fcfcfb", edgecolor=NEUTRAL_GREY,
                   linewidth=2.0, zorder=3)
        ax.scatter(r.mean_nino, i, s=64, color=colour, edgecolor="#fcfcfb",
                   linewidth=1.2, zorder=4)
        ax.text(max(r.all_winters, r.mean_nino) + 0.7, i,
                f"{r.nino_minus_all:+.1f} d  ({r.pct_change_nino:+.0f} %)   r² {r.r2:.0%}",
                va="center", fontsize=8.5, color=colour)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.region.split(' (')[0]} — {r.window}  ({r.what})"
                        for r in d.itertuples()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=d[["all_winters", "mean_nino"]].max().max() * 1.45)
    ax.set_xlabel("days in the window")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=54, color="#fcfcfb", edgecolor=NEUTRAL_GREY, linewidth=2.0,
               label="a normal winter")
    ax.scatter([], [], s=64, color=LESS_SNOW, label="an El Niño winter")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("What El Niño changes, in days — only results that survive "
                 "false-discovery control", pad=10)
    return _save(fig, out, name)


def fig_strength_significance(sig: pd.DataFrame, out: Path,
                              name: str = "fig_strength_significance") -> Path:
    """Is there a strength signal anywhere? A p-value histogram answers it.

    If event magnitude mattered somewhere, the p-values from that family of
    tests would pile up near zero. If it matters nowhere, they are flat — a
    uniform distribution is exactly what pure noise produces. The dashed line
    is that uniform expectation, so any real excess would rise above it at the
    left edge.

    The second panel is the honesty check: with only ~16 winters of a given
    phase per region, a within-phase test can only find a very large
    correlation, so a flat histogram there means "not found", not "not there".
    The nested test, run on every winter, has the power to say more.
    """
    if sig.empty:
        return out / f"{name}.png"
    titles = {"within_nino": "Within El Niño winters only\ndoes a bigger event mean less snow?",
              "within_nina": "Within La Niña winters only",
              "nested": "Does adding magnitude to phase\nexplain more? (every winter)"}
    tests = [t for t in ("within_nino", "within_nina", "nested") if t in set(sig["test"])]
    fig, axes = plt.subplots(2, len(tests), figsize=(4.1 * len(tests), 6.4),
                             gridspec_kw={"height_ratios": [2, 1]}, squeeze=False)
    bins = np.linspace(0, 1, 21)
    for j, test in enumerate(tests):
        g = sig[sig["test"] == test]
        ax = axes[0][j]
        ax.hist(g["p"], bins=bins, color=NEUTRAL_GREY, alpha=0.75, lw=0)
        ax.axhline(len(g) / (len(bins) - 1), color=TEXT, lw=1.6, ls="--",
                   label="what pure chance gives")
        obs = int((g["p"] < 0.05).sum())
        exp = 0.05 * len(g)
        ax.set_title(titles.get(test, test), fontsize=9.5, loc="left")
        ax.set_xlabel("permutation p-value", fontsize=9)
        if j == 0:
            ax.set_ylabel("number of tests")
        colour = LESS_SNOW if obs > exp else TEXT
        ax.text(0.97, 0.94, f"{len(g)} tests\n{obs} at p<0.05\n({exp:.0f} expected by chance)\n"
                            f"0 survive FDR",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=colour,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec=GRID))
        ax.grid(axis="y", color=GRID, lw=0.6); ax.grid(axis="x", visible=False)

        axp = axes[1][j]
        mdr = g["min_detectable_r"].dropna()
        if len(mdr):
            axp.hist(mdr, bins=np.linspace(0, 1, 25), color=MORE_SNOW, alpha=0.7, lw=0)
            axp.axvline(float(mdr.median()), color=TEXT, lw=1.6)
            axp.text(float(mdr.median()) + 0.02, axp.get_ylim()[1] * 0.85,
                     f"median {mdr.median():.2f}", fontsize=8.5, color=TEXT)
        axp.set_xlabel("smallest correlation this sample\ncould reliably find", fontsize=8.5)
        axp.set_xlim(0, 1)
        if j == 0:
            axp.set_ylabel("tests")
        axp.grid(axis="y", color=GRID, lw=0.6); axp.grid(axis="x", visible=False)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Does the strength of an El Niño matter anywhere? No.", y=1.0,
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


def fig_region_pdf_map(dists: dict, meta: dict, out: Path,
                       name: str = "fig_region_pdf_map", window: str = "Midwinter",
                       span: float = 70.0, glyph_w: float = 1.25, glyph_h: float = 1.0,
                       max_lat: float = 52.0, states: list | None = None) -> Path:
    """Every ski region on the map of the West, carrying its whole distribution.

    Forty-odd ranges will not take inline labels — they collide and the map
    becomes unreadable — so each glyph carries a number and the names live in a
    legend beside it, ordered north to south so the list reads down the map.

    Each glyph is a smoothed probability density of the change in powder days,
    anchored at its range on one shared horizontal scale, with a dashed
    hairline at zero. A density astride that line is a region where El Nino
    does nothing measurable. Mass left of zero is red (fewer powder days),
    right is blue.
    """
    from scipy import stats as _st
    items = [(r, d) for r, d in dists.items() if r in meta and meta[r].lat <= max_lat]
    far = [(r, d) for r, d in dists.items() if r in meta and meta[r].lat > max_lat]
    if not items:
        return out / f"{name}.png"
    items.sort(key=lambda kv: (-meta[kv[0]].lat, meta[kv[0]].lon))
    lats = [meta[r].lat for r, _ in items]
    lons = [meta[r].lon for r, _ in items]
    x0, x1 = min(lons) - 2.4, max(lons) + 2.4
    y0, y1 = min(lats) - 2.2, max(lats) + 1.8

    n_leg = len(items) + len(far)
    leg_cols = 2 if n_leg > 22 else 1
    fig = plt.figure(figsize=(11.2 + 3.4 * leg_cols, 10.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[11.2, 3.4 * leg_cols], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")

    ax.set_facecolor("#eef2f6")
    if states is None:
        try:
            states = load_states()
        except Exception:
            states = []
    if states:
        draw_states(ax, states, (x0, x1, y0, y1), land="#fbfaf8", edge="#c3ccd6",
                    label_color="#8a97a5", label_size=11)

    grid = np.linspace(-span, span, 201)
    curves, peak = {}, 0.0
    for r, d in items:
        v = np.clip(d["values"], -span, span)
        if v.std() < 1e-6:
            dens = np.zeros_like(grid)
            dens[len(grid) // 2] = 1.0
        else:
            dens = _st.gaussian_kde(v, bw_method=0.35)(grid)
        curves[r] = dens
        peak = max(peak, dens.max())

    for i, (r, d) in enumerate(items, start=1):
        m = meta[r]
        dens = curves[r]
        x = m.lon + grid / span * (glyph_w / 2)
        y = m.lat + dens / peak * glyph_h * 0.8
        base = np.full_like(x, m.lat)
        neg, pos = grid < 0, grid >= 0
        ax.fill_between(x[neg], base[neg], y[neg], color=LESS_SNOW, alpha=0.88, lw=0, zorder=4)
        ax.fill_between(x[pos], base[pos], y[pos], color=MORE_SNOW, alpha=0.88, lw=0, zorder=4)
        ax.plot(x, y, color=TEXT, lw=0.6, zorder=5)
        ax.plot([x[0], x[-1]], [m.lat, m.lat], color=TEXT, lw=0.6, zorder=5)
        ax.plot([m.lon, m.lon], [m.lat, m.lat + glyph_h * 0.9], color=TEXT, lw=0.9,
                ls=(0, (2.2, 1.8)), zorder=6)
        straddles = d.get("straddles_zero", d["p_zero"] > 0.05)
        ax.annotate(str(i), (m.lon, m.lat), fontsize=7.4, fontweight="bold",
                    xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                    color="#fcfcfb", zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.22",
                              fc=MUTED if straddles else (LESS_SNOW if d["observed"] < 0 else MORE_SNOW),
                              ec="#fcfcfb", lw=0.9))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lats)))))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c3ccd6")

    # key glyph, drawn to the same scale, in the empty south-west corner
    kx, ky = x0 + 1.4, y0 + 0.7
    kh = np.exp(-0.5 * (grid / 20) ** 2)
    xk = kx + grid / span * (glyph_w / 2)
    yk = ky + kh / kh.max() * glyph_h * 0.8
    ax.fill_between(xk[grid < 0], ky, yk[grid < 0], color=LESS_SNOW, alpha=0.6, lw=0, zorder=7)
    ax.fill_between(xk[grid >= 0], ky, yk[grid >= 0], color=MORE_SNOW, alpha=0.6, lw=0, zorder=7)
    ax.plot(xk, yk, color=TEXT, lw=0.6, zorder=7)
    ax.plot([kx, kx], [ky, ky + glyph_h * 0.9], color=TEXT, lw=0.9, ls=(0, (2.2, 1.8)), zorder=7)
    ax.annotate("each glyph, left to right:\n"
                "−70 %   ·   no change   ·   +70 %\n"
                "height is how likely that value is",
                (kx + glyph_w * 0.6, ky), fontsize=7.6, ha="left", va="bottom",
                color=MUTED, linespacing=1.4, zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#d7dee6", alpha=0.92))

    # ---- the legend: number, name, value, ordered north to south
    entries = [(i, r, d, False) for i, (r, d) in enumerate(items, start=1)]
    entries += [(None, r, d, True) for r, d in far]
    per_col = int(np.ceil(len(entries) / leg_cols))
    lax.set_xlim(0, leg_cols)
    lax.set_ylim(0, per_col + 2.4)
    lax.text(0, per_col + 1.9, "Ski regions, north to south",
             fontsize=10.5, fontweight="bold", color=TEXT, va="top",
             family="DejaVu Sans")
    lax.text(0, per_col + 1.15, "change in powder days · n.s. = interval includes zero",
             fontsize=8, color=MUTED, va="top")
    for k, (num, r, d, offmap) in enumerate(entries):
        col, row = divmod(k, per_col)
        ypos = per_col - row - 0.4
        straddles = d.get("straddles_zero", d["p_zero"] > 0.05)
        colour = MUTED if straddles else (LESS_SNOW if d["observed"] < 0 else MORE_SNOW)
        if num is not None:
            lax.plot(col + 0.045, ypos, marker="o", ms=11.5, color=colour,
                     markeredgecolor="#fcfcfb", markeredgewidth=0.9, clip_on=False)
            lax.text(col + 0.045, ypos, str(num), fontsize=6.8, fontweight="bold",
                     color="#fcfcfb", ha="center", va="center", clip_on=False)
        label = r.split(" (")[0]
        if len(label) > 30:
            label = label[:29] + "…"
        suffix = "  (not on map)" if offmap else ""
        lax.text(col + 0.10, ypos, label + suffix, fontsize=8.1, color=TEXT,
                 ha="left", va="center", clip_on=False)
        lax.text(col + 0.95, ypos, f"{d['observed']:+.0f}%" + ("  n.s." if straddles else ""),
                 fontsize=8.1, color=colour, ha="right", va="center", clip_on=False,
                 fontweight="normal" if straddles else "bold")

    n_ns = sum(1 for _, d in items if d.get("straddles_zero", d["p_zero"] > 0.05))
    fig.suptitle("The change in powder days in every ski region, with its uncertainty — "
                 + window.lower(), y=0.965, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.025, "Each glyph is a probability density from 10,000 resamples of the record, "
             "calibrated so its width is an honest sampling distribution. "
             f"{n_ns} of {len(items)} regions straddle zero and are marked n.s.",
             ha="center", fontsize=8.6, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


US_STATES_URL = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
                 "master/data/geojson/us-states.json")
STATE_ABBR = {"Washington": "WA", "Oregon": "OR", "California": "CA", "Nevada": "NV",
              "Idaho": "ID", "Montana": "MT", "Wyoming": "WY", "Utah": "UT",
              "Colorado": "CO", "Arizona": "AZ", "New Mexico": "NM", "South Dakota": "SD",
              "North Dakota": "ND", "Nebraska": "NE", "Kansas": "KS", "Oklahoma": "OK",
              "Texas": "TX", "Alaska": "AK"}


def load_states(path: Path | None = None) -> list:
    """State boundary polygons as plain coordinate rings.

    Cached in the repo so a figure can be redrawn offline; fetched once if the
    cache is missing. Returns ``[(name, [ring, ...]), ...]`` where each ring is
    an (N, 2) array of lon/lat — enough to draw with matplotlib alone, without
    a GIS stack.
    """
    import json
    path = path or Path("data/geo/us-states.json")
    if not path.exists():
        import requests
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(requests.get(US_STATES_URL, timeout=60).text, encoding="utf-8")
    gj = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in gj["features"]:
        name = f["properties"].get("name", "")
        geom = f["geometry"]
        rings = []
        if geom["type"] == "Polygon":
            rings = [np.asarray(r, dtype=float) for r in geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                rings += [np.asarray(r, dtype=float) for r in poly]
        if rings:
            out.append((name, rings))
    return out


def draw_states(ax, states: list, extent: tuple, land: str, edge: str,
                label_color: str, label_size: float = 9.5) -> None:
    """Draw state polygons and their two-letter labels inside ``extent``."""
    from matplotlib.patches import Polygon as MplPolygon
    x0, x1, y0, y1 = extent
    for name, rings in states:
        drew = False
        for ring in rings:
            if ring[:, 0].max() < x0 - 6 or ring[:, 0].min() > x1 + 6:
                continue
            if ring[:, 1].max() < y0 - 6 or ring[:, 1].min() > y1 + 6:
                continue
            ax.add_patch(MplPolygon(ring, closed=True, facecolor=land, edgecolor=edge,
                                    lw=0.9, zorder=1))
            drew = True
        if not drew:
            continue
        abbr = STATE_ABBR.get(name)
        if not abbr:
            continue
        big = max(rings, key=lambda r: len(r))
        cx, cy = float(big[:, 0].mean()), float(big[:, 1].mean())
        if x0 + 0.6 < cx < x1 - 0.6 and y0 + 0.6 < cy < y1 - 0.6:
            ax.text(cx, cy, abbr, fontsize=label_size, color=label_color, ha="center",
                    va="center", fontweight="bold", alpha=0.55, zorder=2)
