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
# Snow convention, used on every ENSO figure: LESS SNOW IS RED, more snow is
# blue. A negative correlation with the ONI means El Nino brings less snow, so
# the scale is reversed relative to the generic diverging ramp above.
SNOW_DIVERGING = LinearSegmentedColormap.from_list(
    "lesssnow_red", ["#7a1f1e", "#e34948", "#f0efec", "#2a78d6", "#0d366b"])
LESS_SNOW, MORE_SNOW = "#e34948", "#2a78d6"
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
    ax.scatter(d.loc[~sig, "longitude"], d.loc[~sig, "latitude"], c=d.loc[~sig, "r"], cmap=SNOW_DIVERGING,
               norm=norm, s=22, edgecolor="#bcbab4", linewidth=0.5, label="p ≥ 0.05")
    s = ax.scatter(d.loc[sig, "longitude"], d.loc[sig, "latitude"], c=d.loc[sig, "r"], cmap=SNOW_DIVERGING,
                   norm=norm, s=46, edgecolor=TEXT, linewidth=0.8, label="p < 0.05")
    cb = fig.colorbar(s, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Correlation with DJF ONI\n(red = El Niño means less snow)")
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


def fig_daily_curve(curve: pd.DataFrame, climatology: np.ndarray, out: Path,
                    name: str = "fig8_daily_enso_curve",
                    regions=("MT", "ID", "WY", "CO", "UT")) -> Path:
    """The ENSO signal day by day through the water year.

    Top panel: mean snowpack climatology, so the reader can see where in the
    season the snow actually is. Bottom panel: correlation of the regional
    standardised anomaly with the DJF ONI on each day, thick where the
    Brown & Harper (2026) calibrated 2σ bounds exclude zero. Both share the
    water-year axis; neither panel carries a second scale.
    """
    from .daily import PERIODS
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.12})

    # period bands, drawn first so everything sits on top of them
    def _day(m, d):
        y = 2001 if m >= 10 else 2002
        return (pd.Timestamp(y, m, d) - pd.Timestamp(2001, 10, 1)).days
    for i, (label, (m0, d0), (m1, d1)) in enumerate(PERIODS):
        a, b = _day(m0, d0), _day(m1, d1)
        for axis in (ax0, ax):
            axis.axvspan(a, b, color=MUTED, alpha=0.05 if i % 2 else 0.10, lw=0)
        ax0.text((a + b) / 2, 1.02, label.replace(" ", "\n"), transform=ax0.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.15)

    ax0.plot(np.arange(len(climatology)), climatology, color=TEXT, lw=1.8)
    ax0.fill_between(np.arange(len(climatology)), 0, climatology, color=MUTED, alpha=0.12, lw=0)
    ax0.set_ylabel("Mean SWE (mm)")
    ax0.set_ylim(0, None)

    x_lo, x_hi = _day(10, 15), _day(6, 20)
    label_at = _day(6, 1)          # label inside the axes, where curves are separated
    for st in regions:
        d = curve[(curve["region"] == st)].sort_values("doy")
        d = d[(d["doy"] >= x_lo) & (d["doy"] <= x_hi)]
        if d.empty:
            continue
        c = STATE_COLORS.get(st, TEXT)
        ax.plot(d["doy"], d["r"], color=c, lw=1.4, alpha=0.55)
        sig = d[d["r_sig_cal"]]
        # thick segments where the relation clears the calibrated 2σ test
        for _, block in sig.groupby((sig["doy"].diff() != 1).cumsum()):
            ax.plot(block["doy"], block["r"], color=c, lw=3.2, solid_capstyle="round")
        near = d.iloc[(d["doy"] - label_at).abs().argsort().iloc[0]]
        ax.annotate(STATE_NAMES.get(st, st), (near["doy"], near["r"]), color=c, fontsize=9,
                    fontweight="bold", xytext=(7, 0), textcoords="offset points", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))

    ax.axhline(0, color=TEXT, lw=0.9)
    ax.set_ylabel("Correlation with DJF ONI\n(below zero: El Niño means less snow)")
    ax.set_xlabel("Water year")
    ticks = [(pd.Timestamp(2001 if m >= 10 else 2002, m, 1) - pd.Timestamp(2001, 10, 1)).days
             for m in (10, 11, 12, 1, 2, 3, 4, 5, 6)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"])
    ax.set_xlim(x_lo, x_hi + 26)
    ax.plot([], [], color=MUTED, lw=3.2, label="significant (calibrated 2σ)")
    ax.plot([], [], color=MUTED, lw=1.4, alpha=0.55, label="not significant")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax0.set_title("When in the season does El Niño act on the snowpack?", pad=26)
    return _save(fig, out, name)


def fig_ski_heatmap(grid: pd.DataFrame, out: Path, name: str = "fig9_ski_region_window",
                    metrics=("mean_swe", "storm_days", "big_storm_days"),
                    value: str = "signed_r2") -> Path:
    """Ski region x window x metric, as small multiples of one measure.

    Rows are ski regions ordered north to south, so the ENSO dipole reads as a
    vertical gradient. Columns are the windows that carry a ski season. Colour
    is the correlation with the DJF ONI on one diverging scale with a neutral
    midpoint; a ring marks a test that survives false-discovery control across
    the whole grid, and every cell is labelled, so colour is never the only
    encoding.
    """
    from .ski import SKI_REGIONS, SKI_WINDOWS, METRICS
    labels = dict(METRICS)
    have = set(grid["region"])
    regions = [r[0] for r in sorted(SKI_REGIONS, key=lambda r: -r[2]) if r[0] in have]
    windows = [w[0] for w in SKI_WINDOWS]
    metrics = [m for m in metrics if m in set(grid["metric"])]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    lim = 0.25 if value == "signed_r2" else 0.6
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for ax, metric in zip(axes, metrics):
        d = grid[grid["metric"] == metric]
        M = np.full((len(regions), len(windows)), np.nan)
        for i, reg in enumerate(regions):
            for j, win in enumerate(windows):
                s = d[(d["region"] == reg) & (d["window"] == win)]
                if not s.empty:
                    M[i, j] = s[value].iloc[0]
        ax.imshow(M, cmap=SNOW_DIVERGING, norm=norm, aspect="auto")
        for i, reg in enumerate(regions):
            for j, win in enumerate(windows):
                s = d[(d["region"] == reg) & (d["window"] == win)]
                if s.empty or not np.isfinite(M[i, j]):
                    continue
                s = s.iloc[0]
                strong = abs(M[i, j]) > 0.62 * lim
                txt = f"{100 * abs(M[i, j]):.0f}%" if value == "signed_r2" else f"{M[i, j]:+.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if strong else TEXT,
                        fontweight="bold" if s["fdr_significant"] else "normal")
                if s["fdr_significant"]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor=TEXT, lw=2.2))
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels([w.replace(" ", "\n") for w in windows], fontsize=8.5)
        ax.set_title(labels.get(metric, metric), fontsize=9.5)
        ax.set_xticks(np.arange(len(windows)) - .5, minor=True)
        ax.set_yticks(np.arange(len(regions)) - .5, minor=True)
        ax.grid(which="minor", color="#fcfcfb", lw=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
    axes[0].set_yticks(range(len(regions)))
    axes[0].set_yticklabels(regions, fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=SNOW_DIVERGING, norm=norm)
    cb = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.015)
    cb.set_label("Variance explained by the ONI (r²), signed\n"
                 "RED = El Niño means LESS snow   BLUE = more snow", fontsize=8.5)
    fig.suptitle("El Niño and the ski season, by region and window of winter", y=0.99,
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "Cells show r² as a percentage of year-to-year variance explained. "
             "Bold value in a ring: survives false-discovery control across all "
             f"{int(grid['n_tests'].iloc[0]) if 'n_tests' in grid else len(grid)} tests. "
             "Regions ordered north (top) to south (bottom).",
             ha="center", fontsize=8, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_distributions(dists: list, out: Path, name: str = "fig10_distributions") -> Path:
    """Probability distributions behind the headline tests.

    For each test, two distributions on one axis: the Brown & Harper (2026)
    subsampling distribution of the estimate (how well pinned down it is, with
    the Gaussian they fit to it) and the permutation null (what chance alone
    produces from the same data). Separation between them IS the significance,
    shown rather than asserted. x is signed r², so distance from zero is
    variance explained and the side is the direction.
    """
    dists = [d for d in dists if d]
    if not dists:
        return out / f"{name}.png"
    ncol = 2
    nrow = int(np.ceil(len(dists) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, d in zip(axes.ravel(), dists):
        null, sub = d["null_signed_r2"], d["sub_signed_r2"]
        lo = min(null.min(), sub.min(), -0.02)
        hi = max(null.max(), sub.max(), 0.02)
        bins = np.linspace(lo, hi, 70)
        ax.hist(null, bins=bins, color=MUTED, alpha=0.45, density=True, lw=0,
                label="chance alone (permutation null)")
        colour = LESS_SNOW if d["r_obs"] < 0 else MORE_SNOW
        ax.hist(sub, bins=bins, color=colour, alpha=0.75, density=True, lw=0,
                label="estimate (80 % subsamples)")
        mu, sd = float(np.mean(sub)), float(np.std(sub, ddof=1))
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                color=TEXT, lw=1.4, ls="--", label="fitted normal")
        ax.axvline(0, color=TEXT, lw=0.9)
        ax.axvline(d["signed_r2_obs"], color=colour, lw=2.0)
        ax.set_yticks([])
        ax.set_xlabel("signed r²   (left = El Niño means less snow)", fontsize=8.5)
        r2 = abs(d["signed_r2_obs"])
        if d["p_perm"] >= 0.05 or r2 < 0.02:
            verdict = "no detectable effect"
        else:
            verdict = f"El Niño = {'less' if d['r_obs'] < 0 else 'more'} snow"
        ax.set_title(f"{d['region']} — {d['window']}, {d['metric']}\n"
                     f"r² = {r2:.0%} of year-to-year variance · {verdict} · "
                     f"n = {d['n']} · p = {d['p_perm']:.4f}", fontsize=8.8, loc="left")
    for ax in axes.ravel()[len(dists):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Probability distributions behind each result", y=1.005,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)
