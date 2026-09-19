"""Resampling-based significance for the ENSO–snowpack relation.

Why resampling at all: 40 SNOTEL winters (and ~15 El Niño winters) is a
small, serially correlated, non-Gaussian sample. Parametric p-values on r
assume independent Gaussian years and are optimistic; the bootstrap and
permutation machinery here does not.

Three schemes, all operating on WATER YEARS as the exchangeable unit (never
stations — stations within a state are strongly co-varying, so resampling
them inflates the effective sample size):

``permutation``     Null distribution of the statistic obtained by shuffling
                    the ENSO index across water years (the snowpack series is
                    untouched, so its own autocorrelation and skew are kept).
                    Gives a p-value that does not assume normality.
``block_bootstrap`` Moving-block bootstrap of (ONI, snowpack) year pairs with
                    block length L (default 2, the e-folding scale of ENSO's
                    year-to-year persistence), for a CI on r, the slope and
                    composite differences that respects serial dependence.
``two_level``       Resample water years, then within each drawn year
                    resample the contributing stations, so the CI also carries
                    the station-sampling uncertainty of the state mean.

``brown_harper_2026`` is the hook for the modified bootstrap of
Brown & Harper (2026, HESS 30, 5735–5748, doi:10.5194/hess-30-5735-2026).
The method text was not reachable from the session that wrote this file, so
the hook raises until it is transcribed — see README "Bootstrap".

Field significance for the per-station map uses the Benjamini–Hochberg
false-discovery-rate control recommended for gridded/climate fields by
Wilks (2016, BAMS, doi:10.1175/BAMS-D-15-00267.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BootResult:
    region: str
    metric: str
    scheme: str
    n: int
    r_obs: float
    p_perm: float                 # two-sided permutation p for r
    r_ci_low: float
    r_ci_high: float
    slope_obs: float
    slope_ci_low: float
    slope_ci_high: float
    diff_nino_rest_obs: float     # mean(El Niño) − mean(rest)
    diff_ci_low: float
    diff_ci_high: float
    p_perm_diff: float            # permutation p for that difference
    n_resamples: int

    def as_dict(self):
        return self.__dict__.copy()


def _stat_r(x, y):
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _stat_slope(x, y):
    if x.std() == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _stat_diff(x, y, thr=0.5):
    m = x >= thr
    if m.sum() < 2 or (~m).sum() < 2:
        return np.nan
    return float(y[m].mean() - y[~m].mean())


def permutation_p(x: np.ndarray, y: np.ndarray, stat, n_perm: int, rng) -> tuple[float, float]:
    """Two-sided p = P(|stat_perm| >= |stat_obs|) with the +1 correction."""
    obs = stat(x, y)
    if not np.isfinite(obs):
        return obs, np.nan
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stat(rng.permutation(x), y)
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return obs, float(p)


def moving_block_indices(n: int, block: int, rng) -> np.ndarray:
    """Indices of one moving-block bootstrap resample of length n."""
    if block <= 1 or n <= block:
        return rng.integers(0, n, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def block_bootstrap_ci(x, y, stat, n_boot: int, block: int, rng, alpha=0.05):
    vals = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = moving_block_indices(n, block, rng)
        vals[i] = stat(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def two_level_bootstrap_ci(years: np.ndarray, x_by_year: dict, station_values: dict,
                           stat, n_boot: int, rng, alpha=0.05):
    """``station_values[year]`` is the array of station anomalies for that
    year; each resample draws years with replacement, then stations within
    each drawn year with replacement, and recomputes the state mean."""
    vals = np.empty(n_boot)
    n = len(years)
    for i in range(n_boot):
        ys = years[rng.integers(0, n, n)]
        xs = np.array([x_by_year[y] for y in ys])
        ms = np.empty(n)
        for k, y in enumerate(ys):
            sv = station_values[y]
            ms[k] = sv[rng.integers(0, len(sv), len(sv))].mean()
        vals[i] = stat(xs, ms)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def brown_harper_2026(x: np.ndarray, y: np.ndarray, **kwargs):
    """Modified bootstrap of Brown & Harper (2026, HESS 30, 5735).

    NOT YET IMPLEMENTED: the paper (and its EGUsphere preprint
    egusphere-2025-4971) could not be fetched from the authoring session.
    Transcribe the resampling scheme from the Methods here and register it
    in :func:`run_bootstrap` under scheme="brown_harper_2026".
    """
    raise NotImplementedError(
        "Brown & Harper (2026) modified bootstrap: method text still to be transcribed "
        "from doi:10.5194/hess-30-5735-2026 — see README 'Bootstrap'.")


def run_bootstrap(series: pd.DataFrame, enso: pd.DataFrame, region: str, metric: str,
                  index_col: str = "oni_djf", scheme: str = "block_bootstrap",
                  n_boot: int = 5000, n_perm: int = 5000, block: int = 2,
                  station_table: pd.DataFrame | None = None, seed: int = 0) -> BootResult | None:
    """``series``: water_year, value. ``station_table`` (for ``two_level``):
    water_year, value per station-year."""
    rng = np.random.default_rng(seed)
    d = series.merge(enso[["water_year", index_col]], on="water_year").dropna(
        subset=["value", index_col]).sort_values("water_year")
    n = len(d)
    if n < 8:
        return None
    x = d[index_col].to_numpy(float)
    y = d["value"].to_numpy(float)

    r_obs, p_r = permutation_p(x, y, _stat_r, n_perm, rng)
    d_obs, p_d = permutation_p(x, y, _stat_diff, n_perm, rng)
    slope_obs = _stat_slope(x, y)

    if scheme == "permutation" or scheme == "block_bootstrap":
        b = block if scheme == "block_bootstrap" else 1
        r_lo, r_hi = block_bootstrap_ci(x, y, _stat_r, n_boot, b, rng)
        s_lo, s_hi = block_bootstrap_ci(x, y, _stat_slope, n_boot, b, rng)
        d_lo, d_hi = block_bootstrap_ci(x, y, _stat_diff, n_boot, b, rng)
    elif scheme == "two_level":
        if station_table is None:
            raise ValueError("two_level needs station_table")
        st = station_table.merge(enso[["water_year", index_col]], on="water_year").dropna(
            subset=["value", index_col])
        years = np.array(sorted(set(st["water_year"]) & set(d["water_year"])))
        xby = {yv: float(st.loc[st.water_year == yv, index_col].iloc[0]) for yv in years}
        sv = {yv: st.loc[st.water_year == yv, "value"].to_numpy(float) for yv in years}
        r_lo, r_hi = two_level_bootstrap_ci(years, xby, sv, _stat_r, n_boot, rng)
        s_lo, s_hi = two_level_bootstrap_ci(years, xby, sv, _stat_slope, n_boot, rng)
        d_lo, d_hi = two_level_bootstrap_ci(years, xby, sv, _stat_diff, n_boot, rng)
    elif scheme == "brown_harper_2026":
        return brown_harper_2026(x, y)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    return BootResult(region, metric, scheme, n, r_obs, p_r, r_lo, r_hi, slope_obs, s_lo, s_hi,
                      d_obs, d_lo, d_hi, p_d, n_boot)


def fdr_field_significance(p: np.ndarray, alpha_fdr: float = 0.10) -> np.ndarray:
    """Benjamini–Hochberg: boolean mask of stations significant at the
    controlled false-discovery rate (Wilks 2016 recommends αFDR = 2·α)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return out
    ps = np.sort(p[ok])
    m = len(ps)
    thresh = alpha_fdr * np.arange(1, m + 1) / m
    passing = np.where(ps <= thresh)[0]
    if len(passing) == 0:
        return out
    p_crit = ps[passing.max()]
    out[ok] = p[ok] <= p_crit
    return out


def station_permutation_p(std: pd.DataFrame, enso: pd.DataFrame, col: str,
                          index_col: str = "oni_djf", n_perm: int = 2000, min_years: int = 15,
                          seed: int = 0) -> pd.DataFrame:
    """Per-station permutation p-values for r, plus BH-FDR field significance."""
    rng = np.random.default_rng(seed)
    d = std.merge(enso[["water_year", index_col]], on="water_year").dropna(subset=[col, index_col])
    rows = []
    for trip, g in d.groupby("stationTriplet"):
        if len(g) < min_years:
            continue
        r, p = permutation_p(g[index_col].to_numpy(float), g[col].to_numpy(float), _stat_r, n_perm, rng)
        rows.append({"stationTriplet": trip, "n": len(g), "r": r, "p_perm": p})
    out = pd.DataFrame(rows, columns=["stationTriplet", "n", "r", "p_perm"])
    if not out.empty:
        out["fdr_significant"] = fdr_field_significance(out["p_perm"].to_numpy())
    return out
