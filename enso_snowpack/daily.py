"""Day-by-day ENSO signal through the water year.

Why this exists: April-1 SWE is a single snapshot that mixes accumulation
with melt already under way, and it cannot show *when* in the season the
ENSO signal acts. Brown & Harper (2026, HESS 30, 5735, Sect. 2.5) analyse
every day of the water year and report the evolution of the daily
distributions; their central result is that the change is NOT monotonic
across the season. This module applies that frame to the ENSO question:
for each day of the water year, correlate the regional standardised
snowpack anomaly with the DJF ONI, with their modified bootstrap.

The seasonal periods are theirs (their Table 1), defined by inflection
points in the SWE and buffering-capacity curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import STATES
from .bootstrap import brown_harper_2026, _stat_r, _stat_slope, delete_d_calibration

# Brown & Harper (2026) Table 1, as (name, start (month, day), end (month, day)).
PERIODS = [
    ("Early Accumulation", (11, 19), (1, 27)),
    ("Core Accumulation", (1, 28), (3, 8)),
    ("Late Accumulation", (3, 9), (4, 17)),
    ("Melt Onset", (4, 18), (6, 7)),
]

DAYS_IN_WY = 366


def day_of_water_year(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (day index with 0 = 1 October, water year).

    Index-safe: the October-1 anchor is built on ``dates``' own index, so a
    non-contiguous index cannot silently union and change length.
    """
    d = pd.Series(pd.to_datetime(dates).to_numpy(), index=range(len(dates)))
    wy = np.where(d.dt.month >= 10, d.dt.year + 1, d.dt.year)
    start = pd.to_datetime(dict(year=pd.Series(wy - 1, index=d.index),
                                month=1, day=1)) + pd.to_timedelta(273, unit="D")
    # 1 Oct is day 273 of a non-leap year and 274 of a leap year; build exactly
    start = pd.to_datetime(pd.DataFrame({"year": wy - 1, "month": 10, "day": 1}, index=d.index))
    return (d - start).dt.days.to_numpy(), wy


def period_of_day(doy: int) -> str | None:
    """Which Brown & Harper period a day-of-water-year index falls in."""
    d = pd.Timestamp(2001, 10, 1) + pd.Timedelta(days=int(doy))
    for name, (m0, d0), (m1, d1) in PERIODS:
        # periods are given as calendar dates inside one water year
        y0 = 2001 if m0 >= 10 else 2002
        y1 = 2001 if m1 >= 10 else 2002
        if pd.Timestamp(y0, m0, d0) <= d <= pd.Timestamp(y1, m1, d1):
            return name
    return None


def build_cube(daily: pd.DataFrame, element: str = "WTEQ",
               min_years: int = 15) -> tuple[np.ndarray, list, np.ndarray]:
    """Dense array ``[station, water_year, day_of_water_year]`` of the element.

    Returns (cube, station_triplets, water_years). Missing days are NaN.
    """
    d = (daily[daily["element"] == element][["stationTriplet", "date", "value"]]
         .dropna().reset_index(drop=True))
    doy, wy = day_of_water_year(d["date"])
    d = d.assign(doy=doy, wy=wy)
    d = d[(d["doy"] >= 0) & (d["doy"] < DAYS_IN_WY)]

    stations = sorted(d["stationTriplet"].unique())
    years = np.arange(int(d["wy"].min()), int(d["wy"].max()) + 1)
    s_idx = {s: i for i, s in enumerate(stations)}
    y_idx = {y: i for i, y in enumerate(years)}

    cube = np.full((len(stations), len(years), DAYS_IN_WY), np.nan, dtype=np.float32)
    cube[d["stationTriplet"].map(s_idx).to_numpy(),
         d["wy"].map(y_idx).to_numpy(),
         d["doy"].to_numpy()] = d["value"].to_numpy(dtype=np.float32)

    # keep stations with enough years of mid-winter data (1 Feb = day 123)
    ok = np.isfinite(cube[:, :, 123]).sum(axis=1) >= min_years
    return cube[ok], [s for s, k in zip(stations, ok) if k], years


def standardize_cube(cube: np.ndarray, years: np.ndarray, detrend: bool = True) -> np.ndarray:
    """Per station and per day, z-score across water years (optionally after
    removing a linear trend in year), so every day is comparable."""
    out = np.full(cube.shape, np.nan, dtype=np.float32)
    x = years.astype(np.float64)
    for s in range(cube.shape[0]):
        block = cube[s].astype(np.float64)            # [year, day]
        good = np.isfinite(block)
        n = good.sum(axis=0)
        use = n >= 10
        if not use.any():
            continue
        b = block[:, use]
        g = good[:, use]
        bf = np.where(g, b, 0.0)
        cnt = g.sum(axis=0)
        mean = bf.sum(axis=0) / cnt
        if detrend:
            xm = np.where(g, x[:, None], 0.0).sum(axis=0) / cnt
            dx = np.where(g, x[:, None] - xm, 0.0)
            dy = np.where(g, b - mean, 0.0)
            denom = (dx * dx).sum(axis=0)
            slope = np.divide((dx * dy).sum(axis=0), denom,
                              out=np.zeros_like(denom), where=denom > 0)
            resid = np.where(g, b - mean - slope * dx, np.nan)
        else:
            resid = np.where(g, b - mean, np.nan)
        sd = np.nanstd(resid, axis=0, ddof=1)
        z = np.divide(resid, sd, out=np.full_like(resid, np.nan), where=sd > 0)
        block_out = np.full(block.shape, np.nan)
        block_out[:, use] = z
        out[s] = block_out
    return out


def regional_daily(z: np.ndarray, stations: list, meta: pd.DataFrame,
                   min_stations: int = 5) -> dict[str, np.ndarray]:
    """Mean standardised anomaly per region: ``{region: [year, day]}``."""
    state = meta.set_index("stationTriplet")["stateCode"].reindex(stations).to_numpy()
    regions: dict[str, np.ndarray] = {}
    groups = {s: state == s for s in STATES}
    groups["North (MT+ID)"] = np.isin(state, ["MT", "ID"])
    groups["South (CO+UT)"] = np.isin(state, ["CO", "UT"])
    groups["ALL"] = np.ones(len(stations), bool)
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        sub = z[mask]
        cnt = np.isfinite(sub).sum(axis=0)
        # np.nanmean warns on all-NaN days (start/end of season); those days are
        # dropped by the min_stations test below anyway, so compute without it.
        total = np.nansum(sub, axis=0)
        mean = np.divide(total, cnt, out=np.full(total.shape, np.nan, dtype=float),
                         where=cnt > 0)
        mean[cnt < min_stations] = np.nan
        regions[name] = mean
    return regions


def bh_vectorized(x: np.ndarray, y: np.ndarray, n_iter: int, omit_fraction: float, rng):
    """Brown & Harper (2026) subsampling, all iterations at once.

    Identical statistics to :func:`bootstrap.brown_harper_2026` (random
    ``1-omit_fraction`` subsets without replacement, no resampling with
    replacement), computed with matrix algebra because the daily curve needs
    ~6000 of these. Returns (r_mean, r_sd, slope_mean, slope_sd).
    """
    n = len(x)
    keep = max(3, int(round(n * (1.0 - omit_fraction))))
    # one random permutation per iteration, take the first `keep` columns
    idx = np.argsort(rng.random((n_iter, n)), axis=1)[:, :keep]
    xs, ys = x[idx], y[idx]
    xm = xs.mean(axis=1, keepdims=True)
    ym = ys.mean(axis=1, keepdims=True)
    dx, dy = xs - xm, ys - ym
    sxy = (dx * dy).sum(axis=1)
    sxx = (dx * dx).sum(axis=1)
    syy = (dy * dy).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sxy / np.sqrt(sxx * syy)
        slope = sxy / sxx
    r = r[np.isfinite(r)]
    slope = slope[np.isfinite(slope)]
    if len(r) < 10 or len(slope) < 10:
        return (np.nan,) * 4
    return float(r.mean()), float(r.std(ddof=1)), float(slope.mean()), float(slope.std(ddof=1))


def daily_enso_curve(region_mean: np.ndarray, years: np.ndarray, enso: pd.DataFrame,
                     index_col: str = "oni_djf", n_iter: int = 2000,
                     omit_fraction: float = 0.20, min_years: int = 20,
                     seed: int = 0) -> pd.DataFrame:
    """For each day of the water year: r and slope against the ENSO index,
    with the Brown & Harper subsampling distribution and its 2σ rule.

    ``region_mean`` is ``[year, day]``.
    """
    e = enso.set_index("water_year")[index_col].reindex(years).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(region_mean.shape[1]):
        y = region_mean[:, day].astype(float)
        ok = np.isfinite(y) & np.isfinite(e)
        n = int(ok.sum())
        if n < min_years:
            continue
        xi, yi = e[ok], y[ok]
        r_mu, r_sd, s_mu, s_sd = bh_vectorized(xi, yi, n_iter, omit_fraction, rng)
        r_lo, r_hi = r_mu - 2 * r_sd, r_mu + 2 * r_sd
        r_sig = bool(np.isfinite(r_sd) and (r_lo > 0 or r_hi < 0))
        s_sig = bool(np.isfinite(s_sd) and (s_mu - 2 * s_sd > 0 or s_mu + 2 * s_sd < 0))
        c = delete_d_calibration(n, omit_fraction)
        rows.append({
            "doy": day, "date": (pd.Timestamp(2001, 10, 1) + pd.Timedelta(days=day)).strftime("%b %d"),
            "period": period_of_day(day), "n": n,
            "r": _stat_r(xi, yi), "r_mean": r_mu, "r_sd": r_sd,
            "r_lo": r_lo, "r_hi": r_hi, "r_sig": r_sig,
            "r_sig_cal": bool(np.isfinite(r_sd) and (r_mu - 2 * r_sd * c > 0 or r_mu + 2 * r_sd * c < 0)),
            "slope": _stat_slope(xi, yi), "slope_mean": s_mu, "slope_sd": s_sd,
            "slope_sig": s_sig,
            "mean_swe_z_nino": float(yi[xi >= 0.5].mean()) if (xi >= 0.5).sum() >= 3 else np.nan,
            "mean_swe_z_nina": float(yi[xi <= -0.5].mean()) if (xi <= -0.5).sum() >= 3 else np.nan,
        })
    return pd.DataFrame(rows)


def period_summary(curve: pd.DataFrame) -> pd.DataFrame:
    """Collapse the daily curve onto the Brown & Harper periods."""
    c = curve.dropna(subset=["period"])
    if c.empty:
        return pd.DataFrame()
    g = c.groupby("period").agg(
        days=("doy", "count"), r_mean=("r", "mean"), r_min=("r", "min"), r_max=("r", "max"),
        frac_sig=("r_sig_cal", "mean"),
        nino=("mean_swe_z_nino", "mean"), nina=("mean_swe_z_nina", "mean")).reset_index()
    order = {name: i for i, (name, _, _) in enumerate(PERIODS)}
    return g.sort_values("period", key=lambda s: s.map(order)).reset_index(drop=True)
