"""ENSO and the ski season: sub-seasonal, by ski region and window of winter.

This is a different question from water supply. A ski operation does not care
about April-1 SWE — that is a runoff metric. It cares whether there is a base
on the ground during the windows that carry the season, and how often it
storms during them. So this module:

* aggregates stations by **destination ski region**, not by state, because a
  state is not a snow climate (the Colorado San Juans and the northern
  Colorado Rockies sit on opposite sides of the ENSO node);
* scores each region-winter inside **ski windows** rather than the whole
  accumulation season;
* uses **base** and **storm frequency** as the metrics, not peak or April-1
  SWE;
* and controls the false-discovery rate across the whole region x window x
  metric grid, because asking ~100 questions at p<0.05 buys ~5 false
  positives for free.

Snow depth (SNWD) begins ~1993 at most SNOTEL sites, so depth-based metrics
have ~25 winters while SWE-based ones have ~45. Storm days are therefore
derived from daily SWE gain, which is both longer and density-independent: a
25 mm water gain is a big storm whether it falls at 6 % or 12 % density.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import fdr_field_significance, permutation_p, _stat_r
from .daily import bh_vectorized, day_of_water_year, delete_d_calibration

# Destination-anchored ski regions: (name, state, lat, lon, radius_km).
SKI_REGIONS = [
    ("NW Montana (Whitefish)",        "MT", 48.45, -114.35, 90),
    ("SW Montana (Big Sky/Bridger)",  "MT", 45.40, -111.30, 95),
    ("N Idaho (Schweitzer/Silver)",   "ID", 47.90, -116.20, 95),
    ("C Idaho (Sun Valley)",          "ID", 43.75, -114.40, 90),
    ("Tetons (Jackson/Targhee)",      "WY", 43.60, -110.85, 80),
    ("S Wyoming (Snowy/Sierra Madre)","WY", 41.35, -106.60, 90),
    ("Wasatch (Alta/Park City)",      "UT", 40.60, -111.60, 70),
    ("S Utah (Brian Head)",           "UT", 37.75, -112.85, 90),
    ("N Colorado (Steamboat)",        "CO", 40.45, -106.80, 70),
    ("I-70 (Summit/Vail)",            "CO", 39.60, -106.20, 65),
    ("Elk Mtns (Aspen/Crested Butte)","CO", 39.00, -106.90, 60),
    ("San Juans (Telluride/Wolf Ck)", "CO", 37.80, -107.60, 90),
]

# Windows that actually carry a ski season, not the hydrologic season.
SKI_WINDOWS = [
    ("Early season",  (11, 1), (12, 15)),   # can it open, and on what
    ("Holidays",      (12, 16), (1, 5)),    # the revenue peak
    ("Midwinter",     (1, 6), (2, 28)),     # powder season
    ("Spring",        (3, 1), (4, 15)),     # spring break to closing
]

# Daily SWE gain thresholds (mm water equivalent).
STORM_MM = 10.0        # ~10-15 cm of snow: a refresh
BIG_STORM_MM = 25.0    # ~25-40 cm: a powder day
BASE_MM = 250.0        # SWE standing in for a comfortably skiable base
MIN_WINTERS = 20       # winters a region-window needs before it is tested


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def assign_regions(stations: pd.DataFrame) -> pd.DataFrame:
    """Map each station to every ski region whose radius contains it.

    Regions may overlap; a station can serve two. Returns
    ``stationTriplet, region, state, km, elevation``.
    """
    rows = []
    for name, state, lat, lon, radius in SKI_REGIONS:
        km = _haversine_km(lat, lon, stations["latitude"].to_numpy(),
                           stations["longitude"].to_numpy())
        hit = km <= radius
        for trip, d, elev in zip(stations.loc[hit, "stationTriplet"], km[hit],
                                 stations.loc[hit, "elevation"]):
            rows.append({"stationTriplet": trip, "region": name, "state": state,
                         "km": round(float(d), 1), "elevation": elev})
    return pd.DataFrame(rows)


def window_mask(doy: np.ndarray, window) -> np.ndarray:
    """Boolean mask of days inside a ski window, given day-of-water-year."""
    (m0, d0), (m1, d1) = window
    def day(m, d):
        y = 2001 if m >= 10 else 2002
        return (pd.Timestamp(y, m, d) - pd.Timestamp(2001, 10, 1)).days
    return (doy >= day(m0, d0)) & (doy <= day(m1, d1))


def station_window_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Per station, water year and ski window: base and storm metrics.

    ``mean_swe`` / ``mean_depth`` are the average standing snowpack in the
    window (the base). ``storm_days`` and ``big_storm_days`` count days whose
    SWE gain clears the thresholds. ``days_with_base`` counts days at or above
    ``BASE_MM``. Storm counts are per 30 days so windows of different length
    are comparable.
    """
    d = daily[daily["element"].isin(["WTEQ", "SNWD"])][
        ["stationTriplet", "element", "date", "value"]].dropna().reset_index(drop=True)
    doy, wy = day_of_water_year(d["date"])
    d = d.assign(doy=doy, wy=wy)
    d = d[(d["doy"] >= 0) & (d["doy"] < 366)]

    swe = d[d["element"] == "WTEQ"].sort_values(["stationTriplet", "date"])
    # daily gain within a station-winter; a reset between water years is not a storm
    gain = swe.groupby(["stationTriplet", "wy"])["value"].diff()
    swe = swe.assign(gain=gain.fillna(0.0).clip(lower=0.0))
    depth = d[d["element"] == "SNWD"]

    out = []
    for wname, *win in SKI_WINDOWS:
        w = (win[0], win[1])
        s = swe[window_mask(swe["doy"].to_numpy(), w)]
        if s.empty:
            continue
        g = s.groupby(["stationTriplet", "wy"])
        m = g.agg(n_days=("value", "size"), mean_swe=("value", "mean"),
                  storm_days=("gain", lambda v: float((v >= STORM_MM).sum())),
                  big_storm_days=("gain", lambda v: float((v >= BIG_STORM_MM).sum())),
                  days_with_base=("value", lambda v: float((v >= BASE_MM).sum()))).reset_index()
        length = int(window_mask(np.arange(366), w).sum())
        m = m[m["n_days"] >= 0.8 * length]          # a window needs real coverage
        for c in ("storm_days", "big_storm_days", "days_with_base"):
            m[c] = 30.0 * m[c] / m["n_days"]        # per 30 days
        dp = depth[window_mask(depth["doy"].to_numpy(), w)]
        if not dp.empty:
            dm = dp.groupby(["stationTriplet", "wy"]).agg(
                mean_depth=("value", "mean"), n_depth=("value", "size")).reset_index()
            dm = dm[dm["n_depth"] >= 0.8 * length].drop(columns="n_depth")
            m = m.merge(dm, on=["stationTriplet", "wy"], how="left")
        else:
            m["mean_depth"] = np.nan
        m["window"] = wname
        out.append(m)
    res = pd.concat(out, ignore_index=True).rename(columns={"wy": "water_year"})
    return res


METRICS = [
    ("mean_swe", "Base (mean SWE in window)"),
    ("mean_depth", "Base (mean snow depth)"),
    ("storm_days", "Storm days per 30 (SWE gain >= 10 mm)"),
    ("big_storm_days", "Powder days per 30 (SWE gain >= 25 mm)"),
    ("days_with_base", "Days per 30 with a skiable base"),
]


def standardize_region_metric(m: pd.DataFrame, region_map: pd.DataFrame, metric: str,
                              min_years: int = 15, min_stations: int = 3) -> pd.DataFrame:
    """Per-station detrended z of ``metric``, averaged over each ski region.

    Returns ``region, window, water_year, value, n_stations``.
    """
    d = m.dropna(subset=[metric])[["stationTriplet", "water_year", "window", metric]]
    if d.empty:
        return pd.DataFrame(columns=["region", "window", "water_year", "value", "n_stations"])
    parts = []
    for (trip, win), g in d.groupby(["stationTriplet", "window"]):
        if len(g) < min_years:
            continue
        g = g.sort_values("water_year")
        v = g[metric].to_numpy(float)
        yr = g["water_year"].to_numpy(float)
        if np.std(v) == 0:
            continue
        slope, icpt = np.polyfit(yr, v, 1)
        resid = v - (slope * yr + icpt)
        sd = resid.std(ddof=1)
        if sd <= 0:
            continue
        parts.append(g.assign(z=resid / sd))
    if not parts:
        return pd.DataFrame(columns=["region", "window", "water_year", "value", "n_stations"])
    z = pd.concat(parts, ignore_index=True).merge(region_map[["stationTriplet", "region"]],
                                                  on="stationTriplet")
    agg = (z.groupby(["region", "window", "water_year"])
             .agg(value=("z", "mean"), n_stations=("z", "size")).reset_index())
    return agg[agg["n_stations"] >= min_stations].reset_index(drop=True)


def test_grid(metrics: pd.DataFrame, region_map: pd.DataFrame, enso: pd.DataFrame,
              index_col: str = "oni_djf", n_iter: int = 10000, n_perm: int = 5000,
              omit_fraction: float = 0.20, alpha_fdr: float = 0.10,
              seed: int = 0) -> pd.DataFrame:
    """Every ski region x window x metric, with FDR control over the whole grid.

    Reports the Brown & Harper (2026) subsampling mean and sd, its raw and
    calibrated 2 sigma verdicts, a permutation p, and whether the test
    survives Benjamini-Hochberg across all tests run here.
    """
    rng = np.random.default_rng(seed)
    e = enso.set_index("water_year")[index_col]
    rows = []
    for metric, label in METRICS:
        reg = standardize_region_metric(metrics, region_map, metric)
        if reg.empty:
            continue
        for (region, window), g in reg.groupby(["region", "window"]):
            g = g.dropna(subset=["value"])
            x = e.reindex(g["water_year"]).to_numpy(float)
            y = g["value"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            n = int(ok.sum())
            if n < MIN_WINTERS:
                continue
            xi, yi = x[ok], y[ok]
            r_mu, r_sd, s_mu, s_sd = bh_vectorized(xi, yi, n_iter, omit_fraction, rng)
            c = delete_d_calibration(n, omit_fraction)
            r_obs, p_perm = permutation_p(xi, yi, _stat_r, n_perm, rng)
            nino, nina = yi[xi >= 0.5], yi[xi <= -0.5]
            rows.append({
                "metric": metric, "metric_label": label, "region": region, "window": window,
                "state": dict((n_, s_) for n_, s_, *_ in SKI_REGIONS).get(region),
                "n_winters": n, "n_stations": int(g["n_stations"].median()),
                "r": r_obs, "bh_r_mean": r_mu, "bh_r_sd": r_sd,
                "sig_2sigma": bool(np.isfinite(r_sd) and (r_mu - 2 * r_sd > 0 or r_mu + 2 * r_sd < 0)),
                "sig_2sigma_cal": bool(np.isfinite(r_sd) and
                                       (r_mu - 2 * r_sd * c > 0 or r_mu + 2 * r_sd * c < 0)),
                "p_perm": p_perm,
                "mean_nino": float(nino.mean()) if len(nino) >= 3 else np.nan,
                "mean_nina": float(nina.mean()) if len(nina) >= 3 else np.nan,
                "n_nino": int(len(nino)), "n_nina": int(len(nina)),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_significant"] = fdr_field_significance(out["p_perm"].to_numpy(), alpha_fdr)
        out["n_tests"] = len(out)
    return out.sort_values("p_perm").reset_index(drop=True)


def window_sign_summary(grid: pd.DataFrame,
                        metrics=("mean_swe", "storm_days", "big_storm_days")) -> pd.DataFrame:
    """How consistently each window leans one way across all ski regions.

    Individually modest correlations mean much more when they share a sign
    across independent regions, so this reports an exact binomial sign test
    per window alongside the mean correlation.
    """
    from scipy import stats
    rows = []
    d = grid[grid["metric"].isin(metrics)]
    for wname, *_ in SKI_WINDOWS:
        w = d[d["window"] == wname]
        if w.empty:
            continue
        neg, tot = int((w["r"] < 0).sum()), int(len(w))
        rows.append({"window": wname, "tests": tot, "negative": neg,
                     "frac_negative": neg / tot, "mean_r": float(w["r"].mean()),
                     "sign_test_p": float(stats.binomtest(neg, tot, 0.5).pvalue)})
    return pd.DataFrame(rows)
