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

# Destination-anchored ski regions.
#
# Each carries the SERVED TERRAIN'S elevation band (base and summit, feet), not
# just a point, because a SNOTEL gauge only represents a ski area to the extent
# it sits in the same snow climate — and elevation is most of that. A valley
# gauge 900 m below the lifts measures a different snowpack, so stations are
# weighted by how well their elevation matches the band (see station_weights).
#
# (name, state, lat, lon, radius_km, base_ft, summit_ft)
SKI_REGIONS = [
    ("Montana Snowbowl (Missoula)",   "MT", 47.023, -113.983, 85, 5000, 7600),
    ("NW Montana (Whitefish)",        "MT", 48.450, -114.350, 90, 4464, 6817),
    ("SW Montana (Big Sky/Bridger)",  "MT", 45.400, -111.300, 95, 6400, 10000),
    ("N Idaho (Schweitzer/Silver)",   "ID", 47.900, -116.200, 95, 4000, 6400),
    ("C Idaho (Sun Valley)",          "ID", 43.750, -114.400, 90, 5750, 9150),
    ("Tetons (Jackson/Targhee)",      "WY", 43.600, -110.850, 80, 6300, 10450),
    ("S Wyoming (Snowy/Sierra Madre)","WY", 41.350, -106.600, 90, 8798, 9663),
    ("Wasatch (Alta/Park City)",      "UT", 40.600, -111.600, 70, 6800, 11000),
    ("S Utah (Brian Head)",           "UT", 37.750, -112.850, 90, 9600, 10970),
    ("N Colorado (Steamboat)",        "CO", 40.450, -106.800, 70, 6900, 10568),
    ("I-70 (Summit/Vail)",            "CO", 39.600, -106.200, 65, 8120, 12998),
    ("Elk Mtns (Aspen/Crested Butte)","CO", 39.000, -106.900, 60, 7945, 12162),
    ("San Juans (Telluride/Wolf Ck)", "CO", 37.800, -107.600, 90, 8725, 13150),
]

# Weighting scales. A station inside the served elevation band is fully
# weighted; outside it the weight falls off over ELEV_SCALE_FT of vertical
# separation, and over DIST_SCALE_FRAC of the region radius horizontally.
ELEV_SCALE_FT = 1200.0
DIST_SCALE_FRAC = 0.6
MIN_WEIGHT = 0.05      # below this a station contributes nothing

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


def station_weights(km: np.ndarray, elevation: np.ndarray,
                    base_ft: float, summit_ft: float, radius_km: float) -> np.ndarray:
    """Weight of each station as a proxy for a ski area's snowpack.

    Two Gaussian kernels multiplied:

    * **elevation** — the vertical gap to the served band ``[base, summit]``
      (zero inside it), falling off over ``ELEV_SCALE_FT``. This is what stops
      a valley-bottom gauge from speaking for a mountain, and it is why the
      weights differ from a plain radius average.
    * **distance** — horizontal separation, falling off over
      ``DIST_SCALE_FRAC x radius``.

    Returns weights in [0, 1]; anything below ``MIN_WEIGHT`` is set to 0.
    """
    elevation = np.asarray(elevation, dtype=float)
    gap = np.where(elevation < base_ft, base_ft - elevation,
                   np.where(elevation > summit_ft, elevation - summit_ft, 0.0))
    w_elev = np.exp(-0.5 * (gap / ELEV_SCALE_FT) ** 2)
    w_dist = np.exp(-0.5 * (np.asarray(km, float) / (DIST_SCALE_FRAC * radius_km)) ** 2)
    w = w_elev * w_dist
    return np.where(w < MIN_WEIGHT, 0.0, w)


def assign_regions(stations: pd.DataFrame) -> pd.DataFrame:
    """Map stations to ski regions with an elevation-aware weight.

    Regions may overlap; a station can serve two, with a different weight in
    each. Returns ``stationTriplet, region, state, km, elevation, elev_gap_ft,
    weight``.
    """
    rows = []
    for name, state, lat, lon, radius, base, summit in SKI_REGIONS:
        km = _haversine_km(lat, lon, stations["latitude"].to_numpy(),
                           stations["longitude"].to_numpy())
        elev = stations["elevation"].to_numpy(dtype=float)
        w = station_weights(km, elev, base, summit, radius)
        hit = (km <= radius) & (w > 0)
        gap = np.where(elev < base, base - elev, np.where(elev > summit, elev - summit, 0.0))
        for trip, d, e, gp, wt in zip(stations.loc[hit, "stationTriplet"], km[hit],
                                      elev[hit], gap[hit], w[hit]):
            rows.append({"stationTriplet": trip, "region": name, "state": state,
                         "km": round(float(d), 1), "elevation": float(e),
                         "elev_gap_ft": round(float(gp), 0), "weight": round(float(wt), 4),
                         "base_ft": base, "summit_ft": summit})
    return pd.DataFrame(rows)


def region_elevation_table(region_map: pd.DataFrame) -> pd.DataFrame:
    """Per region: served band, how many stations, and the weighted elevation
    the region's snowpack estimate actually represents."""
    rows = []
    for name, state, lat, lon, radius, base, summit in SKI_REGIONS:
        g = region_map[region_map["region"] == name]
        if g.empty:
            continue
        w = g["weight"].to_numpy()
        rows.append({"region": name, "state": state, "base_ft": base, "summit_ft": summit,
                     "stations": len(g), "eff_stations": round(float(w.sum() ** 2 / (w ** 2).sum()), 1),
                     "wtd_elev_ft": round(float((g["elevation"] * w).sum() / w.sum()), 0),
                     "median_elev_ft": float(g["elevation"].median()),
                     "in_band": int((g["elev_gap_ft"] == 0).sum())})
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
                              min_years: int = 15, min_eff_stations: float = 2.0) -> pd.DataFrame:
    """Per-station detrended z of ``metric``, combined over each ski region as
    an ELEVATION-WEIGHTED mean (see :func:`station_weights`).

    Returns ``region, window, water_year, value, n_stations, eff_stations``,
    where ``eff_stations`` is Kish's effective sample size of the weights —
    a region carried by one well-placed gauge is reported as such.
    """
    d = m.dropna(subset=[metric])[["stationTriplet", "water_year", "window", metric]]
    if d.empty:
        return pd.DataFrame(columns=["region", "window", "water_year", "value",
                                     "n_stations", "eff_stations"])
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
        return pd.DataFrame(columns=["region", "window", "water_year", "value",
                                     "n_stations", "eff_stations"])
    z = pd.concat(parts, ignore_index=True).merge(
        region_map[["stationTriplet", "region", "weight"]], on="stationTriplet")

    def _agg(g):
        w = g["weight"].to_numpy(float)
        v = g["z"].to_numpy(float)
        return pd.Series({"value": float(np.average(v, weights=w)),
                          "n_stations": len(g),
                          "eff_stations": float(w.sum() ** 2 / (w ** 2).sum())})
    agg = z.groupby(["region", "window", "water_year"]).apply(
        _agg, include_groups=False).reset_index()
    return agg[agg["eff_stations"] >= min_eff_stations].reset_index(drop=True)


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
                "r": r_obs, "r2": r_obs ** 2, "signed_r2": np.sign(r_obs) * r_obs ** 2,
                "bh_r_mean": r_mu, "bh_r_sd": r_sd,
                "bh_r2_mean": r_mu ** 2 + r_sd ** 2, "bh_r2_sd": 2 * abs(r_mu) * r_sd,
                "sig_2sigma": bool(np.isfinite(r_sd) and (r_mu - 2 * r_sd > 0 or r_mu + 2 * r_sd < 0)),
                "sig_2sigma_cal": bool(np.isfinite(r_sd) and
                                       (r_mu - 2 * r_sd * c > 0 or r_mu + 2 * r_sd * c < 0)),
                "p_perm": p_perm,
                "mean_nino": float(nino.mean()) if len(nino) >= 3 else np.nan,
                "mean_nina": float(nina.mean()) if len(nina) >= 3 else np.nan,
                "n_nino": int(len(nino)), "n_nina": int(len(nina)),
                **strength_effect(xi, yi),
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


def strength_effect(x: np.ndarray, y: np.ndarray) -> dict:
    """Does the STRENGTH of an ENSO event matter, beyond its phase?

    Decomposes the explained variance three ways:

    * ``r2_phase`` — variance explained by phase alone (El Nino / Neutral /
      La Nina as three group means). This is what "El Nino years are drier"
      buys you.
    * ``r2_index`` — variance explained by the continuous ONI, which uses
      magnitude as well as sign.
    * ``r2_strength_gain`` — ``r2_index - r2_phase``. POSITIVE means knowing
      how strong the event was tells you something beyond knowing which phase
      it was; at or below zero means magnitude adds nothing.
    * ``r2_within_nino`` / ``r2_within_nina`` — variance explained by the
      index inside one phase only, the sharpest test of magnitude: if a
      stronger El Nino meant less snow, this would be large.
    """
    out = {"r2_phase": np.nan, "r2_index": np.nan, "r2_strength_gain": np.nan,
           "r2_within_nino": np.nan, "r2_within_nina": np.nan,
           "slope_within_nino": np.nan}
    n = len(x)
    if n < 10 or np.std(y) == 0:
        return out
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return out
    phase = np.where(x >= 0.5, 1, np.where(x <= -0.5, -1, 0))
    fit = np.full(n, y.mean())
    for p in (-1, 0, 1):
        m = phase == p
        if m.sum() >= 2:
            fit[m] = y[m].mean()
    out["r2_phase"] = 1.0 - float(((y - fit) ** 2).sum()) / sst
    if np.std(x) > 0:
        out["r2_index"] = float(np.corrcoef(x, y)[0, 1] ** 2)
        out["r2_strength_gain"] = out["r2_index"] - out["r2_phase"]
    for key, m in (("r2_within_nino", x >= 0.5), ("r2_within_nina", x <= -0.5)):
        if m.sum() >= 6 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            out[key] = float(np.corrcoef(x[m], y[m])[0, 1] ** 2)
            if key == "r2_within_nino":
                out["slope_within_nino"] = float(np.polyfit(x[m], y[m], 1)[0])
    return out


def strength_verdict(grid: pd.DataFrame) -> pd.DataFrame:
    """Summarise, across the grid, whether magnitude adds anything to phase."""
    d = grid.dropna(subset=["r2_phase", "r2_index"])
    if d.empty:
        return pd.DataFrame()
    rows = []
    for window, g in d.groupby("window"):
        rows.append({
            "window": window, "tests": len(g),
            "mean_r2_phase": float(g["r2_phase"].mean()),
            "mean_r2_index": float(g["r2_index"].mean()),
            "mean_r2_strength_gain": float(g["r2_strength_gain"].mean()),
            "tests_where_strength_helps": int((g["r2_strength_gain"] > 0.01).sum()),
            "mean_r2_within_nino": float(g["r2_within_nino"].mean()),
        })
    order = {w: i for i, (w, *_) in enumerate(SKI_WINDOWS)}
    return pd.DataFrame(rows).sort_values("window", key=lambda s: s.map(order)).reset_index(drop=True)


def distributions_for(metrics: pd.DataFrame, region_map: pd.DataFrame, enso: pd.DataFrame,
                      region: str, window: str, metric: str, index_col: str = "oni_djf",
                      n_iter: int = 10000, n_perm: int = 10000, omit_fraction: float = 0.20,
                      seed: int = 0) -> dict:
    """Raw distributions behind one test, for plotting.

    Returns the Brown & Harper subsample distribution of the correlation (the
    uncertainty in the estimate) and the permutation null distribution (what
    chance alone produces), both as signed r², plus the observed value.
    """
    rng = np.random.default_rng(seed)
    reg = standardize_region_metric(metrics, region_map, metric)
    g = reg[(reg["region"] == region) & (reg["window"] == window)].dropna(subset=["value"])
    e = enso.set_index("water_year")[index_col]
    x = e.reindex(g["water_year"]).to_numpy(float)
    y = g["value"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < MIN_WINTERS:
        return {}
    keep = max(3, int(round(n * (1.0 - omit_fraction))))
    idx = np.argsort(rng.random((n_iter, n)), axis=1)[:, :keep]
    xs, ys = x[idx], y[idx]
    dx = xs - xs.mean(axis=1, keepdims=True)
    dy = ys - ys.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_sub = (dx * dy).sum(axis=1) / np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
    r_sub = r_sub[np.isfinite(r_sub)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.corrcoef(rng.permutation(x), y)[0, 1]
    null = null[np.isfinite(null)]
    r_obs = float(np.corrcoef(x, y)[0, 1])
    return {"region": region, "window": window, "metric": metric, "n": n,
            "r_obs": r_obs, "signed_r2_obs": np.sign(r_obs) * r_obs ** 2,
            "sub_signed_r2": np.sign(r_sub) * r_sub ** 2,
            "null_signed_r2": np.sign(null) * null ** 2,
            "p_perm": float((np.abs(null) >= abs(r_obs)).sum() + 1) / (len(null) + 1)}
