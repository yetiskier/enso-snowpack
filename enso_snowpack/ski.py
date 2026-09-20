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

from collections import namedtuple

import numpy as np
import pandas as pd

from .bootstrap import fdr_field_significance, permutation_p, _stat_r
from .daily import bh_vectorized, day_of_water_year, delete_d_calibration

# Ski regions, defined by MOUNTAIN RANGE AND SNOW CLIMATE — not by state.
#
# A state boundary is not a snow boundary. The Tahoe basin spans California and
# Nevada; the Bitterroots and the Selkirks span Montana and Idaho; the Tetons
# span Wyoming and Idaho; the Sangre de Cristo spans Colorado and New Mexico.
# Splitting those would average across the divide that matters and merge across
# one that does not. Each region is therefore anchored on a range around real
# ski terrain, carries the states it actually touches, and is labelled with its
# snow climate, because a maritime Cascade snowpack and a continental Colorado
# one respond to the same ocean differently:
#
#   maritime      deep, warm, dense; storm-track position is everything
#   transitional  maritime air that has crossed one barrier
#   intermountain the classic Wasatch/Teton middle ground
#   continental   cold, dry, shallow; interior, far from the source
#
# Fields: name, states, snow climate, lat, lon, radius_km, base_ft, summit_ft.
SkiRegion = namedtuple("SkiRegion", "name states climate lat lon radius_km base_ft summit_ft")

SKI_REGIONS_FULL = [
    # --- coastal Alaska
    SkiRegion("Chugach (Alyeska)", ("AK",), "maritime", 60.97, -149.10, 70, 250, 2750),
    # --- Cascades, west slope
    SkiRegion("North Cascades (Mt Baker)", ("WA",), "maritime", 48.78, -121.60, 70, 3500, 5089),
    SkiRegion("Central Cascades (Stevens/Snoqualmie)", ("WA",), "maritime", 47.55, -121.30, 70, 3000, 5845),
    SkiRegion("S Washington Cascades (Crystal/White Pass)", ("WA",), "maritime", 46.80, -121.45, 70, 4400, 7012),
    SkiRegion("Mt Hood", ("OR",), "maritime", 45.33, -121.71, 60, 4500, 8540),
    # --- Cascades, east slope / rain shadow
    SkiRegion("E Cascades rain shadow (Mission Ridge)", ("WA",), "transitional", 47.29, -120.40, 65, 4570, 6820),
    SkiRegion("Central Oregon (Mt Bachelor)", ("OR",), "transitional", 43.98, -121.69, 70, 6300, 9065),
    SkiRegion("Blues & Wallowas (Anthony Lakes)", ("OR",), "transitional", 45.00, -118.20, 95, 7100, 8000),
    # --- Sierra Nevada (Tahoe deliberately spans CA and NV)
    SkiRegion("N Sierra / Tahoe (Palisades/Heavenly/Rose)", ("CA", "NV"), "maritime", 39.05, -120.10, 80, 6200, 10067),
    SkiRegion("S Sierra (Mammoth/June)", ("CA",), "maritime", 37.70, -119.05, 75, 7953, 11053),
    SkiRegion("Shasta & Trinity", ("CA",), "maritime", 41.35, -122.20, 85, 5500, 7800),
    # --- Great Basin
    SkiRegion("Ruby Mtns & NE Nevada", ("NV",), "continental", 40.60, -115.40, 95, 6500, 10000),
    SkiRegion("Spring Mtns (Lee Canyon)", ("NV",), "continental", 36.32, -115.68, 70, 8510, 11290),
    # --- northern Rockies, maritime-influenced interior (cross MT/ID)
    SkiRegion("Selkirk & Cabinet (Schweitzer/Silver)", ("ID", "MT"), "transitional", 47.90, -116.20, 95, 4000, 6400),
    SkiRegion("Bitterroot & Lolo (Montana Snowbowl)", ("MT", "ID"), "transitional", 47.02, -113.98, 85, 5000, 7600),
    SkiRegion("Whitefish & Flathead Range", ("MT",), "transitional", 48.45, -114.35, 85, 4464, 6817),
    # --- northern Rockies, continental
    SkiRegion("Bridger/Gallatin/Madison (Big Sky)", ("MT",), "continental", 45.40, -111.30, 95, 6400, 10000),
    SkiRegion("Beartooth & Absaroka (Red Lodge)", ("MT", "WY"), "continental", 45.15, -109.50, 90, 7016, 9416),
    # --- central Idaho
    SkiRegion("Sawtooth & Smoky (Sun Valley)", ("ID",), "intermountain", 43.75, -114.40, 90, 5750, 9150),
    SkiRegion("W Central Idaho (Brundage/Tamarack)", ("ID",), "intermountain", 44.95, -116.05, 85, 5840, 7640),
    # --- greater Yellowstone (Tetons span WY and ID)
    SkiRegion("Tetons (Jackson/Targhee)", ("WY", "ID"), "intermountain", 43.60, -110.85, 80, 6300, 10450),
    SkiRegion("Yellowstone & Wind River", ("WY",), "continental", 43.60, -109.80, 110, 7000, 10000),
    SkiRegion("Bighorn Mtns", ("WY",), "continental", 44.35, -107.20, 85, 7500, 9500),
    SkiRegion("Snowy Range & Sierra Madre", ("WY",), "continental", 41.35, -106.60, 90, 8798, 9663),
    # --- Wasatch / Uinta
    SkiRegion("Wasatch (Alta/Snowbird/Park City)", ("UT",), "intermountain", 40.60, -111.60, 70, 6800, 11000),
    SkiRegion("Uinta Mtns", ("UT",), "continental", 40.70, -110.50, 80, 8000, 11000),
    SkiRegion("S Utah (Brian Head/Eagle Point)", ("UT",), "continental", 38.20, -112.40, 110, 9600, 10970),
    # --- Colorado, by range, straddling the ENSO node
    SkiRegion("Park Range (Steamboat)", ("CO",), "continental", 40.45, -106.80, 70, 6900, 10568),
    SkiRegion("Front Range (Winter Park/Loveland/A-Basin)", ("CO",), "continental", 39.80, -105.80, 65, 9000, 13050),
    SkiRegion("Gore & Tenmile (Vail/Summit/Copper)", ("CO",), "continental", 39.55, -106.15, 60, 8120, 12998),
    SkiRegion("Elk Mtns (Aspen/Crested Butte)", ("CO",), "continental", 39.00, -106.90, 60, 7945, 12162),
    SkiRegion("San Juans (Telluride/Purgatory/Wolf Ck)", ("CO",), "continental", 37.80, -107.60, 90, 8725, 13150),
    # --- Sangre de Cristo deliberately spans CO and NM
    SkiRegion("Sangre de Cristo (Taos/Santa Fe)", ("NM", "CO"), "continental", 36.60, -105.45, 100, 9200, 12481),
    # --- southern ranges, monsoon-influenced
    SkiRegion("Jemez & N New Mexico", ("NM",), "continental", 35.90, -106.50, 85, 8500, 11500),
    SkiRegion("Sacramento Mtns (Ski Apache)", ("NM",), "continental", 33.45, -105.75, 90, 9600, 11500),
    SkiRegion("San Francisco Peaks (AZ Snowbowl)", ("AZ",), "continental", 35.33, -111.68, 80, 9200, 11500),
    SkiRegion("White Mtns AZ (Sunrise Park)", ("AZ",), "continental", 33.98, -109.55, 85, 9200, 11100),
    # --- Black Hills
    SkiRegion("Black Hills (Terry Peak)", ("SD",), "continental", 44.33, -103.83, 85, 5800, 7064),
]

# Back-compatible tuple view used by the weighting and plotting code.
SKI_REGIONS = [(r.name, r.states[0], r.lat, r.lon, r.radius_km, r.base_ft, r.summit_ft)
               for r in SKI_REGIONS_FULL]
REGION_META = {r.name: r for r in SKI_REGIONS_FULL}
CLIMATES = ["maritime", "transitional", "intermountain", "continental"]

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
# Powder is measured in INCHES OF SNOW, not water, because a fixed water
# threshold is a maritime yardstick: Cascade new snow runs ~12 % density and
# Colorado's ~7 %, so 25 mm of water is about 8 inches of Cascade snow but 14
# inches of Colorado snow. Counting water alone makes the driest, lightest
# snowpacks look storm-free. Each station's own new-snow ratio is measured
# from the overlap of its depth and water records (see new_snow_ratios) and
# applied to its full, much longer water record.
POWDER_IN = 6.0        # inches of new snow in one day: a powder day
STORM_IN = 2.0         # inches: a refresh
MIN_SWE_GAIN_MM = 2.0  # below this a water gain is gauge noise, not a storm
DEFAULT_RATIO = 10.0   # inches of snow per inch of water where depth is missing

BASE_MM = 250.0        # settled SWE standing in for a comfortably skiable base
STORM_MM = 10.0        # legacy water thresholds, kept for the water-supply tables
BIG_STORM_MM = 25.0
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
            meta = REGION_META[name]
            rows.append({"stationTriplet": trip, "region": name, "state": state,
                         "states": "+".join(meta.states), "climate": meta.climate,
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
        meta = REGION_META[name]
        rows.append({"region": name, "states": "+".join(meta.states), "climate": meta.climate,
                     "base_ft": base, "summit_ft": summit,
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


def new_snow_ratios(daily: pd.DataFrame, min_events: int = 30) -> pd.Series:
    """Inches of new snow per inch of new water, per station.

    Measured from days where the depth and water records overlap and both
    rose: ratio = depth gain / SWE gain. The median over such days is the
    station's typical new-snow density expressed as a depth multiplier, and it
    is what converts the long water record into the snow depth a skier sees.
    Stations without enough paired events fall back to the regional median,
    then to ``DEFAULT_RATIO``.
    """
    d = daily[daily["element"].isin(["WTEQ", "SNWD"])][
        ["stationTriplet", "element", "date", "value"]].dropna()
    wide = d.pivot_table(index=["stationTriplet", "date"], columns="element",
                         values="value", aggfunc="first").reset_index()
    if "SNWD" not in wide or "WTEQ" not in wide:
        return pd.Series(dtype=float)
    wide = wide.sort_values(["stationTriplet", "date"])
    g = wide.groupby("stationTriplet")
    wide["d_swe"] = g["WTEQ"].diff()
    wide["d_dep"] = g["SNWD"].diff()
    ev = wide[(wide["d_swe"] >= 5.0) & (wide["d_dep"] > 0)]
    ratio = ev.assign(ratio=ev["d_dep"] / ev["d_swe"]).groupby("stationTriplet")["ratio"]
    out = ratio.median()
    counts = ratio.size()
    out = out[(counts >= min_events) & out.between(3.0, 30.0)]
    return out


def station_window_metrics(daily: pd.DataFrame, ratios: pd.Series | None = None) -> pd.DataFrame:
    """Per station, water year and ski window: base and storm metrics.

    Storms are counted in INCHES OF NEW SNOW, estimated from the daily water
    gain times the station's own new-snow ratio, so a continental powder day
    and a maritime one mean the same thing to a skier. ``mean_swe`` and
    ``mean_depth`` are the average standing snowpack in the window. Counts are
    given both for the whole window and per 30 days.
    """
    d = daily[daily["element"].isin(["WTEQ", "SNWD"])][
        ["stationTriplet", "element", "date", "value"]].dropna().reset_index(drop=True)
    doy, wy = day_of_water_year(d["date"])
    d = d.assign(doy=doy, wy=wy)
    d = d[(d["doy"] >= 0) & (d["doy"] < 366)]

    swe = d[d["element"] == "WTEQ"].sort_values(["stationTriplet", "date"]).copy()
    gain = swe.groupby(["stationTriplet", "wy"])["value"].diff()
    swe["gain"] = gain.fillna(0.0).clip(lower=0.0)
    if ratios is None:
        ratios = new_snow_ratios(daily)
    med = float(ratios.median()) if len(ratios) else DEFAULT_RATIO
    rmap = swe["stationTriplet"].map(ratios).fillna(med if np.isfinite(med) else DEFAULT_RATIO)
    # inches of new snow = (mm water / 25.4) * (inches of snow per inch of water)
    swe["new_snow_in"] = np.where(swe["gain"] >= MIN_SWE_GAIN_MM,
                                  swe["gain"] / 25.4 * rmap.to_numpy(), 0.0)
    depth = d[d["element"] == "SNWD"]

    out = []
    for wname, *win in SKI_WINDOWS:
        w = (win[0], win[1])
        s = swe[window_mask(swe["doy"].to_numpy(), w)]
        if s.empty:
            continue
        g = s.groupby(["stationTriplet", "wy"])
        m = g.agg(n_days=("value", "size"), mean_swe=("value", "mean"),
                  storm_days=("new_snow_in", lambda v: float((v >= STORM_IN).sum())),
                  big_storm_days=("new_snow_in", lambda v: float((v >= POWDER_IN).sum())),
                  new_snow_in_total=("new_snow_in", "sum"),
                  days_with_base=("value", lambda v: float((v >= BASE_MM).sum()))).reset_index()
        length = int(window_mask(np.arange(366), w).sum())
        m = m[m["n_days"] >= 0.8 * length]
        for c in ("storm_days", "big_storm_days", "days_with_base"):
            m[c + "_per_window"] = length * m[c] / m["n_days"]
            m[c] = 30.0 * m[c] / m["n_days"]
        m["new_snow_in_total"] = length * m["new_snow_in_total"] / m["n_days"]
        m["window_days"] = length
        m["mean_swe_in"] = m["mean_swe"] / 25.4
        dp = depth[window_mask(depth["doy"].to_numpy(), w)]
        if not dp.empty:
            dm = dp.groupby(["stationTriplet", "wy"]).agg(
                mean_depth=("value", "mean"), n_depth=("value", "size")).reset_index()
            dm = dm[dm["n_depth"] >= 0.8 * length].drop(columns="n_depth")
            m = m.merge(dm, on=["stationTriplet", "wy"], how="left")
        else:
            m["mean_depth"] = np.nan
        m["mean_depth_in"] = m["mean_depth"] / 25.4
        m["window"] = wname
        out.append(m)
    res = pd.concat(out, ignore_index=True).rename(columns={"wy": "water_year"})
    return res


METRICS = [
    ("mean_swe", "Base (mean SWE in window)"),
    ("mean_depth", "Base (mean snow depth)"),
    ("storm_days", "Storm days per 30 (>= 2 in of new snow)"),
    ("big_storm_days", "Powder days per 30 (>= 6 in of new snow)"),
    ("days_with_base", "Days per 30 with a skiable base"),
    ("new_snow_in_total", "Total new snow in the window (in)"),
]

# The same quantities in the units a skier actually uses, for the composites.
PHYSICAL_METRICS = [
    ("big_storm_days_per_window", "Powder days (>= 6 in of new snow)", "days"),
    ("storm_days_per_window", "Storm days (>= 2 in of new snow)", "days"),
    ("new_snow_in_total", "Total new snow in the window", "in"),
    ("days_with_base_per_window", "Days with a skiable base", "days"),
    ("mean_swe_in", "Average base (snow water equivalent)", "in"),
    ("mean_depth_in", "Average base (snow depth)", "in"),
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
                "states": "+".join(REGION_META[region].states) if region in REGION_META else None,
                "climate": REGION_META[region].climate if region in REGION_META else None,
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


def physical_composites(metrics: pd.DataFrame, region_map: pd.DataFrame, enso: pd.DataFrame,
                        index_col: str = "oni_djf", min_eff_stations: float = 2.0,
                        min_per_phase: int = 5) -> pd.DataFrame:
    """Actual numbers, in days and inches, by ENSO phase.

    For every region, window and physical measure: the all-winter average, the
    El Nino average, the La Nina average, and the differences — so the answer
    reads "9.7 powder days normally, 7.4 in an El Nino, a loss of 2.3 days
    (-24 %)" instead of a correlation.

    Regional values are elevation-weighted means of the stations (see
    :func:`station_weights`), averaged over the winters of each phase. These
    are raw means, not detrended: they are what actually happened.
    """
    phase = enso.set_index("water_year")["phase"]
    rows = []
    cols = [c for c, _, _ in PHYSICAL_METRICS]
    d = metrics.merge(region_map[["stationTriplet", "region", "weight"]], on="stationTriplet")
    d["phase"] = d["water_year"].map(phase)
    for (region, window), g in d.groupby(["region", "window"]):
        for metric, label, unit in PHYSICAL_METRICS:
            gg = g.dropna(subset=[metric])
            if gg.empty:
                continue
            # weighted regional value per winter
            per_year = (gg.groupby("water_year")
                          .apply(lambda t: pd.Series({
                              "value": float(np.average(t[metric], weights=t["weight"])),
                              "eff": float(t["weight"].sum() ** 2 / (t["weight"] ** 2).sum())}),
                          include_groups=False)
                          .reset_index())
            per_year = per_year[per_year["eff"] >= min_eff_stations]
            if per_year.empty:
                continue
            per_year["phase"] = per_year["water_year"].map(phase)
            allm = float(per_year["value"].mean())
            out = {"region": region, "window": window, "metric": metric,
                   "metric_label": label, "unit": unit,
                   "n_winters": int(len(per_year)), "all_winters": allm}
            for tag, ph in (("nino", "El Nino"), ("nina", "La Nina"), ("neutral", "Neutral")):
                v = per_year[per_year["phase"] == ph]["value"]
                out[f"n_{tag}"] = int(len(v))
                out[f"mean_{tag}"] = float(v.mean()) if len(v) >= min_per_phase else np.nan
            out["nino_minus_all"] = out["mean_nino"] - allm
            out["nina_minus_all"] = out["mean_nina"] - allm
            out["nino_minus_nina"] = out["mean_nino"] - out["mean_nina"]
            out["pct_change_nino"] = 100.0 * out["nino_minus_all"] / allm if allm else np.nan
            out["pct_change_nina"] = 100.0 * out["nina_minus_all"] / allm if allm else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def strength_composites_physical(metrics: pd.DataFrame, region_map: pd.DataFrame,
                                 enso: pd.DataFrame, metric: str = "big_storm_days_per_window",
                                 min_eff_stations: float = 2.0) -> pd.DataFrame:
    """The same physical numbers, split by ENSO STRENGTH bin rather than phase,
    so "does a stronger event cost more powder days" can be read directly."""
    meta = enso.set_index("water_year")[["phase", "strength"]]
    d = metrics.dropna(subset=[metric]).merge(
        region_map[["stationTriplet", "region", "weight"]], on="stationTriplet")
    rows = []
    for (region, window), g in d.groupby(["region", "window"]):
        per_year = (g.groupby("water_year")
                      .apply(lambda t: pd.Series({
                          "value": float(np.average(t[metric], weights=t["weight"])),
                          "eff": float(t["weight"].sum() ** 2 / (t["weight"] ** 2).sum())}),
                      include_groups=False)
                      .reset_index())
        per_year = per_year[per_year["eff"] >= min_eff_stations]
        if per_year.empty:
            continue
        per_year = per_year.join(meta, on="water_year")
        allm = float(per_year["value"].mean())
        for (ph, strength), sub in per_year.groupby(["phase", "strength"]):
            if len(sub) < 2:
                continue
            rows.append({"region": region, "window": window, "metric": metric,
                         "phase": ph, "strength": strength, "n_winters": int(len(sub)),
                         "mean": float(sub["value"].mean()), "all_winters": allm,
                         "diff": float(sub["value"].mean()) - allm,
                         "pct_change": 100.0 * (float(sub["value"].mean()) - allm) / allm if allm else np.nan})
    return pd.DataFrame(rows)


def season_shapes(daily: pd.DataFrame, region_map: pd.DataFrame, enso: pd.DataFrame,
                  regions: list | None = None, element: str = "WTEQ",
                  n_regions: int = 4) -> dict:
    """Mean base through the water year, per region and ENSO phase, in INCHES.

    Returns ``{region: {phase: array over day-of-water-year}}`` for plotting.
    This is the composite a skier can read directly: how deep the base is on
    any given day of an El Nino winter against a La Nina one.
    """
    from .daily import build_cube, DAYS_IN_WY
    phase = enso.set_index("water_year")["phase"]
    if regions is None:
        counts = region_map.groupby("region")["weight"].sum().sort_values(ascending=False)
        regions = list(counts.index[:n_regions])
    cube, stations, years = build_cube(daily, element=element, min_years=10)
    if not stations:
        return {}
    s_idx = {t: i for i, t in enumerate(stations)}
    ph = np.array([phase.get(y, None) for y in years], dtype=object)
    out = {}
    for region in regions:
        rm = region_map[(region_map["region"] == region)
                        & region_map["stationTriplet"].isin(s_idx)]
        if rm.empty:
            continue
        idx = [s_idx[t] for t in rm["stationTriplet"]]
        w = rm["weight"].to_numpy(float)[:, None, None]
        sub = cube[idx].astype(float)
        num = np.nansum(np.where(np.isfinite(sub), sub * w, 0.0), axis=0)
        den = np.nansum(np.where(np.isfinite(sub), w, 0.0), axis=0)
        with np.errstate(invalid="ignore"):
            wmean = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
        per_phase = {}
        for name in ("El Nino", "Neutral", "La Nina"):
            m = ph == name
            if m.sum() < 5:
                continue
            block = wmean[m]
            cnt = np.isfinite(block).sum(axis=0)
            tot = np.nansum(np.where(np.isfinite(block), block, 0.0), axis=0)
            v = np.divide(tot, cnt, out=np.full(DAYS_IN_WY, np.nan), where=cnt >= 3)
            per_phase[name] = v / 25.4          # mm -> inches
        if per_phase:
            out[region] = per_phase
    return out


def significant_composites(comp: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Physical composites joined to the test that backs them, keeping only
    the region-window-measures that survive false-discovery control.

    The statistics and the physical numbers are computed on different views of
    the same data — the tests on detrended z-scores, the composites on raw
    averages — so they are matched on (region, window, measure) and the test's
    r2 and p-value are carried across. Anything unmatched or not significant
    is dropped: this is the "only what survives" view.
    """
    if comp.empty or grid.empty or "fdr_significant" not in grid:
        return comp.iloc[0:0]
    g = grid[grid["fdr_significant"]].copy()
    # composites use the per-window column names; tests use the per-30 names
    g["metric_key"] = g["metric"].where(g["metric"].str.endswith("_in_total"),
                                        g["metric"] + "_per_window")
    swap = {"mean_swe": "mean_swe_in", "mean_depth": "mean_depth_in"}
    g["metric_key"] = g["metric"].map(swap).fillna(g["metric_key"])
    keep = g[["region", "window", "metric_key", "r2", "p_perm", "n_winters"]].rename(
        columns={"metric_key": "metric", "n_winters": "n_test"})
    out = comp.merge(keep, on=["region", "window", "metric"], how="inner",
                     suffixes=("", "_test"))
    return out.sort_values("r2", ascending=False).reset_index(drop=True)


def minimum_detectable_r(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest correlation a sample of ``n`` could reliably detect.

    Needed to tell "there is no effect" apart from "this sample could not have
    seen one". Uses the Fisher z approximation: with ~20 El Nino winters only
    a correlation above ~0.55 would be found 80 % of the time.
    """
    from scipy import stats as _st
    if n < 5:
        return np.nan
    za, zb = _st.norm.ppf(1 - alpha / 2), _st.norm.ppf(power)
    z = (za + zb) / np.sqrt(n - 3)
    return float(np.tanh(z))


def strength_significance(metrics: pd.DataFrame, region_map: pd.DataFrame, enso: pd.DataFrame,
                          index_col: str = "oni_djf", n_perm: int = 5000,
                          alpha_fdr: float = 0.10, min_phase_winters: int = 12,
                          seed: int = 0) -> pd.DataFrame:
    """Does the STRENGTH of an event matter, anywhere, in any window?

    Phase says which way; strength says how far. Three tests per region,
    window and measure:

    * ``within_nino`` — among El Nino winters ONLY, does a larger ONI mean
      less snow? This is the direct question, and the strictest test.
    * ``within_nina`` — the same among La Nina winters.
    * ``nested`` — an F-test of whether adding the continuous index to a
      three-level phase model explains significantly more variance.

    Each gets a permutation p-value, and Benjamini-Hochberg control is applied
    across the whole strength grid. ``min_detectable_r`` records what the
    sample could have found, so a null result can be read honestly.
    """
    from scipy import stats as _st
    rng = np.random.default_rng(seed)
    e = enso.set_index("water_year")
    rows = []
    for metric, label in METRICS:
        reg = standardize_region_metric(metrics, region_map, metric)
        if reg.empty:
            continue
        for (region, window), g in reg.groupby(["region", "window"]):
            g = g.dropna(subset=["value"])
            x = e[index_col].reindex(g["water_year"]).to_numpy(float)
            y = g["value"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < MIN_WINTERS:
                continue
            base = {"region": region, "window": window, "metric": metric,
                    "metric_label": label, "n_total": int(len(x))}
            for tag, mask in (("within_nino", x >= 0.5), ("within_nina", x <= -0.5)):
                xs, ys = x[mask], y[mask]
                if len(xs) < min_phase_winters or np.std(xs) == 0 or np.std(ys) == 0:
                    continue
                r, p = permutation_p(xs, ys, _stat_r, n_perm, rng)
                rows.append({**base, "test": tag, "n": int(len(xs)), "r": r, "r2": r ** 2,
                             "p": p, "min_detectable_r": minimum_detectable_r(len(xs))})
            # nested F-test: phase alone vs phase + the continuous index
            phase = np.where(x >= 0.5, 1, np.where(x <= -0.5, -1, 0))
            fit = np.full(len(y), y.mean())
            for ph in (-1, 0, 1):
                m = phase == ph
                if m.sum() >= 2:
                    fit[m] = y[m].mean()
            rss1 = float(((y - fit) ** 2).sum())
            X = np.column_stack([(phase == -1).astype(float), (phase == 0).astype(float),
                                 (phase == 1).astype(float), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss2 = float(((y - X @ beta) ** 2).sum())
            df2 = len(y) - X.shape[1]
            if df2 > 0 and rss2 > 0 and rss1 >= rss2:
                f = ((rss1 - rss2) / 1.0) / (rss2 / df2)
                pf = float(1 - _st.f.cdf(f, 1, df2))
                rows.append({**base, "test": "nested", "n": int(len(y)),
                             "r": np.nan, "r2": (rss1 - rss2) / max(rss1, 1e-12),
                             "p": pf, "min_detectable_r": minimum_detectable_r(len(y))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_significant"] = fdr_field_significance(out["p"].to_numpy(), alpha_fdr)
        out["n_tests"] = len(out)
    return out.sort_values("p").reset_index(drop=True)
