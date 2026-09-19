"""Derived tables and statistics.

Conventions
-----------
* Water year (WY) N runs 1 Oct N-1 .. 30 Sep N. April 1 SWE of WY N is the
  WTEQ value on N-04-01 (nearest value within ±3 days if that day is missing).
* The ENSO index for WY N is the DJF ONI labelled with year N (Dec N-1 .. Feb N).
  ``oni_winter`` is the mean of the NDJ, DJF and JFM seasons of WY N.
* CPC phase thresholds on DJF ONI: El Niño >= +0.5, La Niña <= -0.5.
  Strength bins (CPC / Null convention): weak 0.5-0.9, moderate 1.0-1.4,
  strong 1.5-1.9, very strong >= 2.0 (absolute value; applied to the
  winter peak of the ONI, i.e. max |ONI| over OND..FMA of the water year).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import STATES

MIN_YEARS_STATION = 15          # years of valid April-1 SWE to keep a station
MIN_MEDIAN_SWE_MM = 50.0        # drop stations with trivial April snowpack
MIN_STATIONS_REGION_YEAR = 5    # stations needed for a regional mean
STRENGTH_BINS = [(0.5, "weak"), (1.0, "moderate"), (1.5, "strong"), (2.0, "very strong")]


# ------------------------------------------------------------- ENSO table ---

def enso_by_water_year(oni: pd.DataFrame) -> pd.DataFrame:
    """One row per water year: DJF ONI, winter-mean ONI, peak |ONI|, phase,
    strength."""
    piv = oni.pivot_table(index="year", columns="season", values="oni")
    out = pd.DataFrame(index=piv.index.rename("water_year"))
    out["oni_djf"] = piv.get("DJF")
    out["oni_ond_prev"] = piv.get("OND").shift(1)      # Oct-Dec of the previous calendar year
    out["oni_ndj_prev"] = piv.get("NDJ").shift(1)
    out["oni_winter"] = pd.concat([out["oni_ndj_prev"], piv.get("DJF"), piv.get("JFM")],
                                  axis=1).mean(axis=1)
    peak_cols = pd.concat([out["oni_ond_prev"], out["oni_ndj_prev"], piv.get("DJF"),
                           piv.get("JFM"), piv.get("FMA")], axis=1)
    idx = peak_cols.abs().idxmax(axis=1)
    out["oni_peak"] = [peak_cols.loc[y, c] if pd.notna(c) else np.nan
                       for y, c in idx.items()]
    out["phase"] = np.select([out["oni_djf"] >= 0.5, out["oni_djf"] <= -0.5],
                             ["El Nino", "La Nina"], default="Neutral")
    out["strength"] = [strength_label(p, v) for p, v in zip(out["phase"], out["oni_peak"])]
    return out.dropna(subset=["oni_djf"]).reset_index()


def strength_label(phase: str, peak: float) -> str:
    if phase == "Neutral" or not np.isfinite(peak):
        return "neutral"
    a = abs(peak)
    label = "weak"
    for thr, name in STRENGTH_BINS:
        if a >= thr:
            label = name
    return label


# ---------------------------------------------------------- station metrics --

def _nearest_value(s: pd.Series, when: pd.Timestamp, tol_days: int) -> float:
    if s.empty:
        return np.nan
    if when in s.index:
        return float(s.loc[when])
    win = s.loc[when - pd.Timedelta(days=tol_days): when + pd.Timedelta(days=tol_days)]
    if win.empty:
        return np.nan
    pos = np.abs((win.index - when).days).argmin()
    return float(win.iloc[pos])


def station_water_year_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Per station × water year: April-1 SWE, peak SWE + date, Oct-Mar
    precipitation (from accumulated PREC) and DJF mean temperature.

    ``long`` is the metric-converted AWDB long form
    (stationTriplet, element, date, value, unit)."""
    rows = []
    for trip, g in long.groupby("stationTriplet"):
        el = {e: gg.set_index("date")["value"].sort_index()
              for e, gg in g.groupby("element")}
        wteq = el.get("WTEQ")
        if wteq is None or wteq.empty:
            continue
        wteq = wteq[~wteq.index.duplicated()]
        prec = el.get("PREC")
        tavg = el.get("TAVG")
        wy_all = np.where(wteq.index.month >= 10, wteq.index.year + 1, wteq.index.year)
        for wy in np.unique(wy_all):
            season = wteq[(wteq.index >= pd.Timestamp(wy - 1, 10, 1)) &
                          (wteq.index <= pd.Timestamp(wy, 7, 31))]
            apr1 = _nearest_value(season, pd.Timestamp(wy, 4, 1), 3)
            if season.empty:
                continue
            # coverage of the accumulation season (Nov-Apr) to trust the peak
            core = season[(season.index >= pd.Timestamp(wy - 1, 11, 1)) &
                          (season.index <= pd.Timestamp(wy, 4, 30))]
            n_core = len(core)
            peak = float(core.max()) if n_core else np.nan
            peak_date = core.idxmax() if n_core else pd.NaT
            row = {"stationTriplet": trip, "water_year": int(wy), "apr1_swe": apr1,
                   "peak_swe": peak, "peak_doy": (peak_date - pd.Timestamp(wy - 1, 10, 1)).days
                   if pd.notna(peak_date) else np.nan,
                   "n_core_days": n_core}
            if prec is not None and not prec.empty:
                prec = prec[~prec.index.duplicated()]
                p_apr = _nearest_value(prec, pd.Timestamp(wy, 4, 1), 3)
                p_oct = _nearest_value(prec, pd.Timestamp(wy - 1, 10, 1), 3)
                # PREC is water-year accumulated; Oct 1 value is ~0 so the
                # difference is robust either way.
                row["precip_oct_mar"] = p_apr - p_oct if np.isfinite(p_apr) and np.isfinite(p_oct) else p_apr
            if tavg is not None and not tavg.empty:
                djf = tavg[(tavg.index >= pd.Timestamp(wy - 1, 12, 1)) &
                           (tavg.index <= pd.Timestamp(wy, 2, 28))]
                row["tavg_djf"] = float(djf.mean()) if len(djf) >= 60 else np.nan
            rows.append(row)
    cols = ["stationTriplet", "water_year", "apr1_swe", "peak_swe", "peak_doy",
            "n_core_days", "precip_oct_mar", "tavg_djf"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols]


def snowcourse_water_year_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Snow courses: the April-1 measurement (semimonthly series)."""
    w = long[long["element"] == "WTEQ"].copy()
    if w.empty:
        return pd.DataFrame(columns=["stationTriplet", "water_year", "apr1_swe"])
    w = w[(w["date"].dt.month == 4) & (w["date"].dt.day <= 7)]
    w["water_year"] = w["date"].dt.year
    out = (w.sort_values("date").groupby(["stationTriplet", "water_year"], as_index=False)
            .first()[["stationTriplet", "water_year", "value"]]
            .rename(columns={"value": "apr1_swe"}))
    return out


# -------------------------------------------------------------- anomalies ---

def standardize(metrics: pd.DataFrame, col: str = "apr1_swe", min_years: int = MIN_YEARS_STATION,
                min_median: float = MIN_MEDIAN_SWE_MM, detrend: bool = True) -> pd.DataFrame:
    """Add ``<col>_pct`` (percent of station median), ``<col>_z`` (z-score) and
    ``<col>_zd`` (z-score of linearly detrended values) per station. Stations
    with fewer than ``min_years`` valid years or a median below ``min_median``
    are dropped."""
    df = metrics.dropna(subset=[col]).copy()
    keep = []
    for trip, g in df.groupby("stationTriplet"):
        if len(g) < min_years or g[col].median() < min_median:
            continue
        g = g.sort_values("water_year").copy()
        med, mu, sd = g[col].median(), g[col].mean(), g[col].std(ddof=1)
        g[f"{col}_pct"] = 100.0 * g[col] / med
        g[f"{col}_z"] = (g[col] - mu) / sd if sd > 0 else np.nan
        if detrend and len(g) >= 3:
            slope, intercept = np.polyfit(g["water_year"], g[col], 1)
            resid = g[col] - (slope * g["water_year"] + intercept)
            rsd = resid.std(ddof=1)
            g[f"{col}_zd"] = resid / rsd if rsd > 0 else np.nan
        else:
            g[f"{col}_zd"] = g[f"{col}_z"]
        keep.append(g)
    if not keep:
        return df.iloc[0:0]
    return pd.concat(keep, ignore_index=True)


def regional_means(std: pd.DataFrame, stations: pd.DataFrame, col: str,
                   min_stations: int = MIN_STATIONS_REGION_YEAR) -> pd.DataFrame:
    """Mean anomaly per (state, water_year) plus a 4-state 'ALL' region and a
    north (MT+ID) / south (CO+UT) split. Columns: region, water_year, value, n."""
    m = std.merge(stations[["stationTriplet", "stateCode"]], on="stationTriplet", how="inner")
    m["region"] = m["stateCode"]
    parts = [m]
    allr = m.copy(); allr["region"] = "ALL"; parts.append(allr)
    north = m[m["stateCode"].isin(["MT", "ID"])].copy(); north["region"] = "North (MT+ID)"
    south = m[m["stateCode"].isin(["CO", "UT"])].copy(); south["region"] = "South (CO+UT)"
    parts += [north, south]
    mm = pd.concat(parts, ignore_index=True)
    agg = (mm.groupby(["region", "water_year"])[col]
             .agg(value="mean", n="count").reset_index())
    return agg[agg["n"] >= min_stations].reset_index(drop=True)


# ------------------------------------------------------------- statistics ---

@dataclass
class CorrResult:
    region: str
    metric: str
    n: int
    pearson_r: float
    pearson_p: float
    r_ci_low: float
    r_ci_high: float
    spearman_rho: float
    spearman_p: float
    slope: float
    slope_se: float
    mean_nino: float
    mean_neutral: float
    mean_nina: float
    n_nino: int
    n_neutral: int
    n_nina: int
    welch_t_nino_vs_rest: float
    welch_p_nino_vs_rest: float
    r_within_nino: float
    p_within_nino: float
    n_within_nino: int

    def as_dict(self):
        return self.__dict__.copy()


def bootstrap_r_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 4:
        return np.nan, np.nan
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        if xi.std() == 0 or yi.std() == 0:
            rs[i] = np.nan
        else:
            rs[i] = np.corrcoef(xi, yi)[0, 1]
    return tuple(np.nanpercentile(rs, [2.5, 97.5]))


def correlate(series: pd.DataFrame, enso: pd.DataFrame, region: str, metric: str,
              index_col: str = "oni_djf", n_boot: int = 2000) -> CorrResult | None:
    """``series``: water_year, value (already regional). Returns the full set
    of statistics for value ~ ENSO index."""
    d = series.merge(enso[["water_year", index_col, "phase"]], on="water_year").dropna(
        subset=["value", index_col])
    n = len(d)
    if n < 8:
        return None
    x, y = d[index_col].to_numpy(float), d["value"].to_numpy(float)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    lo, hi = bootstrap_r_ci(x, y, n_boot)
    nino, neut, nina = (d[d["phase"] == p]["value"] for p in ("El Nino", "Neutral", "La Nina"))
    rest = d[d["phase"] != "El Nino"]["value"]
    if len(nino) >= 3 and len(rest) >= 3:
        t, tp = stats.ttest_ind(nino, rest, equal_var=False)
    else:
        t, tp = np.nan, np.nan
    dn = d[d["phase"] == "El Nino"]
    if len(dn) >= 5:
        rw, pw = stats.pearsonr(dn[index_col], dn["value"])
    else:
        rw, pw = np.nan, np.nan
    return CorrResult(region, metric, n, pr, pp, lo, hi, sr, sp, lr.slope, lr.stderr,
                      nino.mean() if len(nino) else np.nan,
                      neut.mean() if len(neut) else np.nan,
                      nina.mean() if len(nina) else np.nan,
                      len(nino), len(neut), len(nina), t, tp, rw, pw, len(dn))


def station_correlations(std: pd.DataFrame, stations: pd.DataFrame, enso: pd.DataFrame,
                         col: str, index_col: str = "oni_djf",
                         min_years: int = MIN_YEARS_STATION) -> pd.DataFrame:
    """Per-station Pearson r of the anomaly vs the ENSO index, with metadata."""
    d = std.merge(enso[["water_year", index_col]], on="water_year").dropna(subset=[col, index_col])
    rows = []
    for trip, g in d.groupby("stationTriplet"):
        if len(g) < min_years:
            continue
        r, p = stats.pearsonr(g[index_col], g[col])
        rows.append({"stationTriplet": trip, "n": len(g), "r": r, "p": p})
    out = pd.DataFrame(rows, columns=["stationTriplet", "n", "r", "p"])
    return out.merge(stations, on="stationTriplet", how="left")


def strength_composites(series: pd.DataFrame, enso: pd.DataFrame) -> pd.DataFrame:
    """Mean/median anomaly by (phase, strength) for one regional series."""
    d = series.merge(enso[["water_year", "phase", "strength", "oni_peak"]], on="water_year")
    order = {"La Nina": 0, "Neutral": 1, "El Nino": 2}
    sorder = {"very strong": 0, "strong": 1, "moderate": 2, "weak": 3, "neutral": 4}
    g = (d.groupby(["phase", "strength"])["value"]
           .agg(n="count", mean="mean", median="median", sd="std").reset_index())
    g["_o"] = g["phase"].map(order) * 10 + g["strength"].map(sorder)
    return g.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------- nClimDiv --

def nclimdiv_water_year(nc: pd.DataFrame, normal=(1991, 2020)) -> pd.DataFrame:
    """Statewide Nov–Mar precipitation (% of normal) and DJF mean temperature
    anomaly (°C) per water year. Input values are inches / °F."""
    p = nc[(nc["element"] == "pcpn") & (nc["division"] == 0)].copy()
    t = nc[(nc["element"] == "tavg") & (nc["division"] == 0)].copy()
    for df in (p, t):
        df["water_year"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])
    p = p[p["month"].isin([11, 12, 1, 2, 3])]
    pw = p.groupby(["state", "water_year"])["value"].agg(precip_in="sum", n="count").reset_index()
    pw = pw[pw["n"] == 5].drop(columns="n")
    pw["precip_mm"] = pw["precip_in"] * 25.4
    t = t[t["month"].isin([12, 1, 2])]
    tw = t.groupby(["state", "water_year"])["value"].agg(tavg_f="mean", n="count").reset_index()
    tw = tw[tw["n"] == 3].drop(columns="n")
    tw["tavg_c"] = (tw["tavg_f"] - 32) * 5 / 9
    out = pw.merge(tw, on=["state", "water_year"], how="outer")
    parts = []
    for st, g in out.groupby("state"):
        base = g[(g["water_year"] >= normal[0]) & (g["water_year"] <= normal[1])]
        g = g.copy()
        g["precip_pct_normal"] = 100 * g["precip_mm"] / base["precip_mm"].mean()
        g["tavg_anom_c"] = g["tavg_c"] - base["tavg_c"].mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["state", "water_year"]).reset_index(drop=True)
