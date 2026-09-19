"""Command line: ``python -m enso_snowpack fetch|analyze|run``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import STATES, analysis as A, figures as F, fixture as FX
from .fetch import (fetch_all_stations, fetch_mei, fetch_nclimdiv, fetch_oni, fetch_stations,
                    FetchError)
from .report import write_report
from .sources import to_metric

log = logging.getLogger("enso_snowpack")


def _paths(root: Path):
    raw, derived, results = root / "data" / "raw", root / "data" / "derived", root / "results"
    for p in (raw, derived, results):
        p.mkdir(parents=True, exist_ok=True)
    return raw, derived, results


# ------------------------------------------------------------------ fetch ---

def cmd_fetch(args) -> int:
    raw, derived, _ = _paths(args.root)
    end = args.end or date.today().isoformat()
    oni = fetch_oni(raw, args.force)
    oni.to_csv(derived / "oni.csv", index=False)
    log.info("ONI: %d seasons, %d–%d", len(oni), oni.year.min(), oni.year.max())
    mei = fetch_mei(raw, args.force)
    if mei is not None:
        mei.to_csv(derived / "mei.csv", index=False)
    try:
        nc = fetch_nclimdiv(raw, args.force)
        nc.to_csv(derived / "nclimdiv.csv", index=False)
        log.info("nClimDiv: %d monthly values", len(nc))
    except FetchError as exc:
        log.warning("nClimDiv unavailable: %s", exc)
    stations = fetch_stations(raw, "SNTL", args.force)
    stations.to_csv(derived / "stations_SNTL.csv", index=False)
    log.info("SNOTEL stations: %d", len(stations))
    daily = fetch_all_stations(raw, stations, "SNTL", end, args.force, args.max_stations)
    to_metric(daily).to_parquet(derived / "snotel_daily.parquet") if _have_parquet() \
        else to_metric(daily).to_csv(derived / "snotel_daily.csv.gz", index=False)
    log.info("SNOTEL daily rows: %d", len(daily))
    if args.snow_courses:
        try:
            courses = fetch_stations(raw, "SNOW", args.force)
            courses.to_csv(derived / "stations_SNOW.csv", index=False)
            semi = fetch_all_stations(raw, courses, "SNOW", end, args.force, args.max_stations)
            to_metric(semi).to_csv(derived / "snowcourse_semimonthly.csv.gz", index=False)
            log.info("snow courses: %d stations, %d rows", len(courses), len(semi))
        except FetchError as exc:
            log.warning("snow courses unavailable: %s", exc)
    return 0


def _have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _load_daily(derived: Path) -> pd.DataFrame:
    p = derived / "snotel_daily.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(derived / "snotel_daily.csv.gz", parse_dates=["date"])


# ---------------------------------------------------------------- analyze ---

def _inputs(args):
    raw, derived, results = _paths(args.root)
    if args.fixture:
        oni = FX.synthetic_oni()
        stations = FX.synthetic_stations(args.fixture_stations)
        daily = to_metric(FX.synthetic_daily(stations, oni))
        nc = FX.synthetic_nclimdiv(oni)
        return oni, None, stations, daily, nc, None, None
    oni = pd.read_csv(derived / "oni.csv")
    mei = pd.read_csv(derived / "mei.csv") if (derived / "mei.csv").exists() else None
    stations = pd.read_csv(derived / "stations_SNTL.csv")
    daily = _load_daily(derived)
    nc = pd.read_csv(derived / "nclimdiv.csv") if (derived / "nclimdiv.csv").exists() else None
    courses = semi = None
    if (derived / "snowcourse_semimonthly.csv.gz").exists():
        courses = pd.read_csv(derived / "stations_SNOW.csv")
        semi = pd.read_csv(derived / "snowcourse_semimonthly.csv.gz", parse_dates=["date"])
    return oni, mei, stations, daily, nc, courses, semi


def _corr_set(regional, enso, metric, index_col="oni_djf", n_boot=2000):
    rows = []
    for region in ["MT", "ID", "CO", "UT", "North (MT+ID)", "South (CO+UT)", "ALL"]:
        s = regional[regional["region"] == region][["water_year", "value"]]
        c = A.correlate(s, enso, region, metric, index_col, n_boot)
        if c is not None:
            rows.append(c.as_dict())
    return rows


def cmd_analyze(args) -> int:
    oni, mei, stations, daily, nc, courses, semi = _inputs(args)
    raw, derived, results = _paths(args.root)
    enso = A.enso_by_water_year(oni)
    enso.to_csv(results / "enso_water_years.csv", index=False)

    metrics = A.station_water_year_metrics(daily)
    metrics.to_csv(derived / "station_water_year_metrics.csv", index=False)
    ctx = {"enso": enso, "fixture": args.fixture, "min_years": A.MIN_YEARS_STATION,
           "n_stations_total": len(stations), "composites": {}}

    # --- April-1 SWE
    std = A.standardize(metrics, "apr1_swe")
    ctx["n_stations_used"] = std["stationTriplet"].nunique()
    reg = A.regional_means(std, stations, "apr1_swe_zd")
    reg.to_csv(results / "regional_apr1_zd.csv", index=False)
    ctx["corr_apr1"] = _corr_set(reg, enso, "apr1_swe_zd", n_boot=args.n_boot)
    pd.DataFrame(ctx["corr_apr1"]).to_csv(results / "corr_apr1_zd.csv", index=False)
    sc = A.station_correlations(std, stations, enso, "apr1_swe_zd")
    sc.to_csv(results / "station_corr_apr1.csv", index=False)
    ctx["station_corr_apr1"] = sc
    for region in ["MT", "ID", "CO", "UT"]:
        s = reg[reg["region"] == region][["water_year", "value"]]
        if not s.empty:
            ctx["composites"][region] = A.strength_composites(s, enso)
    # percent-of-median sensitivity
    regp = A.regional_means(std.assign(apr1_pct_anom=std["apr1_swe_pct"] - 100), stations, "apr1_pct_anom")
    ctx["corr_pct"] = _corr_set(regp, enso, "apr1_pct_anom", n_boot=args.n_boot)
    pd.DataFrame(ctx["corr_pct"]).to_csv(results / "corr_apr1_pct.csv", index=False)

    # --- peak SWE
    stdp = A.standardize(metrics[metrics["n_core_days"] >= 150], "peak_swe")
    regpk = A.regional_means(stdp, stations, "peak_swe_zd")
    regpk.to_csv(results / "regional_peak_zd.csv", index=False)
    ctx["corr_peak"] = _corr_set(regpk, enso, "peak_swe_zd", n_boot=args.n_boot)
    pd.DataFrame(ctx["corr_peak"]).to_csv(results / "corr_peak_zd.csv", index=False)

    # --- SNOTEL precipitation
    if metrics["precip_oct_mar"].notna().any():
        stdr = A.standardize(metrics, "precip_oct_mar", min_median=50.0)
        regr = A.regional_means(stdr, stations, "precip_oct_mar_zd")
        ctx["corr_snotel_precip"] = _corr_set(regr, enso, "precip_oct_mar_zd", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_snotel_precip"]).to_csv(results / "corr_snotel_precip_zd.csv", index=False)

    # --- snow courses
    if semi is not None and courses is not None and not semi.empty:
        cm = A.snowcourse_water_year_metrics(semi)
        stdc = A.standardize(cm, "apr1_swe")
        ctx["n_courses_used"] = stdc["stationTriplet"].nunique()
        regc = A.regional_means(stdc, courses, "apr1_swe_zd")
        regc.to_csv(results / "regional_courses_apr1_zd.csv", index=False)
        ctx["corr_courses"] = _corr_set(regc, enso, "course_apr1_swe_zd", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_courses"]).to_csv(results / "corr_courses_apr1_zd.csv", index=False)

    # --- nClimDiv weather
    if nc is not None and not nc.empty:
        ctx["have_nclimdiv"] = True
        wy = A.nclimdiv_water_year(nc)
        wy.to_csv(results / "nclimdiv_water_years.csv", index=False)
        regn = wy.rename(columns={"state": "region"}).assign(value=lambda d: d["precip_pct_normal"] - 100)
        regn = regn.dropna(subset=["value"])[["region", "water_year", "value"]]
        ctx["corr_precip"] = _corr_set(regn, enso, "nclimdiv_precip_pct_anom", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_precip"]).to_csv(results / "corr_nclimdiv_precip.csv", index=False)
        regt = wy.rename(columns={"state": "region", "tavg_anom_c": "value"}).dropna(subset=["value"])
        ctx["corr_temp"] = _corr_set(regt[["region", "water_year", "value"]], enso,
                                     "nclimdiv_tavg_djf_anom", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_temp"]).to_csv(results / "corr_nclimdiv_temp.csv", index=False)
        F.fig_scatter_by_state(regn, enso, results, "Nov–Mar precipitation, % of normal − 100",
                               "fig7_nclimdiv_precip")

    # --- MEI sensitivity
    if mei is not None:
        dj = mei[mei["bimonth"] == 1].rename(columns={"year": "water_year", "mei": "mei_dj"})
        e2 = enso.merge(dj[["water_year", "mei_dj"]], on="water_year", how="left")
        ctx["corr_mei"] = _corr_set(reg, e2, "apr1_swe_zd", index_col="mei_dj", n_boot=args.n_boot)

    # --- figures
    F.fig_oni_timeseries(enso, results)
    F.fig_scatter_by_state(reg, enso, results, "April-1 SWE anomaly (station z, detrended)", "fig2_scatter_apr1")
    F.fig_station_map(sc, results, "Per-station correlation of April-1 SWE with DJF ONI", "fig3_station_map_apr1")
    F.fig_phase_boxes(reg, enso, results, "April-1 SWE anomaly (z)", "fig4_phase_boxes_apr1")
    F.fig_nino_strength(reg, enso, results, "April-1 SWE anomaly (z)", "fig5_nino_strength_apr1")
    F.fig_regional_timeseries(reg, enso, results, "April-1 SWE anomaly (z)", "fig6_regional_timeseries_apr1")

    p = write_report(results, ctx)
    log.info("report written: %s", p)
    print(open(p, encoding="utf-8").read()[:3000])
    return 0


def cmd_run(args) -> int:
    if not args.fixture:
        rc = cmd_fetch(args)
        if rc:
            return rc
    return cmd_analyze(args)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="enso_snowpack")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="directory holding data/ and results/ (default: package root)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", cmd_fetch), ("analyze", cmd_analyze), ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.add_argument("--force", action="store_true", help="re-download cached inputs")
        sp.add_argument("--end", default=None, help="end date for station series (YYYY-MM-DD)")
        sp.add_argument("--max-stations", type=int, default=None, help="limit stations (smoke runs)")
        sp.add_argument("--no-snow-courses", dest="snow_courses", action="store_false")
        sp.add_argument("--fixture", action="store_true", help="use the synthetic fixture (offline)")
        sp.add_argument("--fixture-stations", type=int, default=12)
        sp.add_argument("--n-boot", type=int, default=2000)
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
