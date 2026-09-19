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

from . import STATES, analysis as A, figures as F, fixture as FX, bootstrap as B
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
    if args.bundle:
        _unpack_bundle(Path(args.bundle), derived)
    oni = pd.read_csv(derived / "oni.csv")
    mei = pd.read_csv(derived / "mei.csv") if (derived / "mei.csv").exists() else None
    stations = pd.read_csv(derived / "stations_SNTL.csv")
    daily = _load_daily(derived) if _have_daily(derived) else None
    nc = pd.read_csv(derived / "nclimdiv.csv") if (derived / "nclimdiv.csv").exists() else None
    courses = semi = None
    if (derived / "snowcourse_semimonthly.csv.gz").exists():
        courses = pd.read_csv(derived / "stations_SNOW.csv")
        semi = pd.read_csv(derived / "snowcourse_semimonthly.csv.gz", parse_dates=["date"])
    elif (derived / "stations_SNOW.csv").exists():
        courses = pd.read_csv(derived / "stations_SNOW.csv")
    return oni, mei, stations, daily, nc, courses, semi


def _have_daily(derived: Path) -> bool:
    return (derived / "snotel_daily.parquet").exists() or (derived / "snotel_daily.csv.gz").exists()


BUNDLE_FILES = ["oni.csv", "mei.csv", "nclimdiv.csv", "stations_SNTL.csv", "stations_SNOW.csv",
                "station_water_year_metrics.csv", "snowcourse_water_year_metrics.csv"]


def _unpack_bundle(bundle: Path, derived: Path) -> None:
    import tarfile
    with tarfile.open(bundle, "r:gz") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
        for m in tf.getmembers():
            if m.isfile() and Path(m.name).name in BUNDLE_FILES:
                m.name = Path(m.name).name
                tf.extract(m, derived)
    log.info("unpacked %d files from %s into %s", len(names), bundle, derived)


def cmd_bundle(args) -> int:
    """Reduce the (large) daily station data to per-station water-year metrics
    and pack every small derived table into one .tar.gz for transfer."""
    import tarfile
    raw, derived, results = _paths(args.root)
    if args.fixture:
        oni, _, stations, daily, nc, _, _ = _inputs(args)
        oni.to_csv(derived / "oni.csv", index=False)
        stations.to_csv(derived / "stations_SNTL.csv", index=False)
        nc.to_csv(derived / "nclimdiv.csv", index=False)
        A.station_water_year_metrics(daily).to_csv(derived / "station_water_year_metrics.csv", index=False)
    elif _have_daily(derived):
        metrics = A.station_water_year_metrics(_load_daily(derived))
        metrics.to_csv(derived / "station_water_year_metrics.csv", index=False)
        log.info("station metrics: %d station-years", len(metrics))
    if (derived / "snowcourse_semimonthly.csv.gz").exists():
        semi = pd.read_csv(derived / "snowcourse_semimonthly.csv.gz", parse_dates=["date"])
        A.snowcourse_water_year_metrics(semi).to_csv(derived / "snowcourse_water_year_metrics.csv", index=False)
    out = Path(args.out or (args.root / f"enso_snowpack_data_{date.today().isoformat()}.tar.gz"))
    n = 0
    with tarfile.open(out, "w:gz") as tf:
        for name in BUNDLE_FILES:
            p = derived / name
            if p.exists():
                tf.add(p, arcname=name); n += 1
    log.info("bundle written: %s (%d files, %.1f MB)", out, n, out.stat().st_size / 1e6)
    print(out)
    return 0


def _corr_set(regional, enso, metric, index_col="oni_djf", n_boot=2000):
    rows = []
    for region in ["MT", "ID", "WY", "CO", "UT", "North (MT+ID)", "South (CO+UT)", "ALL"]:
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

    if daily is not None:
        metrics = A.station_water_year_metrics(daily)
        metrics.to_csv(derived / "station_water_year_metrics.csv", index=False)
    else:
        metrics = pd.read_csv(derived / "station_water_year_metrics.csv")
        log.info("using precomputed station metrics (%d station-years)", len(metrics))
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
    sp = B.station_permutation_p(std, enso, "apr1_swe_zd", n_perm=min(args.n_perm, 2000))
    sc = sc.merge(sp[["stationTriplet", "p_perm", "fdr_significant"]], on="stationTriplet", how="left")
    sc.to_csv(results / "station_corr_apr1.csv", index=False)
    ctx["station_corr_apr1"] = sc

    # --- resampling significance (water years as the exchangeable unit)
    st_tab = std.merge(stations[["stationTriplet", "stateCode"]], on="stationTriplet")
    boots = []
    for region in ["MT", "ID", "WY", "CO", "UT", "North (MT+ID)", "South (CO+UT)", "ALL"]:
        s_ = reg[reg["region"] == region][["water_year", "value"]]
        if s_.empty:
            continue
        if region in ("MT", "ID", "WY", "CO", "UT"):
            tab = st_tab[st_tab["stateCode"] == region]
        elif region.startswith("North"):
            tab = st_tab[st_tab["stateCode"].isin(["MT", "ID"])]
        elif region.startswith("South"):
            tab = st_tab[st_tab["stateCode"].isin(["CO", "UT"])]
        else:
            tab = st_tab
        tab = tab[["water_year", "apr1_swe_zd"]].rename(columns={"apr1_swe_zd": "value"})
        b = B.run_bootstrap(s_, enso, region, "apr1_swe_zd", scheme=args.bootstrap,
                            n_boot=args.n_boot, n_perm=args.n_perm, station_table=tab,
                            omit_fraction=args.omit_fraction)
        if b is not None:
            boots.append(b.as_dict())
    ctx["boot_apr1"] = boots
    ctx["boot_scheme"] = args.bootstrap
    pd.DataFrame(boots).to_csv(results / f"bootstrap_apr1_{args.bootstrap}.csv", index=False)
    for region in ["MT", "ID", "WY", "CO", "UT"]:
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

    # --- snow depth (SNWD) and April-1 bulk density
    if metrics["apr1_depth"].notna().any():
        stdd = A.standardize(metrics, "apr1_depth", min_median=200.0)
        regd = A.regional_means(stdd, stations, "apr1_depth_zd")
        ctx["corr_depth"] = _corr_set(regd, enso, "apr1_depth_zd", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_depth"]).to_csv(results / "corr_apr1_depth_zd.csv", index=False)
        stdn = A.standardize(metrics, "apr1_density", min_median=0.05, detrend=True)
        regn_ = A.regional_means(stdn, stations, "apr1_density_zd")
        ctx["corr_density"] = _corr_set(regn_, enso, "apr1_density_zd", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_density"]).to_csv(results / "corr_apr1_density_zd.csv", index=False)

    # --- SNOTEL precipitation
    if metrics["precip_oct_mar"].notna().any():
        stdr = A.standardize(metrics, "precip_oct_mar", min_median=50.0)
        regr = A.regional_means(stdr, stations, "precip_oct_mar_zd")
        ctx["corr_snotel_precip"] = _corr_set(regr, enso, "precip_oct_mar_zd", n_boot=args.n_boot)
        pd.DataFrame(ctx["corr_snotel_precip"]).to_csv(results / "corr_snotel_precip_zd.csv", index=False)

    # --- snow courses
    cm = None
    if semi is not None and courses is not None and not semi.empty:
        cm = A.snowcourse_water_year_metrics(semi)
    elif courses is not None and (derived / "snowcourse_water_year_metrics.csv").exists():
        cm = pd.read_csv(derived / "snowcourse_water_year_metrics.csv")
    if cm is not None and not cm.empty:
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
    for name, fn in (("fetch", cmd_fetch), ("analyze", cmd_analyze), ("run", cmd_run),
                     ("bundle", cmd_bundle)):
        sp = sub.add_parser(name)
        sp.add_argument("--bundle", default=None,
                        help="analyze: a .tar.gz from `bundle` to unpack into data/derived first")
        sp.add_argument("--out", default=None, help="bundle: output .tar.gz path")
        sp.add_argument("--force", action="store_true", help="re-download cached inputs")
        sp.add_argument("--end", default=None, help="end date for station series (YYYY-MM-DD)")
        sp.add_argument("--max-stations", type=int, default=None, help="limit stations (smoke runs)")
        sp.add_argument("--no-snow-courses", dest="snow_courses", action="store_false")
        sp.add_argument("--fixture", action="store_true", help="use the synthetic fixture (offline)")
        sp.add_argument("--fixture-stations", type=int, default=12)
        sp.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples / iterations")
        sp.add_argument("--n-perm", type=int, default=5000, help="permutations for null p-values")
        sp.add_argument("--bootstrap", default="brown_harper_2026",
                        choices=["brown_harper_2026", "permutation", "block_bootstrap", "two_level"])
        sp.add_argument("--omit-fraction", type=float, default=0.20,
                        help="Brown & Harper: share of water years omitted per iteration")
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
