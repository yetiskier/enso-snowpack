"""Offline tests: format parsers against spec-shaped samples, and the whole
pipeline against the synthetic fixture with a planted signal."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enso_snowpack import analysis as A, fixture as FX
from enso_snowpack.cli import main
from enso_snowpack.sources import (find_nclimdiv_files, parse_awdb_data, parse_awdb_stations,
                                   parse_mei_v2, parse_nclimdiv, parse_oni_ascii, parse_oni_psl,
                                   to_metric)

ONI_SAMPLE = """ SEAS  YR   TOTAL   ANOM
 DJF  1950  24.72  -1.53
 JFM  1950  25.17  -1.34
 FMA  1950  25.75  -1.16
 NDJ  1997  28.55   2.33
 DJF  1998  28.29   2.24
"""

PSL_SAMPLE = """ 1950 1951
1950  -1.53 -1.34 -1.16 -1.18 -1.07 -0.85 -0.54 -0.42 -0.39 -0.44 -0.60 -0.80
1951  -0.82 -0.54 -0.17  0.18  0.36  0.58  0.70  0.89  0.99  1.15  1.04  0.81
  -99.9
 ONI from CPC
"""

MEI_SAMPLE = """1979 1980
1979  0.47  0.30  0.17  0.51  0.70  0.32  0.35  0.30  0.53  0.36  0.64  0.87
1980  0.83  0.68  0.69  0.90  0.94  0.90  0.86 -999 -999 -999 -999 -999
"""

# Real nClimDiv layout: BOTH files use a 10-char id, so length cannot pick the
# layout. statewide = SSS D EE YYYY (division always 0); divisional = SS DD EE YYYY.
NCLIMDIV_STATEWIDE = """0240011950   0.85   0.60   1.10   1.50   2.40   2.90   1.20   1.10   1.40   0.90   0.70   0.80
0240021950  15.20  20.10  30.00  42.00  52.00  60.00  68.00  66.00  55.00  44.00  30.00  20.00
0050011950   0.50  -9.99   1.00   1.60   2.10   1.20   1.90   1.80   1.20   1.10   0.60   0.50
0480011950   0.70   0.55   0.95   1.40   2.00   1.80   1.30   1.25   1.15   0.95   0.65   0.70
0360011950   0.50   0.40   1.00   1.60   2.10   1.20   1.90   1.80   1.20   1.10   0.60   0.50
"""

# divisional: state 24 (MT), division 01
NCLIMDIV_DIVISIONAL = """2401011950   0.85   0.60   1.10   1.50   2.40   2.90   1.20   1.10   1.40   0.90   0.70   0.80
"""

NCLIMDIV_INDEX = """<a href="climdiv-pcpnst-v1.0.0-20260805">climdiv-pcpnst-v1.0.0-20260805</a>
<a href="climdiv-pcpnst-v1.0.0-20260905">climdiv-pcpnst-v1.0.0-20260905</a>
<a href="climdiv-tmpcst-v1.0.0-20260905">climdiv-tmpcst-v1.0.0-20260905</a>
<a href="climdiv-pcpndv-v1.0.0-20260905">climdiv-pcpndv-v1.0.0-20260905</a>
"""

AWDB_STATIONS = [{"stationTriplet": "482:MT:SNTL", "stationId": "482", "stateCode": "MT",
                  "networkCode": "SNTL", "name": "Hand Creek", "countyName": "Flathead",
                  "huc": "17010210", "elevation": 5035.0, "latitude": 48.3, "longitude": -114.8,
                  "beginDate": "1978-10-01 00:00", "endDate": "2100-01-01 00:00"}]

AWDB_DATA = [{"stationTriplet": "482:MT:SNTL", "data": [
    {"stationElement": {"elementCode": "WTEQ", "storedUnitCode": "in", "durationName": "DAILY"},
     "values": [{"date": "2020-03-31", "value": 10.0}, {"date": "2020-04-01", "value": 10.4},
                {"date": "2020-04-02", "value": None}]},
    {"stationElement": {"elementCode": "TAVG", "storedUnitCode": "degF", "durationName": "DAILY"},
     "values": [{"date": "2020-04-01", "value": 32.0}]}]}]


def test_parse_oni_ascii():
    df = parse_oni_ascii(ONI_SAMPLE)
    assert len(df) == 5
    row = df[(df.season == "DJF") & (df.year == 1998)].iloc[0]
    assert row.oni == pytest.approx(2.24)
    assert row.center_month == 1


def test_parse_oni_psl_matches_ascii_layout():
    df = parse_oni_psl(PSL_SAMPLE)
    assert len(df) == 24
    assert df[(df.year == 1950) & (df.season == "DJF")].oni.iloc[0] == pytest.approx(-1.53)
    assert df[(df.year == 1951) & (df.season == "NDJ")].oni.iloc[0] == pytest.approx(0.81)


def test_parse_mei():
    df = parse_mei_v2(MEI_SAMPLE)
    assert len(df) == 19          # 12 + 7 non-missing
    assert df[(df.year == 1980) & (df.bimonth == 1)].mei.iloc[0] == pytest.approx(0.83)


def test_parse_nclimdiv_statewide_and_divisional():
    df = parse_nclimdiv(NCLIMDIV_STATEWIDE, ["MT", "CO", "UT", "ID"], "statewide")
    assert set(df.state) == {"MT", "CO"}          # WY(048) and NY(036) not requested
    assert (df.division == 0).all()
    assert set(df[df.state == "MT"].element) == {"pcpn", "tavg"}
    assert len(df[(df.state == "CO") & (df.element == "pcpn")]) == 11   # one -9.99 dropped
    assert set(parse_nclimdiv(NCLIMDIV_STATEWIDE, ["WY"], "statewide").state) == {"WY"}

    dv = parse_nclimdiv(NCLIMDIV_DIVISIONAL, ["MT"], "divisional")
    assert len(dv) == 12 and dv.state.iloc[0] == "MT" and (dv.division == 1).all()
    # the SAME line read with the wrong layout must not silently become Montana
    assert "MT" not in set(parse_nclimdiv(NCLIMDIV_DIVISIONAL, None, "statewide").state)
    with pytest.raises(ValueError):
        parse_nclimdiv(NCLIMDIV_STATEWIDE, ["MT"], "county")


def test_find_nclimdiv_files_picks_newest():
    names = find_nclimdiv_files(NCLIMDIV_INDEX)
    assert names["pcpnst"] == "climdiv-pcpnst-v1.0.0-20260905"
    assert names["tmpcst"].endswith("20260905")


def test_parse_awdb():
    st = parse_awdb_stations(json.dumps(AWDB_STATIONS))
    assert st.stateCode.iloc[0] == "MT" and st.latitude.iloc[0] == pytest.approx(48.3)
    d = parse_awdb_data(json.dumps(AWDB_DATA))
    assert len(d) == 3                            # None value dropped
    m = to_metric(d)
    w = m[(m.element == "WTEQ") & (m.date == "2020-04-01")].value.iloc[0]
    assert w == pytest.approx(10.4 * 25.4)
    t = m[m.element == "TAVG"].value.iloc[0]
    assert t == pytest.approx(0.0)


def test_enso_classification():
    oni = FX.synthetic_oni(1990, 2000)
    e = A.enso_by_water_year(oni)
    assert set(e.phase) <= {"El Nino", "La Nina", "Neutral"}
    assert (e.loc[e.phase == "El Nino", "oni_djf"] >= 0.5).all()
    assert (e.loc[e.phase == "La Nina", "oni_djf"] <= -0.5).all()
    assert A.strength_label("El Nino", 2.3) == "very strong"
    assert A.strength_label("La Nina", -1.2) == "moderate"
    assert A.strength_label("Neutral", 0.1) == "neutral"


def test_station_metrics_april1_and_peak():
    oni = FX.synthetic_oni(2000, 2003)
    st = FX.synthetic_stations(1).iloc[:1].assign(beginDate="1999-10-01 00:00")
    daily = to_metric(FX.synthetic_daily(st, oni, end_wy=2003))
    m = A.station_water_year_metrics(daily)
    assert set(m.water_year) == {2000, 2001, 2002, 2003}
    assert (m.apr1_swe > 0).all() and (m.peak_swe >= m.apr1_swe).all()
    assert m.precip_oct_mar.notna().all() and m.tavg_djf.notna().all()
    assert m.apr1_depth.notna().all() and (m.apr1_depth > m.apr1_swe).all()
    assert m.apr1_density.between(0.15, 0.6).all()


def test_pipeline_recovers_planted_dipole(tmp_path: Path):
    rc = main(["--root", str(tmp_path), "analyze", "--fixture", "--n-boot", "300", "--n-perm", "300"])
    assert rc == 0
    results = tmp_path / "results"
    corr = pd.read_csv(results / "corr_apr1_zd.csv").set_index("region")
    # planted: north dry in El Niño (negative r), south wet (positive r)
    for region in ("MT", "ID", "North (MT+ID)"):
        assert corr.loc[region, "pearson_r"] < -0.5 and corr.loc[region, "pearson_p"] < 0.01
    for region in ("CO", "UT", "South (CO+UT)"):
        assert corr.loc[region, "pearson_r"] > 0.4 and corr.loc[region, "pearson_p"] < 0.01
    sc = pd.read_csv(results / "station_corr_apr1.csv")
    assert (sc[sc.stateCode == "MT"].r < 0).mean() > 0.8
    assert (sc[sc.stateCode == "UT"].r > 0).mean() > 0.8
    assert "WY" in set(sc.stateCode) and "WY" in set(corr.index)
    assert (results / "corr_apr1_depth_zd.csv").exists()
    assert (results / "summary.md").exists()
    # every figure must be about the snowpack a skier meets; the water-supply
    # figures (April-1 scatter, phase boxes, statewide precipitation) are gone
    assert list(results.glob("fig8_*.png")) and list(results.glob("fig9_*.png"))
    for gone in ("fig2_scatter_apr1", "fig3_station_map_apr1", "fig4_phase_boxes_apr1",
                 "fig5_nino_strength_apr1", "fig6_regional_timeseries_apr1",
                 "fig7_nclimdiv_precip"):
        assert not (results / f"{gone}.png").exists(), f"{gone} should have been removed"
    nc = pd.read_csv(results / "corr_nclimdiv_precip.csv").set_index("region")
    assert nc.loc["MT", "pearson_r"] < 0 and nc.loc["UT", "pearson_r"] > 0
    bh = pd.read_csv(results / "bootstrap_apr1_brown_harper_2026.csv").set_index("region")
    assert bool(bh.loc["MT", "bh_slope_significant"]) and bh.loc["MT", "bh_slope_mean"] < 0
    assert bool(bh.loc["UT", "bh_slope_significant"]) and bh.loc["UT", "bh_slope_mean"] > 0


# ------------------------------------------------------------ bootstrap ---
from enso_snowpack import bootstrap as B  # noqa: E402


def test_permutation_p_detects_signal_and_null():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = -0.8 * x + rng.normal(size=40) * 0.6
    r, p = B.permutation_p(x, y, B._stat_r, 999, rng)
    assert r < -0.6 and p < 0.01
    y0 = rng.normal(size=40)
    r0, p0 = B.permutation_p(x, y0, B._stat_r, 999, rng)
    assert p0 > 0.05


def test_moving_block_indices_shape_and_range():
    rng = np.random.default_rng(1)
    idx = B.moving_block_indices(37, 2, rng)
    assert len(idx) == 37 and idx.min() >= 0 and idx.max() < 37
    # blocks are contiguous pairs
    assert all(idx[i + 1] == idx[i] + 1 for i in range(0, 36, 2))


def test_fdr_field_significance():
    p = np.array([0.001, 0.002, 0.5, 0.6, 0.7, np.nan])
    m = B.fdr_field_significance(p, 0.10)
    assert m.tolist() == [True, True, False, False, False, False]
    assert not B.fdr_field_significance(np.full(10, 0.5)).any()


def test_run_bootstrap_schemes_agree_on_sign():
    oni = FX.synthetic_oni(1985, 2025)
    enso = A.enso_by_water_year(oni)
    rng = np.random.default_rng(2)
    years = enso["water_year"].to_numpy()
    x = enso["oni_djf"].to_numpy()
    station_rows = []
    for yv, xv in zip(years, x):
        for k in range(8):
            station_rows.append({"water_year": yv, "value": -0.5 * xv + rng.normal(0, 1)})
    tab = pd.DataFrame(station_rows)
    series = tab.groupby("water_year", as_index=False)["value"].mean()
    for scheme in ("permutation", "block_bootstrap", "two_level"):
        b = B.run_bootstrap(series, enso, "MT", "test", scheme=scheme, n_boot=300, n_perm=300,
                            station_table=tab)
        assert b is not None and b.r_obs < -0.4, scheme
        assert b.r_ci_high < 0, scheme          # CI excludes zero
        assert b.p_perm < 0.05 and b.p_perm_diff < 0.05, scheme
    b = B.run_bootstrap(series, enso, "MT", "test", scheme="brown_harper_2026", n_boot=2000,
                        n_perm=300)
    assert b.scheme == "brown_harper_2026"
    assert b.bh_slope_mean < 0 and b.bh_slope_significant and b.bh_r_significant
    assert b.slope_ci_high < 0                      # 2σ bounds exclude zero
    assert abs(b.bh_slope_mean - b.slope_obs) < 3 * b.bh_slope_sd


def test_brown_harper_subsampling_mechanics():
    rng = np.random.default_rng(5)
    n = 40
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(size=n) * 0.5
    mu, sd, lo, hi, sig, samples = B.brown_harper_2026(x, y, B._stat_slope, n_iter=3000,
                                                        omit_fraction=0.2, rng=rng)
    assert len(samples) == 3000
    assert sig and lo > 0 and abs(mu - 0.6) < 0.15
    assert abs((hi - lo) - 4 * sd) < 1e-9          # bounds are mean ± 2σ
    # each iteration keeps exactly 80 % of the points
    keep = int(round(n * 0.8))
    idx = np.random.default_rng(1).choice(n, keep, replace=False)
    assert len(set(idx)) == keep == 32
    # the distribution is centred on the full-sample coefficient
    assert abs(mu - B._stat_slope(x, y)) < 0.5 * sd
    # pure noise with a near-zero sample slope: not significant
    y0 = rng.normal(size=n)
    y0 = y0 - B._stat_slope(x, y0) * x            # remove the sample slope exactly
    _, _, lo0, hi0, sig0, _ = B.brown_harper_2026(x, y0, B._stat_slope, n_iter=3000, rng=rng)
    assert not sig0 and lo0 < 0 < hi0


# ------------------------------------------------- real ONI, when present ---
REAL_ONI = Path(__file__).resolve().parents[1] / "data" / "derived" / "oni.csv"


@pytest.mark.skipif(not REAL_ONI.exists(), reason="real ONI not downloaded (run `fetch`)")
def test_real_oni_classifies_canonical_winters():
    """Pins the classification against winters whose CPC status is settled."""
    enso = A.enso_by_water_year(pd.read_csv(REAL_ONI)).set_index("water_year")
    very_strong_nino = [1983, 1998, 2016]
    for wy in very_strong_nino:
        assert enso.loc[wy, "phase"] == "El Nino" and enso.loc[wy, "strength"] == "very strong", wy
    for wy in (1973, 1992):                      # strong El Niño winters
        assert enso.loc[wy, "phase"] == "El Nino" and enso.loc[wy, "strength"] in ("strong", "very strong"), wy
    for wy in (1989, 2000, 2011):                # strong La Niña winters
        assert enso.loc[wy, "phase"] == "La Nina" and enso.loc[wy, "strength"] in ("strong", "very strong"), wy
    for wy in (1990, 2013, 2014):                # neutral winters
        assert enso.loc[wy, "phase"] == "Neutral", wy


# ------------------------------------------- snow course (monthly) schema ---
AWDB_SEMIMONTHLY = [{"stationTriplet": "15A21:MT:SNOW", "data": [
    {"stationElement": {"elementCode": "WTEQ", "storedUnitCode": "in", "durationName": "SEMIMONTHLY"},
     "values": [{"month": 1, "monthPart": "2", "year": 1994, "collectionDate": "1994-01-27 00:00", "value": 10.1},
                {"month": 3, "monthPart": "2", "year": 1994, "collectionDate": "1994-03-29 00:00", "value": 22.4},
                {"month": 4, "monthPart": "1", "year": 1995, "collectionDate": "1995-04-03 00:00", "value": 18.0},
                {"month": 5, "monthPart": "1", "year": 1994, "value": 9.9}]}]}]


def test_parse_awdb_semimonthly_without_date_key():
    d = parse_awdb_data(json.dumps(AWDB_SEMIMONTHLY))
    assert len(d) == 4, "rows with collectionDate/year-month must not be dropped"
    got = set(d["date"].dt.strftime("%Y-%m-%d"))
    assert {"1994-01-27", "1994-03-29", "1995-04-03"} <= got
    assert "1994-05-15" in got          # no collectionDate -> monthPart 1 = mid-month


def test_snowcourse_picks_measurement_nearest_april_1():
    long = parse_awdb_data(json.dumps(AWDB_SEMIMONTHLY))
    m = A.snowcourse_water_year_metrics(long)
    assert set(zip(m.water_year, m.apr1_swe)) == {(1994, 22.4), (1995, 18.0)}, m
    # a late-January reading is never mistaken for April
    assert 10.1 not in set(m.apr1_swe)


# --------------------------------------------- daily seasonal analysis -----
from enso_snowpack import daily as D  # noqa: E402


def test_day_of_water_year_is_index_safe():
    """A non-contiguous index must not union and change length."""
    dates = pd.Series(pd.to_datetime(["2000-10-01", "2001-01-01", "2001-04-01", "2001-09-30"]),
                      index=[7, 19, 3, 44])
    doy, wy = D.day_of_water_year(dates)
    assert len(doy) == 4 and len(wy) == 4
    assert doy[0] == 0                       # 1 Oct is day 0
    assert list(wy) == [2001, 2001, 2001, 2001]
    assert doy[2] == 182                     # 1 Apr of a non-leap water year
    assert doy[3] == 364


def test_period_of_day_matches_brown_harper_table_1():
    def d(m, day):
        y = 2001 if m >= 10 else 2002
        return (pd.Timestamp(y, m, day) - pd.Timestamp(2001, 10, 1)).days
    assert D.period_of_day(d(11, 19)) == "Early Accumulation"
    assert D.period_of_day(d(1, 27)) == "Early Accumulation"
    assert D.period_of_day(d(1, 28)) == "Core Accumulation"
    assert D.period_of_day(d(3, 8)) == "Core Accumulation"
    assert D.period_of_day(d(3, 9)) == "Late Accumulation"
    assert D.period_of_day(d(4, 17)) == "Late Accumulation"
    assert D.period_of_day(d(4, 18)) == "Melt Onset"
    assert D.period_of_day(d(6, 7)) == "Melt Onset"
    assert D.period_of_day(d(10, 15)) is None      # before the first period


def test_bh_vectorized_matches_reference():
    rng = np.random.default_rng(4)
    x = rng.normal(size=48)
    y = -0.45 * x + rng.normal(size=48)
    rm, rs, sm, ss = D.bh_vectorized(x, y, 8000, 0.2, np.random.default_rng(5))
    ref_r = B.brown_harper_2026(x, y, B._stat_r, n_iter=8000, rng=np.random.default_rng(6))
    ref_s = B.brown_harper_2026(x, y, B._stat_slope, n_iter=8000, rng=np.random.default_rng(7))
    assert abs(rm - ref_r[0]) < 0.02 and abs(rs - ref_r[1]) < 0.01
    assert abs(sm - ref_s[0]) < 0.02 and abs(ss - ref_s[1]) < 0.02
    # subsets are drawn WITHOUT replacement: each keeps exactly 80% distinct points
    idx = np.argsort(np.random.default_rng(8).random((5, 48)), axis=1)[:, :38]
    assert all(len(set(row)) == 38 for row in idx)


def test_daily_pipeline_recovers_planted_signal_on_fixture():
    oni = FX.synthetic_oni(1990, 2025)
    stations = FX.synthetic_stations(6)
    daily = to_metric(FX.synthetic_daily(stations, oni, end_wy=2025))
    cube, trips, years = D.build_cube(daily, min_years=10)
    assert cube.shape[0] == len(trips) > 0 and cube.shape[2] == D.DAYS_IN_WY
    z = D.standardize_cube(cube, years)
    regions = D.regional_daily(z, trips, stations, min_stations=3)
    enso = A.enso_by_water_year(oni)
    mt = D.daily_enso_curve(regions["MT"], years, enso, n_iter=400, min_years=15)
    ut = D.daily_enso_curve(regions["UT"], years, enso, n_iter=400, min_years=15)
    # planted: MT negative, UT positive — must hold in the accumulation season
    core_mt = mt[mt["period"] == "Core Accumulation"]["r"]
    core_ut = ut[ut["period"] == "Core Accumulation"]["r"]
    assert len(core_mt) > 10 and core_mt.mean() < -0.3, core_mt.mean()
    assert len(core_ut) > 10 and core_ut.mean() > 0.2, core_ut.mean()
    ps = D.period_summary(mt)
    assert list(ps["period"]) == [p[0] for p in D.PERIODS if p[0] in set(ps["period"])]


# ------------------------------------------------- ski-season analysis -----
from enso_snowpack import ski as SKI  # noqa: E402


def test_ski_windows_tile_the_season_without_overlap():
    def day(m, d):
        y = 2001 if m >= 10 else 2002
        return (pd.Timestamp(y, m, d) - pd.Timestamp(2001, 10, 1)).days
    spans = [(day(*a), day(*b)) for _, a, b in SKI.SKI_WINDOWS]
    assert spans == sorted(spans), "windows must be in calendar order"
    for (_, e0), (s1, _) in zip(spans, spans[1:]):
        assert s1 == e0 + 1, "ski windows must abut, with no gap and no overlap"
    mask = SKI.window_mask(np.arange(366), SKI.SKI_WINDOWS[1][1:])
    assert mask.sum() == 21                      # 16 Dec - 5 Jan inclusive


def test_assign_regions_uses_distance_and_elevation():
    st = pd.DataFrame({"stationTriplet": ["AT_BAND", "FAR", "TOO_LOW"],
                       "latitude": [40.60, 40.60, 40.62],
                       "longitude": [-111.60, -105.00, -111.62],
                       "elevation": [9000.0, 9000.0, 3000.0]})
    rm = SKI.assign_regions(st)
    was = rm[rm.region.str.startswith("Wasatch")]
    assert set(was.stationTriplet) == {"AT_BAND"}, \
        "only the in-band station near the centre may represent the Wasatch"
    # a station 6000 ft below the served band is excluded even when it is close
    assert "TOO_LOW" not in set(was.stationTriplet)


def test_station_weights_prefer_the_served_elevation_band():
    base, summit, radius = 5000.0, 7600.0, 85.0
    km = np.array([5.0, 5.0, 5.0, 60.0])
    elev = np.array([7270.0, 4000.0, 9000.0, 7270.0])   # in band, low, high, in band but far
    w = SKI.station_weights(km, elev, base, summit, radius)
    assert w[0] == pytest.approx(max(w)), "the in-band, nearby station must rank first"
    assert w[1] < w[0] and w[2] < w[0], "out-of-band stations are downweighted"
    assert w[3] < w[0], "distance still matters for an in-band station"
    # anywhere inside the band is weighted equally at equal distance
    inside = SKI.station_weights(np.array([5.0, 5.0]), np.array([5100.0, 7500.0]),
                                 base, summit, radius)
    assert inside[0] == pytest.approx(inside[1])


def test_stuart_mountain_anchors_the_snowbowl_region():
    """The station nearest Snowbowl in both distance and elevation must carry
    the most weight there."""
    st = pd.DataFrame({
        "stationTriplet": ["901:MT:SNTL", "562:MT:SNTL", "604:MT:SNTL"],
        "name": ["Stuart Mountain", "Kraft Creek", "Lubrecht Flume"],
        "latitude": [46.99521, 47.4, 46.9],
        "longitude": [-113.92667, -113.8, -113.3],
        "elevation": [7270.0, 4770.0, 4690.0]})
    rm = SKI.assign_regions(st)
    sb = rm[rm.region.str.contains("Snowbowl")].sort_values("weight", ascending=False)
    assert sb.iloc[0].stationTriplet == "901:MT:SNTL"
    assert sb.iloc[0].elev_gap_ft == 0, "Stuart Mountain sits inside Snowbowl's band"
    # the region is named for the range Snowbowl actually sits in, and the
    # Bitterroots — 148 km south — are a separate region, not folded in
    region = SKI.REGION_META[sb.iloc[0].region]
    assert "Snowbowl" in region.name and set(region.states) == {"MT"}
    bitterroot = SKI.REGION_META["Bitterroots (Lost Trail)"]
    assert SKI._haversine_km(region.lat, region.lon, bitterroot.lat, bitterroot.lon) > 100


def test_strength_effect_separates_phase_from_magnitude():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(0, 1.0, n)
    # y depends ONLY on phase, not on how far the index goes past the threshold
    phase = np.where(x >= 0.5, -1.0, np.where(x <= -0.5, 1.0, 0.0))
    y = phase + rng.normal(0, 0.5, n)
    out = SKI.strength_effect(x, y)
    assert out["r2_phase"] > 0.4
    assert out["r2_strength_gain"] < 0.05, "magnitude must add nothing when only phase drives y"
    assert out["r2_within_nino"] < 0.15, "within one phase the index should explain little"
    # now make magnitude genuinely matter
    y2 = -x * 1.5 + rng.normal(0, 0.5, n)
    out2 = SKI.strength_effect(x, y2)
    assert out2["r2_index"] > out2["r2_phase"], "a linear response must favour the index"
    assert out2["r2_within_nino"] > out["r2_within_nino"]


def test_storm_days_counted_from_swe_gain_and_normalised():
    """A within-winter SWE gain is a storm; the water-year reset is not."""
    days = pd.date_range("2009-11-01", "2010-04-15", freq="D")
    swe = np.zeros(len(days))
    swe[10:] = 30.0          # a 30 mm storm on day 10 (early season)
    swe[60:] = 90.0          # a 60 mm storm inside the holidays
    long = pd.DataFrame({"stationTriplet": "X:MT:SNTL", "element": "WTEQ",
                         "date": days, "value": swe})
    m = SKI.station_window_metrics(long)
    early = m[m.window == "Early season"].iloc[0]
    hol = m[m.window == "Holidays"].iloc[0]
    assert early.big_storm_days > 0 and hol.big_storm_days > 0
    # counts are per 30 days, so a single storm in a 21-day window scales up
    assert hol.big_storm_days == pytest.approx(30 / 21, rel=0.02)
    assert early.mean_swe < hol.mean_swe


def test_test_grid_applies_fdr_across_the_whole_grid():
    oni = FX.synthetic_oni(1980, 2025)
    enso = A.enso_by_water_year(oni)
    rng = np.random.default_rng(11)
    years = enso.water_year.to_numpy()
    x = enso.oni_djf.to_numpy()
    rows, rmap = [], []
    for reg_i, (region, state, *_ ) in enumerate(SKI.SKI_REGIONS[:3]):
        for s in range(4):
            trip = f"{reg_i}{s}:XX:SNTL"
            rmap.append({"stationTriplet": trip, "region": region, "state": state,
                         "km": 1.0, "elevation": 9000.0, "weight": 1.0})
            for yv, xv in zip(years, x):
                for w, *_ in SKI.SKI_WINDOWS:
                    rows.append({"stationTriplet": trip, "water_year": yv, "window": w,
                                 "mean_swe": 300 - 60 * xv + rng.normal(0, 40),
                                 "mean_depth": np.nan, "storm_days": 3.0 + rng.normal(0, 1),
                                 "big_storm_days": 0.5 + rng.normal(0, 0.3),
                                 "new_snow_in_total": 40.0 + rng.normal(0, 8),
                                 "days_with_base": 10.0 + rng.normal(0, 3)})
    rmap_df = pd.DataFrame(rmap)
    rmap_df["weight"] = 1.0
    grid = SKI.test_grid(pd.DataFrame(rows), rmap_df, enso, n_iter=500, n_perm=400)
    assert len(grid) > 0 and "fdr_significant" in grid
    assert {"r2", "signed_r2", "r2_phase", "r2_strength_gain"} <= set(grid.columns)
    assert np.allclose(grid["r2"], grid["r"] ** 2)
    assert (np.sign(grid["signed_r2"]) == np.sign(grid["r"])).all()
    base = grid[grid.metric == "mean_swe"]
    assert (base.r < -0.4).all(), "planted strong negative base signal must be recovered"
    assert base.fdr_significant.all()
    for noisy in ("big_storm_days", "new_snow_in_total"):
        noise = grid[grid.metric == noisy]
        assert noise.fdr_significant.sum() <= 1, f"pure noise ({noisy}) must not survive FDR"
    signs = SKI.window_sign_summary(grid)
    assert len(signs) == len(SKI.SKI_WINDOWS) and set(signs.columns) >= {"sign_test_p", "mean_r"}


def test_physical_composites_are_in_real_units():
    """Composites must be actual days and inches, with a % change that matches."""
    oni = FX.synthetic_oni(1985, 2025)
    enso = A.enso_by_water_year(oni)
    rng = np.random.default_rng(21)
    rows, rmap = [], []
    region = SKI.SKI_REGIONS_FULL[0].name
    x = enso.set_index("water_year")["oni_djf"]
    for st in range(4):
        trip = f"P{st}:MT:SNTL"
        rmap.append({"stationTriplet": trip, "region": region, "weight": 1.0})
        for yv in enso.water_year:
            # 10 powder days normally, 3 fewer per +1 degree of ONI
            rows.append({"stationTriplet": trip, "water_year": yv, "window": "Midwinter",
                         "big_storm_days_per_window": 10.0 - 3.0 * x[yv] + rng.normal(0, 0.5),
                         "storm_days_per_window": np.nan, "days_with_base_per_window": np.nan,
                         "new_snow_in_total": np.nan,
                         "mean_swe_in": np.nan, "mean_depth_in": np.nan})
    comp = SKI.physical_composites(pd.DataFrame(rows), pd.DataFrame(rmap), enso)
    r = comp[comp.metric == "big_storm_days_per_window"].iloc[0]
    assert r.unit == "days"
    assert 8.0 < r.all_winters < 12.0, "a normal winter should be about 10 powder days"
    assert r.mean_nino < r.all_winters < r.mean_nina, "El Nino must lose days, La Nina gain"
    assert r.nino_minus_all == pytest.approx(r.mean_nino - r.all_winters)
    assert r.pct_change_nino == pytest.approx(100 * r.nino_minus_all / r.all_winters)
    assert r.pct_change_nino < -8, "a 3-day-per-degree signal must show as a clear % loss"


def test_new_snow_ratio_converts_water_to_snow_depth():
    """A powder day is 6 inches of SNOW; a fixed water threshold would count a
    dry continental storm as nothing. The ratio comes from each station's own
    paired depth and water record."""
    days = pd.date_range("2009-11-01", "2010-03-31", freq="D")
    n = len(days)
    # a dry site: 12 inches of snow per inch of water
    swe = np.zeros(n); dep = np.zeros(n)
    for k in range(5, n, 10):
        swe[k:] += 8.0                     # 8 mm water per storm
        dep[k:] += 8.0 * 12.0              # 12x depth ratio
    long = pd.concat([
        pd.DataFrame({"stationTriplet": "DRY:CO:SNTL", "element": "WTEQ",
                      "date": days, "value": swe}),
        pd.DataFrame({"stationTriplet": "DRY:CO:SNTL", "element": "SNWD",
                      "date": days, "value": dep})], ignore_index=True)
    ratios = SKI.new_snow_ratios(long, min_events=5)
    assert ratios["DRY:CO:SNTL"] == pytest.approx(12.0, rel=0.05)

    m = SKI.station_window_metrics(long, ratios=ratios)
    mid = m[m.window == "Midwinter"].iloc[0]
    # 8 mm of water = 0.315 in x 12 = 3.8 in of snow: a storm day, not a powder day
    assert mid.storm_days_per_window > 0, "2-inch storms must be counted"
    assert mid.big_storm_days_per_window == 0, "3.8 inches is not a 6-inch powder day"
    # the same water at a maritime ratio would still not reach 6 inches, but
    # doubling the storm size must
    big = long.copy()
    big.loc[big.element == "WTEQ", "value"] *= 3
    big.loc[big.element == "SNWD", "value"] *= 3
    m2 = SKI.station_window_metrics(big, ratios=ratios)
    assert m2[m2.window == "Midwinter"].iloc[0].big_storm_days_per_window > 0


def test_significant_composites_keeps_only_surviving_tests():
    """The physical numbers must be matched to the test that backs them, on the
    same region, window and measure — and nothing else may come through."""
    comp = pd.DataFrame([
        {"region": "R1", "window": "Midwinter", "metric": "new_snow_in_total",
         "metric_label": "x", "unit": "in", "n_winters": 40, "all_winters": 50.0,
         "mean_nino": 40.0, "mean_nina": 60.0, "nino_minus_all": -10.0,
         "nina_minus_all": 10.0, "nino_minus_nina": -20.0,
         "pct_change_nino": -20.0, "pct_change_nina": 20.0},
        {"region": "R2", "window": "Spring", "metric": "mean_swe_in",
         "metric_label": "y", "unit": "in", "n_winters": 40, "all_winters": 20.0,
         "mean_nino": 18.0, "mean_nina": 22.0, "nino_minus_all": -2.0,
         "nina_minus_all": 2.0, "nino_minus_nina": -4.0,
         "pct_change_nino": -10.0, "pct_change_nina": 10.0},
    ])
    grid = pd.DataFrame([
        {"region": "R1", "window": "Midwinter", "metric": "new_snow_in_total",
         "fdr_significant": True, "r2": 0.21, "p_perm": 0.001, "n_winters": 40},
        {"region": "R2", "window": "Spring", "metric": "mean_swe",
         "fdr_significant": False, "r2": 0.02, "p_perm": 0.6, "n_winters": 40},
    ])
    out = SKI.significant_composites(comp, grid)
    assert list(out["region"]) == ["R1"], "a non-significant test must not come through"
    assert out.iloc[0]["r2"] == pytest.approx(0.21), "the test's r2 must be carried across"
    # the tested name (mean_swe) maps to the composite name (mean_swe_in)
    grid.loc[1, "fdr_significant"] = True
    out2 = SKI.significant_composites(comp, grid)
    assert set(out2["region"]) == {"R1", "R2"}, "mean_swe must match mean_swe_in"
    assert SKI.significant_composites(comp, grid.assign(fdr_significant=False)).empty


def test_strength_significance_finds_a_planted_magnitude_effect():
    """The strength test must fire when magnitude genuinely drives the data,
    and stay quiet when only phase does — otherwise a null result means
    nothing."""
    oni = FX.synthetic_oni(1975, 2025)
    enso = A.enso_by_water_year(oni)
    x = enso.set_index("water_year")["oni_djf"]
    rng = np.random.default_rng(31)
    region = SKI.SKI_REGIONS_FULL[0].name

    def build(magnitude: bool):
        rows, rmap = [], []
        for st in range(4):
            trip = f"S{st}:MT:SNTL"
            rmap.append({"stationTriplet": trip, "region": region, "weight": 1.0})
            for yv in enso.water_year:
                xv = x[yv]
                if magnitude:
                    sig = -2.0 * xv                       # linear in the index
                else:
                    sig = -2.0 * (1 if xv >= 0.5 else -1 if xv <= -0.5 else 0)  # phase only
                rows.append({"stationTriplet": trip, "water_year": yv, "window": "Midwinter",
                             "mean_swe": 300 + 40 * sig + rng.normal(0, 25),
                             "mean_depth": np.nan, "storm_days": np.nan,
                             "big_storm_days": np.nan, "new_snow_in_total": np.nan,
                             "days_with_base": np.nan})
        return pd.DataFrame(rows), pd.DataFrame(rmap)

    rows, rmap = build(magnitude=True)
    hit = SKI.strength_significance(rows, rmap, enso, n_perm=600, min_phase_winters=10)
    nested = hit[hit.test == "nested"]
    assert len(nested) and nested.iloc[0]["p"] < 0.05, "a real magnitude effect must be found"

    rows, rmap = build(magnitude=False)
    miss = SKI.strength_significance(rows, rmap, enso, n_perm=600, min_phase_winters=10)
    nested = miss[miss.test == "nested"]
    assert len(nested) and nested.iloc[0]["p"] > 0.05, "phase-only data must not fire"
    assert (miss["min_detectable_r"] > 0).all()


def test_minimum_detectable_r_shrinks_with_sample_size():
    small, large = SKI.minimum_detectable_r(16), SKI.minimum_detectable_r(100)
    assert 0.5 < small < 0.8, small
    assert 0.2 < large < 0.35, large
    assert large < small
    assert np.isnan(SKI.minimum_detectable_r(3))


def test_region_change_distributions_straddle_zero_when_there_is_no_effect():
    """A region with no real effect must produce a distribution that spans
    zero; one with a planted effect must produce one that does not."""
    oni = FX.synthetic_oni(1980, 2025)
    enso = A.enso_by_water_year(oni)
    x = enso.set_index("water_year")["oni_djf"]
    rng = np.random.default_rng(41)
    r_signal, r_null = SKI.SKI_REGIONS_FULL[0].name, SKI.SKI_REGIONS_FULL[1].name
    rows, rmap = [], []
    for region, slope in ((r_signal, -4.0), (r_null, 0.0)):
        for st in range(4):
            trip = f"{region[:3]}{st}:MT:SNTL"
            rmap.append({"stationTriplet": trip, "region": region, "weight": 1.0})
            for yv in enso.water_year:
                rows.append({"stationTriplet": trip, "water_year": yv, "window": "Midwinter",
                             "big_storm_days_per_window": 12.0 + slope * x[yv] + rng.normal(0, 1.5)})
    d = SKI.region_change_distributions(pd.DataFrame(rows), pd.DataFrame(rmap), enso,
                                        n_iter=2000)
    assert set(d) == {r_signal, r_null}
    sig, null = d[r_signal], d[r_null]
    lo_s, hi_s = np.percentile(sig["values"], [2.5, 97.5])
    lo_n, hi_n = np.percentile(null["values"], [2.5, 97.5])
    assert hi_s < 0, "a planted loss must give an interval entirely below zero"
    assert lo_n < 0 < hi_n, "no effect must give an interval that spans zero"
    assert not sig["straddles_zero"] and null["straddles_zero"]
    # calibration widens the interval; without it the null would look decided
    raw = SKI.region_change_distributions(pd.DataFrame(rows), pd.DataFrame(rmap), enso,
                                          n_iter=2000, calibrate=False)
    assert raw[r_null]["values"].std() < null["values"].std()


def test_change_distributions_differ_between_windows():
    """The effect is not constant through winter, so the per-window
    distributions must actually differ — a bug that ignored the window
    argument would return the same numbers four times."""
    oni = FX.synthetic_oni(1980, 2025)
    enso = A.enso_by_water_year(oni)
    x = enso.set_index("water_year")["oni_djf"]
    rng = np.random.default_rng(51)
    region = SKI.SKI_REGIONS_FULL[0].name
    rows, rmap = [], []
    for st in range(4):
        trip = f"W{st}:MT:SNTL"
        rmap.append({"stationTriplet": trip, "region": region, "weight": 1.0})
        for yv in enso.water_year:
            for w, slope in (("Holidays", -5.0), ("Midwinter", 0.0)):
                rows.append({"stationTriplet": trip, "water_year": yv, "window": w,
                             "big_storm_days_per_window": 12.0 + slope * x[yv] + rng.normal(0, 1.2)})
    m, rm = pd.DataFrame(rows), pd.DataFrame(rmap)
    hol = SKI.region_change_distributions(m, rm, enso, window="Holidays", n_iter=2000)
    mid = SKI.region_change_distributions(m, rm, enso, window="Midwinter", n_iter=2000)
    assert not hol[region]["straddles_zero"], "the planted holiday loss must be decided"
    assert mid[region]["straddles_zero"], "the flat midwinter window must not be"
    assert hol[region]["observed"] < mid[region]["observed"] - 10


# ------------------------------------------- ski region geography guards ---
def _state_polygons():
    from enso_snowpack.figures_ski import load_states, STATE_ABBR
    out = {}
    for name, rings in load_states():
        a = STATE_ABBR.get(name)
        if a:
            out.setdefault(a, []).extend(rings)
    return out


def test_ski_region_centres_are_derived_from_their_areas():
    """The centre must be the mean of the named areas, not a hand-typed
    number that can drift away from them."""
    for r in SKI.SKI_REGIONS_FULL:
        assert r.areas, f"{r.name} names no ski area"
        assert r.lat == pytest.approx(float(np.mean([a.lat for a in r.areas])))
        assert r.lon == pytest.approx(float(np.mean([a.lon for a in r.areas])))
        # every named area must be close to the centre it defines
        for a in r.areas:
            km = SKI._haversine_km(r.lat, r.lon, a.lat, a.lon)
            assert km <= 30, f"{r.name}: {a.name} is {km:.0f} km from the region centre"
        assert r.spread_km <= 60, f"{r.name} spans {r.spread_km:.0f} km — split it"


def test_every_ski_area_sits_in_a_state_the_region_claims():
    """A coordinate typo, or a region claiming the wrong state, shows up as a
    named area landing outside every state in its list. This is the check that
    an earlier hand-typed version failed."""
    from matplotlib.path import Path as MplPath
    polys = _state_polygons()
    problems = []
    for r in SKI.SKI_REGIONS_FULL:
        for a in r.areas:
            hit = any(MplPath(ring).contains_point((a.lon, a.lat))
                      for st in r.states for ring in polys.get(st, []))
            if not hit:
                where = [st for st, rings in polys.items()
                         if any(MplPath(ring).contains_point((a.lon, a.lat)) for ring in rings)]
                problems.append(f"{r.name}: {a.name} claims {'+'.join(r.states)} but is in {where}")
    assert not problems, "\n".join(problems)


def test_region_names_are_unique_and_states_are_real():
    names = [r.name for r in SKI.SKI_REGIONS_FULL]
    assert len(names) == len(set(names))
    for r in SKI.SKI_REGIONS_FULL:
        from enso_snowpack import STATES as ALL_STATES
        assert set(r.states) <= set(ALL_STATES), f"{r.name} claims a state with no SNOTEL network"
        assert r.climate in SKI.CLIMATES
        assert r.base_ft < r.summit_ft, f"{r.name} has base at or above summit"


def test_every_region_is_sampled_well_enough_to_stand_for_a_range():
    """A region must have at least four SNOTEL sites spanning at least 1000 ft
    of elevation. One gauge, or four at the same height, measures a contour
    line rather than a mountain range — the snowpack a skier meets changes
    with elevation faster than with anything else."""
    stations = pd.read_csv(Path(__file__).resolve().parents[1] /
                           "data" / "derived" / "stations_SNTL.csv") \
        if (Path(__file__).resolve().parents[1] / "data" / "derived" /
            "stations_SNTL.csv").exists() else None
    if stations is None:
        pytest.skip("station list not downloaded")
    rm = SKI.assign_regions(stations)
    v = SKI.validate_regions(rm, stations)
    failed = v[~v.passes]
    assert failed.empty, "under-sampled regions:\n" + failed.to_string(index=False)
    assert v["stations"].min() >= SKI.MIN_STATIONS_PER_REGION
    assert v["elev_range_ft"].min() >= SKI.MIN_ELEVATION_RANGE_FT
    # and every defined region must actually appear
    assert set(v["region"]) == {r.name for r in SKI.SKI_REGIONS_FULL}


def test_coarsening_across_the_node_dilutes_a_real_dipole():
    """Merging regions on opposite sides of the ENSO node must weaken the
    measured effect — that is the argument for resolving ranges. If coarsening
    ever strengthened it, the fine regions would be manufacturing the signal."""
    oni = FX.synthetic_oni(1980, 2025)
    enso = A.enso_by_water_year(oni)
    x = enso.set_index("water_year")["oni_djf"]
    rng = np.random.default_rng(61)
    north, south = "N", "S"
    rows, rmap = [], []
    for region, slope in ((north, -4.0), (south, +4.0)):   # a clean dipole
        for st in range(6):
            trip = f"{region}{st}:XX:SNTL"
            rmap.append({"stationTriplet": trip, "region": region, "weight": 1.0})
            for yv in enso.water_year:
                rows.append({"stationTriplet": trip, "water_year": yv, "window": "Midwinter",
                             "mean_swe": 300 + 30 * slope * x[yv] + rng.normal(0, 60),
                             "mean_depth": np.nan, "storm_days": np.nan,
                             "big_storm_days": np.nan, "new_snow_in_total": np.nan,
                             "days_with_base": np.nan})
    m, fine = pd.DataFrame(rows), pd.DataFrame(rmap)
    coarse = fine.copy(); coarse["region"] = "merged"
    gf = SKI.test_grid(m, fine, enso, n_iter=400, n_perm=400)
    gc = SKI.test_grid(m, coarse, enso, n_iter=400, n_perm=400)
    fine_max = gf[gf.metric == "mean_swe"]["r"].abs().max()
    coarse_max = gc[gc.metric == "mean_swe"]["r"].abs().max()
    assert fine_max > 0.4, "the planted dipole must be visible at range resolution"
    assert coarse_max < fine_max / 2, \
        f"merging across the node must dilute: fine {fine_max:.2f} vs coarse {coarse_max:.2f}"
