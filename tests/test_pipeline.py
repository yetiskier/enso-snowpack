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

# statewide: SSS EE YYYY; divisional: SS DD EE YYYY
NCLIMDIV_SAMPLE = """024011950   0.85   0.60   1.10   1.50   2.40   2.90   1.20   1.10   1.40   0.90   0.70   0.80
024021950  15.20  20.10  30.00  42.00  52.00  60.00  68.00  66.00  55.00  44.00  30.00  20.00
005011950   0.50 -9.99   1.00   1.60   2.10   1.20   1.90   1.80   1.20   1.10   0.60   0.50
048011950   0.50   0.40   1.00   1.60   2.10   1.20   1.90   1.80   1.20   1.10   0.60   0.50
2401011950   0.85   0.60   1.10   1.50   2.40   2.90   1.20   1.10   1.40   0.90   0.70   0.80
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
    df = parse_nclimdiv(NCLIMDIV_SAMPLE, ["MT", "CO", "UT", "ID"])
    assert set(df.state) == {"MT", "CO"}          # WY (48) filtered out
    st = df[(df.state == "MT") & (df.division == 0)]
    assert set(st.element) == {"pcpn", "tavg"}
    assert len(df[(df.state == "CO") & (df.element == "pcpn")]) == 11   # one -9.99 dropped
    dv = df[df.division == 1]
    assert len(dv) == 12 and dv.state.iloc[0] == "MT"


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


def test_pipeline_recovers_planted_dipole(tmp_path: Path):
    rc = main(["--root", str(tmp_path), "analyze", "--fixture", "--n-boot", "300", "--n-perm", "300"])
    assert rc == 0
    corr = pd.read_csv(tmp_path / "results" / "corr_apr1_zd.csv").set_index("region")
    # planted: north dry in El Niño (negative r), south wet (positive r)
    for region in ("MT", "ID", "North (MT+ID)"):
        assert corr.loc[region, "pearson_r"] < -0.5 and corr.loc[region, "pearson_p"] < 0.01
    for region in ("CO", "UT", "South (CO+UT)"):
        assert corr.loc[region, "pearson_r"] > 0.4 and corr.loc[region, "pearson_p"] < 0.01
    sc = pd.read_csv(tmp_path / "results" / "station_corr_apr1.csv")
    assert (sc[sc.stateCode == "MT"].r < 0).mean() > 0.8
    assert (sc[sc.stateCode == "UT"].r > 0).mean() > 0.8
    assert (tmp_path / "results" / "summary.md").exists()
    for i in range(1, 8):
        assert list((tmp_path / "results").glob(f"fig{i}_*.png")), f"figure {i} missing"
    nc = pd.read_csv(tmp_path / "results" / "corr_nclimdiv_precip.csv").set_index("region")
    assert nc.loc["MT", "pearson_r"] < 0 and nc.loc["UT", "pearson_r"] > 0
    bh = pd.read_csv(tmp_path / "results" / "bootstrap_apr1_brown_harper_2026.csv").set_index("region")
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
