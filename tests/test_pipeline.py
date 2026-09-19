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
    rc = main(["--root", str(tmp_path), "analyze", "--fixture", "--n-boot", "100"])
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
