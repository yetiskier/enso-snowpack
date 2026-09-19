"""Synthetic stand-in for the real downloads, used by the tests and by
``--fixture`` runs so the whole pipeline can be exercised offline.

The synthetic world has a KNOWN signal — El Niño winters are snow-poor in
the north (MT, ID) and snow-rich in the south (CO, UT), scaled by the ONI —
so a correct pipeline must recover a negative r in the north and a positive
r in the south. Nothing here resembles real measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import STATES
from .sources import ONI_SEASONS

# z per °C of ONI. Northern and maritime states lean negative, southern
# positive, matching the published dipole so the pipeline has a known answer.
FIXTURE_SIGNAL = {"AK": -0.2, "WA": -0.45, "OR": -0.4, "CA": +0.2, "NV": +0.15,
                  "ID": -0.4, "MT": -0.5, "WY": -0.1, "UT": +0.45, "CO": +0.3,
                  "AZ": +0.4, "NM": +0.35, "SD": -0.15}


def synthetic_oni(y0: int = 1950, y1: int = 2025, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = (y1 - y0 + 1) * 12
    # AR(1)-ish red noise with ENSO-like amplitude, smoothed to 3-month means
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.93 * x[i - 1] + rng.normal(0, 0.28)
    x = np.convolve(x, np.ones(3) / 3, mode="same")
    rows = []
    for i in range(n):
        yr, m = y0 + i // 12, i % 12
        rows.append((ONI_SEASONS[m], yr, 26.5 + x[i], round(float(x[i]), 2), m + 1))
    return pd.DataFrame(rows, columns=["season", "year", "total", "oni", "center_month"])


def synthetic_stations(n_per_state: int = 12, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boxes = {"AK": (60.0, 63.0, -150.5, -145.0), "WA": (45.6, 49.0, -122.5, -117.5),
             "OR": (42.0, 46.0, -122.5, -117.5), "CA": (36.5, 41.9, -122.5, -118.5),
             "NV": (36.0, 41.9, -120.0, -114.5), "ID": (42.2, 48.5, -116.8, -111.2),
             "MT": (44.5, 48.9, -115.5, -109.0), "WY": (41.2, 44.9, -110.9, -104.5),
             "UT": (37.2, 41.8, -113.5, -109.2), "CO": (37.2, 40.9, -108.8, -105.0),
             "AZ": (33.5, 36.5, -112.5, -109.0), "NM": (33.0, 37.0, -108.5, -105.0),
             "SD": (43.5, 44.8, -104.5, -103.2)}
    rows = []
    k = 300
    for st in STATES:
        la0, la1, lo0, lo1 = boxes[st]
        for i in range(n_per_state):
            k += 1
            rows.append({"stationTriplet": f"{k}:{st}:SNTL", "stationId": k, "stateCode": st,
                         "networkCode": "SNTL", "name": f"Synthetic {st} {i+1}",
                         "countyName": "Test", "huc": "10000000",
                         "elevation": float(rng.uniform(5500, 10500)),
                         "latitude": float(rng.uniform(la0, la1)),
                         "longitude": float(rng.uniform(lo0, lo1)),
                         "beginDate": f"{int(rng.integers(1979, 1995))}-10-01 00:00",
                         "endDate": "2100-01-01 00:00"})
    return pd.DataFrame(rows)


def synthetic_daily(stations: pd.DataFrame, oni: pd.DataFrame, end_wy: int = 2025,
                    seed: int = 3) -> pd.DataFrame:
    """Daily WTEQ/SNWD/PREC/TAVG (inches, inches, inches, °F — as AWDB stores them)."""
    rng = np.random.default_rng(seed)
    djf = oni[oni["season"] == "DJF"].set_index("year")["oni"]
    frames = []
    for st in stations.itertuples():
        y0 = int(str(st.beginDate)[:4]) + 1
        mean_peak = rng.uniform(8, 30)          # inches
        sd = 0.28 * mean_peak
        for wy in range(y0, end_wy + 1):
            z = FIXTURE_SIGNAL[st.stateCode] * djf.get(wy, 0.0) + rng.normal(0, 1)
            peak = max(0.5, mean_peak + sd * z)
            days = pd.date_range(f"{wy-1}-10-01", f"{wy}-09-30", freq="D")
            t = np.arange(len(days))
            accum = np.clip((t - 30) / 190, 0, 1)                 # builds Nov->mid-Apr
            melt = np.clip((t - 210) / 60, 0, 1)                  # gone by ~mid-June
            swe = peak * accum * (1 - melt)
            prec = peak * 1.6 * np.clip(t / 365, 0, 1)
            tavg = 25 + 30 * np.sin((t - 100) / 365 * 2 * np.pi) + rng.normal(0, 2, len(days))
            frames.append(pd.DataFrame({"stationTriplet": st.stationTriplet, "element": "WTEQ",
                                        "date": days, "value": np.round(swe, 1), "unit": "in"}))
            depth = swe / np.clip(0.22 + 0.25 * accum, 0.2, 0.5)     # density 0.22 -> 0.47
            frames.append(pd.DataFrame({"stationTriplet": st.stationTriplet, "element": "SNWD",
                                        "date": days, "value": np.round(depth, 0), "unit": "in"}))
            frames.append(pd.DataFrame({"stationTriplet": st.stationTriplet, "element": "PREC",
                                        "date": days, "value": np.round(prec, 1), "unit": "in"}))
            frames.append(pd.DataFrame({"stationTriplet": st.stationTriplet, "element": "TAVG",
                                        "date": days, "value": np.round(tavg, 1), "unit": "degF"}))
    return pd.concat(frames, ignore_index=True)


def synthetic_nclimdiv(oni: pd.DataFrame, y0: int = 1950, y1: int = 2025, seed: int = 4) -> pd.DataFrame:
    """Statewide monthly pcpn (in) and tavg (°F) with the same sign structure."""
    rng = np.random.default_rng(seed)
    djf = oni[oni["season"] == "DJF"].set_index("year")["oni"]
    rows = []
    for st in STATES:
        for yr in range(y0, y1 + 1):
            for m in range(1, 13):
                wy = yr + 1 if m >= 10 else yr
                sig = FIXTURE_SIGNAL[st] * djf.get(wy, 0.0) if m in (11, 12, 1, 2, 3) else 0.0
                p = max(0.05, 1.5 + 0.4 * sig + rng.normal(0, 0.4))
                t = 45 + 25 * np.sin((m - 4) / 12 * 2 * np.pi) + 1.5 * djf.get(wy, 0.0) + rng.normal(0, 2)
                rows.append((st, 0, "pcpn", yr, m, round(p, 2)))
                rows.append((st, 0, "tavg", yr, m, round(t, 1)))
    return pd.DataFrame(rows, columns=["state", "division", "element", "year", "month", "value"])
