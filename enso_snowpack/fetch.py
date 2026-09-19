"""Download and cache the raw inputs.

Everything lands under ``data/raw/`` and is never re-downloaded unless
``--force`` is given, so a run that is interrupted resumes where it stopped.

Sources
-------
* ONI:      https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
            (fallback https://psl.noaa.gov/data/correlation/oni.data)
* MEI v2:   https://psl.noaa.gov/enso/mei/data/meiv2.data   (optional)
* SNOTEL /  https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/
  snow courses  (``/stations`` metadata, ``/data`` daily WTEQ/PREC/TAVG,
            semimonthly WTEQ for snow courses)
* nClimDiv: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import STATES
from .sources import (find_nclimdiv_files, parse_awdb_data, parse_awdb_stations,
                      parse_mei_v2, parse_nclimdiv, parse_oni_ascii, parse_oni_psl)

log = logging.getLogger("enso_snowpack.fetch")

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_FALLBACK_URL = "https://psl.noaa.gov/data/correlation/oni.data"
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
NCLIMDIV_BASE = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"

SNOTEL_ELEMENTS = ["WTEQ", "PREC", "TAVG"]
USER_AGENT = "enso-snowpack/0.1 (research; contact via repository)"


class FetchError(RuntimeError):
    pass


def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 120) -> str:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break
        except requests.RequestException as exc:  # network hiccup
            last = repr(exc)
        time.sleep(2 ** attempt)
    raise FetchError(f"GET {url} failed: {last}")


def _cached_text(path: Path, url: str, force: bool = False, params: dict | None = None) -> str:
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    text = _get(url, params=params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- ENSO -----

def fetch_oni(raw_dir: Path, force: bool = False) -> pd.DataFrame:
    try:
        text = _cached_text(raw_dir / "oni.ascii.txt", ONI_URL, force)
        return parse_oni_ascii(text)
    except FetchError as exc:
        log.warning("CPC ONI unavailable (%s); trying PSL mirror", exc)
        text = _cached_text(raw_dir / "oni.data", ONI_FALLBACK_URL, force)
        return parse_oni_psl(text)


def fetch_mei(raw_dir: Path, force: bool = False) -> pd.DataFrame | None:
    try:
        text = _cached_text(raw_dir / "meiv2.data", MEI_URL, force)
        return parse_mei_v2(text)
    except FetchError as exc:
        log.warning("MEI v2 unavailable (%s); continuing without it", exc)
        return None


# ------------------------------------------------------------- nClimDiv -----

def fetch_nclimdiv(raw_dir: Path, force: bool = False,
                   products=("pcpnst", "tmpcst")) -> pd.DataFrame:
    """Statewide monthly precipitation (in) and mean temperature (°F)."""
    index = _cached_text(raw_dir / "nclimdiv_index.html", NCLIMDIV_BASE, force)
    names = find_nclimdiv_files(index)
    frames = []
    for prod in products:
        if prod not in names:
            raise FetchError(f"nClimDiv product {prod!r} not in directory listing")
        text = _cached_text(raw_dir / names[prod], NCLIMDIV_BASE + names[prod], force)
        frames.append(parse_nclimdiv(text, STATES))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- AWDB -----

def fetch_stations(raw_dir: Path, network: str = "SNTL", force: bool = False) -> pd.DataFrame:
    path = raw_dir / f"stations_{network}.json"
    if path.exists() and not force:
        return parse_awdb_stations(path.read_text(encoding="utf-8"))
    triplets = ",".join(f"*:{s}:{network}" for s in STATES)
    text = _get(f"{AWDB_BASE}/stations", params={
        "stationTriplets": triplets,
        "activeOnly": "false",
        "returnForecastPointMetadata": "false",
        "returnReservoirMetadata": "false",
        "returnStationElements": "false",
    })
    json.loads(text)  # validate before caching
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return parse_awdb_stations(text)


def _station_cache_path(raw_dir: Path, triplet: str) -> Path:
    return raw_dir / "awdb" / (triplet.replace(":", "_") + ".csv")


def fetch_station_series(raw_dir: Path, triplet: str, elements, duration: str,
                         begin: str, end: str, force: bool = False,
                         pause: float = 0.2) -> pd.DataFrame:
    """Full-period series for one station, cached as CSV."""
    path = _station_cache_path(raw_dir, triplet)
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    text = _get(f"{AWDB_BASE}/data", params={
        "stationTriplets": triplet,
        "elements": ",".join(elements),
        "duration": duration,
        "beginDate": begin,
        "endDate": end,
        "periodRef": "END",
        "centralTendencyType": "NONE",
        "returnFlags": "false",
        "returnOriginalValues": "false",
        "returnSuspectData": "false",
    })
    df = parse_awdb_data(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(pause)
    return df


def fetch_all_stations(raw_dir: Path, stations: pd.DataFrame, network: str,
                       end: str, force: bool = False, max_stations: int | None = None,
                       progress: bool = True) -> pd.DataFrame:
    """Loop over stations, tolerate individual failures, return long-form data."""
    if network == "SNTL":
        elements, duration = SNOTEL_ELEMENTS, "DAILY"
    else:  # snow courses: manual measurements near the 1st of the month
        elements, duration = ["WTEQ"], "SEMIMONTHLY"
    rows = stations.itertuples()
    frames, failed = [], []
    n = len(stations) if max_stations is None else min(max_stations, len(stations))
    for i, st in enumerate(rows):
        if max_stations is not None and i >= max_stations:
            break
        begin = str(st.beginDate)[:10] if isinstance(st.beginDate, str) else "1900-01-01"
        try:
            df = fetch_station_series(raw_dir, st.stationTriplet, elements, duration,
                                      begin, end, force)
            frames.append(df)
        except (FetchError, ValueError) as exc:
            failed.append((st.stationTriplet, str(exc)[:120]))
            log.warning("station %s failed: %s", st.stationTriplet, exc)
        if progress and (i + 1) % 25 == 0:
            log.info("%s: %d/%d stations fetched", network, i + 1, n)
    if failed:
        (raw_dir / f"failed_{network}.txt").write_text(
            "\n".join(f"{t}\t{m}" for t, m in failed), encoding="utf-8")
        log.warning("%d %s stations failed — see failed_%s.txt", len(failed), network, network)
    if not frames:
        return pd.DataFrame(columns=["stationTriplet", "element", "date", "value", "unit"])
    return pd.concat(frames, ignore_index=True)
