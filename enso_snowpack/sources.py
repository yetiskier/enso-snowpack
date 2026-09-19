"""Parsers for the raw text/JSON formats of each data source.

Every parser is a pure function of the downloaded text so it can be tested
offline. Nothing here touches the network.
"""
from __future__ import annotations

import io
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- ONI ------

ONI_SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
               "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
# Centre month of each overlapping 3-month season (1 = Jan).
ONI_SEASON_CENTER = {s: i + 1 for i, s in enumerate(ONI_SEASONS)}


def parse_oni_ascii(text: str) -> pd.DataFrame:
    """Parse CPC ``oni.ascii.txt``.

    Format::

        SEAS  YR   TOTAL   ANOM
        DJF  1950  24.72  -1.53
        JFM  1950  25.17  -1.34

    ``YR`` for DJF is the year of the Jan/Feb, i.e. the *water year*.
    Returns columns ``season, year, total, oni`` plus ``center_month``.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] not in ONI_SEASON_CENTER:
            continue
        try:
            rows.append((parts[0], int(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    if not rows:
        raise ValueError("no ONI rows parsed — file format changed?")
    df = pd.DataFrame(rows, columns=["season", "year", "total", "oni"])
    df["center_month"] = df["season"].map(ONI_SEASON_CENTER)
    return df.sort_values(["year", "center_month"]).reset_index(drop=True)


def parse_oni_psl(text: str) -> pd.DataFrame:
    """Parse PSL ``oni.data`` (fallback source).

    Format: first line ``1950 2025`` (year range), then one row per year with
    12 values — each is the 3-month running mean centred on that month
    (Jan = DJF, Feb = JFM, ...). Missing = -99.9. Trailing lines are
    metadata. Output matches :func:`parse_oni_ascii` minus ``total``.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    first = lines[0].split()
    y0, y1 = int(first[0]), int(first[1])
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0])
        except ValueError:
            continue
        if not (y0 <= yr <= y1):
            continue
        for m, v in enumerate(parts[1:], start=1):
            val = float(v)
            if val <= -99:
                continue
            rows.append((ONI_SEASONS[m - 1], yr, np.nan, val, m))
    if not rows:
        raise ValueError("no ONI rows parsed from PSL file")
    return pd.DataFrame(rows, columns=["season", "year", "total", "oni", "center_month"])


def parse_mei_v2(text: str) -> pd.DataFrame:
    """Parse PSL ``meiv2.data``: ``1979 2025`` header then ``year v1..v12``
    (bimonthly DJ, JF, ..., ND), missing = -999. Returns ``year, bimonth, mei``
    where ``bimonth`` 1 = DJ (Dec/Jan), 2 = JF, ... 12 = ND.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    y0, y1 = (int(x) for x in lines[0].split()[:2])
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0])
        except ValueError:
            continue
        if not (y0 <= yr <= y1):
            continue
        for b, v in enumerate(parts[1:], start=1):
            val = float(v)
            if val <= -990:
                continue
            rows.append((yr, b, val))
    return pd.DataFrame(rows, columns=["year", "bimonth", "mei"])


# ------------------------------------------------------------ nClimDiv -----

# NCEI nClimDiv state codes (alphabetical order of the 48 contiguous states).
# NCEI nClimDiv statewide codes (Alaska has no nClimDiv division record).
NCLIMDIV_STATE_CODES = {"AZ": 2, "CA": 4, "CO": 5, "ID": 10, "MT": 24, "NV": 26,
                        "NM": 29, "OR": 35, "SD": 39, "UT": 42, "WA": 45, "WY": 48}
NCLIMDIV_ELEMENTS = {"01": "pcpn", "02": "tavg", "27": "tmax", "28": "tmin", "05": "pdsi"}
NCLIMDIV_MISSING = {"pcpn": -9.99, "tavg": -99.9, "tmax": -99.9, "tmin": -99.9, "pdsi": -99.99}


def parse_nclimdiv(text: str, states: Iterable[str] | None = None,
                   layout: str = "statewide") -> pd.DataFrame:
    """Parse an nClimDiv fixed-width file (statewide or divisional).

    Both layouts use a 10-character id followed by 12 monthly values, so the
    layout CANNOT be inferred from its length — the caller says which file
    this is:

    * ``statewide`` (``climdiv-pcpnst`` / ``climdiv-tmpcst``):
      ``SSS`` 3-digit state, ``D`` division (always 0), ``EE`` element,
      ``YYYY`` year.
    * ``divisional`` (``climdiv-pcpndv`` / ``climdiv-tmpcdv``):
      ``SS`` 2-digit state, ``DD`` division, ``EE`` element, ``YYYY`` year.

    Element codes differ per file: the statewide precipitation file uses 01
    and the statewide temperature file 02. Returns long-form ``state,
    division, element, year, month, value`` with missing values dropped;
    ``division`` is 0 for statewide rows.
    """
    if layout not in ("statewide", "divisional"):
        raise ValueError(f"layout must be 'statewide' or 'divisional', not {layout!r}")
    want = None
    if states is not None:
        # Alaska has no nClimDiv statewide record; skip silently rather than fail.
        want = {NCLIMDIV_STATE_CODES[s] for s in states if s in NCLIMDIV_STATE_CODES}
    code_to_state = {v: k for k, v in NCLIMDIV_STATE_CODES.items()}
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        ident = parts[0]
        if len(ident) != 10 or not ident.isdigit():
            continue
        if layout == "statewide":
            st, div = int(ident[0:3]), int(ident[3:4])
        else:
            st, div = int(ident[0:2]), int(ident[2:4])
        el, yr = ident[4:6], int(ident[6:10])
        if want is not None and st not in want:
            continue
        elem = NCLIMDIV_ELEMENTS.get(el)
        if elem is None:
            continue
        miss = NCLIMDIV_MISSING[elem]
        for m, v in enumerate(parts[1:], start=1):
            val = float(v)
            if abs(val - miss) < 1e-6:
                continue
            rows.append((code_to_state.get(st, str(st)), div, elem, yr, m, val))
    return pd.DataFrame(rows, columns=["state", "division", "element", "year", "month", "value"])


def find_nclimdiv_files(index_html: str) -> dict[str, str]:
    """Return ``{'pcpnst': 'climdiv-pcpnst-v1.0.0-20260905', ...}`` from the
    NCEI directory listing, picking the newest date per product."""
    found: dict[str, tuple[str, str]] = {}
    for m in re.finditer(r"climdiv-(\w+)-v([\d.]+)-(\d{8})", index_html):
        product, ver, date = m.group(1), m.group(2), m.group(3)
        name = f"climdiv-{product}-v{ver}-{date}"
        if product not in found or date > found[product][0]:
            found[product] = (date, name)
    return {k: v[1] for k, v in found.items()}


# ----------------------------------------------------------- AWDB REST -----

def parse_awdb_stations(payload: str | list) -> pd.DataFrame:
    """Normalise the AWDB ``/stations`` JSON into a DataFrame."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    cols = ["stationTriplet", "stationId", "stateCode", "networkCode", "name",
            "countyName", "huc", "elevation", "latitude", "longitude",
            "beginDate", "endDate"]
    rows = [{c: rec.get(c) for c in cols} for rec in data]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("elevation", "latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_awdb_data(payload: str | list) -> pd.DataFrame:
    """Normalise the AWDB ``/data`` JSON into long form
    ``stationTriplet, element, date, value, unit``.

    The response is a list of ``{stationTriplet, data: [{stationElement:
    {elementCode, storedUnitCode, ...}, values: [{date, value}, ...]}]}``.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    frames = []
    for rec in data:
        trip = rec.get("stationTriplet")
        for block in rec.get("data", []) or []:
            se = block.get("stationElement", {}) or {}
            elem = se.get("elementCode")
            unit = se.get("storedUnitCode")
            vals = block.get("values", []) or []
            if not vals:
                continue
            df = pd.DataFrame(vals)
            if "value" not in df:
                continue
            if "date" not in df:
                df["date"] = _derive_date(df)
            df = df[["date", "value"]].copy()
            df["stationTriplet"] = trip
            df["element"] = elem
            df["unit"] = unit
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stationTriplet", "element", "date", "value", "unit"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"].astype(str).str.slice(0, 10), errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    return out[["stationTriplet", "element", "date", "value", "unit"]].reset_index(drop=True)


def _derive_date(df: pd.DataFrame) -> pd.Series:
    """Build a date for AWDB records that carry no ``date`` key.

    DAILY records have ``date``. SEMIMONTHLY / MONTHLY records (snow courses,
    and any monthly element) instead carry ``collectionDate`` — the actual day
    the course was measured — plus ``year``/``month`` and, semimonthly, a
    ``monthPart`` of "1" (mid-month) or "2" (end of month). The real
    collection date is preferred because a snow course read on 27 January is
    the February-1 measurement, and the analysis matches on proximity to the
    target day.
    """
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if "collectionDate" in df:
        out = pd.to_datetime(df["collectionDate"].astype(str).str.slice(0, 10), errors="coerce")
    if out.isna().any() and {"year", "month"} <= set(df.columns):
        y = pd.to_numeric(df["year"], errors="coerce")
        m = pd.to_numeric(df["month"], errors="coerce")
        part = pd.to_numeric(df["monthPart"], errors="coerce") if "monthPart" in df else pd.Series(1, index=df.index)
        # part 1 -> mid-month (15th), part 2 -> end of month (28th, safe in Feb)
        day = pd.Series(np.where(part.fillna(1).to_numpy() >= 2, 28, 15), index=df.index)
        fallback = pd.to_datetime(dict(year=y, month=m, day=day), errors="coerce")
        out = out.fillna(fallback)
    return out


def to_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert AWDB values in place-ish: inches → mm, °F → °C, feet elevation
    is handled in the station table separately. Unknown units pass through."""
    df = df.copy()
    inch = df["unit"].astype(str).str.lower().isin(["in", "inch", "inches"])
    df.loc[inch, "value"] = df.loc[inch, "value"] * 25.4
    df.loc[inch, "unit"] = "mm"
    degf = df["unit"].astype(str).str.lower().isin(["degf", "f", "°f"])
    df.loc[degf, "value"] = (df.loc[degf, "value"] - 32.0) * 5.0 / 9.0
    df.loc[degf, "unit"] = "degC"
    return df
