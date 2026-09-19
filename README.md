# El Niño vs. snowpack — MT, ID, CO, UT

Does El Niño strength correlate with winter snowpack in Montana, Idaho,
Colorado and Utah? This directory is a self-contained, reproducible pipeline
that downloads every public input, computes the correlations, and writes a
report with figures.

## Run it

```
cd enso-snowpack
./run.sh                 # downloads (~30–60 min first time, cached after), then analyses
```

or step by step:

```
python3 -m pip install -r requirements.txt
python3 -m enso_snowpack fetch            # data/raw/ + data/derived/, resumable
python3 -m enso_snowpack analyze          # results/summary.md + figures + CSV tables
python3 -m enso_snowpack analyze --fixture   # offline smoke run on synthetic data
python3 -m pytest tests -q                # offline tests
```

Useful flags: `--max-stations 40` for a quick real-data smoke run,
`--no-snow-courses` to skip the manual snow-course network, `--force` to
re-download, `--n-boot 500` for a faster bootstrap.

**Network note.** The Claude Code cloud session that wrote this could not
reach `cpc.ncep.noaa.gov`, `psl.noaa.gov`, `ncei.noaa.gov` or
`wcc.sc.egov.usda.gov` (egress policy), so the pipeline has only been run on
the synthetic fixture. The first real run is on Joel's Linux box; if the AWDB
API rejects a parameter, the failure lands in `data/raw/failed_SNTL.txt` and
the log, and the parser functions in `enso_snowpack/sources.py` are the place
to adjust.

## Data

| Source | What | Period | URL |
|---|---|---|---|
| NOAA CPC ONI | 3-month running Niño-3.4 SST anomaly, the official ENSO index | 1950– | cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (fallback psl.noaa.gov/data/correlation/oni.data) |
| NOAA PSL MEI v2 | Multivariate ENSO Index, sensitivity check | 1979– | psl.noaa.gov/enso/mei/data/meiv2.data |
| NRCS SNOTEL (AWDB REST v1) | daily SWE (`WTEQ`), accumulated precipitation (`PREC`), mean air temperature (`TAVG`) for every SNOTEL station in the four states | ~1979– | wcc.sc.egov.usda.gov/awdbRestApi |
| NRCS snow courses (AWDB, network `SNOW`) | manual April-1 SWE, extends the record before SNOTEL | ~1930s– | same |
| NCEI nClimDiv | statewide monthly precipitation and mean temperature | 1895– | ncei.noaa.gov/pub/data/cirs/climdiv/ |

Raw downloads are cached under `data/raw/` (one CSV per station, so an
interrupted fetch resumes); tidy inputs under `data/derived/`; everything
reported under `results/`. `data/` is git-ignored, `results/` is meant to be
committed once a real run exists.

## Method

1. **Water year** N = 1 Oct N−1 … 30 Sep N. Snowpack metrics per station and
   water year: April-1 SWE (nearest value within ±3 days), peak SWE and its
   date, Oct–Mar precipitation, DJF mean temperature.
2. **ENSO** per water year: DJF ONI (Dec N−1 … Feb N), winter mean
   (NDJ/DJF/JFM), and the winter peak |ONI| (OND…FMA) for strength bins.
   Phases use the CPC ±0.5 °C thresholds; strength bins weak 0.5–0.9,
   moderate 1.0–1.4, strong 1.5–1.9, very strong ≥ 2.0.
3. **Standardise per station**: z-score of the linearly detrended series
   (so long-term trends cannot pose as ENSO signal) — plus percent-of-median
   as a sensitivity. Stations need ≥ 15 valid years and a median April-1
   SWE ≥ 50 mm.
4. **Aggregate**: mean station z per state and water year (≥ 5 stations),
   plus North (MT+ID), South (CO+UT), ALL.
5. **Statistics** per region: Pearson r with a bootstrap 95 % CI, Spearman ρ,
   regression slope per °C of ONI, composite means by phase with a Welch
   t-test (El Niño vs the rest), composites by strength bin, and — the direct
   test of *strength* — r within El Niño winters only. Per-station r is
   mapped so the latitude structure is visible rather than averaged away.
6. **Weather cross-check**: nClimDiv statewide Nov–Mar precipitation (% of
   1991–2020 normal) and DJF temperature anomaly against the same index over
   70+ winters; SNOTEL Oct–Mar precipitation the same way.

## What to expect (from the literature, to be confirmed by the run)

The published ENSO–snowpack relation in the US Rockies is a **latitudinal
dipole**: La Niña winters tend to be snow-rich in the northern Rockies
(MT, ID) and El Niño winters snow-poor there, with the reverse and a weaker
signal in the southern Rockies/Great Basin (southern UT, CO). Colorado sits
near the node and is the least predictable. Correlations in the north are
typically |r| ≈ 0.3–0.5; the relation is not strictly monotonic in strength
— the 2015-16 "Godzilla" El Niño was not a dry year in Montana — which is
exactly what the within-El-Niño rows and the strength composites are there to
test. Treat these as hypotheses; the report's numbers are the finding.

## Layout

```
enso_snowpack/
  sources.py   parsers for each raw format (pure functions, tested offline)
  fetch.py     downloads + per-station cache
  analysis.py  water-year metrics, standardisation, statistics
  figures.py   matplotlib figures
  report.py    results/summary.md
  fixture.py   synthetic data with a planted north-negative / south-positive signal
  cli.py       fetch | analyze | run
tests/         parser tests on spec-shaped samples + end-to-end fixture test
```
