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

**Status.** The code was developed in a sandbox without access to the NOAA
and NRCS servers, so it has been exercised only on the synthetic fixture and
the offline tests. Once a real run exists its `results/` are committed here.
If the AWDB API rejects a parameter, the failure lands in
`data/raw/failed_SNTL.txt` and the log; the parsers in
`enso_snowpack/sources.py` are the place to adjust.

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
7. **Bootstrap / permutation significance** (`enso_snowpack/bootstrap.py`),
   because ~40 winters and ~15 El Niño winters are too few for parametric
   p-values on skewed, serially correlated series. See below.

## Bootstrap

All resampling treats the **water year** as the exchangeable unit; stations
are never resampled as if independent (stations within a state co-vary, and
resampling them would inflate the effective sample size).

| scheme | what it does | gives |
|---|---|---|
| `permutation` | shuffles the ENSO index across years, snowpack untouched | two-sided p for r, and for the El Niño-minus-rest composite difference |
| `block_bootstrap` (default) | moving-block bootstrap of (ONI, snowpack) year pairs, block length 2 (ENSO's persistence scale) | 95 % CI on r, slope, composite difference that respects serial dependence |
| `two_level` | resamples years, then the contributing stations within each year | CI that also carries station-sampling uncertainty of the state mean |
| `brown_harper_2026` | the modified bootstrap of Brown & Harper (2026) | **not yet implemented** — see below |

The per-station map uses per-station permutation p-values with
Benjamini–Hochberg false-discovery-rate control (α_FDR = 0.10), the
field-significance procedure recommended by Wilks (2016).

**Brown & Harper (2026).** J. Brown and J. Harper, *Historical evolution of
snowpack capacity to buffer rain-on-snow runoff in a large Columbia River
headwaters basin*, Hydrol. Earth Syst. Sci. 30, 5735–5748, 2026,
doi:10.5194/hess-30-5735-2026 (preprint egusphere-2025-4971). Its modified
bootstrap is the intended primary significance test. The authoring sandbox
could not fetch the paper, so `bootstrap.brown_harper_2026` is a stub that
raises `NotImplementedError`; transcribe the resampling scheme from the
Methods into that function and register it in `run_bootstrap`. Run with
`python3 -m enso_snowpack analyze --bootstrap brown_harper_2026` once done.

Select a scheme with `--bootstrap <scheme>`; `--n-boot` and `--n-perm` set
the resample counts (5000 each by default).

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
  bootstrap.py permutation, moving-block and two-level bootstraps; FDR field significance
  fixture.py   synthetic data with a planted north-negative / south-positive signal
  cli.py       fetch | analyze | run
tests/         parser tests on spec-shaped samples + end-to-end fixture test
```

## Citation and licence

MIT licence (see `LICENSE`). If you use the results, cite the data providers
(NOAA CPC, NRCS AWDB, NCEI nClimDiv) and Brown & Harper (2026) for the
bootstrap.
