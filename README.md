# El Niño and the ski season — MT, ID, WY, CO, UT

**Does El Niño affect the ski season, where, and when in the winter?**
Reproducible analysis of NOAA's Oceanic Niño Index against NRCS SNOTEL
snowpack for 13 destination ski regions in the northern and central Rockies.

_Last updated 2026-09-19. Results below are from real data:
534 SNOTEL stations, 24.4 million daily observations, water years 1964–2026,
with 25 El Niño, 27 La Niña and 25 neutral winters._

This is deliberately **not** a water-supply study. April-1 snow-water
equivalent is a runoff metric. A ski operation cares about the base underfoot
and how often it storms during the windows that carry a season, so the
headline analysis scores ski regions inside ski windows.

---

## The answer in four points

**1. The effect is real, and it is strongest at Montana Snowbowl.** Of 260
tests, 20 survive false-discovery control across the whole grid, and 8 of
those are Snowbowl. El Niño explains up to **28 %** of the year-to-year
variance in its spring base.

| Window | Measure | winters | r² | El Niño brings | mean z, El Niño | mean z, La Niña | survives FDR |
|---|---|---|---|---|---|---|---|
| Spring | Base (mean snow depth) | 26 | **29%** | less | -0.63 | +0.45 | yes |
| Spring | Base (mean SWE in window) | 59 | **28%** | less | -0.58 | +0.55 | yes |
| Midwinter | Base (mean snow depth) | 26 | **28%** | less | -0.57 | +0.43 | yes |
| Spring | Days per 30 with a skiable base | 56 | **24%** | less | -0.49 | +0.43 | yes |
| Midwinter | Powder days per 30 (SWE gain >= 25 mm) | 59 | **19%** | less | -0.41 | +0.48 | yes |
| Midwinter | Storm days per 30 (SWE gain >= 10 mm) | 59 | **19%** | less | -0.47 | +0.40 | yes |
| Holidays | Base (mean snow depth) | 26 | **18%** | less | -0.50 | +0.31 | no |
| Midwinter | Base (mean SWE in window) | 59 | **16%** | less | -0.42 | +0.46 | yes |

**2. The holidays are hit everywhere, including Colorado and Utah.** Storm
frequency over Christmas and New Year leans negative in **37 of 39** tests
across all 13 regions. This is invisible in a seasonal average, and it lands
on the highest-revenue window of the year.

| Window of winter | tests | leaning to less snow | mean r | sign-test p |
|---|---|---|---|---|
| Early season | 39 | 27 of 39 (69%) | -0.073 | 0.0237 |
| Holidays | 39 | 37 of 39 (95%) | -0.196 | 0.0000 |
| Midwinter | 39 | 34 of 39 (87%) | -0.160 | 0.0000 |
| Spring | 39 | 23 of 39 (59%) | -0.050 | 0.3368 |

**3. ENSO *strength* tells you almost nothing. Phase is what matters.**
This is the clearest negative result in the study. Knowing only whether a
winter is El Niño, La Niña or neutral explains more variance than knowing the
actual ONI value — adding magnitude makes the prediction *worse* in every
window, and within El Niño winters alone the index explains about 4 %.

| Window | r² from phase alone | r² from the continuous ONI | change from adding strength | tests where strength helps | r² within El Niño winters only |
|---|---|---|---|---|---|
| Early season | 5.6% | 3.3% | **-2.3%** | 6 of 65 | 3.5% |
| Holidays | 7.6% | 5.1% | **-2.5%** | 2 of 65 | 4.2% |
| Midwinter | 7.7% | 6.4% | **-1.3%** | 9 of 65 | 3.9% |
| Spring | 9.5% | 5.8% | **-3.7%** | 4 of 65 | 5.0% |

A "very strong" El Niño is not a worse ski year than a weak one. The three
very strong events (1983, 1998, 2016) were not the worst winters in the
northern Rockies. Plan for the phase; ignore the magnitude.

**4. The south is not a mirror image — it is unpredictable.** The San Juans
and southern Utah show weak positive tendencies that never reach
significance. For water supply there is a clean north-negative /
south-positive dipole; for skiing the honest statement about the southern
Colorado Rockies is "no detectable ENSO signal", not "El Niño is good".

### Every result that survives false-discovery control

| Ski region | Window | Measure | winters | r² | permutation p |
|---|---|---|---|---|---|
| N Idaho (Schweitzer/Silver) | Spring | Base (mean snow depth) | 25 | **36%** | 0.0026 |
| NW Montana (Whitefish) | Spring | Base (mean snow depth) | 25 | **34%** | 0.0032 |
| NW Montana (Whitefish) | Midwinter | Base (mean snow depth) | 25 | **30%** | 0.0040 |
| Montana Snowbowl (Missoula) | Spring | Base (mean snow depth) | 26 | **29%** | 0.0040 |
| Montana Snowbowl (Missoula) | Spring | Base (mean SWE in window) | 59 | **28%** | 0.0002 |
| Montana Snowbowl (Missoula) | Midwinter | Base (mean snow depth) | 26 | **28%** | 0.0062 |
| Montana Snowbowl (Missoula) | Spring | Days per 30 with a skiable base | 56 | **24%** | 0.0002 |
| S Wyoming (Snowy/Sierra Madre) | Holidays | Storm days per 30 (SWE gain >= 10 mm) | 46 | **21%** | 0.0012 |
| N Colorado (Steamboat) | Holidays | Storm days per 30 (SWE gain >= 10 mm) | 47 | **19%** | 0.0018 |
| Montana Snowbowl (Missoula) | Midwinter | Powder days per 30 (SWE gain >= 25 mm) | 59 | **19%** | 0.0004 |
| Montana Snowbowl (Missoula) | Midwinter | Storm days per 30 (SWE gain >= 10 mm) | 59 | **19%** | 0.0010 |
| NW Montana (Whitefish) | Spring | Days per 30 with a skiable base | 50 | **18%** | 0.0020 |
| SW Montana (Big Sky/Bridger) | Spring | Base (mean SWE in window) | 60 | **16%** | 0.0016 |
| NW Montana (Whitefish) | Spring | Base (mean SWE in window) | 52 | **16%** | 0.0042 |
| N Idaho (Schweitzer/Silver) | Spring | Base (mean SWE in window) | 46 | **16%** | 0.0056 |
| Montana Snowbowl (Missoula) | Midwinter | Base (mean SWE in window) | 59 | **16%** | 0.0022 |
| Montana Snowbowl (Missoula) | Midwinter | Days per 30 with a skiable base | 59 | **15%** | 0.0020 |
| SW Montana (Big Sky/Bridger) | Midwinter | Base (mean SWE in window) | 60 | **14%** | 0.0046 |
| SW Montana (Big Sky/Bridger) | Midwinter | Days per 30 with a skiable base | 60 | **13%** | 0.0036 |
| SW Montana (Big Sky/Bridger) | Spring | Days per 30 with a skiable base | 60 | **11%** | 0.0072 |

---

## Figures

| File | What it shows |
|---|---|
| `results/fig9_ski_region_window.png` | The headline: r² by ski region (north to south) x window of winter x measure. |
| `results/fig10_distributions.png` | The probability distributions behind each headline result: the subsampling distribution of the estimate against the permutation null. Separation between them is the significance, shown rather than asserted. |
| `results/fig8_daily_enso_curve.png` | The correlation for every day of the water year, against the snowpack climatology. |
| `results/fig3_station_map_apr1.png` | Per-station correlation across all 534 stations, showing the latitudinal dipole. |
| `results/fig1_oni_timeseries.png` | The ONI record with El Niño and La Niña winters marked. |

**Colour convention on every figure: red means less snow, blue means more.**

---

## Ski regions and elevation weighting

A SNOTEL gauge only represents a ski area to the extent it shares its snow
climate, and elevation is most of that. Each region therefore carries the
served terrain's **base-to-summit band**, and every station is weighted by a
Gaussian on its vertical gap to that band (scale 1,200 ft) times a Gaussian on
horizontal distance (scale 0.6 x radius). A valley gauge 900 m below the lifts
counts for little. "Eff. stations" is Kish's effective sample size of those
weights.

| Ski region | served band (ft) | stations | eff. stations | weighted elevation (ft) | in band |
|---|---|---|---|---|---|
| Montana Snowbowl (Missoula) | 5,000–7,600 | 13 | 10.9 | 5,978 | 10 |
| NW Montana (Whitefish) | 4,464–6,817 | 14 | 12.4 | 5,469 | 12 |
| SW Montana (Big Sky/Bridger) | 6,400–10,000 | 20 | 17.2 | 7,848 | 20 |
| N Idaho (Schweitzer/Silver) | 4,000–6,400 | 14 | 12.3 | 5,238 | 13 |
| C Idaho (Sun Valley) | 5,750–9,150 | 20 | 18.1 | 7,569 | 18 |
| Tetons (Jackson/Targhee) | 6,300–10,450 | 15 | 12.3 | 7,943 | 15 |
| S Wyoming (Snowy/Sierra Madre) | 8,798–9,663 | 21 | 19.1 | 9,324 | 6 |
| Wasatch (Alta/Park City) | 6,800–11,000 | 31 | 27.8 | 8,173 | 29 |
| S Utah (Brian Head) | 9,600–10,970 | 17 | 13.3 | 9,177 | 6 |
| N Colorado (Steamboat) | 6,900–10,568 | 18 | 15.9 | 9,367 | 16 |
| I-70 (Summit/Vail) | 8,120–12,998 | 27 | 24.1 | 10,365 | 27 |
| Elk Mtns (Aspen/Crested Butte) | 7,945–12,162 | 14 | 12.8 | 10,172 | 14 |
| San Juans (Telluride/Wolf Ck) | 8,725–13,150 | 27 | 25.5 | 10,495 | 27 |

The Montana Snowbowl region is anchored on **Stuart Mountain SNOTEL**
(901:MT:SNTL, 7,270 ft, 5.3 km from the ski area), which is both the closest
station and the closest in elevation; it carries the highest weight in the
region, 0.99.

Two regions are poorly served and their results should be read with care:
**S Utah (Brian Head)** and **S Wyoming (Snowy Range)** each have only 6
stations inside the served band, because SNOTEL does not reach those
elevations locally.

---

## Method

1. **Water year** N runs 1 Oct N-1 to 30 Sep N. ENSO phase comes from the DJF
   ONI (El Niño at or above +0.5 °C, La Niña at or below -0.5 °C); strength
   bins follow CPC (weak 0.5–0.9, moderate 1.0–1.4, strong 1.5–1.9, very
   strong 2.0 and above) on the winter peak.
2. **Ski windows**: Early season (1 Nov–15 Dec), Holidays (16 Dec–5 Jan),
   Midwinter (6 Jan–28 Feb), Spring (1 Mar–15 Apr). They abut without gaps.
3. **Measures**: base as mean SWE and, where the record allows, mean snow
   depth; storm days and powder days counted from daily SWE gain (10 mm and
   25 mm thresholds, normalised per 30 days); days with a skiable base. Storm
   counts come from SWE gain because it runs longer than snow depth and is
   density-independent — a 25 mm water gain is a big storm at any density.
4. **Standardisation**: per station and window, the z-score of the linearly
   detrended series, so a warming trend cannot pose as an ENSO signal.
   Regional values are the elevation-weighted mean of those z-scores.
5. **Significance**: the modified bootstrap of Brown & Harper (2026) — omit a
   random 20 % of winters, refit, repeat 10,000 times, and require the 2σ
   bounds of the near-Gaussian coefficient distribution to exclude zero.
   Because the raw 2σ rule fires on about 30 % of pure noise (the subsample
   spread is smaller than the sampling error by the delete-d factor
   sqrt((n-d)/d) = 2), a calibrated version is reported alongside, plus a
   permutation p-value. **Benjamini-Hochberg false-discovery control is then
   applied across all 260 tests**, because asking that many questions at
   p below 0.05 buys false positives for free.
6. **Correlations are reported as r²**, the share of year-to-year variance
   explained. Signed r² keeps the direction: negative means El Niño brings
   less.

### Data

| Source | What | Period |
|---|---|---|
| NOAA CPC ONI | 3-month Niño-3.4 SST anomaly, the official ENSO index | 1950– |
| NRCS SNOTEL (AWDB REST v1) | daily SWE, snow depth, precipitation, air temperature; 534 stations | ~1964– |
| NRCS snow courses | manual April-1 SWE, 1,075 courses, extends the record | ~1930s– |
| NCEI nClimDiv | statewide monthly precipitation and temperature | 1895– |
| NOAA PSL MEI v2 | Multivariate ENSO Index, sensitivity check | 1979– |

---

## Running it

```
python3 -m pip install -r requirements.txt
python3 -m enso_snowpack fetch      # ~45 min, resumable, caches per station
python3 -m enso_snowpack analyze    # results/summary.md, figures, CSV tables
python3 -m pytest tests -q          # offline tests
```

`analyze` flags: `--no-ski`, `--no-daily`, `--n-boot`, `--n-perm`,
`--omit-fraction`, and `--bootstrap` to switch resampling scheme.

To move data to a machine that cannot reach NOAA or NRCS: `bundle` reduces the
daily tables to per-station metrics in one small archive, and
`analyze --bundle <file>.tar.gz` runs from it.

### Layout

```
enso_snowpack/
  ski.py       ski regions x windows of winter — the headline analysis
  daily.py     day-by-day correlation through the water year
  bootstrap.py Brown & Harper subsampling, permutation, FDR control
  analysis.py  water-year metrics, standardisation, seasonal statistics
  sources.py   parsers for each raw format (pure functions, tested offline)
  fetch.py     downloads with a per-station cache
  figures.py   all figures (red = less snow)
  report.py    results/summary.md
  fixture.py   synthetic data with a planted signal, for offline tests
```

## Caveats

- **Snow depth starts around 1993**, so depth results rest on about 25 winters
  against 45 to 60 for the SWE-derived ones. They agree in sign and are
  stronger, which is reassuring, but they are the least certain numbers here.
- **SNOTEL is not the ski area.** Even with elevation weighting these are
  nearby mountain gauges, not on-mountain measurements, and they say nothing
  about grooming, snowmaking or aspect.
- **An r² of 15 to 28 % is real but leaves most of the variance
  unexplained.** ENSO shifts the odds; it does not determine a season.
- **Regions overlap** where radii intersect, so neighbouring rows are not
  fully independent. The sign tests treat them as independent and will read
  slightly optimistically.

## Citation

MIT licence. Cite the data providers (NOAA CPC, NRCS AWDB, NCEI nClimDiv) and
Brown & Harper (2026), *Historical evolution of snowpack capacity to buffer
rain-on-snow runoff in a large Columbia River headwaters basin*, Hydrol. Earth
Syst. Sci. 30, 5735–5748, doi:10.5194/hess-30-5735-2026, for the bootstrap.
