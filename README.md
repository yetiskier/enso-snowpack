# El Niño and the ski season — the western United States

**Does El Niño affect the ski season, where, and when in the winter?**
NOAA's Oceanic Niño Index against NRCS SNOTEL snowpack for 42 ski regions
across every SNOTEL state, reported in powder days and inches.

_Last updated 2026-09-21. Real data: 932 SNOTEL stations,
40.2 million daily observations, water years 1964–2026, 25 El Niño, 27 La Niña
and 25 neutral winters._

This is not a water-supply study. April-1 snow-water equivalent is a runoff
metric. A skier cares how often it snows six inches and how deep the base is
in the windows that carry a season, so that is what is measured.

**Report:** `results/report.html`, or `results/El-Nino-and-the-Ski-Season.pdf`
(`make pdf` rebuilds it).

---

## The answer in five points

### 1. The effect is real but geographically narrow

Of 971 tests, **51 survive** Benjamini-Hochberg false-discovery control
across the whole grid. Nearly all sit in the northern Rockies and the Cascades,
where El Niño costs roughly a foot of snow, or in New Mexico and Arizona, where
it adds snow in spring.

**In inches:**

| Ski region | Window | Measure | winters | a normal winter | an El Niño winter | difference | r² |
|---|---|---|---|---|---|---|---|
| Bighorn Mtns (Meadowlark) | Holidays | Total new snow | 48 | 11.5 in | **7.9 in** | **-3.6 in** (-31 %) | 25% |
| Park Range (Steamboat) | Holidays | Total new snow | 47 | 25.1 in | **17.5 in** | **-7.7 in** (-30 %) | 19% |
| Snowy Range & Sierra Madre | Holidays | Total new snow | 48 | 25.3 in | **17.9 in** | **-7.4 in** (-29 %) | 18% |
| Mt Hood | Spring | Base, snow depth | 29 | 69.5 in | **50.4 in** | **-19.2 in** (-28 %) | 25% |
| Central Cascades (Stevens/Snoqualmie) | Spring | Base, snow depth | 28 | 63.7 in | **48.2 in** | **-15.5 in** (-24 %) | 27% |
| Mt Hood | Spring | Base, water equivalent | 48 | 27.3 in | **20.8 in** | **-6.4 in** (-23 %) | 23% |
| Central Cascades (Stevens/Snoqualmie) | Spring | Total new snow | 45 | 50.4 in | **38.6 in** | **-11.8 in** (-23 %) | 21% |
| Central Cascades (Stevens/Snoqualmie) | Spring | Base, water equivalent | 45 | 28.9 in | **23.3 in** | **-5.6 in** (-19 %) | 19% |
| Bitterroots (Lost Trail) | Midwinter | Total new snow | 59 | 59.7 in | **48.1 in** | **-11.5 in** (-19 %) | 15% |
| Missoula ranges (Montana Snowbowl) | Spring | Base, water equivalent | 46 | 19.2 in | **15.7 in** | **-3.5 in** (-18 %) | 27% |
| Coeur d'Alenes (Silver/Lookout) | Spring | Base, water equivalent | 44 | 23.8 in | **19.6 in** | **-4.3 in** (-18 %) | 25% |
| Missoula ranges (Montana Snowbowl) | Midwinter | Total new snow | 46 | 57.3 in | **47.1 in** | **-10.3 in** (-18 %) | 21% |
| Bitterroots (Lost Trail) | Spring | Base, water equivalent | 59 | 18.8 in | **15.9 in** | **-2.9 in** (-15 %) | 21% |
| Missoula ranges (Montana Snowbowl) | Spring | Base, snow depth | 25 | 61.4 in | **52.0 in** | **-9.4 in** (-15 %) | 37% |
| Whitefish & Flathead Range | Spring | Base, snow depth | 25 | 60.7 in | **51.8 in** | **-8.9 in** (-15 %) | 34% |
| Whitefish & Flathead Range | Spring | Base, water equivalent | 52 | 22.2 in | **19.0 in** | **-3.2 in** (-14 %) | 16% |
| Selkirks (Schweitzer) | Spring | Base, snow depth | 25 | 85.1 in | **73.6 in** | **-11.5 in** (-14 %) | 29% |
| Missoula ranges (Montana Snowbowl) | Midwinter | Base, water equivalent | 46 | 13.6 in | **11.8 in** | **-1.8 in** (-13 %) | 17% |
| Bitterroots (Lost Trail) | Midwinter | Base, water equivalent | 59 | 13.2 in | **11.7 in** | **-1.5 in** (-12 %) | 15% |
| Beartooth & Absaroka (Red Lodge) | Spring | Base, water equivalent | 53 | 17.6 in | **15.6 in** | **-2.0 in** (-12 %) | 16% |
| Missoula ranges (Montana Snowbowl) | Midwinter | Base, snow depth | 25 | 51.2 in | **45.5 in** | **-5.7 in** (-11 %) | 36% |
| Madison & Gallatin (Big Sky) | Midwinter | Base, water equivalent | 60 | 11.6 in | **10.5 in** | **-1.2 in** (-10 %) | 13% |
| Bitterroots (Lost Trail) | Midwinter | Base, snow depth | 25 | 44.5 in | **40.1 in** | **-4.4 in** (-10 %) | 31% |
| Madison & Gallatin (Big Sky) | Spring | Base, water equivalent | 60 | 17.2 in | **15.6 in** | **-1.7 in** (-10 %) | 15% |
| Whitefish & Flathead Range | Midwinter | Base, snow depth | 25 | 49.9 in | **45.1 in** | **-4.8 in** (-10 %) | 30% |
| S Sangre de Cristo (Santa Fe) | Early season | Total new snow | 46 | 24.8 in | **31.7 in** | **+6.9 in** (+28 %) | 22% |
| Jemez (Pajarito) | Early season | Total new snow | 46 | 20.0 in | **25.7 in** | **+5.7 in** (+28 %) | 23% |
| S Sangre de Cristo (Santa Fe) | Spring | Base, water equivalent | 46 | 9.5 in | **12.3 in** | **+2.8 in** (+29 %) | 22% |
| Jemez (Pajarito) | Spring | Base, water equivalent | 46 | 7.3 in | **9.7 in** | **+2.4 in** (+33 %) | 23% |
| White Mtns AZ (Sunrise Park) | Spring | Base, water equivalent | 46 | 4.3 in | **6.3 in** | **+2.0 in** (+46 %) | 24% |
| S Sangre de Cristo (Santa Fe) | Early season | Base, snow depth | 25 | 7.0 in | **10.2 in** | **+3.2 in** (+46 %) | 32% |

**In days:**

| Ski region | Window | Measure | winters | a normal winter | an El Niño winter | difference | r² |
|---|---|---|---|---|---|---|---|
| Missoula ranges (Montana Snowbowl) | Midwinter | Powder days (≥6 in) | 46 | 1.4 d | **0.8 d** | **-0.6 d** (-42 %) | 31% |
| Bighorn Mtns (Meadowlark) | Holidays | Storm days (≥2 in) | 48 | 1.7 d | **1.0 d** | **-0.7 d** (-40 %) | 20% |
| Park Range (Steamboat) | Holidays | Storm days (≥2 in) | 47 | 4.3 d | **2.7 d** | **-1.6 d** (-36 %) | 22% |
| Coeur d'Alenes (Silver/Lookout) | Midwinter | Powder days (≥6 in) | 44 | 2.0 d | **1.3 d** | **-0.7 d** (-34 %) | 22% |
| Snowy Range & Sierra Madre | Holidays | Storm days (≥2 in) | 48 | 4.2 d | **2.9 d** | **-1.3 d** (-32 %) | 17% |
| Central Cascades (Stevens/Snoqualmie) | Spring | Storm days (≥2 in) | 45 | 8.5 d | **6.4 d** | **-2.1 d** (-25 %) | 24% |
| Bitterroots (Lost Trail) | Midwinter | Storm days (≥2 in) | 59 | 9.9 d | **7.9 d** | **-2.0 d** (-20 %) | 13% |
| S Washington Cascades (Crystal/White Pass) | Spring | Storm days (≥2 in) | 45 | 9.9 d | **8.1 d** | **-1.8 d** (-18 %) | 16% |
| Madison & Gallatin (Big Sky) | Midwinter | Days with a skiable base | 60 | 32.0 d | **26.9 d** | **-5.1 d** (-16 %) | 13% |
| Missoula ranges (Montana Snowbowl) | Midwinter | Days with a skiable base | 46 | 28.9 d | **24.9 d** | **-4.0 d** (-14 %) | 25% |
| Whitefish & Flathead Range | Spring | Days with a skiable base | 52 | 38.1 d | **33.2 d** | **-4.9 d** (-13 %) | 18% |
| Missoula ranges (Montana Snowbowl) | Spring | Days with a skiable base | 46 | 32.6 d | **28.5 d** | **-4.1 d** (-13 %) | 33% |
| Wind River Range | Spring | Powder days (≥6 in) | 48 | 0.6 d | **0.8 d** | **+0.1 d** (+21 %) | 19% |
| White Mtns AZ (Sunrise Park) | Spring | Storm days (≥2 in) | 46 | 2.3 d | **2.9 d** | **+0.6 d** (+24 %) | 17% |
| S Sangre de Cristo (Santa Fe) | Early season | Storm days (≥2 in) | 46 | 3.9 d | **4.9 d** | **+1.0 d** (+26 %) | 19% |
| Jemez (Pajarito) | Early season | Storm days (≥2 in) | 46 | 3.4 d | **4.3 d** | **+0.9 d** (+28 %) | 20% |
| Chugach (Alyeska) | Holidays | Storm days (≥2 in) | 43 | 4.5 d | **5.7 d** | **+1.2 d** (+28 %) | 17% |
| S Sangre de Cristo (Santa Fe) | Spring | Days with a skiable base | 46 | 22.1 d | **33.3 d** | **+11.2 d** (+51 %) | 29% |
| Jemez (Pajarito) | Spring | Days with a skiable base | 46 | 15.2 d | **23.9 d** | **+8.7 d** (+57 %) | 21% |
| White Mtns AZ (Sunrise Park) | Spring | Days with a skiable base | 46 | 8.2 d | **14.3 d** | **+6.1 d** (+74 %) | 18% |

### 2. The effect moves through the winter

Judged region by region, the number whose 95 % interval excludes zero shifts
window to window — and so does *where* they are. The holidays hit the interior
ranges; midwinter hits the north.

| Window | Regions with an interval excluding zero |
|---|---|
| Early season | 2 of 41 |
| Holidays | 13 of 41 |
| Midwinter | 8 of 41 |
| Spring | 4 of 41 |

### 3. Strength carries no information

| Test | tests run | p<0.05 | expected by chance | survive FDR | smallest r findable |
|---|---|---|---|---|---|
| Adding magnitude to phase (every winter) | 971 | 23 | 49 | **0** | 0.40 |
| Within La Niña winters only | 859 | 47 | 43 | **0** | 0.65 |
| Within El Niño winters only | 789 | 16 | 39 | **0** | 0.67 |

**0 of 2619 survive.** Powder days by strength bin say the same thing plainly:

| Phase and strength | winters | powder days (midwinter) |
|---|---|---|
| La Niña strong | 6 | 1.63 |
| La Niña moderate | 5 | 1.31 |
| La Niña weak | 10 | 1.37 |
| Neutral neutral | 18 | 1.33 |
| El Niño weak | 9 | 1.23 |
| El Niño moderate | 4 | 1.18 |
| El Niño strong | 4 | 1.09 |
| El Niño very strong | 3 | 1.40 |

The bars do not descend. Plan for the phase; ignore the magnitude.

### 4. The regions are the right size

Every region is a mountain range or group of ranges with **at least 4 SNOTEL
sites spanning at least 1,000 ft** of elevation (minimum 4 sites,
median 11, minimum relief 1360 ft) — enforced by
`validate_regions()` and a test. Re-running the whole grid under coarser
groupings of the same station memberships:

| Grouping | regions | tests | survive | rate | largest effect |
|---|---|---|---|---|---|
| ranges (published) | 41 | 971 | 42 | 4.3% | 0.61 |
| latitude bands, 2 deg | 10 | 238 | 42 | 17.6% | 0.51 |
| snow climates | 4 | 96 | 1 | 1.0% | 0.48 |
| latitude zones | 3 | 72 | 20 | 27.8% | 0.52 |
| whole West | 1 | 24 | 13 | 54.2% | 0.37 |

The same number of findings survives at range resolution as at coarse latitude
bands, so the fine division is not being punished into uselessness by multiple
comparisons. The largest effect shrinks monotonically as the grouping coarsens,
because averaging across the ENSO node cancels the two halves of the seesaw.
Grouping by snow climate destroys the signal almost entirely, because those
classes span the node.

### 5. An r² of 15–35 % is real and is not a forecast

ENSO shifts the odds. It does not determine a season.

---

## Regions and coverage

A state boundary is not a snow boundary: Tahoe spans California and Nevada, the
Coeur d'Alenes span Idaho and Montana, the Tetons span Wyoming and Idaho, the
Sangre de Cristo spans Colorado and New Mexico. Each region is anchored on the
ski areas it serves and **its centre is computed from their coordinates**, never
typed by hand — an earlier hand-typed version drifted 61 km from Schweitzer and
grouped Lookout Pass with Montana Snowbowl 138 km away.

Stations are weighted by a Gaussian on the vertical gap to the served
base-to-summit band times one on horizontal distance, so a valley gauge below
the lifts does not speak for a mountain.

| Ski region | states | climate | served band (ft) | stations | eff. | weighted elev (ft) |
|---|---|---|---|---|---|---|
| Beartooth & Absaroka (Red Lodge) | MT+WY | continental | 7,016–9,416 | 14 | 12.4 | 8,101 |
| Bighorn Mtns (Meadowlark) | WY | continental | 7,500–9,500 | 15 | 13.4 | 8,716 |
| Bitterroots (Lost Trail) | MT+ID | transitional | 6,400–8,000 | 13 | 10.8 | 7,000 |
| Blue Mtns (Anthony Lakes) | OR | transitional | 7,100–8,000 | 18 | 13.7 | 5,809 |
| Bridger Range (Bridger Bowl) | MT | continental | 6,100–8,700 | 6 | 5.2 | 7,156 |
| Central Cascades (Stevens/Snoqualmie) | WA | maritime | 3,000–5,845 | 24 | 20.9 | 3,832 |
| Central Oregon (Mt Bachelor) | OR | transitional | 6,300–9,065 | 10 | 7.3 | 5,220 |
| Chugach (Alyeska) | AK | maritime | 250–2,750 | 11 | 9.5 | 1,413 |
| Coeur d'Alenes (Silver/Lookout) | ID+MT | transitional | 4,100–6,300 | 8 | 6.3 | 5,055 |
| E Cascades rain shadow (Mission Ridge) | WA | transitional | 4,570–6,820 | 7 | 5.9 | 4,744 |
| Elk Mtns (Aspen/Crested Butte) | CO | continental | 7,945–12,162 | 15 | 13.8 | 10,139 |
| Front Range (Winter Park/Loveland/A-Basin) | CO | continental | 9,000–13,050 | 26 | 22.2 | 10,403 |
| Gore & Tenmile (Vail/Summit/Copper) | CO | continental | 8,120–12,998 | 25 | 22.5 | 10,388 |
| Jemez (Pajarito) | NM | continental | 8,500–10,441 | 10 | 8.2 | 9,514 |
| Klamath & Siskiyou (Mt Shasta) | CA+OR | maritime | 5,500–7,800 | 8 | 7.7 | 5,607 |
| Madison & Gallatin (Big Sky) | MT | continental | 6,800–11,166 | 10 | 9.2 | 8,173 |
| Markagunt (Brian Head) | UT | continental | 9,600–10,970 | 13 | 9.9 | 9,123 |
| Missoula ranges (Montana Snowbowl) | MT | transitional | 5,000–7,600 | 7 | 6.2 | 6,013 |
| Mt Hood | OR | maritime | 4,500–8,540 | 11 | 9.4 | 4,268 |
| N Sangre de Cristo (Taos) | NM+CO | continental | 9,200–12,481 | 13 | 11.0 | 10,169 |
| N Sierra / Tahoe (Palisades/Heavenly/Rose) | CA+NV | maritime | 6,200–10,067 | 27 | 24.9 | 7,573 |
| North Cascades (Mt Baker) | WA | maritime | 3,500–5,089 | 10 | 8.6 | 4,160 |
| Park Range (Steamboat) | CO | continental | 6,900–10,568 | 19 | 16.5 | 9,355 |
| Ruby Mtns & NE Nevada | NV | continental | 6,500–10,000 | 11 | 10.0 | 7,866 |
| S San Juans (Wolf Creek) | CO | continental | 10,300–11,904 | 16 | 12.4 | 10,970 |
| S Sangre de Cristo (Santa Fe) | NM | continental | 10,350–12,075 | 8 | 5.9 | 10,371 |
| S Sierra (Mammoth/June) | CA | maritime | 7,953–11,053 | 7 | 6.0 | 8,909 |
| S Washington Cascades (Crystal/White Pass) | WA | maritime | 4,400–7,012 | 24 | 18.8 | 4,683 |
| San Francisco Peaks (AZ Snowbowl) | AZ | continental | 9,200–11,500 | 8 | 4.8 | 8,422 |
| San Juans (Telluride/Purgatory) | CO | continental | 8,725–13,150 | 23 | 21.6 | 10,381 |
| Sawtooth & Smoky (Sun Valley) | ID | intermountain | 5,750–9,150 | 21 | 18.6 | 7,514 |
| Selkirks (Schweitzer) | ID | transitional | 4,000–6,400 | 9 | 7.2 | 5,319 |
| Snowy Range & Sierra Madre | WY | continental | 8,798–9,663 | 22 | 18.7 | 9,475 |
| Spring Mtns (Lee Canyon) | NV | continental | 8,510–11,290 | 4 | 4.0 | 8,723 |
| Tetons (Jackson/Targhee) | WY+ID | intermountain | 6,300–10,450 | 11 | 9.3 | 7,874 |
| Tushar Mtns (Eagle Point) | UT | continental | 9,000–10,600 | 11 | 9.4 | 9,359 |
| Uinta Mtns | UT | continental | 8,000–11,000 | 30 | 27.9 | 9,345 |
| W Central Idaho (Brundage/Tamarack) | ID | intermountain | 5,840–7,640 | 11 | 9.9 | 5,937 |
| Wasatch (Alta/Snowbird/Park City) | UT | intermountain | 6,800–11,000 | 30 | 25.2 | 8,173 |
| White Mtns AZ (Sunrise Park) | AZ | continental | 9,200–11,100 | 9 | 7.9 | 8,673 |
| Whitefish & Flathead Range | MT | transitional | 4,464–6,817 | 12 | 10.9 | 5,443 |
| Wind River Range | WY | continental | 7,000–10,000 | 27 | 23.7 | 8,956 |

Two regions are deliberately absent. The Sacramento Mountains have one SNOTEL
site at any radius and the Black Hills three with 970 ft of relief; neither can
stand for a range.

---

## Method

1. **Water year** N runs 1 Oct N−1 to 30 Sep N. Phase from the DJF ONI
   (El Niño ≥ +0.5 °C, La Niña ≤ −0.5 °C); strength bins follow CPC on the
   winter peak.
2. **Ski windows**: Early season (1 Nov–15 Dec), Holidays (16 Dec–5 Jan),
   Midwinter (6 Jan–28 Feb), Spring (1 Mar–15 Apr). They abut without gaps.
3. **A powder day is six inches of snow, not a fixed amount of water.** Cascade
   new snow runs about 12 % density and Colorado's about 7 %, so counting water
   would make the driest snowpacks look storm-free. Each station's new-snow
   ratio is measured from the overlap of its depth and water records (median
   7.5 in of snow per in of water over 508 stations) and applied to its much
   longer water record.
4. **Two views.** Physical composites are raw averages by phase — what happened.
   The tests use per-station detrended z-scores, so a warming trend cannot pose
   as an ENSO signal.
5. **Significance**: the modified bootstrap of Brown & Harper (2026) — omit a
   random 20 % of winters, refit, repeat 10,000 times, require the 2σ bounds to
   exclude zero. The raw rule fires on ~30 % of pure noise, so a calibrated
   version (σ × the delete-d factor √((n−d)/d) = 2) and a permutation p-value
   are reported alongside. Benjamini–Hochberg control is applied across all
   971 tests.
6. **Two standards, stated separately.** The maps judge each region alone; the
   headline tables correct across the whole grid. A region that clears the
   first but not the second is worth watching, not worth relying on.

### Data

| Source | What | Period |
|---|---|---|
| NOAA CPC ONI | 3-month Niño-3.4 SST anomaly | 1950– |
| NRCS SNOTEL | daily SWE, snow depth, precipitation, temperature; 932 stations in 13 states | ~1964– |
| NRCS snow courses | manual April-1 SWE, 1,075 courses | ~1930s– |
| NCEI nClimDiv | statewide monthly precipitation and temperature | 1895– |

## Running it

```
python3 -m pip install -r requirements.txt
python3 -m enso_snowpack fetch      # ~1 h, resumable, caches per station
python3 -m enso_snowpack analyze    # results/, figures, CSV tables
make pdf                            # results/El-Nino-and-the-Ski-Season.pdf
python3 -m pytest tests -q          # 40 offline tests
```

The daily table streams straight to parquet: 40 million rows will not fit in
memory as one frame.

### Layout

```
enso_snowpack/
  ski.py         ski regions x windows — the headline analysis
  figures_ski.py the skier-facing figures, in days and inches
  daily.py       day-by-day correlation through the water year
  bootstrap.py   Brown & Harper subsampling, permutation, FDR control
  analysis.py    water-year metrics and seasonal statistics
  sources.py     parsers for each raw format (pure functions, tested offline)
  fetch.py       downloads, per-station cache, streaming parquet sink
  figures.py     shared style and the statistical figures
  report.py      results/summary.md
  fixture.py     synthetic data with a planted signal, for offline tests
```

## Caveats

- **Snow depth starts around 1993.** New-snow ratios are calibrated on that
  period and applied backward, assuming density has been stable.
- **A SNOTEL gauge is not a ski area.** These are nearby mountain stations, and
  they say nothing about aspect, grooming or snowmaking.
- **Overlapping regions** are not fully independent where radii intersect.
- **Six inches is a choice.** Regions with few powder days show large
  percentage swings on small differences; read the inches columns there.
- **The strength null is bounded by power.** Within-phase tests had ~16 winters
  and could only have found r above ~0.65. The nested test uses every winter,
  could have found 0.40, and did not.

## Citation

MIT licence. Cite NOAA CPC, NRCS AWDB, NCEI nClimDiv, and Brown & Harper
(2026), *Historical evolution of snowpack capacity to buffer rain-on-snow
runoff in a large Columbia River headwaters basin*, Hydrol. Earth Syst. Sci.
30, 5735–5748, doi:10.5194/hess-30-5735-2026, for the bootstrap.
