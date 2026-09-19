"""Render results/summary.md from the computed tables."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import STATE_NAMES


def _f(x, nd=2, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def _stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _verdict(r, p, lo, hi):
    if not np.isfinite(p):
        return "insufficient data"
    if p >= 0.05:
        return "no significant linear relation"
    sign = "wetter" if r > 0 else "drier"
    strength = "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong"
    return f"{strength}: El Niño winters tend to be {sign}"


def corr_table(rows: list[dict]) -> str:
    hdr = ("| Region | n | Pearson r [95% CI] | p | Spearman ρ | slope per +1 °C ONI | "
           "mean El Niño / Neutral / La Niña (n) | Welch p (El Niño vs rest) | r within El Niño (n) | reading |")
    sep = "|" + "---|" * 10
    out = [hdr, sep]
    for c in rows:
        out.append(
            f"| {c['region']} | {c['n']} | {_f(c['pearson_r'], plus=True)} "
            f"[{_f(c['r_ci_low'], plus=True)}, {_f(c['r_ci_high'], plus=True)}] | "
            f"{_f(c['pearson_p'], 3)}{_stars(c['pearson_p'])} | {_f(c['spearman_rho'], plus=True)} | "
            f"{_f(c['slope'], plus=True)} ± {_f(c['slope_se'])} | "
            f"{_f(c['mean_nino'], plus=True)} / {_f(c['mean_neutral'], plus=True)} / {_f(c['mean_nina'], plus=True)} "
            f"({c['n_nino']}/{c['n_neutral']}/{c['n_nina']}) | {_f(c['welch_p_nino_vs_rest'], 3)} | "
            f"{_f(c['r_within_nino'], plus=True)} ({c['n_within_nino']}) | "
            f"{_verdict(c['pearson_r'], c['pearson_p'], c['r_ci_low'], c['r_ci_high'])} |")
    return "\n".join(out)


def composite_table(comp: pd.DataFrame) -> str:
    out = ["| Phase | Strength | n | mean | median | sd |", "|---|---|---|---|---|---|"]
    for r in comp.itertuples():
        out.append(f"| {r.phase} | {r.strength} | {r.n} | {_f(r.mean, plus=True)} | "
                   f"{_f(r.median, plus=True)} | {_f(r.sd)} |")
    return "\n".join(out)


def station_summary(sc: pd.DataFrame) -> str:
    if sc.empty:
        return "_no stations met the record-length criterion_"
    out = ["| State | stations | median r | share r<0 | significant negative | significant positive |",
           "|---|---|---|---|---|---|"]
    for st, g in sc.groupby("stateCode"):
        neg = (g["r"] < 0).mean()
        sn = ((g["p"] < 0.05) & (g["r"] < 0)).sum()
        sp = ((g["p"] < 0.05) & (g["r"] > 0)).sum()
        out.append(f"| {STATE_NAMES.get(st, st)} | {len(g)} | {_f(g['r'].median(), plus=True)} | "
                   f"{100*neg:.0f}% | {sn} | {sp} |")
    return "\n".join(out)


def write_report(out_dir: Path, ctx: dict) -> Path:
    e = ctx["enso"]
    nino = e[e["phase"] == "El Nino"]
    lines = [
        "# El Niño and snowpack in Montana, Idaho, Colorado and Utah",
        "",
        f"_Generated {date.today().isoformat()} by `enso_snowpack` "
        f"({'SYNTHETIC FIXTURE — not real data' if ctx.get('fixture') else 'real data'})._",
        "",
        "## Question",
        "",
        "Is there a correlation between El Niño strength (Oceanic Niño Index, ONI) and the "
        "winter snowpack of the four states, and how strong is it?",
        "",
        "## Data",
        "",
        f"- ENSO: NOAA CPC ONI, water years {int(e['water_year'].min())}–{int(e['water_year'].max())}; "
        f"{len(nino)} El Niño winters ({', '.join(str(int(y)) for y in nino['water_year'])}), "
        f"{(e['phase'] == 'La Nina').sum()} La Niña winters, {(e['phase'] == 'Neutral').sum()} neutral.",
        f"- SNOTEL: {ctx['n_stations_used']} stations with ≥ {ctx['min_years']} valid April-1 SWE "
        f"values (of {ctx['n_stations_total']} in the four states), daily WTEQ / PREC / TAVG.",
    ]
    if ctx.get("n_courses_used"):
        lines.append(f"- Snow courses: {ctx['n_courses_used']} manual courses with ≥ {ctx['min_years']} "
                     "April-1 measurements (extends the record before SNOTEL).")
    if ctx.get("have_nclimdiv"):
        lines.append("- Weather: NCEI nClimDiv statewide monthly precipitation and mean temperature "
                     "(Nov–Mar precipitation as % of the 1991–2020 normal; DJF temperature anomaly).")
    lines += [
        "",
        "Snowpack metric: April-1 snow-water equivalent per station, standardised per station "
        "(z-score of the linearly detrended series, so long-term warming trends cannot masquerade "
        "as ENSO signal), then averaged over the stations of each state for each water year. "
        "Peak SWE is analysed the same way. The ENSO index is the DJF ONI of the water year; "
        "El Niño ≥ +0.5 °C, La Niña ≤ −0.5 °C.",
        "",
        "## Headline: correlation of April-1 SWE anomaly with DJF ONI",
        "",
        corr_table(ctx["corr_apr1"]),
        "",
        "`slope` is in station-z units per °C of ONI. `r within El Niño` is the correlation using "
        "El Niño winters only — this is the direct test of whether *stronger* events matter more. "
        "Stars: * p<0.05, ** p<0.01, *** p<0.001.",
        "",
        "![](fig2_scatter_apr1.png)",
        "",
        "![](fig3_station_map_apr1.png)",
        "",
        "### Per-station correlations (April-1 SWE)",
        "",
        station_summary(ctx["station_corr_apr1"]),
        "",
        "## Peak SWE",
        "",
        corr_table(ctx["corr_peak"]),
        "",
        "## By phase and event strength (April-1 SWE, state means)",
        "",
        "![](fig4_phase_boxes_apr1.png)",
        "",
        "![](fig5_nino_strength_apr1.png)",
        "",
    ]
    for region, comp in ctx["composites"].items():
        lines += [f"### {region}", "", composite_table(comp), ""]
    if ctx.get("corr_pct"):
        lines += ["## Sensitivity: percent-of-median instead of detrended z", "",
                  corr_table(ctx["corr_pct"]), ""]
    if ctx.get("corr_courses"):
        lines += ["## Snow courses (longer record, April-1 SWE)", "", corr_table(ctx["corr_courses"]), ""]
    if ctx.get("corr_precip"):
        lines += ["## Weather: nClimDiv statewide Nov–Mar precipitation (% of normal) vs DJF ONI", "",
                  corr_table(ctx["corr_precip"]), "", "![](fig7_nclimdiv_precip.png)", ""]
    if ctx.get("corr_temp"):
        lines += ["## Weather: nClimDiv statewide DJF temperature anomaly (°C) vs DJF ONI", "",
                  corr_table(ctx["corr_temp"]), ""]
    if ctx.get("corr_snotel_precip"):
        lines += ["## SNOTEL Oct–Mar precipitation anomaly vs DJF ONI", "",
                  corr_table(ctx["corr_snotel_precip"]), ""]
    if ctx.get("corr_mei"):
        lines += ["## Sensitivity: MEI v2 (Dec–Jan) instead of ONI", "", corr_table(ctx["corr_mei"]), ""]
    lines += [
        "![](fig6_regional_timeseries_apr1.png)",
        "",
        "## How to read this",
        "",
        "- A negative r means El Niño winters have *less* snow than average (and La Niña more); "
        "a positive r the reverse. |r| of 0.3 explains about 10 % of year-to-year variance, "
        "0.5 about 25 %.",
        "- The 95 % CI on r is a bootstrap over water years. If it spans zero the sign is not "
        "established.",
        "- `r within El Niño` answers the *strength* question directly but rests on few winters; "
        "treat |r| < 0.4 with n < 12 as not established.",
        "- The nClimDiv rows use 70+ winters, so they are the most robust sign test; the SNOTEL rows "
        "use ~40 but measure the snowpack itself.",
        "",
        "Tables: `corr_*.csv`, `station_corr_*.csv`, `regional_*.csv`, `enso_water_years.csv`.",
    ]
    p = out_dir / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
