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


def boot_table(rows: list[dict]) -> str:
    if rows and rows[0].get("scheme") == "brown_harper_2026":
        hdr = ("| Region | n | slope per +1 °C ONI: mean ± σ | 2σ bounds | significant (2σ) | "
               "significant (calibrated 2σ) | r: mean ± σ | 2σ bounds | significant (2σ) | "
               "El Niño − rest (z): mean ± σ | significant (2σ) | permutation p (r) |")
        out = [hdr, "|" + "---|" * 12]
        for b in rows:
            out.append(
                f"| {b['region']} | {b['n']} | {_f(b['bh_slope_mean'], plus=True)} ± {_f(b['bh_slope_sd'])} | "
                f"[{_f(b['slope_ci_low'], plus=True)}, {_f(b['slope_ci_high'], plus=True)}] | "
                f"{'**yes**' if b['bh_slope_significant'] else 'no'} | "
                f"{'**yes**' if b['bh_slope_significant_cal'] else 'no'} | "
                f"{_f(b['bh_r_mean'], plus=True)} ± {_f(b['bh_r_sd'])} | "
                f"[{_f(b['r_ci_low'], plus=True)}, {_f(b['r_ci_high'], plus=True)}] | "
                f"{'**yes**' if b['bh_r_significant'] else 'no'} | "
                f"{_f(b['bh_diff_mean'], plus=True)} ± {_f(b['bh_diff_sd'])} | "
                f"{'**yes**' if b['bh_diff_significant'] else 'no'} | "
                f"{_f(b['p_perm'], 4)}{_stars(b['p_perm'])} |")
        return "\n".join(out)
    hdr = ("| Region | n | r | permutation p | r 95% CI | slope 95% CI | "
           "El Niño − rest (z) | 95% CI | permutation p |")
    out = [hdr, "|" + "---|" * 9]
    for b in rows:
        out.append(
            f"| {b['region']} | {b['n']} | {_f(b['r_obs'], plus=True)} | {_f(b['p_perm'], 3)}{_stars(b['p_perm'])} | "
            f"[{_f(b['r_ci_low'], plus=True)}, {_f(b['r_ci_high'], plus=True)}] | "
            f"[{_f(b['slope_ci_low'], plus=True)}, {_f(b['slope_ci_high'], plus=True)}] | "
            f"{_f(b['diff_nino_rest_obs'], plus=True)} | "
            f"[{_f(b['diff_ci_low'], plus=True)}, {_f(b['diff_ci_high'], plus=True)}] | "
            f"{_f(b['p_perm_diff'], 3)}{_stars(b['p_perm_diff'])} |")
    return "\n".join(out)


def enso_years_table(enso: pd.DataFrame) -> str:
    """Which water years fall in each phase × strength bin."""
    order = [("El Nino", "very strong"), ("El Nino", "strong"), ("El Nino", "moderate"),
             ("El Nino", "weak"), ("Neutral", "neutral"), ("La Nina", "weak"),
             ("La Nina", "moderate"), ("La Nina", "strong"), ("La Nina", "very strong")]
    out = ["| Phase | Strength (winter peak ONI) | n | Water years (DJF ONI) |", "|---|---|---|---|"]
    for phase, strength in order:
        g = enso[(enso["phase"] == phase) & (enso["strength"] == strength)].sort_values("water_year")
        if g.empty:
            continue
        years = ", ".join(f"{int(r.water_year)} ({r.oni_djf:+.1f})" for r in g.itertuples())
        out.append(f"| {phase} | {strength} | {len(g)} | {years} |")
    return "\n".join(out)


def station_summary(sc: pd.DataFrame) -> str:
    if sc.empty:
        return "_no stations met the record-length criterion_"
    has_fdr = "fdr_significant" in sc
    out = ["| State | stations | median r | share r<0 | p<0.05 negative | p<0.05 positive | FDR-significant (α=0.10) |",
           "|---|---|---|---|---|---|---|"]
    for st, g in sc.groupby("stateCode"):
        neg = (g["r"] < 0).mean()
        sn = ((g["p"] < 0.05) & (g["r"] < 0)).sum()
        sp = ((g["p"] < 0.05) & (g["r"] > 0)).sum()
        fdr = int(g["fdr_significant"].fillna(False).astype(bool).sum()) if has_fdr else "—"
        out.append(f"| {STATE_NAMES.get(st, st)} | {len(g)} | {_f(g['r'].median(), plus=True)} | "
                   f"{100*neg:.0f}% | {sn} | {sp} | {fdr} |")
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
        "## ENSO classification of the water years",
        "",
        "Phase from the DJF ONI (El Niño ≥ +0.5 °C, La Niña ≤ −0.5 °C); strength from the peak "
        "|ONI| over the OND…FMA seasons of the water year, in the CPC bins weak 0.5–0.9, moderate "
        "1.0–1.4, strong 1.5–1.9, very strong ≥ 2.0. The value in parentheses is the DJF ONI.",
        "",
        enso_years_table(e),
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
        f"## Resampling significance (April-1 SWE, scheme `{ctx.get('boot_scheme', '')}`)",
        "",
        ("Brown & Harper (2026) modified bootstrap regression: a random 20 % of the water years is "
         "omitted, the regression of snowpack anomaly on DJF ONI is fitted to the remaining 80 %, "
         "and this is repeated 10 000 times; the coefficient distribution is near-Gaussian and the "
         "relation is significant only where its 2σ bounds exclude zero. The `calibrated 2σ` column "
         "rescales σ by the delete-d jackknife factor sqrt((n−d)/d) (= 2 for 80/20), which turns the "
         "subsample spread into a full-sample standard error; the raw 2σ rule flags ~30 % of "
         "pure-noise cases, the calibrated one ~5 %. The permutation p shuffles the ENSO index across "
         "years as an independent check. These, not the parametric p-values above, are the "
         "significance test to quote."
         if ctx.get("boot_scheme") == "brown_harper_2026" else
         "Permutation p-values shuffle the ENSO index across water years (no normality assumed); "
         "the CIs come from the selected bootstrap scheme with water years as the exchangeable unit. "
         "These, not the parametric p-values above, are the significance test to quote."),
        "",
        boot_table(ctx.get("boot_apr1", [])),
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
