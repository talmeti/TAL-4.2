# -*- coding: utf-8 -*-
"""Analisi statistica TAL vs PySR vs poly per dominio."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPORT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPORT_DIR / "data"

with (DATA_DIR / "tal_4_2_step04_benchmark_v2_FINAL.json").open(encoding="utf-8") as f:
    d = json.load(f)

results = d["results"]
domains = ["Feynman", "ODE", "Dinamici", "Reali"]

def _errs(method, domain=None, exclude_outliers=True):
    out = []
    for r in results:
        if r["method"] != method or not r["ok"]:
            continue
        if domain is not None and r["domain"] != domain:
            continue
        if exclude_outliers and r["final_err"] >= 1e10:
            continue
        out.append(r["final_err"])
    return out

print("=" * 90)
print(" TAL vs PySR per dominio")
print("=" * 90)
print(f"  {'Dominio':<12} {'TAL n':<6} {'TAL med':<14} "
      f"{'PySR n':<7} {'PySR med':<14} {'p-value':<12} {'Signif.'}")
print(f"  {'-'*12} {'-'*6} {'-'*14} {'-'*7} {'-'*14} {'-'*12} {'-'*8}")

for dom in domains:
    tal = _errs("tal", domain=dom)
    pysr = _errs("pysr", domain=dom)

    if not tal or not pysr:
        print(f"  {dom:<12} (dati insufficienti)")
        continue

    tal_med = np.median(tal)
    pysr_med = np.median(pysr)

    if len(tal) >= 3 and len(pysr) >= 3:
        stat, p = stats.mannwhitneyu(tal, pysr, alternative="two-sided")
    else:
        p = float("nan")

    if np.isnan(p):
        signif = "n/a"
    elif p < 0.001:
        signif = "***"
    elif p < 0.01:
        signif = "**"
    elif p < 0.05:
        signif = "*"
    else:
        signif = "ns"

    print(f"  {dom:<12} {len(tal):<6} {tal_med:<14.3e} "
          f"{len(pysr):<7} {pysr_med:<14.3e} {p:<12.4e} {signif}")

print()
print("Legenda: *** p<0.001, ** p<0.01, * p<0.05, ns = non significativo")
print()

print("=" * 90)
print(" TAL vs poly per dominio")
print("=" * 90)
print(f"  {'Dominio':<12} {'TAL n':<6} {'TAL med':<14} "
      f"{'poly n':<7} {'poly med':<14} {'p-value':<12} {'Signif.'}")
print(f"  {'-'*12} {'-'*6} {'-'*14} {'-'*7} {'-'*14} {'-'*12} {'-'*8}")

for dom in domains:
    tal = _errs("tal", domain=dom)
    poly = _errs("poly", domain=dom)

    if not tal or not poly:
        continue

    stat, p = stats.mannwhitneyu(tal, poly, alternative="two-sided")

    if p < 0.001:
        signif = "***"
    elif p < 0.01:
        signif = "**"
    elif p < 0.05:
        signif = "*"
    else:
        signif = "ns"

    print(f"  {dom:<12} {len(tal):<6} {np.median(tal):<14.3e} "
          f"{len(poly):<7} {np.median(poly):<14.3e} {p:<12.4e} {signif}")

print()
print("=" * 90)
print(" TAL vs PySR (globale)")
print("=" * 90)
tal = _errs("tal")
pysr = _errs("pysr")
poly = _errs("poly")

stat, p = stats.mannwhitneyu(tal, pysr, alternative="two-sided")
print(f"  TAL vs PySR : U={stat:.1f}, p={p:.4e}")
stat, p = stats.mannwhitneyu(tal, poly, alternative="two-sided")
print(f"  TAL vs poly : U={stat:.1f}, p={p:.4e}")