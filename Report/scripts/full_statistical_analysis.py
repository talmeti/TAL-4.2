# -*- coding: utf-8 -*-
"""Analisi statistica completa per il report TAL 4.2."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPORT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPORT_DIR / "data"
OUT_PATH = DATA_DIR / "full_stats_summary.txt"

with (DATA_DIR / "tal_4_2_step04_benchmark_v2_FINAL.json").open(encoding="utf-8") as f:
    d = json.load(f)

results = d["results"]
domains = ["Feynman", "ODE", "Dinamici", "Reali"]
methods = ["poly", "pysr", "tal"]


def errs(method, domain=None):
    """Estrae gli errori per metodo/dominio, escludendo outlier (>=1e10)."""
    out = []
    for r in results:
        if r["method"] != method or not r["ok"]:
            continue
        if domain is not None and r["domain"] != domain:
            continue
        if r["final_err"] >= 1e10:
            continue
        out.append(r["final_err"])
    return out


def cohens_d(x, y):
    """Effect size Cohen's d."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return float("inf") if np.mean(x) != np.mean(y) else 0.0
    return (np.mean(y) - np.mean(x)) / sp


def ci95(x):
    """Intervallo di confidenza al 95% della media."""
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    m = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(n)
    h = 1.96 * se
    return (m - h, m + h)


lines = []
lines.append("=" * 100)
lines.append(" TAL 4.2 - Analisi statistica completa")
lines.append("=" * 100)
lines.append("")

# ---------- 1. STATISTICHE DESCRITTIVE GLOBALI ----------
lines.append("--- 1. Statistiche descrittive (globale) ---")
lines.append(f"{'Metodo':<8} {'n':<5} {'mean':<14} {'median':<14} "
             f"{'std':<14} {'CI95_low':<14} {'CI95_high':<14}")
lines.append("-" * 100)

for m in methods:
    vals = np.array(errs(m))
    if len(vals) == 0:
        continue
    lo, hi = ci95(vals)
    lines.append(f"{m:<8} {len(vals):<5} {np.mean(vals):<14.3e} "
                 f"{np.median(vals):<14.3e} {np.std(vals):<14.3e} "
                 f"{lo:<14.3e} {hi:<14.3e}")
lines.append("")

# ---------- 2. MANN-WHITNEY U GLOBALE ----------
lines.append("--- 2. Mann-Whitney U test (globale) ---")
lines.append(f"{'Confronto':<20} {'U':<12} {'p-value':<14} {'Signif.':<8} "
             f"{'Cohen d':<12}")
lines.append("-" * 80)

pairs = [("tal", "poly"), ("tal", "pysr"), ("pysr", "poly")]
for a, b in pairs:
    va, vb = errs(a), errs(b)
    if len(va) < 2 or len(vb) < 2:
        continue
    stat, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
    cd = cohens_d(va, vb)

    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"

    lines.append(f"{a + ' vs ' + b:<20} {stat:<12.1f} {p:<14.4e} "
                 f"{sig:<8} {cd:<12.3f}")
lines.append("")

# ---------- 3. MANN-WHITNEY U PER DOMINIO ----------
lines.append("--- 3. Mann-Whitney U test per dominio (TAL vs PySR) ---")
lines.append(f"{'Dominio':<12} {'TAL n':<6} {'PySR n':<7} {'U':<10} "
             f"{'p-value':<14} {'Signif.':<8} {'Cohen d':<12}")
lines.append("-" * 100)

for dom in domains:
    va = errs("tal", domain=dom)
    vb = errs("pysr", domain=dom)
    if len(va) < 2 or len(vb) < 2:
        continue
    stat, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
    cd = cohens_d(va, vb)

    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"

    lines.append(f"{dom:<12} {len(va):<6} {len(vb):<7} {stat:<10.1f} "
                 f"{p:<14.4e} {sig:<8} {cd:<12.3f}")
lines.append("")

# ---------- 4. MANN-WHITNEY U PER DOMINIO (TAL vs poly) ----------
lines.append("--- 4. Mann-Whitney U test per dominio (TAL vs poly) ---")
lines.append(f"{'Dominio':<12} {'TAL n':<6} {'poly n':<7} {'U':<10} "
             f"{'p-value':<14} {'Signif.':<8} {'Cohen d':<12}")
lines.append("-" * 100)

for dom in domains:
    va = errs("tal", domain=dom)
    vb = errs("poly", domain=dom)
    if len(va) < 2 or len(vb) < 2:
        continue
    stat, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
    cd = cohens_d(va, vb)

    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"

    lines.append(f"{dom:<12} {len(va):<6} {len(vb):<7} {stat:<10.1f} "
                 f"{p:<14.4e} {sig:<8} {cd:<12.3f}")
lines.append("")

# ---------- 5. WILCOXON SIGNED-RANK (differenze appaiate) ----------
lines.append("--- 5. Wilcoxon signed-rank test (TAL vs PySR, run appaiati) ---")
lines.append("")

# Appaia per (dominio, funzione, seed)
pairs_data = {}
for r in results:
    if r["method"] not in ("tal", "pysr"):
        continue
    if not r["ok"] or r["final_err"] >= 1e10:
        continue
    key = (r["domain"], r["function"], r["seed"])
    pairs_data.setdefault(key, {})[r["method"]] = r["final_err"]

tal_paired = []
pysr_paired = []
for key, d_ in pairs_data.items():
    if "tal" in d_ and "pysr" in d_:
        tal_paired.append(d_["tal"])
        pysr_paired.append(d_["pysr"])

if len(tal_paired) >= 3:
    stat, p = stats.wilcoxon(tal_paired, pysr_paired)
    lines.append(f"Run appaiati: n={len(tal_paired)}")
    lines.append(f"Wilcoxon W={stat:.1f}, p={p:.4e}")
    if p < 0.05:
        lines.append("Differenza significativa tra TAL e PySR.")
    else:
        lines.append("Nessuna differenza significativa tra TAL e PySR.")
lines.append("")

# ---------- 6. SOMMARIO ----------
lines.append("--- 6. Sommario interpretativo ---")
lines.append("")
lines.append("Confronti significativi (p < 0.05):")
for a, b in pairs:
    va, vb = errs(a), errs(b)
    if len(va) < 2 or len(vb) < 2:
        continue
    _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
    if p < 0.05:
        lines.append(f"  - {a} vs {b}: p={p:.4e}")
lines.append("")
lines.append("Confronti non significativi (p >= 0.05):")
for a, b in pairs:
    va, vb = errs(a), errs(b)
    if len(va) < 2 or len(vb) < 2:
        continue
    _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
    if p >= 0.05:
        lines.append(f"  - {a} vs {b}: p={p:.4e}")
lines.append("")

lines.append("Per dominio (TAL vs PySR):")
for dom in domains:
    va = errs("tal", domain=dom)
    vb = errs("pysr", domain=dom)
    if len(va) < 2 or len(vb) < 2:
        continue
    _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
    s = "significativo" if p < 0.05 else "non significativo"
    lines.append(f"  - {dom}: p={p:.4e} ({s})")
lines.append("")

lines.append("=" * 100)

text = "\n".join(lines)
OUT_PATH.write_text(text, encoding="utf-8")
print(text)
print()
print(f">>> Salvato in {OUT_PATH}")