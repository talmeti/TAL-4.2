# -*- coding: utf-8 -*-
"""Genera le figure pubblicabili per il report TAL 4.2."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path relativi alla cartella Report/
REPORT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPORT_DIR / "data"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with (DATA_DIR / "tal_4_2_step04_benchmark_v2_FINAL.json").open(encoding="utf-8") as f:
    d = json.load(f)

results = d["results"]
methods = ["poly", "pysr", "tal"]
colors = {"poly": "#888888", "pysr": "#3498db", "tal": "#e74c3c"}

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


# ---------- FIGURA 1: boxplot log10(MSE) per metodo ----------
fig, ax = plt.subplots(figsize=(8, 5))
data = [_errs(m) for m in methods]
data_log = [[np.log10(max(v, 1e-33)) for v in vals] for vals in data]
labels = [f"{m}\n(n={len(v)})" for m, v in zip(methods, data)]

bp = ax.boxplot(data_log, tick_labels=labels, patch_artist=True)
for patch, m in zip(bp["boxes"], methods):
    patch.set_facecolor(colors[m])
    patch.set_alpha(0.7)
ax.set_ylabel("log10(MSE)")
ax.set_title("Distribuzione dell'errore per metodo (quick mode)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_boxplot_method.png", dpi=200)
plt.close(fig)


# ---------- FIGURA 2: solved_rate per metodo ----------
fig, ax = plt.subplots(figsize=(6, 4))
solved = [np.mean(np.array(_errs(m)) < 0.01) * 100 for m in methods]
bars = ax.bar(methods, solved, color=[colors[m] for m in methods], alpha=0.8)
for bar, val in zip(bars, solved):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontsize=11)
ax.set_ylabel("Solved rate (%)")
ax.set_ylim(0, 110)
ax.set_title("Solved rate per metodo")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_solved_rate.png", dpi=200)
plt.close(fig)


# ---------- FIGURA 3: boxplot log10(MSE) per dominio (TAL vs PySR) ----------
domains = ["Feynman", "ODE", "Dinamici", "Reali"]
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
for ax, dom in zip(axes, domains):
    tal_vals = _errs("tal", domain=dom)
    pysr_vals = _errs("pysr", domain=dom)
    data_dom = [
        [np.log10(max(v, 1e-33)) for v in tal_vals],
        [np.log10(max(v, 1e-33)) for v in pysr_vals],
    ]
    bp = ax.boxplot(data_dom, tick_labels=["TAL", "PySR"], patch_artist=True)
    bp["boxes"][0].set_facecolor(colors["tal"])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(colors["pysr"])
    bp["boxes"][1].set_alpha(0.7)
    ax.set_title(dom)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("log10(MSE)")
fig.suptitle("TAL vs PySR: errore per dominio")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_boxplot_domain.png", dpi=200)
plt.close(fig)


# ---------- FIGURA 4: scatter tempo vs errore ----------
fig, ax = plt.subplots(figsize=(8, 5))
for m in methods:
    xs, ys = [], []
    for r in results:
        if r["method"] != m or not r["ok"]:
            continue
        if r["final_err"] >= 1e10:
            continue
        xs.append(r["time"])
        ys.append(max(r["final_err"], 1e-33))
    ax.scatter(xs, ys, alpha=0.5, s=20, color=colors[m], label=m)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Tempo (s, scala log)")
ax.set_ylabel("MSE (scala log)")
ax.set_title("Trade-off tempo / errore")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_scatter_time_error.png", dpi=200)
plt.close(fig)

print(f"Figure generate in {FIG_DIR}")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")