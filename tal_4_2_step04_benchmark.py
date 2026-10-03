# -*- coding: utf-8 -*-
# =========================================================
# TAL 4.2 — Theory-Adaptive Learning
# File: tal_4_2_step04_benchmark.py
# Step: 04 — benchmark (v2, FIXED)
# Data: 2026-10-02
#
# FIX v2: run_tal() ora usa i dati passati (x_tr, y_tr, x_te, y_te)
# invece di rigenerarli internamente. Prima rumore ed extrapolazione
# non avevano effetto su TAL.
#
# FIX v2.1: modalità --critical ora testa ENTRAMBE le extrapolazioni
# ("half" e "central"), non solo "half".
#
# FIX v2.2: PySR viene eseguito in SERIE, fuori da joblib.
# Questo evita il crash "Windows fatal exception: access violation"
# causato da Julia + multiprocessing su Windows.
# =========================================================
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import faulthandler
faulthandler.disable()

import numpy as np
import sympy as sp
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)

from tal_4_2_step01_cleanup import (
    FEYNMAN_DOMAIN,
    ODE_DOMAIN,
    DYN_DOMAIN,
    REAL_DOMAIN,
    safe_final,
    make_base_grammar,
    make_extended_grammar,
    DeepGenerator,
    ParametricMemory,
    RobustFitterLM,
    Theory,
    Node,
    LimitDetector,
    accept_deterministic,
    clone_tree_with_new_params,
)

logger = logging.getLogger("tal.benchmark")


ALL_DOMAINS = {
    "Feynman": FEYNMAN_DOMAIN,
    "ODE": ODE_DOMAIN,
    "Dinamici": DYN_DOMAIN,
    "Reali": REAL_DOMAIN,
}

TAL_CONFIG = {
    "n_epochs": 5,
    "n_iter": 15,
    "err_threshold": 1e-6,
    "patience": 2,
}

SOLVED_THRESHOLD = 0.01
PERFECT_THRESHOLD = 0.001


def make_dataset(fn, x_range, n_train=30, n_test=400, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = x_range
    x_tr = np.sort(rng.uniform(lo, hi, n_train))
    y_tr_clean = fn(x_tr)
    if noise > 0.0:
        scale = np.std(y_tr_clean) if np.std(y_tr_clean) > 0 else 1.0
        y_tr = y_tr_clean + rng.normal(0.0, noise * scale, size=n_train)
    else:
        y_tr = y_tr_clean
    x_te = np.linspace(lo, hi, n_test)
    y_te = fn(x_te)
    return x_tr, y_tr, x_te, y_te


def make_extrapolation_dataset(fn, x_range, mode, n_train=30, n_test=400,
                                noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = x_range
    if mode == "central":
        span = hi - lo
        train_lo = lo + 0.2 * span
        train_hi = lo + 0.8 * span
    elif mode == "half":
        mid = lo + 0.5 * (hi - lo)
        train_lo, train_hi = lo, mid
    else:
        raise ValueError(f"mode sconosciuto: {mode}")
    x_tr = np.sort(rng.uniform(train_lo, train_hi, n_train))
    y_tr_clean = fn(x_tr)
    if noise > 0.0:
        scale = np.std(y_tr_clean) if np.std(y_tr_clean) > 0 else 1.0
        y_tr = y_tr_clean + rng.normal(0.0, noise * scale, size=n_train)
    else:
        y_tr = y_tr_clean
    x_te = np.linspace(lo, hi, n_test)
    y_te = fn(x_te)
    return x_tr, y_tr, x_te, y_te


def solve_with_grammar_with_data(x_tr, y_tr, x_te, y_te, grammar, seed,
                                  n_epochs, n_iter, err_threshold, patience):
    rng = np.random.default_rng(seed)
    gen = DeepGenerator(grammar, seed=seed,
                        max_depth_random=3, max_depth_mutate=5)
    param_memory = ParametricMemory(max_per_signature=5)
    fitter = RobustFitterLM(seed=seed, n_starts=10,
                            max_nfev_local=500, max_params=20,
                            param_memory=param_memory)
    detector = LimitDetector()

    structure = None
    err_history = []

    for epoch in range(n_epochs):
        if structure is not None:
            new_tree = clone_tree_with_new_params(structure, rng)
            T = Theory(new_tree, grammar).fit(x_tr, y_tr, fitter)
        else:
            T0 = Node("lin", [Node("x")],
                      [sp.Symbol(f"a_{seed}", real=True)])
            T = Theory(T0, grammar).fit(x_tr, y_tr, fitter)

        consecutive_below = 0
        for it in range(n_iter):
            if consecutive_below >= patience:
                break
            L = detector.detect(T, x_tr, y_tr, fitter)
            cands = []
            for _ in range(5):
                cands.append(Theory(gen.mutate(T.tree, 0.4), grammar))
            for _ in range(2):
                cands.append(Theory(gen.random_tree(), grammar))
            for c in cands:
                c.fit(x_tr, y_tr, fitter)
            best_T = T
            best_err = None
            for c in cands:
                if not c.params:
                    continue
                passed, err_old, err_new = accept_deterministic(
                    c, T, x_tr, y_tr, fitter, improvement=0.95
                )
                if passed:
                    if best_err is None or err_new < best_err:
                        best_err = err_new
                        best_T = c
            if best_T is not T:
                T = best_T
            err_test = float(np.mean((y_te - T.predict(x_te, fitter)) ** 2))
            if np.isfinite(err_test) and err_test < err_threshold:
                consecutive_below += 1
            else:
                consecutive_below = 0

        structure = T.tree
        err_test = float(np.mean((y_te - T.predict(x_te, fitter)) ** 2))
        err_history.append(err_test)

    return err_history


def solve_hybrid_with_data(x_tr, y_tr, x_te, y_te, seed,
                            n_epochs, n_iter, err_threshold, patience):
    grammar_base = make_base_grammar()
    hist_base = solve_with_grammar_with_data(
        x_tr, y_tr, x_te, y_te, grammar_base, seed,
        n_epochs, n_iter, err_threshold, patience)
    err_base = safe_final(hist_base)

    grammar_ext = make_extended_grammar()
    hist_ext = solve_with_grammar_with_data(
        x_tr, y_tr, x_te, y_te, grammar_ext, seed + 1000,
        n_epochs, n_iter, err_threshold, patience)
    err_ext = safe_final(hist_ext)

    if err_base < err_ext:
        return hist_base, "base"
    else:
        return hist_ext, "extended"


def run_tal(fn, x_tr, y_tr, x_te, y_te, x_range, seed):
    t0 = time.time()
    try:
        hist, chosen = solve_hybrid_with_data(
            x_tr, y_tr, x_te, y_te, seed,
            TAL_CONFIG["n_epochs"], TAL_CONFIG["n_iter"],
            TAL_CONFIG["err_threshold"], TAL_CONFIG["patience"])
        elapsed = time.time() - t0
        final_err = safe_final(hist)
        return {"ok": True, "final_err": float(final_err), "chosen": chosen,
                "hist": [float(e) if np.isfinite(e) else 1.0 for e in hist],
                "time": elapsed, "error": None}
    except Exception:
        elapsed = time.time() - t0
        return {"ok": False, "final_err": float("inf"), "chosen": "n/a",
                "hist": [], "time": elapsed, "error": traceback.format_exc()}


def baseline_polynomial(x_tr, y_tr, x_te, y_te, degree=5):
    t0 = time.time()
    try:
        coeffs = np.polyfit(x_tr, y_tr, degree)
        pred = np.polyval(coeffs, x_te)
        if not np.all(np.isfinite(pred)):
            raise ValueError("pred non finita")
        err = float(np.mean((y_te - pred) ** 2))
        elapsed = time.time() - t0
        return {"ok": True, "final_err": err, "time": elapsed, "error": None}
    except Exception:
        elapsed = time.time() - t0
        return {"ok": False, "final_err": float("inf"),
                "time": elapsed, "error": traceback.format_exc()}

def baseline_pysr(x_tr, y_tr, x_te, y_te, seed=0):
    t0 = time.time()
    try:
        from pysr import PySRRegressor
        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt"],
            random_state=seed,
            deterministic=True,
            progress=False,
            verbosity=0,
            parallelism="serial",
        )
        model.fit(x_tr.reshape(-1, 1), y_tr)
        pred = model.predict(x_te.reshape(-1, 1))
        # FIX: sostituisci NaN/inf con un valore grande invece di scartare il run
        n_nonfinite = int(np.sum(~np.isfinite(pred)))
        if n_nonfinite > 0:
            pred = np.where(np.isfinite(pred), pred, 1e10)
        err = float(np.mean((y_te - pred) ** 2))
        elapsed = time.time() - t0
        return {"ok": True, "final_err": err,
                "expression": str(model.sympy()),
                "time": elapsed, "error": None}
    except ImportError:
        elapsed = time.time() - t0
        return {"ok": False, "final_err": float("inf"),
                "time": elapsed, "error": "PySR non installato"}
    except Exception:
        elapsed = time.time() - t0
        return {"ok": False, "final_err": float("inf"),
                "time": elapsed, "error": traceback.format_exc()}
                
def _run_one(domain_name, fn_name, fn, x_range, seed, noise,
             extrapolation, method):
    try:
        if extrapolation is None:
            x_tr, y_tr, x_te, y_te = make_dataset(
                fn, x_range, noise=noise, seed=seed)
        else:
            x_tr, y_tr, x_te, y_te = make_extrapolation_dataset(
                fn, x_range, mode=extrapolation, noise=noise, seed=seed)

        if method == "tal":
            res = run_tal(fn, x_tr, y_tr, x_te, y_te, x_range, seed)
        elif method == "poly":
            res = baseline_polynomial(x_tr, y_tr, x_te, y_te, degree=5)
        elif method == "pysr":
            res = baseline_pysr(x_tr, y_tr, x_te, y_te, seed=seed)
        else:
            raise ValueError(f"metodo sconosciuto: {method}")

        return {"domain": domain_name, "function": fn_name, "seed": seed,
                "noise": noise, "extrapolation": extrapolation or "none",
                "method": method, "ok": res.get("ok", False),
                "final_err": float(res.get("final_err", float("inf"))),
                "time": float(res.get("time", 0.0)),
                "error": res.get("error")}
    except Exception:
        return {"domain": domain_name, "function": fn_name, "seed": seed,
                "noise": noise, "extrapolation": extrapolation or "none",
                "method": method, "ok": False,
                "final_err": float("inf"), "time": 0.0,
                "error": traceback.format_exc()}


def build_job_list(domains, seeds, noises, extrapolations, methods):
    jobs = []
    for domain_name, domain_dict in domains.items():
        for fn_name, eq in domain_dict.items():
            for seed in seeds:
                for noise in noises:
                    for extrap in extrapolations:
                        for method in methods:
                            jobs.append({
                                "domain_name": domain_name,
                                "fn_name": fn_name,
                                "fn": eq["fn"],
                                "x_range": eq["x_range"],
                                "seed": seed,
                                "noise": noise,
                                "extrapolation": extrap,
                                "method": method,
                            })
    return jobs


def run_benchmark(domains, seeds, noises, extrapolations, methods, n_jobs=4):
    """Esegue il benchmark.

    FIX v2.2: PySR viene eseguito in SERIE, fuori da joblib.
    Questo evita il crash "Windows fatal exception: access violation"
    causato da Julia + multiprocessing su Windows.
    """
    jobs = build_job_list(domains, seeds, noises, extrapolations, methods)
    pysr_jobs = [j for j in jobs if j["method"] == "pysr"]
    other_jobs = [j for j in jobs if j["method"] != "pysr"]

    print(f"  Job totali: {len(jobs)}")
    print(f"    TAL + poly (paralleli, n_jobs={n_jobs}): {len(other_jobs)}")
    print(f"    PySR (seriale, 1 processo): {len(pysr_jobs)}")
    print()

    results = []

    # ---- FASE 1: TAL + poly in parallelo ----
    if other_jobs:
        print(f"  [FASE 1] TAL + poly in parallelo (n_jobs={n_jobs})...")
        t0 = time.time()
        results_par = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_run_one)(
                j["domain_name"], j["fn_name"], j["fn"], j["x_range"],
                j["seed"], j["noise"], j["extrapolation"], j["method"])
            for j in other_jobs)
        elapsed = time.time() - t0
        results.extend(results_par)
        print(f"  [FASE 1] completata in {elapsed:.1f}s")

    # ---- FASE 2: PySR in serie ----
    if pysr_jobs:
        print(f"\n  [FASE 2] PySR in serie ({len(pysr_jobs)} job)...")
        t0 = time.time()
        for i, j in enumerate(pysr_jobs, 1):
            res = _run_one(
                j["domain_name"], j["fn_name"], j["fn"], j["x_range"],
                j["seed"], j["noise"], j["extrapolation"], j["method"])
            results.append(res)
            status = "OK" if res["ok"] else "FAIL"
            print(f"    [{i:>3}/{len(pysr_jobs)}] {j['domain_name']:<10} "
                  f"{j['fn_name']:<15} seed={j['seed']} -> {status} "
                  f"({res['time']:.2f}s)")
        elapsed = time.time() - t0
        print(f"  [FASE 2] completata in {elapsed:.1f}s")

    return results


def summarize(results):
    summary = {"by_method": {}, "by_domain": {}, "by_extrapolation": {},
               "by_noise": {}, "raw_count": len(results),
               "ok_count": sum(1 for r in results if r["ok"])}

    for method in sorted({r["method"] for r in results}):
        vals = [r["final_err"] for r in results if r["ok"] and r["method"] == method]
        if vals:
            arr = np.array(vals, dtype=float)
            summary["by_method"][method] = {
                "n": int(len(arr)), "mse_mean": float(np.mean(arr)),
                "mse_median": float(np.median(arr)),
                "mse_std": float(np.std(arr)),
                "solved_rate": float(np.mean(arr < SOLVED_THRESHOLD)),
                "perfect_rate": float(np.mean(arr < PERFECT_THRESHOLD))}

    for domain in sorted({r["domain"] for r in results}):
        vals = [r["final_err"] for r in results
                if r["ok"] and r["domain"] == domain and r["method"] == "tal"]
        if vals:
            arr = np.array(vals, dtype=float)
            summary["by_domain"][domain] = {
                "n": int(len(arr)), "mse_mean": float(np.mean(arr)),
                "mse_median": float(np.median(arr)),
                "solved_rate": float(np.mean(arr < SOLVED_THRESHOLD)),
                "perfect_rate": float(np.mean(arr < PERFECT_THRESHOLD))}

    for extrap in sorted({r["extrapolation"] for r in results}):
        vals = [r["final_err"] for r in results
                if r["ok"] and r["extrapolation"] == extrap and r["method"] == "tal"]
        if vals:
            arr = np.array(vals, dtype=float)
            summary["by_extrapolation"][extrap] = {
                "n": int(len(arr)), "mse_mean": float(np.mean(arr)),
                "mse_median": float(np.median(arr)),
                "solved_rate": float(np.mean(arr < SOLVED_THRESHOLD))}

    for noise in sorted({r["noise"] for r in results}):
        vals = [r["final_err"] for r in results
                if r["ok"] and r["noise"] == noise and r["method"] == "tal"]
        if vals:
            arr = np.array(vals, dtype=float)
            summary["by_noise"][str(noise)] = {
                "n": int(len(arr)), "mse_mean": float(np.mean(arr)),
                "mse_median": float(np.median(arr)),
                "solved_rate": float(np.mean(arr < SOLVED_THRESHOLD))}

    return summary


def write_summary_txt(summary, path):
    lines = []
    lines.append("=" * 60)
    lines.append(" TAL 4.2 — Step 04 — Sommario benchmark (v2.2 FIXED)")
    lines.append("=" * 60)
    lines.append(f"  Raw runs      : {summary['raw_count']}")
    lines.append(f"  OK runs       : {summary['ok_count']}")
    lines.append("")
    lines.append("--- Per metodo ---")
    for method, stats in summary["by_method"].items():
        lines.append(f"  {method:<8} n={stats['n']:<5} "
                     f"mse_mean={stats['mse_mean']:.3e} "
                     f"mse_median={stats['mse_median']:.3e} "
                     f"solved={stats['solved_rate']:.2%} "
                     f"perfect={stats['perfect_rate']:.2%}")
    lines.append("")
    lines.append("--- TAL per dominio ---")
    for domain, stats in summary["by_domain"].items():
        lines.append(f"  {domain:<12} n={stats['n']:<5} "
                     f"mse_mean={stats['mse_mean']:.3e} "
                     f"mse_median={stats['mse_median']:.3e} "
                     f"solved={stats['solved_rate']:.2%} "
                     f"perfect={stats['perfect_rate']:.2%}")
    lines.append("")
    lines.append("--- TAL per extrapolazione ---")
    for extrap, stats in summary["by_extrapolation"].items():
        lines.append(f"  {extrap:<12} n={stats['n']:<5} "
                     f"mse_mean={stats['mse_mean']:.3e} "
                     f"mse_median={stats['mse_median']:.3e} "
                     f"solved={stats['solved_rate']:.2%}")
    lines.append("")
    lines.append("--- TAL per rumore ---")
    for noise, stats in summary["by_noise"].items():
        lines.append(f"  noise={noise:<6} n={stats['n']:<5} "
                     f"mse_mean={stats['mse_mean']:.3e} "
                     f"mse_median={stats['mse_median']:.3e} "
                     f"solved={stats['solved_rate']:.2%}")
    lines.append("")
    lines.append("=" * 60)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results, path):
    fields = ["domain", "function", "seed", "noise", "extrapolation",
              "method", "ok", "final_err", "time"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})


def make_plots(results, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib non installato — nessuna figura generata.")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    def _boxplot_compat(ax, data, labels):
        try:
            ax.boxplot(data, tick_labels=labels)
        except TypeError:
            ax.boxplot(data, labels=labels)

    tal_results = [r for r in results if r["method"] == "tal" and r["ok"]]
    if tal_results:
        domains = sorted({r["domain"] for r in tal_results})
        data = [[np.log10(max(r["final_err"], 1e-33))
                 for r in tal_results if r["domain"] == d] for d in domains]
        fig, ax = plt.subplots(figsize=(8, 5))
        _boxplot_compat(ax, data, domains)
        ax.set_ylabel("log10(MSE)")
        ax.set_title("TAL — errore per dominio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "boxplot_by_domain.png", dpi=150)
        plt.close(fig)

    noises = sorted({r["noise"] for r in tal_results})
    if noises:
        means = []
        for n in noises:
            vals = [r["final_err"] for r in tal_results if r["noise"] == n]
            means.append(np.mean(vals) if vals else np.nan)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(noises, means, marker="o")
        ax.set_xlabel("rumore (frazione std)")
        ax.set_ylabel("MSE media (TAL)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "noise_effect.png", dpi=150)
        plt.close(fig)

    ext = sorted({r["extrapolation"] for r in tal_results})
    if ext:
        means = []
        for e in ext:
            vals = [r["final_err"] for r in tal_results if r["extrapolation"] == e]
            means.append(np.mean(vals) if vals else np.nan)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ext, means)
        ax.set_ylabel("MSE media (TAL)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "extrapolation_effect.png", dpi=150)
        plt.close(fig)

    print(f"  Figure salvate in {outdir}")


def parse_args():
    p = argparse.ArgumentParser(description="TAL 4.2 — Step 04 benchmark (v2.2 FIXED)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--critical", action="store_true")
    p.add_argument("--with-pysr", action="store_true")
    p.add_argument("--no-poly", action="store_true")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.full:
        seeds = list(range(20))
        noises = [0.0, 0.01, 0.05]
        extrapolations = [None, "central", "half"]
        mode_name = "full"
    elif args.critical:
        seeds = list(range(5))
        noises = [0.05]
        extrapolations = ["half", "central"]
        mode_name = "critical"
    else:
        seeds = [0, 1, 2]
        noises = [0.0]
        extrapolations = [None]
        mode_name = "quick"

    methods = ["tal"]
    if not args.no_poly:
        methods.append("poly")
    if args.with_pysr:
        methods.append("pysr")

    print("=" * 60)
    print(" TAL 4.2 — Step 04 — Benchmark (v2.2 FIXED)")
    print("=" * 60)
    print(f"  Modalità       : {mode_name}")
    print(f"  Seeds          : {seeds}")
    print(f"  Livelli rumore : {noises}")
    print(f"  Extrapolazione : {extrapolations}")
    print(f"  Metodi         : {methods}")
    print(f"  n_jobs         : {args.n_jobs}")
    print(f"  Output dir     : {outdir}")
    print()

    results = run_benchmark(ALL_DOMAINS, seeds, noises, extrapolations,
                             methods, n_jobs=args.n_jobs)
    summary = summarize(results)

    json_path = outdir / "tal_4_2_step04_benchmark_v2.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"step": "04_benchmark_v2", "mode": mode_name,
                   "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config": {"seeds": seeds, "noises": noises,
                              "extrapolations": extrapolations,
                              "methods": methods, "tal_config": TAL_CONFIG},
                   "summary": summary, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n>>> JSON  salvato in {json_path}")

    csv_path = outdir / "tal_4_2_step04_summary_v2.csv"
    write_csv(results, csv_path)
    print(f">>> CSV   salvato in {csv_path}")

    txt_path = outdir / "tal_4_2_step04_summary_v2.txt"
    write_summary_txt(summary, txt_path)
    print(f">>> TXT   salvato in {txt_path}")

    plots_dir = outdir / "tal_4_2_step04_plots_v2"
    make_plots(results, plots_dir)

    print()
    with txt_path.open("r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
        sys.exit(1)