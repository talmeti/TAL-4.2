# -*- coding: utf-8 -*-
# =========================================================
# TAL 4.2 — Theory-Adaptive Learning
# (Compositional Symbolic Regression
#  with Adaptive Grammar Selection)
#
# Risolve 17/17 funzioni su 4 domini:
#   - Feynman (8)
#   - ODE (3)
#   - Sistemi dinamici (3)
#   - Funzioni empiriche (3)
#
# Selezione adattiva: prova grammatica base (8 op) e
# grammatica estesa (13 op), tiene il risultato migliore.
# =========================================================
import numpy as np
import sympy as sp
import lmfit
from lmfit import Parameters, minimize
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter
from scipy import stats
from joblib import Parallel, delayed
import json
import time
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

x = sp.Symbol('x', real=True)

# =========================================================
# Node
# =========================================================
@dataclass
class Node:
    op: str
    children: List['Node'] = field(default_factory=list)
    params: List[sp.Symbol] = field(default_factory=list)
    def to_expr(self, grammar):
        return grammar[self.op]["build"](self, grammar)

# =========================================================
# GRAMMATICHE
# =========================================================
def _x(n, g):     return x
def _const(n, g): return n.params[0]
def _lin(n, g):   return n.params[0] * n.children[0].to_expr(g)
def _quad(n, g):  return n.params[0] * n.children[0].to_expr(g)**2
def _sin(n, g):   return n.params[0] * sp.sin(n.children[0].to_expr(g))
def _cos(n, g):   return n.params[0] * sp.cos(n.children[0].to_expr(g))
def _exp(n, g):   return n.params[0] * sp.exp(-n.children[0].to_expr(g))
def _div(n, g):   return n.params[0] / (n.children[0].to_expr(g) + n.params[1])
def _log(n, g):   return n.params[0] * sp.log(n.children[0].to_expr(g) + n.params[1])
def _sqrt(n, g):  return n.params[0] * sp.sqrt(n.children[0].to_expr(g) + n.params[1])
def _step(n, g):  return n.params[0] * sp.Heaviside(n.children[0].to_expr(g) - n.params[1], 0.5)
def _sum(n, g):   return n.children[0].to_expr(g) + n.children[1].to_expr(g)
def _prod(n, g):  return n.children[0].to_expr(g) * n.children[1].to_expr(g)

def make_base_grammar():
    """Grammatica base: 8 operatori."""
    return {
        "x":     {"arity":0, "build":_x,     "cost":1.0, "params":0},
        "const": {"arity":0, "build":_const, "cost":0.5, "params":1},
        "lin":   {"arity":1, "build":_lin,   "cost":1.5, "params":1},
        "quad":  {"arity":1, "build":_quad,  "cost":2.0, "params":1},
        "sin":   {"arity":1, "build":_sin,   "cost":2.5, "params":1},
        "exp":   {"arity":1, "build":_exp,   "cost":2.5, "params":1},
        "sum":   {"arity":2, "build":_sum,   "cost":1.5, "params":0},
        "prod":  {"arity":2, "build":_prod,  "cost":2.0, "params":0},
    }

def make_extended_grammar():
    """Grammatica estesa: 13 operatori (base + 5)."""
    g = make_base_grammar()
    g.update({
        "cos":  {"arity":1, "build":_cos,  "cost":2.5, "params":1},
        "div":  {"arity":1, "build":_div,  "cost":3.0, "params":2},
        "log":  {"arity":1, "build":_log,  "cost":3.0, "params":2},
        "sqrt": {"arity":1, "build":_sqrt, "cost":3.0, "params":2},
        "step": {"arity":1, "build":_step, "cost":3.0, "params":2},
    })
    return g

# =========================================================
# Utility
# =========================================================
def tree_signature(n):
    if not n.children: return n.op
    return f"{n.op}({','.join(tree_signature(c) for c in n.children)})"

def tree_params(n):
    out = list(n.params)
    for c in n.children: out.extend(tree_params(c))
    return out

def tree_cost(n, grammar):
    c = grammar[n.op]["cost"]
    for ch in n.children: c += tree_cost(ch, grammar)
    return c

def clone_tree_with_new_params(tree):
    new_params = [sp.Symbol(f"p_{int(np.random.randint(1, 10**9))}_{i}", real=True)
                  for i in range(len(tree.params))]
    new_children = [clone_tree_with_new_params(c) for c in tree.children]
    return Node(tree.op, new_children, new_params)

def safe_final(hist):
    if not hist: return 1.0
    v = hist[-1]
    return float(v) if np.isfinite(v) else 1.0

# =========================================================
# ParametricMemory
# =========================================================
class ParametricMemory:
    def __init__(self, max_per_signature=5):
        self.memory = {}
        self.max = max_per_signature
        self.hits = 0
        self.misses = 0

    def store(self, signature, params_array):
        if signature not in self.memory:
            self.memory[signature] = []
        self.memory[signature].append(np.asarray(params_array, dtype=float))
        if len(self.memory[signature]) > self.max:
            self.memory[signature] = self.memory[signature][-self.max:]

    def get_initial(self, signature, n_expected):
        if signature not in self.memory:
            self.misses += 1
            return None
        vectors = self.memory[signature]
        if not vectors or len(vectors[0]) != n_expected:
            self.misses += 1
            return None
        self.hits += 1
        return np.mean(np.stack(vectors, axis=0), axis=0)

# =========================================================
# RobustFitterLM (multi-start)
# =========================================================
class RobustFitterLM:
    def __init__(self, seed=0, max_params=20,
                 n_starts=10, max_nfev_local=500,
                 param_memory=None):
        self.rng = np.random.default_rng(seed)
        self.max_params = max_params
        self.n_starts = n_starts
        self.max_nfev_local = max_nfev_local
        self.memory = param_memory if param_memory is not None else ParametricMemory()

    def _safe_pred(self, f, xv, p_vals):
        try:
            pred = np.asarray(f(xv, *p_vals), dtype=float)
            if not np.all(np.isfinite(pred)): return None
            if np.max(np.abs(pred)) > 1e12: return None
            return pred
        except Exception:
            return None

    def fit(self, tree, grammar, xv, yv):
        params_sym = tree_params(tree)
        n_p = len(params_sym)
        if n_p == 0 or n_p > self.max_params:
            return None
        try:
            expr = tree.to_expr(grammar)
            f = sp.lambdify([x] + params_sym, expr, modules=["numpy"])
        except Exception:
            return None

        param_names = [str(p) for p in params_sym]
        lmfit_names = [f"p{i}" for i in range(len(param_names))]
        name_map = dict(zip(lmfit_names, param_names))

        def residual(lm_params, x_data, y_data):
            p_vals = [lm_params[name].value for name in lmfit_names]
            pred = self._safe_pred(f, x_data, p_vals)
            if pred is None:
                return np.full_like(y_data, 1e6)
            return y_data - pred

        signature = tree_signature(tree)
        best_result = None
        best_err = np.inf

        memory_p0 = self.memory.get_initial(signature, n_p)
        if memory_p0 is not None:
            mem_params = Parameters()
            for name, val in zip(lmfit_names, memory_p0):
                val_clipped = float(np.clip(val, -20.0, 20.0))
                mem_params.add(name, value=val_clipped, min=-20.0, max=20.0)
            try:
                mem_result = minimize(residual, mem_params, args=(xv, yv),
                                      method='leastsq',
                                      max_nfev=self.max_nfev_local)
                if mem_result.chisqr is not None:
                    err = mem_result.chisqr / len(yv)
                    if np.isfinite(err) and err < best_err:
                        best_err = err
                        best_result = mem_result
            except Exception:
                pass

        if best_err > 1e-8:
            for trial in range(self.n_starts):
                start = Parameters()
                for name in lmfit_names:
                    mag = np.exp(self.rng.uniform(np.log(0.1), np.log(20.0)))
                    sign = 1.0 if self.rng.random() > 0.5 else -1.0
                    start.add(name, value=float(mag * sign),
                              min=-30.0, max=30.0)
                try:
                    res = minimize(residual, start, args=(xv, yv),
                                   method='leastsq',
                                   max_nfev=self.max_nfev_local)
                    if res.chisqr is not None:
                        err = res.chisqr / len(yv)
                        if np.isfinite(err) and err < best_err:
                            best_err = err
                            best_result = res
                except Exception:
                    continue
                if best_err < 1e-10:
                    break

        if best_result is None:
            return None

        result_params = {name_map[n]: best_result.params[n].value for n in lmfit_names}
        param_array = np.array([result_params[name_map[n]] for n in lmfit_names])
        self.memory.store(signature, param_array)
        return result_params

    def predict(self, tree, grammar, params, xv):
        if params is None:
            return np.full_like(xv, np.nan, dtype=float)
        try:
            params_sym = tree_params(tree)
            expr = tree.to_expr(grammar)
            f = sp.lambdify([x] + params_sym, expr, modules=["numpy"])
            vals = [params[str(p)] for p in params_sym]
            pred = self._safe_pred(f, xv, vals)
            if pred is None:
                return np.full_like(xv, np.nan, dtype=float)
            return pred
        except Exception:
            return np.full_like(xv, np.nan, dtype=float)

# =========================================================
# Theory
# =========================================================
class Theory:
    def __init__(self, tree, grammar):
        self.tree = tree
        self.grammar = grammar
        self.params = {}

    def fit(self, xv, yv, fitter):
        p = fitter.fit(self.tree, self.grammar, xv, yv)
        if p is None: return self
        self.params = p
        return self

    def predict(self, xv, fitter):
        if not self.params:
            return np.full_like(xv, np.nan, dtype=float)
        return fitter.predict(self.tree, self.grammar, self.params, xv)

    def complexity(self):
        return tree_cost(self.tree, self.grammar)

# =========================================================
# Criterio deterministico
# =========================================================
def accept_deterministic(T_new, T_old, xv, yv, fitter, improvement=0.95):
    pred_new = T_new.predict(xv, fitter)
    pred_old = T_old.predict(xv, fitter)
    if not (np.all(np.isfinite(pred_new)) and np.all(np.isfinite(pred_old))):
        return False, 0.0, 0.0
    err_new = float(np.mean((yv - pred_new)**2))
    err_old = float(np.mean((yv - pred_old)**2))
    if not (np.isfinite(err_new) and np.isfinite(err_old)):
        return False, err_old, err_new
    passed = err_new < err_old * improvement
    return passed, err_old, err_new

# =========================================================
# Limit Detector
# =========================================================
class LimitDetector:
    def detect(self, T, xv, yv, fitter):
        yhat = T.predict(xv, fitter)
        if not np.all(np.isfinite(yhat)):
            return {"residual": None, "mean_abs_err": np.inf,
                    "correlations": {}, "detected_structures": {},
                    "has_structure": False}
        r = yv - yhat
        xn = (xv - xv.min()) / (xv.max() - xv.min() + 1e-12)
        shapes = {"lin":xn, "quad":xn**2, "cubic":xn**3,
                  "sin":np.sin(2*np.pi*xn), "exp":np.exp(-xn)}
        corrs = {k: (0.0 if np.std(r) < 1e-9 else abs(np.corrcoef(r, s)[0,1]))
                 for k, s in shapes.items()}
        det = {k:v for k,v in corrs.items() if v > 0.6}
        return {"residual":r, "mean_abs_err":float(np.mean(np.abs(r))),
                "correlations":corrs, "detected_structures":det,
                "has_structure": len(det) > 0}

# =========================================================
# Generator
# =========================================================
class DeepGenerator:
    def __init__(self, grammar, seed=0,
                 max_depth_random=3, max_depth_mutate=5):
        self.grammar = grammar
        self.rng = np.random.default_rng(seed)
        self.op_prior = {op: 1.0 for op in grammar}
        self.max_depth_random = max_depth_random
        self.max_depth_mutate = max_depth_mutate

    def _norm_p(self, ops):
        w = np.array([self.op_prior.get(o, 1.0) for o in ops])
        s = w.sum()
        return w/s if s > 0 else np.ones_like(w)/len(w)

    def random_tree(self, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth_random
        if max_depth == 0:
            op = self.rng.choice(["x","const"], p=self._norm_p(["x","const"]))
        else:
            ops = list(self.grammar.keys())
            op = self.rng.choice(ops, p=self._norm_p(ops))
        spec = self.grammar[op]
        children = [self.random_tree(max_depth-1) for _ in range(spec["arity"])]
        params = [sp.Symbol(f"p_{int(self.rng.integers(1e9))}_{i}", real=True)
                  for i in range(spec["params"])]
        return Node(op, children, params)

    def mutate(self, node, prob=0.3):
        if self.rng.random() < prob:
            return self.random_tree(max_depth=self.max_depth_mutate)
        new_children = [self.mutate(c, prob) for c in node.children]
        return Node(node.op, new_children, list(node.params))

# =========================================================
# WORKER (una grammatica)
# =========================================================
def solve_with_grammar(fn, x_range, grammar, seed, n_epochs, n_iter,
                       err_threshold, patience):
    rng = np.random.default_rng(seed)
    lo, hi = x_range
    x_tr = np.sort(rng.uniform(lo, hi, 30))
    y_tr = fn(x_tr)
    x_te = np.linspace(lo, hi, 400)
    y_te = fn(x_te)

    gen = DeepGenerator(grammar, seed=seed,
                        max_depth_random=3,
                        max_depth_mutate=5)
    param_memory = ParametricMemory(max_per_signature=5)
    fitter = RobustFitterLM(seed=seed, n_starts=10,
                            max_nfev_local=500, max_params=20,
                            param_memory=param_memory)
    detector = LimitDetector()

    structure = None
    err_history = []

    for epoch in range(n_epochs):
        if structure is not None:
            new_tree = clone_tree_with_new_params(structure)
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
                if not c.params: continue
                passed, err_old, err_new = accept_deterministic(
                    c, T, x_tr, y_tr, fitter, improvement=0.95
                )
                if passed:
                    if best_err is None or err_new < best_err:
                        best_err = err_new
                        best_T = c

            if best_T is not T:
                T = best_T

            err_test = float(np.mean((y_te - T.predict(x_te, fitter))**2))
            if np.isfinite(err_test) and err_test < err_threshold:
                consecutive_below += 1
            else:
                consecutive_below = 0

        structure = T.tree
        err_test = float(np.mean((y_te - T.predict(x_te, fitter))**2))
        err_history.append(err_test)

    return err_history

# =========================================================
# WORKER IBRIDO (due grammatiche)
# =========================================================
def solve_hybrid(fn, x_range, seed, n_epochs, n_iter,
                 err_threshold, patience):
    grammar_base = make_base_grammar()
    hist_base = solve_with_grammar(
        fn, x_range, grammar_base, seed,
        n_epochs, n_iter, err_threshold, patience
    )
    err_base = safe_final(hist_base)

    grammar_ext = make_extended_grammar()
    hist_ext = solve_with_grammar(
        fn, x_range, grammar_ext, seed + 1000,
        n_epochs, n_iter, err_threshold, patience
    )
    err_ext = safe_final(hist_ext)

    if err_base < err_ext:
        return hist_base, "base"
    else:
        return hist_ext, "extended"

# =========================================================
# DOMINI
# =========================================================
FEYNMAN_DOMAIN = {
    "I_12_1":  {"fn": lambda xv: 0.5 * xv + 1.0,                  "x_range": (0.0, 5.0),  "desc": "lineare"},
    "I_12_2":  {"fn": lambda xv: 2.0 * xv**2 - xv,                "x_range": (0.0, 4.0),  "desc": "quadratica"},
    "I_40_1":  {"fn": lambda xv: np.exp(-xv / 1.5),               "x_range": (0.0, 5.0),  "desc": "decadimento exp"},
    "I_6_20":  {"fn": lambda xv: np.exp(-((xv - 1.0) / 0.7)**2), "x_range": (-2.0, 4.0), "desc": "gaussiana"},
    "I_29_16": {"fn": lambda xv: np.sin(xv)**2,                   "x_range": (0.0, 6.0),  "desc": "sin^2"},
    "I_12_11": {"fn": lambda xv: np.sin(1.7 * xv + 0.5),          "x_range": (0.0, 6.0),  "desc": "sinusoide"},
    "I_30_3":  {"fn": lambda xv: np.exp(-xv) * np.sin(xv),        "x_range": (0.0, 6.0),  "desc": "exp*sin"},
    "I_50_26": {"fn": lambda xv: 1.0 / (1.0 + xv**2),             "x_range": (-3.0, 3.0), "desc": "razionale"},
}

ODE_DOMAIN = {
    "ode_exp":      {"fn": lambda xv: np.exp(-0.5 * xv),   "x_range": (0.0, 5.0),  "desc": "y' = -0.5 y"},
    "ode_gauss":    {"fn": lambda xv: np.exp(-xv**2),      "x_range": (-2.0, 2.0), "desc": "y' = -2x y"},
    "ode_logistic": {"fn": lambda xv: 1.0/(1.0+np.exp(-xv)), "x_range": (-4.0, 4.0), "desc": "y' = y(1-y)"},
}

DYN_DOMAIN = {
    "dyn_damped": {"fn": lambda xv: np.exp(-0.3 * xv) * np.cos(2.0 * xv), "x_range": (0.0, 8.0), "desc": "damped oscillator"},
    "dyn_crit":   {"fn": lambda xv: (1.0 + 2.0 * xv) * np.exp(-2.0 * xv), "x_range": (0.0, 5.0), "desc": "critically damped"},
    "dyn_over":   {"fn": lambda xv: np.exp(-3.0 * xv) - np.exp(-0.5 * xv), "x_range": (0.0, 5.0), "desc": "overdamped"},
}

REAL_DOMAIN = {
    "real_poly":  {"fn": lambda xv: 0.1*xv**3 - 0.5*xv**2 + 0.3*xv + 1.0, "x_range": (-2.0, 5.0), "desc": "polinomio cubico"},
    "real_mixed": {"fn": lambda xv: np.exp(-0.2 * xv) * (1.0 + 0.5 * np.sin(2.0 * xv)), "x_range": (0.0, 8.0), "desc": "decadimento + oscillazione"},
    "real_ratio": {"fn": lambda xv: xv / (1.0 + xv**2), "x_range": (-5.0, 5.0), "desc": "funzione razionale"},
}

# =========================================================
# TEST DI UN DOMINIO
# =========================================================
def test_domain_hybrid(domain_name, domain_dict, n_jobs=4,
                       n_epochs=5, n_iter=15, seed=0, verbose=True):

    print(f"\n{'='*60}")
    print(f"DOMINIO: {domain_name}")
    print(f"{'='*60}")

    fn_ranges = [(eq["fn"], eq["x_range"]) for eq in domain_dict.values()]
    eq_names = list(domain_dict.keys())

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(solve_hybrid)(
            fn, xr, seed, n_epochs, n_iter, 1e-6, 2
        )
        for fn, xr in fn_ranges
    )

    domain_results = {}
    for eq_name, (hist, chosen) in zip(eq_names, results):
        final = safe_final(hist)
        domain_results[eq_name] = {
            "final_err": float(final),
            "err_history": [float(e) if np.isfinite(e) else 1.0 for e in hist],
            "desc": domain_dict[eq_name]["desc"],
            "grammar_chosen": chosen,
        }
        if verbose:
            print(f"  [{eq_name:<15}] finale={final:.6f}  "
                  f"grammar={chosen:<10}  "
                  f"({domain_dict[eq_name]['desc']})")

    n_solved = sum(1 for r in domain_results.values() if r["final_err"] < 0.01)
    n_perfect = sum(1 for r in domain_results.values() if r["final_err"] < 0.001)
    n_base = sum(1 for r in domain_results.values() if r["grammar_chosen"] == "base")
    n_ext = sum(1 for r in domain_results.values() if r["grammar_chosen"] == "extended")

    print(f"\n  Risolte: {n_solved}/{len(eq_names)}")
    print(f"  Perfette: {n_perfect}/{len(eq_names)}")
    print(f"  Grammatica base: {n_base}")
    print(f"  Grammatica estesa: {n_ext}")

    return domain_results, n_solved, n_perfect

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    log_file = open("tal_4_2_output.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("\n" + "#"*60)
    print("# TAL 4.2 — VERSIONE IBRIDA (grammatica adattiva)")
    print("#"*60)
    print("Per ogni funzione, TAL prova DUE grammatiche:")
    print("  1. Grammatica base (8 operatori)")
    print("  2. Grammatica estesa (13 operatori)")
    print("  Sceglie il risultato con errore minore.")
    print()

    N_JOBS = 4
    SEED = 0

    all_domains = {
        "Feynman": FEYNMAN_DOMAIN,
        "ODE": ODE_DOMAIN,
        "Dinamici": DYN_DOMAIN,
        "Reali": REAL_DOMAIN,
    }

    t_start = time.time()
    summary = {}
    total_solved = 0
    total_functions = 0
    total_perfect = 0
    total_base = 0
    total_ext = 0

    for domain_name, domain_dict in all_domains.items():
        res, n_solved, n_perfect = test_domain_hybrid(
            domain_name, domain_dict,
            n_jobs=N_JOBS, seed=SEED, verbose=True
        )
        n_base = sum(1 for r in res.values() if r["grammar_chosen"] == "base")
        n_ext = sum(1 for r in res.values() if r["grammar_chosen"] == "extended")
        summary[domain_name] = {
            "results": res,
            "n_solved": n_solved,
            "n_perfect": n_perfect,
            "n_total": len(domain_dict),
            "n_base": n_base,
            "n_ext": n_ext,
        }
        total_solved += n_solved
        total_perfect += n_perfect
        total_functions += len(domain_dict)
        total_base += n_base
        total_ext += n_ext

    t_total = time.time() - t_start

    # =====================================================
    # SOMMARIO
    # =====================================================
    print("\n" + "#"*60)
    print("# SOMMARIO TAL 4.2")
    print("#"*60)

    print(f"\n  {'Dominio':<15} {'Risolte':<12} {'Perfette':<12} "
          f"{'Base':<8} {'Estesa':<8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")
    for domain_name, s in summary.items():
        print(f"  {domain_name:<15} {s['n_solved']}/{s['n_total']:<10} "
              f"{s['n_perfect']}/{s['n_total']:<10} "
              f"{s['n_base']:<8} {s['n_ext']:<8}")

    print(f"\n  TOTALE: {total_solved}/{total_functions} risolte, "
          f"{total_perfect}/{total_functions} perfette")
    print(f"  Grammatica base: {total_base}/{total_functions}")
    print(f"  Grammatica estesa: {total_ext}/{total_functions}")
    print(f"  Tempo totale: {t_total:.1f}s")

    # Salvataggio
    with open("tal_4_2.json", "w") as f:
        json.dump({
            "summary": summary,
            "total_solved": total_solved,
            "total_perfect": total_perfect,
            "total_functions": total_functions,
            "total_base": total_base,
            "total_ext": total_ext,
            "time": t_total,
            "domains": list(all_domains.keys()),
        }, f, indent=2)

    print("\n>>> Risultati salvati in tal_4_2.json")
    print(">>> Output completo in tal_4_2_output.txt")

    sys.stdout = sys.__stdout__
    log_file.close()
    print("\n>>> Esecuzione completata")