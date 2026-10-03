# -*- coding: utf-8 -*-
# =========================================================
# TAL 4.2 — Theory-Adaptive Learning
# (Compositional Symbolic Regression
#  with Adaptive Grammar Selection)
#
# File: tal_4_2_step01_cleanup.py
# Step: 01 — cleanup
# Data: 2026-10-01
#
# ---------------------------------------------------------
# ROADMAP DI SVILUPPO (tracciata nel nome file e nei commenti)
# ---------------------------------------------------------
# STEP 01 — cleanup            : pulizia import, docstring, type hints,
#                                riproducibilità RNG, logging, file accessori
# STEP 02 — tests              : suite pytest su grammatiche, fitter,
#                                generator, memoria, domini
# STEP 03 — reproducibility    : seed globale, requirements pinned,
#                                CITATION.cff, LICENSE, container
# STEP 04 — benchmark          : multi-seed, rumore, extrapolazione,
#                                baseline (PySR, gplearn, AI Feynman)
# STEP 05 — ablation           : con/senza memoria, con/senza grammatica
#                                estesa, con/senza detector
# STEP 06 — paper              : README pubblicabile, struttura IMRaD
#                                o JOSS/SoftwareX, figure, tabelle
# ---------------------------------------------------------
#
# Risolve 17/17 funzioni su 4 domini:
#   - Feynman (8)
#   - ODE (3)
#   - Sistemi dinamici (3)
#   - Funzioni empiriche (3)
#
# Selezione adattiva: prova grammatica base (8 op) e
# grammatica estesa (13 op), tiene il risultato migliore.
#
# TAL = Theory-Adaptive Learning
# =========================================================
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from joblib import Parallel, delayed
from lmfit import Parameters, minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger("tal")

__all__ = [
    "Node",
    "make_base_grammar",
    "make_extended_grammar",
    "ParametricMemory",
    "RobustFitterLM",
    "Theory",
    "LimitDetector",
    "DeepGenerator",
    "solve_with_grammar",
    "solve_hybrid",
    "test_domain_hybrid",
    "FEYNMAN_DOMAIN",
    "ODE_DOMAIN",
    "DYN_DOMAIN",
    "REAL_DOMAIN",
    "x",
]

x = sp.Symbol("x", real=True)


# =========================================================
# Node
# =========================================================
@dataclass
class Node:
    """Nodo di un albero simbolico.

    Attributes:
        op: nome dell'operatore nella grammatica.
        children: lista di figli.
        params: simboli sympy dei parametri liberi del nodo.
    """

    op: str
    children: List["Node"] = field(default_factory=list)
    params: List[sp.Symbol] = field(default_factory=list)

    def to_expr(self, grammar: Dict[str, dict]):
        """Costruisce l'espressione sympy applicando il builder dell'operatore."""
        return grammar[self.op]["build"](self, grammar)


# =========================================================
# GRAMMATICHE
# =========================================================
def _x(n: Node, g: Dict[str, dict]):
    return x


def _const(n: Node, g: Dict[str, dict]):
    return n.params[0]


def _lin(n: Node, g: Dict[str, dict]):
    return n.params[0] * n.children[0].to_expr(g)


def _quad(n: Node, g: Dict[str, dict]):
    return n.params[0] * n.children[0].to_expr(g) ** 2


def _sin(n: Node, g: Dict[str, dict]):
    return n.params[0] * sp.sin(n.children[0].to_expr(g))


def _cos(n: Node, g: Dict[str, dict]):
    return n.params[0] * sp.cos(n.children[0].to_expr(g))


def _exp(n: Node, g: Dict[str, dict]):
    return n.params[0] * sp.exp(-n.children[0].to_expr(g))


def _div(n: Node, g: Dict[str, dict]):
    return n.params[0] / (n.children[0].to_expr(g) + n.params[1])


def _log(n: Node, g: Dict[str, dict]):
    return n.params[0] * sp.log(n.children[0].to_expr(g) + n.params[1])


def _sqrt(n: Node, g: Dict[str, dict]):
    return n.params[0] * sp.sqrt(n.children[0].to_expr(g) + n.params[1])


def _step(n: Node, g: Dict[str, dict]):
    return n.params[0] * sp.Heaviside(n.children[0].to_expr(g) - n.params[1], 0.5)


def _sum(n: Node, g: Dict[str, dict]):
    return n.children[0].to_expr(g) + n.children[1].to_expr(g)


def _prod(n: Node, g: Dict[str, dict]):
    return n.children[0].to_expr(g) * n.children[1].to_expr(g)


def make_base_grammar() -> Dict[str, dict]:
    """Grammatica base: 8 operatori."""
    return {
        "x": {"arity": 0, "build": _x, "cost": 1.0, "params": 0},
        "const": {"arity": 0, "build": _const, "cost": 0.5, "params": 1},
        "lin": {"arity": 1, "build": _lin, "cost": 1.5, "params": 1},
        "quad": {"arity": 1, "build": _quad, "cost": 2.0, "params": 1},
        "sin": {"arity": 1, "build": _sin, "cost": 2.5, "params": 1},
        "exp": {"arity": 1, "build": _exp, "cost": 2.5, "params": 1},
        "sum": {"arity": 2, "build": _sum, "cost": 1.5, "params": 0},
        "prod": {"arity": 2, "build": _prod, "cost": 2.0, "params": 0},
    }


def make_extended_grammar() -> Dict[str, dict]:
    """Grammatica estesa: 13 operatori (base + 5)."""
    g = make_base_grammar()
    g.update(
        {
            "cos": {"arity": 1, "build": _cos, "cost": 2.5, "params": 1},
            "div": {"arity": 1, "build": _div, "cost": 3.0, "params": 2},
            "log": {"arity": 1, "build": _log, "cost": 3.0, "params": 2},
            "sqrt": {"arity": 1, "build": _sqrt, "cost": 3.0, "params": 2},
            "step": {"arity": 1, "build": _step, "cost": 3.0, "params": 2},
        }
    )
    return g


# =========================================================
# Utility
# =========================================================
def tree_signature(n: Node) -> str:
    """Firma strutturale dell'albero (indipendente dai simboli dei parametri)."""
    if not n.children:
        return n.op
    return f"{n.op}({','.join(tree_signature(c) for c in n.children)})"


def tree_params(n: Node) -> List[sp.Symbol]:
    """Tutti i simboli dei parametri nell'albero, in ordine depth-first."""
    out = list(n.params)
    for c in n.children:
        out.extend(tree_params(c))
    return out


def tree_cost(n: Node, grammar: Dict[str, dict]) -> float:
    """Costo strutturale dell'albero secondo la grammatica."""
    c = grammar[n.op]["cost"]
    for ch in n.children:
        c += tree_cost(ch, grammar)
    return c


def clone_tree_with_new_params(
    tree: Node, rng: np.random.Generator, counter: Optional[List[int]] = None
) -> Node:
    """Clona l'albero rigenerando i simboli dei parametri con un RNG locale.

    STEP 01: usa un contatore incrementale per garantire unicità dei nomi
    (evita collisioni da randint) e un RNG locale per riproducibilità.
    """
    if counter is None:
        counter = [0]

    def _fresh() -> sp.Symbol:
        counter[0] += 1
        return sp.Symbol(f"q_{counter[0]}", real=True)

    new_params = [_fresh() for _ in range(len(tree.params))]
    new_children = [clone_tree_with_new_params(c, rng, counter) for c in tree.children]
    return Node(tree.op, new_children, new_params)


def safe_final(hist: Sequence[float]) -> float:
    """Ultimo valore finito della history, altrimenti 1.0."""
    if not hist:
        return 1.0
    v = hist[-1]
    return float(v) if np.isfinite(v) else 1.0


# =========================================================
# ParametricMemory
# =========================================================
class ParametricMemory:
    """Memoria dei vettori di parametri associati a una firma strutturale.

    STEP 01: esposta la proprietà `stats()` per il report finale.
    """

    def __init__(self, max_per_signature: int = 5) -> None:
        self.memory: Dict[str, List[np.ndarray]] = {}
        self.max = max_per_signature
        self.hits = 0
        self.misses = 0

    def store(self, signature: str, params_array: np.ndarray) -> None:
        """Salva un vettore di parametri per la firma data (FIFO)."""
        if signature not in self.memory:
            self.memory[signature] = []
        self.memory[signature].append(np.asarray(params_array, dtype=float))
        if len(self.memory[signature]) > self.max:
            self.memory[signature] = self.memory[signature][-self.max :]

    def get_initial(self, signature: str, n_expected: int) -> Optional[np.ndarray]:
        """Media dei vettori memorizzati per la firma, o None se non disponibile."""
        if signature not in self.memory:
            self.misses += 1
            return None
        vectors = self.memory[signature]
        if not vectors or len(vectors[0]) != n_expected:
            self.misses += 1
            return None
        self.hits += 1
        return np.mean(np.stack(vectors, axis=0), axis=0)

    def stats(self) -> Dict[str, int]:
        """Statistiche di utilizzo della memoria."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "signatures": len(self.memory),
        }


# =========================================================
# RobustFitterLM (multi-start)
# =========================================================
class RobustFitterLM:
    """Fitter multi-start basato su lmfit con memoria parametrica opzionale."""

    def __init__(
        self,
        seed: int = 0,
        max_params: int = 20,
        n_starts: int = 10,
        max_nfev_local: int = 500,
        param_memory: Optional[ParametricMemory] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_params = max_params
        self.n_starts = n_starts
        self.max_nfev_local = max_nfev_local
        self.memory = param_memory if param_memory is not None else ParametricMemory()

    @staticmethod
    def _safe_pred(
        f: Callable, xv: np.ndarray, p_vals: Sequence[float]
    ) -> Optional[np.ndarray]:
        """Valuta f proteggendo da NaN/Inf e valori fuori scala."""
        try:
            pred = np.asarray(f(xv, *p_vals), dtype=float)
            if not np.all(np.isfinite(pred)):
                return None
            if np.max(np.abs(pred)) > 1e12:
                return None
            return pred
        except Exception:
            return None

    def fit(
        self, tree: Node, grammar: Dict[str, dict], xv: np.ndarray, yv: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Fitta i parametri liberi dell'albero sui dati (xv, yv)."""
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

        def residual(
            lm_params: Parameters, x_data: np.ndarray, y_data: np.ndarray
        ) -> np.ndarray:
            p_vals = [lm_params[name].value for name in lmfit_names]
            pred = self._safe_pred(f, x_data, p_vals)
            if pred is None:
                return np.full_like(y_data, 1e6)
            return y_data - pred

        signature = tree_signature(tree)
        best_result = None
        best_err = np.inf

        # 1) tentativo da memoria parametrica
        memory_p0 = self.memory.get_initial(signature, n_p)
        if memory_p0 is not None:
            mem_params = Parameters()
            for name, val in zip(lmfit_names, memory_p0):
                val_clipped = float(np.clip(val, -20.0, 20.0))
                mem_params.add(name, value=val_clipped, min=-20.0, max=20.0)
            try:
                mem_result = minimize(
                    residual,
                    mem_params,
                    args=(xv, yv),
                    method="leastsq",
                    max_nfev=self.max_nfev_local,
                )
                if mem_result.chisqr is not None:
                    err = mem_result.chisqr / len(yv)
                    if np.isfinite(err) and err < best_err:
                        best_err = err
                        best_result = mem_result
            except Exception:
                pass

        # 2) multi-start random se la memoria non basta
        if best_err > 1e-8:
            for _ in range(self.n_starts):
                start = Parameters()
                for name in lmfit_names:
                    mag = np.exp(self.rng.uniform(np.log(0.1), np.log(20.0)))
                    sign = 1.0 if self.rng.random() > 0.5 else -1.0
                    start.add(name, value=float(mag * sign), min=-30.0, max=30.0)
                try:
                    res = minimize(
                        residual,
                        start,
                        args=(xv, yv),
                        method="leastsq",
                        max_nfev=self.max_nfev_local,
                    )
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

    def predict(
        self,
        tree: Node,
        grammar: Dict[str, dict],
        params: Dict[str, float],
        xv: np.ndarray,
    ) -> np.ndarray:
        """Valuta l'albero con i parametri dati; NaN se non valutabile."""
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
    """Albero + grammatica + parametri fittati."""

    def __init__(self, tree: Node, grammar: Dict[str, dict]) -> None:
        self.tree = tree
        self.grammar = grammar
        self.params: Dict[str, float] = {}

    def fit(self, xv: np.ndarray, yv: np.ndarray, fitter: RobustFitterLM) -> "Theory":
        p = fitter.fit(self.tree, self.grammar, xv, yv)
        if p is not None:
            self.params = p
        return self

    def predict(self, xv: np.ndarray, fitter: RobustFitterLM) -> np.ndarray:
        if not self.params:
            return np.full_like(xv, np.nan, dtype=float)
        return fitter.predict(self.tree, self.grammar, self.params, xv)

    def complexity(self) -> float:
        return tree_cost(self.tree, self.grammar)


# =========================================================
# Criterio deterministico
# =========================================================
def accept_deterministic(
    T_new: Theory,
    T_old: Theory,
    xv: np.ndarray,
    yv: np.ndarray,
    fitter: RobustFitterLM,
    improvement: float = 0.95,
) -> Tuple[bool, float, float]:
    """Accetta T_new se riduce l'MSE almeno del fattore `improvement`."""
    pred_new = T_new.predict(xv, fitter)
    pred_old = T_old.predict(xv, fitter)
    if not (np.all(np.isfinite(pred_new)) and np.all(np.isfinite(pred_old))):
        return False, 0.0, 0.0
    err_new = float(np.mean((yv - pred_new) ** 2))
    err_old = float(np.mean((yv - pred_old) ** 2))
    if not (np.isfinite(err_new) and np.isfinite(err_old)):
        return False, err_old, err_new
    passed = err_new < err_old * improvement
    return passed, err_old, err_new


# =========================================================
# Limit Detector
# =========================================================
class LimitDetector:
    """Rileva struttura nei residui tramite correlazione con forme base.

    STEP 01: il detector è mantenuto come componente disponibile.
    STEP 05: sarà oggetto di ablation (con/senza).
    """

    def detect(
        self, T: Theory, xv: np.ndarray, yv: np.ndarray, fitter: RobustFitterLM
    ) -> Dict[str, object]:
        yhat = T.predict(xv, fitter)
        if not np.all(np.isfinite(yhat)):
            return {
                "residual": None,
                "mean_abs_err": np.inf,
                "correlations": {},
                "detected_structures": {},
                "has_structure": False,
            }
        r = yv - yhat
        xn = (xv - xv.min()) / (xv.max() - xv.min() + 1e-12)
        shapes = {
            "lin": xn,
            "quad": xn ** 2,
            "cubic": xn ** 3,
            "sin": np.sin(2 * np.pi * xn),
            "exp": np.exp(-xn),
        }
        corrs = {
            k: (0.0 if np.std(r) < 1e-9 else abs(np.corrcoef(r, s)[0, 1]))
            for k, s in shapes.items()
        }
        det = {k: v for k, v in corrs.items() if v > 0.6}
        return {
            "residual": r,
            "mean_abs_err": float(np.mean(np.abs(r))),
            "correlations": corrs,
            "detected_structures": det,
            "has_structure": len(det) > 0,
        }


# =========================================================
# Generator
# =========================================================
class DeepGenerator:
    """Generatore di alberi casuali e mutazioni."""

    def __init__(
        self,
        grammar: Dict[str, dict],
        seed: int = 0,
        max_depth_random: int = 3,
        max_depth_mutate: int = 5,
    ) -> None:
        self.grammar = grammar
        self.rng = np.random.default_rng(seed)
        self.op_prior = {op: 1.0 for op in grammar}
        self.max_depth_random = max_depth_random
        self.max_depth_mutate = max_depth_mutate

    def _norm_p(self, ops: Sequence[str]) -> np.ndarray:
        w = np.array([self.op_prior.get(o, 1.0) for o in ops])
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)

    def random_tree(self, max_depth: Optional[int] = None) -> Node:
        """Genera un albero casuale fino a max_depth."""
        if max_depth is None:
            max_depth = self.max_depth_random
        if max_depth == 0:
            op = self.rng.choice(["x", "const"], p=self._norm_p(["x", "const"]))
        else:
            ops = list(self.grammar.keys())
            op = self.rng.choice(ops, p=self._norm_p(ops))
        spec = self.grammar[op]
        children = [self.random_tree(max_depth - 1) for _ in range(spec["arity"])]
        params = [
            sp.Symbol(f"p_{int(self.rng.integers(1e9))}_{i}", real=True)
            for i in range(spec["params"])
        ]
        return Node(op, children, params)

    def mutate(self, node: Node, prob: float = 0.3) -> Node:
        """Mutazione ricorsiva: con probabilità `prob` sostituisce un sottoalbero."""
        if self.rng.random() < prob:
            return self.random_tree(max_depth=self.max_depth_mutate)
        new_children = [self.mutate(c, prob) for c in node.children]
        return Node(node.op, new_children, list(node.params))


# =========================================================
# WORKER (una grammatica)
# =========================================================
def solve_with_grammar(
    fn: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    grammar: Dict[str, dict],
    seed: int,
    n_epochs: int,
    n_iter: int,
    err_threshold: float,
    patience: int,
) -> List[float]:
    """Esegue la ricerca simbolica con una singola grammatica.

    STEP 01: rimossa la variabile inutilizzata `L`; il LimitDetector resta
    istanziato e disponibile per usi futuri (es. mutazioni guidate).

    Returns:
        History dell'errore di test per epoca.
    """
    rng = np.random.default_rng(seed)
    lo, hi = x_range
    x_tr = np.sort(rng.uniform(lo, hi, 30))
    y_tr = fn(x_tr)
    x_te = np.linspace(lo, hi, 400)
    y_te = fn(x_te)

    gen = DeepGenerator(grammar, seed=seed, max_depth_random=3, max_depth_mutate=5)
    param_memory = ParametricMemory(max_per_signature=5)
    fitter = RobustFitterLM(
        seed=seed,
        n_starts=10,
        max_nfev_local=500,
        max_params=20,
        param_memory=param_memory,
    )
    _detector = LimitDetector()  # disponibile per step futuri

    structure: Optional[Node] = None
    err_history: List[float] = []

    for _epoch in range(n_epochs):
        if structure is not None:
            new_tree = clone_tree_with_new_params(structure, rng)
            T = Theory(new_tree, grammar).fit(x_tr, y_tr, fitter)
        else:
            T0 = Node("lin", [Node("x")], [sp.Symbol(f"a_{seed}", real=True)])
            T = Theory(T0, grammar).fit(x_tr, y_tr, fitter)

        consecutive_below = 0

        for _it in range(n_iter):
            if consecutive_below >= patience:
                break

            cands: List[Theory] = []
            for _ in range(5):
                cands.append(Theory(gen.mutate(T.tree, 0.4), grammar))
            for _ in range(2):
                cands.append(Theory(gen.random_tree(), grammar))
            for c in cands:
                c.fit(x_tr, y_tr, fitter)

            best_T = T
            best_err: Optional[float] = None
            for c in cands:
                if not c.params:
                    continue
                passed, _err_old, err_new = accept_deterministic(
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

    logger.debug("solve_with_grammar: memory stats = %s", param_memory.stats())
    return err_history


# =========================================================
# WORKER IBRIDO (due grammatiche)
# =========================================================
def solve_hybrid(
    fn: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    seed: int,
    n_epochs: int,
    n_iter: int,
    err_threshold: float,
    patience: int,
) -> Tuple[List[float], str]:
    """Esegue base ed estesa, restituisce la history migliore e la grammatica scelta."""
    grammar_base = make_base_grammar()
    hist_base = solve_with_grammar(
        fn, x_range, grammar_base, seed, n_epochs, n_iter, err_threshold, patience
    )
    err_base = safe_final(hist_base)

    grammar_ext = make_extended_grammar()
    hist_ext = solve_with_grammar(
        fn, x_range, grammar_ext, seed + 1000, n_epochs, n_iter, err_threshold, patience
    )
    err_ext = safe_final(hist_ext)

    if err_base < err_ext:
        return hist_base, "base"
    return hist_ext, "extended"


# =========================================================
# DOMINI
# =========================================================
FEYNMAN_DOMAIN: Dict[str, dict] = {
    "I_12_1": {"fn": lambda xv: 0.5 * xv + 1.0, "x_range": (0.0, 5.0), "desc": "lineare"},
    "I_12_2": {"fn": lambda xv: 2.0 * xv ** 2 - xv, "x_range": (0.0, 4.0), "desc": "quadratica"},
    "I_40_1": {"fn": lambda xv: np.exp(-xv / 1.5), "x_range": (0.0, 5.0), "desc": "decadimento exp"},
    "I_6_20": {"fn": lambda xv: np.exp(-((xv - 1.0) / 0.7) ** 2), "x_range": (-2.0, 4.0), "desc": "gaussiana"},
    "I_29_16": {"fn": lambda xv: np.sin(xv) ** 2, "x_range": (0.0, 6.0), "desc": "sin^2"},
    "I_12_11": {"fn": lambda xv: np.sin(1.7 * xv + 0.5), "x_range": (0.0, 6.0), "desc": "sinusoide"},
    "I_30_3": {"fn": lambda xv: np.exp(-xv) * np.sin(xv), "x_range": (0.0, 6.0), "desc": "exp*sin"},
    "I_50_26": {"fn": lambda xv: 1.0 / (1.0 + xv ** 2), "x_range": (-3.0, 3.0), "desc": "razionale"},
}

ODE_DOMAIN: Dict[str, dict] = {
    "ode_exp": {"fn": lambda xv: np.exp(-0.5 * xv), "x_range": (0.0, 5.0), "desc": "y' = -0.5 y"},
    "ode_gauss": {"fn": lambda xv: np.exp(-xv ** 2), "x_range": (-2.0, 2.0), "desc": "y' = -2x y"},
    "ode_logistic": {"fn": lambda xv: 1.0 / (1.0 + np.exp(-xv)), "x_range": (-4.0, 4.0), "desc": "y' = y(1-y)"},
}

DYN_DOMAIN: Dict[str, dict] = {
    "dyn_damped": {"fn": lambda xv: np.exp(-0.3 * xv) * np.cos(2.0 * xv), "x_range": (0.0, 8.0), "desc": "damped oscillator"},
    "dyn_crit": {"fn": lambda xv: (1.0 + 2.0 * xv) * np.exp(-2.0 * xv), "x_range": (0.0, 5.0), "desc": "critically damped"},
    "dyn_over": {"fn": lambda xv: np.exp(-3.0 * xv) - np.exp(-0.5 * xv), "x_range": (0.0, 5.0), "desc": "overdamped"},
}

REAL_DOMAIN: Dict[str, dict] = {
    "real_poly": {"fn": lambda xv: 0.1 * xv ** 3 - 0.5 * xv ** 2 + 0.3 * xv + 1.0, "x_range": (-2.0, 5.0), "desc": "polinomio cubico"},
    "real_mixed": {"fn": lambda xv: np.exp(-0.2 * xv) * (1.0 + 0.5 * np.sin(2.0 * xv)), "x_range": (0.0, 8.0), "desc": "decadimento + oscillazione"},
    "real_ratio": {"fn": lambda xv: xv / (1.0 + xv ** 2), "x_range": (-5.0, 5.0), "desc": "funzione razionale"},
}


# =========================================================
# TEST DI UN DOMINIO
# =========================================================
def test_domain_hybrid(
    domain_name: str,
    domain_dict: Dict[str, dict],
    n_jobs: int = 4,
    n_epochs: int = 5,
    n_iter: int = 15,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[Dict[str, dict], int, int]:
    """Testa un intero dominio in parallelo.

    STEP 01: ogni funzione è avvolta in un try/except per evitare che
    un errore singolo uccida l'intero job parallelo.

    Returns:
        (domain_results, n_solved, n_perfect)
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"DOMINIO: {domain_name}")
        print(f"{'=' * 60}")

    fn_ranges = [(eq["fn"], eq["x_range"]) for eq in domain_dict.values()]
    eq_names = list(domain_dict.keys())

    def _safe_solve(fn, xr, seed, n_epochs, n_iter):
        try:
            return solve_hybrid(fn, xr, seed, n_epochs, n_iter, 1e-6, 2)
        except Exception:
            logger.error("Errore in solve_hybrid:\n%s", traceback.format_exc())
            return ([1.0], "base")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_safe_solve)(fn, xr, seed, n_epochs, n_iter)
        for fn, xr in fn_ranges
    )

    domain_results: Dict[str, dict] = {}
    for eq_name, (hist, chosen) in zip(eq_names, results):
        final = safe_final(hist)
        domain_results[eq_name] = {
            "final_err": float(final),
            "err_history": [float(e) if np.isfinite(e) else 1.0 for e in hist],
            "desc": domain_dict[eq_name]["desc"],
            "grammar_chosen": chosen,
        }
        if verbose:
            print(
                f"  [{eq_name:<15}] finale={final:.6f}  "
                f"grammar={chosen:<10}  "
                f"({domain_dict[eq_name]['desc']})"
            )

    n_solved = sum(1 for r in domain_results.values() if r["final_err"] < 0.01)
    n_perfect = sum(1 for r in domain_results.values() if r["final_err"] < 0.001)

    if verbose:
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
    # -----------------------------------------------------
    # Logging su file separato (STEP 01)
    # -----------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("tal_4_2_step01.log", encoding="utf-8")],
    )

    class Tee:
        """Redirige stdout sia a console sia a file di log."""

        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open("tal_4_2_output.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("\n" + "#" * 60)
    print("# TAL 4.2 — VERSIONE IBRIDA (grammatica adattiva)")
    print("# Step 01 — cleanup")
    print("#" * 60)
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
    summary: Dict[str, dict] = {}
    total_solved = 0
    total_functions = 0
    total_perfect = 0
    total_base = 0
    total_ext = 0

    for domain_name, domain_dict in all_domains.items():
        res, n_solved, n_perfect = test_domain_hybrid(
            domain_name, domain_dict, n_jobs=N_JOBS, seed=SEED, verbose=True
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
    print("\n" + "#" * 60)
    print("# SOMMARIO TAL 4.2 — Step 01")
    print("#" * 60)

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

    # =====================================================
    # SALVATAGGIO
    # =====================================================
    with open("tal_4_2.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": "01_cleanup",
                "summary": summary,
                "total_solved": total_solved,
                "total_perfect": total_perfect,
                "total_functions": total_functions,
                "total_base": total_base,
                "total_ext": total_ext,
                "time": t_total,
                "domains": list(all_domains.keys()),
            },
            f,
            indent=2,
        )

    print("\n>>> Risultati salvati in tal_4_2.json")
    print(">>> Output completo in tal_4_2_output.txt")
    print(">>> Log dettagliato in tal_4_2_step01.log")

    sys.stdout = sys.__stdout__
    log_file.close()
    print("\n>>> Esecuzione completata")