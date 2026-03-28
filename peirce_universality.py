#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peirce_universality_publishable.py (gepatcht)

Publishable-grade universality pipeline (vollständig, lokal ausführbar)
Patch-Highlights:
- Grid-Validierung (verhindert Null-Abstände)
- Robuste laplace_matrix (kein divide-by-zero)
- Massmatrix-Checks
- Heat-Fallback: LinearOperator + sparse LU (splu) statt inv(M)
- Debug-Logging für kritische Matrixstatistiken
- Sauberer Fallback-Pfad, wenn Heat-Estimation fehlschlägt

Usage:
  python peirce_universality_publishable.py --mode quick --out results.json
"""
import argparse
import json
import time
import math
import sys
from itertools import product
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator, splu

# ------------------ Grid / Operators ------------------
def build_grid(kind, N=400, r_min=1e-6, r_max=10.0):
    if kind == "log":
        return np.geomspace(r_min, r_max, N)
    if kind == "linear":
        return np.linspace(r_min, r_max, N)
    if kind == "mixed":
        n1 = max(10, N // 3)
        r1 = np.geomspace(r_min, 0.1, n1)
        r2 = np.linspace(0.1, r_max, N - n1)
        return np.concatenate((r1, r2))
    raise ValueError("unknown grid kind")

def validate_grid(r):
    # ensure strictly increasing positive grid
    if np.any(np.diff(r) <= 0):
        raise ValueError("Grid must be strictly increasing: check build_grid parameters (found nonpositive spacing).")
    if np.any(r <= 0):
        raise ValueError("Grid contains nonpositive radii; use r_min>0 or handle origin separately.")

def laplace_matrix(r):
    """
    Robust discrete Laplace on non-uniform grid with Dirichlet BC at both ends.
    Avoids division by zero by validating spacing first.
    """
    N = len(r)
    h = np.diff(r)
    if np.any(h <= 0):
        raise ValueError("Nonpositive grid spacing detected in laplace_matrix.")
    data = []
    rows = []
    cols = []
    for i in range(N):
        if i == 0:
            # Dirichlet at left (regularized origin)
            rows.append(i); cols.append(i); data.append(1.0)
        elif i == N - 1:
            # Dirichlet at right
            rows.append(i); cols.append(i); data.append(1.0)
        else:
            h1 = h[i - 1]; h2 = h[i]
            denom = h1 * h2 * (h1 + h2) + 1e-30
            a = 2.0 / (h1 * (h1 + h2))
            b = -2.0 / (h1 * h2)
            c = 2.0 / (h2 * (h1 + h2))
            rows += [i, i, i]; cols += [i - 1, i, i + 1]; data += [a, b, c]
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))

def mass_matrix(r):
    N = len(r)
    w = np.empty(N)
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    for i in range(1, N - 1):
        w[i] = 0.5 * (r[i + 1] - r[i - 1])
    # ensure positivity
    w = np.maximum(w, 1e-30)
    return sp.diags(w, 0, format='csr')

def build_potential(r, r_eps=1e-5, Z=1.0, clamp=None, extra=None):
    # Coulomb-like test potential regularized at origin plus optional extra
    V = -Z / (r + r_eps)
    if extra is not None:
        V = V + np.asarray(extra)
    if clamp is not None:
        V = np.clip(V, -abs(clamp), abs(clamp))
    return V

def build_selfadjoint_operator(r, r_eps=1e-5, Z=1.0, clamp=1e3, extra=None):
    """
    Build generalized eigenproblem L x = lambda M x
    with L = -laplace + diag(V) and M = mass matrix
    """
    L = laplace_matrix(r)
    M = mass_matrix(r)
    V = build_potential(r, r_eps=r_eps, Z=Z, clamp=clamp, extra=extra)
    # Add potential as diagonal (symmetric)
    L = -L + sp.diags(V, 0, format='csr')
    return L.tocsr(), M.tocsr()

# ------------------ Eigenvalue solvers and heat fallback ------------------
def solve_generalized_eig(L, M, k=60):
    """
    Solve generalized eigenproblem L x = lambda M x for smallest algebraic eigenvalues.
    Prefer eigsh for symmetric generalized problem via shift-invert if available.
    """
    try:
        n = L.shape[0]
        k_use = min(k, max(6, n - 2))
        # shift-invert around sigma=0 to get smallest magnitude generalized eigenvalues
        vals = sla.eigsh(L, k=k_use, M=M, sigma=0.0, which='LM', return_eigenvectors=False)
        return np.sort(np.real(vals))
    except Exception:
        try:
            # fallback: form A = M^{-1} L via solving linear systems (avoid explicit inv)
            # Attempt to compute a few eigenvalues of the generalized problem via ARPACK on A
            Minv = None
            try:
                Minv = sla.inv(M.tocsc())
            except Exception:
                Minv = None
            if Minv is not None:
                A = Minv.dot(L)
                n = A.shape[0]
                k_use = min(k, max(6, n - 2))
                vals = sla.eigsh(A, k=k_use, which='SA', return_eigenvectors=False)
                return np.sort(np.real(vals))
            else:
                return None
        except Exception:
            return None

def heat_traces_from_vals(vals, t_list=(1e-2, 1e-3, 1e-4)):
    if vals is None:
        return None
    vals = np.array(vals)
    vals = vals - np.min(vals)  # shift to non-negative
    vals = vals[np.abs(vals) > 1e-12]
    if vals.size == 0:
        return None
    return [float(np.sum(np.exp(-t * vals))) for t in t_list]

def heat_trace_stochastic(L, M, t, m=32):
    """
    Stochastic Hutchinson trace estimation for Tr(exp(-t M^{-1} L))
    Uses expm_multiply on a LinearOperator A defined by matvec = solve(M, L @ v).
    """
    try:
        from scipy.sparse.linalg import expm_multiply
    except Exception:
        expm_multiply = None
    n = L.shape[0]
    # quick NaN/Inf checks
    if np.isnan(L.data).any() or np.isnan(M.data).any() or np.isinf(L.data).any() or np.isinf(M.data).any():
        return None
    # factorize M once (sparse LU). If M is singular, splu will raise.
    try:
        lu = splu(M.tocsc())
    except Exception:
        return None
    def matvec(v):
        Lv = L.dot(v)
        return lu.solve(Lv)
    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    s = 0.0
    for _ in range(m):
        z = np.random.choice([-1.0, 1.0], size=n)
        if expm_multiply is None:
            return None
        try:
            y = expm_multiply(-t * A, z)
        except Exception:
            return None
        s += float(np.dot(z, y))
    return float(s / m)

# ------------------ Signatures and normalization ------------------
def scale_normalize(vals):
    """
    Shift and scale eigenvalues to remove trivial rescaling:
      - shift so min(vals) == 0
      - scale by median(|vals|)
    Returns sorted positive array.
    """
    vals = np.array(vals)
    vals = np.sort(np.real(vals))
    vals = vals - np.min(vals)
    med = np.median(np.abs(vals) + 1e-30)
    if med == 0:
        med = 1.0
    vals = vals / med
    return vals

def eta_signature(vals, s=0.05):
    if vals is None:
        return None
    vals = vals[np.abs(vals) > 1e-12]
    if vals.size == 0:
        return None
    return float(np.sum(np.sign(vals) * (np.abs(vals) ** (-s))))

def ratio_signature(vals):
    if vals is None:
        return None
    if len(vals) < 4:
        return None
    half = len(vals) // 2
    a = np.mean(vals[:half]); b = np.mean(vals[half:])
    if abs(b) < 1e-30:
        return None
    return float(a / b)

def flow_signature(vals):
    if vals is None:
        return None
    if len(vals) < 3:
        return None
    return float(np.mean(np.diff(vals)))

def log_spacing_bulk(vals, frac_lo=0.25, frac_hi=0.75):
    if vals is None:
        return None
    n = len(vals)
    if n < 10:
        return None
    lo = max(1, int(n * frac_lo))
    hi = min(n - 1, int(n * frac_hi))
    bulk = vals[lo:hi]
    if len(bulk) < 3:
        return None
    return float(np.mean(np.diff(np.log(bulk + 1e-30))))

# ------------------ Bootstrap & sensitivity ------------------
def bootstrap_model_eval(model, eval_fn, nboot=40, perturb_scale=1e-5):
    samples = []
    for _ in range(nboot):
        try:
            s = eval_fn(model, perturb=True, perturb_scale=perturb_scale)
            samples.append(s)
        except Exception:
            samples.append(None)
    keys = ["eta", "ratio", "flow", "log_spacing", "heat_t0", "heat_t1", "heat_t2"]
    out = {}
    for k in keys:
        vals = [v.get(k) for v in samples if v and v.get(k) is not None]
        if not vals:
            out[k] = None
        else:
            out[k] = {"lo": float(np.percentile(vals, 2.5)), "med": float(np.percentile(vals, 50)), "hi": float(np.percentile(vals, 97.5))}
    return out

def sensitivity_by_axis(flat_results, axis_values, key):
    vals = np.array([r.get(key) for r in flat_results], dtype=float)
    if np.isnan(vals).all():
        return {ax: None for ax in axis_values.keys()}
    total_var = np.nanvar(vals)
    if total_var == 0 or np.isnan(total_var):
        return {ax: 0.0 for ax in axis_values.keys()}
    sens = {}
    for ax, axvals in axis_values.items():
        groups = defaultdict(list)
        for i, v in enumerate(axvals):
            groups[v].append(vals[i])
        means = [np.nanmean(g) for g in groups.values() if len(g) > 0]
        sens[ax] = float(np.nanvar(means) / (total_var + 1e-30))
    return sens

# ------------------ Model evaluation pipeline ------------------
def evaluate_model(model, k_eig=60, nboot=30, do_bootstrap=True, perturb=False, perturb_scale=1e-5):
    # build and validate grid
    r = build_grid(model["grid"], N=model.get("N", 400))
    try:
        validate_grid(r)
    except Exception as e:
        # return a clear failure record
        return {"eta": None, "ratio": None, "flow": None, "log_spacing": None, "heat": [None, None, None], "vals_len": 0, "error": str(e)}
    # example extra potential (short-range) - replace with production V_extra if available
    V_extra = np.exp(-r * 10.0)
    L, M = build_selfadjoint_operator(r, r_eps=model["r_eps"], Z=model.get("Z", 1.0), clamp=model["diag_clamp"], extra=V_extra)
    # debug logging for matrix stats
    try:
        min_mass = float(M.diagonal().min())
    except Exception:
        min_mass = None
    max_L = float(np.max(np.abs(L.data))) if L.data.size > 0 else None
    # attempt eigen-solve
    vals = solve_generalized_eig(L, M, k=k_eig)
    if vals is None:
        # fallback: try relaxed r_eps
        L2, M2 = build_selfadjoint_operator(r, r_eps=max(1e-6, model["r_eps"] * 10), Z=model.get("Z", 1.0), clamp=model["diag_clamp"], extra=V_extra)
        vals = solve_generalized_eig(L2, M2, k=max(20, k_eig // 2))
        if vals is None:
            # cannot compute eigenvalues: attempt stochastic heat trace fallback and return partial info
            heat = []
            for t in [1e-2, 1e-3, 1e-4]:
                ht = heat_trace_stochastic(L, M, t, m=32)
                heat.append(ht)
            return {"eta": None, "ratio": None, "flow": None, "log_spacing": None, "heat": heat, "vals_len": 0, "min_mass": min_mass, "max_L": max_L}
    # scale-normalize
    vals = scale_normalize(vals)
    # if perturb requested, perturb potential and recompute quickly
    if perturb:
        noise = 1.0 + perturb_scale * np.random.normal(size=len(r))
        Vp = V_extra * noise
        Lp, Mp = build_selfadjoint_operator(r, r_eps=model["r_eps"], Z=model.get("Z", 1.0), clamp=model["diag_clamp"], extra=Vp)
        vals2 = solve_generalized_eig(Lp, Mp, k=max(20, k_eig // 2))
        if vals2 is None:
            return {"eta": None, "ratio": None, "flow": None, "log_spacing": None, "heat": None}
        vals2 = scale_normalize(vals2)
        return {
            "eta": eta_signature(vals2),
            "ratio": ratio_signature(vals2),
            "flow": flow_signature(vals2),
            "log_spacing": log_spacing_bulk(vals2),
            "heat": heat_traces_from_vals(vals2),
            "vals_len": len(vals2),
            "min_mass": min_mass,
            "max_L": max_L
        }
    # normal evaluation
    sig = {}
    sig["eta"] = eta_signature(vals)
    sig["ratio"] = ratio_signature(vals)
    sig["flow"] = flow_signature(vals)
    sig["log_spacing"] = log_spacing_bulk(vals)
    heat = heat_traces_from_vals(vals)
    if heat is None:
        # fallback to stochastic trace on operator
        heat = []
        for t in [1e-2, 1e-3, 1e-4]:
            ht = heat_trace_stochastic(L, M, t, m=32)
            heat.append(ht)
    sig["heat"] = heat
    sig["vals_len"] = len(vals)
    sig["min_mass"] = min_mass
    sig["max_L"] = max_L
    if do_bootstrap:
        sig["bootstrap"] = bootstrap_model_eval(model, evaluate_model, nboot=nboot)
    else:
        sig["bootstrap"] = None
    return sig

# ------------------ Aggregation & classification ------------------
def aggregate_results(flat_results, signature_keys):
    agg = {}
    for key in signature_keys:
        vals = np.array([r.get(key) for r in flat_results], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            agg[key] = {"median": None, "mad": None, "score": None}
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        score = float(mad / (abs(med) + 1e-30)) if med is not None else None
        agg[key] = {"median": med, "mad": mad, "score": score}
    return agg

def classify_score(score, t1=0.05, t2=0.2):
    if score is None:
        return "unknown"
    if score < t1:
        return "universal"
    if score < t2:
        return "relatively_stable"
    return "not_universal"

# ------------------ Model generation ------------------
def generate_models(params):
    keys = list(params.keys())
    combos = list(product(*[params[k] for k in keys]))
    models = []
    for combo in combos:
        model = dict(zip(keys, combo))
        model.setdefault("Z", 1.0)
        models.append(model)
    return models

# ------------------ CLI / Main ------------------
def main():
    parser = argparse.ArgumentParser(description="Peirce universality publishable pipeline (gepatcht)")
    parser.add_argument('--mode', choices=['quick', 'full', 'eta_focus', 'r_eps_sweep', 'heat_debug'], default='quick')
    parser.add_argument('--out', default=r"C:\math\universality_publishable.json")
    parser.add_argument('--nboot', type=int, default=40)
    parser.add_argument('--k', type=int, default=60)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'quick':
        params = {"grid": ["log", "mixed"], "r_eps": [1e-5, 1e-4], "diag_clamp": [1e2, 1e3], "smear_sigma": [0.0], "N": [300]}
        nboot = max(10, args.nboot // 2); k_eig = max(30, args.k // 2)
    elif args.mode == 'eta_focus':
        params = {"grid": ["log", "mixed", "linear"], "r_eps": [1e-6, 1e-5, 1e-4], "diag_clamp": [1e2, 1e3], "smear_sigma": [0.0, 0.005], "N": [400]}
        nboot = max(50, args.nboot); k_eig = max(80, args.k)
    elif args.mode == 'r_eps_sweep':
        params = {"grid": ["mixed"], "r_eps": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "diag_clamp": [1e2], "smear_sigma": [0.0], "N": [400]}
        nboot = max(20, args.nboot // 2); k_eig = max(40, args.k // 2)
    elif args.mode == 'heat_debug':
        params = {"grid": ["mixed"], "r_eps": [1e-5, 1e-4], "diag_clamp": [1e2, 1e3], "smear_sigma": [0.0], "N": [300]}
        nboot = max(10, args.nboot // 2); k_eig = max(60, args.k // 2)
    else:
        params = {"grid": ["log", "linear", "mixed"], "r_eps": [1e-6, 1e-5, 1e-4], "diag_clamp": [1e2, 1e3, None], "smear_sigma": [0.0, 0.005, 0.02], "N": [400, 800]}
        nboot = args.nboot; k_eig = args.k

    models = generate_models(params)
    axis_values = {ax: [] for ax in params.keys()}

    results = []
    start = time.time()
    for i, model in enumerate(models):
        res = evaluate_model(model, k_eig=k_eig, nboot=nboot, do_bootstrap=True)
        results.append(res)
        for ax in params.keys():
            axis_values[ax].append(model[ax])
        if (i + 1) % 5 == 0 or (i + 1) == len(models):
            print(f"[{i+1}/{len(models)}] models evaluated, elapsed {time.time()-start:.1f}s", file=sys.stderr)

    # flatten results using bootstrap medians when available
    flat_results = []
    for r in results:
        entry = {}
        entry["eta"] = r.get("bootstrap", {}).get("eta", {}).get("med") if r.get("bootstrap") and r["bootstrap"].get("eta") else r.get("eta")
        entry["log_spacing"] = r.get("bootstrap", {}).get("log_spacing", {}).get("med") if r.get("bootstrap") and r["bootstrap"].get("log_spacing") else r.get("log_spacing")
        # heat t0..t2
        if r.get("bootstrap") and r["bootstrap"].get("heat_t0"):
            entry["heat_t0"] = r["bootstrap"]["heat_t0"]["med"]; entry["heat_t1"] = r["bootstrap"]["heat_t1"]["med"]; entry["heat_t2"] = r["bootstrap"]["heat_t2"]["med"]
        elif r.get("heat"):
            h = r.get("heat")
            entry["heat_t0"], entry["heat_t1"], entry["heat_t2"] = (h[0] if h and len(h) > 0 else None, h[1] if h and len(h) > 1 else None, h[2] if h and len(h) > 2 else None)
        else:
            entry["heat_t0"] = entry["heat_t1"] = entry["heat_t2"] = None
        flat_results.append(entry)

    signature_keys = ["eta", "log_spacing", "heat_t0", "heat_t1", "heat_t2"]
    agg = aggregate_results(flat_results, signature_keys)
    sens = {k: sensitivity_by_axis(flat_results, axis_values, k) for k in signature_keys}
    classification = {k: classify_score(agg.get(k, {}).get("score")) for k in signature_keys}

    out = {
        "meta": {"mode": args.mode, "n_models": len(models), "nboot": nboot, "k_eig": k_eig, "time_s": time.time() - start},
        "params": params,
        "aggregate": agg,
        "sensitivity": sens,
        "classification": classification,
        "raw_results_count": len(results)
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("Universality pipeline complete. Results written to", args.out)

if __name__ == "__main__":
    main()
