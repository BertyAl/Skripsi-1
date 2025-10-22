# -*- coding: utf-8 -*-
# portfolio_opt_gurobi.py  —  Markowitz (Continuous & MIQP) + near-PSD
import math
import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    _GUROBI_OK = True
    _GUROBI_ERR = None
except Exception as e:
    _GUROBI_OK = False
    _GUROBI_ERR = e


# =========================
# Utilities
# =========================
def _validate_gurobi():
    if not _GUROBI_OK:
        raise RuntimeError(
            "Gurobi tidak tersedia/berlisensi di environment ini. "
            f"Detail error: {repr(_GUROBI_ERR)}"
        )


def _to_log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if len(prices) < 2:
        return np.array([], dtype=float)
    prices = np.clip(prices, 1e-12, None)
    return np.diff(np.log(prices))


def _expected_return_from_path(prices: np.ndarray, mode: str = "simple") -> float:
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if len(prices) < 2:
        return 0.0
    if mode == "log":
        r = _to_log_returns_from_prices(prices)
        return float(np.mean(r)) if len(r) else 0.0
    r = prices[1:] / np.clip(prices[:-1], 1e-12, None) - 1.0
    return float(np.mean(r)) if len(r) else 0.0


def _covariance_from_paths(paths: dict, use_log: bool = True) -> pd.DataFrame:
    min_len = min(len(v) for v in paths.values() if isinstance(v, (list, np.ndarray)))
    if min_len < 2:
        raise ValueError("future_predictions terlalu pendek untuk membangun kovarians.")
    R = {}
    for tkr, p in paths.items():
        arr = np.asarray(p, dtype=float)[:min_len]
        if use_log:
            r = _to_log_returns_from_prices(arr)
        else:
            r = arr[1:] / np.clip(arr[:-1], 1e-12, None) - 1.0
        R[tkr] = r
    dfR = pd.DataFrame(R)  # shape (T-1, N)
    return dfR.cov()       # sample covariance (N×N)


def _near_psd(S: np.ndarray, lam: float = 0.10, jitter: float = 1e-8) -> np.ndarray:
    """Shrinkage ke diag + jitter + simetrisasi agar PSD/near-PSD."""
    S = np.asarray(S, dtype=float)
    S = (S + S.T) / 2.0
    D = np.diag(np.diag(S))
    S = (1.0 - lam) * S + lam * D
    S = S + jitter * np.eye(S.shape[0])
    S[np.abs(S) < 1e-16] = 0.0
    return (S + S.T) / 2.0


def _mu_bounds(mu: np.ndarray, wmax: float | None, allow_short: bool) -> tuple[float, float]:
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    if allow_short:
        wmax_eff = 1.0 if wmax is None else float(wmax)
    else:
        wmax_eff = 1.0 if wmax is None else min(1.0, float(wmax))
        if wmax_eff * n < 1.0:
            wmax_eff = 1.0 / n

    # μ_max: isi ke aset μ terbesar sampai habis budget
    order_desc = np.argsort(-mu)
    remain = 1.0
    mu_max = 0.0
    for idx in order_desc:
        take = min(wmax_eff, remain)
        mu_max += take * mu[idx]
        remain -= take
        if remain <= 1e-12:
            break

    # μ_min: isi ke aset μ terendah
    order_asc = np.argsort(mu)
    remain = 1.0
    mu_min = 0.0
    for idx in order_asc:
        take = min(wmax_eff, remain)
        mu_min += take * mu[idx]
        remain -= take
        if remain <= 1e-12:
            break
    return float(mu_min), float(mu_max)


# =========================
# Build μ (vector) & Σ (matrix) dari results
# =========================
def build_mu_sigma_from_results(
        results: list,
        use_log_for_mu: bool = False,
        use_log_for_cov: bool = True,
        shrink_lam: float = 0.10,
        jitter: float = 1e-8,
        drop_flat_std_thresh: float = 1e-8,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    results[i] minimal:
      - 'ticker': str
      - 'future_predictions': list[float]
      - (opsional) 'expected_return': float
    """
    clean = []
    for r in results:
        if not isinstance(r, dict):
            continue
        t = r.get("ticker")
        fp = r.get("future_predictions") or r.get("future_values")
        if not t or not isinstance(fp, (list, np.ndarray)) or len(fp) < 2:
            continue
        # buang aset dengan jalur 'flat' (variansi return ~ 0)
        arr = np.asarray(fp, dtype=float)
        rets = np.diff(np.log(np.clip(arr, 1e-12, None)))
        if len(rets) < 2 or np.std(rets) <= drop_flat_std_thresh:
            continue
        clean.append({"ticker": t, "fp": arr, "exp": r.get("expected_return")})

    if not clean:
        raise ValueError("Tidak ada aset valid untuk membangun μ dan Σ.")

    mu_dict, paths = {}, {}
    for obj in clean:
        tkr, fp, exp = obj["ticker"], obj["fp"], obj["exp"]
        paths[tkr] = fp
        if exp is not None:
            mu_dict[tkr] = float(exp)
        else:
            mu_dict[tkr] = _expected_return_from_path(
                fp, mode=("log" if use_log_for_mu else "simple")
            )
    mu = pd.Series(mu_dict).sort_index()

    Sigma = _covariance_from_paths(paths, use_log=use_log_for_cov).reindex(index=mu.index, columns=mu.index)
    Sigma = Sigma.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # near-PSD
    S_np = _near_psd(Sigma.to_numpy(), lam=shrink_lam, jitter=jitter)
    Sigma = pd.DataFrame(S_np, index=mu.index, columns=mu.index)

    return mu, Sigma


# =========================
# Continuous solvers
# =========================
def solve_min_variance_for_target_return(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        mu_target: float,
        wmax: float | None = None,
        allow_short: bool = False,
        obj_scale: float = 1.0,
) -> dict:
    _validate_gurobi()
    mu_v = mu.to_numpy()
    S = Sigma.to_numpy()
    n = len(mu)

    # feasibility guard
    if not allow_short:
        wmax_eff = 1.0 if wmax is None else min(1.0, float(wmax))
        if wmax_eff * n < 1.0:
            wmax_eff = 1.0 / n
    else:
        wmax_eff = 1.0 if wmax is None else float(wmax)

    mu_min, mu_max = _mu_bounds(mu_v, wmax=wmax_eff, allow_short=allow_short)
    mu_tgt = float(np.clip(mu_target, mu_min, mu_max))

    m = gp.Model("MinVar_TargetReturn")
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2  # jaring pengaman numerik

    lb, ub = (-1.0, wmax_eff) if allow_short else (0.0, wmax_eff)
    x = m.addMVar(n, lb=lb, ub=ub, name="x")

    m.addConstr(x.sum() == 1.0, name="budget")
    m.addConstr(mu_v @ x >= mu_tgt, name="min_return")
    m.setObjective(obj_scale * (x @ S @ x), gp.GRB.MINIMIZE)
    m.optimize()

    if m.Status != gp.GRB.OPTIMAL:
        return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}

    w = pd.Series(x.X, index=mu.index, name="weight")
    risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
    eret = float(mu_v @ x.X)
    return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}


def solve_max_return_with_risk_cap(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        sigma_cap: float,
        wmax: float | None = None,
        allow_short: bool = False,
        obj_scale: float = 1.0,
) -> dict:
    _validate_gurobi()
    mu_v = mu.to_numpy()
    S = Sigma.to_numpy()
    n = len(mu)

    if not allow_short:
        wmax_eff = 1.0 if wmax is None else min(1.0, float(wmax))
        if wmax_eff * n < 1.0:
            wmax_eff = 1.0 / n
    else:
        wmax_eff = 1.0 if wmax is None else float(wmax)

    m = gp.Model("MaxRet_RiskCap")
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2

    lb, ub = (-1.0, wmax_eff) if allow_short else (0.0, wmax_eff)
    x = m.addMVar(n, lb=lb, ub=ub, name="x")

    m.addConstr(x.sum() == 1.0, name="budget")
    m.addQConstr(x @ S @ x <= (sigma_cap ** 2), name="risk_cap")
    m.setObjective(obj_scale * (mu_v @ x), gp.GRB.MAXIMIZE)
    m.optimize()

    if m.Status != gp.GRB.OPTIMAL:
        return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}

    w = pd.Series(x.X, index=mu.index, name="weight")
    risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
    eret = float(mu_v @ x.X)
    return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}


# =========================
# MIQP with cardinality (y) and L/U coupling
# =========================
def solve_miqp_minvar_target(
        mu: pd.Series,
        Sigma: pd.DataFrame,
        mu_target: float,
        K: int,
        L: float | np.ndarray | None = None,
        U: float | np.ndarray | None = None,
        obj_scale: float = 1.0,
) -> dict:
    """
    min x' Σ x
    s.t. sum x = 1, μ'x ≥ mu_target,
         sum y ≤ K,
         L_i y_i ≤ x_i ≤ U_i y_i, y_i ∈ {0,1}
    Notes:
      - Long-only (x_i ≥ 0) implicit via L_i ≥ 0.
      - L/U bisa scalar atau array sepanjang n.
    """
    _validate_gurobi()
    mu_v = mu.to_numpy()
    S = Sigma.to_numpy()
    n = len(mu)

    # default L/U
    if L is None:
        L = np.zeros(n)
    if U is None:
        U = np.ones(n)
    L = np.broadcast_to(np.asarray(L, float), (n,))
    U = np.broadcast_to(np.asarray(U, float), (n,))
    # jaga L<=U dan [0,1]
    L = np.clip(L, 0.0, 1.0)
    U = np.clip(U, 0.0, 1.0)
    U = np.maximum(U, L + 1e-12)  # elak L>U

    # clamp mu_target agar feasible secara "budget-only"
    # pakai wmax = max(U) utk bound konservatif
    mu_min, mu_max = _mu_bounds(mu_v, wmax=float(np.max(U)), allow_short=False)
    mu_tgt = float(np.clip(mu_target, mu_min, mu_max))

    m = gp.Model("MIQP_MinVar_Target")
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2

    x = m.addMVar(n, lb=0.0, ub=1.0, name="x")
    y = m.addMVar(n, vtype=gp.GRB.BINARY, name="y")

    # Budget & target return
    m.addConstr(x.sum() == 1.0, name="budget")
    m.addConstr(mu_v @ x >= mu_tgt, name="min_return")

    # Kardinalitas
    m.addConstr(y.sum() <= int(max(1, K)), name="cardinality")

    # L/U coupling: L_i y_i ≤ x_i ≤ U_i y_i
    for i in range(n):
        m.addConstr(L[i] * y[i] <= x[i], name=f"low_{i}")
        m.addConstr(x[i] <= U[i] * y[i], name=f"up_{i}")

    # Objective
    m.setObjective(obj_scale * (x @ S @ x), gp.GRB.MINIMIZE)
    m.optimize()

    if m.Status != gp.GRB.OPTIMAL:
        return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}

    w = pd.Series(x.X, index=mu.index, name="weight")
    risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
    eret = float(mu_v @ x.X)
    return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}


# =========================
# Public API for Flask
# =========================
def optimize_portfolio_gurobi(
        results: list,
        dana: float,
        mode: str = "minvar_target",   # "minvar_target" | "maxret_cap"
        mu_target: float | None = None,
        sigma_cap: float | None = None,
        # Continuous bound:
        wmax: float | None = 0.5,
        allow_short: bool = False,
        # MIQP options (aktif jika K is not None):
        K: int | None = None,
        L: float | np.ndarray | None = None,
        U: float | np.ndarray | None = None,
        # Builder options:
        use_log_for_mu: bool = False,
        use_log_for_cov: bool = True,
        shrink_lam: float = 0.10,
        jitter: float = 1e-8,
        rf_daily: float = 0.03/252,
) -> dict:
    """
    Return:
      {
        'df_alloc': DataFrame[ticker, weight, nominal, exp_return, contrib_return],
        'summary':  {'exp_return': float, 'risk': float, 'sharpe': float, 'status': str}
      }
    """
    mu, Sigma = build_mu_sigma_from_results(
        results,
        use_log_for_mu=use_log_for_mu,
        use_log_for_cov=use_log_for_cov,
        shrink_lam=shrink_lam,
        jitter=jitter,
    )

    # Trivial case: hanya 1 aset
    if len(mu) == 1 and not allow_short and K in (None, 1):
        tkr = mu.index[0]
        w = 1.0
        port_ret = float(mu.iloc[0])
        port_risk = float(np.sqrt(max(Sigma.iloc[0, 0], 0.0)))
        sharpe = (port_ret - rf_daily) / port_risk if port_risk > 0 else np.nan
        df_alloc = pd.DataFrame([{
            "ticker": tkr, "weight": w, "nominal": float(dana),
            "exp_return": port_ret, "contrib_return": port_ret * w
        }])
        return {"df_alloc": df_alloc, "summary": {"exp_return": port_ret, "risk": port_risk, "sharpe": sharpe, "status": "trivial"}}

    # Pilih solver
    if K is not None:
        # MIQP
        if mu_target is None:
            mu_target = float(mu.median())
        sol = solve_miqp_minvar_target(
            mu, Sigma, mu_target=mu_target, K=int(max(1, K)), L=L, U=U
        )
    else:
        # Continuous
        if mode == "minvar_target":
            if mu_target is None:
                mu_target = float(mu.median())
            sol = solve_min_variance_for_target_return(
                mu, Sigma, mu_target=mu_target, wmax=wmax, allow_short=allow_short
            )
        elif mode == "maxret_cap":
            if sigma_cap is None:
                diag_std = np.sqrt(np.maximum(np.diag(Sigma.to_numpy()), 0))
                sigma_cap = float(np.nanmedian(diag_std))
            sol = solve_max_return_with_risk_cap(
                mu, Sigma, sigma_cap=sigma_cap, wmax=wmax, allow_short=allow_short
            )
        else:
            raise ValueError("mode harus 'minvar_target' atau 'maxret_cap'.")

    status = sol["status"]
    w = sol["weights"]
    port_risk = sol["risk"]
    port_ret  = sol["exp_return"]

    if status != "optimal" or w.empty:
        return {
            "df_alloc": pd.DataFrame(columns=["ticker","weight","nominal","exp_return","contrib_return"]),
            "summary":  {"exp_return": math.nan, "risk": math.nan, "sharpe": math.nan, "status": status}
        }

    df_alloc = pd.DataFrame({"ticker": w.index, "weight": w.values})
    df_alloc["nominal"] = df_alloc["weight"] * float(dana)
    df_alloc["exp_return"] = df_alloc["ticker"].map(mu.to_dict())
    df_alloc["contrib_return"] = df_alloc["weight"] * df_alloc["exp_return"]

    sharpe = (port_ret - rf_daily) / port_risk if port_risk > 0 else np.nan

    return {
        "df_alloc": df_alloc.sort_values("weight", ascending=False).reset_index(drop=True),
        "summary":  {"exp_return": port_ret, "risk": port_risk, "sharpe": sharpe, "status": status}
    }
