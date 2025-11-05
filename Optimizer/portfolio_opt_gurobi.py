from __future__ import annotations

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

# mengecek ketersedian lisensi
def _validate_gurobi():
    if not _GUROBI_OK:
        raise RuntimeError(
            "Gurobi tidak tersedia/berlisensi di environment ini. "
            f"Detail error: {repr(_GUROBI_ERR)}"
        )

# mengubah nilai harga menjadi return
def _to_log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if len(prices) < 2:
        return np.array([], dtype=float)
    prices = np.clip(prices, 1e-12, None)
    return np.diff(np.log(prices))

# menghitung nilai dari expected return
def _expected_return_from_path(prices: np.ndarray, mode: str = "simple") -> float:
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if len(prices) < 2:
        return 0.0
    if mode == "log":
        r = _to_log_returns_from_prices(prices)
        return float(np.mean(r)) if len(r) else 0.0
    r = prices[1:] / np.clip(prices[:-1], 1e-12, None) - 1.0
    return float(np.mean(r)) if len(r) else 0.0

# menghitung nilai dari matriks kovarians
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

# menambahkan sedikit nilai kedalam diagonal jika matriks bermasalah
def _near_psd(S: np.ndarray, lam: float = 0.10, jitter: float = 1e-8) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    S = (S + S.T) / 2.0
    D = np.diag(np.diag(S))
    S = (1.0 - lam) * S + lam * D
    S = S + jitter * np.eye(S.shape[0])
    S[np.abs(S) < 1e-16] = 0.0
    return (S + S.T) / 2.0

# menghitung nilai batasan pada expected return
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



# penghubahan data xLSTM menjadi input return dan matriks covarians
# dengan memanggil fungsi sebelumnya
def build_mu_sigma_from_results(
        results: list,
        use_log_for_mu: bool = False,
        use_log_for_cov: bool = True,
        shrink_lam: float = 0.10,
        jitter: float = 1e-8,
        drop_flat_std_thresh: float = 1e-8,
) -> tuple[pd.Series, pd.DataFrame]:
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


#fungsi yang digunakan untuk melakukan optimasi alokasi portofolio
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
    m.Params.OutputFlag = 1
    m.Params.NonConvex = 2  # jaring pengaman numerik

    lb, ub = (-1.0, wmax_eff) if allow_short else (0.0, wmax_eff) #fungsi 30d shortselling
    x = m.addMVar(n, lb=lb, ub=ub, name="x")


    m.addConstr(x.sum() == 1.0, name="budget") #fungsi 30c
    m.addConstr(mu_v @ x >= mu_tgt, name="min_return")#fungsi 30b
    m.setObjective(obj_scale * (x @ S @ x), gp.GRB.MINIMIZE)# fungsi 30a
    m.optimize()

    if m.Status != gp.GRB.OPTIMAL:
        return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}

    w = pd.Series(x.X, index=mu.index, name="weight")
    risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
    eret = float(mu_v @ x.X)
    return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}


def optimize_portfolio_gurobi(
        results: list,
        dana: float,
        mode: str = "minvar_target",
        mu_target: float | None = None,
        # Continuous bound:
        wmax: float | None = 0.5,
        allow_short: bool = False,
        # MIQP options (aktif jika K is not None):
        K: int | None = None,
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
    if len(mu) == 1 and not allow_short and K in (None, 1):
        tkr = mu.index[0]
        w = 1.0
        var_ii = float(Sigma.iloc[0, 0])
        asset_sigma = float(np.sqrt(max(var_ii, 0.0)))
        port_ret = float(mu.iloc[0])
        port_risk = asset_sigma
        sharpe = (port_ret - rf_daily) / port_risk if port_risk > 0 else np.nan

        # konsistenkan kolom dengan non-trivial branch
        df_alloc = pd.DataFrame([{
            "ticker": tkr,
            "weight": w,
            "nominal": float(dana),
            "exp_return": port_ret,
            "contrib_return": port_ret * w,
            "asset_sigma": asset_sigma,       # <-- penting
            "mrc": var_ii * w,                # (Σ w)_i = Σ_ii * 1
            "rc_var": var_ii * (w**2),        # kontribusi ke var portofolio
            "rc_var_pct": 1.0,                # 100%
            "rc_risk": port_risk,             # kontribusi ke risk (σ_p)
            "rc_risk_pct": 1.0,               # 100%
        }])

        return {
            "df_alloc": df_alloc,
            "summary": {"exp_return": port_ret, "risk": port_risk, "sharpe": sharpe, "status": "trivial"}
        }
    if mode == "minvar_target":
        if mu_target is None:
            mu_target = float(mu.median())
        sol = solve_min_variance_for_target_return(
            mu, Sigma, mu_target=mu_target, wmax=wmax, allow_short=allow_short
        )

    status = sol["status"]
    w = sol["weights"]
    port_risk = sol["risk"]
    port_ret  = sol["exp_return"]

    if status != "optimal" or w.empty:
        return {
            "df_alloc": pd.DataFrame(columns=[
                "ticker","weight","nominal","exp_return","contrib_return",
                "asset_sigma","mrc","rc_var","rc_var_pct","rc_risk","rc_risk_pct"
            ]),
            "summary":  {"exp_return": math.nan, "risk": math.nan, "sharpe": math.nan, "status": status}
        }

    # Tabel alokasi
    df_alloc = pd.DataFrame({"ticker": w.index, "weight": w.values})
    df_alloc["nominal"] = df_alloc["weight"] * float(dana)
    df_alloc["exp_return"] = df_alloc["ticker"].map(mu.to_dict())
    df_alloc["contrib_return"] = df_alloc["weight"] * df_alloc["exp_return"]

    # === Risiko per saham & kontribusi risiko ===
    S = Sigma.reindex(index=w.index, columns=w.index).to_numpy()
    wv = df_alloc["weight"].to_numpy()

    # Risiko portofolio (recompute untuk konsistensi)
    port_var = float(wv @ S @ wv)
    port_risk = float(np.sqrt(max(port_var, 0.0)))

    # Volatilitas (risiko) per aset: sqrt(diagonal)
    asset_var = np.diag(S)
    asset_sigma = np.sqrt(np.maximum(asset_var, 0.0))

    # MRC (marginal risk contrib): (Σ w)_i
    mrc = S @ wv

    # RC (variance): w_i * (Σ w)_i  -> sum = port_var
    rc_var = wv * mrc
    rc_var_pct = rc_var / port_var if port_var > 0 else np.full_like(rc_var, np.nan)

    # RC (risk): w_i * (Σ w)_i / sigma_p -> sum = sigma_p
    rc_risk = rc_var / port_risk if port_risk > 0 else np.full_like(rc_var, np.nan)
    rc_risk_pct = rc_risk / port_risk if port_risk > 0 else np.full_like(rc_risk, np.nan)

    df_alloc["asset_sigma"] = asset_sigma              # risiko per saham (harian)
    df_alloc["mrc"] = mrc                              # marginal risk contribution
    df_alloc["rc_var"] = rc_var                        # kontribusi ke VAR portofolio
    df_alloc["rc_var_pct"] = rc_var_pct                # persentase kontribusi VAR
    df_alloc["rc_risk"] = rc_risk                      # kontribusi ke RISK (σ_p)
    df_alloc["rc_risk_pct"] = rc_risk_pct              # persentase kontribusi RISK

    sharpe = (port_ret - rf_daily) / port_risk if port_risk > 0 else np.nan

    return {
        "df_alloc": df_alloc.sort_values("weight", ascending=False).reset_index(drop=True),
        "summary":  {"exp_return": port_ret, "risk": port_risk, "sharpe": sharpe, "status": status}
    }