# Optimizer/Markowitz.py
import pandas as pd
import numpy as np
from .portfolio_opt_gurobi import optimize_portfolio_gurobi

def run_markowitz_optimization(results, dana: float):
    """
    Jalankan optimasi portofolio berdasarkan hasil prediksi dari xLSTM.

    Parameters
    ----------
    results : list[dict]
        Setiap dict minimal berisi:
            - 'ticker': str
            - 'future_predictions': list[float] (harga prediksi harian)
            - (opsional) 'expected_return': float
    dana : float
        Total dana investasi pengguna.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        df_alloc : DataFrame berisi bobot portofolio
        summary  : dict berisi ringkasan (expected return, risk, Sharpe, status)
    """

    # Safety check
    if not results or not isinstance(results, list):
        raise ValueError("Results dari prediksi xLSTM tidak valid atau kosong.")

    opt = optimize_portfolio_gurobi(
        results=results,
        dana=dana,
        mode='minvar_target',  # gunakan minimasi risiko dengan target return
        mu_target=0.005,       # target return harian (≈0.1%)
        wmax=1,              # batas maksimum bobot per saham = 100% = 1
        allow_short=False,     # tidak boleh short
        use_log_for_mu=False,  # gunakan simple return untuk interpretasi yang mudah
        use_log_for_cov=True,  # gunakan log-return untuk stabilitas kovarians
    )

    df_alloc = opt["df_alloc"]
    summary = opt["summary"]
    return df_alloc, summary

