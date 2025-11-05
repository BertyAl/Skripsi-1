# Optimizer/Markowitz.py
from .portfolio_opt_gurobi import optimize_portfolio_gurobi

def run_markowitz_optimization(results, dana: float):
    # Safety check
    if not results or not isinstance(results, list):
        raise ValueError("Results dari prediksi xLSTM tidak valid atau kosong.")

    opt = optimize_portfolio_gurobi(
        results=results,
        dana=dana,
        mode='minvar_target',  # gunakan minimasi risiko dengan target return
        mu_target=None,       # target return harian (≈0.5%)
        wmax=1,              # batas maksimum bobot per saham = 100% = 1
        allow_short=False,     # tidak boleh short
        use_log_for_mu=True,  # gunakan  return untuk interpretasi yang mudah
        use_log_for_cov=True,  # gunakan log-return untuk stabilitas kovarians
    )

    df_alloc = opt["df_alloc"]
    summary = opt["summary"]
    return df_alloc, summary

