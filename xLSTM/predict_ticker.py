# from __future__ import annotations
# import os, math, json, pickle
# from typing import Any, Dict, List, Tuple, Optional
# from datetime import datetime
#
# import numpy as np
# import pandas as pd
#
# # ---------------------
# # Optional: GUROBI
# # ---------------------
# try:
#     import gurobipy as gp
#     _GUROBI_OK = True
#     _GUROBI_ERR = None
# except Exception as e:
#     _GUROBI_OK = False
#     _GUROBI_ERR = e
#
# # ---------------------
# # Torch & friends
# # ---------------------
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# # ---------------------
# # Data & indicators
# # ---------------------
# import yfinance as yf
# from pandas.tseries.offsets import BDay
# from ta.momentum import RSIIndicator
# from ta.volatility import BollingerBands
#
# # ---------------------
# # Your xLSTM
# # ---------------------
# from xLSTM.xLSTM import xLSTM
#
#
# # =============================================================================
# # Helpers
# # =============================================================================
# def _validate_gurobi():
#     if not _GUROBI_OK:
#         raise RuntimeError(
#             "Gurobi tidak tersedia/berlisensi di environment ini. "
#             f"Detail error: {repr(_GUROBI_ERR)}"
#         )
#
# def _to_log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
#     prices = np.asarray(prices, dtype=float).reshape(-1)
#     if len(prices) < 2:
#         return np.array([], dtype=float)
#     prices = np.clip(prices, 1e-12, None)
#     return np.diff(np.log(prices))
#
# def _expected_return_from_path(prices: np.ndarray, mode: str = "simple") -> float:
#     prices = np.asarray(prices, dtype=float).reshape(-1)
#     if len(prices) < 2:
#         return 0.0
#     if mode == "log":
#         r = _to_log_returns_from_prices(prices)
#         return float(np.mean(r)) if len(r) else 0.0
#     r = prices[1:] / np.clip(prices[:-1], 1e-12, None) - 1.0
#     return float(np.mean(r)) if len(r) else 0.0
#
# def _covariance_from_paths(paths: Dict[str, np.ndarray], use_log: bool = True) -> pd.DataFrame:
#     min_len = min(len(v) for v in paths.values() if isinstance(v, (list, np.ndarray)))
#     if min_len < 2:
#         raise ValueError("future_predictions terlalu pendek untuk membangun kovarians.")
#     R = {}
#     for tkr, p in paths.items():
#         arr = np.asarray(p, dtype=float)[:min_len]
#         if use_log:
#             r = _to_log_returns_from_prices(arr)
#         else:
#             r = arr[1:] / np.clip(arr[:-1], 1e-12, None) - 1.0
#         R[tkr] = r
#     dfR = pd.DataFrame(R)  # (T-1, N)
#     return dfR.cov()
#
# def _near_psd(S: np.ndarray, lam: float = 0.10, jitter: float = 1e-8) -> np.ndarray:
#     S = np.asarray(S, dtype=float)
#     S = (S + S.T) / 2.0
#     D = np.diag(np.diag(S))
#     S = (1.0 - lam) * S + lam * D
#     S = S + jitter * np.eye(S.shape[0])
#     S[np.abs(S) < 1e-16] = 0.0
#     return (S + S.T) / 2.0
#
# def _mu_bounds(mu: np.ndarray, wmax: float | None, allow_short: bool) -> tuple[float, float]:
#     mu = np.asarray(mu, dtype=float)
#     n = len(mu)
#     if allow_short:
#         wmax_eff = 1.0 if wmax is None else float(wmax)
#     else:
#         wmax_eff = 1.0 if wmax is None else min(1.0, float(wmax))
#         if wmax_eff * n < 1.0:
#             wmax_eff = 1.0 / n
#
#     order_desc = np.argsort(-mu)
#     remain = 1.0
#     mu_max = 0.0
#     for idx in order_desc:
#         take = min(wmax_eff, remain)
#         mu_max += take * mu[idx]
#         remain -= take
#         if remain <= 1e-12:
#             break
#
#     order_asc = np.argsort(mu)
#     remain = 1.0
#     mu_min = 0.0
#     for idx in order_asc:
#         take = min(wmax_eff, remain)
#         mu_min += take * mu[idx]
#         remain -= take
#         if remain <= 1e-12:
#             break
#     return float(mu_min), float(mu_max)
#
#
# # =============================================================================
# # Build μ & Σ dari daftar hasil prediksi
# # =============================================================================
# def build_mu_sigma_from_results(
#         results: List[Dict[str, Any]],
#         use_log_for_mu: bool = False,
#         use_log_for_cov: bool = True,
#         shrink_lam: float = 0.10,
#         jitter: float = 1e-8,
#         drop_flat_std_thresh: float = 1e-8,
# ) -> tuple[pd.Series, pd.DataFrame]:
#     clean = []
#     for r in results:
#         if not isinstance(r, dict):
#             continue
#         t = r.get("ticker")
#         fp = r.get("future_values") or r.get("future_predictions")
#         if not t or not isinstance(fp, (list, np.ndarray)) or len(fp) < 2:
#             continue
#         arr = np.asarray(fp, dtype=float)
#         rets = np.diff(np.log(np.clip(arr, 1e-12, None)))
#         if len(rets) < 2 or np.std(rets) <= drop_flat_std_thresh:
#             continue
#         clean.append({"ticker": t, "fp": arr, "exp": r.get("expected_return")})
#
#     if not clean:
#         raise ValueError("Tidak ada aset valid untuk membangun μ dan Σ.")
#
#     mu_dict, paths = {}, {}
#     for obj in clean:
#         tkr, fp, exp = obj["ticker"], obj["fp"], obj["exp"]
#         paths[tkr] = fp
#         if exp is not None:
#             mu_dict[tkr] = float(exp)
#         else:
#             mu_dict[tkr] = _expected_return_from_path(
#                 fp, mode=("log" if use_log_for_mu else "simple")
#             )
#     mu = pd.Series(mu_dict).sort_index()
#
#     Sigma = _covariance_from_paths(paths, use_log=use_log_for_cov).reindex(index=mu.index, columns=mu.index)
#     Sigma = Sigma.replace([np.inf, -np.inf], np.nan).fillna(0.0)
#     S_np = _near_psd(Sigma.to_numpy(), lam=shrink_lam, jitter=jitter)
#     Sigma = pd.DataFrame(S_np, index=mu.index, columns=mu.index)
#     return mu, Sigma
#
#
# # =============================================================================
# # Gurobi solvers (Continuous & MIQP)
# # =============================================================================
# def solve_min_variance_for_target_return(
#         mu: pd.Series,
#         Sigma: pd.DataFrame,
#         mu_target: float,
#         wmax: float | None = None,
#         allow_short: bool = False,
#         obj_scale: float = 1.0,
# ) -> dict:
#     _validate_gurobi()
#     mu_v = mu.to_numpy()
#     S = Sigma.to_numpy()
#     n = len(mu)
#
#     if not allow_short:
#         wmax_eff = 1.0 if wmax is None else min(1.0, float(wmax))
#         if wmax_eff * n < 1.0:
#             wmax_eff = 1.0 / n
#     else:
#         wmax_eff = 1.0 if wmax is None else float(wmax)
#
#     mu_min, mu_max = _mu_bounds(mu_v, wmax=wmax_eff, allow_short=allow_short)
#     mu_tgt = float(np.clip(mu_target, mu_min, mu_max))
#
#     m = gp.Model("MinVar_TargetReturn")
#     m.Params.OutputFlag = 0
#     m.Params.NonConvex = 2
#
#     lb, ub = (-1.0, wmax_eff) if allow_short else (0.0, wmax_eff)
#     x = m.addMVar(n, lb=lb, ub=ub, name="x")
#
#     m.addConstr(x.sum() == 1.0, name="budget")
#     m.addConstr(mu_v @ x >= mu_tgt, name="min_return")
#     m.setObjective(obj_scale * (x @ S @ x), gp.GRB.MINIMIZE)
#     m.optimize()
#
#     if m.Status != gp.GRB.OPTIMAL:
#         return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}
#
#     w = pd.Series(x.X, index=mu.index, name="weight")
#     risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
#     eret = float(mu_v @ x.X)
#     return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}
#
#
# def solve_max_return_with_risk_cap(
#         mu: pd.Series,
#         Sigma: pd.DataFrame,
#         sigma_cap: float,
#         wmax: float | None = None,
#         allow_short: bool = False,
#         obj_scale: float = 1.0,
# ) -> dict:
#     _validate_gurobi()
#     mu_v = mu.to_numpy()
#     S = Sigma.to_numpy()
#     n = len(mu)
#
#     if not allow_short:
#         wmax_eff = 1.0 if wmax is None else min(1.0, float(wmax))
#         if wmax_eff * n < 1.0:
#             wmax_eff = 1.0 / n
#     else:
#         wmax_eff = 1.0 if wmax is None else float(wmax)
#
#     m = gp.Model("MaxRet_RiskCap")
#     m.Params.OutputFlag = 0
#     m.Params.NonConvex = 2
#
#     lb, ub = (-1.0, wmax_eff) if allow_short else (0.0, wmax_eff)
#     x = m.addMVar(n, lb=lb, ub=ub, name="x")
#
#     m.addConstr(x.sum() == 1.0, name="budget")
#     m.addQConstr(x @ S @ x <= (sigma_cap ** 2), name="risk_cap")
#     m.setObjective(obj_scale * (mu_v @ x), gp.GRB.MAXIMIZE)
#     m.optimize()
#
#     if m.Status != gp.GRB.OPTIMAL:
#         return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}
#
#     w = pd.Series(x.X, index=mu.index, name="weight")
#     risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
#     eret = float(mu_v @ x.X)
#     return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}
#
#
# def solve_miqp_minvar_target(
#         mu: pd.Series,
#         Sigma: pd.DataFrame,
#         mu_target: float,
#         K: int,
#         L: float | np.ndarray | None = None,
#         U: float | np.ndarray | None = None,
#         obj_scale: float = 1.0,
# ) -> dict:
#     """
#     min x' Σ x
#     s.t. sum x = 1, μ'x ≥ mu_target,
#          sum y ≤ K,
#          L_i y_i ≤ x_i ≤ U_i y_i, y_i ∈ {0,1}
#     Long-only diatur via L>=0.
#     """
#     _validate_gurobi()
#     mu_v = mu.to_numpy()
#     S = Sigma.to_numpy()
#     n = len(mu)
#
#     if L is None:
#         L = np.zeros(n)
#     if U is None:
#         U = np.ones(n)
#     L = np.broadcast_to(np.asarray(L, float), (n,))
#     U = np.broadcast_to(np.asarray(U, float), (n,))
#     L = np.clip(L, 0.0, 1.0)
#     U = np.clip(U, 0.0, 1.0)
#     U = np.maximum(U, L + 1e-12)
#
#     mu_min, mu_max = _mu_bounds(mu_v, wmax=float(np.max(U)), allow_short=False)
#     mu_tgt = float(np.clip(mu_target, mu_min, mu_max))
#
#     m = gp.Model("MIQP_MinVar_Target")
#     m.Params.OutputFlag = 0
#     m.Params.NonConvex = 2
#
#     x = m.addMVar(n, lb=0.0, ub=1.0, name="x")
#     y = m.addMVar(n, vtype=gp.GRB.BINARY, name="y")
#
#     m.addConstr(x.sum() == 1.0, name="budget")
#     m.addConstr(mu_v @ x >= mu_tgt, name="min_return")
#     m.addConstr(y.sum() <= int(max(1, K)), name="cardinality")
#     for i in range(n):
#         m.addConstr(L[i] * y[i] <= x[i], name=f"low_{i}")
#         m.addConstr(x[i] <= U[i] * y[i], name=f"up_{i}")
#
#     m.setObjective(obj_scale * (x @ S @ x), gp.GRB.MINIMIZE)
#     m.optimize()
#
#     if m.Status != gp.GRB.OPTIMAL:
#         return {"status": f"not_optimal({m.Status})", "weights": pd.Series(dtype=float), "risk": math.nan, "exp_return": math.nan}
#
#     w = pd.Series(x.X, index=mu.index, name="weight")
#     risk = float(np.sqrt(max(float(x.X @ S @ x.X), 0.0)))
#     eret = float(mu_v @ x.X)
#     return {"status": "optimal", "weights": w, "risk": risk, "exp_return": eret}
#
#
# # =============================================================================
# # Predict per ticker (train-or-load) + forecast
# # =============================================================================
# def predict_ticker_xlstm(ticker: str, log=print) -> Dict[str, Any]:
#     """
#     - Download daily OHLC 3y dari yfinance (drop volume=0), simpan CSV (no Adj Close)
#     - Target = log-return(OHLC-avg)
#     - Train xLSTM (atau load bila bundle ada)
#     - Evaluasi 1-step (MAE/RMSE/MAPE) di harga
#     - Forecast autoregresif 60 business days dengan guardrails
#     """
#     # ----------------- Download & save CSV -----------------
#     log(f"[INFO] Downloading {ticker}...")
#     df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False, progress=False)
#     if df.empty or "Close" not in df.columns:
#         raise ValueError(f"Tidak ada data untuk {ticker}")
#
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)
#
#     if "Volume" in df.columns:
#         df = df[df["Volume"].fillna(0) != 0]
#
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     save_dir = os.path.join(base_dir, "data", "ticker_daily")
#     os.makedirs(save_dir, exist_ok=True)
#     csv_path = os.path.join(save_dir, f"{ticker}_daily.csv")
#
#     df_to_save = df.copy()
#     if "Adj Close" in df_to_save.columns:
#         df_to_save = df_to_save.drop(columns=["Adj Close"])
#     df_to_save.index = (
#         pd.to_datetime(df_to_save.index).tz_localize(None)
#         if getattr(df_to_save.index, "tz", None)
#         else pd.to_datetime(df_to_save.index)
#     )
#     df_to_save.sort_index(inplace=True)
#     df_to_save.to_csv(csv_path, index_label="Date")
#     log(f"[SAVE] CSV: {csv_path}")
#
#     last_date = pd.to_datetime(df.index[-1])
#     if getattr(last_date, "tzinfo", None) is not None:
#         last_date = last_date.tz_localize(None)
#     next_bd = last_date + BDay(1)
#
#     # ----------------- Build series & returns -----------------
#     need = ["Open", "High", "Low", "Close"]
#     for c in need:
#         if c not in df.columns:
#             raise ValueError(f"Kolom {c} tidak ada pada data {ticker}.")
#     price = df[need].mean(axis=1).astype("float64").sort_index()
#     ret = np.log(price / price.shift(1)).dropna()
#     if len(ret) < 200:
#         raise ValueError("Data terlalu pendek untuk training (butuh > 200 titik).")
#
#     seq_len   = 16
#     batch_sz  = 32
#     split_idx = int(len(ret) * 0.9)
#
#     train_r = ret.iloc[:split_idx].values.reshape(-1, 1)
#     test_r  = ret.iloc[split_idx:].values.reshape(-1, 1)
#
#     y_scaler = StandardScaler()
#     train_r_s = y_scaler.fit_transform(train_r)
#     test_r_s  = y_scaler.transform(test_r)
#
#     def make_seq(arr1d: np.ndarray, L: int):
#         N = len(arr1d)
#         if N <= L:
#             return torch.empty(0, L, 1), torch.empty(0, 1)
#         X = np.zeros((N - L, L, 1), dtype=np.float32)
#         y = np.zeros((N - L, 1), dtype=np.float32)
#         for i in range(N - L):
#             X[i, :, 0] = arr1d[i:i+L, 0]
#             y[i, 0] = arr1d[i+L, 0]
#         return torch.from_numpy(X), torch.from_numpy(y)
#
#     trainX, trainY = make_seq(train_r_s, seq_len)
#     testX,  testY  = make_seq(test_r_s,  seq_len)
#     if len(trainX) == 0 or len(testX) == 0:
#         raise ValueError("Dataset terlalu pendek; perpanjang period atau kurangi seq_len.")
#
#     train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=batch_sz, shuffle=True)
#     test_loader  = DataLoader(TensorDataset(testX,  testY),  batch_size=batch_sz, shuffle=False)
#
#     # ----------------- Model bundle path -----------------
#     model_dir = os.path.join(base_dir, "Data", "models", ticker)
#     os.makedirs(model_dir, exist_ok=True)
#     model_path  = os.path.join(model_dir, "model.pth")
#     scaler_path = os.path.join(model_dir, "scaler.pkl")
#     config_path = os.path.join(model_dir, "config.json")
#
#     # ----------------- Build model -----------------
#     input_size, head_size, num_heads = 1, 32, 2
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = xLSTM(input_size, head_size, num_heads, layers='msm', batch_first=True).to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     # ----------------- Train OR Load -----------------
#     loaded = False
#     if os.path.exists(model_path) and os.path.exists(scaler_path):
#         try:
#             model.load_state_dict(torch.load(model_path, map_location=device))
#             with open(scaler_path, "rb") as f:
#                 y_scaler = pickle.load(f)
#             loaded = True
#             log(f"[LOAD] Loaded model & scaler for {ticker}")
#         except Exception as e:
#             log(f"[LOAD-WARN] Gagal load bundle, retrain: {e}")
#
#     if not loaded:
#         log("[STEP] Training xLSTM (target: return)…")
#         epochs = 10
#         for ep in tqdm(range(epochs), desc=f"Training {ticker}"):
#             model.train()
#             running = 0.0
#             for Xb, yb in train_loader:
#                 Xb = Xb.to(device); yb = yb.to(device)
#                 optimizer.zero_grad()
#                 out, _ = model(Xb)
#                 pred = out[:, -1, :]
#                 loss = criterion(pred, yb)
#                 loss.backward()
#                 optimizer.step()
#                 running += loss.item()
#             log(f"  [EPOCH {ep+1}] loss={running/max(1,len(train_loader)):.6f}")
#
#         torch.save(model.state_dict(), model_path)
#         with open(scaler_path, "wb") as f:
#             pickle.dump(y_scaler, f)
#         config = {
#             "ticker": ticker, "seq_len": seq_len, "input_size": input_size,
#             "head_size": head_size, "num_heads": num_heads, "epochs": 10,
#             "training_date": datetime.now().isoformat()
#         }
#         with open(config_path, "w") as f:
#             json.dump(config, f, indent=4)
#         log(f"[SAVE] Bundle saved to {model_dir}")
#
#     # ----------------- Evaluate 1-step in PRICE -----------------
#     test_start_pos = split_idx + seq_len
#     price_arr = price.values.astype("float64")
#
#     model.eval()
#     test_pred_s = []
#     with torch.no_grad():
#         for Xb, _ in test_loader:
#             Xb = Xb.to(device)
#             out, _ = model(Xb)
#             test_pred_s.extend(out[:, -1, :].cpu().numpy())
#     test_pred_s = np.array(test_pred_s).reshape(-1, 1)
#     test_pred_r = y_scaler.inverse_transform(test_pred_s)
#
#     ref_prices = price_arr[test_start_pos-1 : test_start_pos-1 + len(test_pred_r)]
#     pred_prices = ref_prices * np.exp(test_pred_r.flatten())
#     actual_prices = price_arr[test_start_pos : test_start_pos + len(test_pred_r)]
#
#     rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
#     mae  = float(mean_absolute_error(actual_prices, pred_prices))
#     mape = float(np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100.0)
#     log(f"[DONE] {ticker} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")
#
#     # ----------------- 60-day forecast w/ guardrails -----------------
#     H = 60
#     sim_prices = list(price_arr)
#     last_P = float(sim_prices[-1])
#
#     full_r = np.log(price / price.shift(1)).dropna().values.reshape(-1, 1)
#     full_r_s = y_scaler.transform(full_r)
#     last_seq = full_r_s[-16:].copy()
#
#     def rolling_stats(prices: list):
#         s = pd.Series(prices, dtype="float64")
#         ret_ = np.log(s / s.shift(1))
#         mu60 = float(ret_.rolling(60).mean().iloc[-1]) if len(ret_) >= 60 else 0.0
#         sig20 = float(ret_.rolling(20).std().iloc[-1]) if len(ret_) >= 20 else (np.std(ret_) if len(ret_)>1 else 0.0)
#         ma20 = float(s.rolling(20).mean().iloc[-1]) if len(s) >= 20 else float(s.iloc[-1])
#         rsi = float(RSIIndicator(close=s, window=14).rsi().iloc[-1]) if len(s) >= 15 else 50.0
#         bb  = BollingerBands(close=s) if len(s) >= 21 else None
#         bbw = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1] / s.rolling(20).mean().iloc[-1]) \
#             if bb is not None and pd.notna(s.rolling(20).mean().iloc[-1]) else 0.0
#         return mu60, sig20, ma20, rsi, bbw
#
#     future_vals, future_dates = [], [(last_date + BDay(i)).date().isoformat() for i in range(1, H+1)]
#     with torch.no_grad():
#         for h in range(H):
#             X = torch.from_numpy(last_seq.reshape(1, 16, 1)).float().to(device)
#             out, _ = model(X)
#             r_hat_s = out[:, -1, :].cpu().numpy()[0, 0]
#             r_hat   = float(y_scaler.inverse_transform([[r_hat_s]])[0, 0])
#
#             mu60, sig20, ma20, rsi, bbw = rolling_stats(sim_prices)
#             alpha = 0.30 * ((h+1)/H)
#             r_hat = (1.0 - alpha) * r_hat + alpha * mu60
#             kappa = 0.05
#             gap = np.log(last_P) - np.log(max(ma20, 1e-8))
#             r_hat = r_hat - kappa * gap
#             c = 2.5
#             cap = c * (sig20 if np.isfinite(sig20) and sig20 > 0 else 0.02)
#             r_hat = float(np.clip(r_hat, -cap, cap))
#
#             next_P = last_P * np.exp(r_hat)
#             future_vals.append(float(next_P))
#             sim_prices.append(float(next_P))
#             last_P = float(next_P)
#
#             r_hat_s_feedback = float(y_scaler.transform([[r_hat]])[0, 0])
#             last_seq = np.vstack([last_seq[1:], [r_hat_s_feedback]])
#
#     pred_1d       = float(future_vals[0])
#     pred_3mo      = float(future_vals[-1])
#     pred_1d_date  = future_dates[0]
#     pred_3mo_date = future_dates[-1]
#
#     actual_list = [float(x) for x in actual_prices.tolist()]
#     preds_list  = [float(x) for x in pred_prices.tolist()]
#
#     return {
#         "ticker": ticker,
#         "mae": float(mae), "rmse": float(rmse), "mape": float(mape),
#         "actual": actual_list, "preds": preds_list,
#         "last_price": float(price.values[-1]),
#         "next_pred": float(pred_1d),
#         "last_date": last_date.date().isoformat(),
#         "next_bd":   next_bd.date().isoformat(),
#         "pred_1d":   pred_1d, "pred_1d_date": pred_1d_date,
#         "pred_3mo":  pred_3mo, "pred_3mo_date": pred_3mo_date,
#         "future_dates":  future_dates,
#         "future_values": [float(v) for v in future_vals],
#         "csv_path": csv_path,
#     }
#
# #
# # # =============================================================================
# # # Public API: Optimize from PREDICTIONS (μ & Σ dari future_values)
# # # =============================================================================
# # def optimize_from_predictions(
# #         results: List[Dict[str, Any]],
# #         dana: float,
# #         mode: str = "minvar_target",   # "minvar_target" | "maxret_cap"
# #         mu_target: float | None = None,
# #         sigma_cap: float | None = None,
# #         wmax: float | None = 0.5,
# #         allow_short: bool = False,
# #         K: int | None = None,
# #         L: float | np.ndarray | None = None,
# #         U: float | np.ndarray | None = None,
# #         use_log_for_mu: bool = False,
# #         use_log_for_cov: bool = True,
# #         shrink_lam: float = 0.10,
# #         jitter: float = 1e-8,
# #         rf_daily: float = 0.03/252,
# # ) -> dict:
# #     mu, Sigma = build_mu_sigma_from_results(
# #         results,
# #         use_log_for_mu=use_log_for_mu,
# #         use_log_for_cov=use_log_for_cov,
# #         shrink_lam=shrink_lam,
# #         jitter=jitter,
# #     )
# #
# #     if len(mu) == 1 and not allow_short and K in (None, 1):
# #         tkr = mu.index[0]
# #         w = 1.0
# #         port_ret = float(mu.iloc[0])
# #         port_risk = float(np.sqrt(max(Sigma.iloc[0, 0], 0.0)))
# #         sharpe = (port_ret - rf_daily) / port_risk if port_risk > 0 else np.nan
# #         df_alloc = pd.DataFrame([{
# #             "ticker": tkr, "weight": w, "nominal": float(dana),
# #             "exp_return": port_ret, "contrib_return": port_ret * w
# #         }])
# #         return {"df_alloc": df_alloc, "summary": {"exp_return": port_ret, "risk": port_risk, "sharpe": sharpe, "status": "trivial"}}
# #
# #     if K is not None:
# #         if mu_target is None:
# #             mu_target = float(mu.median())
# #         sol = solve_miqp_minvar_target(mu, Sigma, mu_target=mu_target, K=int(max(1, K)), L=L, U=U)
# #     else:
# #         if mode == "minvar_target":
# #             if mu_target is None:
# #                 mu_target = float(mu.median())
# #             sol = solve_min_variance_for_target_return(mu, Sigma, mu_target=mu_target, wmax=wmax, allow_short=allow_short)
# #         elif mode == "maxret_cap":
# #             if sigma_cap is None:
# #                 diag_std = np.sqrt(np.maximum(np.diag(Sigma.to_numpy()), 0))
# #                 sigma_cap = float(np.nanmedian(diag_std))
# #             sol = solve_max_return_with_risk_cap(mu, Sigma, sigma_cap=sigma_cap, wmax=wmax, allow_short=allow_short)
# #         else:
# #             raise ValueError("mode harus 'minvar_target' atau 'maxret_cap'.")
# #
# #     status = sol["status"]; w = sol["weights"]; port_risk = sol["risk"]; port_ret = sol["exp_return"]
# #     if status != "optimal" or w.empty:
# #         return {
# #             "df_alloc": pd.DataFrame(columns=["ticker","weight","nominal","exp_return","contrib_return"]),
# #             "summary":  {"exp_return": math.nan, "risk": math.nan, "sharpe": math.nan, "status": status}
# #         }
# #
# #     df_alloc = pd.DataFrame({"ticker": w.index, "weight": w.values})
# #     df_alloc["nominal"] = df_alloc["weight"] * float(dana)
# #     df_alloc["exp_return"] = df_alloc["ticker"].map(mu.to_dict())
# #     df_alloc["contrib_return"] = df_alloc["weight"] * df_alloc["exp_return"]
# #     sharpe = (port_ret - rf_daily) / port_risk if port_risk > 0 else np.nan
# #
# #     return {
# #         "df_alloc": df_alloc.sort_values("weight", ascending=False).reset_index(drop=True),
# #         "summary":  {"exp_return": port_ret, "risk": port_risk, "sharpe": sharpe, "status": status}
# #     }
# #
# #
# # # =============================================================================
# # # Orchestrator: predict many tickers -> optimize
# # # =============================================================================
# # def run_full_pipeline(
# #         tickers: List[str],
# #         dana: float,
# #         mode: str = "minvar_target",
# #         mu_target: float | None = None,
# #         sigma_cap: float | None = None,
# #         K: int | None = None,
# #         L: float | np.ndarray | None = None,
# #         U: float | np.ndarray | None = None,
# #         allow_short: bool = False,
# #         wmax: float | None = 0.5,
# #         use_log_for_mu: bool = False,
# #         use_log_for_cov: bool = True,
# #         shrink_lam: float = 0.10,
# #         jitter: float = 1e-8,
# #         log=print,
# # ) -> dict:
# #     results, logs_per_ticker = [], {}
# #     for code in tickers:
# #         tkr = f"{code}.JK" if not code.endswith(".JK") else code
# #         tlogs = []
# #         def _log(m): tlogs.append(str(m)); log(m)
# #         try:
# #             res = predict_ticker_xlstm(tkr, log=_log)
# #             if not res.get("future_values"):
# #                 raise ValueError("future_values kosong")
# #             results.append(res)
# #         except Exception as e:
# #             _log(f"[ERR] {tkr}: {e}")
# #         logs_per_ticker[tkr] = tlogs
# #
# #     if len(results) < 2:
# #         return {
# #             "df_alloc": pd.DataFrame(columns=["ticker","weight","nominal","exp_return","contrib_return"]),
# #             "summary":  {"exp_return": np.nan, "risk": np.nan, "sharpe": np.nan, "status": "need>=2_assets"},
# #             "logs": logs_per_ticker
# #         }
# #
# #     out = optimize_from_predictions(
# #         results=results,
# #         dana=float(dana),
# #         mode=mode,
# #         mu_target=mu_target,
# #         sigma_cap=sigma_cap,
# #         wmax=wmax,
# #         allow_short=allow_short,
# #         K=K, L=L, U=U,
# #         use_log_for_mu=use_log_for_mu,
# #         use_log_for_cov=use_log_for_cov,
# #         shrink_lam=shrink_lam,
# #         jitter=jitter,
# #     )
# #     out["logs"] = logs_per_ticker
# #     return out
# #
# #
# # # =============================================================================
# # # Example CLI usage
# # # =============================================================================
# # if __name__ == "__main__":
# #     tickers = ["BBCA","BBRI","TLKM","AALI"]
# #     dana = 25_000_000
# #
# #     # Example 1: Continuous min-variance @ median(mu)
# #     res = run_full_pipeline(
# #         tickers=tickers,
# #         dana=dana,
# #         mode="minvar_target",
# #         mu_target=None,   # use median(mu)
# #         wmax=0.5,
# #     )
# #     print("\n=== Allocation (minvar@median) ===")
# #     print(res["df_alloc"])
# #     print("Summary:", res["summary"])
# #
# #     # # Example 2: MIQP (enable by uncommenting)
# #     # res_miqp = run_full_pipeline(
# #     #     tickers=tickers,
# #     #     dana=dana,
# #     #     K=3, L=0.05, U=0.45,   # at most 3 assets, 5–45% each
# #     # )
# #     # print("\n=== Allocation (MIQP) ===")
# #     # print(res_miqp["df_alloc"])
# #     # print("Summary:", res_miqp["summary"])


# services/predict_xlstm.py
# -*- coding: utf-8 -*-
"""
HANYA modul prediksi xLSTM + forecasting 60 hari.
Bagian Gurobi/Markowitz SUDAH DIPISAH ke services/portfolio_opt_gurobi.py

=== BAGIAN YANG DIHAPUS & DIPINDAH (penanda) ===
# [REMOVED → portfolio_opt_gurobi.py]
# - _validate_gurobi
# - _covariance_from_paths
# - _near_psd
# - _mu_bounds
# - build_mu_sigma_from_results
# - solve_min_variance_for_target_return
# - solve_max_return_with_risk_cap
# - solve_miqp_minvar_target
# - optimize_portfolio_gurobi
# ==============================================
"""

from __future__ import annotations
import os, json, pickle
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

import yfinance as yf
from pandas.tseries.offsets import BDay
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# xLSTM kamu
from xLSTM.xLSTM import xLSTM


def predict_ticker_xlstm(ticker: str, log=print) -> Dict[str, Any]:
    """
    - Download OHLC 3y dari yfinance -> simpan CSV (no Adj Close)
    - Target = log-return (OHLC-avg)
    - Train-or-load xLSTM
    - Evaluasi 1-step di harga
    - Forecast 60 hari bursa dengan guardrails
    """
    # ---------- Download & save CSV ----------
    log(f"[INFO] Downloading {ticker}...")
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Tidak ada data untuk {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Volume" in df.columns:
        df = df[df["Volume"].fillna(0) != 0]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(base_dir, "data", "ticker_daily")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{ticker}_daily.csv")

    df_to_save = df.copy()
    if "Adj Close" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["Adj Close"])
    df_to_save.index = (
        pd.to_datetime(df_to_save.index).tz_localize(None)
        if getattr(df_to_save.index, "tz", None)
        else pd.to_datetime(df_to_save.index)
    )
    df_to_save.sort_index(inplace=True)
    df_to_save.to_csv(csv_path, index_label="Date")
    log(f"[SAVE] CSV: {csv_path}")

    last_date = pd.to_datetime(df.index[-1])
    if getattr(last_date, "tzinfo", None) is not None:
        last_date = last_date.tz_localize(None)
    next_bd = last_date + BDay(1)

    # ---------- Series & returns ----------
    need = ["Open", "High", "Low", "Close"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Kolom {c} tidak ada pada data {ticker}.")
    price = df[need].mean(axis=1).astype("float64").sort_index()
    ret = np.log(price / price.shift(1)).dropna()
    if len(ret) < 200:
        raise ValueError("Data terlalu pendek untuk training (butuh > 200 titik).")

    seq_len   = 16
    batch_sz  = 32
    split_idx = int(len(ret) * 0.9)

    train_r = ret.iloc[:split_idx].values.reshape(-1, 1)
    test_r  = ret.iloc[split_idx:].values.reshape(-1, 1)

    y_scaler = StandardScaler()
    train_r_s = y_scaler.fit_transform(train_r)
    test_r_s  = y_scaler.transform(test_r)

    def make_seq(arr1d: np.ndarray, L: int):
        N = len(arr1d)
        if N <= L:
            return torch.empty(0, L, 1), torch.empty(0, 1)
        X = np.zeros((N - L, L, 1), dtype=np.float32)
        y = np.zeros((N - L, 1), dtype=np.float32)
        for i in range(N - L):
            X[i, :, 0] = arr1d[i:i+L, 0]
            y[i, 0] = arr1d[i+L, 0]
        return torch.from_numpy(X), torch.from_numpy(y)

    trainX, trainY = make_seq(train_r_s, seq_len)
    testX,  testY  = make_seq(test_r_s,  seq_len)
    if len(trainX) == 0 or len(testX) == 0:
        raise ValueError("Dataset terlalu pendek; perpanjang period atau kurangi seq_len.")

    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=batch_sz, shuffle=True)
    test_loader  = DataLoader(TensorDataset(testX,  testY),  batch_size=batch_sz, shuffle=False)

    # ---------- Model bundle paths ----------
    model_dir = os.path.join(base_dir, "Data", "models", ticker)
    os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, "model.pth")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    config_path = os.path.join(model_dir, "config.json")

    # ---------- Model ----------
    input_size, head_size, num_heads = 1, 32, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xLSTM(input_size, head_size, num_heads, layers='msm', batch_first=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Load or Train ----------
    loaded = False
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            with open(scaler_path, "rb") as f:
                y_scaler = pickle.load(f)
            loaded = True
            log(f"[LOAD] Loaded model & scaler for {ticker}")
        except Exception as e:
            log(f"[LOAD-WARN] Gagal load bundle, retrain: {e}")

    if not loaded:
        log("[STEP] Training xLSTM (target: return)…")
        epochs = 10
        for ep in tqdm(range(epochs), desc=f"Training {ticker}"):
            model.train()
            running = 0.0
            for Xb, yb in train_loader:
                Xb = Xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                out, _ = model(Xb)
                pred = out[:, -1, :]
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                running += loss.item()
            log(f"  [EPOCH {ep+1}] loss={running/max(1,len(train_loader)):.6f}")

        torch.save(model.state_dict(), model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(y_scaler, f)
        config = {
            "ticker": ticker, "seq_len": seq_len, "input_size": input_size,
            "head_size": head_size, "num_heads": num_heads, "epochs": 10,
            "training_date": datetime.now().isoformat()
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        log(f"[SAVE] Bundle saved to {model_dir}")

    # ---------- Evaluate 1-step in PRICE ----------
    test_start_pos = split_idx + seq_len
    price_arr = price.values.astype("float64")

    model.eval()
    test_pred_s = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            out, _ = model(Xb)
            test_pred_s.extend(out[:, -1, :].cpu().numpy())
    test_pred_s = np.array(test_pred_s).reshape(-1, 1)
    test_pred_r = y_scaler.inverse_transform(test_pred_s)

    ref_prices   = price_arr[test_start_pos-1 : test_start_pos-1 + len(test_pred_r)]
    pred_prices  = ref_prices * np.exp(test_pred_r.flatten())
    actual_prices= price_arr[test_start_pos : test_start_pos + len(test_pred_r)]

    rmse = float(np.sqrt(mean_squared_error(actual_prices, pred_prices)))
    mae  = float(mean_absolute_error(actual_prices, pred_prices))
    mape = float(np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100.0)
    log(f"[DONE] {ticker} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")

    # ---------- Forecast 60 days ----------
    H = 60
    sim_prices = list(price_arr)
    last_P = float(sim_prices[-1])

    full_r = np.log(price / price.shift(1)).dropna().values.reshape(-1, 1)
    full_r_s = y_scaler.transform(full_r)
    last_seq = full_r_s[-16:].copy()

    def rolling_stats(prices: list):
        s = pd.Series(prices, dtype="float64")
        ret_ = np.log(s / s.shift(1))
        mu60 = float(ret_.rolling(60).mean().iloc[-1]) if len(ret_) >= 60 else 0.0
        sig20 = float(ret_.rolling(20).std().iloc[-1]) if len(ret_) >= 20 else (np.std(ret_) if len(ret_)>1 else 0.0)
        ma20 = float(s.rolling(20).mean().iloc[-1]) if len(s) >= 20 else float(s.iloc[-1])
        rsi = float(RSIIndicator(close=s, window=14).rsi().iloc[-1]) if len(s) >= 15 else 50.0
        bb  = BollingerBands(close=s) if len(s) >= 21 else None
        bbw = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1] / s.rolling(20).mean().iloc[-1]) \
            if bb is not None and pd.notna(s.rolling(20).mean().iloc[-1]) else 0.0
        return mu60, sig20, ma20, rsi, bbw

    future_vals, future_dates = [], [(last_date + BDay(i)).date().isoformat() for i in range(1, H+1)]
    with torch.no_grad():
        for h in range(H):
            X = torch.from_numpy(last_seq.reshape(1, 16, 1)).float().to(device)
            out, _ = model(X)
            r_hat_s = out[:, -1, :].cpu().numpy()[0, 0]
            r_hat   = float(y_scaler.inverse_transform([[r_hat_s]])[0, 0])

            mu60, sig20, ma20, rsi, bbw = rolling_stats(sim_prices)
            alpha = 0.30 * ((h+1)/H)
            r_hat = (1.0 - alpha) * r_hat + alpha * mu60
            kappa = 0.05
            gap = np.log(last_P) - np.log(max(ma20, 1e-8))
            r_hat = r_hat - kappa * gap
            c = 2.5
            cap = c * (sig20 if np.isfinite(sig20) and sig20 > 0 else 0.02)
            r_hat = float(np.clip(r_hat, -cap, cap))

            next_P = last_P * np.exp(r_hat)
            future_vals.append(float(next_P))
            sim_prices.append(float(next_P))
            last_P = float(next_P)

            r_hat_s_feedback = float(y_scaler.transform([[r_hat]])[0, 0])
            last_seq = np.vstack([last_seq[1:], [r_hat_s_feedback]])

    pred_1d       = float(future_vals[0])
    pred_3mo      = float(future_vals[-1])
    pred_1d_date  = future_dates[0]
    pred_3mo_date = future_dates[-1]

    actual_list = [float(x) for x in actual_prices.tolist()]
    preds_list  = [float(x) for x in pred_prices.tolist()]

    return {
        "ticker": ticker,
        "mae": float(mae), "rmse": float(rmse), "mape": float(mape),
        "actual": actual_list, "preds": preds_list,
        "last_price": float(price.values[-1]),
        "next_pred": float(pred_1d),
        "last_date": last_date.date().isoformat(),
        "next_bd":   next_bd.date().isoformat(),
        "pred_1d":   pred_1d, "pred_1d_date": pred_1d_date,
        "pred_3mo":  pred_3mo, "pred_3mo_date": pred_3mo_date,
        "future_dates":  future_dates,
        "future_values": [float(v) for v in future_vals],
        "csv_path": csv_path,
    }
