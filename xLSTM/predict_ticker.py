from __future__ import annotations
import os, json, pickle
from typing import Any, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import BDay
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from xLSTM.xLSTM import xLSTM


def predict_ticker_xlstm(ticker: str, log=print) -> Dict[str, Any]:

    log(f"[INFO] Downloading {ticker}...")
    #melakukan pengambilan data dari yfinance
    df = yf.download(ticker, period="10y", interval="1d", auto_adjust=False, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Tidak ada data untuk {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    #tidak menggunakan  data yang memiliki transaksi volume 0 (karena harga akan sama)
    if "Volume" in df.columns:
        df = df[df["Volume"].fillna(0) != 0]
    #menyimpan hasil uduhan data haarian tiap ticker
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(base_dir, "data", "ticker_daily")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{ticker}_daily.csv")
    df_to_save = df.copy()
    if "Adj Close" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["Adj Close"])

    #agar berurutan sesuai dengan tanggal
    df_to_save.index = (
        pd.to_datetime(df_to_save.index).tz_localize(None)
        if getattr(df_to_save.index, "tz", None)
        else pd.to_datetime(df_to_save.index)
    )
    df_to_save.sort_index(inplace=True)
    df_to_save.to_csv(csv_path, index_label="Date")
    log(f"[SAVE] CSV: {csv_path}")

    #menghitung hari berikutnya menggunakan Bday, semisal diprediksi dijumat sabtu dan minggu libur
    #sistem akan memprediksinya ke hari bursa buka yaitu hari senin
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
        raise ValueError("Data terlalu pendek untuk training (butuh > 200 data).")
    #menyetel nilai sequend window dan membagi data training : latih sebanyak 90:10
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

    # mempersiapkan path untuk menyimpan model
    model_dir = os.path.join(base_dir, "Data", "models", ticker)
    os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, "model.pth")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    config_path = os.path.join(model_dir, "config.json")

    # Model
    input_size, head_size, num_heads = 1, 32, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xLSTM(input_size, head_size, num_heads, layers='msm', batch_first=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #jika suddah terdapat data sebelumnya yang ditraining, maka akan di load kembali
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

    #melakukan evaluasi harga RMSE MAE MAPE
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

    #melakukan prediksi sebanyak 60 hari
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

            tilt = (rsi - 50.0) / 100.0
            r_hat += 0.10 * tilt * (sig20 if np.isfinite(sig20) else 0.02)

            c = 2.5
            cap = c * (sig20 if np.isfinite(sig20) and sig20 > 0 else 0.02)
            cap *= (1.0 + np.clip(bbw, 0.0, 0.5))
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

