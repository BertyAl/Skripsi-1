import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from xLSTM.xLSTM import xLSTM  # pastikan path ini benar

# ============================================================
# Fungsi prediksi xLSTM berdasarkan ticker saham
# ============================================================
def predict_ticker_xlstm(ticker, log=lambda msg: print(msg)):
    log(f"[INFO] Mengunduh data {ticker} dari yfinance...")
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"Tidak ada data untuk {ticker}")

    close = df[['Close']].astype('float32').values
    log(f"[STEP] Jumlah data: {len(close)} baris")

    # Normalisasi
    scaler = MinMaxScaler((0, 1))
    dataset = scaler.fit_transform(close)
    log("[STEP] Normalisasi selesai")

    # Dataset dan parameter
    seq_len = 8
    batch_size = 16

    def create_dataset_np(data, seq_len):
        N = len(data)
        X = np.zeros((N - seq_len, seq_len, 1), dtype=np.float32)
        y = np.zeros((N - seq_len, 1), dtype=np.float32)
        for i in range(N - seq_len):
            X[i, :, 0] = data[i:i + seq_len, 0]
            y[i, 0] = data[i + seq_len, 0]
        return torch.from_numpy(X), torch.from_numpy(y)

    split = int(len(dataset) * 0.9)
    train, test = dataset[:split], dataset[split:]
    trainX, trainY = create_dataset_np(train, seq_len)
    testX, testY = create_dataset_np(test, seq_len)

    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(testX, testY), batch_size=batch_size, shuffle=False)

    # Model
    input_size, head_size, num_heads = 1, 32, 2
    model = xLSTM(input_size, head_size, num_heads, layers='msm', batch_first=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training singkat
    log("[STEP] Mulai training model xLSTM...")
    epochs = 20
    for epoch in tqdm(range(epochs), desc=f"Training {ticker}"):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            out, _ = model(X)
            out = out[:, -1, :]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log(f"  [EPOCH {epoch+1}] loss={total_loss/len(train_loader):.6f}")

    # Evaluasi
    log("[STEP] Evaluasi model...")
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            out, _ = model(X)
            preds.extend(out[:, -1, :].numpy())

    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    actual_inv = scaler.inverse_transform(testY.numpy().reshape(-1, 1))

    mae = mean_absolute_error(actual_inv, preds_inv)
    log(f"[DONE] Selesai: {ticker} | MAE={mae:.4f}")

    return {
        "ticker": ticker,
        "mae": float(mae),
        "actual": actual_inv.flatten().tolist(),
        "preds": preds_inv.flatten().tolist(),
        "last_price": float(actual_inv[-1]),
        "next_pred": float(preds_inv[-1]),
    }
