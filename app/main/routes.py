import os, csv, glob
from pathlib import Path
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from xLSTM.predict_ticker import predict_ticker_xlstm
from Optimizer.Markowitz import run_markowitz_optimization  # perhatikan path impor sesuai strukturmu

try:
    import yfinance as yf
except ImportError:
    yf = None

bp = Blueprint('main', __name__)

# =======================
#  Utility
# =======================
def find_csv_path():
    """Cari file CSV daftar saham IHSG."""
    project_root = Path(current_app.root_path).parent
    data_dir = project_root / "Data" / "Ticker"
    patterns = [
        "ihsg_tickers.csv", "ihsg_tickers.CSV",
        "IHSG_Tickers.csv", "IHSG_Tickers.CSV"
    ]
    for pat in patterns:
        p = data_dir / pat
        if p.exists():
            return str(p)
    for p in glob.glob(str(data_dir / "ihsg_tickers.*")):
        if Path(p).is_file():
            return p
    return None


def load_tickers():
    """Baca CSV → list[(code, name)]."""
    path = find_csv_path()
    tickers = []
    if not path:
        flash("File CSV daftar IHSG tidak ditemukan di Data/Ticker.", "warning")
        return tickers

    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip().upper()
                name = (row.get("name") or "").strip()
                if code:
                    tickers.append((code, name or code))
    except Exception as e:
        flash(f"Gagal membaca CSV: {e}", "danger")

    if not tickers:
        flash("CSV terbaca, tapi kosong atau header salah (harus 'code,name').", "warning")
    return tickers


def to_yf_symbol(code: str) -> str:
    return f"{code}.JK"

# =======================
#  ROUTES
# =======================
@bp.route('/')
def index():
    return render_template('index.html', title='Home')

@bp.route('/dashboard', methods=['GET'])
def dashboard():
    tickers = load_tickers()
    return render_template('dashboard.html', title='Dashboard', tickers=tickers)

@bp.route('/result', methods=['GET', 'POST'])
def about():
    if request.method == 'GET':
        return redirect(url_for('main.dashboard'))

    amount = request.form.get('amount', type=float)
    # ✅ gunakan getlist('tickers[]') agar bisa ambil banyak saham dari checkbox
    selected_codes = request.form.getlist('tickers[]')

    if not selected_codes:
        flash("Pilih minimal satu saham.", "warning")
        return redirect(url_for('main.dashboard'))

    results = []
    logs_per_ticker = {}

    for code in selected_codes:
        ticker = f"{code}.JK"
        logs = []

        def log_func(msg):
            print(msg)
            logs.append(str(msg))
        #
        # try:
        #     result = predict_ticker_xlstm(ticker, log=log_func)
        #     results.append(result)
        # except Exception as e:
        #     logs.append(f"[ERROR] {e}")
        #     flash(f"Gagal memproses {ticker}: {e}", "danger")

        try:
            res = predict_ticker_xlstm(ticker, log=log_func)
            # pastikan field wajib ada untuk optimizer
            if not res.get("future_values"):
                raise ValueError("future_values kosong")
            # siapkan expected_return sederhana (mean daily return dari future_values)
            fv = res["future_values"]
            arr = np.array(fv, dtype=float)

            if len(arr) >= 2 and arr.min() > 0:
                rets = (arr[1:] - arr[:-1]) / arr[:-1]
                res["expected_return"] = float(np.nanmean(rets))
            else:
                res["expected_return"] = 0.0

            # beri nama yang dikenali optimizer:
            res["future_predictions"] = res["future_values"]
            results.append(res)
        except Exception as e:
            logs.append(f"[ERROR] {e}")
            flash(f"Gagal memproses {ticker}: {e}", "danger")


        logs_per_ticker[ticker] = logs

        # >>> INI BAGIAN YANG MENGGANTIKAN all_prediction_results <<<
    if results:
        try:
            df_alloc, summary = run_markowitz_optimization(results, amount)
            df_alloc_records = df_alloc.to_dict(orient="records")
        except Exception as e:
            flash(f"Optimasi portofolio gagal: {e}", "danger")
            df_alloc_records, summary = [], {}
    else:
        df_alloc_records, summary = [], {}
        flash("Tidak ada hasil prediksi yang valid.", "warning")

    return render_template(
        # 'about.html',
        # title='Hasil Prediksi xLSTM',
        # amount=amount,
        # results=results,
        # logs_per_ticker=logs_per_ticker

        'about.html',
        title='Hasil Prediksi xLSTM',
        amount=amount,
        results=results,
        logs_per_ticker=logs_per_ticker,
        df_alloc=df_alloc_records,   # untuk tabel alokasi
        summary=summary              # ringkasan (return, risk, Sharpe)
    )