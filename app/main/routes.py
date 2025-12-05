# import csv, glob
# import os
# import pandas as pd
# import json
# from pathlib import Path
# import numpy as np
# from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
# from xLSTM.predict_ticker import predict_ticker_xlstm
# from Optimizer.Markowitz import run_markowitz_optimization  # perhatikan path impor sesuai strukturmu
#
# try:
#     import yfinance as yf
# except ImportError:
#     yf = None
#
# bp = Blueprint('main', __name__)
# SECTOR_FOLDER = 'Data/Ticker/Sectors'
# # =======================
# #  Utility
# # =======================
#
# def find_csv_path():
#     """Cari file CSV daftar saham IHSG."""
#     project_root = Path(current_app.root_path).parent
#     data_dir = project_root / "Data" / "Ticker"
#     patterns = [
#         "ihsg_tickers.csv", "ihsg_tickers.CSV",
#         "IHSG_Tickers.csv", "IHSG_Tickers.CSV"
#     ]
#     for pat in patterns:
#         p = data_dir / pat
#         if p.exists():
#             return str(p)
#     for p in glob.glob(str(data_dir / "ihsg_tickers.*")):
#         if Path(p).is_file():
#             return p
#     return None
#
# def get_tickers_grouped_by_sector():
#     sectors_data = {}
#
#     # Cek apakah folder ada
#     if os.path.exists(SECTOR_FOLDER):
#         for filename in os.listdir(SECTOR_FOLDER):
#             if filename.endswith(".csv"):
#                 # Nama sektor diambil dari nama file (misal: "Energy.csv" -> "Energy")
#                 sector_name = filename.replace(".csv", "")
#
#                 try:
#                     # Baca CSV
#                     file_path = os.path.join(SECTOR_FOLDER, filename)
#                     df = pd.read_csv(file_path)
#
#                     # Asumsi kolom di CSV adalah 'code' dan 'name'
#                     # Konversi ke list of dict: [{'code': 'AAAA', 'name': 'Perusahaan A'}, ...]
#                     records = df[['code', 'name']].to_dict(orient='records')
#                     sectors_data[sector_name] = records
#                 except Exception as e:
#                     print(f"Error reading {filename}: {e}")
#
#     return sectors_data
#
# def load_tickers():
#     """Baca CSV → list[(code, name)]."""
#     path = find_csv_path()
#     tickers = []
#     if not path:
#         flash("File CSV daftar IHSG tidak ditemukan di Data/Ticker.", "warning")
#         return tickers
#
#     try:
#         with open(path, newline='', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 code = (row.get("code") or "").strip().upper()
#                 name = (row.get("name") or "").strip()
#                 if code:
#                     tickers.append((code, name or code))
#     except Exception as e:
#         flash(f"Gagal membaca CSV: {e}", "danger")
#
#     if not tickers:
#         flash("CSV terbaca, tapi kosong atau header salah (harus 'code,name').", "warning")
#     return tickers
#
#
# def to_yf_symbol(code: str) -> str:
#     return f"{code}.JK"
#
# # =======================
# #  ROUTES
# # =======================
# @bp.route('/')
# def index():
#     return render_template('index.html', title='Home')
#
# @bp.route('/dashboard', methods=['GET'])
# def dashboard():
#     tickers = load_tickers()
#     return render_template('dashboard.html', title='Dashboard', tickers=tickers)
#
# @bp.route('/result', methods=['GET', 'POST'])
# def about():
#     if request.method == 'GET':
#         return redirect(url_for('main.dashboard'))
#
#     amount = request.form.get('amount', type=float)
#     # ✅ gunakan getlist('tickers[]') agar bisa ambil banyak saham dari checkbox
#     selected_codes = request.form.getlist('tickers[]')
#
#     if not selected_codes:
#         flash("Pilih minimal satu saham.", "warning")
#         return redirect(url_for('main.dashboard'))
#
#     results = []
#     logs_per_ticker = {}
#
#     for code in selected_codes:
#         ticker = f"{code}.JK"
#         logs = []
#
#         def log_func(msg):
#             print(msg)
#             logs.append(str(msg))
#
#         try:
#             res = predict_ticker_xlstm(ticker, log=log_func)
#             # pastikan field wajib ada untuk optimizer
#             if not res.get("future_values"):
#                 raise ValueError("future_values kosong")
#             # siapkan expected_return sederhana (mean daily return dari future_values)
#             fv = res["future_values"]
#             arr = np.array(fv, dtype=float)
#
#             if len(arr) >= 2 and arr.min() > 0:
#                 rets = (arr[1:] - arr[:-1]) / arr[:-1]
#                 res["expected_return"] = float(np.nanmean(rets))
#             else:
#                 res["expected_return"] = 0.0
#
#             # beri nama yang dikenali optimizer:
#             res["future_predictions"] = res["future_values"]
#             results.append(res)
#         except Exception as e:
#             logs.append(f"[ERROR] {e}")
#             flash(f"Gagal memproses {ticker}: {e}", "danger")
#
#         logs_per_ticker[ticker] = logs
#
#         # >>> INI BAGIAN YANG MENGGANTIKAN all_prediction_results <<<
#     if results:
#         try:
#             df_alloc, summary = run_markowitz_optimization(results, amount)
#             df_alloc_records = df_alloc.to_dict(orient="records")
#         except Exception as e:
#             flash(f"Optimasi portofolio gagal: {e}", "danger")
#             df_alloc_records, summary = [], {}
#     else:
#         df_alloc_records, summary = [], {}
#         flash("Tidak ada hasil prediksi yang valid.", "warning")
#
#     return render_template(
#
#         'about.html',
#         title='Hasil Prediksi xLSTM',
#         amount=amount,
#         results=results,
#         logs_per_ticker=logs_per_ticker,
#         df_alloc=df_alloc_records,   # untuk tabel alokasi
#         summary=summary              # ringkasan (return, risk, Sharpe)
#     )
import csv, glob
import os
import pandas as pd
import json
from pathlib import Path
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from xLSTM.predict_ticker import predict_ticker_xlstm
from Optimizer.Markowitz import run_markowitz_optimization

try:
    import yfinance as yf
except ImportError:
    yf = None

bp = Blueprint('main', __name__)

# =======================
#  Utility
# =======================

def find_project_root():
    """Mencari root folder proyek berdasarkan lokasi file aplikasi."""
    # Mengambil parent dari folder tempat blueprint ini berada
    return Path(current_app.root_path).parent

def find_csv_path():
    """Cari file CSV daftar saham IHSG."""
    project_root = find_project_root()
    data_dir = project_root / "Data" / "Ticker"

    patterns = [
        "ihsg_tickers.csv", "ihsg_tickers.CSV",
        "IHSG_Tickers.csv", "IHSG_Tickers.CSV"
    ]
    for pat in patterns:
        p = data_dir / pat
        if p.exists():
            return str(p)

    # Fallback pencarian glob
    for p in glob.glob(str(data_dir / "ihsg_tickers.*")):
        if Path(p).is_file():
            return p
    return None

def get_tickers_grouped_by_sector():
    """Membaca file CSV per sektor dan mengembalikan dictionary."""
    sectors_data = {}

    # Gunakan path absolute agar lebih aman
    project_root = find_project_root()
    sector_dir = project_root / "Data" / "Ticker" / "Sectors"

    # Cek apakah folder ada
    if sector_dir.exists():
        for filename in os.listdir(sector_dir):
            if filename.endswith(".csv") or filename.endswith(".CSV"):
                # Nama sektor diambil dari nama file (misal: "Energy.csv" -> "Energy")
                sector_name = os.path.splitext(filename)[0] # Menghapus ekstensi .csv

                try:
                    file_path = sector_dir / filename
                    # Baca CSV
                    df = pd.read_csv(file_path)

                    # Pastikan nama kolom standar (lowercase) untuk keamanan
                    df.columns = [c.lower() for c in df.columns]

                    if 'code' in df.columns and 'name' in df.columns:
                        # Konversi ke list of dict: [{'code': 'AAAA', 'name': 'Perusahaan A'}, ...]
                        records = df[['code', 'name']].to_dict(orient='records')
                        sectors_data[sector_name] = records
                except Exception as e:
                    print(f"Error reading sector file {filename}: {e}")
    else:
        print(f"Warning: Folder sektor tidak ditemukan di {sector_dir}")

    return sectors_data

def load_tickers():
    """Baca CSV Utama → list[(code, name)]."""
    path = find_csv_path()
    tickers = []
    if not path:
        # Jangan flash error di sini jika ingin silent fail, tapi untuk debug biarkan saja
        # flash("File CSV daftar IHSG tidak ditemukan di Data/Ticker.", "warning")
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
        pass # Optional: flash warning
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
    # 1. Load data saham default (opsional, jika masih dipakai di bagian lain)
    tickers = load_tickers()

    # 2. Load data per sektor untuk Dropdown (INI YANG PENTING UTK KODE HTML BARU)
    sectors_data = get_tickers_grouped_by_sector()

    return render_template(
        'dashboard.html',
        title='Dashboard',
        tickers=tickers,
        sectors_data=sectors_data
    )

@bp.route('/result', methods=['GET', 'POST'])
def about():
    if request.method == 'GET':
        return redirect(url_for('main.dashboard'))

    amount = request.form.get('amount', type=float)
    # ✅ Ambil list tickers dari checkbox
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

        try:
            res = predict_ticker_xlstm(ticker, log=log_func)

            # Validasi hasil prediksi
            if not res.get("future_values"):
                raise ValueError("future_values kosong")

            # Hitung expected_return sederhana
            fv = res["future_values"]
            arr = np.array(fv, dtype=float)

            if len(arr) >= 2 and arr.min() > 0:
                rets = (arr[1:] - arr[:-1]) / arr[:-1]
                res["expected_return"] = float(np.nanmean(rets))
            else:
                res["expected_return"] = 0.0

            # Mapping nama field untuk optimizer
            res["future_predictions"] = res["future_values"]
            results.append(res)
        except Exception as e:
            logs.append(f"[ERROR] {e}")
            # Opsional: Jangan flash per saham error agar UI tidak penuh, cukup log
            print(f"Gagal memproses {ticker}: {e}")

        logs_per_ticker[ticker] = logs

    # Jalankan Optimasi Markowitz jika ada hasil prediksi valid
    if results:
        try:
            df_alloc, summary = run_markowitz_optimization(results, amount)
            df_alloc_records = df_alloc.to_dict(orient="records")
        except Exception as e:
            flash(f"Optimasi portofolio gagal: {e}", "danger")
            df_alloc_records, summary = [], {}
    else:
        df_alloc_records, summary = [], {}
        flash("Tidak ada hasil prediksi yang valid untuk saham yang dipilih.", "warning")

    return render_template(
        'about.html',
        title='Hasil Prediksi xLSTM',
        amount=amount,
        results=results,
        logs_per_ticker=logs_per_ticker,
        df_alloc=df_alloc_records,
        summary=summary
    )