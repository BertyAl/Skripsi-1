# from flask import Blueprint, render_template
#
# bp = Blueprint('main', __name__)
#
# @bp.route('/')
# def index():
#     return render_template('index.html', title='Home')
#
# @bp.route('/about')
# def about():
#     return render_template('about.html', title='About')
#
# @bp.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html', title='dashboard')

# from flask import Blueprint, render_template, request, redirect, url_for
# import yfinance as yf
#
# bp = Blueprint('main', __name__)

# app/main/routes.py
# import os, csv
# from flask import Blueprint, render_template, request, redirect, url_for, flash
#
# try:
#     import yfinance as yf
# except ImportError:
#     yf = None  # biar app tetap jalan, nanti kita cek saat pemakaian
#
# bp = Blueprint('main', __name__)
#
#
# # 👉 Contoh subset ticker IHSG (silakan tambah sendiri)
# IHSG_TICKERS = [
#     ("BBCA", "Bank Central Asia Tbk"),
#     ("BBRI", "Bank Rakyat Indonesia Tbk"),
#     ("BMRI", "Bank Mandiri Tbk"),
#     ("BBNI", "Bank Negara Indonesia Tbk"),
#     ("TLKM", "Telkom Indonesia Tbk"),
#     ("ASII", "Astra International Tbk"),
#     ("UNVR", "Unilever Indonesia Tbk"),
#     ("ICBP", "Indofood CBP Sukses Makmur Tbk"),
#     ("INDF", "Indofood Sukses Makmur Tbk"),
#     ("ANTM", "Aneka Tambang Tbk"),
#     ("TOWR", "Sarana Menara Nusantara Tbk"),
#     ("ADRO", "Adaro Energy Indonesia Tbk"),
#     ("MDKA", "Merdeka Copper Gold Tbk"),
#     ("GOTO", "GoTo Gojek Tokopedia Tbk"),
#     ("BUKA", "Bukalapak.com Tbk"),
# ]
#
# def to_yf_symbol(code: str) -> str:
#     return f"{code}.JK"
#
# @bp.route('/', methods=['GET'])
# def index():
#     # Kirim daftar ticker ke template untuk dropdown
#     return render_template('index.html', title='Home', tickers=IHSG_TICKERS)
#
# @bp.route('/about', methods=['GET', 'POST'])
# def about():
#     if request.method == 'GET':
#         # Arahkan balik ke form bila diakses langsung
#         return redirect(url_for('main.index'))
#
#     # Ambil input dari form
#     amount = request.form.get('amount', type=float)
#     selected_codes = request.form.getlist('tickers')  # misal ['BBRI','TLKM']
#     if not selected_codes:
#         # Jika tidak memilih apa-apa, balik ke form
#         return redirect(url_for('main.index'))
#
#     yf_symbols = [to_yf_symbol(c) for c in selected_codes]
#
#     # (Opsional) Ambil sedikit data untuk validasi (bisa Anda hapus bila tidak perlu)
#     try:
#         df = yf.download(yf_symbols, period="6mo", interval="1d", auto_adjust=True, progress=False)
#         # NOTE: Lanjutkan ke logika optimasi di sini…
#     except Exception as e:
#         # Gagal ambil data — tetap render halaman hasil dengan info yang ada
#         df = None
#
#     return render_template('about.html',
#                            title='About',
#                            amount=amount,
#                            selected_codes=selected_codes,
#                            yf_symbols=yf_symbols,
#                            has_data=(df is not None and not df.empty))
#
# @bp.route('/dashboard', methods=['GET'])
# def dashboard():
#     return render_template('dashboard.html', title='dashboard')

# app/main/routes.py
import os, csv, glob
from pathlib import Path
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app

try:
    import yfinance as yf
except ImportError:
    yf = None

bp = Blueprint('main', __name__)

def find_csv_path():
    """
    Cari file CSV daftar IHSG dengan beberapa strategi:
    - Prioritas: <project_root>/Data/Ticker/ihsg_tickers.CSV (sesuai yang kamu sebut)
    - Fallback:  glob "ihsg_tickers.*" di folder Data/Ticker
    """
    # root project = parent dari folder "app"
    project_root = Path(current_app.root_path).parent

    # target folder
    data_dir = project_root / "Data" / "Ticker"

    # kandidat 1: persis seperti yang kamu tulis
    candidate1 = data_dir / "ihsg_tickers.CSV"
    if candidate1.exists():
        return str(candidate1)

    # kandidat 2: case-insensitive glob di Data/Ticker
    patterns = [
        str(data_dir / "ihsg_tickers.csv"),
        str(data_dir / "ihsg_tickers.CSV"),
        str(data_dir / "IHSG_Tickers.csv"),
        str(data_dir / "IHSG_Tickers.CSV"),
        str(data_dir / "ihsg_tickers.*"),
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            if Path(p).is_file():
                return p

    # terakhir: None -> biar nanti di-handle dengan flash
    return None

def load_tickers():
    """
    Baca CSV -> list[(code, name)].
    Struktur CSV minimal: header 'code,name'
    """
    path = find_csv_path()
    tickers = []

    if not path:
        flash("File CSV daftar IHSG tidak ditemukan di Data/Ticker (cari ihsg_tickers.CSV).", "warning")
        return tickers

    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip().upper()
                name = (row.get("name") or "").strip()
                if code:
                    tickers.append((code, name or code))
    except UnicodeDecodeError:
        # fallback encoding Windows
        with open(path, newline='', encoding='cp1252') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip().upper()
                name = (row.get("name") or "").strip()
                if code:
                    tickers.append((code, name or code))
    except Exception as e:
        flash(f"Gagal membaca CSV: {e}", "danger")

    if not tickers:
        flash("CSV terbaca, tetapi tidak ada baris valid (cek header 'code,name').", "warning")

    return tickers

def to_yf_symbol(code: str) -> str:
    return f"{code}.JK"

@bp.route('/', methods=['GET'])
def index():
    # jangan redirect — render halaman index
    return render_template('index.html', title='Home')

# @bp.route('/', methods=['GET'])
# def index():
#     # Arahkan semua ke dashboard agar form hanya di sana
#     return redirect(url_for('main.dashboard'))

@bp.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html', title='Dashboard', tickers=load_tickers())

# @bp.route('/about', methods=['GET','POST'])
# def about():
#     return render_template('about.html',title='About')

# @bp.route('/about', methods=['GET','POST'])
# def about():
#     if request.method == 'GET':
#         return redirect(url_for('main.dashboard'))
#
#     amount = request.form.get('amount', type=float)
#     selected_codes = request.form.getlist('tickers')
#
#     if not selected_codes:
#         flash("Pilih minimal satu saham.", "warning")
#         return redirect(url_for('main.dashboard'))
#
#     if yf is None:
#         flash("Paket 'yfinance' belum terpasang. Jalankan: pip install yfinance pandas numpy", "danger")
#         return redirect(url_for('main.dashboard'))
#
#     yf_symbols = [to_yf_symbol(c) for c in selected_codes]
#     try:
#         df = yf.download(yf_symbols, period="6mo", interval="1d", auto_adjust=True, progress=False)
#         has_data = (not df.empty)
#     except Exception as e:
#         has_data = False
#         flash(f"Gagal mengambil data yfinance: {e}", "danger")
#
#     return render_template('about.html',
#                            title='About',
#                            amount=amount,
#                            selected_codes=selected_codes,
#                            yf_symbols=yf_symbols,
#                            has_data=has_data)
# from xLSTM.predict_ticker import predict_stock_xlstm
#
# @bp.route('/about', methods=['GET','POST'])
# def about():
#     if request.method == 'GET':
#         return redirect(url_for('main.dashboard'))
#
#     amount = request.form.get('amount', type=float)
#     selected_codes = request.form.getlist('tickers')
#
#     if not selected_codes:
#         flash("Pilih minimal satu saham.", "warning")
#         return redirect(url_for('main.dashboard'))
#
#     results = []
#     for code in selected_codes:
#         ticker = f"{code}.JK"
#         try:
#             result = predict_stock_xlstm(ticker)
#             results.append(result)
#         except Exception as e:
#             flash(f"Gagal memprediksi {ticker}: {e}", "danger")
#
#     return render_template('about.html',
#                            title='Hasil Prediksi xLSTM',
#                            amount=amount,
#                            results=results)
from xLSTM.predict_ticker import predict_ticker_xlstm

@bp.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'GET':
        return redirect(url_for('main.dashboard'))

    amount = request.form.get('amount', type=float)
    selected_codes = request.form.getlist('tickers')

    if not selected_codes:
        flash("Pilih minimal satu saham.", "warning")
        return redirect(url_for('main.dashboard'))

    results = []
    logs_per_ticker = {}

    for code in selected_codes:
        ticker = f"{code}.JK"
        logs = []
        def log_func(msg):  # logger untuk tiap ticker
            print(msg)       # tampil di terminal Flask
            logs.append(str(msg))  # simpan ke halaman HTML

        try:
            result = predict_ticker_xlstm(ticker, log=log_func)
            results.append(result)
        except Exception as e:
            logs.append(f"[ERROR] {e}")
            flash(f"Gagal memproses {ticker}: {e}", "danger")

        logs_per_ticker[ticker] = logs

    return render_template('about.html',
                           title='Hasil Prediksi xLSTM',
                           amount=amount,
                           results=results,
                           logs_per_ticker=logs_per_ticker)
