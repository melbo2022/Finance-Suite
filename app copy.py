#金融電卓(積立、ローン、年金）を表示させるか、非表示にするかを選択してください（30行目）
#-----------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
from __future__ import annotations

# Combined app: Option Borrow vs Hedge, FX Options Visualizer (Put/Call), and Investment calculators
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import io, base64
import numpy as np
import math
from datetime import datetime as dt, date
from math import isfinite, log, log1p, exp

# ------- Matplotlib (headless, portable fonts) -------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib import rcParams
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False

#-------------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "replace-this-key"

#金融電卓の表示・非表示フラグ-----------------------------------------------------------
app.config["SHOW_FIN_TOOLS"] = True #←　非表示にするときはFalse

# どのテンプレでも使える共通コンテキスト
@app.context_processor
def inject_flags():
    return {
        "show_fin_tools": app.config.get("SHOW_FIN_TOOLS", True)
    }

#-----------------------------------------------------------------------------------------

# =====================================================
# ================ Option Calcurate =================
# =====================================================

def to_float(x, default):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

#-------------------------------------------------------------------------------------------------
def _set_ylim_tight(ax, arrays, pad=0.08):
    """
    Y軸レンジを、与えた配列群（損益カーブ）だけで決める。pad は上下の余白率。
    """
    y_min = min(np.min(a) for a in arrays)
    y_max = max(np.max(a) for a in arrays)
    if y_max <= y_min:
        y_max = y_min + 1.0  # 退避
    span = y_max - y_min
    ax.set_ylim(y_min - pad * span, y_max + pad * span)


#----------------------------------------------------------------------------------------------------
# app.py（ホームルート）
@app.route("/")
def home():
    return render_template("home.html", show_fin_tools=app.config["SHOW_FIN_TOOLS"])


# -----------------------------------------------------------------------------------------------

# =====================================================
# ============== FX Options Visualizer ================
# =====================================================


# ---------------- ユーティリティ --------------------------------------------------------------------------

# ポイント数のクランプ関数
def clamp_points(points):
    """
    グラフの分解能（points）を安全な範囲に丸める。
    51 ～ 2001 の範囲に収め、数値化できない場合は既定 251。
    """
    try:
        p = int(float(points))
    except Exception:
        p = 251
    return max(51, min(p, 2001))


# ---------------- 数式（GK/正規CDF） -------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    """標準正規の累積分布。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def garman_kohlhagen_put(S0, K, r_dom, r_for, sigma, T):
    """
    FXプット（Garman–Kohlhagen）。返り値は JPY/USD（1USDあたりのプレミアム）。
    r_dom, r_for, sigma は年率（実数）、T は年（= months/12）。
    """
    if sigma <= 0.0 or T <= 0.0:
        return max(K - S0, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r_dom - r_for + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    from_term = K * math.exp(-r_dom * T) * norm_cdf(-d2)
    spot_term = S0 * math.exp(-r_for * T) * norm_cdf(-d1)
    return from_term - spot_term  # JPY per USD

def garman_kohlhagen_call(S0, K, r_dom, r_for, sigma, T):
    """
    FXコール（Garman–Kohlhagen）。返り値は JPY/USD（1USDあたりのプレミアム）。
    r_dom, r_for, sigma は年率（実数）、T は年（= months/12）。
    """
    if sigma <= 0.0 or T <= 0.0:
        return max(S0 - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r_dom - r_for + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    spot_term = S0 * math.exp(-r_for * T) * norm_cdf(d1)
    from_term = K * math.exp(-r_dom * T) * norm_cdf(d2)
    return spot_term - from_term  # JPY per USD

def _arange_inclusive(start: float, stop: float, step: float):
    """
    start から stop まで step 刻みで生成（stop を含める）。
    浮動小数の誤差を吸収するように少し余裕を持って計算します。
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if start > stop:
        start, stop = stop, start

    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    arr = start + step * np.arange(n)
    # 誤差で stop を僅かに超える値を除去
    return arr[arr <= stop + 1e-9]

# ---------------- 損益ロジック -------------------------------------------------------------------------------------------------
def payoff_components_put(S_T, S0, K, premium, qty):
    """
    現物USD、プット買い、合成（Protective Put）の各損益を返す（JPY建て）。
    premium は JPY/USD、qty は USD 数量。
    """
    spot_pl = (S_T - S0) * qty
    put_pl  = (np.maximum(K - S_T, 0.0) - premium) * qty
    combo_pl = spot_pl + put_pl
    return {"spot": spot_pl, "put": put_pl, "combo": combo_pl}

def payoff_components_call(S_T, S0, K, premium, qty):
    """
    Call用の損益内訳（JPY）。
    spot: USDショートの損益、opt: コール買い（プレミアム込み）、combo: 合成
    """
    spot_pl = (S_T - S0) * (-qty)                          # USDショートの損益
    call_pl = (np.maximum(S_T - K, 0.0) - premium) * qty   # コール損益（プレミアム込み）
    combo_pl = spot_pl + call_pl
    return {"spot": spot_pl, "opt": call_pl, "combo": combo_pl}

def build_grid_and_rows_put(S0, K, premium, qty, smin, smax, points):
    """
    グリッド（S_T配列）とグラフ・表に使うデータを生成。
    """
    if smin >= smax:
        smin, smax = (min(smin, smax), max(smin, smax) + 1.0)
    points = clamp_points(points)
    S_T = np.linspace(smin, smax, points)
    pl = payoff_components_put(S_T, S0, K, premium, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "put":  float(pl["put"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(points)]
    return S_T, pl, rows

def build_grid_and_rows_call(S0, K, premium, qty, smin, smax, points):
    """
    Call用のグリッド生成＆テーブル行作成。
    """
    if smin >= smax:
        smin, smax = (min(smin, smax), max(smin, smax) + 1.0)
    points = clamp_points(points)
    S_T = np.linspace(smin, smax, points)
    pl = payoff_components_call(S_T, S0, K, premium, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "opt":  float(pl["opt"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(points)]
    return S_T, pl, rows

# ---------------- 描画（メイン：M表記へ） --------------------------------------------------------------------------------
def _format_y_as_m(ax):
    """Y軸を百万円（M）表記にする。"""
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

def draw_chart_put(S_T, pl, S0, K, floor_value):
    """
    Protective Put の損益グラフを描画。Y軸はM（百万円）表記。
    """
    fig = plt.figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot(111)

    # ライン
    ax.plot(S_T, pl["spot"], label="Spot USD P/L (vs today)")
    ax.plot(S_T, pl["put"],  label="Long Put P/L (incl. premium)")
    ax.plot(S_T, pl["combo"], linewidth=2, label="Protective Put Combo P/L")

    # 基準線
    ax.axhline(0, linewidth=1)

    # 参考線：S0 / K（ラベルも簡潔に）
    ax.axvline(S0, linestyle="--", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)

    # Loss floor の水平線（線のみ）
    ax.axhline(floor_value, linestyle=":", linewidth=1)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Protective Put: P/L vs Terminal USD/JPY")
    _format_y_as_m(ax)  # ★ M表記
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_chart_call(S_T, pl, S0, K, floor_value):
    """
    Protective Call の損益グラフを描画。Y軸はM（百万円）表記。
    """
    fig = plt.figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot(111)

    # ライン
    ax.plot(S_T, pl["spot"], label="Short USD Spot P/L (vs today)")
    ax.plot(S_T, pl["opt"],  label="Long Call P/L (incl. premium)")
    ax.plot(S_T, pl["combo"], linewidth=2, label="Protective Call Combo P/L")

    # 基準線
    ax.axhline(0, linewidth=1)

    # 参考線：S0 / K（簡潔なラベル）
    ax.axvline(S0, linestyle="--", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)

    # Loss floor（線のみ）
    ax.axhline(floor_value, linestyle=":", linewidth=1)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Protective Call: P/L vs Terminal USD/JPY")
    _format_y_as_m(ax)  # ★ M表記
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

# ---------------- 比較グラフ（2本：Combo vs 借入利息、Y軸M表記） ----------------
def draw_compare_put(S_T, combo_pl, finance_cost):
    """
    Protective Put Combo と 借入利息(一定額) の比較グラフ。
    借入利息ラインはコストとして見やすいよう -finance_cost で水平線にする。
    """
    fig = plt.figure(figsize=(7, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Protective Put Combo P/L")
    ax.axhline(-finance_cost, linestyle="--", linewidth=1.5, label="Borrow Cost (flat)")
    ax.axhline(0, linewidth=1)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Compare: Put Combo vs Borrow")
    _format_y_as_m(ax)  # ★ M表記
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_compare_call(S_T, combo_pl, finance_cost):
    """
    Protective Call Combo と 借入利息(一定額) の比較グラフ。
    借入利息ラインは -finance_cost の水平線。
    """
    fig = plt.figure(figsize=(7, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2,color="green",label="Protective Call Combo P/L")
    ax.axhline(-finance_cost, linestyle="--", linewidth=1.5, label="Borrow Cost (flat)")
    ax.axhline(0, linewidth=1)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Compare: Call Combo vs Borrow")
    _format_y_as_m(ax)  # ★ M表記
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

# ---------------- 画面ルート（PUT） ----------------
@app.route("/fx/put", methods=["GET", "POST"])
def fx_put():
    """
    Protective Put 可視化画面。
    Premiumはユーザ入力ではなく、Volatility/金利/満期からGK式で算出。
    借入率（年率%）と期間（月）は表示用の利息計算に用いる。
    """
    defaults = dict(
        S0=150.0, K=148.0,
        vol=10.0,                 # 年率％
        r_dom=1.6,                # JPY金利（年率％）
        r_for=4.2,                # USD金利（年率％）
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0,               # 満期（月）
        borrow_rate=4.2           # 借入年率（％）…表示用
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K = fget("K", float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
        borrow_rate = fget("borrow_rate", float, defaults["borrow_rate"])
    else:
        S0 = defaults["S0"]; K = defaults["K"]; vol = defaults["vol"]
        r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]; borrow_rate = defaults["borrow_rate"]

    points = clamp_points(points)

    # --- GK式でプレミアム( JPY/USD )を算出 ---
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    premium = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # JPY/USD

    # グリッドと損益（算出premiumを使用）
    S_T, pl, rows = build_grid_and_rows_put(S0, K, premium, qty, smin, smax, points)

    # Loss floor（従来通り）
    floor_value = (K - S0 - premium) * qty

    # オプション料と借入利息（表示用）
    premium_jpy = premium * qty
    notional_jpy = S0 * qty
    premium_pct_of_qty = (premium_jpy / notional_jpy * 100.0) if notional_jpy > 0 else 0.0

    # 借入利息は「Quantityに対する利息（元本= S0×Quantity）」で算出
    finance_jpy = qty * S0 * (borrow_rate / 100.0) * (months / 12.0)

    # メイングラフ（M表記）
    fig = draw_chart_put(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 比較グラフ（Combo vs Borrow、M表記）
    fig_cmp = draw_compare_put(S_T, pl["combo"], finance_jpy)
    buf2 = io.BytesIO(); fig_cmp.savefig(buf2, format="png"); plt.close(fig_cmp); buf2.seek(0)
    png_b64_put_compare = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_put.html",
        png_b64=png_b64,
        png_b64_put_compare=png_b64_put_compare,
        # 入力（Vol/金利/満期）
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力（算出値）
        premium=premium,                    # JPY/USD（CSVや表示に供給）
        floor=floor_value,
        premium_cost=premium_jpy,
        finance_cost=finance_jpy,
        premium_pct=premium_pct_of_qty,
        rows=rows
    )

@app.route("/fx/download_csv_put", methods=["POST"])
def fx_download_csv_put():
    """
    Protective Put のグリッドをCSVでダウンロード。
    ※ premium はテンプレート側から算出済みの値がPOSTされる前提。
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    S0 = fget("S0", float, 150.0)
    K = fget("K", float, 148.0)
    premium = fget("premium", float, 0.74)  # 画面で算出された値が hidden で渡ってくる
    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 130.0)
    smax = fget("smax", float, 160.0)
    points = clamp_points(fget("points", float, 251))

    S_T, pl, _ = build_grid_and_rows_put(S0, K, premium, qty, smin, smax, points)

    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["S_T(USD/JPY)", "Spot_PnL(JPY)", "Put_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(points):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}"
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data, mimetype="text/csv",
        as_attachment=True, download_name="protective_put_pnl.csv"
    )

# ---------------- 画面ルート（CALL） ----------------
@app.route("/fx/call", methods=["GET", "POST"])
def fx_call():
    """
    Call 版：Premium は Volatility/金利/満期から GK 式で算出。
    オプション料の名目比（%）と、借入利息（元本= S0×Quantity）も表示。
    """
    defaults = dict(
        S0=150.0, K=152.0,
        vol=10.0,               # 年率％（ボラ）
        r_dom=1.6,              # JPY金利（年率％）
        r_for=4.2,              # USD金利（年率％）
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0,             # 満期（月）
        borrow_rate=4.2         # 借入年率（％）…表示用
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K = fget("K", float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
        borrow_rate = fget("borrow_rate", float, defaults["borrow_rate"])
    else:
        S0 = defaults["S0"]; K = defaults["K"]; vol = defaults["vol"]
        r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]; borrow_rate = defaults["borrow_rate"]

    points = clamp_points(points)

    # --- GK式でコール・プレミアム( JPY/USD )を算出 ---
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    premium = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)

    # グリッドと損益（算出premiumを使用）
    S_T, pl, rows = build_grid_and_rows_call(S0, K, premium, qty, smin, smax, points)

    # Loss floor（従来式）
    floor_value = (S0 - K - premium) * qty

    # 表示用：オプション料と借入利息・名目比
    premium_jpy = premium * qty
    notional_jpy = S0 * qty
    premium_pct_of_qty = (premium_jpy / notional_jpy * 100.0) if notional_jpy > 0 else 0.0
    finance_jpy = qty * S0 * (borrow_rate / 100.0) * (months / 12.0)

    # メイングラフ（M表記）
    fig = draw_chart_call(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 比較グラフ（Combo vs Borrow、M表記）
    fig_cmp = draw_compare_call(S_T, pl["combo"], finance_jpy)
    buf2 = io.BytesIO(); fig_cmp.savefig(buf2, format="png"); plt.close(fig_cmp); buf2.seek(0)
    png_b64_call_compare = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_call.html",
        png_b64=png_b64,
        png_b64_call_compare=png_b64_call_compare,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力（算出値）
        premium=premium,                 # JPY/USD
        floor=floor_value,
        premium_cost=premium_jpy,
        finance_cost=finance_jpy,
        premium_pct=premium_pct_of_qty,
        rows=rows
    )

@app.route("/fx/download_csv_call", methods=["POST"])
def fx_download_csv_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default
    S0 = fget("S0", float, 150.0)
    K = fget("K", float, 152.0)
    premium = fget("premium", float, 0.62)  # 画面で算出された値（hidden）を受ける
    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 130.0)
    smax = fget("smax", float, 160.0)
    points = clamp_points(fget("points", float, 251))

    S_T, pl, _ = build_grid_and_rows_call(S0, K, premium, qty, smin, smax, points)

    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["S_T(USD/JPY)", "ShortSpot_PnL(JPY)", "Call_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(points):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['opt'][i]:.6f}",
            f"{pl['combo'][i]:.6f}"
        ])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data, mimetype="text/csv",
        as_attachment=True, download_name="protective_call_pnl.csv"
    )

#---------------------------------------------------------------------------------------------------------------------------------------
#PUTの売り
# =====================================================
# ==================== Put 売り版 =====================
# =====================================================

# 損益ロジック（Put 売り）
def payoff_components_put_short(S_T, S0, K, premium, qty):
    """
    現物USD（ロング）、プット売り、合成（Spot + Short Put）の各損益（JPY）。
    premium は JPY/USD（受取プレミアム）、qty は USD 数量。
    """
    spot_pl = (S_T - S0) * qty
    short_put_pl = (premium - np.maximum(K - S_T, 0.0)) * qty  # 売り：受取 - 支払
    combo_pl = spot_pl + short_put_pl
    return {"spot": spot_pl, "short_put": short_put_pl, "combo": combo_pl}

def build_grid_and_rows_put_short(S0, K, premium, qty, smin, smax, points):
    """Putショート用グリッドとテーブル行。"""
    if smin >= smax:
        smin, smax = (min(smin, smax), max(smin, smax) + 1.0)
    points = clamp_points(points)
    S_T = np.linspace(smin, smax, points)
    pl = payoff_components_put_short(S_T, S0, K, premium, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "short_put": float(pl["short_put"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(points)]
    return S_T, pl, rows

# 描画（Put 売り）
def draw_chart_put_short(S_T, pl, S0, K, floor_value):
    """
    Spot + Short Put（Covered Put）の損益グラフ。Y軸はM表記。
    """
    fig = plt.figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["spot"],      label="Spot USD P/L (vs today)")
    ax.plot(S_T, pl["short_put"], label="Short Put P/L (incl. premium)")
    ax.plot(S_T, pl["combo"],     linewidth=2, label="Spot + Short Put Combo P/L")

    # ← 損益データのみでY軸を決定（Loss floorは無視）
    _set_ylim_tight(ax, [pl["spot"], pl["short_put"], pl["combo"]])

    # 基準線
    ax.axhline(0, linewidth=1)

    # 縦の参考線（S0/K）
    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1)
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)

    # Loss floor：レンジ内なら線、外なら注記
    ymin, ymax = ax.get_ylim()
    if ymin <= floor_value <= ymax:
        ax.axhline(floor_value, linestyle=":", linewidth=1)
    else:
        ax.annotate(f"Loss floor ≈ {floor_value/1e6:.1f}M (below)",
                    xy=(S0, ymin), xytext=(6, -14), textcoords="offset points",
                    va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Covered Put (Spot + Short Put): P/L vs Terminal USD/JPY")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_compare_put_short(S_T, combo_pl, finance_cost):
    """
    Covered Put Combo と 借入利息の比較（Y軸M表記）。
    借入利息ラインはレンジ内にあるときだけ描画。レンジ外なら注記。
    """
    fig = plt.figure(figsize=(7, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    # 1) まずコンボ線だけ描画し、そのデータでY軸を決める
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Covered Put Combo P/L")
    _set_ylim_tight(ax, [combo_pl])  # 借入コストはスケーリングに使わない

    # 2) 基準線
    ax.axhline(0, linewidth=1)

    # 3) 借入利息ライン：レンジ内なら線、外なら注記
    ymin, ymax = ax.get_ylim()
    borrow_level = -finance_cost
    if ymin <= borrow_level <= ymax:
        ax.axhline(borrow_level, linestyle="--", linewidth=1.5, label="Borrow Cost (flat)")
    else:
        pos = "below" if borrow_level < ymin else "above"
        ax.annotate(f"Borrow cost ≈ {borrow_level/1e6:.1f}M ({pos})",
                    xy=(S_T[len(S_T)//2], ymin if pos=='below' else ymax),
                    xytext=(6, -14 if pos=='below' else 6),
                    textcoords="offset points", va="top" if pos=='below' else "bottom",
                    ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Compare: Covered Put Combo vs Borrow")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

# 画面ルート（PUT 売り）
@app.route("/fx/put-sell", methods=["GET", "POST"])
def fx_put_sell():
    """
    Putの売り（ショート）版。
    Premium は Vol/金利/満期からGK式で算出した理論価格を「受取」として扱う。
    既存Put買い版と同じ入出力・表示体系。
    """
    defaults = dict(
        S0=150.0, K=148.0,
        vol=10.0,                 # 年率％
        r_dom=1.6,                # JPY金利（年率％）
        r_for=4.2,                # USD金利（年率％）
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0,               # 満期（月）
        borrow_rate=4.2           # 借入年率（％）
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K = fget("K", float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
        borrow_rate = fget("borrow_rate", float, defaults["borrow_rate"])
    else:
        S0 = defaults["S0"]; K = defaults["K"]; vol = defaults["vol"]
        r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]; borrow_rate = defaults["borrow_rate"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）。売りなので「受取」として同額を用いる。
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    premium = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_put_short(S0, K, premium, qty, smin, smax, points)

    # ① 理論上の最悪損益（参考）
    floor_value = (-S0 - K + premium) * qty

    # ② レンジ内の最小損益（画面表示用）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor = float(pl["combo"][idx_min])    # 最小損益（JPY）
    range_floor_st = float(S_T[idx_min])         # その時のレート S_T

    # 受取プレミアムと借入利息（表示用）
    premium_jpy = premium * qty          # 受取額
    notional_jpy = S0 * qty
    premium_pct_of_qty = (premium_jpy / notional_jpy * 100.0) if notional_jpy > 0 else 0.0

    finance_jpy = qty * S0 * (borrow_rate / 100.0) * (months / 12.0)

    # メイングラフ
    fig = draw_chart_put_short(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 比較グラフ（Combo vs Borrow）
    fig_cmp = draw_compare_put_short(S_T, pl["combo"], finance_jpy)
    buf2 = io.BytesIO(); fig_cmp.savefig(buf2, format="png"); plt.close(fig_cmp); buf2.seek(0)
    png_b64_put_short_compare = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_put_sell.html",
        png_b64=png_b64,
        png_b64_put_short_compare=png_b64_put_short_compare,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力（算出値）
        premium=premium,                    # JPY/USD
        floor=floor_value,                  # 理論 floor（参考）
        premium_recv=premium_jpy,           # 受取額
        finance_cost=finance_jpy,
        premium_pct=premium_pct_of_qty,
        rows=rows,
        # 追加（レンジ内の最小値）
        range_floor=range_floor,
        range_floor_st=range_floor_st
    )

# CSV（Put 売り）
@app.route("/fx/download_csv_put_sell", methods=["POST"])
def fx_download_csv_put_sell():
    """
    Putショート版のグリッドCSVダウンロード。
    ※ premium は画面側で算出済みの値がPOSTされる前提。
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    S0 = fget("S0", float, 150.0)
    K = fget("K", float, 148.0)
    premium = fget("premium", float, 0.74)
    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 130.0)
    smax = fget("smax", float, 160.0)
    points = clamp_points(fget("points", float, 251))

    S_T, pl, _ = build_grid_and_rows_put_short(S0, K, premium, qty, smin, smax, points)

    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["S_T(USD/JPY)", "Spot_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(points):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['short_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}"
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data, mimetype="text/csv",
        as_attachment=True, download_name="covered_put_pnl.csv"
    )



# ==================== Call 売り版（Covered Call） ================================================================================================

def payoff_components_call_short(S_T, S0, K, premium, qty):
    """
    現物USD（ロング）、コール売り、合成（Spot + Short Call）の各損益（JPY）。
    premium は JPY/USD（受取プレミアム）、qty は USD 数量。
    """
    spot_pl = (S_T - S0) * qty
    short_call_pl = (premium - np.maximum(S_T - K, 0.0)) * qty  # 売り：受取 - 支払
    combo_pl = spot_pl + short_call_pl
    return {"spot": spot_pl, "short_call": short_call_pl, "combo": combo_pl}

def build_grid_and_rows_call_short(S0, K, premium, qty, smin, smax, points):
    """Callショート用グリッドとテーブル行。"""
    if smin >= smax:
        smin, smax = (min(smin, smax), max(smin, smax) + 1.0)
    points = clamp_points(points)
    S_T = np.linspace(smin, smax, points)
    pl = payoff_components_call_short(S_T, S0, K, premium, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "short_call": float(pl["short_call"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(points)]
    return S_T, pl, rows

def draw_chart_call_short(S_T, pl, S0, K, floor_value):
    """
    Spot + Short Call（Covered Call）の損益グラフ。Y軸はM表記。
    """
    fig = plt.figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["spot"],        label="Spot USD P/L (vs today)")
    ax.plot(S_T, pl["short_call"],  label="Short Call P/L (incl. premium)")
    ax.plot(S_T, pl["combo"], linewidth=2, label="Spot + Short Call Combo P/L")

    # 損益カーブのみでY軸決定（極端値に引っ張られない）
    _set_ylim_tight(ax, [pl["spot"], pl["short_call"], pl["combo"]])

    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1)
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)

    ymin, ymax = ax.get_ylim()
    if ymin <= floor_value <= ymax:
        ax.axhline(floor_value, linestyle=":", linewidth=1)
    else:
        ax.annotate(f"Loss floor ≈ {floor_value/1e6:.1f}M (outside range)",
                    xy=(S0, ymin), xytext=(6, -14), textcoords="offset points",
                    va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Covered Call (Spot + Short Call): P/L vs Terminal USD/JPY")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_compare_call_short(S_T, combo_pl, finance_cost):
    """
    Covered Call Combo と 借入利息の比較（Y軸M表記）。
    借入利息ラインはレンジ内にあるときだけ描画。外なら注記。
    """
    fig = plt.figure(figsize=(7, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Covered Call Combo P/L")
    _set_ylim_tight(ax, [combo_pl])   # 借入コストはスケーリングに使わない
    ax.axhline(0, linewidth=1)

    ymin, ymax = ax.get_ylim()
    borrow_level = -finance_cost
    if ymin <= borrow_level <= ymax:
        ax.axhline(borrow_level, linestyle="--", linewidth=1.5, label="Borrow Cost (flat)")
    else:
        pos = "below" if borrow_level < ymin else "above"
        ax.annotate(f"Borrow cost ≈ {borrow_level/1e6:.1f}M ({pos})",
                    xy=(S_T[len(S_T)//2], ymin if pos=='below' else ymax),
                    xytext=(6, -14 if pos=='below' else 6),
                    textcoords="offset points",
                    va="top" if pos=='below' else "bottom", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Compare: Covered Call Combo vs Borrow")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

@app.route("/fx/call-sell", methods=["GET", "POST"])
def fx_call_sell():
    """
    Callの売り（Covered Call）。
    Premium は GK式コール理論値を「受取」として扱う。
    UI・出力は Put売り版と同じ構成。
    """
    defaults = dict(
        S0=150.0, K=152.0,
        vol=10.0,                 # 年率％
        r_dom=1.6,                # JPY金利（年率％）
        r_for=4.2,                # USD金利（年率％）
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0,               # 満期（月）
        borrow_rate=4.2           # 借入年率（％）
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K = fget("K", float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
        borrow_rate = fget("borrow_rate", float, defaults["borrow_rate"])
    else:
        S0 = defaults["S0"]; K = defaults["K"]; vol = defaults["vol"]
        r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]; borrow_rate = defaults["borrow_rate"]

    points = clamp_points(points)

    # GK式コール理論プレミアム（受取）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    premium = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_call_short(S0, K, premium, qty, smin, smax, points)

    # ① 理論上の最悪損益（S_T→0 で最小）：(-S0 + premium) × qty
    floor_value = (-S0 + premium) * qty

    # ② レンジ内の最小損益（画面表示用）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor = float(pl["combo"][idx_min])
    range_floor_st = float(S_T[idx_min])

    # 受取プレミアムと借入利息（表示用）
    premium_jpy = premium * qty
    notional_jpy = S0 * qty
    premium_pct_of_qty = (premium_jpy / notional_jpy * 100.0) if notional_jpy > 0 else 0.0
    finance_jpy = qty * S0 * (borrow_rate / 100.0) * (months / 12.0)

    # グラフ
    fig = draw_chart_call_short(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig_cmp = draw_compare_call_short(S_T, pl["combo"], finance_jpy)
    buf2 = io.BytesIO(); fig_cmp.savefig(buf2, format="png"); plt.close(fig_cmp); buf2.seek(0)
    png_b64_call_short_compare = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_call_sell.html",
        png_b64=png_b64,
        png_b64_call_short_compare=png_b64_call_short_compare,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力
        premium=premium,                  # JPY/USD
        floor=floor_value,                # 理論 floor（参考）
        premium_recv=premium_jpy,         # 受取額
        finance_cost=finance_jpy,
        premium_pct=premium_pct_of_qty,
        rows=rows,
        range_floor=range_floor,          # レンジ内最小損益
        range_floor_st=range_floor_st
    )

@app.route("/fx/download_csv_call_sell", methods=["POST"])
def fx_download_csv_call_sell():
    """Covered Call のグリッドCSVダウンロード。"""
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    S0 = fget("S0", float, 150.0)
    K = fget("K", float, 152.0)
    premium = fget("premium", float, 0.74)
    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 130.0)
    smax = fget("smax", float, 160.0)
    points = clamp_points(fget("points", float, 251))

    S_T, pl, _ = build_grid_and_rows_call_short(S0, K, premium, qty, smin, smax, points)

    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["S_T(USD/JPY)", "Spot_PnL(JPY)", "ShortCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(points):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['combo'][i]:.6f}"
        ])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="covered_call_pnl.csv")
#-------------------------------------------------------------------------------------------------------------------------------------------------
# ==債権ヘッジ===================================================
# ================= Zero-Cost（Collar） ================
# =====================================================

# 損益ロジック（ゼロコスト：Long Put + Short Call）
def payoff_components_zero_cost(S_T, S0, Kp, Kc, prem_put, prem_call, qty):
    """
    現物USD（ロング）、プット買い、コール売り、合成（Spot + Long Put + Short Call）の各損益（JPY）。
    prem_put / prem_call は JPY/USD（Putは支払=マイナス、Callは受取=プラスの向きに注意）
      - Long Put P/L  = (-prem_put + max(Kp - S_T, 0)) * qty
      - Short Call P/L= ( prem_call - max(S_T - Kc, 0)) * qty
    """
    spot_pl       = (S_T - S0) * qty
    long_put_pl   = (-prem_put + np.maximum(Kp - S_T, 0.0)) * qty
    short_call_pl = ( prem_call - np.maximum(S_T - Kc, 0.0)) * qty
    combo_pl      = spot_pl + long_put_pl + short_call_pl
    return {
        "spot": spot_pl,
        "long_put": long_put_pl,
        "short_call": short_call_pl,
        "combo": combo_pl
    }


def build_grid_and_rows_zero_cost(
    S0, Kp, Kc, prem_put, prem_call, qty,
    smin, smax, points, step: float = 0.25
):
    """
    ゼロコスト用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    points は互換のため受け取るが、刻み幅優先のため実際の点数は step で決まる。
    """
    if smin > smax:
        smin, smax = smax, smin

    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_zero_cost(S_T, S0, Kp, Kc, prem_put, prem_call, qty)

    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "long_put": float(pl["long_put"][i]),
        "short_call": float(pl["short_call"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]

    return S_T, pl, rows


# 描画（ゼロコスト：Spot/Put/Call/Comboの4本）
def draw_chart_zero_cost(S_T, pl, S0, Kp, Kc):
    """
    Zero-Cost Collar の損益グラフ。Y軸はM表記。
    4本表示：Spot（黒）/ Long Put（青）/ Short Call（赤）/ Combo（緑）
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["spot"],        label="Spot USD P/L (vs today)", color="black")
    ax.plot(S_T, pl["long_put"],    label="Long Put P/L",            color="blue")
    ax.plot(S_T, pl["short_call"],  label="Short Call P/L",          color="red")
    ax.plot(S_T, pl["combo"],       label="Combo (Spot+Put-Call)",   color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["spot"], pl["long_put"], pl["short_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    # 縦の参考線（S0/Kp/Kc）
    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1)
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kp, linestyle=":", linewidth=1)
    ax.text(Kp, y_top, f"Kp={Kp:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kc, linestyle=":", linewidth=1)
    ax.text(Kc, y_top, f"Kc={Kc:.1f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Zero-Cost Collar: Spot + Long Put + Short Call (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


def draw_compare_zero_cost(S_T, combo_pl, finance_cost):
    """
    合成（Zero-Cost Combo）と 借入利息の比較（Y軸M表記）。
    借入利息ラインはレンジ内にあるときだけ描画。レンジ外なら注記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Zero-Cost Combo P/L")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)

    ymin, ymax = ax.get_ylim()
    borrow_level = -finance_cost
    if ymin <= borrow_level <= ymax:
        ax.axhline(borrow_level, linestyle="--", linewidth=1.5, label="Borrow Cost (flat)")
    else:
        pos = "below" if borrow_level < ymin else "above"
        ax.annotate(f"Borrow cost ≈ {borrow_level/1e6:.1f}M ({pos})",
                    xy=(S_T[len(S_T)//2], ymin if pos=='below' else ymax),
                    xytext=(6, -14 if pos=='below' else 6),
                    textcoords="offset points",
                    va="top" if pos=='below' else "bottom",
                    ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Compare: Zero-Cost Combo vs Borrow")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# 画面ルート（ゼロコスト）
@app.route("/fx/zero-cost", methods=["GET", "POST"])
def fx_zero_cost():
    """
    ゼロコスト・コリドー（Long Put + Short Call）の損益を表示。
    Premium は GK式で算出（JPY/USD）。理想は prem_call ≈ prem_put（純額≈0）。
    既存Put売り版と同じ入出力・表示体系。
    """
    defaults = dict(
        S0=150.0, Kp=148.0, Kc=152.0,
        vol=10.0,                 # 年率％
        r_dom=1.6,                # JPY金利（年率％）
        r_for=4.2,                # USD金利（年率％）
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0,               # 満期（月）
        borrow_rate=4.2           # 借入年率（％）
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0     = fget("S0", float, defaults["S0"])
        Kp     = fget("Kp", float, defaults["Kp"])
        Kc     = fget("Kc", float, defaults["Kc"])
        vol    = fget("vol", float, defaults["vol"])
        r_dom  = fget("r_dom", float, defaults["r_dom"])
        r_for  = fget("r_for", float, defaults["r_for"])
        qty    = fget("qty", float, defaults["qty"])
        smin   = fget("smin", float, defaults["smin"])
        smax   = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
        borrow_rate = fget("borrow_rate", float, defaults["borrow_rate"])
    else:
        S0 = defaults["S0"]; Kp = defaults["Kp"]; Kc = defaults["Kc"]
        vol = defaults["vol"]; r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]; borrow_rate = defaults["borrow_rate"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_put  = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)   # 支払（Long Put）
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call）

    # ネット・プレミアム（JPY/USD）…理想は ≈ 0
    premium_net = prem_call - prem_put

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_zero_cost(S0, Kp, Kc, prem_put, prem_call, qty, smin, smax, points)

    # レンジ内の最小損益（参考表示）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor     = float(pl["combo"][idx_min])    # 最小損益（JPY）
    range_floor_st  = float(S_T[idx_min])            # その時のレート S_T

    # プレミアム金額（JPY）
    prem_put_jpy   = prem_put * qty       # 支払（Long Put）
    prem_call_jpy  = prem_call * qty      # 受取（Short Call）
    premium_net_jpy = prem_call_jpy - prem_put_jpy   # ≈ 0 が理想

    # 借入利息（表示用）
    notional_jpy = S0 * qty
    finance_jpy = notional_jpy * (borrow_rate / 100.0) * (months / 12.0)

    # グラフ①（4本）
    fig = draw_chart_zero_cost(S_T, pl, S0, Kp, Kc)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②（Combo vs Borrow）
    fig_cmp = draw_compare_zero_cost(S_T, pl["combo"], finance_jpy)
    buf2 = io.BytesIO(); fig_cmp.savefig(buf2, format="png"); plt.close(fig_cmp); buf2.seek(0)
    png_b64_zero_cost_compare = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_zero_cost.html",
        png_b64=png_b64,
        png_b64_zero_cost_compare=png_b64_zero_cost_compare,
        # 入力
        S0=S0, Kp=Kp, Kc=Kc, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力（算出値）
        prem_put=prem_put, prem_call=prem_call, premium_net=premium_net,   # JPY/USD
        prem_put_jpy=prem_put_jpy, prem_call_jpy=prem_call_jpy, premium_net_jpy=premium_net_jpy,
        finance_cost=finance_jpy,
        range_floor=range_floor, range_floor_st=range_floor_st,
        rows=rows
    )


# CSV（ゼロコスト）
@app.route("/fx/download_csv_zero_cost", methods=["POST"])
def fx_download_csv_zero_cost():
    """
    Zero-Cost（Collar）のグリッドCSVダウンロード。
    prem_put / prem_call は JPY/USD（画面側で算出済みの値がPOSTされる想定）。
    出力列: S_T(USD/JPY), Spot_PnL(JPY), LongPut_PnL(JPY), ShortCall_PnL(JPY), Combo_PnL(JPY)
    レート刻みは 0.25 固定。
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    S0        = fget("S0", float, 150.0)
    Kp        = fget("Kp", float, 148.0)
    Kc        = fget("Kc", float, 152.0)
    prem_put  = fget("prem_put", float, 0.80)   # JPY/USD
    prem_call = fget("prem_call", float, 0.80)  # JPY/USD
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)      # 互換のため受け取るが、実際は step 優先
    step      = 0.25

    # 0.25刻みでグリッド生成
    S_T, pl, _ = build_grid_and_rows_zero_cost(
        S0, Kp, Kc, prem_put, prem_call, qty, smin, smax, points, step=step
    )

    import csv, io
    # UTF-8 with BOM でExcelでも文字化けしないように出力
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "Spot_PnL(JPY)", "LongPut_PnL(JPY)", "ShortCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['long_put'][i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="zero_cost_collar_pnl.csv",
    )

# =================================================＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝====
# ===== Zero-Cost（Long Call + Short Put：輸入向け）=====
# =====================================================

def payoff_components_zero_cost_longcall(S_T, S0, Kp, Kc, prem_call, prem_put, qty):
    """
    輸入実需を想定：基準は USD ショート。
    prem_call/prem_put は JPY/USD（Callは支払=マイナス、Putは受取=プラス）。
      Long Call P/L  = (-prem_call + max(S_T - Kc, 0)) * qty
      Short Put  P/L = ( prem_put   - max(Kp - S_T, 0)) * qty
    """
    # ★変更：USDショート基準（円安でコスト増 → P/Lはマイナス）にする
    spot_pl       = -(S_T - S0) * qty

    long_call_pl  = (-prem_call + np.maximum(S_T - Kc, 0.0)) * qty
    short_put_pl  = ( prem_put   - np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl      = spot_pl + long_call_pl + short_put_pl
    return {
        "spot": spot_pl,
        "long_call": long_call_pl,
        "short_put": short_put_pl,
        "combo": combo_pl
    }


def build_grid_and_rows_zero_cost_longcall(
    S0, Kp, Kc, prem_call, prem_put, qty, smin, smax, points, step: float = 0.25
):
    """ゼロコスト（Long Call + Short Put）用グリッドと行データ。レートは0.25刻み。"""
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_zero_cost_longcall(S_T, S0, Kp, Kc, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "long_call": float(pl["long_call"][i]),
        "short_put": float(pl["short_put"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_zero_cost_longcall(S_T, pl, S0, Kp, Kc):
    """
    損益グラフ（4本表示）：Short USD（黒）/ Long Call（青）/ Short Put（赤）/ Combo（緑）
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    # ★変更：凡例を Short USD に
    ax.plot(S_T, pl["spot"],       label="Short USD P/L (Importer baseline)", color="black")
    ax.plot(S_T, pl["long_call"],  label="Long Call P/L",                     color="blue")
    ax.plot(S_T, pl["short_put"],  label="Short Put P/L",                     color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Short USD + Call - Put)",    color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["spot"], pl["long_call"], pl["short_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kp, linestyle=":",  linewidth=1); ax.text(Kp, y_top, f"Kp={Kp:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kc, linestyle=":",  linewidth=1); ax.text(Kc, y_top, f"Kc={Kc:.1f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Zero-Cost (Importer): Short USD + Long Call + Short Put (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


def draw_compare_zero_cost_longcall(S_T, combo_pl, finance_cost):
    """
    合成（Zero-Cost Combo）と 借入利息の比較（Y軸M表記）。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Zero-Cost Combo P/L")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)

    ymin, ymax = ax.get_ylim()
    borrow_level = -finance_cost
    if ymin <= borrow_level <= ymax:
        ax.axhline(borrow_level, linestyle="--", linewidth=1.5, label="Borrow Cost (flat)")
    else:
        pos = "below" if borrow_level < ymin else "above"
        ax.annotate(f"Borrow cost ≈ {borrow_level/1e6:.1f}M ({pos})",
                    xy=(S_T[len(S_T)//2], ymin if pos=='below' else ymax),
                    xytext=(6, -14 if pos=='below' else 6),
                    textcoords="offset points",
                    va="top" if pos=='below' else "bottom",
                    ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Compare: Zero-Cost (Importer) Combo vs Borrow")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


@app.route("/fx/zero-cost-call", methods=["GET", "POST"])
def fx_zero_cost_long_call():
    """
    ゼロコスト（Long Call + Short Put）。輸入企業が円安リスクをヘッジする想定。
    Premium は GK式で算出。理想は prem_put ≈ prem_call（純額≈0）。
    """
    defaults = dict(
        S0=150.0, Kp=148.0, Kc=152.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0,
        borrow_rate=4.2
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0     = fget("S0", float, defaults["S0"])
        Kp     = fget("Kp", float, defaults["Kp"])   # Put strike（Short）
        Kc     = fget("Kc", float, defaults["Kc"])   # Call strike（Long）
        vol    = fget("vol", float, defaults["vol"])
        r_dom  = fget("r_dom", float, defaults["r_dom"])
        r_for  = fget("r_for", float, defaults["r_for"])
        qty    = fget("qty", float, defaults["qty"])
        smin   = fget("smin", float, defaults["smin"])
        smax   = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
        borrow_rate = fget("borrow_rate", float, defaults["borrow_rate"])
    else:
        S0 = defaults["S0"]; Kp = defaults["Kp"]; Kc = defaults["Kc"]
        vol = defaults["vol"]; r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]; borrow_rate = defaults["borrow_rate"]

    points = clamp_points(points)

    # GK式プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Call）
    prem_put  = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)   # 受取（Short Put）

    premium_net = prem_put - prem_call  # （受取−支払）≈ 0 が理想

    # グリッドと損益（0.25刻み）
    S_T, pl, rows = build_grid_and_rows_zero_cost_longcall(
        S0, Kp, Kc, prem_call, prem_put, qty, smin, smax, points, step=0.25
    )

    # レンジ内最小損益（参考）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min])
    range_floor_st = float(S_T[idx_min])

    # 金額換算
    prem_call_jpy   = prem_call * qty   # 支払
    prem_put_jpy    = prem_put  * qty   # 受取
    premium_net_jpy = prem_put_jpy - prem_call_jpy

    notional_jpy = S0 * qty
    finance_jpy  = notional_jpy * (borrow_rate / 100.0) * (months / 12.0)

    # グラフ①（4本）
    fig = draw_chart_zero_cost_longcall(S_T, pl, S0, Kp, Kc)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②（Combo vs Borrow）
    fig_cmp = draw_compare_zero_cost_longcall(S_T, pl["combo"], finance_jpy)
    buf2 = io.BytesIO(); fig_cmp.savefig(buf2, format="png"); plt.close(fig_cmp); buf2.seek(0)
    png_b64_zero_cost_compare = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_zero_cost_long_call.html",
        png_b64=png_b64,
        png_b64_zero_cost_compare=png_b64_zero_cost_compare,
        # 入力
        S0=S0, Kp=Kp, Kc=Kc, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_net=premium_net,  # JPY/USD
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_net_jpy=premium_net_jpy,
        finance_cost=finance_jpy,
        range_floor=range_floor, range_floor_st=range_floor_st,
        rows=rows
    )


@app.route("/fx/download_csv_zero_cost_longcall", methods=["POST"])
def fx_download_csv_zero_cost_longcall():
    """
    Zero-Cost（Long Call + Short Put）版のCSV。
    出力: S_T, ShortUSD_PnL, LongCall_PnL, ShortPut_PnL, Combo_PnL（JPY）
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    S0        = fget("S0", float, 150.0)
    Kp        = fget("Kp", float, 148.0)
    Kc        = fget("Kc", float, 152.0)
    prem_call = fget("prem_call", float, 0.80)  # JPY/USD（支払）
    prem_put  = fget("prem_put",  float, 0.80)  # JPY/USD（受取）
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_zero_cost_longcall(
        S0, Kp, Kc, prem_call, prem_put, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "ShortUSD_PnL(JPY)", "LongCall_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['long_call'][i]:.6f}",
            f"{pl['short_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="zero_cost_longcall_pnl.csv",
    )


# ==債権ヘッジ====================================================================================================================
# ===================== Straddle =====================
# ====================================================

# 損益ロジック（ストラドル：Long Call + Long Put）
def payoff_components_straddle(S_T, K, prem_call, prem_put, qty):
    """
    Long Call と Long Put（同一ストライクK）の損益（JPY）。
    prem_call / prem_put は JPY/USD（支払のためどちらもマイナスで計上）
      - Long Call P/L = (-prem_call + max(S_T - K, 0)) * qty
      - Long Put  P/L = (-prem_put  + max(K - S_T, 0)) * qty
    """
    long_call_pl = (-prem_call + np.maximum(S_T - K, 0.0)) * qty
    long_put_pl  = (-prem_put  + np.maximum(K - S_T, 0.0)) * qty
    combo_pl     = long_call_pl + long_put_pl
    return {
        "long_call": long_call_pl,
        "long_put":  long_put_pl,
        "combo":     combo_pl
    }


def build_grid_and_rows_straddle(K, prem_call, prem_put, qty,
                                 smin, smax, points, step: float = 0.25):
    """
    ストラドル用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    points は互換のため受け取るが、刻み幅優先のため実際の点数は step で決まる。
    """
    if smin > smax:
        smin, smax = smax, smin

    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_straddle(S_T, K, prem_call, prem_put, qty)

    rows = [{
        "st": float(S_T[i]),
        "long_call": float(pl["long_call"][i]),
        "long_put":  float(pl["long_put"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]

    return S_T, pl, rows


# 描画（ストラドル：Call/Put/Comboの3本 + 目安線）
def draw_chart_straddle(S_T, pl, S0, K, be_low, be_high):
    """
    Straddle の損益グラフ。Y軸はM表記。
    3本表示：Long Call（青）/ Long Put（赤）/ Combo（緑）
    縦線：S0, K, Break-even(下/上)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_call"], label="Long Call P/L", color="blue")
    ax.plot(S_T, pl["long_put"],  label="Long Put P/L",  color="red")
    ax.plot(S_T, pl["combo"],     label="Combo (Call+Put)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_call"], pl["long_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    # 参考縦線：現値S0、行使K、損益分岐点
    ax.axvline(S0, linestyle="--", linewidth=1)
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_low, linestyle="--", linewidth=1)
    ax.text(be_low, y_top, f"BE−={be_low:.2f}", va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1)
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Straddle: Long Call + Long Put (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# ブレークイーブン注力版（同じ合成線だがBEを強調）
def draw_straddle_breakeven(S_T, combo_pl, be_low, be_high):
    """
    合成（Straddle）と損益分岐点（2本）を表示。Y軸M表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Straddle Combo P/L")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low, linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low, y_top, f"BE−={be_low:.2f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Straddle: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# 画面ルート（ストラドル）
@app.route("/fx/straddle", methods=["GET", "POST"])
def fx_straddle():
    """
    ストラドル（Long Call + Long Put, 同一K）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    既存のゼロコストと同じ入出力/表示体系（カード/グラフ/テーブル/CSV）に準拠。
    """
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0,                 # 年率％
        r_dom=1.6,                # JPY金利（年率％）
        r_for=4.2,                # USD金利（年率％）
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0                # 満期（月）
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0     = fget("S0", float, defaults["S0"])
        K      = fget("K", float, defaults["K"])
        vol    = fget("vol", float, defaults["vol"])
        r_dom  = fget("r_dom", float, defaults["r_dom"])
        r_for  = fget("r_for", float, defaults["r_for"])
        qty    = fget("qty", float, defaults["qty"])
        smin   = fget("smin", float, defaults["smin"])
        smax   = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0 = defaults["S0"]; K = defaults["K"]
        vol = defaults["vol"]; r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Call）
    prem_put  = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)   # 支払（Long Put）

    # 合計プレミアム（JPY/USD）
    premium_sum = prem_call + prem_put

    # ブレークイーブン（USD/JPY）
    be_low  = K - premium_sum
    be_high = K + premium_sum

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_straddle(K, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内の最小損益（参考表示）……理論上は -合計プレミアム（K近傍）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min])   # 最小損益（JPY）
    range_floor_st = float(S_T[idx_min])           # その時のレート S_T

    # プレミアム金額（JPY）
    prem_call_jpy = prem_call * qty
    prem_put_jpy  = prem_put * qty
    premium_sum_jpy = prem_call_jpy + prem_put_jpy

    # グラフ①（3本：Call/Put/Combo + S0/K/BE）
    fig = draw_chart_straddle(S_T, pl, S0, K, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②（Break-even強調）
    fig_be = draw_straddle_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_straddle_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_straddle.html",
        png_b64=png_b64,
        png_b64_straddle_be=png_b64_straddle_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力（算出値）
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,   # JPY/USD
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        rows=rows
    )


# CSV（ストラドル）
@app.route("/fx/download_csv_straddle", methods=["POST"])
def fx_download_csv_straddle():
    """
    Straddle（Long Call + Long Put）のグリッドCSVダウンロード。
    prem_call / prem_put は JPY/USD（画面側で算出済みの値がPOSTされる想定）。
    出力列: S_T(USD/JPY), LongCall_PnL(JPY), LongPut_PnL(JPY), Combo_PnL(JPY)
    レート刻みは 0.25 固定。
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    K         = fget("K", float, 150.0)
    prem_call = fget("prem_call", float, 0.80)   # JPY/USD
    prem_put  = fget("prem_put",  float, 0.80)   # JPY/USD
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)       # 互換のため受け取るが、実際は step 優先
    step      = 0.25

    # 0.25刻みでグリッド生成
    S_T, pl, _ = build_grid_and_rows_straddle(
        K, prem_call, prem_put, qty, smin, smax, points, step=step
    )

    import csv, io
    # UTF-8 with BOM でExcelでも文字化けしないように出力
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "LongPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_call'][i]:.6f}",
            f"{pl['long_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="straddle_pnl.csv",
    )

# ==債権ヘッジ===================================================
# ================= Short Straddle =====================
# ====================================================

# ================= Short Straddle =====================
# =====================================================

# 損益ロジック（ショート・ストラドル：Short Call + Short Put）
def payoff_components_short_straddle(S_T, K, prem_call, prem_put, qty):
    """
    Short Call と Short Put（同一ストライクK）の損益（JPY）。
    prem_call / prem_put は JPY/USD（受取のため＋で計上）
      - Short Call P/L = (prem_call - max(S_T - K, 0)) * qty
      - Short Put  P/L = (prem_put  - max(K - S_T, 0)) * qty
    """
    short_call_pl = (prem_call - np.maximum(S_T - K, 0.0)) * qty
    short_put_pl  = (prem_put  - np.maximum(K - S_T, 0.0)) * qty
    combo_pl      = short_call_pl + short_put_pl
    return {
        "short_call": short_call_pl,
        "short_put":  short_put_pl,
        "combo":      combo_pl
    }


def build_grid_and_rows_short_straddle(K, prem_call, prem_put, qty,
                                       smin, smax, points, step: float = 0.25):
    """
    ショート・ストラドル用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    points は互換のため受け取るが、刻み幅優先。
    """
    if smin > smax:
        smin, smax = smax, smin

    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_short_straddle(S_T, K, prem_call, prem_put, qty)

    rows = [{
        "st": float(S_T[i]),
        "short_call": float(pl["short_call"][i]),
        "short_put":  float(pl["short_put"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]

    return S_T, pl, rows


# 描画（ショートストラドル：ShortCall/ShortPut/Comboの3本 + 目安線）
def draw_chart_short_straddle(S_T, pl, S0, K, be_low, be_high):
    """
    Short Straddle の損益グラフ。Y軸はM表記。
    3本表示：Short Call（青）/ Short Put（赤）/ Combo（緑）
    縦線：S0, K, Break-even(下/上)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["short_call"], label="Short Call P/L", color="blue")
    ax.plot(S_T, pl["short_put"],  label="Short Put P/L",  color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Short Straddle)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short_call"], pl["short_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1)
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_low, linestyle="--", linewidth=1)
    ax.text(be_low, y_top, f"BE−={be_low:.2f}", va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1)
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Straddle: Short Call + Short Put (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# ブレークイーブン注力版（ショート・ストラドル）
def draw_short_straddle_breakeven(S_T, combo_pl, be_low, be_high):
    """
    合成（Short Straddle）と損益分岐点（2本）を表示。Y軸M表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Short Straddle Combo P/L")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low, linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low, y_top, f"BE−={be_low:.2f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Straddle: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# 画面ルート（ショートストラドル）
@app.route("/fx/straddle-short", methods=["GET", "POST"])
def fx_straddle_short():
    """
    ショート・ストラドル（Short Call + Short Put, 同一K）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0     = fget("S0", float, defaults["S0"])
        K      = fget("K", float, defaults["K"])
        vol    = fget("vol", float, defaults["vol"])
        r_dom  = fget("r_dom", float, defaults["r_dom"])
        r_for  = fget("r_for", float, defaults["r_for"])
        qty    = fget("qty", float, defaults["qty"])
        smin   = fget("smin", float, defaults["smin"])
        smax   = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0 = defaults["S0"]; K = defaults["K"]
        vol = defaults["vol"]; r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call）
    prem_put  = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)   # 受取（Short Put）

    # 合計プレミアム（JPY/USD）
    premium_sum = prem_call + prem_put

    # ブレークイーブン
    be_low  = K - premium_sum
    be_high = K + premium_sum

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_short_straddle(K, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内最小/最大損益
    idx_min = int(np.argmin(pl["combo"]))
    idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min])
    range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max])
    range_cap_st   = float(S_T[idx_max])

    # プレミアム金額（JPY）
    prem_call_jpy = prem_call * qty
    prem_put_jpy  = prem_put * qty
    premium_sum_jpy = prem_call_jpy + prem_put_jpy

    # グラフ①（損益）
    fig = draw_chart_short_straddle(S_T, pl, S0, K, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②（損益分岐点強調）← 追加
    fig_be = draw_short_straddle_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_short_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_straddle_short.html",
        png_b64=png_b64,
        png_b64_short_be=png_b64_short_be,  # ← 追加
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（ショートストラドル）
@app.route("/fx/download_csv_straddle_short", methods=["POST"])
def fx_download_csv_straddle_short():
    """
    Short Straddle（Short Call + Short Put）のCSV出力。
    出力列: S_T, ShortCall_PnL, ShortPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    K         = fget("K", float, 150.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_short_straddle(
        K, prem_call, prem_put, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "ShortCall_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['short_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="short_straddle_pnl.csv",
    )


# ==債権ヘッジ===================================================
# ===================== Strangle ======================
# ====================================================

# 損益ロジック（ロング・ストラングル：Long Call(Kc) + Long Put(Kp)）
def payoff_components_strangle(S_T, Kc, Kp, prem_call, prem_put, qty):
    """
    Long Call と Long Put（異なる行使価格 Kc, Kp）の損益（JPY）。
    prem_call / prem_put は JPY/USD（支払のためどちらもマイナスで計上）
      - Long Call P/L = (-prem_call + max(S_T - Kc, 0)) * qty
      - Long Put  P/L = (-prem_put  + max(Kp - S_T, 0)) * qty
    """
    long_call_pl = (-prem_call + np.maximum(S_T - Kc, 0.0)) * qty
    long_put_pl  = (-prem_put  + np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl     = long_call_pl + long_put_pl
    return {"long_call": long_call_pl, "long_put": long_put_pl, "combo": combo_pl}


def build_grid_and_rows_strangle(Kc, Kp, prem_call, prem_put, qty,
                                 smin, smax, points, step: float = 0.25):
    """
    ストラングル用のレートグリッドを 0.25 刻みで生成。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_strangle(S_T, Kc, Kp, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_call": float(pl["long_call"][i]),
        "long_put":  float(pl["long_put"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_strangle(S_T, pl, S0, Kc, Kp, be_low, be_high):
    """
    Strangle の損益グラフ。Y軸M表記。
    3本：Long Call（青）/ Long Put（赤）/ Combo（緑）
    縦線：S0, Kp, Kc, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_call"], label="Long Call P/L", color="blue")
    ax.plot(S_T, pl["long_put"],  label="Long Put P/L",  color="red")
    ax.plot(S_T, pl["combo"],     label="Combo (Strangle)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["long_call"], pl["long_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kp, linestyle=":",  linewidth=1); ax.text(Kp, y_top, f"Kp={Kp:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kc, linestyle=":",  linewidth=1); ax.text(Kc, y_top, f"Kc={Kc:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(be_low,  linestyle="--", linewidth=1); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Strangle: Long Call(Kc) + Long Put(Kp) (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_strangle_breakeven(S_T, combo_pl, be_low, be_high):
    """
    合成（Strangle）と損益分岐点（2本）を表示。Y軸M表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Strangle Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Strangle: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（ロング・ストラングル）
@app.route("/fx/strangle", methods=["GET", "POST"])
def fx_strangle():
    """
    ロング・ストラングル（Long Call(Kc) + Long Put(Kp)）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, Kc=152.0, Kp=148.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        Kc = fget("Kc", float, defaults["Kc"])
        Kp = fget("Kp", float, defaults["Kp"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; Kc=defaults["Kc"]; Kp=defaults["Kp"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Call）
    prem_put  = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)   # 支払（Long Put）

    # 合計プレミアムとBE
    premium_sum = prem_call + prem_put
    be_low  = Kp - premium_sum
    be_high = Kc + premium_sum

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_strangle(Kc, Kp, prem_call, prem_put, qty, smin, smax, points)

    # 参考：レンジ内最小損益（概ね -合計プレミアム）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min])
    range_floor_st = float(S_T[idx_min])

    # プレミアム金額（JPY）
    prem_call_jpy = prem_call * qty
    prem_put_jpy  = prem_put  * qty
    premium_sum_jpy = prem_call_jpy + prem_put_jpy

    # グラフ
    fig = draw_chart_strangle(S_T, pl, S0, Kc, Kp, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig_be = draw_strangle_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_strangle_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_strangle.html",
        png_b64=png_b64,
        png_b64_strangle_be=png_b64_strangle_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        rows=rows
    )


# CSV（ロング・ストラングル）
@app.route("/fx/download_csv_strangle", methods=["POST"])
def fx_download_csv_strangle():
    """
    Long Strangle のグリッドCSVダウンロード。
    出力列: S_T, LongCall_PnL, LongPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    Kc        = fget("Kc", float, 152.0)
    Kp        = fget("Kp", float, 148.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_strangle(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "LongPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([f"{S_T[i]:.6f}", f"{pl['long_call'][i]:.6f}", f"{pl['long_put'][i]:.6f}", f"{pl['combo'][i]:.6f}"])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="strangle_pnl.csv")


# ================= Short Strangle ====================

# ================= Short Strangle ====================

def payoff_components_short_strangle(S_T, Kc, Kp, prem_call, prem_put, qty):
    """
    Short Call と Short Put（異なる行使価格 Kc, Kp）の損益（JPY）。
      - Short Call P/L = (prem_call - max(S_T - Kc, 0)) * qty
      - Short Put  P/L = (prem_put  - max(Kp - S_T, 0)) * qty
    prem_call / prem_put は JPY/USD（受取はプラスで計上）
    """
    short_call_pl = (prem_call - np.maximum(S_T - Kc, 0.0)) * qty
    short_put_pl  = (prem_put  - np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl      = short_call_pl + short_put_pl
    return {"short_call": short_call_pl, "short_put": short_put_pl, "combo": combo_pl}


def build_grid_and_rows_short_strangle(Kc, Kp, prem_call, prem_put, qty,
                                       smin, smax, points, step: float = 0.25):
    """
    ショート・ストラングル用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    points は互換のため受け取るが、実際の点数は step により決まる。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_short_strangle(S_T, Kc, Kp, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_call": float(pl["short_call"][i]),
        "short_put":  float(pl["short_put"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_short_strangle(S_T, pl, S0, Kc, Kp, be_low, be_high):
    """
    Short Strangle の損益グラフ。Y軸M表記。
    3本：Short Call（青）/ Short Put（赤）/ Combo（緑）
    縦線：S0, Kp, Kc, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["short_call"], label="Short Call P/L", color="blue")
    ax.plot(S_T, pl["short_put"],  label="Short Put P/L",  color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Short Strangle)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short_call"], pl["short_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1)
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(Kp, linestyle=":",  linewidth=1)
    ax.text(Kp, y_top, f"Kp={Kp:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(Kc, linestyle=":",  linewidth=1)
    ax.text(Kc, y_top, f"Kc={Kc:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_low,  linestyle="--", linewidth=1)
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)

    ax.axvline(be_high, linestyle="--", linewidth=1)
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Strangle: Short Call(Kc) + Short Put(Kp) (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


def draw_short_strangle_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Short Strangle の合成損益と損益分岐点（2本）を表示。Y軸はM表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Short Strangle Combo P/L")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)

    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Strangle: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# 画面ルート（ショート・ストラングル）
@app.route("/fx/strangle-short", methods=["GET", "POST"])
def fx_strangle_short():
    defaults = dict(
        S0=150.0, Kc=152.0, Kp=148.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        Kc = fget("Kc", float, defaults["Kc"])
        Kp = fget("Kp", float, defaults["Kp"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0 = defaults["S0"]; Kc = defaults["Kc"]; Kp = defaults["Kp"]
        vol = defaults["vol"]; r_dom = defaults["r_dom"]; r_for = defaults["r_for"]
        qty = defaults["qty"]; smin = defaults["smin"]; smax = defaults["smax"]
        points = defaults["points"]; months = defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call）
    prem_put  = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)   # 受取（Short Put）

    # 受取プレミアム合計と損益分岐点
    premium_sum = prem_call + prem_put
    be_low  = Kp - premium_sum
    be_high = Kc + premium_sum

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_short_strangle(Kc, Kp, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内 最大/最小（参考）
    idx_min = int(np.argmin(pl["combo"]))
    idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min]); range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max]); range_cap_st   = float(S_T[idx_max])

    # プレミアム金額（JPY）
    prem_call_jpy   = prem_call * qty
    prem_put_jpy    = prem_put  * qty
    premium_sum_jpy = prem_call_jpy + prem_put_jpy

    # グラフ①：全体P/L
    fig = draw_chart_short_strangle(S_T, pl, S0, Kc, Kp, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：BE強調
    fig_be = draw_short_strangle_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_strangle_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_strangle_short.html",
        png_b64=png_b64,
        png_b64_strangle_be=png_b64_strangle_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（ショート・ストラングル）
@app.route("/fx/download_csv_strangle_short", methods=["POST"])
def fx_download_csv_strangle_short():
    """
    Short Strangle のCSV出力。
    出力列: S_T, ShortCall_PnL, ShortPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default

    Kc        = fget("Kc", float, 152.0)
    Kp        = fget("Kp", float, 148.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_short_strangle(
        Kc, Kp, prem_call, prem_put, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "ShortCall_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['short_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    data.seek(0)
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="short_strangle_pnl.csv",
    )

# ===================== Strip (2x Put + 1x Call) ======================

def payoff_components_strip(S_T, K, prem_call, prem_put, qty):
    """
    Long Strip: Long Call(1) + Long Put(2) の損益（JPY）。
    prem_call / prem_put は JPY/USD（支払＝マイナス計上）
      - Long Call P/L  = (-prem_call + max(S_T - K, 0)) * qty
      - 2x Long Put    = (-2*prem_put + 2*max(K - S_T, 0)) * qty
    """
    long_call_pl = (-prem_call + np.maximum(S_T - K, 0.0)) * qty
    long_put2_pl = (-2.0 * prem_put + 2.0 * np.maximum(K - S_T, 0.0)) * qty
    combo_pl     = long_call_pl + long_put2_pl
    return {"long_call": long_call_pl, "long_put2": long_put2_pl, "combo": combo_pl}


def build_grid_and_rows_strip(K, prem_call, prem_put, qty, smin, smax, points, step: float = 0.25):
    """
    ストリップ用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_strip(S_T, K, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_call": float(pl["long_call"][i]),
        "long_put2": float(pl["long_put2"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_strip(S_T, pl, S0, K, be_low, be_high):
    """
    Strip の損益グラフ。Y軸M表記。
    3本：Long Call（青）/ 2×Long Put（赤）/ Combo（緑）
    縦線：S0, K, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_call"], label="Long Call P/L", color="blue")
    ax.plot(S_T, pl["long_put2"], label="2× Long Put P/L", color="red")
    ax.plot(S_T, pl["combo"],     label="Combo (Strip)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_call"], pl["long_put2"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K,  linestyle=":",  linewidth=1); ax.text(K,  y_top, f"K={K:.1f}",   va="top", ha="left", fontsize=9)
    ax.axvline(be_low,  linestyle="--", linewidth=1); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Strip: Long Put×2 + Long Call (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_strip_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Strip 合成損益と損益分岐点（2本）を表示。Y軸M表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Strip Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Strip: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（ロング・ストリップ）
@app.route("/fx/strip", methods=["GET", "POST"])
def fx_strip():
    """
    ロング・ストリップ（Long Put×2 + Long Call, 同一K）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K  = fget("K",  float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K=defaults["K"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Call）
    prem_put  = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)   # 支払（Long Put）

    # 合計プレミアム（JPY/USD）と損益分岐点
    premium_sum = prem_call + 2.0 * prem_put
    # S <= K 側： (-prem_call -2prem_put) + 2(K - S) = 0 → S = K - (prem_call + 2*prem_put)/2
    be_low  = K - premium_sum / 2.0
    # S >= K 側： (-prem_call -2prem_put) + (S - K) = 0 → S = K + (prem_call + 2*prem_put)
    be_high = K + premium_sum

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_strip(K, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内 最小損益（参考）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min])
    range_floor_st = float(S_T[idx_min])

    # プレミアム金額（JPY）
    prem_call_jpy = prem_call * qty
    prem_put_jpy  = prem_put  * qty
    premium_sum_jpy = prem_call_jpy + 2.0 * prem_put_jpy

    # グラフ①：全体P/L
    fig = draw_chart_strip(S_T, pl, S0, K, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：BE強調
    fig_be = draw_strip_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_strip_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_strip.html",
        png_b64=png_b64,
        png_b64_strip_be=png_b64_strip_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        rows=rows
    )


# CSV（ロング・ストリップ）
@app.route("/fx/download_csv_strip", methods=["POST"])
def fx_download_csv_strip():
    """
    Strip のグリッドCSVダウンロード。
    出力列: S_T, LongCall_PnL, LongPut2x_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K         = fget("K", float, 150.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_strip(K, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "LongPut2x_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_call'][i]:.6f}",
            f"{pl['long_put2'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="strip_pnl.csv")


# ===================== Short Strip (2x Short Put + Short Call) ======================

def payoff_components_short_strip(S_T, K, prem_call, prem_put, qty):
    """
    Short Strip: Short Call(1) + Short Put(2) の損益（JPY）。
    prem_call / prem_put は JPY/USD（受取＝プラス計上）
      - Short Call P/L = ( prem_call - max(S_T - K, 0)) * qty
      - 2x Short Put   = ( 2*prem_put - 2*max(K - S_T, 0)) * qty
    """
    short_call_pl = ( prem_call - np.maximum(S_T - K, 0.0)) * qty
    short_put2_pl = ( 2.0 * prem_put - 2.0 * np.maximum(K - S_T, 0.0)) * qty
    combo_pl      = short_call_pl + short_put2_pl
    return {"short_call": short_call_pl, "short_put2": short_put2_pl, "combo": combo_pl}


def build_grid_and_rows_short_strip(K, prem_call, prem_put, qty, smin, smax, points, step: float = 0.25):
    """
    ショート・ストリップ用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_short_strip(S_T, K, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_call": float(pl["short_call"][i]),
        "short_put2": float(pl["short_put2"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_short_strip(S_T, pl, S0, K, be_low, be_high):
    """
    Short Strip の損益グラフ。Y軸M表記。
    3本：Short Call（青）/ 2×Short Put（赤）/ Combo（緑）
    縦線：S0, K, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["short_call"], label="Short Call P/L", color="blue")
    ax.plot(S_T, pl["short_put2"], label="2× Short Put P/L", color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Short Strip)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short_call"], pl["short_put2"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K,  linestyle=":",  linewidth=1); ax.text(K,  y_top, f"K={K:.1f}",   va="top", ha="left", fontsize=9)
    ax.axvline(be_low,  linestyle="--", linewidth=1); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Strip: Short Put×2 + Short Call (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_short_strip_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Short Strip 合成損益と損益分岐点（2本）を表示。Y軸M表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Short Strip Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Strip: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（ショート・ストリップ）
@app.route("/fx/strip-short", methods=["GET", "POST"])
def fx_strip_short():
    """
    ショート・ストリップ（Short Put×2 + Short Call, 同一K）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K  = fget("K",  float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K=defaults["K"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call）
    prem_put  = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)   # 受取（Short Put）

    # 受取プレミアム合計（JPY/USD）と損益分岐点
    premium_sum = prem_call + 2.0 * prem_put
    be_low  = K - premium_sum / 2.0
    be_high = K + premium_sum

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_short_strip(K, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内 最大/最小（参考）
    idx_min = int(np.argmin(pl["combo"]))
    idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min]); range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max]); range_cap_st   = float(S_T[idx_max])

    # プレミアム金額（JPY）
    prem_call_jpy   = prem_call * qty
    prem_put_jpy    = prem_put  * qty
    premium_sum_jpy = prem_call_jpy + 2.0 * prem_put_jpy

    # グラフ①：全体P/L
    fig = draw_chart_short_strip(S_T, pl, S0, K, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：BE強調
    fig_be = draw_short_strip_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_strip_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_strip_short.html",
        png_b64=png_b64,
        png_b64_strip_be=png_b64_strip_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（ショート・ストリップ）
@app.route("/fx/download_csv_strip_short", methods=["POST"])
def fx_download_csv_strip_short():
    """
    Short Strip のグリッドCSVダウンロード。
    出力列: S_T, ShortCall_PnL, ShortPut2x_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K         = fget("K", float, 150.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_short_strip(K, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "ShortCall_PnL(JPY)", "ShortPut2x_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['short_put2'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="short_strip_pnl.csv")


# ===================== Strap (2x Call + 1x Put) ======================

def payoff_components_strap(S_T, K, prem_call, prem_put, qty):
    """
    Long Strap: Long Call(2) + Long Put(1) の損益（JPY）。
    prem_call / prem_put は JPY/USD（支払＝マイナス計上）
      - 2x Long Call = (-2*prem_call + 2*max(S_T - K, 0)) * qty
      - Long Put     = (-prem_put    +   max(K - S_T, 0)) * qty
    """
    long_call2_pl = (-2.0 * prem_call + 2.0 * np.maximum(S_T - K, 0.0)) * qty
    long_put_pl   = ( -prem_put      +       np.maximum(K - S_T, 0.0)) * qty
    combo_pl      = long_call2_pl + long_put_pl
    return {"long_call2": long_call2_pl, "long_put": long_put_pl, "combo": combo_pl}


def build_grid_and_rows_strap(K, prem_call, prem_put, qty, smin, smax, points, step: float = 0.25):
    """
    ストラップ用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_strap(S_T, K, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_call2": float(pl["long_call2"][i]),
        "long_put":   float(pl["long_put"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_strap(S_T, pl, S0, K, be_low, be_high):
    """
    Strap の損益グラフ。Y軸M表記。
    3本：2×Long Call（青）/ Long Put（赤）/ Combo（緑）
    縦線：S0, K, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_call2"], label="2× Long Call P/L", color="blue")
    ax.plot(S_T, pl["long_put"],   label="Long Put P/L",     color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Strap)",    color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_call2"], pl["long_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K,  linestyle=":",  linewidth=1); ax.text(K,  y_top, f"K={K:.1f}",   va="top", ha="left", fontsize=9)
    ax.axvline(be_low,  linestyle="--", linewidth=1); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Strap: Long Call×2 + Long Put (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_strap_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Strap 合成損益と損益分岐点（2本）を表示。Y軸M表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Strap Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Strap: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（ロング・ストラップ）
@app.route("/fx/strap", methods=["GET", "POST"])
def fx_strap():
    """
    ロング・ストラップ（Long Call×2 + Long Put, 同一K）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K  = fget("K",  float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K=defaults["K"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Call）
    prem_put  = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)   # 支払（Long Put）

    # 合計プレミアム（JPY/USD）と損益分岐点
    premium_sum = 2.0 * prem_call + prem_put
    # S <= K 側： (-2C - P) + (K - S) = 0 → S = K - (2C + P)
    be_low  = K - premium_sum
    # S >= K 側： (-2C - P) + 2(S - K) = 0 → S = K + (2C + P)/2
    be_high = K + premium_sum / 2.0

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_strap(K, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内 最小損益（参考）
    idx_min = int(np.argmin(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min])
    range_floor_st = float(S_T[idx_min])

    # プレミアム金額（JPY）
    prem_call_jpy = prem_call * qty
    prem_put_jpy  = prem_put  * qty
    premium_sum_jpy = 2.0 * prem_call_jpy + prem_put_jpy

    # グラフ①：全体P/L
    fig = draw_chart_strap(S_T, pl, S0, K, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：BE強調
    fig_be = draw_strap_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_strap_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_strap.html",
        png_b64=png_b64,
        png_b64_strap_be=png_b64_strap_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        rows=rows
    )


# CSV（ロング・ストラップ）
@app.route("/fx/download_csv_strap", methods=["POST"])
def fx_download_csv_strap():
    """
    Strap のグリッドCSVダウンロード。
    出力列: S_T, LongCall2x_PnL, LongPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K         = fget("K", float, 150.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_strap(K, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "LongCall2x_PnL(JPY)", "LongPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_call2'][i]:.6f}",
            f"{pl['long_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="strap_pnl.csv")

# ===================== Short Strap (2x Short Call + 1x Short Put) ======================

def payoff_components_short_strap(S_T, K, prem_call, prem_put, qty):
    """
    Short Strap: Short Call(2) + Short Put(1) の損益（JPY）。
    prem_call / prem_put は JPY/USD（受取＝プラス計上）
      - 2x Short Call = ( 2*prem_call - 2*max(S_T - K, 0)) * qty
      - Short Put     = (   prem_put  -   max(K - S_T, 0)) * qty
    """
    short_call2_pl = ( 2.0 * prem_call - 2.0 * np.maximum(S_T - K, 0.0)) * qty
    short_put_pl   = (       prem_put  -       np.maximum(K - S_T, 0.0)) * qty
    combo_pl       = short_call2_pl + short_put_pl
    return {"short_call2": short_call2_pl, "short_put": short_put_pl, "combo": combo_pl}


def build_grid_and_rows_short_strap(K, prem_call, prem_put, qty, smin, smax, points, step: float = 0.25):
    """
    ショート・ストラップ用のレートグリッドを 0.25 刻みで生成し、行データを返す。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_short_strap(S_T, K, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_call2": float(pl["short_call2"][i]),
        "short_put":   float(pl["short_put"][i]),
        "combo":       float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_short_strap(S_T, pl, S0, K, be_low, be_high):
    """
    Short Strap の損益グラフ。Y軸M表記。
    3本：2×Short Call（青）/ Short Put（赤）/ Combo（緑）
    縦線：S0, K, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["short_call2"], label="2× Short Call P/L", color="blue")
    ax.plot(S_T, pl["short_put"],   label="Short Put P/L",     color="red")
    ax.plot(S_T, pl["combo"],       label="Combo (Short Strap)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short_call2"], pl["short_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K,  linestyle=":",  linewidth=1); ax.text(K,  y_top, f"K={K:.1f}",   va="top", ha="left", fontsize=9)
    ax.axvline(be_low,  linestyle="--", linewidth=1); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Strap: Short Call×2 + Short Put (P/L)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_short_strap_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Short Strap 合成損益と損益分岐点（2本）を表示。Y軸はM表記。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Short Strap Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Strap: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（ショート・ストラップ）
@app.route("/fx/strap-short", methods=["GET", "POST"])
def fx_strap_short():
    """
    ショート・ストラップ（Short Call×2 + Short Put, 同一K）の損益を表示。
    Premium は GK式で算出（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K  = fget("K",  float, defaults["K"])
        vol = fget("vol", float, defaults["vol"])
        r_dom = fget("r_dom", float, defaults["r_dom"])
        r_for = fget("r_for", float, defaults["r_for"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K=defaults["K"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call）
    prem_put  = garman_kohlhagen_put(S0, K, r_dom/100.0, r_for/100.0, sigma, T)   # 受取（Short Put）

    # 受取プレミアム合計（JPY/USD）と損益分岐点
    premium_sum = 2.0 * prem_call + prem_put
    # S <= K: 2C + P - (K - S) = 0 → S = K - (2C + P)
    be_low  = K - premium_sum
    # S >= K: 2C + P - 2(S - K) = 0 → S = K + (2C + P)/2
    be_high = K + premium_sum / 2.0

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_short_strap(K, prem_call, prem_put, qty, smin, smax, points)

    # レンジ内 最大/最小（参考）
    idx_min = int(np.argmin(pl["combo"]))
    idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min]); range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max]); range_cap_st   = float(S_T[idx_max])

    # プレミアム金額（JPY）
    prem_call_jpy   = prem_call * qty
    prem_put_jpy    = prem_put  * qty
    premium_sum_jpy = 2.0 * prem_call_jpy + prem_put_jpy

    # グラフ①：全体P/L
    fig = draw_chart_short_strap(S_T, pl, S0, K, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：BE強調
    fig_be = draw_short_strap_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_strap_short_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_strap_short.html",
        png_b64=png_b64,
        png_b64_strap_short_be=png_b64_strap_short_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（ショート・ストラップ）
@app.route("/fx/download_csv_strap_short", methods=["POST"])
def fx_download_csv_strap_short():
    """
    Short Strap のグリッドCSVダウンロード。
    出力列: S_T, ShortCall2x_PnL, ShortPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K         = fget("K", float, 150.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_short_strap(K, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "ShortCall2x_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['short_call2'][i]:.6f}",
            f"{pl['short_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="short_strap_pnl.csv")

# ===============================================================================================================================================
# =============== Butterfly (Call) =====================
# =====================================================

def payoff_components_bfly_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty):
    """
    Long Call Butterfly: +C(K1) -2C(K2) +C(K3)
    prem_* は JPY/USD（Long=支払, Short=受取の符号は式で処理）
    """
    leg1 = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty         # +C(K1)
    leg2 = ( 2*prem_c2 - 2*np.maximum(S_T - K2, 0.0)) * qty     # -2C(K2)
    leg3 = (-prem_c3 + np.maximum(S_T - K3, 0.0)) * qty         # +C(K3)
    combo = leg1 + leg2 + leg3
    return {"c_k1": leg1, "c_k2x2": leg2, "c_k3": leg3, "combo": combo}


def build_grid_and_rows_bfly_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty,
                                  smin, smax, points, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_bfly_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty)
    rows = [{
        "st": float(S_T[i]),
        "c_k1":   float(pl["c_k1"][i]),
        "c_k2x2": float(pl["c_k2x2"][i]),
        "c_k3":   float(pl["c_k3"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def _find_breakevens_from_grid(S_T, y):
    """
    グリッド上のゼロ交差を線形補間で最大2点返す（一般形K1<K2<K3でもOK）。
    """
    bes = []
    for i in range(1, len(S_T)):
        y0, y1 = y[i-1], y[i]
        if y0 == 0:
            bes.append(S_T[i-1])
        if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
            x0, x1 = S_T[i-1], S_T[i]
            t = abs(y0) / (abs(y0) + abs(y1))
            bes.append(x0 + (x1 - x0) * t)
    # 重複除去
    uniq = []
    for x in bes:
        if not any(abs(x - u) < 1e-6 for u in uniq):
            uniq.append(x)
    return uniq[:2]


def draw_chart_bfly_call(S_T, pl, S0, K1, K2, K3, be_vals):
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["c_k1"],   label="Long Call(K1) P/L", color="blue")
    ax.plot(S_T, pl["c_k2x2"], label="-2× Call(K2) P/L",  color="red")
    ax.plot(S_T, pl["c_k3"],   label="Long Call(K3) P/L", color="purple")
    ax.plot(S_T, pl["combo"],  label="Combo (Call Butterfly)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["c_k1"], pl["c_k2x2"], pl["c_k3"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (K1,"K1",":"), (K2,"K2",":"), (K3,"K3",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Long Call Butterfly: +C(K1) − 2C(K2) + C(K3)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_bfly_call_breakeven(S_T, combo_pl, be_vals):
    """
    Call Butterfly の合成P/Lと損益分岐点にフォーカスしたグラフ。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Call Butterfly Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Call Butterfly: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


@app.route("/fx/butterfly-call", methods=["GET","POST"])
def fx_butterfly_call():
    defaults = dict(
        S0=150.0, K1=148.0, K2=150.0, K3=152.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=160.0, points=321,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0",float,defaults["S0"]); K1=fget("K1",float,defaults["K1"])
        K2=fget("K2",float,defaults["K2"]); K3=fget("K3",float,defaults["K3"])
        vol=fget("vol",float,defaults["vol"]); r_dom=fget("r_dom",float,defaults["r_dom"])
        r_for=fget("r_for",float,defaults["r_for"]); qty=fget("qty",float,defaults["qty"])
        smin=fget("smin",float,defaults["smin"]); smax=fget("smax",float,defaults["smax"])
        points=int(fget("points",float,defaults["points"])); months=fget("months",float,defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GKプレミアム（JPY/USD）
    T = max(months, 0.0001)/12.0; sigma = max(vol,0.0)/100.0
    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)
    prem_c3 = garman_kohlhagen_call(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)

    # ネット・デビット（支払, JPY/USD）
    premium_net = prem_c1 - 2.0*prem_c2 + prem_c3
    premium_net_jpy = premium_net * qty

    # グリッド
    S_T, pl, rows = build_grid_and_rows_bfly_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty, smin, smax, points)

    # 損益分岐点（数値近似）
    be_vals = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = be_vals[0] if len(be_vals)>0 else None
    be_high = be_vals[1] if len(be_vals)>1 else None

    # レンジ内の最大/最小損益
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_bfly_call(S_T, pl, S0, K1, K2, K3, be_vals)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_bfly_call_breakeven(S_T, pl["combo"], be_vals)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_bfly_call_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_butterfly_call.html",
        png_b64=png_b64,
        png_b64_bfly_call_be=png_b64_bfly_call_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2, prem_c3=prem_c3, premium_net=premium_net,
        premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


@app.route("/fx/download_csv_butterfly_call", methods=["POST"])
def fx_download_csv_butterfly_call():
    """
    Call Butterfly のCSV
    列: S_T, CallK1_PnL, Short2CallK2_PnL, CallK3_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    K1=fget("K1",float,148.0); K2=fget("K2",float,150.0); K3=fget("K3",float,152.0)
    prem_c1=fget("prem_c1",float,0.8); prem_c2=fget("prem_c2",float,0.8); prem_c3=fget("prem_c3",float,0.8)
    qty=fget("qty",float,1_000_000.0); smin=fget("smin",float,140.0); smax=fget("smax",float,160.0)
    points=fget("points",float,321); step=0.25

    S_T, pl, _ = build_grid_and_rows_bfly_call(K1,K2,K3,prem_c1,prem_c2,prem_c3,qty,smin,smax,points,step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)","CallK1_PnL(JPY)","Short2CallK2_PnL(JPY)","CallK3_PnL(JPY)","Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['c_k1'][i]:.6f}", f"{pl['c_k2x2'][i]:.6f}", f"{pl['c_k3'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="butterfly_call_pnl.csv")


# =============== Butterfly (Put) =====================

def payoff_components_bfly_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty):
    """
    Long Put Butterfly: +P(K1) -2P(K2) +P(K3)  （K1<K2<K3）
    """
    leg1 = (-prem_p1 + np.maximum(K1 - S_T, 0.0)) * qty         # +P(K1)
    leg2 = ( 2*prem_p2 - 2*np.maximum(K2 - S_T, 0.0)) * qty     # -2P(K2)
    leg3 = (-prem_p3 + np.maximum(K3 - S_T, 0.0)) * qty         # +P(K3)
    combo = leg1 + leg2 + leg3
    return {"p_k1": leg1, "p_k2x2": leg2, "p_k3": leg3, "combo": combo}


def build_grid_and_rows_bfly_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty,
                                 smin, smax, points, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_bfly_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty)
    rows = [{
        "st": float(S_T[i]),
        "p_k1":   float(pl["p_k1"][i]),
        "p_k2x2": float(pl["p_k2x2"][i]),
        "p_k3":   float(pl["p_k3"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_bfly_put(S_T, pl, S0, K1, K2, K3, be_vals):
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["p_k1"],   label="Long Put(K1) P/L", color="blue")
    ax.plot(S_T, pl["p_k2x2"], label="-2× Put(K2) P/L",  color="red")
    ax.plot(S_T, pl["p_k3"],   label="Long Put(K3) P/L", color="purple")
    ax.plot(S_T, pl["combo"],  label="Combo (Put Butterfly)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["p_k1"], pl["p_k2x2"], pl["p_k3"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (K1,"K1",":"), (K2,"K2",":"), (K3,"K3",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Long Put Butterfly: +P(K1) − 2P(K2) + P(K3)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_bfly_put_breakeven(S_T, combo_pl, be_vals):
    """
    Put Butterfly の合成P/Lと損益分岐点にフォーカスしたグラフ。
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Put Butterfly Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Put Butterfly: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


@app.route("/fx/butterfly-put", methods=["GET","POST"])
def fx_butterfly_put():
    defaults = dict(
        S0=150.0, K1=148.0, K2=150.0, K3=152.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=160.0, points=321,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0",float,defaults["S0"]); K1=fget("K1",float,defaults["K1"])
        K2=fget("K2",float,defaults["K2"]); K3=fget("K3",float,defaults["K3"])
        vol=fget("vol",float,defaults["vol"]); r_dom=fget("r_dom",float,defaults["r_dom"])
        r_for=fget("r_for",float,defaults["r_for"]); qty=fget("qty",float,defaults["qty"])
        smin=fget("smin",float,defaults["smin"]); smax=fget("smax",float,defaults["smax"])
        points=int(fget("points",float,defaults["points"])); months=fget("months",float,defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GKプレミアム（JPY/USD）
    T = max(months, 0.0001)/12.0; sigma = max(vol,0.0)/100.0
    prem_p1 = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)
    prem_p2 = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)
    prem_p3 = garman_kohlhagen_put(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)

    # ネット・デビット（支払, JPY/USD）
    premium_net = prem_p1 - 2.0*prem_p2 + prem_p3
    premium_net_jpy = premium_net * qty

    # グリッド
    S_T, pl, rows = build_grid_and_rows_bfly_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty, smin, smax, points)

    # 損益分岐点（数値近似）
    be_vals = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = be_vals[0] if len(be_vals)>0 else None
    be_high = be_vals[1] if len(be_vals)>1 else None

    # レンジ内の最大/最小損益
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_bfly_put(S_T, pl, S0, K1, K2, K3, be_vals)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_bfly_put_breakeven(S_T, pl["combo"], be_vals)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_bfly_put_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_butterfly_put.html",
        png_b64=png_b64,
        png_b64_bfly_put_be=png_b64_bfly_put_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_p1=prem_p1, prem_p2=prem_p2, prem_p3=prem_p3, premium_net=premium_net,
        premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


@app.route("/fx/download_csv_butterfly_put", methods=["POST"])
def fx_download_csv_butterfly_put():
    """
    Put Butterfly のCSV
    列: S_T, PutK1_PnL, Short2PutK2_PnL, PutK3_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    K1=fget("K1",float,148.0); K2=fget("K2",float,150.0); K3=fget("K3",float,152.0)
    prem_p1=fget("prem_p1",float,0.8); prem_p2=fget("prem_p2",float,0.8); prem_p3=fget("prem_p3",float,0.8)
    qty=fget("qty",float,1_000_000.0); smin=fget("smin",float,140.0); smax=fget("smax",float,160.0)
    points=fget("points",float,321); step=0.25

    S_T, pl, _ = build_grid_and_rows_bfly_put(K1,K2,K3,prem_p1,prem_p2,prem_p3,qty,smin,smax,points,step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)","PutK1_PnL(JPY)","Short2PutK2_PnL(JPY)","PutK3_PnL(JPY)","Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['p_k1'][i]:.6f}", f"{pl['p_k2x2'][i]:.6f}", f"{pl['p_k3'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="butterfly_put_pnl.csv")

# ================= Iron Condor (Long wings + short inner) =====================

def payoff_components_iron_condor(S_T, K1, K2, K3, K4,
                                  prem_p1, prem_p2, prem_c3, prem_c4, qty):
    """
    Iron Condor（K1<K2<K3<K4）:
      Long Put(K1), Short Put(K2), Short Call(K3), Long Call(K4)
    prem_* は JPY/USD（Long=支払, Short=受取の向きは式で処理）
    """
    long_put_k1   = (-prem_p1 + np.maximum(K1 - S_T, 0.0)) * qty
    short_put_k2  = ( prem_p2 - np.maximum(K2 - S_T, 0.0)) * qty
    short_call_k3 = ( prem_c3 - np.maximum(S_T - K3, 0.0)) * qty
    long_call_k4  = (-prem_c4 + np.maximum(S_T - K4, 0.0)) * qty
    combo = long_put_k1 + short_put_k2 + short_call_k3 + long_call_k4
    return {
        "long_put_k1": long_put_k1,
        "short_put_k2": short_put_k2,
        "short_call_k3": short_call_k3,
        "long_call_k4": long_call_k4,
        "combo": combo
    }


def build_grid_and_rows_iron_condor(K1, K2, K3, K4,
                                    prem_p1, prem_p2, prem_c3, prem_c4, qty,
                                    smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_iron_condor(S_T, K1, K2, K3, K4,
                                       prem_p1, prem_p2, prem_c3, prem_c4, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_put_k1":   float(pl["long_put_k1"][i]),
        "short_put_k2":  float(pl["short_put_k2"][i]),
        "short_call_k3": float(pl["short_call_k3"][i]),
        "long_call_k4":  float(pl["long_call_k4"][i]),
        "combo":         float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_iron_condor(S_T, pl, S0, K1, K2, K3, K4, be_low, be_high):
    """
    Iron Condor の損益グラフ（Y軸M表記）
    5本：Long Put(K1)（青）/ Short Put(K2)（赤）/ Short Call(K3)（オレンジ）/ Long Call(K4)（紫）/ 合成（緑）
    縦線：S0, K1..K4, BE(±)
    """
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_put_k1"],   label="Long Put(K1) P/L",   color="blue")
    ax.plot(S_T, pl["short_put_k2"],  label="Short Put(K2) P/L",  color="red")
    ax.plot(S_T, pl["short_call_k3"], label="Short Call(K3) P/L", color="orange")
    ax.plot(S_T, pl["long_call_k4"],  label="Long Call(K4) P/L",  color="purple")
    ax.plot(S_T, pl["combo"],         label="Combo (Iron Condor)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_put_k1"], pl["short_put_k2"], pl["short_call_k3"], pl["long_call_k4"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (K1,"K1",":"), (K2,"K2",":"), (K3,"K3",":"), (K4,"K4",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_low,  linestyle="--", linewidth=1.2); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.2); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Iron Condor: Long Put(K1) + Short Put(K2) + Short Call(K3) + Long Call(K4)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_iron_condor_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Iron Condor の合成損益と損益分岐点だけにフォーカスしたグラフ（Y軸M表記）
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Iron Condor Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Iron Condor: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（Iron Condor）
@app.route("/fx/iron-condor", methods=["GET", "POST"])
def fx_iron_condor():
    """
    Iron Condor（Long Put K1, Short Put K2, Short Call K3, Long Call K4）の損益を表示。
    Premium は GK式（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K1=146.0, K2=148.0, K3=152.0, K4=154.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=160.0, points=321,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        K3=fget("K3", float, defaults["K3"])
        K4=fget("K4", float, defaults["K4"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]; K4=defaults["K4"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_p1 = garman_kohlhagen_put (S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long Put
    prem_p2 = garman_kohlhagen_put (S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short Put
    prem_c3 = garman_kohlhagen_call(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # Short Call
    prem_c4 = garman_kohlhagen_call(S0, K4, r_dom/100.0, r_for/100.0, sigma, T)  # Long Call

    # 受取クレジット（JPY/USD）と損益分岐点
    credit = (prem_p2 + prem_c3) - (prem_p1 + prem_c4)
    credit_jpy = credit * qty
    be_low  = K2 - credit
    be_high = K3 + credit

    # 翼幅と最大損失（JPY/USD → JPY）
    wing_lower = max(K2 - K1, 0.0)
    wing_upper = max(K4 - K3, 0.0)
    wing_max   = max(wing_lower, wing_upper)
    max_loss_per_usd = max(wing_max - credit, 0.0)
    max_loss_jpy     = max_loss_per_usd * qty

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_iron_condor(
        K1, K2, K3, K4, prem_p1, prem_p2, prem_c3, prem_c4, qty, smin, smax, points
    )

    # レンジ内 最大/最小（参考）
    idx_min = int(np.argmin(pl["combo"]))
    idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min]); range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max]); range_cap_st   = float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_iron_condor(S_T, pl, S0, K1, K2, K3, K4, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_iron_condor_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_iron_condor_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_iron_condor.html",
        png_b64=png_b64,
        png_b64_iron_condor_be=png_b64_iron_condor_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力（プレミアム・信用/損失・BE）
        prem_p1=prem_p1, prem_p2=prem_p2, prem_c3=prem_c3, prem_c4=prem_c4,
        credit=credit, credit_jpy=credit_jpy,
        be_low=be_low, be_high=be_high,
        wing_lower=wing_lower, wing_upper=wing_upper, wing_max=wing_max,
        max_loss_per_usd=max_loss_per_usd, max_loss_jpy=max_loss_jpy,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Iron Condor）
@app.route("/fx/download_csv_iron_condor", methods=["POST"])
def fx_download_csv_iron_condor():
    """
    Iron Condor のCSV出力。
    列: S_T, LongPutK1_PnL, ShortPutK2_PnL, ShortCallK3_PnL, LongCallK4_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K1        = fget("K1", float, 146.0)
    K2        = fget("K2", float, 148.0)
    K3        = fget("K3", float, 152.0)
    K4        = fget("K4", float, 154.0)
    prem_p1   = fget("prem_p1", float, 0.80)
    prem_p2   = fget("prem_p2", float, 0.80)
    prem_c3   = fget("prem_c3", float, 0.80)
    prem_c4   = fget("prem_c4", float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 140.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 321)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_iron_condor(
        K1, K2, K3, K4, prem_p1, prem_p2, prem_c3, prem_c4, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        "S_T(USD/JPY)",
        "LongPutK1_PnL(JPY)",
        "ShortPutK2_PnL(JPY)",
        "ShortCallK3_PnL(JPY)",
        "LongCallK4_PnL(JPY)",
        "Combo_PnL(JPY)"
    ])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_put_k1'][i]:.6f}",
            f"{pl['short_put_k2'][i]:.6f}",
            f"{pl['short_call_k3'][i]:.6f}",
            f"{pl['long_call_k4'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="iron_condor_pnl.csv")

# ================= Iron Butterfly (Short body + long wings) ===================

def payoff_components_iron_butterfly(S_T, K1, Kmid, K4,
                                     prem_p1, prem_pmid, prem_cmid, prem_c4, qty):
    """
    Iron Butterfly（K1 < Kmid < K4）:
      Long Put(K1), Short Put(Kmid), Short Call(Kmid), Long Call(K4)
    prem_* は JPY/USD（Long=支払, Short=受取は式で処理）
    """
    long_put_k1    = (-prem_p1   + np.maximum(K1   - S_T, 0.0)) * qty
    short_put_kmid = ( prem_pmid - np.maximum(Kmid - S_T, 0.0)) * qty
    short_call_kmid= ( prem_cmid - np.maximum(S_T  - Kmid, 0.0)) * qty
    long_call_k4   = (-prem_c4   + np.maximum(S_T  - K4,   0.0)) * qty
    combo = long_put_k1 + short_put_kmid + short_call_kmid + long_call_k4
    return {
        "long_put_k1": long_put_k1,
        "short_put_kmid": short_put_kmid,
        "short_call_kmid": short_call_kmid,
        "long_call_k4": long_call_k4,
        "combo": combo
    }


def build_grid_and_rows_iron_butterfly(K1, Kmid, K4,
                                       prem_p1, prem_pmid, prem_cmid, prem_c4, qty,
                                       smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_iron_butterfly(S_T, K1, Kmid, K4,
                                          prem_p1, prem_pmid, prem_cmid, prem_c4, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_put_k1":     float(pl["long_put_k1"][i]),
        "short_put_kmid":  float(pl["short_put_kmid"][i]),
        "short_call_kmid": float(pl["short_call_kmid"][i]),
        "long_call_k4":    float(pl["long_call_k4"][i]),
        "combo":           float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_iron_butterfly(S_T, pl, S0, K1, Kmid, K4, be_low, be_high):
    """
    Iron Butterfly の損益グラフ（Y軸M表記）
    5本：Long Put(K1)/Short Put(Kmid)/Short Call(Kmid)/Long Call(K4)/Combo
    縦線：S0, K1, Kmid, K4, BE(±)
    """
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_put_k1"],     label="Long Put(K1) P/L",     color="blue")
    ax.plot(S_T, pl["short_put_kmid"],  label="Short Put(Kmid) P/L",  color="red")
    ax.plot(S_T, pl["short_call_kmid"], label="Short Call(Kmid) P/L", color="orange")
    ax.plot(S_T, pl["long_call_k4"],    label="Long Call(K4) P/L",    color="purple")
    ax.plot(S_T, pl["combo"],           label="Combo (Iron Butterfly)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_put_k1"], pl["short_put_kmid"], pl["short_call_kmid"], pl["long_call_k4"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (K1,"K1",":"), (Kmid,"Kmid",":"), (K4,"K4",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    ax.axvline(be_low,  linestyle="--", linewidth=1.2); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.2); ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Iron Butterfly: Long Put(K1) + Short Put/Call(Kmid) + Long Call(K4)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_iron_butterfly_breakeven(S_T, combo_pl, be_low, be_high):
    """
    Iron Butterfly の合成損益と損益分岐点フォーカス（Y軸M表記）
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Iron Butterfly Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(be_low,  linestyle="--", linewidth=1.5, label="Break-even (Lower)")
    ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.5, label="Break-even (Upper)")
    ax.text(be_high, y_top, f"BE+={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Iron Butterfly: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（Iron Butterfly）
@app.route("/fx/iron-butterfly", methods=["GET", "POST"])
def fx_iron_butterfly():
    """
    Iron Butterfly（Long Put K1, Short Put/Call Kmid, Long Call K4）の損益を表示。
    Premium は GK式（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, K1=148.0, Kmid=150.0, K4=152.0,   # 対称の例
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=160.0, points=321,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        Kmid=fget("Kmid", float, defaults["Kmid"])
        K4=fget("K4", float, defaults["K4"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; Kmid=defaults["Kmid"]; K4=defaults["K4"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_p1   = garman_kohlhagen_put (S0, K1,   r_dom/100.0, r_for/100.0, sigma, T)  # Long Put
    prem_pmid = garman_kohlhagen_put (S0, Kmid, r_dom/100.0, r_for/100.0, sigma, T)  # Short Put
    prem_cmid = garman_kohlhagen_call(S0, Kmid, r_dom/100.0, r_for/100.0, sigma, T)  # Short Call
    prem_c4   = garman_kohlhagen_call(S0, K4,   r_dom/100.0, r_for/100.0, sigma, T)  # Long Call

    # 受取クレジット（JPY/USD）と損益分岐点
    credit = (prem_pmid + prem_cmid) - (prem_p1 + prem_c4)
    credit_jpy = credit * qty
    be_low  = Kmid - credit
    be_high = Kmid + credit

    # 翼幅と最大損失（JPY/USD → JPY）
    wing_lower = max(Kmid - K1, 0.0)
    wing_upper = max(K4   - Kmid, 0.0)
    wing_max   = max(wing_lower, wing_upper)
    max_loss_per_usd = max(wing_max - credit, 0.0)
    max_loss_jpy     = max_loss_per_usd * qty

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_iron_butterfly(
        K1, Kmid, K4, prem_p1, prem_pmid, prem_cmid, prem_c4, qty, smin, smax, points
    )

    # レンジ内 最大/最小（参考）
    idx_min = int(np.argmin(pl["combo"]))
    idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min]); range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max]); range_cap_st   = float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_iron_butterfly(S_T, pl, S0, K1, Kmid, K4, be_low, be_high)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_iron_butterfly_breakeven(S_T, pl["combo"], be_low, be_high)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_iron_bfly_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_iron_butterfly.html",
        png_b64=png_b64,
        png_b64_iron_bfly_be=png_b64_iron_bfly_be,
        # 入力
        S0=S0, K1=K1, Kmid=Kmid, K4=K4, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力（プレミアム・信用/損失・BE）
        prem_p1=prem_p1, prem_pmid=prem_pmid, prem_cmid=prem_cmid, prem_c4=prem_c4,
        credit=credit, credit_jpy=credit_jpy,
        be_low=be_low, be_high=be_high,
        wing_lower=wing_lower, wing_upper=wing_upper, wing_max=wing_max,
        max_loss_per_usd=max_loss_per_usd, max_loss_jpy=max_loss_jpy,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Iron Butterfly）
@app.route("/fx/download_csv_iron_butterfly", methods=["POST"])
def fx_download_csv_iron_butterfly():
    """
    Iron Butterfly のCSV出力。
    列: S_T, LongPutK1_PnL, ShortPutKmid_PnL, ShortCallKmid_PnL, LongCallK4_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K1        = fget("K1", float, 148.0)
    Kmid      = fget("Kmid", float, 150.0)
    K4        = fget("K4", float, 152.0)
    prem_p1   = fget("prem_p1", float, 0.80)
    prem_pmid = fget("prem_pmid", float, 0.80)
    prem_cmid = fget("prem_cmid", float, 0.80)
    prem_c4   = fget("prem_c4", float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 140.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 321)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_iron_butterfly(
        K1, Kmid, K4, prem_p1, prem_pmid, prem_cmid, prem_c4, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        "S_T(USD/JPY)",
        "LongPutK1_PnL(JPY)",
        "ShortPutKmid_PnL(JPY)",
        "ShortCallKmid_PnL(JPY)",
        "LongCallK4_PnL(JPY)",
        "Combo_PnL(JPY)"
    ])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_put_k1'][i]:.6f}",
            f"{pl['short_put_kmid'][i]:.6f}",
            f"{pl['short_call_kmid'][i]:.6f}",
            f"{pl['long_call_k4'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="iron_butterfly_pnl.csv")

# ================= Seagull (Long Put + Short Call + Long Call) ================

def payoff_components_seagull(S_T, Kp, Kc1, Kc2, prem_put, prem_call1, prem_call2, qty):
    """
    Seagull 構成（例：Kp < S0 < Kc1 < Kc2）
      + Long Put (Kp)
      + Short Call (Kc1)
      + Long  Call (Kc2)   ※ショート・コールのテールをロング・コールで抑える
    prem_* は JPY/USD（Long=支払, Short=受取の向きは式で処理）
    """
    long_put_pl    = (-prem_put   + np.maximum(Kp  - S_T, 0.0)) * qty
    short_call1_pl = ( prem_call1 - np.maximum(S_T - Kc1, 0.0)) * qty
    long_call2_pl  = (-prem_call2 + np.maximum(S_T - Kc2, 0.0)) * qty
    combo_pl       = long_put_pl + short_call1_pl + long_call2_pl
    return {
        "long_put": long_put_pl,
        "short_call1": short_call1_pl,
        "long_call2": long_call2_pl,
        "combo": combo_pl
    }


def build_grid_and_rows_seagull(Kp, Kc1, Kc2, prem_put, prem_call1, prem_call2, qty,
                                smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_seagull(S_T, Kp, Kc1, Kc2, prem_put, prem_call1, prem_call2, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_put":    float(pl["long_put"][i]),
        "short_call1": float(pl["short_call1"][i]),
        "long_call2":  float(pl["long_call2"][i]),
        "combo":       float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


# （Butterfly 実装で既に追加済みなら再定義不要）
def _find_breakevens_from_grid(S_T, y):
    """
    グリッド上のゼロ交差を線形補間で抽出。最大2点返す。
    """
    bes = []
    for i in range(1, len(S_T)):
        y0, y1 = y[i-1], y[i]
        if y0 == 0:
            bes.append(S_T[i-1])
        if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
            x0, x1 = S_T[i-1], S_T[i]
            t = abs(y0) / (abs(y0) + abs(y1))
            bes.append(x0 + (x1 - x0) * t)
    # 重複/近接の除去
    uniq = []
    for x in bes:
        if not any(abs(x - u) < 1e-6 for u in uniq):
            uniq.append(x)
    return uniq[:2]


def draw_chart_seagull(S_T, pl, S0, Kp, Kc1, Kc2, be_vals):
    """
    Seagull 損益（Y軸M表記）
    4本：Long Put（青）/ Short Call(Kc1)（赤）/ Long Call(Kc2)（紫）/ 合成（緑）
    縦線：S0, Kp, Kc1, Kc2, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_put"],    label="Long Put(Kp) P/L",     color="blue")
    ax.plot(S_T, pl["short_call1"], label="Short Call(Kc1) P/L",  color="red")
    ax.plot(S_T, pl["long_call2"],  label="Long Call(Kc2) P/L",   color="purple")
    ax.plot(S_T, pl["combo"],       label="Combo (Seagull)",      color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_put"], pl["short_call1"], pl["long_call2"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (Kp,"Kp",":"), (Kc1,"Kc1",":"), (Kc2,"Kc2",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Seagull: Long Put(Kp) + Short Call(Kc1) + Long Call(Kc2)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_seagull_breakeven(S_T, combo_pl, be_vals):
    """
    Seagull の合成P/Lと損益分岐点をフォーカス表示
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Seagull Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Seagull: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（Seagull）
@app.route("/fx/seagull", methods=["GET", "POST"])
def fx_seagull():
    """
    Seagull（Long Put Kp + Short Call Kc1 + Long Call Kc2）の損益を表示。
    Premium は GK式（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, Kp=148.0, Kc1=152.0, Kc2=155.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=162.0, points=353,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0   = fget("S0", float, defaults["S0"])
        Kp   = fget("Kp", float, defaults["Kp"])
        Kc1  = fget("Kc1", float, defaults["Kc1"])
        Kc2  = fget("Kc2", float, defaults["Kc2"])
        vol  = fget("vol", float, defaults["vol"])
        r_dom= fget("r_dom", float, defaults["r_dom"])
        r_for= fget("r_for", float, defaults["r_for"])
        qty  = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; Kp=defaults["Kp"]; Kc1=defaults["Kc1"]; Kc2=defaults["Kc2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_put   = garman_kohlhagen_put (S0, Kp,  r_dom/100.0, r_for/100.0, sigma, T)
    prem_call1 = garman_kohlhagen_call(S0, Kc1, r_dom/100.0, r_for/100.0, sigma, T)
    prem_call2 = garman_kohlhagen_call(S0, Kc2, r_dom/100.0, r_for/100.0, sigma, T)

    # ネット・プレミアム（受取 − 支払）
    # = Short Call(Kc1) − Long Put(Kp) − Long Call(Kc2)
    premium_net = prem_call1 - prem_put - prem_call2
    premium_net_jpy = premium_net * qty

    # グリッド
    S_T, pl, rows = build_grid_and_rows_seagull(Kp, Kc1, Kc2, prem_put, prem_call1, prem_call2, qty, smin, smax, points)

    # 損益分岐点（数値近似）
    be_vals = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = be_vals[0] if len(be_vals) > 0 else None
    be_high = be_vals[1] if len(be_vals) > 1 else None

    # レンジ内 最大/最小
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_seagull(S_T, pl, S0, Kp, Kc1, Kc2, be_vals)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_seagull_breakeven(S_T, pl["combo"], be_vals)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_seagull_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_seagull.html",
        png_b64=png_b64,
        png_b64_seagull_be=png_b64_seagull_be,
        # 入力
        S0=S0, Kp=Kp, Kc1=Kc1, Kc2=Kc2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_put=prem_put, prem_call1=prem_call1, prem_call2=prem_call2,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Seagull）
@app.route("/fx/download_csv_seagull", methods=["POST"])
def fx_download_csv_seagull():
    """
    Seagull のCSV出力。
    列: S_T, LongPut_PnL, ShortCallKc1_PnL, LongCallKc2_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    Kp        = fget("Kp", float, 148.0)
    Kc1       = fget("Kc1", float, 152.0)
    Kc2       = fget("Kc2", float, 155.0)
    prem_put  = fget("prem_put", float, 0.80)
    prem_call1= fget("prem_call1", float, 0.80)
    prem_call2= fget("prem_call2", float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 140.0)
    smax      = fget("smax", float, 162.0)
    points    = fget("points", float, 353)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_seagull(Kp, Kc1, Kc2, prem_put, prem_call1, prem_call2,
                                             qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongPut_PnL(JPY)", "ShortCallKc1_PnL(JPY)", "LongCallKc2_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_put'][i]:.6f}",
            f"{pl['short_call1'][i]:.6f}",
            f"{pl['long_call2'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="seagull_pnl.csv")

# ================= Risk Reversal (Long Call + Short Put) ======================

def payoff_components_risk_reversal(S_T, Kc, Kp, prem_call, prem_put, qty):
    """
    Long Call(Kc) と Short Put(Kp) の損益（JPY）。
      - Long Call P/L  = (-prem_call + max(S_T - Kc, 0)) * qty
      - Short Put  P/L = ( prem_put  - max(Kp - S_T, 0)) * qty
    """
    long_call_pl = (-prem_call + np.maximum(S_T - Kc, 0.0)) * qty
    short_put_pl = ( prem_put  - np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl     = long_call_pl + short_put_pl
    return {"long_call": long_call_pl, "short_put": short_put_pl, "combo": combo_pl}


def build_grid_and_rows_risk_reversal(Kc, Kp, prem_call, prem_put, qty,
                                      smin, smax, points, step: float = 0.25):
    """
    0.25刻みのレートグリッドを作成し、RRの各損益を返す。
    points は互換のため受け取り、実際の点数は step により決まる。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_risk_reversal(S_T, Kc, Kp, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_call": float(pl["long_call"][i]),
        "short_put": float(pl["short_put"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


# （すでに他戦略で定義済みなら再定義不要）
def _find_breakevens_from_grid(S_T, y):
    """
    グリッド上のゼロ交差（損益分岐点）を線形補間で抽出。最大2点返す。
    """
    bes = []
    for i in range(1, len(S_T)):
        y0, y1 = y[i-1], y[i]
        if y0 == 0:
            bes.append(S_T[i-1])
        if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
            x0, x1 = S_T[i-1], S_T[i]
            t = abs(y0) / (abs(y0) + abs(y1))
            bes.append(x0 + (x1 - x0) * t)
    uniq = []
    for x in bes:
        if not any(abs(x - u) < 1e-6 for u in uniq):
            uniq.append(x)
    return uniq[:2]


def draw_chart_risk_reversal(S_T, pl, S0, Kc, Kp, be_vals):
    """
    Risk Reversal の損益グラフ。Y軸はM表記。
    3本：Long Call（青）/ Short Put（赤）/ Combo（緑）
    縦線：S0, Kp, Kc, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_call"], label="Long Call(Kc) P/L",  color="blue")
    ax.plot(S_T, pl["short_put"], label="Short Put(Kp) P/L",  color="red")
    ax.plot(S_T, pl["combo"],     label="Combo (Risk Reversal)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_call"], pl["short_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (Kp,"Kp",":"), (Kc,"Kc",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Risk Reversal: Long Call(Kc) + Short Put(Kp)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_rr_breakeven(S_T, combo_pl, be_vals):
    """
    Risk Reversal の合成P/Lと損益分岐点フォーカス表示
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="RR Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Risk Reversal: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（Risk Reversal）
@app.route("/fx/risk-reversal", methods=["GET", "POST"])
def fx_risk_reversal():
    """
    Risk Reversal（Long Call Kc + Short Put Kp）の損益を表示。
    Premium は GK式（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, Kc=152.0, Kp=148.0,     # Kp < S0 < Kc を推奨
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=162.0, points=353,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0   = fget("S0", float, defaults["S0"])
        Kc   = fget("Kc", float, defaults["Kc"])
        Kp   = fget("Kp", float, defaults["Kp"])
        vol  = fget("vol", float, defaults["vol"])
        r_dom= fget("r_dom", float, defaults["r_dom"])
        r_for= fget("r_for", float, defaults["r_for"])
        qty  = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; Kc=defaults["Kc"]; Kp=defaults["Kp"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Call）
    prem_put  = garman_kohlhagen_put (S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Put）

    # ネット・プレミアム（受取 − 支払）
    premium_net      = prem_put - prem_call
    prem_call_jpy    = prem_call * qty
    prem_put_jpy     = prem_put  * qty
    premium_net_jpy  = premium_net * qty

    # グリッド
    S_T, pl, rows = build_grid_and_rows_risk_reversal(Kc, Kp, prem_call, prem_put, qty, smin, smax, points)

    # 損益分岐点（数値近似）
    be_vals = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = be_vals[0] if len(be_vals) > 0 else None
    be_high = be_vals[1] if len(be_vals) > 1 else None

    # レンジ内 最大/最小
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_risk_reversal(S_T, pl, S0, Kc, Kp, be_vals)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_rr_breakeven(S_T, pl["combo"], be_vals)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_rr_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_risk_reversal.html",
        png_b64=png_b64,
        png_b64_rr_be=png_b64_rr_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Risk Reversal）
@app.route("/fx/download_csv_risk_reversal", methods=["POST"])
def fx_download_csv_risk_reversal():
    """
    Risk Reversal のCSV出力。
    列: S_T, LongCall_PnL, ShortPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    Kc        = fget("Kc", float, 152.0)
    Kp        = fget("Kp", float, 148.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 140.0)
    smax      = fget("smax", float, 162.0)
    points    = fget("points", float, 353)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_risk_reversal(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_call'][i]:.6f}",
            f"{pl['short_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="risk_reversal_pnl.csv")

# ================= Short Risk Reversal (Short Call + Long Put) =================

def payoff_components_risk_reversal_short(S_T, Kc, Kp, prem_call, prem_put, qty):
    """
    Short Call(Kc) と Long Put(Kp) の損益（JPY）。
      - Short Call P/L = ( prem_call - max(S_T - Kc, 0)) * qty
      - Long  Put  P/L = (-prem_put  + max(Kp - S_T, 0)) * qty
    """
    short_call_pl = ( prem_call - np.maximum(S_T - Kc, 0.0)) * qty
    long_put_pl   = (-prem_put  + np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl      = short_call_pl + long_put_pl
    return {"short_call": short_call_pl, "long_put": long_put_pl, "combo": combo_pl}


def build_grid_and_rows_risk_reversal_short(Kc, Kp, prem_call, prem_put, qty,
                                            smin, smax, points, step: float = 0.25):
    """
    0.25刻みのレートグリッドを作成し、Short RR の各損益を返す。
    points は互換のため受け取り、実際の点数は step により決まる。
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_risk_reversal_short(S_T, Kc, Kp, prem_call, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_call": float(pl["short_call"][i]),
        "long_put":   float(pl["long_put"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_risk_reversal_short(S_T, pl, S0, Kc, Kp, be_vals):
    """
    Short Risk Reversal の損益グラフ。Y軸はM表記。
    3本：Short Call（青）/ Long Put（赤）/ Combo（緑）
    縦線：S0, Kp, Kc, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["short_call"], label="Short Call(Kc) P/L", color="blue")
    ax.plot(S_T, pl["long_put"],   label="Long  Put(Kp) P/L",  color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Short Risk Reversal)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short_call"], pl["long_put"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (Kp,"Kp",":"), (Kc,"Kc",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Risk Reversal: Short Call(Kc) + Long Put(Kp)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_rr_short_breakeven(S_T, combo_pl, be_vals):
    """
    Short Risk Reversal の合成P/Lと損益分岐点フォーカス表示
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Short RR Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Risk Reversal: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# 画面ルート（Short Risk Reversal）
@app.route("/fx/risk-reversal-short", methods=["GET", "POST"])
def fx_risk_reversal_short():
    """
    Short Risk Reversal（Short Call Kc + Long Put Kp）の損益を表示。
    Premium は GK式（JPY/USD）。
    """
    defaults = dict(
        S0=150.0, Kc=152.0, Kp=148.0,     # Kp < S0 < Kc を推奨
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=162.0, points=353,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0   = fget("S0", float, defaults["S0"])
        Kc   = fget("Kc", float, defaults["Kc"])
        Kp   = fget("Kp", float, defaults["Kp"])
        vol  = fget("vol", float, defaults["vol"])
        r_dom= fget("r_dom", float, defaults["r_dom"])
        r_for= fget("r_for", float, defaults["r_for"])
        qty  = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; Kc=defaults["Kc"]; Kp=defaults["Kp"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)

    # GK式プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call）
    prem_put  = garman_kohlhagen_put (S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Put）

    # ネット・プレミアム（受取 − 支払）＝ Call受取 − Put支払
    premium_net      = prem_call - prem_put
    prem_call_jpy    = prem_call * qty
    prem_put_jpy     = prem_put  * qty
    premium_net_jpy  = premium_net * qty

    # グリッド
    S_T, pl, rows = build_grid_and_rows_risk_reversal_short(Kc, Kp, prem_call, prem_put, qty, smin, smax, points)

    # 損益分岐点（数値近似）
    be_vals = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = be_vals[0] if len(be_vals) > 0 else None
    be_high = be_vals[1] if len(be_vals) > 1 else None

    # レンジ内 最大/最小
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # グラフ①：全体P/L
    fig = draw_chart_risk_reversal_short(S_T, pl, S0, Kc, Kp, be_vals)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # グラフ②：損益分岐点フォーカス
    fig_be = draw_rr_short_breakeven(S_T, pl["combo"], be_vals)
    buf2 = io.BytesIO(); fig_be.savefig(buf2, format="png"); plt.close(fig_be); buf2.seek(0)
    png_b64_rr_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_risk_reversal_short.html",
        png_b64=png_b64,
        png_b64_rr_be=png_b64_rr_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put,
        prem_call_jpy=prem_call_jpy, prem_put_jpy=prem_put_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Short Risk Reversal）
@app.route("/fx/download_csv_risk_reversal_short", methods=["POST"])
def fx_download_csv_risk_reversal_short():
    """
    Short Risk Reversal のCSV出力。
    列: S_T, ShortCall_PnL, LongPut_PnL, Combo_PnL
    """
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    Kc        = fget("Kc", float, 152.0)
    Kp        = fget("Kp", float, 148.0)
    prem_call = fget("prem_call", float, 0.80)
    prem_put  = fget("prem_put",  float, 0.80)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 140.0)
    smax      = fget("smax", float, 162.0)
    points    = fget("points", float, 353)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_risk_reversal_short(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortCall_PnL(JPY)", "LongPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['long_put'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="short_risk_reversal_pnl.csv")

# ================= Backspread (Call / Put) =====================================

def payoff_components_backspread_call(S_T, K_short, K_long, prem_short, prem_long, qty, n_short=1, n_long=2):
    """
    Call Backspread（例：Short 1 @K_short + Long n_long @K_long）の損益（JPY）
      - Short Call P/L = ( prem_short - max(S_T - K_short, 0) ) * qty * n_short
      - Long  Call P/L = (-prem_long + max(S_T - K_long, 0)) * qty * n_long
    """
    short_leg = (prem_short - np.maximum(S_T - K_short, 0.0)) * qty * n_short
    long_leg  = (-prem_long  + np.maximum(S_T - K_long, 0.0)) * qty * n_long
    combo     = short_leg + long_leg
    return {"short": short_leg, "long": long_leg, "combo": combo}


def build_grid_and_rows_backspread_call(K_short, K_long, prem_short, prem_long, qty, n_short, n_long,
                                        smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_backspread_call(S_T, K_short, K_long, prem_short, prem_long, qty, n_short, n_long)
    rows = [{
        "st": float(S_T[i]),
        "short": float(pl["short"][i]),
        "long":  float(pl["long"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def payoff_components_backspread_put(S_T, K_short, K_long, prem_short, prem_long, qty, n_short=1, n_long=2):
    """
    Put Backspread（例：Short 1 @K_short + Long n_long @K_long）の損益（JPY）
      - Short Put P/L = ( prem_short - max(K_short - S_T, 0) ) * qty * n_short
      - Long  Put P/L = (-prem_long  + max(K_long - S_T, 0)) * qty * n_long
    ※ 通常は K_long < K_short （下側を厚く買う）
    """
    short_leg = (prem_short - np.maximum(K_short - S_T, 0.0)) * qty * n_short
    long_leg  = (-prem_long  + np.maximum(K_long - S_T, 0.0)) * qty * n_long
    combo     = short_leg + long_leg
    return {"short": short_leg, "long": long_leg, "combo": combo}


def build_grid_and_rows_backspread_put(K_short, K_long, prem_short, prem_long, qty, n_short, n_long,
                                       smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_backspread_put(S_T, K_short, K_long, prem_short, prem_long, qty, n_short, n_long)
    rows = [{
        "st": float(S_T[i]),
        "short": float(pl["short"][i]),
        "long":  float(pl["long"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


# 損益分岐点抽出（未定義なら定義）
try:
    _find_breakevens_from_grid  # type: ignore
except NameError:
    def _find_breakevens_from_grid(S_T, y):
        bes = []
        for i in range(1, len(S_T)):
            y0, y1 = y[i-1], y[i]
            if y0 == 0:
                bes.append(S_T[i-1])
            if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
                x0, x1 = S_T[i-1], S_T[i]
                t = abs(y0) / (abs(y0) + abs(y1))
                bes.append(x0 + (x1 - x0) * t)
        uniq = []
        for x in bes:
            if not any(abs(x - u) < 1e-6 for u in uniq):
                uniq.append(x)
        return uniq[:2]


def draw_chart_backspread(S_T, pl, S0, K_short, K_long, be_vals, is_call: bool, n_short: int, n_long: int):
    """
    Backspread（Call/Put共通）の損益グラフ。Y軸M表記。
    3本：Short Leg（青）/ Long Leg（赤）/ Combo（緑）
    縦線：S0, K_short, K_long, BE(±)
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["short"], label=f"Short {'Call' if is_call else 'Put'} x{n_short} P/L", color="blue")
    ax.plot(S_T, pl["long"],  label=f"Long  {'Call' if is_call else 'Put'} x{n_long} P/L",  color="red")
    title = f"{'Call' if is_call else 'Put'} Backspread: Short x{n_short} @K1 + Long x{n_long} @K2"
    ax.plot(S_T, pl["combo"], label="Combo (Backspread)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short"], pl["long"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (K_short,"K1",":"), (K_long,"K2",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_be_backspread(S_T, combo_pl, be_vals, is_call: bool, n_short: int, n_long: int):
    """
    Backspread の合成P/Lと損益分岐点フォーカス
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{'Call' if is_call else 'Put'} Backspread: Break-even Focus (x{n_short}/x{n_long})")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# -------- Call Backspread 画面 --------
@app.route("/fx/backspread-call", methods=["GET", "POST"])
def fx_backspread_call():
    defaults = dict(
        S0=150.0, K_short=150.0, K_long=155.0,
        n_short=1, n_long=2,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=170.0, points=481,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0      = fget("S0", float, defaults["S0"])
        K_short = fget("K_short", float, defaults["K_short"])
        K_long  = fget("K_long", float, defaults["K_long"])
        n_short = int(fget("n_short", float, defaults["n_short"]))
        n_long  = int(fget("n_long", float, defaults["n_long"]))
        vol     = fget("vol", float, defaults["vol"])
        r_dom   = fget("r_dom", float, defaults["r_dom"])
        r_for   = fget("r_for", float, defaults["r_for"])
        qty     = fget("qty", float, defaults["qty"])
        smin    = fget("smin", float, defaults["smin"])
        smax    = fget("smax", float, defaults["smax"])
        points  = int(fget("points", float, defaults["points"]))
        months  = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K_short=defaults["K_short"]; K_long=defaults["K_long"]
        n_short=defaults["n_short"]; n_long=defaults["n_long"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma = max(vol,0.0)/100.0

    prem_short = garman_kohlhagen_call(S0, K_short, r_dom/100.0, r_for/100.0, sigma, T)  # 受取
    prem_long  = garman_kohlhagen_call(S0, K_long,  r_dom/100.0, r_for/100.0, sigma, T)  # 支払

    # ネット・プレミアム（受取 − 支払）
    premium_net      = n_short*prem_short - n_long*prem_long
    prem_short_jpy   = prem_short * qty * n_short
    prem_long_jpy    = prem_long  * qty * n_long
    premium_net_jpy  = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_backspread_call(K_short, K_long, prem_short, prem_long, qty, n_short, n_long,
                                                        smin, smax, points)

    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = bes[0] if len(bes)>0 else None
    be_high = bes[1] if len(bes)>1 else None

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_backspread(S_T, pl, S0, K_short, K_long, bes, True, n_short, n_long)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_backspread(S_T, pl["combo"], bes, True, n_short, n_long)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_backspread_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K_short=K_short, K_long=K_long, n_short=n_short, n_long=n_long,
        vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_short=prem_short, prem_long=prem_long,
        prem_short_jpy=prem_short_jpy, prem_long_jpy=prem_long_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Call Backspread）
@app.route("/fx/download_csv_backspread_call", methods=["POST"])
def fx_download_csv_backspread_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K_short  = fget("K_short", float, 150.0)
    K_long   = fget("K_long",  float, 155.0)
    prem_sh  = fget("prem_short", float, 1.20)
    prem_lo  = fget("prem_long",  float, 0.80)
    n_short  = int(fget("n_short", float, 1))
    n_long   = int(fget("n_long",  float, 2))
    qty      = fget("qty", float, 1_000_000.0)
    smin     = fget("smin", float, 140.0)
    smax     = fget("smax", float, 170.0)
    points   = fget("points", float, 481)
    step     = 0.25

    S_T, pl, _ = build_grid_and_rows_backspread_call(K_short, K_long, prem_sh, prem_lo, qty, n_short, n_long,
                                                     smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortLeg_PnL(JPY)", "LongLeg_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short'][i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="backspread_call_pnl.csv")


# -------- Put Backspread 画面 --------
@app.route("/fx/backspread-put", methods=["GET", "POST"])
def fx_backspread_put():
    defaults = dict(
        S0=150.0, K_short=150.0, K_long=145.0,  # 通常 K_long < K_short
        n_short=1, n_long=2,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=481,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0      = fget("S0", float, defaults["S0"])
        K_short = fget("K_short", float, defaults["K_short"])
        K_long  = fget("K_long", float, defaults["K_long"])
        n_short = int(fget("n_short", float, defaults["n_short"]))
        n_long  = int(fget("n_long", float, defaults["n_long"]))
        vol     = fget("vol", float, defaults["vol"])
        r_dom   = fget("r_dom", float, defaults["r_dom"])
        r_for   = fget("r_for", float, defaults["r_for"])
        qty     = fget("qty", float, defaults["qty"])
        smin    = fget("smin", float, defaults["smin"])
        smax    = fget("smax", float, defaults["smax"])
        points  = int(fget("points", float, defaults["points"]))
        months  = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K_short=defaults["K_short"]; K_long=defaults["K_long"]
        n_short=defaults["n_short"]; n_long=defaults["n_long"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma = max(vol,0.0)/100.0

    prem_short = garman_kohlhagen_put(S0, K_short, r_dom/100.0, r_for/100.0, sigma, T)  # 受取
    prem_long  = garman_kohlhagen_put(S0, K_long,  r_dom/100.0, r_for/100.0, sigma, T)  # 支払

    premium_net      = n_short*prem_short - n_long*prem_long
    prem_short_jpy   = prem_short * qty * n_short
    prem_long_jpy    = prem_long  * qty * n_long
    premium_net_jpy  = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_backspread_put(K_short, K_long, prem_short, prem_long, qty, n_short, n_long,
                                                       smin, smax, points)

    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = bes[0] if len(bes)>0 else None
    be_high = bes[1] if len(bes)>1 else None

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_backspread(S_T, pl, S0, K_short, K_long, bes, False, n_short, n_long)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_backspread(S_T, pl["combo"], bes, False, n_short, n_long)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_backspread_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K_short=K_short, K_long=K_long, n_short=n_short, n_long=n_long,
        vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_short=prem_short, prem_long=prem_long,
        prem_short_jpy=prem_short_jpy, prem_long_jpy=prem_long_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Put Backspread）
@app.route("/fx/download_csv_backspread_put", methods=["POST"])
def fx_download_csv_backspread_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K_short  = fget("K_short", float, 150.0)
    K_long   = fget("K_long",  float, 145.0)
    prem_sh  = fget("prem_short", float, 1.20)
    prem_lo  = fget("prem_long",  float, 0.80)
    n_short  = int(fget("n_short", float, 1))
    n_long   = int(fget("n_long",  float, 2))
    qty      = fget("qty", float, 1_000_000.0)
    smin     = fget("smin", float, 130.0)
    smax     = fget("smax", float, 160.0)
    points   = fget("points", float, 481)
    step     = 0.25

    S_T, pl, _ = build_grid_and_rows_backspread_put(K_short, K_long, prem_sh, prem_lo, qty, n_short, n_long,
                                                    smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortLeg_PnL(JPY)", "LongLeg_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short'][i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="backspread_put_pnl.csv")

# ================= Broken Wing Butterfly (Call / Put) ==========================
# 構成（ロングBWB）：Long1 @K1、Short2 @K2、Long1 @K3（K1 < K2 < K3、左右非対称）
# Call/Put でそれぞれの原資産依存部のみ変わります。

def payoff_components_bwb_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty):
    """
    Call Broken Wing Butterfly（Long1@K1, Short2@K2, Long1@K3）の損益（JPY）
      - Long Call(K1)  : (-prem_c1 + max(S_T - K1, 0)) * qty
      - Short Call(K2)x2: ( 2*prem_c2 - 2*max(S_T - K2, 0)) * qty
      - Long Call(K3)  : (-prem_c3 + max(S_T - K3, 0)) * qty
    """
    long_low   = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty
    short_mid  = ( 2.0*prem_c2 - 2.0*np.maximum(S_T - K2, 0.0)) * qty
    long_high  = (-prem_c3 + np.maximum(S_T - K3, 0.0)) * qty
    combo      = long_low + short_mid + long_high
    return {"long_low": long_low, "short_mid": short_mid, "long_high": long_high, "combo": combo}


def build_grid_and_rows_bwb_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty,
                                 smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_bwb_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_low":  float(pl["long_low"][i]),
        "short_mid": float(pl["short_mid"][i]),
        "long_high": float(pl["long_high"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def payoff_components_bwb_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty):
    """
    Put Broken Wing Butterfly（Long1@K1, Short2@K2, Long1@K3）の損益（JPY）
    （K1 < K2 < K3 を仮定）
      - Long Put(K1)   : (-prem_p1 + max(K1 - S_T, 0)) * qty
      - Short Put(K2)x2: ( 2*prem_p2 - 2*max(K2 - S_T, 0)) * qty
      - Long Put(K3)   : (-prem_p3 + max(K3 - S_T, 0)) * qty
    """
    long_low   = (-prem_p1 + np.maximum(K1 - S_T, 0.0)) * qty
    short_mid  = ( 2.0*prem_p2 - 2.0*np.maximum(K2 - S_T, 0.0)) * qty
    long_high  = (-prem_p3 + np.maximum(K3 - S_T, 0.0)) * qty
    combo      = long_low + short_mid + long_high
    return {"long_low": long_low, "short_mid": short_mid, "long_high": long_high, "combo": combo}


def build_grid_and_rows_bwb_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty,
                                smin, smax, points, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_bwb_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_low":  float(pl["long_low"][i]),
        "short_mid": float(pl["short_mid"][i]),
        "long_high": float(pl["long_high"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


# 損益分岐点抽出（未定義なら定義）
try:
    _find_breakevens_from_grid  # type: ignore
except NameError:
    def _find_breakevens_from_grid(S_T, y):
        bes = []
        for i in range(1, len(S_T)):
            y0, y1 = y[i-1], y[i]
            if y0 == 0:
                bes.append(S_T[i-1])
            if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
                x0, x1 = S_T[i-1], S_T[i]
                t = abs(y0) / (abs(y0) + abs(y1))  # ← 既存に合わせるなら修正してください
                bes.append(x0 + (x1 - x0) * t)
        # ↑もし既存にバグがなければ、] を ) に修正してお使いください:
        # t = abs(y0) / (abs(y0) + abs(y1))
        uniq = []
        for x in bes:
            if not any(abs(x - u) < 1e-6 for u in uniq):
                uniq.append(x)
        return uniq[:2]


def draw_chart_bwb(S_T, pl, S0, K1, K2, K3, be_vals, is_call: bool):
    """
    Broken Wing Butterfly（Call/Put共通）の損益グラフ。Y軸M表記。
    4本：Long@K1（青）/ Short@K2×2（赤）/ Long@K3（紫）/ Combo（緑）
    縦線：S0, K1, K2, K3, BE(±)
    """
    fig = plt.figure(figsize=(7.6, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_low"],  label=f"Long {'Call' if is_call else 'Put'} @K1",  color="blue")
    ax.plot(S_T, pl["short_mid"], label=f"Short {'Call' if is_call else 'Put'} @K2 x2", color="red")
    ax.plot(S_T, pl["long_high"], label=f"Long {'Call' if is_call else 'Put'} @K3", color="purple")
    ax.plot(S_T, pl["combo"],     label="Combo (Broken Wing BF)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_low"], pl["short_mid"], pl["long_high"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0,"S0","--"), (K1,"K1",":"), (K2,"K2",":"), (K3,"K3",":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{'Call' if is_call else 'Put'} Broken Wing Butterfly (Long1@K1, Short2@K2, Long1@K3)")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_be_bwb(S_T, combo_pl, be_vals, is_call: bool):
    """
    Broken Wing BF の合成P/Lと損益分岐点フォーカス
    """
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{ i+1 }={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{'Call' if is_call else 'Put'} BWB: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# -------- Call Broken Wing BF 画面 ---------------------------------------------
@app.route("/fx/brokenwing-call", methods=["GET", "POST"])
def fx_brokenwing_call():
    defaults = dict(
        S0=150.0,
        K1=148.0, K2=150.0, K3=154.0,  # 非対称（K2-K1=2, K3-K2=4）
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=140.0, smax=165.0, points=401,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0   = fget("S0", float, defaults["S0"])
        K1   = fget("K1", float, defaults["K1"])
        K2   = fget("K2", float, defaults["K2"])
        K3   = fget("K3", float, defaults["K3"])
        vol  = fget("vol", float, defaults["vol"])
        r_dom= fget("r_dom", float, defaults["r_dom"])
        r_for= fget("r_for", float, defaults["r_for"])
        qty  = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long）
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short）x2
    prem_c3 = garman_kohlhagen_call(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long）

    # ネット・プレミアム（受取 − 支払）= 2*Call(K2) − Call(K1) − Call(K3)
    premium_net     = 2.0*prem_c2 - prem_c1 - prem_c3
    prem_c1_jpy     = prem_c1 * qty
    prem_c2_jpy     = prem_c2 * qty * 2.0
    prem_c3_jpy     = prem_c3 * qty
    premium_net_jpy = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_bwb_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty,
                                                 smin, smax, points)

    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = bes[0] if len(bes)>0 else None
    be_high = bes[1] if len(bes)>1 else None

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_bwb(S_T, pl, S0, K1, K2, K3, bes, True)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_bwb(S_T, pl["combo"], bes, True)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_brokenwing_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2, prem_c3=prem_c3,
        prem_c1_jpy=prem_c1_jpy, prem_c2_jpy=prem_c2_jpy, prem_c3_jpy=prem_c3_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Call Broken Wing BF）
@app.route("/fx/download_csv_brokenwing_call", methods=["POST"])
def fx_download_csv_brokenwing_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K1   = fget("K1", float, 148.0)
    K2   = fget("K2", float, 150.0)
    K3   = fget("K3", float, 154.0)
    prem_c1 = fget("prem_c1", float, 0.0)
    prem_c2 = fget("prem_c2", float, 0.0)
    prem_c3 = fget("prem_c3", float, 0.0)
    qty  = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 140.0)
    smax = fget("smax", float, 165.0)
    points = fget("points", float, 401)
    step = 0.25

    S_T, pl, _ = build_grid_and_rows_bwb_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty,
                                              smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long@K1_PnL(JPY)", "Short@K2x2_PnL(JPY)", "Long@K3_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_low'][i]:.6f}",
            f"{pl['short_mid'][i]:.6f}",
            f"{pl['long_high'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="brokenwing_call_pnl.csv")


# -------- Put Broken Wing BF 画面 ----------------------------------------------
@app.route("/fx/brokenwing-put", methods=["GET", "POST"])
def fx_brokenwing_put():
    defaults = dict(
        S0=150.0,
        K1=144.0, K2=150.0, K3=152.0,  # 非対称（K2-K1=6, K3-K2=2）
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=401,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0   = fget("S0", float, defaults["S0"])
        K1   = fget("K1", float, defaults["K1"])
        K2   = fget("K2", float, defaults["K2"])
        K3   = fget("K3", float, defaults["K3"])
        vol  = fget("vol", float, defaults["vol"])
        r_dom= fget("r_dom", float, defaults["r_dom"])
        r_for= fget("r_for", float, defaults["r_for"])
        qty  = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    prem_p1 = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long）
    prem_p2 = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short）x2
    prem_p3 = garman_kohlhagen_put(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long）

    # ネット・プレミアム（受取 − 支払）= 2*Put(K2) − Put(K1) − Put(K3)
    premium_net     = 2.0*prem_p2 - prem_p1 - prem_p3
    prem_p1_jpy     = prem_p1 * qty
    prem_p2_jpy     = prem_p2 * qty * 2.0
    prem_p3_jpy     = prem_p3 * qty
    premium_net_jpy = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_bwb_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty,
                                                smin, smax, points)

    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    be_low  = bes[0] if len(bes)>0 else None
    be_high = bes[1] if len(bes)>1 else None

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_bwb(S_T, pl, S0, K1, K2, K3, bes, False)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_bwb(S_T, pl["combo"], bes, False)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_brokenwing_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_p1=prem_p1, prem_p2=prem_p2, prem_p3=prem_p3,
        prem_p1_jpy=prem_p1_jpy, prem_p2_jpy=prem_p2_jpy, prem_p3_jpy=prem_p3_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Put Broken Wing BF）
@app.route("/fx/download_csv_brokenwing_put", methods=["POST"])
def fx_download_csv_brokenwing_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default

    K1   = fget("K1", float, 144.0)
    K2   = fget("K2", float, 150.0)
    K3   = fget("K3", float, 152.0)
    prem_p1 = fget("prem_p1", float, 0.0)
    prem_p2 = fget("prem_p2", float, 0.0)
    prem_p3 = fget("prem_p3", float, 0.0)
    qty  = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 130.0)
    smax = fget("smax", float, 160.0)
    points = fget("points", float, 401)
    step = 0.25

    S_T, pl, _ = build_grid_and_rows_bwb_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty,
                                             smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long@K1_PnL(JPY)", "Short@K2x2_PnL(JPY)", "Long@K3_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['long_low'][i]:.6f}",
            f"{pl['short_mid'][i]:.6f}",
            f"{pl['long_high'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="brokenwing_put_pnl.csv")

# ================= Verticals (Bull/Bear Call, Bull/Bear Put) ===================
# 既存ユーティリティ: np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が定義済みである前提。

# --- 汎用：損益分岐点検出（未定義なら定義） ---
try:
    _find_breakevens_from_grid  # type: ignore
except NameError:
    def _find_breakevens_from_grid(S_T, y):
        bes = []
        for i in range(1, len(S_T)):
            y0, y1 = y[i-1], y[i]
            if y0 == 0:
                bes.append(S_T[i-1])
            if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
                x0, x1 = S_T[i-1], S_T[i]
                t = abs(y0) / (abs(y0) + abs(y1))
                bes.append(x0 + (x1 - x0) * t)
        uniq = []
        for x in bes:
            if not any(abs(x - u) < 1e-6 for u in uniq):
                uniq.append(x)
        return uniq[:2]

# --- 共通ビルダー（Call Vertical） ---
def build_grid_and_rows_vertical_call(K_long, K_short, prem_long, prem_short, qty,
                                      smin, smax, points, step: float = 0.25):
    """
    Call Vertical のグリッドと各レッグ損益。
      Long Call  @K_long: (-prem_long + max(S_T - K_long, 0)) * qty
      Short Call @K_short: ( prem_short - max(S_T - K_short, 0)) * qty
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    long_pl  = (-prem_long + np.maximum(S_T - K_long, 0.0)) * qty
    short_pl = ( prem_short - np.maximum(S_T - K_short, 0.0)) * qty
    combo    = long_pl + short_pl
    rows = [{
        "st": float(S_T[i]),
        "long":  float(long_pl[i]),
        "short": float(short_pl[i]),
        "combo": float(combo[i]),
    } for i in range(len(S_T))]
    pl = {"long": long_pl, "short": short_pl, "combo": combo}
    return S_T, pl, rows

# --- 共通ビルダー（Put Vertical） ---
def build_grid_and_rows_vertical_put(K_long, K_short, prem_long, prem_short, qty,
                                     smin, smax, points, step: float = 0.25):
    """
    Put Vertical のグリッドと各レッグ損益。
      Long Put  @K_long: (-prem_long + max(K_long - S_T, 0)) * qty
      Short Put @K_short: ( prem_short - max(K_short - S_T, 0)) * qty
    """
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    long_pl  = (-prem_long + np.maximum(K_long - S_T, 0.0)) * qty
    short_pl = ( prem_short - np.maximum(K_short - S_T, 0.0)) * qty
    combo    = long_pl + short_pl
    rows = [{
        "st": float(S_T[i]),
        "long":  float(long_pl[i]),
        "short": float(short_pl[i]),
        "combo": float(combo[i]),
    } for i in range(len(S_T))]
    pl = {"long": long_pl, "short": short_pl, "combo": combo}
    return S_T, pl, rows

# --- 共通：グラフ描画（P/L＋縦線） ---
def draw_chart_vertical(S_T, pl, S0, K1, K2, be_vals, is_call: bool, title: str):
    """
    3本：Long（青）/ Short（赤）/ Combo（緑）、縦線：S0, K1, K2, BE。
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, pl["long"],  label=f"Long {'Call' if is_call else 'Put'} P/L",  color="blue")
    ax.plot(S_T, pl["short"], label=f"Short {'Call' if is_call else 'Put'} P/L", color="red")
    ax.plot(S_T, pl["combo"], label="Combo (Vertical)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long"], pl["short"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, lab, style in [(S0, "S0", "--"), (K1, "K1", ":"), (K2, "K2", ":")]:
        ax.axvline(x, linestyle=style, linewidth=1)
        ax.text(x, y_top, f"{lab}={x:.1f}", va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# --- 共通：損益分岐点フォーカス ---
def draw_be_vertical(S_T, combo_pl, be_vals, title: str):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{title}: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# ----------------------------- Bull Call Spread --------------------------------
@app.route("/fx/vertical-bull-call", methods=["GET", "POST"])
def fx_vertical_bull_call():
    defaults = dict(
        S0=150.0, K1=148.0, K2=152.0,  # Long@K1 < Short@K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=241,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months,0.0001)/12.0; sigma = max(vol,0.0)/100.0

    prem_long  = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long@K1（支払）
    prem_short = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short@K2（受取）
    premium_net = prem_short - prem_long
    prem_long_jpy = prem_long * qty
    prem_short_jpy= prem_short * qty
    premium_net_jpy = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_vertical_call(K1, K2, prem_long, prem_short, qty,
                                                      smin, smax, points)
    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Bull Call Spread (Long Call@K1, Short Call@K2)"
    fig = draw_chart_vertical(S_T, pl, S0, K1, K2, bes, True, title)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_vertical(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_vertical_bull_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        prem_long=prem_long, prem_short=prem_short,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Bull Call）
@app.route("/fx/download_csv_vertical_bull_call", methods=["POST"])
def fx_download_csv_vertical_bull_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 148.0)
    K2 = fget("K2", float, 152.0)
    prem_long  = fget("prem_long",  float, 1.20)
    prem_short = fget("prem_short", float, 0.80)
    qty   = fget("qty", float, 1_000_000.0)
    smin  = fget("smin", float, 130.0)
    smax  = fget("smax", float, 160.0)
    points= fget("points", float, 241)
    step  = 0.25

    S_T, pl, _ = build_grid_and_rows_vertical_call(K1, K2, prem_long, prem_short, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "ShortCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['short'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="vertical_bull_call.csv")

# ----------------------------- Bear Call Spread --------------------------------
@app.route("/fx/vertical-bear-call", methods=["GET", "POST"])
def fx_vertical_bear_call():
    defaults = dict(
        S0=150.0, K1=150.0, K2=155.0,  # Short@K1 < Long@K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=165.0, points=241,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months,0.0001)/12.0; sigma = max(vol,0.0)/100.0

    prem_short = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Short@K1（受取）
    prem_long  = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Long@K2（支払）
    premium_net = prem_short - prem_long
    prem_long_jpy = prem_long * qty
    prem_short_jpy= prem_short * qty
    premium_net_jpy = premium_net * qty

    # Long@K2, Short@K1 をビルド関数に渡す
    S_T, pl, rows = build_grid_and_rows_vertical_call(K2, K1, prem_long, prem_short, qty,
                                                      smin, smax, points)
    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Bear Call Spread (Short Call@K1, Long Call@K2)"
    fig = draw_chart_vertical(S_T, pl, S0, K1, K2, bes, True, title)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_vertical(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_vertical_bear_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        prem_long=prem_long, prem_short=prem_short,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Bear Call）
@app.route("/fx/download_csv_vertical_bear_call", methods=["POST"])
def fx_download_csv_vertical_bear_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 150.0)
    K2 = fget("K2", float, 155.0)
    prem_long  = fget("prem_long",  float, 0.80)
    prem_short = fget("prem_short", float, 1.20)
    qty   = fget("qty", float, 1_000_000.0)
    smin  = fget("smin", float, 135.0)
    smax  = fget("smax", float, 165.0)
    points= fget("points", float, 241)
    step  = 0.25

    S_T, pl, _ = build_grid_and_rows_vertical_call(K2, K1, prem_long, prem_short, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "ShortCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['short'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="vertical_bear_call.csv")

# ----------------------------- Bull Put Spread ---------------------------------
@app.route("/fx/vertical-bull-put", methods=["GET", "POST"])
def fx_vertical_bull_put():
    defaults = dict(
        S0=150.0, K1=150.0, K2=145.0,  # Short@K1 > Long@K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=241,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months,0.0001)/12.0; sigma = max(vol,0.0)/100.0

    prem_short = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Short@K1（受取）
    prem_long  = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Long@K2（支払）
    premium_net = prem_short - prem_long
    prem_long_jpy = prem_long * qty
    prem_short_jpy= prem_short * qty
    premium_net_jpy = premium_net * qty

    # Long@K2, Short@K1 をビルド関数に渡す
    S_T, pl, rows = build_grid_and_rows_vertical_put(K2, K1, prem_long, prem_short, qty,
                                                     smin, smax, points)
    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Bull Put Spread (Short Put@K1, Long Put@K2)"
    fig = draw_chart_vertical(S_T, pl, S0, K1, K2, bes, False, title)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_vertical(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_vertical_bull_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        prem_long=prem_long, prem_short=prem_short,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Bull Put）
@app.route("/fx/download_csv_vertical_bull_put", methods=["POST"])
def fx_download_csv_vertical_bull_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 150.0)
    K2 = fget("K2", float, 145.0)
    prem_long  = fget("prem_long",  float, 0.80)
    prem_short = fget("prem_short", float, 1.20)
    qty   = fget("qty", float, 1_000_000.0)
    smin  = fget("smin", float, 130.0)
    smax  = fget("smax", float, 160.0)
    points= fget("points", float, 241)
    step  = 0.25

    S_T, pl, _ = build_grid_and_rows_vertical_put(K2, K1, prem_long, prem_short, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongPut_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['short'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="vertical_bull_put.csv")

# ----------------------------- Bear Put Spread ---------------------------------
@app.route("/fx/vertical-bear-put", methods=["GET", "POST"])
def fx_vertical_bear_put():
    defaults = dict(
        S0=150.0, K1=152.0, K2=147.0,  # Long@K1 > Short@K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=160.0, points=241,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months,0.0001)/12.0; sigma = max(vol,0.0)/100.0

    prem_long  = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long@K1（支払）
    prem_short = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short@K2（受取）
    premium_net = prem_short - prem_long
    prem_long_jpy = prem_long * qty
    prem_short_jpy= prem_short * qty
    premium_net_jpy = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_vertical_put(K1, K2, prem_long, prem_short, qty,
                                                     smin, smax, points)
    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Bear Put Spread (Long Put@K1, Short Put@K2)"
    fig = draw_chart_vertical(S_T, pl, S0, K1, K2, bes, False, title)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_vertical(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_vertical_bear_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        prem_long=prem_long, prem_short=prem_short,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Bear Put）
@app.route("/fx/download_csv_vertical_bear_put", methods=["POST"])
def fx_download_csv_vertical_bear_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 152.0)
    K2 = fget("K2", float, 147.0)
    prem_long  = fget("prem_long",  float, 1.20)
    prem_short = fget("prem_short", float, 0.80)
    qty   = fget("qty", float, 1_000_000.0)
    smin  = fget("smin", float, 130.0)
    smax  = fget("smax", float, 160.0)
    points= fget("points", float, 241)
    step  = 0.25

    S_T, pl, _ = build_grid_and_rows_vertical_put(K1, K2, prem_long, prem_short, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongPut_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['short'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="vertical_bear_put.csv")

# ================= Vanilla Condor (Calls / Puts) ===================
# 前提：np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が既に定義済み。

# --- 損益分岐点抽出（未定義なら定義） ---
try:
    _find_breakevens_from_grid  # type: ignore
except NameError:
    def _find_breakevens_from_grid(S_T, y):
        bes = []
        for i in range(1, len(S_T)):
            y0, y1 = y[i-1], y[i]
            if y0 == 0:
                bes.append(S_T[i-1])
            if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
                x0, x1 = S_T[i-1], S_T[i]
                t = abs(y0) / (abs(y0) + abs(y1))
                bes.append(x0 + (x1 - x0) * t)
        uniq = []
        for x in bes:
            if not any(abs(x - u) < 1e-6 for u in uniq):
                uniq.append(x)
        return uniq[:2]

# ---------- Call Condor: payoff / grid / draw ----------
def payoff_components_condor_call(S_T, K1, K2, K3, K4,
                                  prem_c1, prem_c2, prem_c3, prem_c4, qty):
    """
    Long Call@K1, Short Call@K2, Short Call@K3, Long Call@K4
    prem_* は JPY/USD。Longは支払(マイナス), Shortは受取(プラス)に注意。
    """
    long1   = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty
    short2  = ( prem_c2 - np.maximum(S_T - K2, 0.0)) * qty
    short3  = ( prem_c3 - np.maximum(S_T - K3, 0.0)) * qty
    long4   = (-prem_c4 + np.maximum(S_T - K4, 0.0)) * qty
    combo   = long1 + short2 + short3 + long4
    return {"long1": long1, "short2": short2, "short3": short3, "long4": long4, "combo": combo}

def build_grid_and_rows_condor_call(K1, K2, K3, K4,
                                    prem_c1, prem_c2, prem_c3, prem_c4, qty,
                                    smin, smax, points, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_condor_call(S_T, K1, K2, K3, K4, prem_c1, prem_c2, prem_c3, prem_c4, qty)
    rows = [{
        "st": float(S_T[i]),
        "long1":  float(pl["long1"][i]),
        "short2": float(pl["short2"][i]),
        "short3": float(pl["short3"][i]),
        "long4":  float(pl["long4"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_condor(S_T, pl, S0, Ks, be_vals, title, leg_labels):
    """
    5本：Long1(青)/Short2(赤)/Short3(赤点線)/Long4(青点線)/Combo(緑)
    縦線：S0, K1..K4, BE(2本)
    """
    K1, K2, K3, K4 = Ks
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long1"],  label=leg_labels["long1"],  color="blue")
    ax.plot(S_T, pl["short2"], label=leg_labels["short2"], color="red")
    ax.plot(S_T, pl["short3"], label=leg_labels["short3"], color="red", linestyle="--")
    ax.plot(S_T, pl["long4"],  label=leg_labels["long4"],  color="blue", linestyle="--")
    ax.plot(S_T, pl["combo"],  label="Combo (Condor)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long1"], pl["short2"], pl["short3"], pl["long4"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    for i, K in enumerate(Ks, start=1):
        ax.axvline(K, linestyle=":", linewidth=1); ax.text(K, y_top, f"K{i}={K:.1f}", va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_condor(S_T, combo_pl, be_vals, title):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{title}: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# ---------- Put Condor: payoff / grid ----------
def payoff_components_condor_put(S_T, K1, K2, K3, K4,
                                 prem_p1, prem_p2, prem_p3, prem_p4, qty):
    """
    Long Put@K1, Short Put@K2, Short Put@K3, Long Put@K4（K1<K2<K3<K4）
    """
    long1   = (-prem_p1 + np.maximum(K1 - S_T, 0.0)) * qty
    short2  = ( prem_p2 - np.maximum(K2 - S_T, 0.0)) * qty
    short3  = ( prem_p3 - np.maximum(K3 - S_T, 0.0)) * qty
    long4   = (-prem_p4 + np.maximum(K4 - S_T, 0.0)) * qty
    combo   = long1 + short2 + short3 + long4
    return {"long1": long1, "short2": short2, "short3": short3, "long4": long4, "combo": combo}

def build_grid_and_rows_condor_put(K1, K2, K3, K4,
                                   prem_p1, prem_p2, prem_p3, prem_p4, qty,
                                   smin, smax, points, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_condor_put(S_T, K1, K2, K3, K4, prem_p1, prem_p2, prem_p3, prem_p4, qty)
    rows = [{
        "st": float(S_T[i]),
        "long1":  float(pl["long1"][i]),
        "short2": float(pl["short2"][i]),
        "short3": float(pl["short3"][i]),
        "long4":  float(pl["long4"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

# ----------------------- 画面ルート：Call Condor（ロング） -----------------------
@app.route("/fx/vanilla-condor-call", methods=["GET", "POST"])
def fx_vanilla_condor_call():
    defaults = dict(
        S0=150.0, K1=145.0, K2=150.0, K3=155.0, K4=160.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=170.0, points=281,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        K3=fget("K3", float, defaults["K3"])
        K4=fget("K4", float, defaults["K4"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]; K4=defaults["K4"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    # GKで各レッグのプレミアム（JPY/USD）
    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_c3 = garman_kohlhagen_call(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_c4 = garman_kohlhagen_call(S0, K4, r_dom/100.0, r_for/100.0, sigma, T)  # Long

    premium_net = (prem_c2 + prem_c3) - (prem_c1 + prem_c4)  # 受取−支払（Long Condorは通常マイナス）
    prem_long_sum_jpy  = (prem_c1 + prem_c4) * qty
    prem_short_sum_jpy = (prem_c2 + prem_c3) * qty
    premium_net_jpy    = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_condor_call(
        K1, K2, K3, K4, prem_c1, prem_c2, prem_c3, prem_c4, qty, smin, smax, points
    )
    bes = _find_breakevens_from_grid(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Call Condor (Long@K1, Short@K2, Short@K3, Long@K4)"
    leg_labels = {"long1": "Long Call@K1 P/L", "short2": "Short Call@K2 P/L",
                  "short3": "Short Call@K3 P/L", "long4": "Long Call@K4 P/L"}
    fig = draw_chart_condor(S_T, pl, S0, (K1,K2,K3,K4), bes, title, leg_labels)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_condor(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_condor_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2, prem_c3=prem_c3, prem_c4=prem_c4,
        premium_net=premium_net, prem_long_sum_jpy=prem_long_sum_jpy,
        prem_short_sum_jpy=prem_short_sum_jpy, premium_net_jpy=premium_net_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Call Condor）
@app.route("/fx/download_csv_condor_call", methods=["POST"])
def fx_download_csv_condor_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 145.0)
    K2 = fget("K2", float, 150.0)
    K3 = fget("K3", float, 155.0)
    K4 = fget("K4", float, 160.0)
    prem_c1 = fget("prem_c1", float, 1.50)
    prem_c2 = fget("prem_c2", float, 1.10)
    prem_c3 = fget("prem_c3", float, 0.80)
    prem_c4 = fget("prem_c4", float, 0.55)
    qty    = fget("qty", float, 1_000_000.0)
    smin   = fget("smin", float, 135.0)
    smax   = fget("smax", float, 170.0)
    points = fget("points", float, 281)
    step   = 0.25

    S_T, pl, _ = build_grid_and_rows_condor_call(K1, K2, K3, K4, prem_c1, prem_c2, prem_c3, prem_c4,
                                                 qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long1_Call_PnL(JPY)", "Short2_Call_PnL(JPY)",
                "Short3_Call_PnL(JPY)", "Long4_Call_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long1'][i]:.6f}", f"{pl['short2'][i]:.6f}",
                    f"{pl['short3'][i]:.6f}", f"{pl['long4'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True,
                     download_name="vanilla_condor_call_pnl.csv")

# ----------------------- 画面ルート：Put Condor（ロング） -----------------------
@app.route("/fx/vanilla-condor-put", methods=["GET", "POST"])
def fx_vanilla_condor_put():
    defaults = dict(
        S0=150.0, K1=145.0, K2=150.0, K3=155.0, K4=160.0,  # 形はCallと同じ配置
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=170.0, points=281,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        K3=fget("K3", float, defaults["K3"])
        K4=fget("K4", float, defaults["K4"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]; K4=defaults["K4"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    prem_p1 = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long
    prem_p2 = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_p3 = garman_kohlhagen_put(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_p4 = garman_kohlhagen_put(S0, K4, r_dom/100.0, r_for/100.0, sigma, T)  # Long

    premium_net = (prem_p2 + prem_p3) - (prem_p1 + prem_p4)
    prem_long_sum_jpy  = (prem_p1 + prem_p4) * qty
    prem_short_sum_jpy = (prem_p2 + prem_p3) * qty
    premium_net_jpy    = premium_net * qty

    S_T, pl, rows = build_grid_and_rows_condor_put(
        K1, K2, K3, K4, prem_p1, prem_p2, prem_p3, prem_p4, qty, smin, smax, points
    )
    bes = _find_breakevens_from_grid(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Put Condor (Long@K1, Short@K2, Short@K3, Long@K4)"
    leg_labels = {"long1": "Long Put@K1 P/L", "short2": "Short Put@K2 P/L",
                  "short3": "Short Put@K3 P/L", "long4": "Long Put@K4 P/L"}
    fig = draw_chart_condor(S_T, pl, S0, (K1,K2,K3,K4), bes, title, leg_labels)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_condor(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_condor_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_p1=prem_p1, prem_p2=prem_p2, prem_p3=prem_p3, prem_p4=prem_p4,
        premium_net=premium_net, prem_long_sum_jpy=prem_long_sum_jpy,
        prem_short_sum_jpy=prem_short_sum_jpy, premium_net_jpy=premium_net_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Put Condor）
@app.route("/fx/download_csv_condor_put", methods=["POST"])
def fx_download_csv_condor_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 145.0)
    K2 = fget("K2", float, 150.0)
    K3 = fget("K3", float, 155.0)
    K4 = fget("K4", float, 160.0)
    prem_p1 = fget("prem_p1", float, 0.55)
    prem_p2 = fget("prem_p2", float, 0.80)
    prem_p3 = fget("prem_p3", float, 1.10)
    prem_p4 = fget("prem_p4", float, 1.50)
    qty    = fget("qty", float, 1_000_000.0)
    smin   = fget("smin", float, 135.0)
    smax   = fget("smax", float, 170.0)
    points = fget("points", float, 281)
    step   = 0.25

    S_T, pl, _ = build_grid_and_rows_condor_put(K1, K2, K3, K4, prem_p1, prem_p2, prem_p3, prem_p4,
                                                qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long1_Put_PnL(JPY)", "Short2_Put_PnL(JPY)",
                "Short3_Put_PnL(JPY)", "Long4_Put_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long1'][i]:.6f}", f"{pl['short2'][i]:.6f}",
                    f"{pl['short3'][i]:.6f}", f"{pl['long4'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True,
                     download_name="vanilla_condor_put_pnl.csv")

# ======================= Ladder (Call / Put) =======================
# 前提：np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が既に定義済み。

# --- 損益分岐点抽出（共通ヘルパ；未定義なら定義） ---
try:
    _find_breakevens_from_grid  # type: ignore
except NameError:
    def _find_breakevens_from_grid(S_T, y):
        bes = []
        for i in range(1, len(S_T)):
            y0, y1 = y[i-1], y[i]
            if y0 == 0:
                bes.append(S_T[i-1])
            if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
                x0, x1 = S_T[i-1], S_T[i]
                t = abs(y0) / (abs(y0) + abs(y1)) if (abs(y0) + abs(y1))>0 else 0.0
                bes.append(x0 + (x1 - x0) * t)
        uniq = []
        for x in bes:
            if not any(abs(x - u) < 1e-6 for u in uniq):
                uniq.append(x)
        return uniq[:2]

# ================= Long Call Ladder =================
# 構成：Long Call@K1、Short Call@K2、Short Call@K3（K1<K2<K3）

def payoff_components_ladder_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty):
    """
    Long Call(K1), Short Call(K2), Short Call(K3) の損益（JPY）。
    prem_* は JPY/USD。Longは支払(マイナス), Shortは受取(プラス)。
    """
    long1  = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty
    short2 = ( prem_c2 - np.maximum(S_T - K2, 0.0)) * qty
    short3 = ( prem_c3 - np.maximum(S_T - K3, 0.0)) * qty
    combo  = long1 + short2 + short3
    return {"long1": long1, "short2": short2, "short3": short3, "combo": combo}

def build_grid_and_rows_ladder_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty,
                                    smin, smax, points, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_ladder_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty)
    rows = [{
        "st": float(S_T[i]),
        "long1":  float(pl["long1"][i]),
        "short2": float(pl["short2"][i]),
        "short3": float(pl["short3"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_ladder(S_T, pl, S0, Ks, be_vals, title, leg_labels):
    """
    4本：Long1（青）/ Short2（赤）/ Short3（赤点線）/ Combo（緑）
    縦線：S0, K1..K3, BE（最大2本）
    """
    K1, K2, K3 = Ks
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long1"],  label=leg_labels["long1"],  color="blue")
    ax.plot(S_T, pl["short2"], label=leg_labels["short2"], color="red")
    ax.plot(S_T, pl["short3"], label=leg_labels["short3"], color="red", linestyle="--")
    ax.plot(S_T, pl["combo"],  label="Combo (Ladder)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long1"], pl["short2"], pl["short3"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    for i, K in enumerate(Ks, start=1):
        ax.axvline(K, linestyle=":", linewidth=1); ax.text(K, y_top, f"K{i}={K:.1f}", va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_ladder(S_T, combo_pl, be_vals, title):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_vals):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i+1}")
        ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{title}: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# 画面ルート：Long Call Ladder
@app.route("/fx/ladder-call", methods=["GET", "POST"])
def fx_ladder_call():
    defaults = dict(
        S0=150.0, K1=148.0, K2=152.0, K3=156.0,  # K1<K2<K3
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=170.0, points=281,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        K3=fget("K3", float, defaults["K3"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    # GK式プレミアム（JPY/USD）
    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_c3 = garman_kohlhagen_call(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # Short

    # ネット・プレミアム：受取−支払（＋ならクレジット、−ならデビット）
    premium_net = (prem_c2 + prem_c3) - prem_c1
    premium_net_jpy = premium_net * qty
    prem_long_sum_jpy  = prem_c1 * qty
    prem_short_sum_jpy = (prem_c2 + prem_c3) * qty

    S_T, pl, rows = build_grid_and_rows_ladder_call(K1, K2, K3, prem_c1, prem_c2, prem_c3,
                                                    qty, smin, smax, points)
    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Long Call Ladder: Long@K1, Short@K2, Short@K3"
    leg_labels = {"long1": "Long Call@K1 P/L", "short2": "Short Call@K2 P/L", "short3": "Short Call@K3 P/L"}
    fig = draw_chart_ladder(S_T, pl, S0, (K1, K2, K3), bes, title, leg_labels)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_ladder(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_ladder_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2, prem_c3=prem_c3,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_long_sum_jpy=prem_long_sum_jpy, prem_short_sum_jpy=prem_short_sum_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Long Call Ladder）
@app.route("/fx/download_csv_ladder_call", methods=["POST"])
def fx_download_csv_ladder_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 148.0)
    K2 = fget("K2", float, 152.0)
    K3 = fget("K3", float, 156.0)
    prem_c1 = fget("prem_c1", float, 1.20)
    prem_c2 = fget("prem_c2", float, 0.90)
    prem_c3 = fget("prem_c3", float, 0.65)
    qty    = fget("qty", float, 1_000_000.0)
    smin   = fget("smin", float, 135.0)
    smax   = fget("smax", float, 170.0)
    points = fget("points", float, 281)
    step   = 0.25

    S_T, pl, _ = build_grid_and_rows_ladder_call(K1, K2, K3, prem_c1, prem_c2, prem_c3,
                                                 qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long1_Call_PnL(JPY)", "Short2_Call_PnL(JPY)", "Short3_Call_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long1'][i]:.6f}", f"{pl['short2'][i]:.6f}",
                    f"{pl['short3'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="ladder_call_pnl.csv")

# ================= Long Put Ladder =================
# 構成：Short Put@K1、Short Put@K2、Long Put@K3（K1<K2<K3）
#   ※ Putは高いストライクほど価値が高いので、ラダーの「ロング」は高いK側をLongにします

def payoff_components_ladder_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty):
    """
    Short Put(K1), Short Put(K2), Long Put(K3) の損益（JPY）。
    （K1<K2<K3。Longは支払、Shortは受取）
    """
    short1 = ( prem_p1 - np.maximum(K1 - S_T, 0.0)) * qty
    short2 = ( prem_p2 - np.maximum(K2 - S_T, 0.0)) * qty
    long3  = (-prem_p3 + np.maximum(K3 - S_T, 0.0)) * qty
    combo  = short1 + short2 + long3
    return {"short1": short1, "short2": short2, "long3": long3, "combo": combo}

def build_grid_and_rows_ladder_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty,
                                   smin, smax, points, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_ladder_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty)
    rows = [{
        "st": float(S_T[i]),
        "short1": float(pl["short1"][i]),
        "short2": float(pl["short2"][i]),
        "long3":  float(pl["long3"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

# 画面ルート：Long Put Ladder
@app.route("/fx/ladder-put", methods=["GET", "POST"])
def fx_ladder_put():
    defaults = dict(
        S0=150.0, K1=144.0, K2=148.0, K3=152.0,  # K1<K2<K3
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=165.0, points=281,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        K1=fget("K1", float, defaults["K1"])
        K2=fget("K2", float, defaults["K2"])
        K3=fget("K3", float, defaults["K3"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    # GK式プレミアム（JPY/USD）
    prem_p1 = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_p2 = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_p3 = garman_kohlhagen_put(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # Long

    premium_net = (prem_p1 + prem_p2) - prem_p3
    premium_net_jpy = premium_net * qty
    prem_long_sum_jpy  = prem_p3 * qty
    prem_short_sum_jpy = (prem_p1 + prem_p2) * qty

    S_T, pl, rows = build_grid_and_rows_ladder_put(K1, K2, K3, prem_p1, prem_p2, prem_p3,
                                                   qty, smin, smax, points)
    bes = _find_breakevens_from_grid(S_T, pl["combo"])
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    title = "Long Put Ladder: Short@K1, Short@K2, Long@K3"
    # （役割がわかるよう、凡例は各レッグで明記）
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, pl["short1"], label="Short Put@K1 P/L", color="red")
    ax.plot(S_T, pl["short2"], label="Short Put@K2 P/L", color="red", linestyle="--")
    ax.plot(S_T, pl["long3"],  label="Long Put@K3 P/L",  color="blue")
    ax.plot(S_T, pl["combo"],  label="Combo (Ladder)",   color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["short1"], pl["short2"], pl["long3"], pl["combo"]])
    ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, K in enumerate((K1, K2, K3), start=1):
        ax.axvline(K, linestyle=":", linewidth=1); ax.text(K, y_top, f"K{i}={K:.1f}", va="top", ha="left", fontsize=9)
    for i, be in enumerate(bes):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i+1}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 損益分岐点フォーカス
    fig2 = draw_be_ladder(S_T, pl["combo"], bes, title)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_ladder_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_p1=prem_p1, prem_p2=prem_p2, prem_p3=prem_p3,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_long_sum_jpy=prem_long_sum_jpy, prem_short_sum_jpy=prem_short_sum_jpy,
        be_vals=bes,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Long Put Ladder）
@app.route("/fx/download_csv_ladder_put", methods=["POST"])
def fx_download_csv_ladder_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1 = fget("K1", float, 144.0)
    K2 = fget("K2", float, 148.0)
    K3 = fget("K3", float, 152.0)
    prem_p1 = fget("prem_p1", float, 0.55)
    prem_p2 = fget("prem_p2", float, 0.80)
    prem_p3 = fget("prem_p3", float, 1.10)
    qty    = fget("qty", float, 1_000_000.0)
    smin   = fget("smin", float, 130.0)
    smax   = fget("smax", float, 165.0)
    points = fget("points", float, 281)
    step   = 0.25

    S_T, pl, _ = build_grid_and_rows_ladder_put(K1, K2, K3, prem_p1, prem_p2, prem_p3,
                                                qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Short1_Put_PnL(JPY)", "Short2_Put_PnL(JPY)", "Long3_Put_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short1'][i]:.6f}", f"{pl['short2'][i]:.6f}",
                    f"{pl['long3'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="ladder_put_pnl.csv")

# =========================== Guts (Long / Short) ===========================
# 前提：np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が既に定義済み。

# ---------------- 基本パーツ：P/L・グリッド・描画 ----------------

def payoff_components_guts_long(S_T, Kc, Kp, prem_call, prem_put, qty):
    """
    Long Guts = Long ITM Call(Kc) + Long ITM Put(Kp), with Kc < Kp.
      - Long Call P/L = (-prem_call + max(S_T - Kc, 0)) * qty
      - Long Put  P/L = (-prem_put  + max(Kp - S_T, 0)) * qty
    """
    long_call_pl = (-prem_call + np.maximum(S_T - Kc, 0.0)) * qty
    long_put_pl  = (-prem_put  + np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl     = long_call_pl + long_put_pl
    return {"long_call": long_call_pl, "long_put": long_put_pl, "combo": combo_pl}

def payoff_components_guts_short(S_T, Kc, Kp, prem_call, prem_put, qty):
    """
    Short Guts = Short ITM Call(Kc) + Short ITM Put(Kp), with Kc < Kp.
      - Short Call P/L = ( prem_call - max(S_T - Kc, 0)) * qty
      - Short Put  P/L = ( prem_put  - max(Kp - S_T, 0)) * qty
    """
    short_call_pl = (prem_call - np.maximum(S_T - Kc, 0.0)) * qty
    short_put_pl  = (prem_put  - np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl      = short_call_pl + short_put_pl
    return {"short_call": short_call_pl, "short_put": short_put_pl, "combo": combo_pl}

def build_grid_and_rows_guts(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, *,
                             step: float = 0.25, is_short: bool = False):
    """0.25刻みでレートグリッド生成 + 行データ化（Long/Short両対応）"""
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    if is_short:
        pl = payoff_components_guts_short(S_T, Kc, Kp, prem_call, prem_put, qty)
        rows = [{
            "st": float(S_T[i]),
            "short_call": float(pl["short_call"][i]),
            "short_put":  float(pl["short_put"][i]),
            "combo":      float(pl["combo"][i]),
        } for i in range(len(S_T))]
    else:
        pl = payoff_components_guts_long(S_T, Kc, Kp, prem_call, prem_put, qty)
        rows = [{
            "st": float(S_T[i]),
            "long_call": float(pl["long_call"][i]),
            "long_put":  float(pl["long_put"][i]),
            "combo":     float(pl["combo"][i]),
        } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_guts(S_T, pl, S0, Kc, Kp, be_low, be_high, *, is_short: bool):
    """
    Gutsの損益グラフ（Y軸はM表記）
      - Long：3本（Long Call 青 / Long Put 赤 / Combo 緑）
      - Short：3本（Short Call 青 / Short Put 赤 / Combo 緑）
    縦線：S0, Kc, Kp, BE(±)
    """
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    if is_short:
        ax.plot(S_T, pl["short_call"], label="Short Call P/L", color="blue")
        ax.plot(S_T, pl["short_put"],  label="Short Put P/L",  color="red")
    else:
        ax.plot(S_T, pl["long_call"], label="Long Call P/L", color="blue")
        ax.plot(S_T, pl["long_put"],  label="Long Put P/L",  color="red")

    ax.plot(S_T, pl["combo"], label="Combo (Guts)", color="green", linewidth=2)

    series = [pl["combo"]]
    if is_short:
        series += [pl["short_call"], pl["short_put"]]
    else:
        series += [pl["long_call"], pl["long_put"]]
    _set_ylim_tight(ax, series)
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.axvline(S0, linestyle="--", linewidth=1); ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kc, linestyle=":",  linewidth=1); ax.text(Kc, y_top, f"Kc={Kc:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(Kp, linestyle=":",  linewidth=1); ax.text(Kp, y_top, f"Kp={Kp:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(be_low,  linestyle="--", linewidth=1.2); ax.text(be_low,  y_top, f"BE−={be_low:.2f}",  va="top", ha="left", fontsize=9)
    ax.axvline(be_high, linestyle="--", linewidth=1.2); ax.text(be_high, y_top, f"BE＋={be_high:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Short Guts" if is_short else "Long Guts")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_guts(S_T, combo_pl, be_low, be_high, title):
    """損益分岐点フォーカス用の簡易グラフ"""
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate([be_low, be_high], start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{title}: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# -------------------------- 画面ルート：Long Guts --------------------------
@app.route("/fx/guts", methods=["GET", "POST"])
def fx_guts():
    defaults = dict(
        S0=150.0, Kc=148.0, Kp=152.0,   # Kc < Kp（ITMコール＆ITMプット）
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=165.0, points=281,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        Kc=fget("Kc", float, defaults["Kc"])
        Kp=fget("Kp", float, defaults["Kp"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; Kc=defaults["Kc"]; Kp=defaults["Kp"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    # Kc<Kp の体裁に揃える
    if Kc > Kp:
        Kc, Kp = Kp, Kc

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    # GK式プレミアム（JPY/USD）
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # Long
    prem_put  = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)   # Long
    premium_sum = prem_call + prem_put
    premium_sum_jpy = premium_sum * qty

    # 損益分岐点（解析）
    be_low  = Kp - premium_sum
    be_high = Kc + premium_sum

    # グリッドとP/L
    S_T, pl, rows = build_grid_and_rows_guts(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, is_short=False)

    # レンジ内の最大・最小（参考）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：総合
    fig = draw_chart_guts(S_T, pl, S0, Kc, Kp, be_low, be_high, is_short=False)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = draw_be_guts(S_T, pl["combo"], be_low, be_high, "Long Guts")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_guts.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Long Guts）
@app.route("/fx/download_csv_guts", methods=["POST"])
def fx_download_csv_guts():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    Kc        = fget("Kc", float, 148.0)
    Kp        = fget("Kp", float, 152.0)
    prem_call = fget("prem_call", float, 2.00)
    prem_put  = fget("prem_put",  float, 2.00)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 165.0)
    points    = fget("points", float, 281)
    step      = 0.25

    if Kc > Kp: Kc, Kp = Kp, Kc
    S_T, pl, _ = build_grid_and_rows_guts(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, step=step, is_short=False)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongCall_PnL(JPY)", "LongPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long_call'][i]:.6f}", f"{pl['long_put'][i]:.6f}", f"{pl['combo'][i]:.6f}"])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="guts_long_pnl.csv")

# ------------------------- 画面ルート：Short Guts -------------------------
@app.route("/fx/guts-short", methods=["GET", "POST"])
def fx_guts_short():
    defaults = dict(
        S0=150.0, Kc=148.0, Kp=152.0,   # Kc < Kp（ITMコール＆ITMプット）
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=165.0, points=281,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0", float, defaults["S0"])
        Kc=fget("Kc", float, defaults["Kc"])
        Kp=fget("Kp", float, defaults["Kp"])
        vol=fget("vol", float, defaults["vol"])
        r_dom=fget("r_dom", float, defaults["r_dom"])
        r_for=fget("r_for", float, defaults["r_for"])
        qty=fget("qty", float, defaults["qty"])
        smin=fget("smin", float, defaults["smin"])
        smax=fget("smax", float, defaults["smax"])
        points=int(fget("points", float, defaults["points"]))
        months=fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; Kc=defaults["Kc"]; Kp=defaults["Kp"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    if Kc > Kp:
        Kc, Kp = Kp, Kc

    points = clamp_points(points)
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0

    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_put  = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)   # Short
    premium_sum = prem_call + prem_put
    premium_sum_jpy = premium_sum * qty  # 受取合計

    # 損益分岐点（解析）
    be_low  = Kp - premium_sum
    be_high = Kc + premium_sum

    S_T, pl, rows = build_grid_and_rows_guts(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, is_short=True)

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_guts(S_T, pl, S0, Kc, Kp, be_low, be_high, is_short=True)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_guts(S_T, pl["combo"], be_low, be_high, "Short Guts")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_guts_short.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_call=prem_call, prem_put=prem_put, premium_sum=premium_sum, premium_sum_jpy=premium_sum_jpy,
        be_low=be_low, be_high=be_high,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Short Guts）
@app.route("/fx/download_csv_guts_short", methods=["POST"])
def fx_download_csv_guts_short():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    Kc        = fget("Kc", float, 148.0)
    Kp        = fget("Kp", float, 152.0)
    prem_call = fget("prem_call", float, 2.00)
    prem_put  = fget("prem_put",  float, 2.00)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 165.0)
    points    = fget("points", float, 281)
    step      = 0.25

    if Kc > Kp: Kc, Kp = Kp, Kc
    S_T, pl, _ = build_grid_and_rows_guts(Kc, Kp, prem_call, prem_put, qty, smin, smax, points, step=step, is_short=True)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortCall_PnL(JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_call'][i]:.6f}", f"{pl['short_put'][i]:.6f}", f"{pl['combo'][i]:.6f}"])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="guts_short_pnl.csv")

# ======================== Ratio Spread (1×2) =========================
# 前提：np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が既に定義済み。

# ---------- 共通：描画（損益） ----------
def _draw_chart_ratio(S_T, series_dict, S0, K1, K2, be_list, title, leg_labels):
    """
    損益グラフ。series_dictは {"long":..., "shortN":..., "combo":...}
    縦線：S0 / K1 / K2 / BE(s)
    """
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, series_dict["long"],   label=leg_labels["long"],   color="blue")
    ax.plot(S_T, series_dict["shortN"], label=leg_labels["shortN"], color="red")
    ax.plot(S_T, series_dict["combo"],  label="Combo (Ratio 1×2)", color="green", linewidth=2)

    _set_ylim_tight(ax, [series_dict["long"], series_dict["shortN"], series_dict["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    # 参照線
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1)
        ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    # BE（最大2本想定）
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def _draw_be_focus_ratio(S_T, combo_pl, be_list, title):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{title}: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# ---------- 1) Call Ratio Spread（Long 1 @K1, Short 2 @K2, K1<K2） ----------
def payoff_components_ratio_call(S_T, K1, K2, prem_c1, prem_c2, qty, n_short=2):
    long1   = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty
    shortN  = ( n_short * prem_c2 - n_short * np.maximum(S_T - K2, 0.0)) * qty
    combo   = long1 + shortN
    return {"long": long1, "shortN": shortN, "combo": combo}

def build_grid_and_rows_ratio_call(K1, K2, prem_c1, prem_c2, qty, smin, smax, points, *,
                                   step: float = 0.25, n_short: int = 2):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_ratio_call(S_T, K1, K2, prem_c1, prem_c2, qty, n_short=n_short)
    rows = [{
        "st": float(S_T[i]),
        "long":   float(pl["long"][i]),
        "shortN": float(pl["shortN"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

@app.route("/fx/ratio-call", methods=["GET", "POST"])
def fx_ratio_call():
    defaults = dict(
        S0=150.0, K1=148.0, K2=152.0,   # K1 < K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=170.0, points=281,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2"); vol=fget("vol")
        r_dom=fget("r_dom"); r_for=fget("r_for"); qty=fget("qty")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; vol=defaults["vol"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]; qty=defaults["qty"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]; months=defaults["months"]

    if K1 > K2: K1, K2 = K2, K1
    points = clamp_points(points)

    T = max(months, 0.0001)/12.0
    sigma = max(vol, 0.0)/100.0

    # GK式プレミアム（JPY/USD）
    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long@K1
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Shortx2@K2

    # ネット・プレミアム（受取−支払）
    premium_net = 2*prem_c2 - prem_c1
    premium_net_jpy = premium_net * qty
    prem_long_jpy  = prem_c1 * qty
    prem_short_jpy = 2 * prem_c2 * qty

    # 損益分岐点（解析）
    # BE_mid（K1～K2 内にあるとき）：S = K1 - premium_net（= K1 + デビット額）
    be_list = []
    be_mid  = K1 - premium_net
    if K1 <= be_mid <= K2:
        be_list.append(be_mid)
    # BE_high（K2 以上）：S = 2*K2 - K1 + premium_net
    be_high = 2*K2 - K1 + premium_net
    if be_high >= K2:
        be_list.append(be_high)

    # グリッド＆P/L
    S_T, pl, rows = build_grid_and_rows_ratio_call(K1, K2, prem_c1, prem_c2, qty, smin, smax, points)

    # 参考（レンジ内の最大・最小）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：総合
    leg_labels = {"long": "Long Call@K1 P/L", "shortN": "Short 2 Calls@K2 P/L"}
    fig = _draw_chart_ratio(S_T, pl, S0, K1, K2, be_list, "Call Ratio Spread (1×2)", leg_labels)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = _draw_be_focus_ratio(S_T, pl["combo"], be_list, "Call Ratio Spread (1×2)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_ratio_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Call Ratio）
@app.route("/fx/download_csv_ratio_call", methods=["POST"])
def fx_download_csv_ratio_call():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1=fget("K1", float, 148.0); K2=fget("K2", float, 152.0)
    prem_c1=fget("prem_c1", float, 1.60); prem_c2=fget("prem_c2", float, 1.10)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 135.0); smax=fget("smax", float, 170.0)
    points=fget("points", float, 281); step=0.25
    if K1 > K2: K1, K2 = K2, K1
    S_T, pl, _ = build_grid_and_rows_ratio_call(K1, K2, prem_c1, prem_c2, qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long_Call_K1_PnL(JPY)", "Short_2Calls_K2_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['shortN'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="ratio_call_1x2_pnl.csv")

# ---------- 2) Put Ratio Spread（Long 1 @K2, Short 2 @K1, K1<K2） ----------
def payoff_components_ratio_put(S_T, K1, K2, prem_p1, prem_p2, qty, n_short=2):
    shortN = ( n_short * prem_p1 - n_short * np.maximum(K1 - S_T, 0.0)) * qty  # Short 2 @K1
    long1  = (-prem_p2 + np.maximum(K2 - S_T, 0.0)) * qty                      # Long 1 @K2
    combo  = shortN + long1
    return {"long": long1, "shortN": shortN, "combo": combo}

def build_grid_and_rows_ratio_put(K1, K2, prem_p1, prem_p2, qty, smin, smax, points, *,
                                  step: float = 0.25, n_short: int = 2):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_ratio_put(S_T, K1, K2, prem_p1, prem_p2, qty, n_short=n_short)
    rows = [{
        "st": float(S_T[i]),
        "shortN": float(pl["shortN"][i]),
        "long":   float(pl["long"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

@app.route("/fx/ratio-put", methods=["GET", "POST"])
def fx_ratio_put():
    defaults = dict(
        S0=150.0, K1=148.0, K2=152.0,   # K1 < K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=165.0, points=281,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try: return cast(val)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2"); vol=fget("vol")
        r_dom=fget("r_dom"); r_for=fget("r_for"); qty=fget("qty")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; vol=defaults["vol"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]; qty=defaults["qty"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]; months=defaults["months"]

    if K1 > K2: K1, K2 = K2, K1
    points = clamp_points(points)

    T = max(months, 0.0001)/12.0
    sigma = max(vol, 0.0)/100.0

    # GK式プレミアム（JPY/USD）
    prem_p1 = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)   # Shortx2@K1
    prem_p2 = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)   # Long@K2

    # ネット・プレミアム（受取−支払）
    premium_net = 2*prem_p1 - prem_p2
    premium_net_jpy = premium_net * qty
    prem_long_jpy  = prem_p2 * qty
    prem_short_jpy = 2 * prem_p1 * qty

    # 損益分岐点（解析）
    # BE_low（K1 以下）：S = 2*K1 - K2 - premium_net
    be_list = []
    be_low = 2*K1 - K2 - premium_net
    if be_low <= K1:
        be_list.append(be_low)
    # BE_high（K1～K2 内、デビット時のみ）：S = K2 + premium_net（premium_net<0のとき K1～K2 に入る）
    be_high = K2 + premium_net
    if (K1 <= be_high <= K2):
        be_list.append(be_high)

    # グリッド＆P/L
    S_T, pl, rows = build_grid_and_rows_ratio_put(K1, K2, prem_p1, prem_p2, qty, smin, smax, points)

    # 参考（レンジ内の最大・最小）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：総合
    leg_labels = {"long": "Long Put@K2 P/L", "shortN": "Short 2 Puts@K1 P/L"}
    fig = _draw_chart_ratio(S_T, pl, S0, K1, K2, be_list, "Put Ratio Spread (1×2)", leg_labels)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = _draw_be_focus_ratio(S_T, pl["combo"], be_list, "Put Ratio Spread (1×2)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_ratio_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_p1=prem_p1, prem_p2=prem_p2,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Put Ratio）
@app.route("/fx/download_csv_ratio_put", methods=["POST"])
def fx_download_csv_ratio_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try: return cast(val)
        except Exception: return default
    K1=fget("K1", float, 148.0); K2=fget("K2", float, 152.0)
    prem_p1=fget("prem_p1", float, 0.90); prem_p2=fget("prem_p2", float, 1.30)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 130.0); smax=fget("smax", float, 165.0)
    points=fget("points", float, 281); step=0.25
    if K1 > K2: K1, K2 = K2, K1
    S_T, pl, _ = build_grid_and_rows_ratio_put(K1, K2, prem_p1, prem_p2, qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Short_2Puts_K1_PnL(JPY)", "Long_Put_K2_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['shortN'][i]:.6f}", f"{pl['long'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="ratio_put_1x2_pnl.csv")


# ====================== Christmas Tree (1–3–2) ======================
# 前提：np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が既に定義済み。

# ---------- 共通：P/Lビルドと描画 ----------
def payoff_components_christmas_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty):
    """
    Long Call CT (1–3–2): +1 Call@K1, -3 Call@K2, +2 Call@K3  (K1 < K2 < K3)
      Long Call @K: (-prem + max(S-K,0)) * qty
      Short Call@K: ( prem - max(S-K,0)) * qty
    """
    long1   = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty
    short3  = ( 3*prem_c2 - 3*np.maximum(S_T - K2, 0.0)) * qty
    long2   = ( -2*prem_c3 + 2*np.maximum(S_T - K3, 0.0)) * qty
    combo   = long1 + short3 + long2
    return {"long1": long1, "short3": short3, "long2": long2, "combo": combo}

def payoff_components_christmas_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty):
    """
    Long Put CT (1–3–2): +1 Put@K1, -3 Put@K2, +2 Put@K3  (K1 > K2 > K3)
      Long Put @K:  (-prem + max(K-S,0)) * qty
      Short Put@K:  ( prem - max(K-S,0)) * qty
    """
    long1   = (-prem_p1 + np.maximum(K1 - S_T, 0.0)) * qty
    short3  = ( 3*prem_p2 - 3*np.maximum(K2 - S_T, 0.0)) * qty
    long2   = ( -2*prem_p3 + 2*np.maximum(K3 - S_T, 0.0)) * qty
    combo   = long1 + short3 + long2
    return {"long1": long1, "short3": short3, "long2": long2, "combo": combo}

def build_grid_and_rows_ct_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty,
                                smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_christmas_call(S_T, K1, K2, K3, prem_c1, prem_c2, prem_c3, qty)
    rows = [{
        "st": float(S_T[i]),
        "long1":  float(pl["long1"][i]),
        "short3": float(pl["short3"][i]),
        "long2":  float(pl["long2"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def build_grid_and_rows_ct_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty,
                               smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_christmas_put(S_T, K1, K2, K3, prem_p1, prem_p2, prem_p3, qty)
    rows = [{
        "st": float(S_T[i]),
        "long1":  float(pl["long1"][i]),
        "short3": float(pl["short3"][i]),
        "long2":  float(pl["long2"][i]),
        "combo":  float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def _draw_chart_ct(S_T, pl, S0, K1, K2, K3, be_list, title, leg_long1, leg_short3, leg_long2):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long1"],  label=leg_long1,  color="blue")
    ax.plot(S_T, pl["short3"], label=leg_short3, color="red")
    ax.plot(S_T, pl["long2"],  label=leg_long2,  color="purple")
    ax.plot(S_T, pl["combo"],  label="Combo (1–3–2)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long1"], pl["short3"], pl["long2"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":"), (K3, f"K3={K3:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(title)
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def _draw_be_focus_ct(S_T, combo_pl, be_list, title):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"{title}: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# ---------- 1) Call Christmas Tree (1–3–2) ----------
@app.route("/fx/christmas-call", methods=["GET", "POST"])
def fx_christmas_call():
    defaults = dict(
        S0=150.0, K1=148.0, K2=152.0, K3=156.0,   # K1 < K2 < K3
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=175.0, points=321,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2"); K3=fget("K3")
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax")
        points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    # K1<K2<K3 に整列（プレミアムは後で計算するのでOK）
    Ks = sorted([K1, K2, K3]); K1, K2, K3 = Ks[0], Ks[1], Ks[2]
    points = clamp_points(points)

    T = max(months, 0.0001)/12.0
    sigma = max(vol, 0.0)/100.0

    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long 1
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short 3
    prem_c3 = garman_kohlhagen_call(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)  # Long 2

    # ネット・プレミアム（受取 − 支払）
    premium_net = 3*prem_c2 - (prem_c1 + 2*prem_c3)
    premium_net_jpy = premium_net * qty
    prem_long_jpy  = (prem_c1 + 2*prem_c3) * qty
    prem_short_jpy = (3*prem_c2) * qty

    # 損益分岐点（解析）
    # Region 定義から：BE1 = K1 - premium_net（K1〜K2にあるとき）、BE2 = (3K2 - K1 + premium_net)/2（K2〜K3にあるとき）
    be_list = []
    be1 = K1 - premium_net
    if K1 <= be1 <= K2:
        be_list.append(be1)
    be2 = (3.0*K2 - K1 + premium_net)/2.0
    if K2 <= be2 <= K3:
        be_list.append(be2)

    # グリッド＆P/L
    S_T, pl, rows = build_grid_and_rows_ct_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty, smin, smax, points)

    # 参考（レンジ内の最大・最小）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：総合
    fig = _draw_chart_ct(
        S_T, pl, S0, K1, K2, K3, be_list,
        "Call Christmas Tree (1–3–2)",
        leg_long1="Long 1 Call@K1 P/L", leg_short3="Short 3 Calls@K2 P/L", leg_long2="Long 2 Calls@K3 P/L"
    )
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = _draw_be_focus_ct(S_T, pl["combo"], be_list, "Call Christmas Tree (1–3–2)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_christmas_call.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2, prem_c3=prem_c3,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Call Christmas Tree）
@app.route("/fx/download_csv_christmas_call", methods=["POST"])
def fx_download_csv_christmas_call():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    K1=fget("K1", float, 148.0); K2=fget("K2", float, 152.0); K3=fget("K3", float, 156.0)
    prem_c1=fget("prem_c1", float, 1.80); prem_c2=fget("prem_c2", float, 1.30); prem_c3=fget("prem_c3", float, 0.95)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 135.0); smax=fget("smax", float, 175.0)
    points=fget("points", float, 321); step=0.25

    S_T, pl, _ = build_grid_and_rows_ct_call(K1, K2, K3, prem_c1, prem_c2, prem_c3, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long1_Call_K1_PnL(JPY)", "Short3_Calls_K2_PnL(JPY)", "Long2_Calls_K3_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long1'][i]:.6f}", f"{pl['short3'][i]:.6f}", f"{pl['long2'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="christmas_tree_call_pnl.csv")

# ---------- 2) Put Christmas Tree (1–3–2) ----------
@app.route("/fx/christmas-put", methods=["GET", "POST"])
def fx_christmas_put():
    defaults = dict(
        S0=150.0, K1=152.0, K2=148.0, K3=144.0,   # K1 > K2 > K3
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=130.0, smax=165.0, points=321,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2"); K3=fget("K3")
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax")
        points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    # K1>K2>K3 に整列
    Ks = sorted([K1, K2, K3], reverse=True); K1, K2, K3 = Ks[0], Ks[1], Ks[2]
    points = clamp_points(points)

    T = max(months, 0.0001)/12.0
    sigma = max(vol, 0.0)/100.0

    prem_p1 = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)   # Long 1
    prem_p2 = garman_kohlhagen_put(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)   # Short 3
    prem_p3 = garman_kohlhagen_put(S0, K3, r_dom/100.0, r_for/100.0, sigma, T)   # Long 2

    # ネット・プレミアム（受取 − 支払）
    premium_net = 3*prem_p2 - (prem_p1 + 2*prem_p3)
    premium_net_jpy = premium_net * qty
    prem_long_jpy  = (prem_p1 + 2*prem_p3) * qty
    prem_short_jpy = (3*prem_p2) * qty

    # 損益分岐点（解析）
    # BE1 = K1 + premium_net（K2〜K1にあるとき）、BE2 = (3K2 - K1 - premium_net)/2（K3〜K2にあるとき）
    be_list = []
    be1 = K1 + premium_net
    if K2 <= be1 <= K1:
        be_list.append(be1)
    be2 = (3.0*K2 - K1 - premium_net)/2.0
    if K3 <= be2 <= K2:
        be_list.append(be2)

    # グリッド＆P/L
    S_T, pl, rows = build_grid_and_rows_ct_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty, smin, smax, points)

    # 参考（レンジ内の最大・最小）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：総合
    fig = _draw_chart_ct(
        S_T, pl, S0, K1, K2, K3, be_list,
        "Put Christmas Tree (1–3–2)",
        leg_long1="Long 1 Put@K1 P/L", leg_short3="Short 3 Puts@K2 P/L", leg_long2="Long 2 Puts@K3 P/L"
    )
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = _draw_be_focus_ct(S_T, pl["combo"], be_list, "Put Christmas Tree (1–3–2)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_christmas_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_p1=prem_p1, prem_p2=prem_p2, prem_p3=prem_p3,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_long_jpy=prem_long_jpy, prem_short_jpy=prem_short_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Put Christmas Tree）
@app.route("/fx/download_csv_christmas_put", methods=["POST"])
def fx_download_csv_christmas_put():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    K1=fget("K1", float, 152.0); K2=fget("K2", float, 148.0); K3=fget("K3", float, 144.0)
    prem_p1=fget("prem_p1", float, 1.80); prem_p2=fget("prem_p2", float, 1.25); prem_p3=fget("prem_p3", float, 0.85)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 130.0); smax=fget("smax", float, 165.0)
    points=fget("points", float, 321); step=0.25

    S_T, pl, _ = build_grid_and_rows_ct_put(K1, K2, K3, prem_p1, prem_p2, prem_p3, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Long1_Put_K1_PnL(JPY)", "Short3_Puts_K2_PnL(JPY)", "Long2_Puts_K3_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long1'][i]:.6f}", f"{pl['short3'][i]:.6f}", f"{pl['long2'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="christmas_tree_put_pnl.csv")

# ======================== Jade Lizard =========================
# 前提：np, plt, io, base64, request, render_template, send_file,
# garman_kohlhagen_call, garman_kohlhagen_put, _arange_inclusive, _set_ylim_tight,
# _format_y_as_m, clamp_points が既に定義済み。

# ---- P/L ----
def payoff_components_jade_lizard(S_T, Kp, K1, K2, prem_put, prem_c1, prem_c2, qty):
    """
    Jade Lizard: Short Put(Kp) + Short Call(K1) + Long Call(K2) with Kp < K1 < K2
      Short Put : ( prem_put - max(Kp - S, 0)) * qty
      Short Call: ( prem_c1  - max(S - K1, 0)) * qty
      Long  Call: (-prem_c2  + max(S - K2, 0)) * qty
    """
    short_put  = ( prem_put - np.maximum(Kp - S_T, 0.0)) * qty
    short_call = ( prem_c1  - np.maximum(S_T - K1, 0.0)) * qty
    long_call  = (-prem_c2  + np.maximum(S_T - K2, 0.0)) * qty
    combo = short_put + short_call + long_call
    return {"short_put": short_put, "short_call": short_call, "long_call": long_call, "combo": combo}

def build_grid_and_rows_jade(Kp, K1, K2, prem_put, prem_c1, prem_c2, qty,
                             smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_jade_lizard(S_T, Kp, K1, K2, prem_put, prem_c1, prem_c2, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_put":  float(pl["short_put"][i]),
        "short_call": float(pl["short_call"][i]),
        "long_call":  float(pl["long_call"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

# ---- Draw ----
def draw_chart_jade(S_T, pl, S0, Kp, K1, K2, be_list):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["short_put"],  label="Short Put P/L",  color="red")
    ax.plot(S_T, pl["short_call"], label="Short Call P/L", color="blue")
    ax.plot(S_T, pl["long_call"],  label="Long Call P/L",  color="purple")
    ax.plot(S_T, pl["combo"],      label="Combo (Jade Lizard)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["short_put"], pl["short_call"], pl["long_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (Kp, f"Kp={Kp:.1f}", ":"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)"); ax.set_ylabel("P/L (JPY)")
    ax.set_title("Jade Lizard: Short Put(Kp) + Short Call(K1) + Long Call(K2)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_jade(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L"); _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}"); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)"); ax.set_ylabel("P/L (JPY)"); ax.set_title("Jade Lizard: Break-even Focus")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout(); return fig

# ---- Route ----
@app.route("/fx/jade-lizard", methods=["GET", "POST"])
def fx_jade_lizard():
    defaults = dict(
        S0=150.0, Kp=146.0, K1=152.0, K2=156.0,  # Kp < K1 < K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000, smin=130.0, smax=170.0, points=321, months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); Kp=fget("Kp"); K1=fget("K1"); K2=fget("K2")
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax"); points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; Kp=defaults["Kp"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]; months=defaults["months"]

    # 並び整える
    Ks_sorted = sorted([Kp, K1, K2]); Kp, K1, K2 = Ks_sorted[0], Ks_sorted[1], Ks_sorted[2]
    points = clamp_points(points)

    # GK premium（JPY/USD）
    T = max(months, 0.0001)/12.0; sigma = max(vol, 0.0)/100.0
    prem_put = garman_kohlhagen_put(S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)
    prem_c1  = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)
    prem_c2  = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)

    # ネット・クレジット
    premium_net = prem_put + prem_c1 - prem_c2
    premium_net_jpy = premium_net * qty
    prem_recv_jpy = (prem_put + prem_c1) * qty
    prem_pay_jpy  = prem_c2 * qty
    width_call = K2 - K1
    upside_buffer = premium_net - width_call  # >=0 なら上昇側ノーリスク

    # 損益分岐点
    be_list = []
    be_low = Kp - premium_net
    if be_low <= Kp: be_list.append(be_low)
    be_mid = K1 + premium_net
    if K1 <= be_mid <= K2: be_list.append(be_mid)

    # グリッド
    S_T, pl, rows = build_grid_and_rows_jade(Kp, K1, K2, prem_put, prem_c1, prem_c2, qty, smin, smax, points)

    # 参考
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図
    fig = draw_chart_jade(S_T, pl, S0, Kp, K1, K2, be_list)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_jade(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_jade_lizard.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, Kp=Kp, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_put=prem_put, prem_c1=prem_c1, prem_c2=prem_c2,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        prem_recv_jpy=prem_recv_jpy, prem_pay_jpy=prem_pay_jpy,
        width_call=width_call, upside_buffer=upside_buffer,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Jade Lizard）
@app.route("/fx/download_csv_jade_lizard", methods=["POST"])
def fx_download_csv_jade_lizard():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    Kp=fget("Kp", float, 146.0); K1=fget("K1", float, 152.0); K2=fget("K2", float, 156.0)
    prem_put=fget("prem_put", float, 1.10); prem_c1=fget("prem_c1", float, 1.40); prem_c2=fget("prem_c2", float, 0.95)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 130.0); smax=fget("smax", float, 170.0)
    points=fget("points", float, 321); step=0.25

    S_T, pl, _ = build_grid_and_rows_jade(Kp, K1, K2, prem_put, prem_c1, prem_c2, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortPut_PnL(JPY)", "ShortCall_PnL(JPY)", "LongCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_put'][i]:.6f}", f"{pl['short_call'][i]:.6f}", f"{pl['long_call'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="jade_lizard_pnl.csv")


# ======================== Big Lizard =========================
# Short Straddle (Put+Call @K1) + Long OTM Call @K2
def payoff_components_big_lizard(S_T, K1, K2, prem_put1, prem_call1, prem_call2, qty):
    short_put  = ( prem_put1  - np.maximum(K1 - S_T, 0.0)) * qty
    short_call = ( prem_call1 - np.maximum(S_T - K1, 0.0)) * qty
    long_call  = (-prem_call2 + np.maximum(S_T - K2, 0.0)) * qty
    combo = short_put + short_call + long_call
    return {"short_put": short_put, "short_call": short_call, "long_call": long_call, "combo": combo}

def build_grid_and_rows_big(K1, K2, prem_put1, prem_call1, prem_call2, qty,
                            smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_big_lizard(S_T, K1, K2, prem_put1, prem_call1, prem_call2, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_put":  float(pl["short_put"][i]),
        "short_call": float(pl["short_call"][i]),
        "long_call":  float(pl["long_call"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_big(S_T, pl, S0, K1, K2, be_list, width, credit):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["short_put"],  label="Short Put@K1 P/L",  color="red")
    ax.plot(S_T, pl["short_call"], label="Short Call@K1 P/L", color="blue")
    ax.plot(S_T, pl["long_call"],  label="Long Call@K2 P/L",  color="purple")
    ax.plot(S_T, pl["combo"],      label="Combo (Big Lizard)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["short_put"], pl["short_call"], pl["long_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)"); ax.set_ylabel("P/L (JPY)")
    title_extra = " (Upside OK)" if credit >= width else ""
    ax.set_title(f"Big Lizard: Short Straddle@K1 + Long Call@K2{title_extra}")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_big(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L"); _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}"); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)"); ax.set_ylabel("P/L (JPY)"); ax.set_title("Big Lizard: Break-even Focus")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout(); return fig

@app.route("/fx/big-lizard", methods=["GET", "POST"])
def fx_big_lizard():
    defaults = dict(
        S0=150.0, K1=150.0, K2=156.0,  # Short Put @K1, Short Call @K1, Long Call @K2
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000, smin=130.0, smax=170.0, points=321, months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2")
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax"); points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]; months=defaults["months"]

    if K2 < K1: K1, K2 = K2, K1
    points = clamp_points(points)

    # GK premium
    T = max(months, 0.0001)/12.0; sigma = max(vol, 0.0)/100.0
    prem_put1  = garman_kohlhagen_put(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)
    prem_call1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)
    prem_call2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)

    premium_net = prem_put1 + prem_call1 - prem_call2
    premium_net_jpy = premium_net * qty
    width_call = K2 - K1
    upside_buffer = premium_net - width_call  # >=0 で上昇側ノーリスク

    # 損益分岐点
    be_list = []
    be_low = K1 - premium_net
    if be_low <= K1: be_list.append(be_low)
    be_mid = K1 + premium_net
    if K1 <= be_mid <= K2: be_list.append(be_mid)

    # グリッド
    S_T, pl, rows = build_grid_and_rows_big(K1, K2, prem_put1, prem_call1, prem_call2, qty, smin, smax, points)

    # 参考
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図
    fig = draw_chart_big(S_T, pl, S0, K1, K2, be_list, width_call, premium_net)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_big(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_big_lizard.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_put1=prem_put1, prem_call1=prem_call1, prem_call2=prem_call2,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        width_call=width_call, upside_buffer=upside_buffer,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Big Lizard）
@app.route("/fx/download_csv_big_lizard", methods=["POST"])
def fx_download_csv_big_lizard():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    K1=fget("K1", float, 150.0); K2=fget("K2", float, 156.0)
    prem_put1=fget("prem_put1", float, 1.70); prem_call1=fget("prem_call1", float, 1.70); prem_call2=fget("prem_call2", float, 1.10)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 130.0); smax=fget("smax", float, 170.0)
    points=fget("points", float, 321); step=0.25

    S_T, pl, _ = build_grid_and_rows_big(K1, K2, prem_put1, prem_call1, prem_call2, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortPut_K1_PnL(JPY)", "ShortCall_K1_PnL(JPY)", "LongCall_K2_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_put'][i]:.6f}", f"{pl['short_call'][i]:.6f}", f"{pl['long_call'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="big_lizard_pnl.csv")

# ========================= Box (Long) =========================
# 概要：Long Box = Bull Call Spread (+C@K1, -C@K2) ＋ Bear Put Spread (+P@K2, -P@K1)
# K1 < K2 を推奨。理論上、満期損益は「(K2 - K1) - ネットコスト」で一定（S_T に依存せず）。

def payoff_components_box_long(S_T, K1, K2, prem_c1, prem_c2, prem_p1, prem_p2, qty):
    """
    各レッグと合成の損益（JPY）。Premiumは JPY/USD。
      Long Call@K1: (-prem_c1 + max(S - K1, 0)) * qty
      Short Call@K2: ( prem_c2 - max(S - K2, 0)) * qty
      Long Put @K2: (-prem_p2 + max(K2 - S, 0)) * qty
      Short Put @K1: ( prem_p1 - max(K1 - S, 0)) * qty
    合成（Box）は理論上フラット（一定値）。
    """
    long_call_k1  = (-prem_c1 + np.maximum(S_T - K1, 0.0)) * qty
    short_call_k2 = ( prem_c2 - np.maximum(S_T - K2, 0.0)) * qty
    long_put_k2   = (-prem_p2 + np.maximum(K2 - S_T, 0.0)) * qty
    short_put_k1  = ( prem_p1 - np.maximum(K1 - S_T, 0.0)) * qty
    combo = long_call_k1 + short_call_k2 + long_put_k2 + short_put_k1
    return {
        "long_call_k1": long_call_k1,
        "short_call_k2": short_call_k2,
        "long_put_k2": long_put_k2,
        "short_put_k1": short_put_k1,
        "combo": combo
    }

def build_grid_and_rows_box_long(K1, K2, prem_c1, prem_c2, prem_p1, prem_p2, qty,
                                 smin, smax, points, *, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_box_long(S_T, K1, K2, prem_c1, prem_c2, prem_p1, prem_p2, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_call_k1":  float(pl["long_call_k1"][i]),
        "short_call_k2": float(pl["short_call_k2"][i]),
        "long_put_k2":   float(pl["long_put_k2"][i]),
        "short_put_k1":  float(pl["short_put_k1"][i]),
        "combo":         float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_box_long(S_T, pl, S0, K1, K2, flat_pl):
    """
    Box（Long）の損益グラフ。5本：4レッグ＋合成。合成は理論上フラット。
    """
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["long_call_k1"],  label="Long Call @K1 P/L",  color="blue")
    ax.plot(S_T, pl["short_call_k2"], label="Short Call @K2 P/L", color="red")
    ax.plot(S_T, pl["long_put_k2"],   label="Long Put  @K2 P/L",  color="purple")
    ax.plot(S_T, pl["short_put_k1"],  label="Short Put @K1 P/L",  color="orange")
    ax.plot(S_T, pl["combo"],         label="Combo (Box Long)",    color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_call_k1"], pl["short_call_k2"], pl["long_put_k2"], pl["short_put_k1"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    # 参考縦線
    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1)
        ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    # 合成がフラットである旨の注記
    ax.annotate(f"flat ≈ {flat_pl/1e6:.2f}M JPY", xy=(S_T[len(S_T)//2], np.median(pl["combo"])),
                xytext=(6, -10), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Long Box: (+C@K1, -C@K2) + (+P@K2, -P@K1)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# 画面ルート（Long Box）
@app.route("/fx/box", methods=["GET", "POST"])
def fx_box():
    defaults = dict(
        S0=150.0, K1=148.0, K2=152.0,          # K1 < K2 を推奨
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        smin=135.0, smax=165.0, points=241,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try:
                return cast(v)
            except Exception:
                return default if default is not None else defaults[name]
        S0     = fget("S0", float, defaults["S0"])
        K1     = fget("K1", float, defaults["K1"])
        K2     = fget("K2", float, defaults["K2"])
        vol    = fget("vol", float, defaults["vol"])
        r_dom  = fget("r_dom", float, defaults["r_dom"])
        r_for  = fget("r_for", float, defaults["r_for"])
        qty    = fget("qty", float, defaults["qty"])
        smin   = fget("smin", float, defaults["smin"])
        smax   = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
        months = fget("months", float, defaults["months"])
    else:
        S0=defaults["S0"]; K1=defaults["K1"]; K2=defaults["K2"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]

    # K1 < K2 になるよう整列（安全）
    if K1 > K2:
        K1, K2 = K2, K1

    points = clamp_points(points)

    # GK式でプレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_c1 = garman_kohlhagen_call(S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Long
    prem_c2 = garman_kohlhagen_call(S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_p1 = garman_kohlhagen_put (S0, K1, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_p2 = garman_kohlhagen_put (S0, K2, r_dom/100.0, r_for/100.0, sigma, T)  # Long

    # ネット・コスト（支払 - 受取） … Box は通常支払超（デビット）
    #   支払：Long C@K1 + Long P@K2
    #   受取：Short C@K2 + Short P@K1
    premium_pay   = prem_c1 + prem_p2
    premium_recv  = prem_c2 + prem_p1
    premium_net   = premium_pay - premium_recv       # >0 なら支払超
    premium_net_jpy = premium_net * qty
    payoff_locked   = (K2 - K1) * qty                # 満期の固定受取（JPY）
    # 合成PLは理論上一定： (K2-K1 - premium_net) * qty
    flat_pl = (K2 - K1 - premium_net) * qty

    # グリッド
    S_T, pl, rows = build_grid_and_rows_box_long(K1, K2, prem_c1, prem_c2, prem_p1, prem_p2,
                                                 qty, smin, smax, points)

    # レンジ内 最大/最小（理論上は同値）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor    = float(pl["combo"][idx_min]); range_floor_st = float(S_T[idx_min])
    range_cap      = float(pl["combo"][idx_max]); range_cap_st   = float(S_T[idx_max])

    # 図
    fig = draw_chart_box_long(S_T, pl, S0, K1, K2, flat_pl)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return render_template(
        "fx_box.html",
        png_b64=png_b64,
        # 入力
        S0=S0, K1=K1, K2=K2, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_c1=prem_c1, prem_c2=prem_c2, prem_p1=prem_p1, prem_p2=prem_p2,
        premium_pay=premium_pay*qty, premium_recv=premium_recv*qty,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        payoff_locked=payoff_locked, flat_pl=flat_pl,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Long Box）
@app.route("/fx/download_csv_box", methods=["POST"])
def fx_download_csv_box():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try:
            return cast(v)
        except Exception:
            return default
    K1        = fget("K1", float, 148.0)
    K2        = fget("K2", float, 152.0)
    prem_c1   = fget("prem_c1", float, 1.90)
    prem_c2   = fget("prem_c2", float, 1.30)
    prem_p1   = fget("prem_p1", float, 1.30)
    prem_p2   = fget("prem_p2", float, 1.90)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 135.0)
    smax      = fget("smax", float, 165.0)
    points    = fget("points", float, 241)
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_box_long(K1, K2, prem_c1, prem_c2, prem_p1, prem_p2,
                                              qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongCall_K1_PnL(JPY)", "ShortCall_K2_PnL(JPY)",
                "LongPut_K2_PnL(JPY)", "ShortPut_K1_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long_call_k1'][i]:.6f}", f"{pl['short_call_k2'][i]:.6f}",
                    f"{pl['long_put_k2'][i]:.6f}", f"{pl['short_put_k1'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="box_long_pnl.csv")

# =================== Participating Collar =====================
# 現物USDロング + Long Put（100%） + Short Call（参加率 r × ノーション）
# 既存のユーティリティ（np, plt, io, base64, request, render_template, send_file,
# _arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points,
# garman_kohlhagen_call, garman_kohlhagen_put）が前提。

def payoff_components_participating_collar(S_T, S0, Kp, Kc, prem_put, prem_call, qty, part):
    """
    Participating Collar の各損益（JPY）。
      - Spot P/L        = (S_T - S0) * qty
      - Long Put  P/L   = (-prem_put + max(Kp - S_T, 0)) * qty
      - Short Call P/L  = ( prem_call - max(S_T - Kc, 0)) * (qty * part)
        ※ prem_put/prem_call は JPY/USD、part は 0.0〜1.0 を推奨（比率）
    """
    part = max(0.0, min(1.0, float(part)))  # ガード
    spot_pl       = (S_T - S0) * qty
    long_put_pl   = (-prem_put + np.maximum(Kp - S_T, 0.0)) * qty
    short_call_pl = ( prem_call - np.maximum(S_T - Kc, 0.0)) * (qty * part)
    combo_pl      = spot_pl + long_put_pl + short_call_pl
    return {
        "spot": spot_pl,
        "long_put": long_put_pl,
        "short_call": short_call_pl,
        "combo": combo_pl
    }

def build_grid_and_rows_participating_collar(S0, Kp, Kc, prem_put, prem_call, qty, part,
                                             smin, smax, points, *, step: float = 0.25):
    """0.25刻みのレート・グリッド生成と行データ整形。"""
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_participating_collar(S_T, S0, Kp, Kc, prem_put, prem_call, qty, part)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "long_put": float(pl["long_put"][i]),
        "short_call": float(pl["short_call"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def _find_break_evens(S_T, y):
    """数値的に損益分岐点（y=0）を検出（線形補間）。最大2〜3点想定だが任意数に対応。"""
    bes = []
    for i in range(len(S_T)-1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            # 線形補間でゼロ位置
            x0, x1 = S_T[i], S_T[i+1]
            x = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            bes.append(x)
    # 近接重複を整理
    bes_sorted = []
    for b in sorted(bes):
        if not bes_sorted or abs(b - bes_sorted[-1]) > 1e-6:
            bes_sorted.append(b)
    return bes_sorted

def draw_chart_participating_collar(S_T, pl, S0, Kp, Kc, be_list, part):
    """
    Participating Collar の損益グラフ。4本：Spot/Long Put/Short Call/Combo
    参考線：S0, Kp, Kc, BE
    """
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120); ax = fig.add_subplot(111)

    ax.plot(S_T, pl["spot"],        label="Spot USD P/L (vs today)", color="black")
    ax.plot(S_T, pl["long_put"],    label="Long Put P/L",            color="blue")
    ax.plot(S_T, pl["short_call"],  label=f"Short Call P/L (×{part:.2f})", color="red")
    ax.plot(S_T, pl["combo"],       label="Combo (Spot+Put−r·Call)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["spot"], pl["long_put"], pl["short_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (Kp, f"Kp={Kp:.1f}", ":"), (Kc, f"Kc={Kc:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"Participating Collar (Call ratio r={part:.2f})")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_participating(S_T, combo_pl, be_list):
    """合成損益×損益分岐点フォーカス。"""
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Participating Collar: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# -------- 画面ルート --------
@app.route("/fx/participating-collar", methods=["GET", "POST"])
def fx_participating_collar():
    """
    Participating Collar（Spot + Long Put + Short Call×r）
    r=コール売りノーション比（0〜1）。上方の参加度合いは 1−r。
    """
    defaults = dict(
        S0=150.0, Kp=148.0, Kc=152.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        part=0.50,                 # ← コール売りノーション比（0〜1）
        smin=130.0, smax=160.0, points=251,
        months=1.0
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); Kp=fget("Kp"); Kc=fget("Kc")
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax")
        points=int(fget("points")); months=fget("months")
        # part は 0〜100% の数値を入れてしまうことがあるため、POST名を両対応
        part_input = request.form.get("part", "")
        if part_input == "" and "part_pct" in request.form:
            part = float(request.form.get("part_pct", "50.0")) / 100.0
        else:
            part = fget("part", float, defaults["part"])
    else:
        S0=defaults["S0"]; Kp=defaults["Kp"]; Kc=defaults["Kc"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]; months=defaults["months"]; part=defaults["part"]

    # 整理
    points = clamp_points(points)
    part = max(0.0, min(1.0, float(part)))

    # GK式プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    prem_put  = garman_kohlhagen_put (S0, Kp, r_dom/100.0, r_for/100.0, sigma, T)  # 支払（Long Put）
    prem_call = garman_kohlhagen_call(S0, Kc, r_dom/100.0, r_for/100.0, sigma, T)  # 受取（Short Call, ただし r 倍）

    # プレミアム金額（JPY）
    prem_put_jpy    = prem_put  * qty
    prem_call_jpy   = prem_call * (qty * part)
    premium_net     = prem_call * part - prem_put     # JPY/USD
    premium_net_jpy = prem_call_jpy - prem_put_jpy

    # グリッドと損益
    S_T, pl, rows = build_grid_and_rows_participating_collar(S0, Kp, Kc, prem_put, prem_call, qty, part,
                                                             smin, smax, points)

    # 損益分岐点（数値計算）
    be_list = _find_break_evens(S_T, pl["combo"])
    be_low  = be_list[0] if len(be_list) >= 1 else None
    be_high = be_list[1] if len(be_list) >= 2 else None

    # レンジ内の最大/最小
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：総合
    fig = draw_chart_participating_collar(S_T, pl, S0, Kp, Kc, be_list, part)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = draw_be_focus_participating(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_participating_collar.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, Kp=Kp, Kc=Kc, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, part=part,
        # 出力（算出値）
        prem_put=prem_put, prem_call=prem_call,
        prem_put_jpy=prem_put_jpy, prem_call_jpy=prem_call_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be_low=be_low, be_high=be_high, be_vals=be_list,
        participation_share=(1.0 - part),  # 上方参加率
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )


# CSV（Participating Collar）
@app.route("/fx/download_csv_participating_collar", methods=["POST"])
def fx_download_csv_participating_collar():
    """
    Participating Collar のグリッドCSVダウンロード。
    prem_put / prem_call は JPY/USD（画面側で算出済みの値がPOSTされる想定）。
    出力列: S_T, Spot_PnL, LongPut_PnL, ShortCall_PnL(r), Combo_PnL
    """
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default

    S0        = fget("S0", float, 150.0)
    Kp        = fget("Kp", float, 148.0)
    Kc        = fget("Kc", float, 152.0)
    prem_put  = fget("prem_put", float, 0.80)   # JPY/USD
    prem_call = fget("prem_call", float, 0.80)  # JPY/USD
    qty       = fget("qty", float, 1_000_000.0)
    part      = fget("part", float, 0.50)
    smin      = fget("smin", float, 130.0)
    smax      = fget("smax", float, 160.0)
    points    = fget("points", float, 251)      # step 優先
    step      = 0.25

    S_T, pl, _ = build_grid_and_rows_participating_collar(S0, Kp, Kc, prem_put, prem_call, qty, part,
                                                          smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["S_T(USD/JPY)", "Spot_PnL(JPY)", "LongPut_PnL(JPY)",
                     f"ShortCall_PnL(r={part:.2f})(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        writer.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['spot'][i]:.6f}",
            f"{pl['long_put'][i]:.6f}",
            f"{pl['short_call'][i]:.6f}",
            f"{pl['combo'][i]:.6f}",
        ])

    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(
        data, mimetype="text/csv", as_attachment=True,
        download_name="participating_collar_pnl.csv",
    )

# ===================== Conversion / Reversal =====================
# 依存：np, plt, io, base64, request, render_template, send_file
# ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# プレミアム：garman_kohlhagen_call, garman_kohlhagen_put

# ---- 共通：簡易BE検出（水平線でも動くよう閾値で判定） ----
def _be_from_constant_line(S_T, y, eps=1e-6):
    # 水平線のゼロ近傍を検出。通常はBEなし（[]）だが、
    # flatがほぼ0なら代表点を1つ返す（見栄え用）
    if len(y) == 0:
        return []
    val = float(y[0])
    if abs(val) < eps:
        midx = len(S_T) // 2
        return [float(S_T[midx])]
    return []


# ======================= Conversion =============================
# Long Spot + Long Put@K + Short Call@K
def payoff_components_conversion(S_T, S0, K, prem_put, prem_call, qty):
    spot_pl       = (S_T - S0) * qty
    long_put_pl   = (-prem_put + np.maximum(K - S_T, 0.0)) * qty
    short_call_pl = ( prem_call - np.maximum(S_T - K, 0.0)) * qty
    combo_pl      = spot_pl + long_put_pl + short_call_pl
    return {"spot": spot_pl, "long_put": long_put_pl, "short_call": short_call_pl, "combo": combo_pl}

def build_grid_and_rows_conversion(S0, K, prem_put, prem_call, qty,
                                   smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_conversion(S_T, S0, K, prem_put, prem_call, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot":       float(pl["spot"][i]),
        "long_put":   float(pl["long_put"][i]),
        "short_call": float(pl["short_call"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_conversion(S_T, pl, S0, K, flat_pl):
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["spot"],       label="Spot USD P/L",  color="black")
    ax.plot(S_T, pl["long_put"],   label="Long Put P/L",  color="blue")
    ax.plot(S_T, pl["short_call"], label="Short Call P/L",color="red")
    ax.plot(S_T, pl["combo"],      label="Combo (Conversion)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["spot"], pl["long_put"], pl["short_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K, f"K={K:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    ax.annotate(f"flat ≈ {flat_pl/1e6:.2f}M JPY", xy=(S_T[len(S_T)//2], np.median(pl["combo"])),
                xytext=(6, -10), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Conversion: Long Spot + Long Put(K) + Short Call(K)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_conversion(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    if be_list:
        for i, be in enumerate(be_list, start=1):
            ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
            ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    else:
        ax.text(S_T[len(S_T)//2], y_top, "No break-even (flat line)", va="top", ha="center", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Conversion: Break-even Focus")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

@app.route("/fx/conversion", methods=["GET", "POST"])
def fx_conversion():
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000, smin=135.0, smax=165.0, points=241,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K=fget("K"); vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax"); points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K=defaults["K"]; vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma = max(vol, 0.0)/100.0
    prem_put  = garman_kohlhagen_put (S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # Long
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # Short

    # 定数（理論上フラット）： qty * ( (K - S0) + (call - put) )
    flat_pl = ((K - S0) + (prem_call - prem_put)) * qty

    # Grid
    S_T, pl, rows = build_grid_and_rows_conversion(S0, K, prem_put, prem_call, qty, smin, smax, points)

    # 参考
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    prem_put_jpy = prem_put * qty; prem_call_jpy = prem_call * qty; premium_net = prem_call - prem_put
    premium_net_jpy = prem_call_jpy - prem_put_jpy

    # BE（平坦なので通常なし）
    be_list = _be_from_constant_line(S_T, pl["combo"])

    # 図
    fig = draw_chart_conversion(S_T, pl, S0, K, flat_pl)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_conversion(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_conversion.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_put=prem_put, prem_call=prem_call,
        prem_put_jpy=prem_put_jpy, prem_call_jpy=prem_call_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        flat_pl=flat_pl,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        be_vals=be_list,
        rows=rows
    )


# CSV（Conversion）
@app.route("/fx/download_csv_conversion", methods=["POST"])
def fx_download_csv_conversion():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    S0=fget("S0", float, 150.0); K=fget("K", float, 150.0)
    prem_put=fget("prem_put", float, 1.70); prem_call=fget("prem_call", float, 1.70)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0)
    points=fget("points", float, 241); step=0.25

    S_T, pl, _ = build_grid_and_rows_conversion(S0, K, prem_put, prem_call, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Spot_PnL(JPY)", "LongPut_PnL(JPY)", "ShortCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['spot'][i]:.6f}", f"{pl['long_put'][i]:.6f}", f"{pl['short_call'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="conversion_pnl.csv")


# ========================= Reversal ============================
# Short Spot + Short Put@K + Long Call@K
def payoff_components_reversal(S_T, S0, K, prem_put, prem_call, qty):
    short_spot_pl = (S0 - S_T) * qty
    short_put_pl  = ( prem_put  - np.maximum(K - S_T, 0.0)) * qty
    long_call_pl  = (-prem_call + np.maximum(S_T - K, 0.0)) * qty
    combo_pl      = short_spot_pl + short_put_pl + long_call_pl
    return {"short_spot": short_spot_pl, "short_put": short_put_pl, "long_call": long_call_pl, "combo": combo_pl}

def build_grid_and_rows_reversal(S0, K, prem_put, prem_call, qty,
                                 smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl  = payoff_components_reversal(S_T, S0, K, prem_put, prem_call, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_spot": float(pl["short_spot"][i]),
        "short_put":  float(pl["short_put"][i]),
        "long_call":  float(pl["long_call"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_reversal(S_T, pl, S0, K, flat_pl):
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["short_spot"], label="Short Spot P/L", color="black")
    ax.plot(S_T, pl["short_put"],  label="Short Put P/L",  color="red")
    ax.plot(S_T, pl["long_call"],  label="Long Call P/L",  color="blue")
    ax.plot(S_T, pl["combo"],      label="Combo (Reversal)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["short_spot"], pl["short_put"], pl["long_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K, f"K={K:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    ax.annotate(f"flat ≈ {flat_pl/1e6:.2f}M JPY", xy=(S_T[len(S_T)//2], np.median(pl["combo"])),
                xytext=(6, -10), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Reversal: Short Spot + Short Put(K) + Long Call(K)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_reversal(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    if be_list:
        for i, be in enumerate(be_list, start=1):
            ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
            ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    else:
        ax.text(S_T[len(S_T)//2], y_top, "No break-even (flat line)", va="top", ha="center", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Reversal: Break-even Focus")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

@app.route("/fx/reversal", methods=["GET", "POST"])
def fx_reversal():
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000, smin=135.0, smax=165.0, points=241,
        months=1.0
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K=fget("K"); vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); smin=fget("smin"); smax=fget("smax"); points=int(fget("points")); months=fget("months")
    else:
        S0=defaults["S0"]; K=defaults["K"]; vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]; months=defaults["months"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma = max(vol, 0.0)/100.0
    prem_put  = garman_kohlhagen_put (S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # Short
    prem_call = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T)  # Long

    # 定数（理論上フラット）： qty * ( (S0 - K) + (put - call) ) = - Conversion の値
    flat_pl = ((S0 - K) + (prem_put - prem_call)) * qty

    # Grid
    S_T, pl, rows = build_grid_and_rows_reversal(S0, K, prem_put, prem_call, qty, smin, smax, points)

    # 参考
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    prem_put_jpy = prem_put * qty; prem_call_jpy = prem_call * qty; premium_net = prem_put - prem_call
    premium_net_jpy = prem_put_jpy - prem_call_jpy

    # BE（平坦なので通常なし）
    be_list = _be_from_constant_line(S_T, pl["combo"])

    # 図
    fig = draw_chart_reversal(S_T, pl, S0, K, flat_pl)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_reversal(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_reversal.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months,
        # 出力
        prem_put=prem_put, prem_call=prem_call,
        prem_put_jpy=prem_put_jpy, prem_call_jpy=prem_call_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        flat_pl=flat_pl,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        be_vals=be_list,
        rows=rows
    )


# CSV（Reversal）
@app.route("/fx/download_csv_reversal", methods=["POST"])
def fx_download_csv_reversal():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    S0=fget("S0", float, 150.0); K=fget("K", float, 150.0)
    prem_put=fget("prem_put", float, 1.70); prem_call=fget("prem_call", float, 1.70)
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0)
    points=fget("points", float, 241); step=0.25

    S_T, pl, _ = build_grid_and_rows_reversal(S0, K, prem_put, prem_call, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortSpot_PnL(JPY)", "ShortPut_PnL(JPY)", "LongCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_spot'][i]:.6f}", f"{pl['short_put'][i]:.6f}", f"{pl['long_call'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="reversal_pnl.csv")


# ========================= Jelly Roll (Time-Box) =========================
# 同一行使 K、異なる満期 T1 < T2 で
#   Jelly Roll ＝ ( C(K,T1) − P(K,T1) ) − ( C(K,T2) − P(K,T2) )
# ＝ Synthetic Fwd(T1) − Synthetic Fwd(T2)
# 価値は理論上 S に依存せず、ボラにも依存しない（パリティ）。

def payoff_components_jelly_roll(S_T, val_syn_T1_jpy, val_short_syn_T2_jpy):
    """
    配列で描画用の水平線データを返す。
      val_syn_T1_jpy      = (C1 - P1) * qty   （JPY）
      val_short_syn_T2_jpy= -(C2 - P2) * qty  （JPY）
    """
    syn_T1 = np.full_like(S_T, float(val_syn_T1_jpy))
    short_T2 = np.full_like(S_T, float(val_short_syn_T2_jpy))
    combo = syn_T1 + short_T2
    return {"syn_T1": syn_T1, "short_syn_T2": short_T2, "combo": combo}

def build_grid_and_rows_jelly_roll(K, prem_c1, prem_p1, prem_c2, prem_p2, qty,
                                   smin, smax, points, *, step: float = 0.25):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))

    # JPY 金額（水平）
    val_syn_T1_jpy = (prem_c1 - prem_p1) * qty
    val_short_syn_T2_jpy = -(prem_c2 - prem_p2) * qty

    pl = payoff_components_jelly_roll(S_T, val_syn_T1_jpy, val_short_syn_T2_jpy)
    rows = [{
        "st": float(S_T[i]),
        "syn_T1": float(pl["syn_T1"][i]),
        "short_syn_T2": float(pl["short_syn_T2"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows, val_syn_T1_jpy, val_short_syn_T2_jpy

def _be_from_constant_line_jr(S_T, y, eps=1e-6):
    """
    Jelly Roll は通常フラットのため BE なし。ゼロに極めて近い場合のみ見栄え用に中央点を返す。
    """
    if len(y) == 0: return []
    yy = float(np.median(y))
    if abs(yy) < eps:
        return [float(S_T[len(S_T)//2])]
    return []

def draw_chart_jelly_roll(S_T, pl, S0, K, T1m, T2m, flat_pl):
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["syn_T1"],       label=f"Synthetic Fwd @T1({T1m:.1f}m)", color="blue")
    ax.plot(S_T, pl["short_syn_T2"], label=f"− Synthetic Fwd @T2({T2m:.1f}m)", color="red")
    ax.plot(S_T, pl["combo"],        label="Combo (Jelly Roll)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["syn_T1"], pl["short_syn_T2"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    # 参考縦線
    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K, f"K={K:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1)
        ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    ax.annotate(f"flat ≈ {flat_pl/1e6:.2f}M JPY",
                xy=(S_T[len(S_T)//2], np.median(pl["combo"])),
                xytext=(6, -10), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Jelly Roll (Time-Box): (C−P)@T1 − (C−P)@T2")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_be_focus_jelly_roll(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    if be_list:
        for i, be in enumerate(be_list, start=1):
            ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
            ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    else:
        ax.text(S_T[len(S_T)//2], y_top, "No break-even (flat)", va="top", ha="center", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Jelly Roll: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# ---------- 画面ルート ----------
@app.route("/fx/jelly-roll", methods=["GET", "POST"])
def fx_jelly_roll():
    defaults = dict(
        S0=150.0, K=150.0,
        vol=10.0,                 # ※理論値はボラ非依存（パリティ）が、表示は他と統一
        r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        months1=1.0,              # T1（短期）
        months2=3.0,              # T2（長期）
        smin=135.0, smax=165.0, points=241
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K=fget("K"); vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months1=fget("months1"); months2=fget("months2")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        S0=defaults["S0"]; K=defaults["K"]; vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months1=defaults["months1"]; months2=defaults["months2"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    # T1 < T2 に揃える
    if months1 > months2:
        months1, months2 = months2, months1

    points = clamp_points(points)
    T1 = max(months1, 0.0001)/12.0
    T2 = max(months2, 0.0001)/12.0
    sigma = max(vol, 0.0)/100.0

    # GKで C-P（JPY/USD）
    prem_c1 = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T1)
    prem_p1 = garman_kohlhagen_put (S0, K, r_dom/100.0, r_for/100.0, sigma, T1)
    prem_c2 = garman_kohlhagen_call(S0, K, r_dom/100.0, r_for/100.0, sigma, T2)
    prem_p2 = garman_kohlhagen_put (S0, K, r_dom/100.0, r_for/100.0, sigma, T2)

    # 値（JPY/USD）と JPY 金額
    syn_T1 = (prem_c1 - prem_p1)                     # per USD
    syn_T2 = (prem_c2 - prem_p2)                     # per USD
    syn_T1_jpy = syn_T1 * qty
    short_syn_T2_jpy = -syn_T2 * qty
    flat_pl = syn_T1_jpy + short_syn_T2_jpy          # 合成はフラット

    # パリティ確認（参考）：C-P = DF_for*S0 − DF_dom*K
    df_dom1 = np.exp(- (r_dom/100.0) * T1)
    df_for1 = np.exp(- (r_for/100.0) * T1)
    df_dom2 = np.exp(- (r_dom/100.0) * T2)
    df_for2 = np.exp(- (r_for/100.0) * T2)
    parity1 = df_for1 * S0 - df_dom1 * K
    parity2 = df_for2 * S0 - df_dom2 * K
    parity_diff_jpy = (parity1 - parity2) * qty  # ≈ flat_pl と一致

    # グリッド
    S_T, pl, rows, val_syn_T1_jpy, val_short_syn_T2_jpy = build_grid_and_rows_jelly_roll(
        K, prem_c1, prem_p1, prem_c2, prem_p2, qty, smin, smax, points
    )

    # 損益分岐点（通常なし）
    be_list = _be_from_constant_line_jr(S_T, pl["combo"])

    # 図① 全体
    fig = draw_chart_jelly_roll(S_T, pl, S0, K, months1, months2, flat_pl)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図② BEフォーカス
    fig2 = draw_be_focus_jelly_roll(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_jelly_roll.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        months1=months1, months2=months2,
        smin=smin, smax=smax, points=points,
        # 出力
        prem_c1=prem_c1, prem_p1=prem_p1, prem_c2=prem_c2, prem_p2=prem_p2,
        syn_T1=syn_T1, syn_T2=syn_T2,
        syn_T1_jpy=syn_T1_jpy, short_syn_T2_jpy=short_syn_T2_jpy,
        flat_pl=flat_pl,
        parity_diff_jpy=parity_diff_jpy,
        be_vals=be_list,
        rows=rows
    )


# ---------------- CSV（Jelly Roll） ----------------
@app.route("/fx/download_csv_jelly_roll", methods=["POST"])
def fx_download_csv_jelly_roll():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default

    K         = fget("K", float, 150.0)
    prem_c1   = fget("prem_c1", float, 1.70)
    prem_p1   = fget("prem_p1", float, 1.70)
    prem_c2   = fget("prem_c2", float, 1.20)
    prem_p2   = fget("prem_p2", float, 1.20)
    qty       = fget("qty", float, 1_000_000.0)
    smin      = fget("smin", float, 135.0)
    smax      = fget("smax", float, 165.0)
    points    = fget("points", float, 241)
    step      = 0.25

    S_T, pl, _rows, _v1, _v2 = build_grid_and_rows_jelly_roll(
        K, prem_c1, prem_p1, prem_c2, prem_p2, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "SynFwd_T1(JPY)", "Short_SynFwd_T2(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}",
                    f"{pl['syn_T1'][i]:.6f}",
                    f"{pl['short_syn_T2'][i]:.6f}",
                    f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="jelly_roll_pnl.csv")

# ======================= FX Option Calendar Spread =======================
# 依存：np, plt, io, base64, request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put

def _price_far_at_T1_vectorized(S_vals, opt_type, K_far, r_dom, r_for, sigma_far, T_rem):
    """
    近月満期T1時点での遠月オプション価値（残存T_rem）をSごとに評価。
    GK関数がスカラ前提でも動くようvectorize。
    返り値は np.array（JPY/USD）。
    """
    S_vals = np.asarray(S_vals, dtype=float)
    out = np.empty_like(S_vals, dtype=float)
    if opt_type == "call":
        for i, s in enumerate(S_vals):
            out[i] = garman_kohlhagen_call(float(s), K_far, r_dom, r_for, sigma_far, T_rem)
    else:
        for i, s in enumerate(S_vals):
            out[i] = garman_kohlhagen_put(float(s), K_far, r_dom, r_for, sigma_far, T_rem)
    return out

def payoff_components_calendar_spread(
    S_T, opt_type, S0, K_near, K_far,
    prem_near, prem_far,
    r_dom, r_for, sigma_far, T_rem, qty
):
    """
    近月ショート + 遠月ロングのカレンダー・スプレッドをT1で評価。
    - near_short_pl = ( +prem_near  - intrinsic_near_T1(S_T) ) * qty
    - far_long_pl  = ( -prem_far    + value_far_T1(S_T, T_rem) ) * qty
    combo = near_short_pl + far_long_pl
    prem_* は JPY/USD。qty はUSD名目。
    """
    if opt_type == "call":
        intrinsic_near = np.maximum(S_T - K_near, 0.0)
    else:
        intrinsic_near = np.maximum(K_near - S_T, 0.0)

    far_val_T1 = _price_far_at_T1_vectorized(S_T, opt_type, K_far, r_dom, r_for, sigma_far, T_rem)

    near_short_pl = (prem_near - intrinsic_near) * qty
    far_long_pl   = (-prem_far + far_val_T1) * qty
    combo_pl      = near_short_pl + far_long_pl

    return {
        "near_short": near_short_pl,
        "far_long":   far_long_pl,
        "combo":      combo_pl,
        "far_val_T1": far_val_T1,        # 参考：JPY/USD
        "intrinsic":  intrinsic_near,    # 参考：JPY/USD
    }

def build_grid_and_rows_calendar_spread(
    opt_type, S0, K_near, K_far, prem_near, prem_far, r_dom, r_for, sigma_far, T_rem,
    qty, smin, smax, points, *, step: float = 0.25
):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_calendar_spread(
        S_T, opt_type, S0, K_near, K_far, prem_near, prem_far, r_dom, r_for, sigma_far, T_rem, qty
    )
    rows = [{
        "st": float(S_T[i]),
        "near_short": float(pl["near_short"][i]),
        "far_long":   float(pl["far_long"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def _find_break_evens(S_T, y):
    bes = []
    for i in range(len(S_T)-1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            x0, x1 = S_T[i], S_T[i+1]
            x = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            bes.append(x)
    # 近接重複の整理
    res = []
    for v in sorted(bes):
        if not res or abs(v - res[-1]) > 1e-6:
            res.append(v)
    return res

def draw_chart_calendar_spread(S_T, pl, S0, K_near, K_far, be_list, opt_type):
    fig = plt.figure(figsize=(7.5, 4.8), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["near_short"], label="Short (Near) P/L", color="red")
    ax.plot(S_T, pl["far_long"],   label="Long (Far) MtM P/L", color="blue")
    ax.plot(S_T, pl["combo"],      label="Combo (Calendar) P/L", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["near_short"], pl["far_long"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K_near, f"K_near={K_near:.1f}", ":"), (K_far, f"K_far={K_far:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at T1 (USD/JPY)")
    ax.set_ylabel("P/L at T1 (JPY)")
    ax.set_title(f"Calendar Spread ({opt_type.title()}): Short Near + Long Far (T1 MtM)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_calendar_spread(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L @T1")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Spot at T1 (USD/JPY)")
    ax.set_ylabel("P/L at T1 (JPY)")
    ax.set_title("Calendar Spread: Break-even Focus")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# ---------------- 画面ルート ----------------
@app.route("/fx/calendar-spread", methods=["GET", "POST"])
def fx_calendar_spread():
    """
    Calendar Spread：近月ショート＋遠月ロングを T1 で評価（同スト or ダイアゴナル）。
    opt_type: call/put 切替。
    """
    defaults = dict(
        opt_type="call",
        S0=150.0,
        K_near=150.0, K_far=150.0,   # 同値がデフォルト
        vol_near=10.0, vol_far=11.0, # 年率％（テナーごとに別ボラ）
        r_dom=1.6, r_for=4.2,        # 年率％
        qty=1_000_000,
        months_near=1.0, months_far=3.0,
        smin=135.0, smax=165.0, points=241
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        opt_type = request.form.get("opt_type", defaults["opt_type"]).strip().lower()
        if opt_type not in ("call","put"): opt_type = "call"
        S0=fget("S0"); K_near=fget("K_near"); K_far=fget("K_far")
        vol_near=fget("vol_near"); vol_far=fget("vol_far")
        r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months_near=fget("months_near"); months_far=fget("months_far")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        opt_type=defaults["opt_type"]
        S0=defaults["S0"]; K_near=defaults["K_near"]; K_far=defaults["K_far"]
        vol_near=defaults["vol_near"]; vol_far=defaults["vol_far"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months_near=defaults["months_near"]; months_far=defaults["months_far"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    # 期限の整列（far > near）
    if months_near > months_far:
        months_near, months_far = months_far, months_near
        K_near, K_far = K_far, K_near  # ダイアゴナルの入替にも対応
        vol_near, vol_far = vol_far, vol_near

    points = clamp_points(points)

    # パラメータ
    T_near = max(months_near, 0.0001) / 12.0
    T_far  = max(months_far , 0.0001) / 12.0
    T_rem  = max(T_far - T_near, 1e-6)                # 残存
    sigma_near = max(vol_near, 0.0) / 100.0
    sigma_far  = max(vol_far , 0.0) / 100.0
    rD = r_dom/100.0; rF = r_for/100.0

    # 近月/遠月の初期プレミアム（JPY/USD）
    if opt_type == "call":
        prem_near = garman_kohlhagen_call(S0, K_near, rD, rF, sigma_near, T_near)
        prem_far  = garman_kohlhagen_call(S0, K_far , rD, rF, sigma_far , T_far )
    else:
        prem_near = garman_kohlhagen_put (S0, K_near, rD, rF, sigma_near, T_near)
        prem_far  = garman_kohlhagen_put (S0, K_far , rD, rF, sigma_far , T_far )

    # 損益グリッド（T1評価）
    S_T, pl, rows = build_grid_and_rows_calendar_spread(
        opt_type, S0, K_near, K_far, prem_near, prem_far, rD, rF, sigma_far, T_rem,
        qty, smin, smax, points
    )

    # BE
    be_list = _find_break_evens(S_T, pl["combo"])
    be1 = be_list[0] if len(be_list) >= 1 else None
    be2 = be_list[1] if len(be_list) >= 2 else None

    # 参考（レンジ内）
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 金額系
    prem_near_jpy = prem_near * qty
    prem_far_jpy  = prem_far  * qty
    premium_net   = prem_far - prem_near          # per USD（通常 Long Calendar はデビット）
    premium_net_jpy = (prem_far_jpy - prem_near_jpy)

    # 図
    fig = draw_chart_calendar_spread(S_T, pl, S0, K_near, K_far, be_list, opt_type)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_calendar_spread(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_calendar_spread.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        opt_type=opt_type, S0=S0,
        K_near=K_near, K_far=K_far,
        vol_near=vol_near, vol_far=vol_far,
        r_dom=r_dom, r_for=r_for, qty=qty,
        months_near=months_near, months_far=months_far,
        smin=smin, smax=smax, points=points,
        # 出力
        prem_near=prem_near, prem_far=prem_far,
        prem_near_jpy=prem_near_jpy, prem_far_jpy=prem_far_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be1=be1, be2=be2, be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# ---------------- CSV（Calendar Spread） ----------------
@app.route("/fx/download_csv_calendar_spread", methods=["POST"])
def fx_download_csv_calendar_spread():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    opt_type = request.form.get("opt_type", "call").strip().lower()
    if opt_type not in ("call","put"): opt_type = "call"
    S0 = fget("S0", float, 150.0)
    K_near = fget("K_near", float, 150.0)
    K_far  = fget("K_far",  float, 150.0)
    prem_near = fget("prem_near", float, 1.50)
    prem_far  = fget("prem_far",  float, 1.80)
    rD = fget("rD", float, 0.016)
    rF = fget("rF", float, 0.042)
    sigma_far = fget("sigma_far", float, 0.11)
    T_rem = fget("T_rem", float, 0.1667)
    qty = fget("qty", float, 1_000_000.0)
    smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0); points=fget("points", float, 241)
    step=0.25

    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_calendar_spread(S_T, opt_type, S0, K_near, K_far, prem_near, prem_far, rD, rF, sigma_far, T_rem, qty)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "NearShort_PnL(JPY)", "FarLong_MtM_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}", f"{pl['near_short'][i]:.6f}", f"{pl['far_long'][i]:.6f}", f"{pl['combo'][i]:.6f}"
        ])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="calendar_spread_T1_pnl.csv")

# ======================= FX Option Double Calendar =======================
# 依存：np, plt, io, base64, request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put

def _price_far_at_T1_vec_call(S_vals, K_far, r_dom, r_for, sigma_far, T_rem):
    S_vals = np.asarray(S_vals, dtype=float)
    out = np.empty_like(S_vals, dtype=float)
    for i, s in enumerate(S_vals):
        out[i] = garman_kohlhagen_call(float(s), K_far, r_dom, r_for, sigma_far, T_rem)
    return out

def _price_far_at_T1_vec_put(S_vals, K_far, r_dom, r_for, sigma_far, T_rem):
    S_vals = np.asarray(S_vals, dtype=float)
    out = np.empty_like(S_vals, dtype=float)
    for i, s in enumerate(S_vals):
        out[i] = garman_kohlhagen_put(float(s), K_far, r_dom, r_for, sigma_far, T_rem)
    return out

def payoff_components_double_calendar(
    S_T,
    # params
    S0,
    Kc_near, Kc_far,
    Kp_near, Kp_far,
    prem_near_call, prem_far_call,
    prem_near_put,  prem_far_put,
    r_dom, r_for, sigma_far, T_rem,
    qty
):
    """
    Double Calendar：近月ショート（Call/Put）＋遠月ロング（Call/Put）を T1 で評価。
      - Near short call  : ( +prem_near_call - max(S - Kc_near, 0) ) * qty
      - Far  long call   : ( -prem_far_call  + value_call_T1(S, Kc_far, T_rem) ) * qty
      - Near short put   : ( +prem_near_put  - max(Kp_near - S, 0) ) * qty
      - Far  long put    : ( -prem_far_put   + value_put_T1(S,  Kp_far, T_rem) ) * qty
    """
    intrinsic_call = np.maximum(S_T - Kc_near, 0.0)
    intrinsic_put  = np.maximum(Kp_near - S_T, 0.0)

    far_call_T1 = _price_far_at_T1_vec_call(S_T, Kc_far, r_dom, r_for, sigma_far, T_rem)
    far_put_T1  = _price_far_at_T1_vec_put (S_T, Kp_far, r_dom, r_for, sigma_far, T_rem)

    near_short_call = (prem_near_call - intrinsic_call) * qty
    far_long_call   = (-prem_far_call + far_call_T1)   * qty
    call_cal_pl     = near_short_call + far_long_call

    near_short_put  = (prem_near_put  - intrinsic_put ) * qty
    far_long_put    = (-prem_far_put  + far_put_T1 )    * qty
    put_cal_pl      = near_short_put + far_long_put

    combo_pl = call_cal_pl + put_cal_pl

    return {
        "near_short_call": near_short_call,
        "far_long_call":   far_long_call,
        "call_cal":        call_cal_pl,
        "near_short_put":  near_short_put,
        "far_long_put":    far_long_put,
        "put_cal":         put_cal_pl,
        "combo":           combo_pl,
    }

def build_grid_and_rows_double_calendar(
    S0,
    Kc_near, Kc_far,
    Kp_near, Kp_far,
    prem_near_call, prem_far_call,
    prem_near_put,  prem_far_put,
    r_dom, r_for, sigma_far, T_rem,
    qty, smin, smax, points, *, step: float = 0.25
):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))

    pl = payoff_components_double_calendar(
        S_T,
        S0,
        Kc_near, Kc_far, Kp_near, Kp_far,
        prem_near_call, prem_far_call, prem_near_put, prem_far_put,
        r_dom, r_for, sigma_far, T_rem,
        qty
    )

    rows = [{
        "st": float(S_T[i]),
        "near_short_call": float(pl["near_short_call"][i]),
        "far_long_call":   float(pl["far_long_call"][i]),
        "near_short_put":  float(pl["near_short_put"][i]),
        "far_long_put":    float(pl["far_long_put"][i]),
        "call_cal":        float(pl["call_cal"][i]),
        "put_cal":         float(pl["put_cal"][i]),
        "combo":           float(pl["combo"][i]),
    } for i in range(len(S_T))]

    return S_T, pl, rows

def _find_break_evens_generic(S_T, y):
    bes = []
    for i in range(len(S_T)-1):
        y0, y1 = float(y[i]), float(y[i+1])
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            x0, x1 = float(S_T[i]), float(S_T[i+1])
            x = x0 + (x1 - x0) * (-y0) / (y1 - y0)
            bes.append(float(x))
    # 近接重複の整理
    res = []
    for v in sorted(bes):
        if not res or abs(v - res[-1]) > 1e-6:
            res.append(v)
    return res

def draw_chart_double_calendar(S_T, pl, S0, Kc_near, Kc_far, Kp_near, Kp_far, be_list):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["call_cal"], label="Call Calendar P/L", color="blue")
    ax.plot(S_T, pl["put_cal"],  label="Put  Calendar P/L", color="red")
    ax.plot(S_T, pl["combo"],    label="Combo (Double Calendar) P/L", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["call_cal"], pl["put_cal"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    # 参考線（多いので簡潔に）
    for x, txt, ls in [
        (S0,        f"S0={S0:.1f}",           "--"),
        (Kc_near,   f"KcN={Kc_near:.1f}",     ":"),
        (Kc_far,    f"KcF={Kc_far:.1f}",      ":"),
        (Kp_near,   f"KpN={Kp_near:.1f}",     ":"),
        (Kp_far,    f"KpF={Kp_far:.1f}",      ":"),
    ]:
        ax.axvline(x, linestyle=ls, linewidth=1)
        ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at T1 (USD/JPY)")
    ax.set_ylabel("P/L at T1 (JPY)")
    ax.set_title("Double Calendar: Short Near + Long Far (Calls & Puts)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_be_focus_double_calendar(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L @T1")
    _set_ylim_tight(ax, [combo_pl])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at T1 (USD/JPY)")
    ax.set_ylabel("P/L at T1 (JPY)")
    ax.set_title("Double Calendar: Break-even Focus")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

# ---------------- 画面ルート ----------------
@app.route("/fx/calendar-double", methods=["GET", "POST"])
def fx_calendar_double():
    """
    Double Calendar（Call＋Put）：近月ショート＋遠月ロングを T1 で評価。
    K は Call/Put それぞれ近月と遠月を別指定（＝ダブル・ダイアゴナルも可）。
    """
    defaults = dict(
        S0=150.0,
        Kc_near=150.0, Kc_far=150.0,
        Kp_near=150.0, Kp_far=150.0,
        vol_near=10.0, vol_far=11.0,   # 年率％（テナー別）
        r_dom=1.6, r_for=4.2,          # 年率％
        qty=1_000_000,
        months_near=1.0, months_far=3.0,
        smin=135.0, smax=165.0, points=241,
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0")
        Kc_near=fget("Kc_near"); Kc_far=fget("Kc_far")
        Kp_near=fget("Kp_near"); Kp_far=fget("Kp_far")
        vol_near=fget("vol_near"); vol_far=fget("vol_far")
        r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty")
        months_near=fget("months_near"); months_far=fget("months_far")
        smin=fget("smin"); smax=fget("smax")
        points=int(fget("points"))
    else:
        S0=defaults["S0"]
        Kc_near=defaults["Kc_near"]; Kc_far=defaults["Kc_far"]
        Kp_near=defaults["Kp_near"]; Kp_far=defaults["Kp_far"]
        vol_near=defaults["vol_near"]; vol_far=defaults["vol_far"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]
        months_near=defaults["months_near"]; months_far=defaults["months_far"]
        smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]

    # 期限の整列（far > near）
    if months_near > months_far:
        months_near, months_far = months_far, months_near
        # 近遠を入替（ダイアゴナル整合）
        Kc_near, Kc_far = Kc_far, Kc_near
        Kp_near, Kp_far = Kp_far, Kp_near

    points = clamp_points(points)

    # パラメータ
    T_near = max(months_near, 0.0001) / 12.0
    T_far  = max(months_far , 0.0001) / 12.0
    T_rem  = max(T_far - T_near, 1e-6)
    sigma_near = max(vol_near, 0.0) / 100.0
    sigma_far  = max(vol_far , 0.0) / 100.0
    rD = r_dom/100.0; rF = r_for/100.0

    # 初期プレミアム（JPY/USD）
    prem_near_call = garman_kohlhagen_call(S0, Kc_near, rD, rF, sigma_near, T_near)
    prem_far_call  = garman_kohlhagen_call(S0, Kc_far,  rD, rF, sigma_far,  T_far )
    prem_near_put  = garman_kohlhagen_put (S0, Kp_near, rD, rF, sigma_near, T_near)
    prem_far_put   = garman_kohlhagen_put (S0, Kp_far,  rD, rF, sigma_far,  T_far )

    # 損益グリッド（T1評価）
    S_T, pl, rows = build_grid_and_rows_double_calendar(
        S0,
        Kc_near, Kc_far, Kp_near, Kp_far,
        prem_near_call, prem_far_call, prem_near_put, prem_far_put,
        rD, rF, sigma_far, T_rem,
        qty, smin, smax, points
    )

    # BE
    be_list = _find_break_evens_generic(S_T, pl["combo"])
    # 表示用（最大3本）
    be1 = be_list[0] if len(be_list) > 0 else None
    be2 = be_list[1] if len(be_list) > 1 else None
    be3 = be_list[2] if len(be_list) > 2 else None

    # レンジ内参考
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 金額系
    prem_near_call_jpy = prem_near_call * qty
    prem_far_call_jpy  = prem_far_call  * qty
    prem_near_put_jpy  = prem_near_put  * qty
    prem_far_put_jpy   = prem_far_put   * qty
    premium_net_jpy = (prem_far_call_jpy + prem_far_put_jpy) - (prem_near_call_jpy + prem_near_put_jpy)
    premium_net = (prem_far_call + prem_far_put) - (prem_near_call + prem_near_put)

    # 図
    fig = draw_chart_double_calendar(S_T, pl, S0, Kc_near, Kc_far, Kp_near, Kp_far, be_list)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_double_calendar(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_calendar_double.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0,
        Kc_near=Kc_near, Kc_far=Kc_far, Kp_near=Kp_near, Kp_far=Kp_far,
        vol_near=vol_near, vol_far=vol_far,
        r_dom=r_dom, r_for=r_for, qty=qty,
        months_near=months_near, months_far=months_far,
        smin=smin, smax=smax, points=points,
        # 出力
        prem_near_call=prem_near_call, prem_far_call=prem_far_call,
        prem_near_put=prem_near_put,   prem_far_put=prem_far_put,
        prem_near_call_jpy=prem_near_call_jpy, prem_far_call_jpy=prem_far_call_jpy,
        prem_near_put_jpy=prem_near_put_jpy,   prem_far_put_jpy=prem_far_put_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be1=be1, be2=be2, be3=be3, be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# ---------------- CSV（Double Calendar） ----------------
@app.route("/fx/download_csv_calendar_double", methods=["POST"])
def fx_download_csv_calendar_double():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default

    Kc_near = fget("Kc_near", float, 150.0)
    Kc_far  = fget("Kc_far",  float, 150.0)
    Kp_near = fget("Kp_near", float, 150.0)
    Kp_far  = fget("Kp_far",  float, 150.0)

    prem_near_call = fget("prem_near_call", float, 1.50)
    prem_far_call  = fget("prem_far_call",  float, 1.80)
    prem_near_put  = fget("prem_near_put",  float, 1.50)
    prem_far_put   = fget("prem_far_put",   float, 1.80)

    rD = fget("rD", float, 0.016)
    rF = fget("rF", float, 0.042)
    sigma_far = fget("sigma_far", float, 0.11)
    T_rem = fget("T_rem", float, 0.1667)

    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 135.0)
    smax = fget("smax", float, 165.0)
    points = fget("points", float, 241)
    step = 0.25

    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_double_calendar(
        S_T,
        S0=0.0,  # 未使用
        Kc_near=Kc_near, Kc_far=Kc_far,
        Kp_near=Kp_near, Kp_far=Kp_far,
        prem_near_call=prem_near_call, prem_far_call=prem_far_call,
        prem_near_put=prem_near_put,   prem_far_put=prem_far_put,
        r_dom=rD, r_for=rF, sigma_far=sigma_far, T_rem=T_rem,
        qty=qty
    )

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)",
                "NearShortCall_PnL(JPY)", "FarLongCall_PnL(JPY)", "CallCalendar_PnL(JPY)",
                "NearShortPut_PnL(JPY)",  "FarLongPut_PnL(JPY)",  "PutCalendar_PnL(JPY)",
                "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['near_short_call'][i]:.6f}", f"{pl['far_long_call'][i]:.6f}", f"{pl['call_cal'][i]:.6f}",
            f"{pl['near_short_put'][i]:.6f}",  f"{pl['far_long_put'][i]:.6f}",  f"{pl['put_cal'][i]:.6f}",
            f"{pl['combo'][i]:.6f}"
        ])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="double_calendar_T1_pnl.csv")

# ======================= FX Option Double Diagonal =======================
# 依存：np, plt, io, base64, request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put
# 補助関数：_price_far_at_T1_vec_call, _price_far_at_T1_vec_put, _find_break_evens_generic
#   （↑ Double Calendar 実装で定義済みのものを使用）

def payoff_components_double_diagonal(
    S_T,
    S0,
    Kc_near, Kc_far,
    Kp_near, Kp_far,
    prem_near_call, prem_far_call,
    prem_near_put,  prem_far_put,
    r_dom, r_for, sigma_far, T_rem,
    qty
):
    """
    Double Diagonal：近月ショート（Call/Put, Kc_near/Kp_near）＋遠月ロング（Call/Put, Kc_far/Kp_far）を
    T1 で評価（近月はインストリンシック、遠月は残存 T_rem のGK理論価値）。
    """
    intrinsic_call = np.maximum(S_T - Kc_near, 0.0)
    intrinsic_put  = np.maximum(Kp_near - S_T, 0.0)

    far_call_T1 = _price_far_at_T1_vec_call(S_T, Kc_far, r_dom, r_for, sigma_far, T_rem)
    far_put_T1  = _price_far_at_T1_vec_put (S_T, Kp_far, r_dom, r_for, sigma_far, T_rem)

    near_short_call = (prem_near_call - intrinsic_call) * qty
    far_long_call   = (-prem_far_call + far_call_T1)   * qty
    call_diag_pl    = near_short_call + far_long_call

    near_short_put  = (prem_near_put  - intrinsic_put ) * qty
    far_long_put    = (-prem_far_put  + far_put_T1 )    * qty
    put_diag_pl     = near_short_put + far_long_put

    combo_pl = call_diag_pl + put_diag_pl

    return {
        "near_short_call": near_short_call,
        "far_long_call":   far_long_call,
        "call_diag":       call_diag_pl,
        "near_short_put":  near_short_put,
        "far_long_put":    far_long_put,
        "put_diag":        put_diag_pl,
        "combo":           combo_pl,
    }

def build_grid_and_rows_double_diagonal(
    S0,
    Kc_near, Kc_far,
    Kp_near, Kp_far,
    prem_near_call, prem_far_call,
    prem_near_put,  prem_far_put,
    r_dom, r_for, sigma_far, T_rem,
    qty, smin, smax, points, *, step: float = 0.25
):
    if smin > smax:
        smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))

    pl = payoff_components_double_diagonal(
        S_T,
        S0,
        Kc_near, Kc_far, Kp_near, Kp_far,
        prem_near_call, prem_far_call, prem_near_put, prem_far_put,
        r_dom, r_for, sigma_far, T_rem,
        qty
    )

    rows = [{
        "st": float(S_T[i]),
        "near_short_call": float(pl["near_short_call"][i]),
        "far_long_call":   float(pl["far_long_call"][i]),
        "call_diag":       float(pl["call_diag"][i]),
        "near_short_put":  float(pl["near_short_put"][i]),
        "far_long_put":    float(pl["far_long_put"][i]),
        "put_diag":        float(pl["put_diag"][i]),
        "combo":           float(pl["combo"][i]),
    } for i in range(len(S_T))]

    return S_T, pl, rows

def draw_chart_double_diagonal(S_T, pl, S0, Kc_near, Kc_far, Kp_near, Kp_far, be_list):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120)
    ax = fig.add_subplot(111)

    ax.plot(S_T, pl["call_diag"], label="Call Diagonal P/L", color="blue")
    ax.plot(S_T, pl["put_diag"],  label="Put  Diagonal P/L", color="red")
    ax.plot(S_T, pl["combo"],     label="Combo (Double Diagonal) P/L", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["call_diag"], pl["put_diag"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [
        (S0,        f"S0={S0:.1f}",       "--"),
        (Kc_near,   f"KcN={Kc_near:.1f}", ":"),
        (Kc_far,    f"KcF={Kc_far:.1f}",  ":"),
        (Kp_near,   f"KpN={Kp_near:.1f}", ":"),
        (Kp_far,    f"KpF={Kp_far:.1f}",  ":"),
    ]:
        ax.axvline(x, linestyle=ls, linewidth=1)
        ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at T1 (USD/JPY)")
    ax.set_ylabel("P/L at T1 (JPY)")
    ax.set_title("Double Diagonal: Short Near + Long Far (Calls & Puts, K遠近ズラし)")
    _format_y_as_m(ax)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_be_focus_double_diagonal(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L @T1")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at T1 (USD/JPY)")
    ax.set_ylabel("P/L at T1 (JPY)")
    ax.set_title("Double Diagonal: Break-even Focus")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# ---------------- 画面ルート ----------------
@app.route("/fx/double-diagonal", methods=["GET", "POST"])
def fx_double_diagonal():
    """
    Double Diagonal：近月ショート＋遠月ロング（Call/Put 両サイド）、近遠で K をずらす構成。
    T1（近月）で評価（近月＝インストリンシック、遠月＝残存の理論価値）。
    """
    defaults = dict(
        S0=150.0,
        Kc_near=151.0, Kc_far=153.0,   # 例：Call は遠月をよりOTMへ
        Kp_near=149.0, Kp_far=147.0,   # 例：Put  も遠月をよりOTMへ
        vol_near=10.0, vol_far=11.5,   # 年率％（テナー別）
        r_dom=1.6, r_for=4.2,          # 年率％
        qty=1_000_000,
        months_near=1.0, months_far=3.0,
        smin=135.0, smax=165.0, points=241,
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0")
        Kc_near=fget("Kc_near"); Kc_far=fget("Kc_far")
        Kp_near=fget("Kp_near"); Kp_far=fget("Kp_far")
        vol_near=fget("vol_near"); vol_far=fget("vol_far")
        r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty")
        months_near=fget("months_near"); months_far=fget("months_far")
        smin=fget("smin"); smax=fget("smax")
        points=int(fget("points"))
    else:
        S0=defaults["S0"]
        Kc_near=defaults["Kc_near"]; Kc_far=defaults["Kc_far"]
        Kp_near=defaults["Kp_near"]; Kp_far=defaults["Kp_far"]
        vol_near=defaults["vol_near"]; vol_far=defaults["vol_far"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]
        months_near=defaults["months_near"]; months_far=defaults["months_far"]
        smin=defaults["smin"]; smax=defaults["smax"]
        points=defaults["points"]

    # 期限の整列（far > near）
    if months_near > months_far:
        months_near, months_far = months_far, months_near
        # 近遠を入替（整合）
        Kc_near, Kc_far = Kc_far, Kc_near
        Kp_near, Kp_far = Kp_far, Kp_near

    points = clamp_points(points)

    # パラメータ
    T_near = max(months_near, 0.0001) / 12.0
    T_far  = max(months_far , 0.0001) / 12.0
    T_rem  = max(T_far - T_near, 1e-6)
    sigma_near = max(vol_near, 0.0) / 100.0
    sigma_far  = max(vol_far , 0.0) / 100.0
    rD = r_dom/100.0; rF = r_for/100.0

    # 初期プレミアム（JPY/USD）
    prem_near_call = garman_kohlhagen_call(S0, Kc_near, rD, rF, sigma_near, T_near)
    prem_far_call  = garman_kohlhagen_call(S0, Kc_far,  rD, rF, sigma_far,  T_far )
    prem_near_put  = garman_kohlhagen_put (S0, Kp_near, rD, rF, sigma_near, T_near)
    prem_far_put   = garman_kohlhagen_put (S0, Kp_far,  rD, rF, sigma_far,  T_far )

    # 損益グリッド（T1評価）
    S_T, pl, rows = build_grid_and_rows_double_diagonal(
        S0,
        Kc_near, Kc_far, Kp_near, Kp_far,
        prem_near_call, prem_far_call, prem_near_put, prem_far_put,
        rD, rF, sigma_far, T_rem,
        qty, smin, smax, points
    )

    # BE
    be_list = _find_break_evens_generic(S_T, pl["combo"])
    be1 = be_list[0] if len(be_list) > 0 else None
    be2 = be_list[1] if len(be_list) > 1 else None
    be3 = be_list[2] if len(be_list) > 2 else None

    # レンジ内参考
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 金額系
    prem_near_call_jpy = prem_near_call * qty
    prem_far_call_jpy  = prem_far_call  * qty
    prem_near_put_jpy  = prem_near_put  * qty
    prem_far_put_jpy   = prem_far_put   * qty
    premium_net_jpy = (prem_far_call_jpy + prem_far_put_jpy) - (prem_near_call_jpy + prem_near_put_jpy)
    premium_net = (prem_far_call + prem_far_put) - (prem_near_call + prem_near_put)

    # 図
    fig = draw_chart_double_diagonal(S_T, pl, S0, Kc_near, Kc_far, Kp_near, Kp_far, be_list)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_double_diagonal(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_double_diagonal.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0,
        Kc_near=Kc_near, Kc_far=Kc_far, Kp_near=Kp_near, Kp_far=Kp_far,
        vol_near=vol_near, vol_far=vol_far,
        r_dom=r_dom, r_for=r_for, qty=qty,
        months_near=months_near, months_far=months_far,
        smin=smin, smax=smax, points=points,
        # 出力
        prem_near_call=prem_near_call, prem_far_call=prem_far_call,
        prem_near_put=prem_near_put,   prem_far_put=prem_far_put,
        prem_near_call_jpy=prem_near_call_jpy, prem_far_call_jpy=prem_far_call_jpy,
        prem_near_put_jpy=prem_near_put_jpy,   prem_far_put_jpy=prem_far_put_jpy,
        premium_net=premium_net, premium_net_jpy=premium_net_jpy,
        be1=be1, be2=be2, be3=be3, be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# ---------------- CSV（Double Diagonal） ----------------
@app.route("/fx/download_csv_double_diagonal", methods=["POST"])
def fx_download_csv_double_diagonal():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default

    Kc_near = fget("Kc_near", float, 151.0)
    Kc_far  = fget("Kc_far",  float, 153.0)
    Kp_near = fget("Kp_near", float, 149.0)
    Kp_far  = fget("Kp_far",  float, 147.0)

    prem_near_call = fget("prem_near_call", float, 1.55)
    prem_far_call  = fget("prem_far_call",  float, 1.95)
    prem_near_put  = fget("prem_near_put",  float, 1.50)
    prem_far_put   = fget("prem_far_put",   float, 1.90)

    rD = fget("rD", float, 0.016)
    rF = fget("rF", float, 0.042)
    sigma_far = fget("sigma_far", float, 0.115)
    T_rem = fget("T_rem", float, 0.1667)

    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 135.0)
    smax = fget("smax", float, 165.0)
    points = fget("points", float, 241)
    step = 0.25

    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_double_diagonal(
        S_T,
        S0=0.0,  # 未使用
        Kc_near=Kc_near, Kc_far=Kc_far,
        Kp_near=Kp_near, Kp_far=Kp_far,
        prem_near_call=prem_near_call, prem_far_call=prem_far_call,
        prem_near_put=prem_near_put,   prem_far_put=prem_far_put,
        r_dom=rD, r_for=rF, sigma_far=sigma_far, T_rem=T_rem,
        qty=qty
    )

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)",
                "NearShortCall_PnL(JPY)", "FarLongCall_PnL(JPY)", "CallDiagonal_PnL(JPY)",
                "NearShortPut_PnL(JPY)",  "FarLongPut_PnL(JPY)",  "PutDiagonal_PnL(JPY)",
                "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([
            f"{S_T[i]:.6f}",
            f"{pl['near_short_call'][i]:.6f}", f"{pl['far_long_call'][i]:.6f}", f"{pl['call_diag'][i]:.6f}",
            f"{pl['near_short_put'][i]:.6f}",  f"{pl['far_long_put'][i]:.6f}",  f"{pl['put_diag'][i]:.6f}",
            f"{pl['combo'][i]:.6f}"
        ])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="double_diagonal_T1_pnl.csv")

# ======================= Ratio Butterfly / Condor =======================
# 依存：np, plt, io, base64, request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put

# 汎用BE検出（ゼロクロスを線形補間）
def _find_break_evens_generic(S_T, y):
    y = np.asarray(y, dtype=float)
    S_T = np.asarray(S_T, dtype=float)
    bes = []
    for i in range(len(S_T) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            x0, x1 = S_T[i], S_T[i+1]
            x = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            bes.append(x)
    # 近接重複の整理
    res = []
    for v in sorted(bes):
        if not res or abs(v - res[-1]) > 1e-6:
            res.append(v)
    return res

# ---------------- Ratio Butterfly ----------------
def payoff_components_ratio_bfly(
    S_T, opt_type,
    K1, K2, K3,
    prem1, prem2, prem3,
    m_long1, m_short2, m_long3,
    qty
):
    """レシオ・バタフライ：Long@K1 ×m1、Short@K2 ×m2、Long@K3 ×m3"""
    if opt_type == "call":
        p1 = np.maximum(S_T - K1, 0.0)
        p2 = np.maximum(S_T - K2, 0.0)
        p3 = np.maximum(S_T - K3, 0.0)
    else:
        p1 = np.maximum(K1 - S_T, 0.0)
        p2 = np.maximum(K2 - S_T, 0.0)
        p3 = np.maximum(K3 - S_T, 0.0)

    leg1 = m_long1  * (-prem1 + p1) * qty
    leg2 = m_short2 * ( +prem2 - p2) * qty
    leg3 = m_long3  * (-prem3 + p3) * qty
    combo = leg1 + leg2 + leg3
    return {"leg1": leg1, "leg2": leg2, "leg3": leg3, "combo": combo}

def build_grid_and_rows_ratio_bfly(
    opt_type, K1, K2, K3, prem1, prem2, prem3, m1, m2, m3,
    qty, smin, smax, points, *, step: float = 0.25
):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_ratio_bfly(S_T, opt_type, K1, K2, K3, prem1, prem2, prem3, m1, m2, m3, qty)
    rows = [{
        "st": float(S_T[i]),
        "leg1": float(pl["leg1"][i]),
        "leg2": float(pl["leg2"][i]),
        "leg3": float(pl["leg3"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_ratio_bfly(S_T, pl, S0, K1, K2, K3, be_list, opt_type, m1, m2, m3):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["leg1"], label=f"Long @{K1:.1f} ×{m1}", color="blue")
    ax.plot(S_T, pl["leg2"], label=f"Short @{K2:.1f} ×{m2}", color="orange")
    ax.plot(S_T, pl["leg3"], label=f"Long @{K3:.1f} ×{m3}", color="purple")
    ax.plot(S_T, pl["combo"], label="Combo (Ratio Butterfly)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["leg1"], pl["leg2"], pl["leg3"], pl["combo"]]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":"), (K3, f"K3={K3:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"Ratio Butterfly ({opt_type.title()})")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_ratio_bfly(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY"); ax.set_ylabel("P/L (JPY)"); ax.set_title("Break-even (Ratio Butterfly)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

@app.route("/fx/ratio-butterfly", methods=["GET", "POST"])
def fx_ratio_butterfly():
    defaults = dict(
        opt_type="call",
        S0=150.0,
        K1=148.0, K2=150.0, K3=152.0,
        m1=1, m2=2, m3=1,          # 1:-2:+1 が標準、m2 を可変に
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000, months=1.0,
        smin=135.0, smax=165.0, points=241
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        opt_type = request.form.get("opt_type", defaults["opt_type"]).strip().lower()
        if opt_type not in ("call","put"): opt_type="call"
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2"); K3=fget("K3")
        m1=int(fget("m1", int, defaults["m1"])); m2=int(fget("m2", int, defaults["m2"])); m3=int(fget("m3", int, defaults["m3"]))
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months=fget("months")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        opt_type=defaults["opt_type"]; S0=defaults["S0"]
        K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]
        m1=defaults["m1"]; m2=defaults["m2"]; m3=defaults["m3"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months=defaults["months"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma = max(vol,0.0)/100.0; rD=r_dom/100.0; rF=r_for/100.0

    if opt_type=="call":
        prem1 = garman_kohlhagen_call(S0,K1,rD,rF,sigma,T)
        prem2 = garman_kohlhagen_call(S0,K2,rD,rF,sigma,T)
        prem3 = garman_kohlhagen_call(S0,K3,rD,rF,sigma,T)
    else:
        prem1 = garman_kohlhagen_put (S0,K1,rD,rF,sigma,T)
        prem2 = garman_kohlhagen_put (S0,K2,rD,rF,sigma,T)
        prem3 = garman_kohlhagen_put (S0,K3,rD,rF,sigma,T)

    S_T, pl, rows = build_grid_and_rows_ratio_bfly(opt_type, K1, K2, K3, prem1, prem2, prem3, m1, m2, m3, qty, smin, smax, points)
    be_list = _find_break_evens_generic(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # プレミアム純額（JPY）
    premium_net_per_usd = (-m1*prem1 + m2*prem2 - m3*prem3)   # Long/Short の符号
    premium_net_jpy = premium_net_per_usd * qty

    # 図
    fig = draw_chart_ratio_bfly(S_T, pl, S0, K1, K2, K3, be_list, opt_type, m1, m2, m3)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_ratio_bfly(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_ratio_butterfly.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        opt_type=opt_type, S0=S0, K1=K1, K2=K2, K3=K3, m1=m1, m2=m2, m3=m3,
        vol=vol, r_dom=r_dom, r_for=r_for, qty=qty, months=months,
        smin=smin, smax=smax, points=points,
        # 出力
        premium_net_per_usd=premium_net_per_usd, premium_net_jpy=premium_net_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows,
        prem1=prem1, prem2=prem2, prem3=prem3
    )

# CSV（Ratio Butterfly）
@app.route("/fx/download_csv_ratio_butterfly", methods=["POST"])
def fx_download_csv_ratio_butterfly():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    opt_type = request.form.get("opt_type","call").strip().lower()
    if opt_type not in ("call","put"): opt_type="call"
    K1=fget("K1", float, 148.0); K2=fget("K2", float, 150.0); K3=fget("K3", float, 152.0)
    prem1=fget("prem1", float, 1.8); prem2=fget("prem2", float, 1.5); prem3=fget("prem3", float, 1.2)
    m1=int(fget("m1", int, 1)); m2=int(fget("m2", int, 2)); m3=int(fget("m3", int, 1))
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0); points=fget("points", float, 241)
    step=0.25
    S_T, pl, _ = build_grid_and_rows_ratio_bfly(opt_type, K1, K2, K3, prem1, prem2, prem3, m1, m2, m3, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Leg1_PnL(JPY)", "Leg2_PnL(JPY)", "Leg3_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['leg1'][i]:.6f}", f"{pl['leg2'][i]:.6f}", f"{pl['leg3'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="ratio_butterfly_pnl.csv")

# ---------------- Ratio Condor ----------------
def payoff_components_ratio_condor(
    S_T, opt_type,
    K1, K2, K3, K4,
    prem1, prem2, prem3, prem4,
    m_long1, m_short2, m_short3, m_long4,
    qty
):
    """レシオ・コンドル：Long@K1 ×m1、Short@K2 ×m2、Short@K3 ×m3、Long@K4 ×m4"""
    if opt_type == "call":
        p1 = np.maximum(S_T - K1, 0.0)
        p2 = np.maximum(S_T - K2, 0.0)
        p3 = np.maximum(S_T - K3, 0.0)
        p4 = np.maximum(S_T - K4, 0.0)
    else:
        p1 = np.maximum(K1 - S_T, 0.0)
        p2 = np.maximum(K2 - S_T, 0.0)
        p3 = np.maximum(K3 - S_T, 0.0)
        p4 = np.maximum(K4 - S_T, 0.0)

    leg1 = m_long1  * (-prem1 + p1) * qty
    leg2 = m_short2 * ( +prem2 - p2) * qty
    leg3 = m_short3 * ( +prem3 - p3) * qty
    leg4 = m_long4  * (-prem4 + p4) * qty
    combo = leg1 + leg2 + leg3 + leg4
    return {"leg1": leg1, "leg2": leg2, "leg3": leg3, "leg4": leg4, "combo": combo}

def build_grid_and_rows_ratio_condor(
    opt_type, K1, K2, K3, K4, prem1, prem2, prem3, prem4, m1, m2, m3, m4,
    qty, smin, smax, points, *, step: float = 0.25
):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_ratio_condor(S_T, opt_type, K1, K2, K3, K4, prem1, prem2, prem3, prem4, m1, m2, m3, m4, qty)
    rows = [{
        "st": float(S_T[i]),
        "leg1": float(pl["leg1"][i]),
        "leg2": float(pl["leg2"][i]),
        "leg3": float(pl["leg3"][i]),
        "leg4": float(pl["leg4"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_ratio_condor(S_T, pl, S0, K1, K2, K3, K4, be_list, opt_type, m1, m2, m3, m4):
    fig = plt.figure(figsize=(7.8, 5.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["leg1"], label=f"Long @{K1:.1f} ×{m1}", color="blue")
    ax.plot(S_T, pl["leg2"], label=f"Short @{K2:.1f} ×{m2}", color="orange")
    ax.plot(S_T, pl["leg3"], label=f"Short @{K3:.1f} ×{m3}", color="red")
    ax.plot(S_T, pl["leg4"], label=f"Long @{K4:.1f} ×{m4}", color="purple")
    ax.plot(S_T, pl["combo"], label="Combo (Ratio Condor)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["leg1"], pl["leg2"], pl["leg3"], pl["leg4"], pl["combo"]]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt in [(S0,f"S0={S0:.1f}"), (K1,f"K1={K1:.1f}"), (K2,f"K2={K2:.1f}"), (K3,f"K3={K3:.1f}"), (K4,f"K4={K4:.1f}")]:
        ls = "--" if x==S0 else ":"
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)

    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title(f"Ratio Condor ({opt_type.title()})")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_ratio_condor(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY"); ax.set_ylabel("P/L (JPY)"); ax.set_title("Break-even (Ratio Condor)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

@app.route("/fx/ratio-condor", methods=["GET", "POST"])
def fx_ratio_condor():
    defaults = dict(
        opt_type="call",
        S0=150.0,
        K1=146.0, K2=149.0, K3=151.0, K4=154.0,
        m1=1, m2=2, m3=2, m4=1,          # 1:-2:-2:+1 を標準に
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000, months=1.0,
        smin=135.0, smax=165.0, points=241
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        opt_type = request.form.get("opt_type", defaults["opt_type"]).strip().lower()
        if opt_type not in ("call","put"): opt_type="call"
        S0=fget("S0"); K1=fget("K1"); K2=fget("K2"); K3=fget("K3"); K4=fget("K4")
        m1=int(fget("m1", int, defaults["m1"])); m2=int(fget("m2", int, defaults["m2"]))
        m3=int(fget("m3", int, defaults["m3"])); m4=int(fget("m4", int, defaults["m4"]))
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months=fget("months")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        opt_type=defaults["opt_type"]; S0=defaults["S0"]
        K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]; K4=defaults["K4"]
        m1=defaults["m1"]; m2=defaults["m2"]; m3=defaults["m3"]; m4=defaults["m4"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months=defaults["months"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma = max(vol,0.0)/100.0; rD=r_dom/100.0; rF=r_for/100.0

    if opt_type=="call":
        prem1 = garman_kohlhagen_call(S0,K1,rD,rF,sigma,T)
        prem2 = garman_kohlhagen_call(S0,K2,rD,rF,sigma,T)
        prem3 = garman_kohlhagen_call(S0,K3,rD,rF,sigma,T)
        prem4 = garman_kohlhagen_call(S0,K4,rD,rF,sigma,T)
    else:
        prem1 = garman_kohlhagen_put (S0,K1,rD,rF,sigma,T)
        prem2 = garman_kohlhagen_put (S0,K2,rD,rF,sigma,T)
        prem3 = garman_kohlhagen_put (S0,K3,rD,rF,sigma,T)
        prem4 = garman_kohlhagen_put (S0,K4,rD,rF,sigma,T)

    S_T, pl, rows = build_grid_and_rows_ratio_condor(opt_type, K1, K2, K3, K4, prem1, prem2, prem3, prem4, m1, m2, m3, m4, qty, smin, smax, points)
    be_list = _find_break_evens_generic(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    premium_net_per_usd = (-m1*prem1 + m2*prem2 + m3*prem3 - m4*prem4)
    premium_net_jpy = premium_net_per_usd * qty

    fig = draw_chart_ratio_condor(S_T, pl, S0, K1, K2, K3, K4, be_list, opt_type, m1, m2, m3, m4)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_ratio_condor(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_ratio_condor.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        opt_type=opt_type, S0=S0, K1=K1, K2=K2, K3=K3, K4=K4,
        m1=m1, m2=m2, m3=m3, m4=m4,
        vol=vol, r_dom=r_dom, r_for=r_for, qty=qty, months=months,
        smin=smin, smax=smax, points=points,
        # 出力
        premium_net_per_usd=premium_net_per_usd, premium_net_jpy=premium_net_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows,
        prem1=prem1, prem2=prem2, prem3=prem3, prem4=prem4
    )

# CSV（Ratio Condor）
@app.route("/fx/download_csv_ratio_condor", methods=["POST"])
def fx_download_csv_ratio_condor():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    opt_type = request.form.get("opt_type","call").strip().lower()
    if opt_type not in ("call","put"): opt_type="call"
    K1=fget("K1", float, 146.0); K2=fget("K2", float, 149.0); K3=fget("K3", float, 151.0); K4=fget("K4", float, 154.0)
    prem1=fget("prem1", float, 2.0); prem2=fget("prem2", float, 1.6); prem3=fget("prem3", float, 1.4); prem4=fget("prem4", float, 1.1)
    m1=int(fget("m1", int, 1)); m2=int(fget("m2", int, 2)); m3=int(fget("m3", int, 2)); m4=int(fget("m4", int, 1))
    qty=fget("qty", float, 1_000_000.0); smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0); points=fget("points", float, 241)
    step=0.25
    S_T, pl, _ = build_grid_and_rows_ratio_condor(opt_type, K1, K2, K3, K4, prem1, prem2, prem3, prem4, m1, m2, m3, m4, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "Leg1_PnL(JPY)", "Leg2_PnL(JPY)", "Leg3_PnL(JPY)", "Leg4_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['leg1'][i]:.6f}", f"{pl['leg2'][i]:.6f}", f"{pl['leg3'][i]:.6f}", f"{pl['leg4'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="ratio_condor_pnl.csv")

# ======================= Double Calendar (Straddle / Strangle) =======================
# 依存：import numpy as np, import io, import base64, import matplotlib.pyplot as plt
#       from flask import request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put

# 汎用：損益分岐点（ゼロクロス）検出
def _find_break_evens_generic(S_T, y):
    S_T = np.asarray(S_T, dtype=float)
    y = np.asarray(y, dtype=float)
    bes = []
    for i in range(len(S_T) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            x0, x1 = S_T[i], S_T[i+1]
            x = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            bes.append(x)
    # 近接重複の整理
    res = []
    for v in sorted(bes):
        if not res or abs(v - res[-1]) > 1e-6:
            res.append(v)
    return res


# ---------- Double Calendar (Straddle) ----------
def payoff_components_double_calendar_straddle(
    S_T, K,
    # 初期プレミアム（JPY/USD）
    prem_far_call, prem_far_put,
    prem_near_call, prem_near_put,
    # 近満期時点でのファー・レッグ価値評価用
    r_dom, r_for, sigma_far, T_rem,
    qty
):
    """
    ダブル・カレンダー（ストラドル）：
      Long Far Call/Put @K、Short Near Call/Put @K
    P/Lは「近満期時点」の関数として算出：
      - Near（Short）は 受取プレミアム − 近満期での内在価値
      - Far（Long）は −支払プレミアム ＋ 近満期時点における残存期間 T_rem のGK価格
    """
    S_T = np.asarray(S_T, dtype=float)

    # Near legs (short)
    short_near_call_pl = (prem_near_call - np.maximum(S_T - K, 0.0)) * qty
    short_near_put_pl  = (prem_near_put  - np.maximum(K - S_T, 0.0)) * qty

    # Far legs (long) ― 近満期時点での残存期間 T_rem の理論価格で評価
    # garman_kohlhagen_* はスカラー想定でも np.vectorize で対応
    _gk_call_vec = np.vectorize(lambda s: garman_kohlhagen_call(s, K, r_dom, r_for, sigma_far, max(T_rem, 1e-8)))
    _gk_put_vec  = np.vectorize(lambda s: garman_kohlhagen_put (s, K, r_dom, r_for, sigma_far, max(T_rem, 1e-8)))

    val_far_call_at_near = _gk_call_vec(S_T)
    val_far_put_at_near  = _gk_put_vec (S_T)

    long_far_call_pl = (-prem_far_call + val_far_call_at_near) * qty
    long_far_put_pl  = (-prem_far_put  + val_far_put_at_near)  * qty

    near_pair_pl = short_near_call_pl + short_near_put_pl
    far_pair_pl  = long_far_call_pl   + long_far_put_pl
    combo_pl     = near_pair_pl + far_pair_pl

    return {
        "short_near_call": short_near_call_pl,
        "short_near_put":  short_near_put_pl,
        "long_far_call":   long_far_call_pl,
        "long_far_put":    long_far_put_pl,
        "near_pair":       near_pair_pl,
        "far_pair":        far_pair_pl,
        "combo":           combo_pl
    }


def build_grid_and_rows_double_calendar_straddle(
    K, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
    r_dom, r_for, sigma_far, T_rem, qty,
    smin, smax, points, *, step: float = 0.25
):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_double_calendar_straddle(
        S_T, K, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
        r_dom, r_for, sigma_far, T_rem, qty
    )
    rows = [{
        "st": float(S_T[i]),
        "near_call": float(pl["short_near_call"][i]),
        "near_put":  float(pl["short_near_put"][i]),
        "far_call":  float(pl["long_far_call"][i]),
        "far_put":   float(pl["long_far_put"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_double_calendar_straddle(S_T, pl, S0, K, be_list):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["near_pair"], label="Near Pair (Short C+P)", color="orange")
    ax.plot(S_T, pl["far_pair"],  label="Far Pair (Long C+P)",   color="blue")
    ax.plot(S_T, pl["combo"],     label="Combo (Double Calendar Straddle)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["near_pair"], pl["far_pair"], pl["combo"]]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K, f"K={K:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at Near Expiry (USD/JPY)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Double Calendar (Straddle) @Near Expiry")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


def draw_be_focus_generic(S_T, combo_pl, be_list, title):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Spot at Near Expiry (USD/JPY)"); ax.set_ylabel("P/L (JPY)"); ax.set_title(title)
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


@app.route("/fx/double-calendar-straddle", methods=["GET", "POST"])
def fx_double_calendar_straddle():
    defaults = dict(
        S0=150.0, K=150.0,
        vol_near=10.0, vol_far=10.0,
        r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        months_near=1.0, months_far=2.0,   # far > near
        smin=135.0, smax=165.0, points=241
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); K=fget("K")
        vol_near=fget("vol_near"); vol_far=fget("vol_far")
        r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months_near=fget("months_near"); months_far=fget("months_far")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        S0=defaults["S0"]; K=defaults["K"]
        vol_near=defaults["vol_near"]; vol_far=defaults["vol_far"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months_near=defaults["months_near"]; months_far=defaults["months_far"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    points = clamp_points(points)
    # 時間換算
    T_near = max(months_near, 0.0001)/12.0
    T_far  = max(months_far,  T_near + 0.0001)/12.0    # far > near を担保
    T_rem  = max(T_far - T_near, 1e-8)

    sigma_near = max(vol_near,0.0)/100.0
    sigma_far  = max(vol_far, 0.0)/100.0
    rD=r_dom/100.0; rF=r_for/100.0

    # 初期プレミアム（JPY/USD）
    prem_near_call = garman_kohlhagen_call(S0, K, rD, rF, sigma_near, T_near)
    prem_near_put  = garman_kohlhagen_put (S0, K, rD, rF, sigma_near, T_near)
    prem_far_call  = garman_kohlhagen_call(S0, K, rD, rF, sigma_far,  T_far)
    prem_far_put   = garman_kohlhagen_put (S0, K, rD, rF, sigma_far,  T_far)

    # 純額（Long far, Short near）
    premium_net_per_usd = (-prem_far_call - prem_far_put) + (prem_near_call + prem_near_put)
    premium_net_jpy = premium_net_per_usd * qty

    # グリッド評価（近満期）
    S_T, pl, rows = build_grid_and_rows_double_calendar_straddle(
        K, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
        rD, rF, sigma_far, T_rem, qty, smin, smax, points
    )

    be_list = _find_break_evens_generic(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 図①：全体
    fig = draw_chart_double_calendar_straddle(S_T, pl, S0, K, be_list)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：BEフォーカス
    fig2 = draw_be_focus_generic(S_T, pl["combo"], be_list, "Break-even (Double Calendar Straddle)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_double_calendar_straddle.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K=K, vol_near=vol_near, vol_far=vol_far, r_dom=r_dom, r_for=r_for,
        qty=qty, months_near=months_near, months_far=months_far,
        smin=smin, smax=smax, points=points,
        # 出力
        premium_net_per_usd=premium_net_per_usd, premium_net_jpy=premium_net_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows,
        prem_near_call=prem_near_call, prem_near_put=prem_near_put,
        prem_far_call=prem_far_call, prem_far_put=prem_far_put
    )


# CSV（Double Calendar Straddle）
@app.route("/fx/download_csv_double_calendar_straddle", methods=["POST"])
def fx_download_csv_double_calendar_straddle():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    S0=fget("S0", float, 150.0); K=fget("K", float, 150.0)
    vol_near=fget("vol_near", float, 10.0); vol_far=fget("vol_far", float, 10.0)
    r_dom=fget("r_dom", float, 1.6); r_for=fget("r_for", float, 4.2)
    qty=fget("qty", float, 1_000_000.0)
    months_near=fget("months_near", float, 1.0); months_far=fget("months_far", float, 2.0)
    smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0); points=fget("points", float, 241)
    step=0.25

    T_near = max(months_near, 0.0001)/12.0
    T_far  = max(months_far,  T_near + 0.0001)/12.0
    T_rem  = max(T_far - T_near, 1e-8)
    sigma_near=max(vol_near,0.0)/100.0; sigma_far=max(vol_far,0.0)/100.0
    rD=r_dom/100.0; rF=r_for/100.0

    prem_near_call = garman_kohlhagen_call(S0, K, rD, rF, sigma_near, T_near)
    prem_near_put  = garman_kohlhagen_put (S0, K, rD, rF, sigma_near, T_near)
    prem_far_call  = garman_kohlhagen_call(S0, K, rD, rF, sigma_far,  T_far)
    prem_far_put   = garman_kohlhagen_put (S0, K, rD, rF, sigma_far,  T_far)

    S_T, pl, _ = build_grid_and_rows_double_calendar_straddle(
        K, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
        rD, rF, sigma_far, T_rem, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "NearCall_PnL(JPY)", "NearPut_PnL(JPY)", "FarCall_PnL(JPY)", "FarPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_near_call'][i]:.6f}", f"{pl['short_near_put'][i]:.6f}", f"{pl['long_far_call'][i]:.6f}", f"{pl['long_far_put'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="double_calendar_straddle_pnl.csv")


# ---------- Double Calendar (Strangle) ----------
def payoff_components_double_calendar_strangle(
    S_T, Kc, Kp,
    prem_far_call, prem_far_put,
    prem_near_call, prem_near_put,
    r_dom, r_for, sigma_far, T_rem,
    qty
):
    """
    ダブル・カレンダー（ストラングル）：
      Long Far Call @Kc, Long Far Put @Kp
      Short Near Call @Kc, Short Near Put @Kp
    評価はストラドル版と同様（近満期時点）。
    """
    S_T = np.asarray(S_T, dtype=float)

    short_near_call_pl = (prem_near_call - np.maximum(S_T - Kc, 0.0)) * qty
    short_near_put_pl  = (prem_near_put  - np.maximum(Kp - S_T, 0.0)) * qty

    _gk_call_vec = np.vectorize(lambda s, K: garman_kohlhagen_call(s, K, r_dom, r_for, sigma_far, max(T_rem, 1e-8)))
    _gk_put_vec  = np.vectorize(lambda s, K: garman_kohlhagen_put (s, K, r_dom, r_for, sigma_far, max(T_rem, 1e-8)))

    val_far_call_at_near = _gk_call_vec(S_T, Kc)
    val_far_put_at_near  = _gk_put_vec (S_T, Kp)

    long_far_call_pl = (-prem_far_call + val_far_call_at_near) * qty
    long_far_put_pl  = (-prem_far_put  + val_far_put_at_near)  * qty

    near_pair_pl = short_near_call_pl + short_near_put_pl
    far_pair_pl  = long_far_call_pl   + long_far_put_pl
    combo_pl     = near_pair_pl + far_pair_pl

    return {
        "short_near_call": short_near_call_pl,
        "short_near_put":  short_near_put_pl,
        "long_far_call":   long_far_call_pl,
        "long_far_put":    long_far_put_pl,
        "near_pair":       near_pair_pl,
        "far_pair":        far_pair_pl,
        "combo":           combo_pl
    }


def build_grid_and_rows_double_calendar_strangle(
    Kc, Kp, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
    r_dom, r_for, sigma_far, T_rem, qty,
    smin, smax, points, *, step: float = 0.25
):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_double_calendar_strangle(
        S_T, Kc, Kp, prem_far_call, prem_far_put, prem_near_call, prem_near_put, r_dom, r_for, sigma_far, T_rem, qty
    )
    rows = [{
        "st": float(S_T[i]),
        "near_call": float(pl["short_near_call"][i]),
        "near_put":  float(pl["short_near_put"][i]),
        "far_call":  float(pl["long_far_call"][i]),
        "far_put":   float(pl["long_far_put"][i]),
        "combo":     float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows


def draw_chart_double_calendar_strangle(S_T, pl, S0, Kc, Kp, be_list):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["near_pair"], label="Near Pair (Short C+P)", color="orange")
    ax.plot(S_T, pl["far_pair"],  label="Far Pair (Long C+P)",   color="blue")
    ax.plot(S_T, pl["combo"],     label="Combo (Double Calendar Strangle)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["near_pair"], pl["far_pair"], pl["combo"]]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (Kp, f"Kp={Kp:.1f}", ":"), (Kc, f"Kc={Kc:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Spot at Near Expiry (USD/JPY)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Double Calendar (Strangle) @Near Expiry")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


@app.route("/fx/double-calendar-strangle", methods=["GET", "POST"])
def fx_double_calendar_strangle():
    defaults = dict(
        S0=150.0, Kc=152.0, Kp=148.0,
        vol_near=10.0, vol_far=10.0,
        r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        months_near=1.0, months_far=2.0,
        smin=135.0, smax=165.0, points=241
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); Kc=fget("Kc"); Kp=fget("Kp")
        vol_near=fget("vol_near"); vol_far=fget("vol_far")
        r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months_near=fget("months_near"); months_far=fget("months_far")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        S0=defaults["S0"]; Kc=defaults["Kc"]; Kp=defaults["Kp"]
        vol_near=defaults["vol_near"]; vol_far=defaults["vol_far"]
        r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months_near=defaults["months_near"]; months_far=defaults["months_far"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    points = clamp_points(points)
    T_near = max(months_near, 0.0001)/12.0
    T_far  = max(months_far,  T_near + 0.0001)/12.0
    T_rem  = max(T_far - T_near, 1e-8)

    sigma_near=max(vol_near,0.0)/100.0; sigma_far=max(vol_far,0.0)/100.0
    rD=r_dom/100.0; rF=r_for/100.0

    prem_near_call = garman_kohlhagen_call(S0, Kc, rD, rF, sigma_near, T_near)
    prem_near_put  = garman_kohlhagen_put (S0, Kp, rD, rF, sigma_near, T_near)
    prem_far_call  = garman_kohlhagen_call(S0, Kc, rD, rF, sigma_far,  T_far)
    prem_far_put   = garman_kohlhagen_put (S0, Kp, rD, rF, sigma_far,  T_far)

    premium_net_per_usd = (-prem_far_call - prem_far_put) + (prem_near_call + prem_near_put)
    premium_net_jpy = premium_net_per_usd * qty

    S_T, pl, rows = build_grid_and_rows_double_calendar_strangle(
        Kc, Kp, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
        rD, rF, sigma_far, T_rem, qty, smin, smax, points
    )

    be_list = _find_break_evens_generic(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_double_calendar_strangle(S_T, pl, S0, Kc, Kp, be_list)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_generic(S_T, pl["combo"], be_list, "Break-even (Double Calendar Strangle)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_double_calendar_strangle.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, Kc=Kc, Kp=Kp, vol_near=vol_near, vol_far=vol_far, r_dom=r_dom, r_for=r_for,
        qty=qty, months_near=months_near, months_far=months_far,
        smin=smin, smax=smax, points=points,
        # 出力
        premium_net_per_usd=premium_net_per_usd, premium_net_jpy=premium_net_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows,
        prem_near_call=prem_near_call, prem_near_put=prem_near_put,
        prem_far_call=prem_far_call, prem_far_put=prem_far_put
    )


# CSV（Double Calendar Strangle）
@app.route("/fx/download_csv_double_calendar_strangle", methods=["POST"])
def fx_download_csv_double_calendar_strangle():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    S0=fget("S0", float, 150.0); Kc=fget("Kc", float, 152.0); Kp=fget("Kp", float, 148.0)
    vol_near=fget("vol_near", float, 10.0); vol_far=fget("vol_far", float, 10.0)
    r_dom=fget("r_dom", float, 1.6); r_for=fget("r_for", float, 4.2)
    qty=fget("qty", float, 1_000_000.0)
    months_near=fget("months_near", float, 1.0); months_far=fget("months_far", float, 2.0)
    smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0); points=fget("points", float, 241)
    step=0.25

    T_near = max(months_near, 0.0001)/12.0
    T_far  = max(months_far,  T_near + 0.0001)/12.0
    T_rem  = max(T_far - T_near, 1e-8)
    sigma_near=max(vol_near,0.0)/100.0; sigma_far=max(vol_far,0.0)/100.0
    rD=r_dom/100.0; rF=r_for/100.0

    prem_near_call = garman_kohlhagen_call(S0, Kc, rD, rF, sigma_near, T_near)
    prem_near_put  = garman_kohlhagen_put (S0, Kp, rD, rF, sigma_near, T_near)
    prem_far_call  = garman_kohlhagen_call(S0, Kc, rD, rF, sigma_far,  T_far)
    prem_far_put   = garman_kohlhagen_put (S0, Kp, rD, rF, sigma_far,  T_far)

    S_T, pl, _ = build_grid_and_rows_double_calendar_strangle(
        Kc, Kp, prem_far_call, prem_far_put, prem_near_call, prem_near_put,
        rD, rF, sigma_far, T_rem, qty, smin, smax, points, step=step
    )

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "NearCall_PnL(JPY)", "NearPut_PnL(JPY)", "FarCall_PnL(JPY)", "FarPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_near_call'][i]:.6f}", f"{pl['short_near_put'][i]:.6f}", f"{pl['long_far_call'][i]:.6f}", f"{pl['long_far_put'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="double_calendar_strangle_pnl.csv")

# ======================= Broken-Wing Iron Condor =======================
# 依存：numpy as np, io, base64, matplotlib.pyplot as plt
#       from flask import request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put

# 損益分岐点（ゼロクロス）検出（既に定義済みならこの再定義は無視されます）
def _find_break_evens_generic(S_T, y):
    S_T = np.asarray(S_T, dtype=float)
    y = np.asarray(y, dtype=float)
    bes = []
    for i in range(len(S_T) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            x0, x1 = S_T[i], S_T[i+1]
            x = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            bes.append(x)
    res = []
    for v in sorted(bes):
        if not res or abs(v - res[-1]) > 1e-6:
            res.append(v)
    return res

def payoff_components_bw_iron_condor(
    S_T, K1, K2, K3, K4,
    prem_long_put, prem_short_put, prem_short_call, prem_long_call,
    qty
):
    """
    Broken-Wing Iron Condor（同満期）：
      Long Put @K1, Short Put @K2, Short Call @K3, Long Call @K4
      ※通常 K1 < K2 < K3 < K4、かつ翼幅が非対称（K2-K1 ≠ K4-K3）
    P/L（JPY）：
      Long Put  = (-prem_long_put  + max(K1 - S_T, 0)) * qty
      Short Put = ( prem_short_put - max(K2 - S_T, 0)) * qty
      Short Call= ( prem_short_call- max(S_T - K3, 0)) * qty
      Long Call = (-prem_long_call + max(S_T - K4, 0)) * qty
    """
    long_put_pl   = (-prem_long_put  + np.maximum(K1 - S_T, 0.0)) * qty
    short_put_pl  = ( prem_short_put - np.maximum(K2 - S_T, 0.0)) * qty
    short_call_pl = ( prem_short_call- np.maximum(S_T - K3, 0.0)) * qty
    long_call_pl  = (-prem_long_call + np.maximum(S_T - K4, 0.0)) * qty
    combo_pl = long_put_pl + short_put_pl + short_call_pl + long_call_pl
    return {
        "long_put": long_put_pl, "short_put": short_put_pl,
        "short_call": short_call_pl, "long_call": long_call_pl,
        "combo": combo_pl
    }

def build_grid_and_rows_bw_ic(K1, K2, K3, K4,
                              prem_long_put, prem_short_put, prem_short_call, prem_long_call,
                              qty, smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_bw_iron_condor(S_T, K1, K2, K3, K4,
                                          prem_long_put, prem_short_put, prem_short_call, prem_long_call, qty)
    rows = [{
        "st": float(S_T[i]),
        "long_put":   float(pl["long_put"][i]),
        "short_put":  float(pl["short_put"][i]),
        "short_call": float(pl["short_call"][i]),
        "long_call":  float(pl["long_call"][i]),
        "combo":      float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_bw_ic(S_T, pl, S0, K1, K2, K3, K4, be_list):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["long_put"],   label=f"Long Put @{K1:.1f}",  color="blue")
    ax.plot(S_T, pl["short_put"],  label=f"Short Put @{K2:.1f}", color="orange")
    ax.plot(S_T, pl["short_call"], label=f"Short Call @{K3:.1f}",color="red")
    ax.plot(S_T, pl["long_call"],  label=f"Long Call @{K4:.1f}", color="purple")
    ax.plot(S_T, pl["combo"],      label="Combo (BW Iron Condor)", color="green", linewidth=2)

    _set_ylim_tight(ax, [pl["long_put"], pl["short_put"], pl["short_call"], pl["long_call"], pl["combo"]])
    ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (K1, f"K1={K1:.1f}", ":"), (K2, f"K2={K2:.1f}", ":"), (K3, f"K3={K3:.1f}", ":"), (K4, f"K4={K4:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.2)
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Broken-Wing Iron Condor (P/L)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

def draw_be_focus_bw_ic(S_T, combo_pl, be_list):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Break-even (Broken-Wing Iron Condor)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

# 画面ルート
@app.route("/fx/broken-wing-iron-condor", methods=["GET", "POST"])
def fx_broken_wing_iron_condor():
    defaults = dict(
        S0=150.0,
        K1=145.0,  # Long Put (lower wing)
        K2=148.0,  # Short Put (inner short)
        K3=152.0,  # Short Call (inner short)
        K4=156.0,  # Long Call (upper wing)
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        months=1.0,
        smin=135.0, smax=165.0, points=241
    )

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0")
        K1=fget("K1"); K2=fget("K2"); K3=fget("K3"); K4=fget("K4")
        vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months=fget("months")
        smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        S0=defaults["S0"]
        K1=defaults["K1"]; K2=defaults["K2"]; K3=defaults["K3"]; K4=defaults["K4"]
        vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months=defaults["months"]
        smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    points = clamp_points(points)

    # GK式で理論プレミアム（JPY/USD）
    T = max(months, 0.0001) / 12.0
    sigma = max(vol, 0.0) / 100.0
    rD = r_dom / 100.0; rF = r_for / 100.0

    prem_long_put   = garman_kohlhagen_put (S0, K1, rD, rF, sigma, T)  # 支払（Long）
    prem_short_put  = garman_kohlhagen_put (S0, K2, rD, rF, sigma, T)  # 受取（Short）
    prem_short_call = garman_kohlhagen_call(S0, K3, rD, rF, sigma, T)  # 受取（Short）
    prem_long_call  = garman_kohlhagen_call(S0, K4, rD, rF, sigma, T)  # 支払（Long）

    # ネット・プレミアム（JPY/USD）：通常はクレジット
    premium_net_per_usd = (prem_short_put + prem_short_call) - (prem_long_put + prem_long_call)
    premium_net_jpy = premium_net_per_usd * qty

    # グリッド＆P/L
    S_T, pl, rows = build_grid_and_rows_bw_ic(K1, K2, K3, K4,
                                              prem_long_put, prem_short_put, prem_short_call, prem_long_call,
                                              qty, smin, smax, points)

    # 損益レンジ
    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    # 損益分岐点（数値的）
    be_list = _find_break_evens_generic(S_T, pl["combo"])

    # 翼幅と理論最大損失の目安（クレジット基準、デビットでも式は成立）
    width_put  = max(0.0, K2 - K1)
    width_call = max(0.0, K4 - K3)
    worst_put_loss_jpy  = max(0.0, width_put  * qty - premium_net_jpy)   # 下側に走った場合
    worst_call_loss_jpy = max(0.0, width_call * qty - premium_net_jpy)   # 上側に走った場合

    # 図①：全体損益
    fig = draw_chart_bw_ic(S_T, pl, S0, K1, K2, K3, K4, be_list)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 図②：損益分岐点フォーカス
    fig2 = draw_be_focus_bw_ic(S_T, pl["combo"], be_list)
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_broken_wing_iron_condor.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, K1=K1, K2=K2, K3=K3, K4=K4, vol=vol, r_dom=r_dom, r_for=r_for,
        qty=qty, months=months, smin=smin, smax=smax, points=points,
        # 出力
        prem_long_put=prem_long_put, prem_short_put=prem_short_put,
        prem_short_call=prem_short_call, prem_long_call=prem_long_call,
        premium_net_per_usd=premium_net_per_usd, premium_net_jpy=premium_net_jpy,
        width_put=width_put, width_call=width_call,
        worst_put_loss_jpy=worst_put_loss_jpy, worst_call_loss_jpy=worst_call_loss_jpy,
        be_vals=be_list,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Broken-Wing Iron Condor）
@app.route("/fx/download_csv_broken_wing_iron_condor", methods=["POST"])
def fx_download_csv_broken_wing_iron_condor():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default

    K1=fget("K1", float, 145.0); K2=fget("K2", float, 148.0); K3=fget("K3", float, 152.0); K4=fget("K4", float, 156.0)
    prem_long_put  = fget("prem_long_put",  float, 1.10)
    prem_short_put = fget("prem_short_put", float, 1.40)
    prem_short_call= fget("prem_short_call",float, 1.35)
    prem_long_call = fget("prem_long_call", float, 1.00)
    qty=fget("qty", float, 1_000_000.0)
    smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0)
    points=fget("points", float, 241); step=0.25

    S_T, pl, _ = build_grid_and_rows_bw_ic(K1, K2, K3, K4,
                                           prem_long_put, prem_short_put, prem_short_call, prem_long_call,
                                           qty, smin, smax, points, step=step)
    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "LongPut_PnL(JPY)", "ShortPut_PnL(JPY)", "ShortCall_PnL(JPY)", "LongCall_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['long_put'][i]:.6f}", f"{pl['short_put'][i]:.6f}", f"{pl['short_call'][i]:.6f}", f"{pl['long_call'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="broken_wing_iron_condor_pnl.csv")

# ======================= Covered Call / Cash-Secured Put =======================
# 依存：numpy as np, io, base64, matplotlib.pyplot as plt
#       from flask import request, render_template, send_file
# 既存ユーティリティ：_arange_inclusive, _set_ylim_tight, _format_y_as_m, clamp_points
# 価格関数：garman_kohlhagen_call, garman_kohlhagen_put

# 損益分岐点（ゼロクロス）検出（他で定義済みでも上書き可）
def _find_break_evens_generic(S_T, y):
    S_T = np.asarray(S_T, dtype=float)
    y = np.asarray(y, dtype=float)
    bes = []
    for i in range(len(S_T) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0.0:
            bes.append(float(S_T[i]))
        elif y0 * y1 < 0.0:
            x0, x1 = S_T[i], S_T[i+1]
            x = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            bes.append(x)
    res = []
    for v in sorted(bes):
        if not res or abs(v - res[-1]) > 1e-6:
            res.append(v)
    return res

def draw_be_focus_generic(S_T, combo_pl, be_list, title):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, combo_pl, linewidth=2, color="green", label="Combo P/L")
    _set_ylim_tight(ax, [combo_pl]); ax.axhline(0, linewidth=1)
    y_top = ax.get_ylim()[1]
    for i, be in enumerate(be_list, start=1):
        ax.axvline(be, linestyle="--", linewidth=1.5, label=f"BE{i}")
        ax.text(be, y_top, f"BE{i}={be:.2f}", va="top", ha="left", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)"); ax.set_ylabel("P/L (JPY)")
    ax.set_title(title); _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig


# ---------- Cash-Secured Put ----------
def payoff_components_cash_secured_put(S_T, Kp, prem_put, qty):
    """
    Cash-Secured Put = Put売り（必要現金確保）
      Short Put P/L  = (prem_put - max(Kp - S_T, 0)) * qty
      Combo (=Short Put)
    """
    short_put_pl = (prem_put - np.maximum(Kp - S_T, 0.0)) * qty
    combo_pl = short_put_pl.copy()
    return {"short_put": short_put_pl, "combo": combo_pl}

def build_grid_and_rows_cash_secured_put(Kp, prem_put, qty, smin, smax, points, *, step: float = 0.25):
    if smin > smax: smin, smax = smax, smin
    S_T = _arange_inclusive(float(smin), float(smax), float(step))
    pl = payoff_components_cash_secured_put(S_T, Kp, prem_put, qty)
    rows = [{
        "st": float(S_T[i]),
        "short_put": float(pl["short_put"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(len(S_T))]
    return S_T, pl, rows

def draw_chart_cash_secured_put(S_T, pl, S0, Kp, be):
    fig = plt.figure(figsize=(7.6, 4.9), dpi=120); ax = fig.add_subplot(111)
    ax.plot(S_T, pl["short_put"], label="Short Put P/L", color="blue")
    ax.plot(S_T, pl["combo"],     label="Combo (Cash-Secured Put)", color="green", linewidth=2)
    _set_ylim_tight(ax, [pl["short_put"], pl["combo"]]); ax.axhline(0, linewidth=1)

    y_top = ax.get_ylim()[1]
    for x, txt, ls in [(S0, f"S0={S0:.1f}", "--"), (Kp, f"Kp={Kp:.1f}", ":")]:
        ax.axvline(x, linestyle=ls, linewidth=1); ax.text(x, y_top, txt, va="top", ha="left", fontsize=9)
    if be is not None:
        ax.axvline(be, linestyle="--", linewidth=1.2); ax.text(be, y_top, f"BE={be:.2f}", va="top", ha="left", fontsize=9)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Cash-Secured Put: Short Put (P/L)")
    _format_y_as_m(ax); ax.legend(loc="best"); ax.grid(True, linewidth=0.3); fig.tight_layout()
    return fig

@app.route("/fx/cash-secured-put", methods=["GET", "POST"])
def fx_cash_secured_put():
    defaults = dict(
        S0=150.0, Kp=148.0,
        vol=10.0, r_dom=1.6, r_for=4.2,
        qty=1_000_000,
        months=1.0,
        smin=135.0, smax=165.0, points=241
    )
    if request.method == "POST":
        def fget(name, cast=float, default=None):
            v = request.form.get(name, "")
            try: return cast(v)
            except Exception: return default if default is not None else defaults[name]
        S0=fget("S0"); Kp=fget("Kp"); vol=fget("vol"); r_dom=fget("r_dom"); r_for=fget("r_for")
        qty=fget("qty"); months=fget("months"); smin=fget("smin"); smax=fget("smax"); points=int(fget("points"))
    else:
        S0=defaults["S0"]; Kp=defaults["Kp"]; vol=defaults["vol"]; r_dom=defaults["r_dom"]; r_for=defaults["r_for"]
        qty=defaults["qty"]; months=defaults["months"]; smin=defaults["smin"]; smax=defaults["smax"]; points=defaults["points"]

    points = clamp_points(points)
    T = max(months, 0.0001)/12.0; sigma=max(vol,0.0)/100.0; rD=r_dom/100.0; rF=r_for/100.0

    prem_put = garman_kohlhagen_put(S0, Kp, rD, rF, sigma, T)     # 受取（Short）
    prem_put_jpy = prem_put * qty

    be_formula = Kp - prem_put
    max_loss_per_usd = -(Kp - prem_put)       # S_T→0 の下限
    max_loss_jpy = max_loss_per_usd * qty     # 負値（損失）

    S_T, pl, rows = build_grid_and_rows_cash_secured_put(Kp, prem_put, qty, smin, smax, points)
    be_list = _find_break_evens_generic(S_T, pl["combo"])

    idx_min = int(np.argmin(pl["combo"])); idx_max = int(np.argmax(pl["combo"]))
    range_floor, range_floor_st = float(pl["combo"][idx_min]), float(S_T[idx_min])
    range_cap,   range_cap_st   = float(pl["combo"][idx_max]), float(S_T[idx_max])

    fig = draw_chart_cash_secured_put(S_T, pl, S0, Kp, be_formula)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    fig2 = draw_be_focus_generic(S_T, pl["combo"], be_list, "Break-even (Cash-Secured Put)")
    buf2 = io.BytesIO(); fig2.savefig(buf2, format="png"); plt.close(fig2); buf2.seek(0)
    png_b64_be = base64.b64encode(buf2.getvalue()).decode("ascii")

    return render_template(
        "fx_cash_secured_put.html",
        png_b64=png_b64, png_b64_be=png_b64_be,
        # 入力
        S0=S0, Kp=Kp, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty, months=months,
        smin=smin, smax=smax, points=points,
        # 出力
        prem_put=prem_put, prem_put_jpy=prem_put_jpy,
        be_formula=be_formula, max_loss_per_usd=max_loss_per_usd, max_loss_jpy=max_loss_jpy,
        range_floor=range_floor, range_floor_st=range_floor_st,
        range_cap=range_cap, range_cap_st=range_cap_st,
        rows=rows
    )

# CSV（Cash-Secured Put）
@app.route("/fx/download_csv_cash_secured_put", methods=["POST"])
def fx_download_csv_cash_secured_put():
    def fget(name, cast=float, default=None):
        v = request.form.get(name, "")
        try: return cast(v)
        except Exception: return default
    Kp=fget("Kp", float, 148.0)
    prem_put=fget("prem_put", float, 1.00)
    qty=fget("qty", float, 1_000_000.0)
    smin=fget("smin", float, 135.0); smax=fget("smax", float, 165.0); points=fget("points", float, 241)
    step=0.25

    S_T, pl, _ = build_grid_and_rows_cash_secured_put(Kp, prem_put, qty, smin, smax, points, step=step)

    import csv, io
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["S_T(USD/JPY)", "ShortPut_PnL(JPY)", "Combo_PnL(JPY)"])
    for i in range(len(S_T)):
        w.writerow([f"{S_T[i]:.6f}", f"{pl['short_put'][i]:.6f}", f"{pl['combo'][i]:.6f}"])
    data = io.BytesIO(buf.getvalue().encode("utf-8-sig")); data.seek(0)
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="cash_secured_put_pnl.csv")


# ===============================================================================================================================================
# ============== Investment calculators ===============
# =====================================================

def parse_float(value, default=0.0):
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        v = float(value)
        if not isfinite(v):
            raise ValueError
        return v
    except Exception:
        return default

def fv_lump_sum(pv: float, r_m: float, n: int) -> float:
    if n <= 0:
        return pv
    return pv * ((1 + r_m) ** n)

def fv_annuity(pmt: float, r_m: float, n: int, due: bool) -> float:
    if n <= 0 or pmt == 0:
        return 0.0
    if r_m == 0:
        fv = pmt * n
    else:
        fv = pmt * (((1 + r_m) ** n - 1) / r_m)
    if due:
        fv *= (1 + r_m)
    return fv

def fv_total(pv: float, pmt: float, r_m: float, n: int, due: bool) -> float:
    return fv_lump_sum(pv, r_m, n) + fv_annuity(pmt, r_m, n, due)

def bisection_solve(func, lo, hi, tol=1e-10, max_iter=200):
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = func(mid)
        if abs(f_mid) < tol or (hi - lo) / 2 < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2

def downsample_yearly(series):
    out = [series[i] for i in range(0, len(series), 12)]
    if (len(series) - 1) % 12 != 0:
        out.append(series[-1])
    return out

def build_balance_series_savings(pv0: float, pmt: float, r_m: float, n: int, due_begin: bool):
    series = [max(0.0, pv0)]
    S = pv0
    for _ in range(n):
        if due_begin:
            S = (S + pmt) * (1.0 + r_m)
        else:
            S = S * (1.0 + r_m) + pmt
        series.append(S)
    return series

def build_balance_series_loan(L0: float, PMT: float, r_m: float, n: int):
    series = [max(0.0, L0)]
    S = L0
    for _ in range(n):
        S = S * (1.0 + r_m) - PMT
        series.append(S)
    return series

#---------------------------------------------------------------------------------------------------------------------
@app.route("/savings", methods=["GET", "POST"])
def page_savings():
    result = None
    PV_eff = PMT_eff = r_eff = None
    n_eff = 0
    due_eff = False

    if request.method == "POST":
        solve   = (request.form.get("solve") or "fv").strip()
        target_fv = parse_float(request.form.get("target_fv", "0"))
        pv        = parse_float(request.form.get("pv", "0"))
        pmt       = parse_float(request.form.get("pmt", "0"))
        years     = parse_float(request.form.get("years", "0"))
        annual    = parse_float(request.form.get("annual", "0"))
        due       = (request.form.get("due") or "begin") == "begin"

        if annual < -100 or annual > 100:
            flash("年率の範囲が不正です。", "danger")
            return render_template("savings.html", result=None)

        n   = int(round(years * 12)) if years > 0 else 0
        r_m = annual / 100.0 / 12.0

        def af_fv(r, n, due_flag):
            if n <= 0:
                return 0.0
            if r == 0.0:
                af = float(n)
            else:
                af = ((1 + r) ** n - 1.0) / r
            if due_flag and r != 0.0:
                af *= (1 + r)
            return af

        if solve == "fv":
            if years < 0:
                flash("年数は0以上で指定してください。", "danger")
                return render_template("savings.html", result=None)
            n = int(round(max(0.0, years) * 12))
            fv = fv_total(pv, pmt, r_m, n, due)
            result = {"solve": solve, "fv": round(fv, 2), "months": n}
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt, r_m, n, due

        elif solve == "pv":
            if years < 0:
                flash("年数は0以上で指定してください。", "danger")
                return render_template("savings.html", result=None)
            n   = int(round(max(0.0, years) * 12))
            fvP = fv_annuity(pmt, r_m, n, due)
            denom = (1 + r_m) ** n if n > 0 else 1.0
            pv_req = (target_fv - fvP) / denom
            result = {"solve": solve, "pv": round(pv_req, 2), "months": n}
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv_req, pmt, r_m, n, due

        elif solve == "pmt":
            if n <= 0:
                flash("年数は正の値で指定してください。", "danger")
                return render_template("savings.html", result=None)
            denom = float(n) if r_m == 0.0 else af_fv(r_m, n, due)
            if abs(denom) < 1e-15:
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("savings.html", result=None)
            fv_pv = fv_lump_sum(pv, r_m, n)
            pmt_req = (target_fv - fv_pv) / denom
            result = {"solve": solve, "pmt": round(pmt_req, 2), "months": n}
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt_req, r_m, n, due

        elif solve == "years":
            def g(n_float):
                n_ = max(0, int(round(n_float)))
                return fv_total(pv, pmt, r_m, n_, due) - target_fv
            lo, hi = 0.0, 1200.0
            val_lo, val_hi = g(lo), g(hi)
            if val_lo == 0:
                n_sol = 0
            elif val_lo * val_hi > 0:
                flash("目標額に到達できません。入力を見直してください。", "warning")
                return render_template("savings.html", result=None)
            else:
                for _ in range(200):
                    mid = (lo + hi) / 2.0
                    vm  = g(mid)
                    if abs(vm) < 1e-6 or (hi - lo) < 1e-6:
                        break
                    if val_lo * vm <= 0:
                        hi, val_hi = mid, vm
                    else:
                        lo, val_lo = mid, vm
                n_sol = int(round((lo + hi) / 2.0))
            years_needed = n_sol / 12.0
            result = {"solve": solve, "months": n_sol, "years": round(years_needed, 3)}
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt, r_m, n_sol, due

        elif solve == "rate":
            if n <= 0:
                flash("年数は正の値で指定してください。", "danger")
                return render_template("savings.html", result=None)
            def f(rm):
                return fv_total(pv, pmt, rm, n, due) - target_fv
            lo, hi = -0.95/12.0, 1.0/12.0
            r_m_sol = bisection_solve(f, lo, hi, tol=1e-12, max_iter=300)
            if r_m_sol is None:
                flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
            else:
                annual_pct = (((1 + r_m_sol) ** 12) - 1) * 100.0
                result = {
                    "solve": solve,
                    "annual_rate_pct": round(annual_pct, 6),
                    "monthly_rate_pct": round(r_m_sol * 100.0, 6),
                    "months": n,
                }
                PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt, r_m_sol, n, due
        else:
            flash("解く対象（どれを求めるか）を選択してください。", "danger")

        if result is not None:
            try:
                if n_eff > 0:
                    series_m = build_balance_series_savings(PV_eff or 0.0, PMT_eff or 0.0, r_eff or 0.0, n_eff, due_eff or False)
                    series_y = downsample_yearly(series_m)
                    labels_y = [f"{i}年" for i in range(len(series_y))]
                    result["chart_labels"] = labels_y
                    result["chart_data"]   = [round(x, 2) for x in series_y]
                else:
                    result["chart_labels"] = []
                    result["chart_data"]   = []
            except Exception:
                result["chart_labels"] = []
                result["chart_data"]   = []

    return render_template("savings.html", result=result)
#-------------------------------------------------------------------------------------------------------------------------------------
@app.route("/loan", methods=["GET", "POST"])
def page_loan():
    result = None
    L_eff = PMT_eff = r_eff = None
    n_eff = 0

    if request.method == "POST":
        solve   = (request.form.get("solve") or "payment").strip()
        L       = parse_float(request.form.get("loan_amount", "0"))
        years   = parse_float(request.form.get("years", "0"))
        PMT     = parse_float(request.form.get("monthly_payment", "0"))
        annual  = parse_float(request.form.get("annual", "0"))
        B       = parse_float(request.form.get("residual", "0"))

        if annual < -100 or annual > 100:
            flash("金利（年率）の範囲が不正です。", "danger")
            return redirect(url_for("page_loan"))
        if years < 0:
            flash("返済年数は0以上を指定してください。", "danger")
            return redirect(url_for("page_loan"))
        if B < 0:
            flash("最終残存元本（バルーン）は0以上で入力してください。", "danger")
            return redirect(url_for("page_loan"))

        n   = int(round(years * 12)) if years > 0 else 0
        r_m = annual / 100.0 / 12.0

        def pmt_from(L_, r_, n_, B_):
            if n_ <= 0 or L_ <= 0:
                return None
            if r_ == 0.0:
                return (L_ - B_) / n_
            if 1.0 + r_ <= 0.0:
                return None
            t = -n_ * log1p(r_)
            if t > 700.0:
                return None
            inv = exp(t)
            denom = 1.0 - inv
            if abs(denom) < 1e-15:
                return None
            return ((L_ - B_ * inv) * r_) / denom

        if solve == "payment":
            if n <= 0 or L <= 0:
                flash("借入金額と返済年数は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))
            p = pmt_from(L, r_m, n, B)
            if p is None or (r_m == 0.0 and L < B):
                flash("計算が不安定か、条件が不成立です（残存元本が大きすぎる等）。", "warning")
            else:
                result = {"solve": solve, "monthly_payment": round(p, 2), "n": n}
                L_eff, PMT_eff, r_eff, n_eff = L, p, r_m, n

        elif solve == "amount":
            if n <= 0 or PMT <= 0:
                flash("月額返済額と返済年数は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))
            if r_m == 0.0:
                L_req = PMT * n + B
            else:
                if 1.0 + r_m <= 0.0:
                    flash("金利が不正です。", "danger")
                    return redirect(url_for("page_loan"))
                t = -n * log1p(r_m)
                if t > 700.0:
                    flash("計算が不安定です。入力値を見直してください。", "warning")
                    return redirect(url_for("page_loan"))
                inv = exp(t)
                L_req = B * inv + PMT * (1.0 - inv) / r_m
            result = {"solve": solve, "loan_amount": round(L_req, 2), "n": n}
            L_eff, PMT_eff, r_eff, n_eff = L_req, PMT, r_m, n

        elif solve == "years":
            if L <= 0 or PMT <= 0:
                flash("借入金額と月額返済額は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))

            if r_m == 0.0:
                if L <= B:
                    flash("無利子では残存元本が大きすぎます（返済が成立しません）。", "warning")
                    return redirect(url_for("page_loan"))
                n_real = (L - B) / PMT
            else:
                if 1.0 + r_m <= 0.0:
                    flash("金利が不正です。", "danger")
                    return redirect(url_for("page_loan"))
                i = r_m
                denom = (L - PMT / i)
                if abs(denom) < 1e-15:
                    flash("返済条件が成立しません（PMTが金利相当と一致）。", "warning")
                    return redirect(url_for("page_loan"))
                rhs = (B - PMT / i) / denom
                if rhs <= 0.0:
                    flash("その条件では返済年数の解が見つかりません。パラメータを見直してください。", "warning")
                    return redirect(url_for("page_loan"))
                n_real = log(rhs) / log1p(i)
                if (not isfinite(n_real)) or n_real < 0:
                    flash("計算が不安定です。入力値を見直してください。", "warning")
                    return redirect(url_for("page_loan"))

            n_req = int(round(n_real))
            result = {"solve": solve, "months": n_req, "years": round(n_req / 12.0, 3)}
            L_eff, PMT_eff, r_eff, n_eff = L, PMT, r_m, n_req

        elif solve == "rate":
            if L <= 0 or PMT <= 0 or n <= 0:
                flash("借入金額・返済年数・月額返済額は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))
            pmt_r0 = (L - B) / n
            if r_m == 0.0 and abs(PMT - pmt_r0) < 1e-12:
                result = {"solve": solve, "monthly_rate_pct": 0.0, "annual_rate_pct": 0.0}
                L_eff, PMT_eff, r_eff, n_eff = L, PMT, 0.0, n
                return render_template("loan.html", result=result)

            def safe_f(r):
                v = pmt_from(L, r, n, B)
                if v is None or not isfinite(v):
                    return None
                return v - PMT

            grid = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
            vals = []
            for r in grid:
                fval = safe_f(r)
                if fval is not None:
                    vals.append((r, fval))

            r_sol = None
            for r, fval in vals:
                if abs(fval) < 1e-12:
                    r_sol = r
                    break

            if r_sol is None:
                bracket = None
                for i in range(len(vals) - 1):
                    r1, f1 = vals[i]
                    r2, f2 = vals[i + 1]
                    if f1 * f2 <= 0:
                        bracket = (r1, r2)
                        break
                if bracket is None:
                    flash("与えられた条件では金利の解が見つかりません。", "warning")
                    return render_template("loan.html", result=None)
                lo, hi = bracket
                r_sol = bisection_solve(lambda x: safe_f(x), lo, hi, tol=1e-12, max_iter=300)

            if r_sol is None:
                flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
            else:
                annual_pct = (((1.0 + r_sol) ** 12) - 1.0) * 100.0
                result = {
                    "solve": solve,
                    "monthly_rate_pct": round(r_sol * 100.0, 6),
                    "annual_rate_pct": round(annual_pct, 6),
                }
                L_eff, PMT_eff, r_eff, n_eff = L, PMT, r_sol, n
        else:
            flash("解く対象（どれを求めるか）を選択してください。", "danger")

        if result is not None:
            try:
                if n_eff > 0 and L_eff is not None and PMT_eff is not None and r_eff is not None:
                    series_m = build_balance_series_loan(L_eff, PMT_eff, r_eff, n_eff)
                    series_y = downsample_yearly(series_m)
                    labels_y = [f"{i}年" for i in range(len(series_y))]
                    result["chart_labels"] = labels_y
                    result["chart_data"]   = [round(x if x >= 0 else 0.0, 2) for x in series_y]
                else:
                    result["chart_labels"] = []
                    result["chart_data"]   = []
            except Exception:
                result["chart_labels"] = []
                result["chart_data"]   = []

    return render_template("loan.html", result=result)
#-------------------------------------------------------------------------------------------------------------------------------
@app.route("/drawdown", methods=["GET", "POST"])
def page_drawdown():
    from math import isfinite as _isfinite, log1p as _log1p, exp as _exp, log as _log
    result = None

    def annuity_factor(r_, n_, due_begin_):
        if n_ <= 0:
            return 0.0
        if r_ == 0.0:
            af = float(n_)
        else:
            if 1.0 + r_ <= 0.0:
                return None
            t = n_ * _log1p(r_)
            if t > 700.0:
                return None
            af = (_exp(t) - 1.0) / r_
        if due_begin_ and r_ != 0.0:
            af *= (1.0 + r_)
        return af

    def pow1pr_n(r_, n_):
        if n_ == 0:
            return 1.0
        if 1.0 + r_ <= 0.0:
            return None
        t = n_ * _log1p(r_)
        if t > 700.0:
            return None
        return _exp(t)

    def build_balance_series(PV0, WD0, r_, n_, due_begin_):
        series = [max(0.0, PV0)]
        S = PV0
        for _ in range(n_):
            if not due_begin_:
                S = S * (1.0 + r_) - WD0
            else:
                S = (S - WD0) * (1.0 + r_)
            series.append(S)
        return series

    def downsample_yearly_local(series):
        out = [series[i] for i in range(0, len(series), 12)]
        if (len(series) - 1) % 12 != 0:
            out.append(series[-1])
        return out

    if request.method == "POST":
        solve    = (request.form.get("solve") or "withdrawal").strip()
        PV       = parse_float(request.form.get("pv", "0"))
        WD       = parse_float(request.form.get("withdrawal", "0"))
        years    = parse_float(request.form.get("years", "0"))
        annual   = parse_float(request.form.get("annual", "0"))
        B        = parse_float(request.form.get("residual", "0"))
        due_str  = (request.form.get("due") or "end").strip()
        due_begin = (due_str == "begin")

        if annual < -100 or annual > 100:
            flash("利回り（年率）の範囲が不正です。", "danger")
            return render_template("drawdown.html", result=None)
        if years < 0:
            flash("取崩年数は0以上で入力してください。", "danger")
            return render_template("drawdown.html", result=None)
        if B < 0:
            flash("残存金額は0以上で入力してください。", "danger")
            return render_template("drawdown.html", result=None)

        n = int(round(years * 12)) if years > 0 else 0
        r = annual / 100.0 / 12.0

        if solve == "withdrawal":
            if n <= 0:
                flash("取崩年数を正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            X  = pow1pr_n(r, n)
            AF = annuity_factor(r, n, due_begin)
            if X is None or AF in (None, 0.0):
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("drawdown.html", result=None)
            WD_req = (PV - B / X) / (AF / X) if r != 0.0 else (PV - B) / n
            result = {"solve": solve, "withdrawal": round(WD_req, 2), "n": n}
            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD_req, n, r, due_begin

        elif solve == "pv":
            if n <= 0:
                flash("取崩年数を正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            X  = pow1pr_n(r, n)
            AF = annuity_factor(r, n, due_begin)
            if X is None or AF is None:
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("drawdown.html", result=None)
            PV_req = (B + WD * AF) / X if r != 0.0 else (B + WD * n)
            result = {"solve": solve, "pv": round(PV_req, 2), "n": n}
            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV_req, WD, n, r, due_begin

        elif solve == "residual":
            if n <= 0:
                flash("取崩年数を正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            X  = pow1pr_n(r, n)
            AF = annuity_factor(r, n, due_begin)
            if X is None or AF is None:
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("drawdown.html", result=None)
            B_req = PV * X - WD * AF if r != 0.0 else (PV - WD * n)
            result = {"solve": solve, "residual": round(B_req, 2), "n": n}
            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n, r, due_begin

        elif solve == "years":
            if WD <= 0:
                flash("取崩月額は正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            if r == 0.0:
                n_real = (PV - B) / WD
                if n_real < 0 or not isfinite(n_real):
                    flash("その条件では到達できません。", "warning")
                    return render_template("drawdown.html", result=None)
                n_req = max(0, int(round(n_real)))
            else:
                if 1.0 + r <= 0.0:
                    flash("金利が不正です。", "danger")
                    return render_template("drawdown.html", result=None)
                A = WD * ((1.0 + r) if due_begin else 1.0) / r
                denom = (PV - A)
                if abs(denom) < 1e-15:
                    flash("条件が特異です（PMT が金利相当と一致）。", "warning")
                    return render_template("drawdown.html", result=None)
                rhs = (B - A) / denom
                if rhs <= 0.0 or not isfinite(rhs):
                    flash("その条件では到達できません（パラメータを見直してください）。", "warning")
                    return render_template("drawdown.html", result=None)
                n_real = log(rhs) / log1p(r)
                if n_real < 0 or not isfinite(n_real):
                    flash("計算が不安定です。入力値を見直してください。", "warning")
                    return render_template("drawdown.html", result=None)
                n_req = max(0, int(round(n_real)))
            result = {"solve": solve, "months": n_req, "years": round(n_req / 12.0, 3)}
            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n_req, r, due_begin

        elif solve == "rate":
            if WD <= 0 or n <= 0:
                flash("取崩月額と取崩年数は正の値を入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            WD_r0 = (PV - B) / n
            if abs(WD - WD_r0) < 1e-12:
                result = {"solve": solve, "monthly_rate_pct": 0.0, "annual_rate_pct": 0.0, "n": n}
                PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n, 0.0, due_begin
            else:
                def WD_from(PV_, r_, n_, B_, due_begin_):
                    X  = pow1pr_n(r_, n_)
                    AF = annuity_factor(r_, n_, due_begin_)
                    if X is None or AF in (None, 0.0):
                        return None
                    return (PV_ * X - B_) / AF if r_ != 0.0 else (PV_ - B_) / n_
                def f(r_):
                    v = WD_from(PV, r_, n, B, due_begin)
                    if v is None or not isfinite(v):
                        return None
                    return v - WD
                grid = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
                vals = []
                for r_try in grid:
                    fv = f(r_try)
                    if fv is not None:
                        vals.append((r_try, fv))
                r_sol = None
                for r_try, fv in vals:
                    if abs(fv) < 1e-12:
                        r_sol = r_try
                        break
                bracket = None
                if r_sol is None:
                    for i in range(len(vals) - 1):
                        r1, f1 = vals[i]
                        r2, f2 = vals[i + 1]
                        if f1 * f2 <= 0:
                            bracket = (r1, r2)
                            break
                if r_sol is None and bracket is None:
                    flash("与えられた条件では利回りの解が見つかりません。", "warning")
                    return render_template("drawdown.html", result=None)
                if r_sol is None:
                    lo, hi = bracket
                    r_sol = bisection_solve(lambda x: f(x), lo, hi, tol=1e-12, max_iter=300)
                if r_sol is None:
                    flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
                    return render_template("drawdown.html", result=None)
                annual_pct = (((1.0 + r_sol) ** 12) - 1.0) * 100.0
                result = {
                    "solve": solve,
                    "monthly_rate_pct": round(r_sol * 100.0, 6),
                    "annual_rate_pct": round(annual_pct, 6),
                    "n": n,
                }
                PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n, r_sol, due_begin
        else:
            flash("解く対象（どれを求めるか）を選択してください。", "danger")
            return render_template("drawdown.html", result=None)

        try:
            series_m = build_balance_series(PV_eff, WD_eff, r_eff, n_eff, due_eff)
            series_y = downsample_yearly_local(series_m)
            labels_y = [f"{i}年" for i in range(len(series_y))]
            result["chart_labels"] = labels_y
            result["chart_data"]   = [round(x, 2) for x in series_y]
        except Exception:
            result["chart_labels"] = []
            result["chart_data"]   = []

    return render_template("drawdown.html", result=result)

# -------------- Run -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
