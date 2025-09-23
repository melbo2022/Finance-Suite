#金融電卓(積立、ローン、年金）を表示させるか、非表示にするかを選択してください（40行目）
#-----------------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
from __future__ import annotations

# Combined app: Option Borrow vs Hedge, FX Options Visualizer (Put/Call), and Investment calculators
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import io, base64
import numpy as np
from math import isfinite, log, log1p, exp

# ------- Matplotlib (headless, portable fonts) -------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib import rcParams
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False
#------------------------------------------------------
# 追加：ポイント数のクランプ関数
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

#-------------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "replace-this-key"

#金融電卓の表示・非表示フラグ-----------------------------------------------------------
app.config["SHOW_FIN_TOOLS"] = False #←　非表示にするときはFalse

# どのテンプレでも使える共通コンテキスト
@app.context_processor
def inject_flags():
    return {
        "show_fin_tools": app.config.get("SHOW_FIN_TOOLS", True)
    }

#-----------------------------------------------------------------------------------------

# =====================================================
# ================ Option Borrow tool =================
# =====================================================

import math
from datetime import datetime as dt, date

def to_float(x, default):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def norm_cdf(x: float) -> float:
    # Φ(x) = 0.5 * (1 + erf(x/√2))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

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


# ---------------- ユーティリティ ----------------
def clamp_points(points):
    """
    グラフ分解能pointsを安全な範囲に丸める。51～2001、数値化できないときは251。
    """
    try:
        p = int(float(points))
    except Exception:
        p = 251
    return max(51, min(p, 2001))

# ---------------- 数式（GK/正規CDF） -------------
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

# ---------------- 損益ロジック ----------------
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

# ---------------- 描画（メイン：M表記へ） ----------------
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
