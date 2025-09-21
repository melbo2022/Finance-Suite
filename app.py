#金融電卓(積立、ローン、年金）を表示させるか、非表示にするかを選択してください（38行目）
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
app.config["SHOW_FIN_TOOLS"] = False   # ← 表示したいとき True / 非表示にしたいとき False
#-----------------------------------------------------------------------------------------

# =====================================================
# ================ Option Borrow tool =================
# =====================================================

import math
from datetime import datetime as dt, date
# Flask の import は既存のファイル側にある想定です: from flask import request, render_template

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


#----------------------------------------------------------------------------------------------------
# app.py（ホームルート）
@app.route("/")
def home():
    return render_template("home.html", show_fin_tools=app.config["SHOW_FIN_TOOLS"])


# -----------------------------------------------------------------------------------------------
#hedge compare
# PUT と 借入 の比較
# -----------------------------------------------------------------------------------------------
@app.route("/hedge", methods=["GET", "POST"])
def borrow_index():
    params = {
        "notional_usd": 1_000_000.0,
        "usd_rate_annual": 4.2,         # r_f (%/y, USD)
        "jpy_rate_annual": 1.6,         # r_d (%/y, JPY)
        "spot_jpy_per_usd": 150.0,      # S
        "strike_jpy_per_usd": 148.0,    # K
        "vol_annual": 11.0,             # σ (%/y)
        "months": 1.0,                  # fallback
        "trade_date": "2025-09-19",
        "expiry_date": "2025-11-18",
        "use_dates": True,
    }

    result = None
    scenarios = []

    if request.method == "POST":
        for k in params.keys():
            v = request.form.get(k, params[k])
            if k in ("trade_date", "expiry_date"):
                params[k] = v or params[k]
            elif k == "use_dates":
                params[k] = str(v).lower() in ("1", "true", "on", "yes")
            else:
                params[k] = to_float(v, params[k])

        N = params["notional_usd"]
        S = params["spot_jpy_per_usd"]
        K = params["strike_jpy_per_usd"]
        r_f = params["usd_rate_annual"] / 100.0
        r_d = params["jpy_rate_annual"] / 100.0
        sigma = params["vol_annual"] / 100.0

        # T
        if params["use_dates"]:
            try:
                trade = dt.strptime(params["trade_date"], "%Y-%m-%d").date()
                expiry = dt.strptime(params["expiry_date"], "%Y-%m-%d").date()
                days = max((expiry - trade).days, 0)
                T = days / 365.0  # Actual/365
            except Exception:
                T = params["months"] / 12.0
        else:
            T = params["months"] / 12.0

        # 借入コスト
        borrow_cost_usd = N * r_f * T
        borrow_cost_jpy = borrow_cost_usd * S

        # Garman–Kohlhagen (PUT)
        if T > 0:
            d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
        else:
            d1 = d2 = float("inf") if S > K else float("-inf")

        # P = K e^{-r_d T} N(-d2) - S e^{-r_f T} N(-d1)
        put_premium_per_usd = (
            K * math.exp(-r_d * T) * norm_cdf(-d2)
            - S * math.exp(-r_f * T) * norm_cdf(-d1)
        )

        option_cost_jpy = N * put_premium_per_usd
        option_cost_usd = option_cost_jpy / S

        # 参考値
        delta_opt_vs_borrow = (option_cost_jpy - borrow_cost_jpy) / N
        delta_jpy_breakeven = option_cost_jpy / N

        # シナリオ（※PUT 側はご提示の式を踏襲）
        moves = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        for d in moves:
            # Spot損益（ご提示のロジック）
            kdiff = K - S
            if d >= 0:
                spot_pnl_jpy = d * N
            else:
                spot_pnl_jpy = max(d, kdiff) * N

            # オプション料は固定（実務表示）
            option_pnl_jpy = -option_cost_jpy

            # 借入の損益（固定）
            borrow_pnl_jpy = -borrow_cost_jpy

            option_plus_spot_pnl_jpy = spot_pnl_jpy + option_pnl_jpy
            diff_vs_borrow = option_plus_spot_pnl_jpy - borrow_pnl_jpy
            better = (
                "オプション" if option_plus_spot_pnl_jpy > borrow_pnl_jpy
                else ("借入" if option_plus_spot_pnl_jpy < borrow_pnl_jpy else "同等")
            )

            scenarios.append({
                "move": d,
                "option_pnl_jpy": option_pnl_jpy,
                "spot_pnl_jpy": spot_pnl_jpy,
                "borrow_pnl_jpy": borrow_pnl_jpy,
                "diff_vs_borrow": diff_vs_borrow,
                "better": better
            })

        # 追加の境界（未表示）
        raw_neg = (option_cost_jpy - borrow_cost_jpy) / N
        raw_pos = (option_cost_jpy - borrow_cost_jpy) / (2.0 * N)
        delta_plusspot_vs_borrow_neg = raw_neg if raw_neg < 0 else None
        delta_plusspot_vs_borrow_pos = raw_pos if raw_pos >= 0 else None

        result = {
            "borrow_cost_usd": borrow_cost_usd,
            "borrow_cost_jpy": borrow_cost_jpy,
            "option_cost_usd": option_cost_usd,
            "option_cost_jpy": option_cost_jpy,
            "put_premium_per_usd": put_premium_per_usd,
            "T": T,
            "d1": d1,
            "d2": d2,
            "delta_opt_vs_borrow": delta_opt_vs_borrow,
            "delta_plusspot_vs_borrow_neg": delta_plusspot_vs_borrow_neg,
            "delta_plusspot_vs_borrow_pos": delta_plusspot_vs_borrow_pos,
            "delta_jpy_breakeven": delta_jpy_breakeven,
        }

    return render_template("option_borrow_put.html", params=params, result=result, scenarios=scenarios)

# -----------------------------------------------------------------------------------------------
#hedge compare
# CALL と 借入 の比較
# -----------------------------------------------------------------------------------------------
@app.route("/hedge_call", methods=["GET", "POST"])
def hedge_call():
    """
    CALL（ドル買いヘッジ）
      - プレミアム：Garman–Kohlhagen（CALL）
      - Spot損益（ドル買い視点）：(S - min(S', K)) * N で円安側は K で頭打ち
      - オプション損益（表示）：オプション料は固定（-option_cost_jpy）
      - 合計：Spot + Option（固定料）
      - 借入コスト：固定（比較用）
    """
    params = {
        "notional_usd": 1_000_000.0,
        "usd_rate_annual": 4.2,      # r_f (%/y, USD)
        "jpy_rate_annual": 1.6,      # r_d (%/y, JPY)
        "spot_jpy_per_usd": 150.0,   # S
        "strike_jpy_per_usd": 152.0, # K
        "vol_annual": 11.0,          # σ (%/y)
        "months": 1.0,               # 満期（年）= months/12
        "use_dates": False,
    }

    result = None
    scenarios = []

    if request.method == "POST":
        for k in params.keys():
            v = request.form.get(k, params[k])
            if k == "use_dates":
                params[k] = str(v).lower() in ("1", "true", "on", "yes")
            else:
                params[k] = to_float(v, params[k])

        N = params["notional_usd"]
        S = params["spot_jpy_per_usd"]
        K = params["strike_jpy_per_usd"]
        r_f = params["usd_rate_annual"] / 100.0
        r_d = params["jpy_rate_annual"] / 100.0
        sigma = params["vol_annual"] / 100.0
        T = max(params["months"], 0.0) / 12.0

        # 借入コスト
        borrow_cost_usd = N * r_f * T
        borrow_cost_jpy = borrow_cost_usd * S

        # Garman–Kohlhagen (CALL)
        if T > 0 and sigma > 0:
            d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
        else:
            d1 = d2 = float("inf") if S > K else float("-inf")

        # C = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)
        call_premium_per_usd = (
            S * math.exp(-r_f * T) * norm_cdf(d1)
            - K * math.exp(-r_d * T) * norm_cdf(d2)
        )
        option_cost_jpy = N * call_premium_per_usd
        option_cost_usd = option_cost_jpy / S

        # シナリオ（ドル買い視点・Kで頭打ち）
        moves = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        for d in moves:
            S_prime = S + d  # 想定レート（円/ドル）

            # 市場での支払コスト差：S' > K なら K で買えるため円安側は K で頭打ち
            effective_rate = min(S_prime, K)
            spot_pnl_jpy = (S - effective_rate) * N  # 円高でプラス、円安は K で下限固定

            # オプション料は固定（実務表示）
            option_pnl_jpy = -option_cost_jpy

            # 借入損益（固定）
            borrow_pnl_jpy = -borrow_cost_jpy

            total_pnl = spot_pnl_jpy + option_pnl_jpy
            diff_vs_borrow = total_pnl - borrow_pnl_jpy
            better = (
                "オプション" if total_pnl > borrow_pnl_jpy
                else ("借入" if total_pnl < borrow_pnl_jpy else "同等")
            )

            scenarios.append({
                "move": d,
                "spot_pnl_jpy": spot_pnl_jpy,
                "option_pnl_jpy": option_pnl_jpy,
                "borrow_pnl_jpy": borrow_pnl_jpy,
                "diff_vs_borrow": diff_vs_borrow,
                "better": better
            })

        result = {
            "borrow_cost_usd": borrow_cost_usd,
            "borrow_cost_jpy": borrow_cost_jpy,
            "option_cost_usd": option_cost_usd,
            "option_cost_jpy": option_cost_jpy,
            "call_premium_per_usd": call_premium_per_usd,
            "T": T,
            "d1": d1,
            "d2": d2,
        }

    return render_template("option_borrow_call.html", params=params, result=result, scenarios=scenarios)



#----------------------------------------------------------------------------------------------------------------------------
# =====================================================
# ============== FX Options Visualizer ================
# =====================================================
#oputionの買い

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

# ---------------- 描画 ----------------
def draw_chart_put(S_T, pl, S0, K, floor_value, premium_jpy, finance_jpy):
    """
    Protective Put の損益グラフを描画。
    ※ Premium/Financing の文字ラベルは描かない（別カードで表示）。
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

    # Loss floor の水平線（線のみ。文字は描かない）
    ax.axhline(floor_value, linestyle=":", linewidth=1)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Protective Put: P/L vs Terminal USD/JPY")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

# ---------------- 画面ルート ----------------
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
    finance_jpy = qty * S0 * (borrow_rate / 100.0) * (months / 12.0)

    # 描画
    fig = draw_chart_put(S_T, pl, S0, K, floor_value, premium_jpy, finance_jpy)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return render_template(
        "fx_put.html",
        png_b64=png_b64,
        # 入力（Vol/金利/満期）
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力（算出値）
        premium=premium,                    # JPY/USD（CSVや表示に供給）
        floor=floor_value,
        premium_cost=premium_jpy,
        finance_cost=finance_jpy,
        total_cost=premium_jpy + finance_jpy,
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
    premium = fget("premium", float, 0.74)  # ← 画面で算出された値が hidden で渡ってくる
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

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
# ====== CALLの買い======

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

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


# ====== 既存: P/Lヘルパ ======
def payoff_components_call(S_T, S0, K, premium, qty):
    """
    Call用の損益内訳（JPY）。
    spot: USDショートの損益、opt: コール買い（プレミアム込み）、combo: 合成
    """
    spot_pl = (S_T - S0) * (-qty)                          # USDショートの損益
    call_pl = (np.maximum(S_T - K, 0.0) - premium) * qty   # コール損益（プレミアム込み）
    combo_pl = spot_pl + call_pl
    return {"spot": spot_pl, "opt": call_pl, "combo": combo_pl}

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

def draw_chart_call(S_T, pl, S0, K, floor_value):
    """
    Protective Call の損益グラフを描画。
    ※ Premium/Financing 等の数値ラベルは描かない（別カードで表示）。
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

    # Loss floor（線のみ。文字は描かない）
    ax.axhline(floor_value, linestyle=":", linewidth=1)

    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Protective Call: P/L vs Terminal USD/JPY")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig


# ====== ルート: /fx/call  ======
@app.route("/fx/call", methods=["GET", "POST"])
def fx_call():
    """
    Call 版：Premium はユーザ入力ではなく、Volatility/金利/満期から GK 式で算出。
    さらにオプション料の名目比（%）も表示する。
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
    # グラフ
    fig = draw_chart_call(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return render_template(
        "fx_call.html",
        png_b64=png_b64,
        # 入力
        S0=S0, K=K, vol=vol, r_dom=r_dom, r_for=r_for, qty=qty,
        smin=smin, smax=smax, points=points, months=months, borrow_rate=borrow_rate,
        # 出力（算出値）
        premium=premium,                 # JPY/USD（CSVや表示に供給）
        floor=floor_value,
        premium_cost=premium_jpy,
        finance_cost=finance_jpy,
        total_cost=premium_jpy + finance_jpy,
        premium_pct=premium_pct_of_qty,  # 名目比 %
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
    premium = fget("premium", float, 0.62)  # ← 画面で算出された値（hidden）を受ける
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

# =====================================================
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
#--------------------------------------------------------------------------------------------------------------------

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
