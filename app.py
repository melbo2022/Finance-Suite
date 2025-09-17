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

app = Flask(__name__)
app.secret_key = "replace-this-key"

# =====================================================
# ================ Option Borrow tool =================
# =====================================================

def to_float(x, default):
    try:
        v = float(x)
        if not isfinite(v):
            return default
        return v
    except Exception:
        return default

@app.route("/hedge", methods=["GET", "POST"])
def borrow_index():
    # Default parameters (sensible starting values)
    params = {
        "notional_usd": 2000000.0,
        "usd_rate_annual": 4.2,     # % per year
        "jpy_rate_annual": 1.6,     # % per year
        "option_premium_pct_per_month": 2.0,  # % of notional per month
        "spot_jpy_per_usd": 150.0,  # JPY per USD
        "months": 1.0
    }
    result = None
    scenarios = []

    if request.method == "POST":
        for k in params.keys():
            params[k] = to_float(request.form.get(k, params[k]), params[k])

        N = params["notional_usd"]
        months = params["months"]
        spot = params["spot_jpy_per_usd"]
        usd_rate = params["usd_rate_annual"] / 100.0
        jpy_rate = params["jpy_rate_annual"] / 100.0
        opt_prem_m = params["option_premium_pct_per_month"] / 100.0

        # Borrowing cost = notional * (usd_rate - jpy_rate) * (months/12)
        borrow_cost_usd = N * max(usd_rate - jpy_rate, 0.0) * (months / 12.0)
        borrow_cost_jpy = borrow_cost_usd * spot

        # Option cost = notional * opt_prem_m * months
        option_cost_usd = N * opt_prem_m * months
        option_cost_jpy = option_cost_usd * spot

        # Thresholds (ΔJPY required per USD)
        delta_jpy_vs_borrow = (option_cost_jpy - borrow_cost_jpy) / N
        delta_jpy_breakeven = option_cost_jpy / N

        moves = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        for d in moves:
            option_payoff_jpy = max(d, 0.0) * N
            option_pnl_jpy = option_payoff_jpy - option_cost_jpy
            borrow_pnl_jpy = -borrow_cost_jpy
            better = "オプション" if option_pnl_jpy > borrow_pnl_jpy else ("借入" if option_pnl_jpy < borrow_pnl_jpy else "同等")
            scenarios.append({
                "move": d,
                "option_pnl_jpy": option_pnl_jpy,
                "borrow_pnl_jpy": borrow_pnl_jpy,
                "better": better
            })

        result = {
            "borrow_cost_usd": borrow_cost_usd,
            "borrow_cost_jpy": borrow_cost_jpy,
            "option_cost_usd": option_cost_usd,
            "option_cost_jpy": option_cost_jpy,
            "delta_jpy_vs_borrow": delta_jpy_vs_borrow,
            "delta_jpy_breakeven": delta_jpy_breakeven,
        }

    return render_template("option_borrow.html", params=params, result=result, scenarios=scenarios)

# =====================================================
# ============== FX Options Visualizer ================
# =====================================================

def payoff_components_put(S_T, S0, K, premium, qty):
    spot_pl = (S_T - S0) * qty
    put_pl  = (np.maximum(K - S_T, 0.0) - premium) * qty
    combo_pl = spot_pl + put_pl
    return {"spot": spot_pl, "put": put_pl, "combo": combo_pl}

def payoff_components_call(S_T, S0, K, premium, qty):
    spot_pl = (S_T - S0) * (-qty)  # short USD spot
    call_pl = (np.maximum(S_T - K, 0.0) - premium) * qty
    combo_pl = spot_pl + call_pl
    return {"spot": spot_pl, "opt": call_pl, "combo": combo_pl}

def clamp_points(points):
    points = int(points)
    return max(51, min(points, 2001))

def build_grid_and_rows_put(S0, K, premium, qty, smin, smax, points):
    if smin >= smax:
        smin, smax = (min(smin, smax), max(smin, smax) + 1.0)
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
    if smin >= smax:
        smin, smax = (min(smin, smax), max(smin, smax) + 1.0)
    S_T = np.linspace(smin, smax, points)
    pl = payoff_components_call(S_T, S0, K, premium, qty)
    rows = [{
        "st": float(S_T[i]),
        "spot": float(pl["spot"][i]),
        "opt":  float(pl["opt"][i]),
        "combo": float(pl["combo"][i]),
    } for i in range(points)]
    return S_T, pl, rows

def draw_chart_put(S_T, pl, S0, K, floor_value):
    fig = plt.figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, pl["spot"], label="Spot USD P/L (vs today)")
    ax.plot(S_T, pl["put"],  label="Long Put P/L (incl. premium)")
    ax.plot(S_T, pl["combo"], linewidth=2, label="Protective Put Combo P/L")
    ax.axhline(0, linewidth=1)
    ax.axvline(S0, linestyle="--", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)
    ax.axhline(floor_value, linestyle=":", linewidth=1)
    ax.text(S_T[-1], floor_value, f" Loss floor = {floor_value:,.0f}", va="bottom", ha="right", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Protective Put: P/L vs Terminal USD/JPY")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

def draw_chart_call(S_T, pl, S0, K, floor_value):
    fig = plt.figure(figsize=(7, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(S_T, pl["spot"], label="Short USD Spot P/L (vs today)")
    ax.plot(S_T, pl["opt"],  label="Long Call P/L (incl. premium)")
    ax.plot(S_T, pl["combo"], linewidth=2, label="Protective Call Combo P/L")
    ax.axhline(0, linewidth=1)
    ax.axvline(S0, linestyle="--", linewidth=1)
    y_top = ax.get_ylim()[1]
    ax.text(S0, y_top, f"S0={S0:.1f}", va="top", ha="left", fontsize=9)
    ax.axvline(K, linestyle=":", linewidth=1)
    ax.text(K, y_top, f"K={K:.1f}", va="top", ha="left", fontsize=9)
    ax.axhline(floor_value, linestyle=":", linewidth=1)
    ax.text(S_T[-1], floor_value, f" Loss floor = {floor_value:,.0f}", va="bottom", ha="right", fontsize=9)
    ax.set_xlabel("Terminal USD/JPY (Spot at Expiry)")
    ax.set_ylabel("P/L (JPY)")
    ax.set_title("Protective Call: P/L vs Terminal USD/JPY")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    return fig

@app.route("/fx/put", methods=["GET", "POST"])
def fx_put():
    defaults = dict(S0=150.0, K=145.0, premium=1.2, qty=1_000_000, smin=120.0, smax=170.0, points=251)

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K = fget("K", float, defaults["K"])
        premium = fget("premium", float, defaults["premium"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
    else:
        S0, K, premium, qty, smin, smax, points = defaults.values()

    points = clamp_points(points)
    S_T, pl, rows = build_grid_and_rows_put(S0, K, premium, qty, smin, smax, points)
    floor_value = (K - S0 - premium) * qty

    fig = draw_chart_put(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return render_template(
        "fx_put.html",
        png_b64=png_b64,
        S0=S0, K=K, premium=premium, qty=qty,
        smin=smin, smax=smax, points=points,
        floor=floor_value,
        rows=rows
    )

@app.route("/fx/download_csv_put", methods=["POST"])
def fx_download_csv_put():
    def fget(name, cast=float, default=None):
        val = request.form.get(name, "")
        try:
            return cast(val)
        except Exception:
            return default
    S0 = fget("S0", float, 150.0)
    K = fget("K", float, 145.0)
    premium = fget("premium", float, 1.2)
    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 120.0)
    smax = fget("smax", float, 170.0)
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

@app.route("/fx/call", methods=["GET", "POST"])
def fx_call():
    defaults = dict(S0=150.0, K=155.0, premium=1.2, qty=1_000_000, smin=120.0, smax=170.0, points=251)

    if request.method == "POST":
        def fget(name, cast=float, default=None):
            val = request.form.get(name, "")
            try:
                return cast(val)
            except Exception:
                return default if default is not None else defaults[name]
        S0 = fget("S0", float, defaults["S0"])
        K = fget("K", float, defaults["K"])
        premium = fget("premium", float, defaults["premium"])
        qty = fget("qty", float, defaults["qty"])
        smin = fget("smin", float, defaults["smin"])
        smax = fget("smax", float, defaults["smax"])
        points = int(fget("points", float, defaults["points"]))
    else:
        S0, K, premium, qty, smin, smax, points = defaults.values()

    points = clamp_points(points)
    S_T, pl, rows = build_grid_and_rows_call(S0, K, premium, qty, smin, smax, points)
    floor_value = (S0 - K - premium) * qty

    fig = draw_chart_call(S_T, pl, S0, K, floor_value)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return render_template(
        "fx_call.html",
        png_b64=png_b64,
        S0=S0, K=K, premium=premium, qty=qty,
        smin=smin, smax=smax, points=points,
        floor=floor_value,
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
    K = fget("K", float, 155.0)
    premium = fget("premium", float, 1.2)
    qty = fget("qty", float, 1_000_000.0)
    smin = fget("smin", float, 120.0)
    smax = fget("smax", float, 170.0)
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

@app.route("/")
def home():
    return render_template("home.html")

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

# -------------- Run --------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
