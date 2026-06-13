"""
Strategy Backtest — Full-Rule Equity Curve
──────────────────────────────────────────
Tests the COMPLETE momentum strategy rules over history to answer:
"If I had run these rules for the past N years, what would the equity curve
 look like? What was the worst drawdown I'd have faced and had to survive?"

This is the source of 底氣 (conviction to act). Seeing the worst −30% drawdown
and the eventual recovery is what allows you to stay in the trade when it hurts.
Without this, you're flying blind on risk tolerance.

Strategy rules:
  1. Monthly rebalance: rank universe by composite price-momentum score
     composite = 0.6 × mom_12_1  +  0.4 × mom_6m  (point-in-time valid)
  2. Hold equal-weighted top quintile (top 20% of universe, ~10 names/50)
  3. Transaction cost: 0.15% per side × estimated turnover each month
  4. Long-only; no leverage; monthly hold

Benchmark: SPY (buy-and-hold over same period).

Scope note: this validates the PRICE-BASED signals (≈37% of composite weight).
Revision, PEAD, quality, short-interest are validated via live snapshots (track record).
This is the floor: the full strategy should do BETTER once those signals are validated.

Output:
  reports/strategy_backtest.json  — stats (for alerts/portfolio integration)
  reports/strategy_backtest.html  — equity curve + charts (browser)
  Console summary

Usage:
  python strategy_backtest.py            # default 50-name universe, 6y
  python strategy_backtest.py --period 10y
  python strategy_backtest.py --top 0.25 --tc 0.002
"""
from __future__ import annotations
import json
import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'TSM', 'AMD', 'QCOM',
    'JPM', 'BAC', 'GS', 'V', 'MA', 'UNH', 'JNJ', 'LLY', 'PFE', 'MRK',
    'XOM', 'CVX', 'CAT', 'HON', 'GE', 'PG', 'KO', 'PEP', 'WMT', 'COST',
    'HD', 'MCD', 'NKE', 'DIS', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC', 'CSCO',
    'GLW', 'NOK', 'WDC', 'STX', 'COHR', 'T', 'VZ', 'TXN', 'MU', 'AMAT',
]
TC_PER_SIDE = 0.0015   # 0.15% per trade (realistic for mid-large cap ETF-style execution)


def _download(universe: list, period: str) -> tuple:
    tickers = list(set(universe + ['SPY']))
    print(f"  [strat_bt] Downloading {len(tickers)} tickers ({period})...")
    raw = yf.download(tickers, period=period, interval='1d',
                      auto_adjust=True, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    monthly = close.resample('ME').last()
    monthly = monthly.dropna(axis=1, thresh=int(len(monthly) * 0.6))

    spy_monthly = None
    if 'SPY' in monthly.columns:
        spy_monthly = monthly['SPY'].copy()
        monthly = monthly.drop(columns=['SPY'])

    # Remove non-universe names that slipped in
    monthly = monthly[[c for c in monthly.columns if c in universe]]
    print(f"  [strat_bt] {monthly.shape[1]} universe names, {monthly.shape[0]} months")
    return monthly, spy_monthly


def _composite(monthly: pd.DataFrame, loc: int) -> pd.Series:
    """Compute composite score at index loc (point-in-time, uses data up to loc-1)."""
    if loc < 13:
        return pd.Series(dtype=float)
    skip = monthly.iloc[loc - 1]    # 1-month-ago price (skip last month for 12-1)
    p12  = monthly.iloc[loc - 13]   # 12 months ago
    p6   = monthly.iloc[loc - 7]    # 6 months ago
    mom_12_1 = skip / p12 - 1
    mom_6m   = skip / p6  - 1
    return (0.6 * mom_12_1 + 0.4 * mom_6m).dropna()


def run_strategy(universe: list = None, period: str = '6y',
                 top_pct: float = 0.20, tc_per_side: float = TC_PER_SIDE) -> dict:
    universe = universe or UNIVERSE
    monthly, spy = _download(universe, period)

    port_rets, bench_rets, dates_list = [], [], []
    prev_holdings: set = set()

    for i in range(13, len(monthly)):
        t     = monthly.index[i]
        t_lag = monthly.index[i - 1]

        # ── 1. Compute this month's return on LAST period's holdings ──
        if prev_holdings:
            p0 = monthly.loc[t_lag, list(prev_holdings)].dropna()
            p1 = monthly.loc[t, list(prev_holdings)].dropna()
            common = p0.index.intersection(p1.index)
            port_ret = float((p1[common] / p0[common] - 1).mean()) if len(common) > 0 else 0.0
        else:
            port_ret = 0.0

        # ── 2. Select new holdings (at month-end t) ──
        scores = _composite(monthly, i).dropna()
        n_pick = max(1, int(len(scores) * top_pct))
        new_holdings = set(scores.nlargest(n_pick).index.tolist())

        # ── 3. Transaction cost on turnover ──
        if prev_holdings:
            sold   = prev_holdings - new_holdings
            bought = new_holdings  - prev_holdings
            turnover_pct = (len(sold) + len(bought)) / 2 / max(len(new_holdings), 1)
            tc = min(turnover_pct * 2 * tc_per_side, 0.012)
        else:
            tc = 0.0

        net_ret = port_ret - tc

        # ── 4. Benchmark return ──
        bench_ret = 0.0
        if spy is not None and t_lag in spy.index and t in spy.index:
            sv0, sv1 = spy.loc[t_lag], spy.loc[t]
            if sv0 and sv0 > 0:
                bench_ret = float(sv1 / sv0 - 1)

        port_rets.append(net_ret)
        bench_rets.append(bench_ret)
        dates_list.append(t)
        prev_holdings = new_holdings

    if not port_rets:
        return {'error': 'Insufficient data for strategy simulation'}

    port_s  = pd.Series(port_rets,  index=dates_list)
    bench_s = pd.Series(bench_rets, index=dates_list)

    # ── Equity curves (indexed to 100) ──
    eq = (1 + port_s).cumprod()  * 100
    bq = (1 + bench_s).cumprod() * 100

    # ── Drawdown ──
    dd    = (eq / eq.cummax()) - 1
    max_dd = float(dd.min())
    trough_date = dd.idxmin()
    peak_date   = eq[:trough_date].idxmax()
    recovery_ser = eq[trough_date:][eq[trough_date:] >= float(eq[peak_date])]
    recovery_date = str(recovery_ser.index[0].date()) if not recovery_ser.empty else 'Not yet recovered'

    # ── Return statistics ──
    n_months = len(port_s)
    cagr     = float(eq.iloc[-1] / 100) ** (12 / n_months) - 1
    mean_m   = float(port_s.mean())
    std_m    = float(port_s.std(ddof=1))
    sharpe   = (mean_m / std_m * np.sqrt(12)) if std_m > 0 else 0.0

    ds       = port_s[port_s < 0]
    ds_std   = float(ds.std(ddof=1)) if len(ds) > 1 else std_m
    sortino  = (mean_m / ds_std * np.sqrt(12)) if ds_std > 0 else 0.0
    calmar   = abs(cagr / max_dd) if max_dd != 0 else 0.0
    win_rate = float((port_s > 0).mean())

    # ── Benchmark stats ──
    bq_cagr  = float(bq.iloc[-1] / 100) ** (12 / n_months) - 1
    bq_mean  = float(bench_s.mean())
    bq_std   = float(bench_s.std(ddof=1))
    bq_sharpe = (bq_mean / bq_std * np.sqrt(12)) if bq_std > 0 else 0.0
    bq_dd    = float(((bq / bq.cummax()) - 1).min())

    # ── Annual returns ──
    ann_strat = {}
    for yr, grp in port_s.groupby(port_s.index.year):
        ann_strat[int(yr)] = round(float((1 + grp).prod() - 1), 4)
    ann_bench = {}
    for yr, grp in bench_s.groupby(bench_s.index.year):
        ann_bench[int(yr)] = round(float((1 + grp).prod() - 1), 4)

    worst_yr = min(ann_strat.values()) if ann_strat else None
    best_yr  = max(ann_strat.values()) if ann_strat else None

    roll12 = port_s.rolling(12).apply(lambda x: float((1 + x).prod()) - 1, raw=False)
    worst_12m = float(roll12.min()) if not roll12.dropna().empty else None
    best_12m  = float(roll12.max()) if not roll12.dropna().empty else None

    return {
        'n_months': n_months,
        'n_names': monthly.shape[1],
        'top_pct': top_pct,
        'tc_per_side': tc_per_side,
        'period_start': str(dates_list[0].date()),
        'period_end':   str(dates_list[-1].date()),
        # Strategy
        'cagr':            round(cagr, 4),
        'sharpe':          round(sharpe, 2),
        'sortino':         round(sortino, 2),
        'max_drawdown':    round(max_dd, 4),
        'calmar':          round(calmar, 2),
        'win_rate_monthly': round(win_rate, 3),
        'worst_year':      round(worst_yr, 4) if worst_yr is not None else None,
        'best_year':       round(best_yr, 4)  if best_yr  is not None else None,
        'worst_12m_rolling': round(worst_12m, 4) if worst_12m is not None else None,
        'best_12m_rolling':  round(best_12m, 4)  if best_12m  is not None else None,
        'max_dd_peak':     str(peak_date.date()),
        'max_dd_trough':   str(trough_date.date()),
        'max_dd_recovery': recovery_date,
        # Benchmark
        'benchmark_cagr':   round(bq_cagr, 4),
        'benchmark_sharpe': round(bq_sharpe, 2),
        'benchmark_max_dd': round(bq_dd, 4),
        'active_return':    round(cagr - bq_cagr, 4),
        # Annual
        'annual_returns':   ann_strat,
        'benchmark_annual': ann_bench,
        # Curves
        'equity_curve':     {str(k.date()): round(float(v), 2) for k, v in eq.items()},
        'benchmark_curve':  {str(k.date()): round(float(v), 2) for k, v in bq.items()},
        'drawdown_curve':   {str(k.date()): round(float(v) * 100, 2) for k, v in dd.items()},
    }


def save_json(r: dict) -> str:
    path = os.path.join(REPORTS_DIR, 'strategy_backtest.json')
    slim = {k: v for k, v in r.items()
            if k not in ('equity_curve', 'benchmark_curve', 'drawdown_curve')}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, indent=2)
    print(f"  [ok] Stats saved -> {path}")

    # Also save curve data so the unified dashboard can embed charts inline
    curves_path = os.path.join(REPORTS_DIR, 'strategy_backtest_curves.json')
    curves = {
        'dates':     list(r.get('equity_curve', {}).keys()),
        'equity':    list(r.get('equity_curve', {}).values()),
        'benchmark': list(r.get('benchmark_curve', {}).values()),
        'drawdown':  list(r.get('drawdown_curve', {}).values()),
        'annual_returns':   r.get('annual_returns', {}),
        'benchmark_annual': r.get('benchmark_annual', {}),
    }
    with open(curves_path, 'w', encoding='utf-8') as f:
        json.dump(curves, f)
    print(f"  [ok] Curves saved -> {curves_path}")
    return path


def generate_html(r: dict) -> str:
    if r.get('error'):
        return ''

    eq  = r['equity_curve']
    bq  = r['benchmark_curve']
    dd  = r['drawdown_curve']
    ann = r.get('annual_returns', {})
    b_ann = r.get('benchmark_annual', {})

    dates_j  = json.dumps(list(eq.keys()))
    eq_j     = json.dumps(list(eq.values()))
    bq_j     = json.dumps(list(bq.values()))
    dd_j     = json.dumps(list(dd.values()))
    yrs_j    = json.dumps([str(y) for y in sorted(ann.keys())])
    strat_j  = json.dumps([round(ann.get(y, 0) * 100, 1) for y in sorted(ann.keys())])
    bench_j  = json.dumps([round(b_ann.get(y, 0) * 100, 1) for y in sorted(ann.keys())])

    def pct(v): return f"{v*100:+.1f}%" if v is not None else "—"
    def f2(v):  return f"{v:.2f}" if v is not None else "—"

    cagr_c  = '#27ae60' if (r['cagr'] or 0) > 0 else '#e74c3c'
    dd_c    = '#e74c3c' if (r['max_drawdown'] or 0) < -0.15 else '#e67e22' if (r['max_drawdown'] or 0) < -0.08 else '#27ae60'
    sharpe_c = '#27ae60' if (r['sharpe'] or 0) > 1.0 else '#e67e22' if (r['sharpe'] or 0) > 0.5 else '#e74c3c'
    act_c   = '#27ae60' if (r['active_return'] or 0) > 0 else '#e74c3c'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Strategy Backtest — Equity Curve</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f7;color:#1a1a2e}}
.top-bar{{background:#0d3b6e;color:white;padding:14px 28px}}
.top-bar h1{{font-size:14pt}}
.top-bar .sub{{font-size:8.5pt;opacity:0.75;margin-top:3px}}
.container{{max-width:1200px;margin:0 auto;padding:20px}}
.stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px}}
.stat{{background:white;border-radius:8px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.stat .lbl{{font-size:7.5pt;color:#6b7280;text-transform:uppercase;letter-spacing:.5px}}
.stat .val{{font-size:21pt;font-weight:700;margin-top:3px}}
.stat .sub{{font-size:7.8pt;color:#9ca3af;margin-top:2px}}
.card{{background:white;border-radius:8px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:18px}}
.card h3{{font-size:10.5pt;color:#0d3b6e;margin-bottom:12px}}
.two{{display:grid;grid-template-columns:2fr 1fr;gap:16px}}
.warn{{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:4px;margin-bottom:16px;font-size:8.5pt;line-height:1.7}}
.info{{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:4px;margin-bottom:16px;font-size:8.5pt;line-height:1.7}}
</style></head><body>
<div class="top-bar">
  <h1>Strategy Backtest — Momentum Top Quintile (Price Factors Only)</h1>
  <div class="sub">Universe: {r['n_names']} names &middot; {r['n_months']} months ({r['period_start']} &rarr; {r['period_end']}) &middot; TC {r['tc_per_side']*100:.2f}%/side &middot; Long-only monthly rebalance</div>
</div>
<div class="container">

<div class="warn">
  <strong>Scope:</strong> Validates the <em>price momentum</em> component only (composite = 0.6&times;12-1 mom + 0.4&times;6m mom, top {r['top_pct']*100:.0f}% of universe). Revision momentum, PEAD, short interest and quality are validated via the live track record (snapshot.py). This is the <strong>floor</strong> — the complete strategy should do better once all signals accumulate live history.
</div>

<div class="stats-grid">
  <div class="stat"><div class="lbl">Strategy CAGR</div><div class="val" style="color:{cagr_c}">{pct(r['cagr'])}</div><div class="sub">SPY: {pct(r['benchmark_cagr'])}</div></div>
  <div class="stat"><div class="lbl">Max Drawdown</div><div class="val" style="color:{dd_c}">{pct(r['max_drawdown'])}</div><div class="sub">{r['max_dd_peak']} &rarr; {r['max_dd_trough']}</div></div>
  <div class="stat"><div class="lbl">Sharpe (Ann.)</div><div class="val" style="color:{sharpe_c}">{f2(r['sharpe'])}</div><div class="sub">Sortino: {f2(r['sortino'])}</div></div>
  <div class="stat"><div class="lbl">Active Return</div><div class="val" style="color:{act_c}">{pct(r['active_return'])}</div><div class="sub">Win rate: {r['win_rate_monthly']*100:.1f}% months</div></div>
  <div class="stat"><div class="lbl">Calmar Ratio</div><div class="val">{f2(r['calmar'])}</div><div class="sub">CAGR / |MaxDD|</div></div>
  <div class="stat"><div class="lbl">Best Year</div><div class="val" style="color:#27ae60">{pct(r['best_year'])}</div><div class="sub">Best 12m rolling: {pct(r['best_12m_rolling'])}</div></div>
  <div class="stat"><div class="lbl">Worst Year</div><div class="val" style="color:#e74c3c">{pct(r['worst_year'])}</div><div class="sub">Worst 12m rolling: {pct(r['worst_12m_rolling'])}</div></div>
  <div class="stat"><div class="lbl">DD Recovery</div><div class="val" style="font-size:11pt;padding-top:6px">{r['max_dd_recovery']}</div><div class="sub">From trough</div></div>
</div>

<div class="card"><h3>Equity Curve (Indexed to 100)</h3><canvas id="eqC" height="70"></canvas></div>
<div class="two">
  <div class="card"><h3>Annual Returns vs SPY</h3><canvas id="annC" height="150"></canvas></div>
  <div class="card"><h3>Drawdown (%)</h3><canvas id="ddC" height="150"></canvas></div>
</div>

<div class="info">
  <strong>What this means for position sizing:</strong> The worst drawdown is <strong>{pct(r['max_drawdown'])}</strong> over this period.
  On a $1M portfolio at 100% of normal sizing, that's a peak-to-trough loss of <strong>${abs(r['max_drawdown'])*1e6:,.0f}</strong> that you'd have to sit through.
  The regime filter (regime.py) reduces exposure in bad tapes to limit this in practice.
  Recovery took until {r['max_dd_recovery']}.
  Calmar {f2(r['calmar'])} means every 1% of drawdown risk earned {f2(r['calmar'])}% of CAGR — above 0.5 is acceptable for a long-only strategy.
</div>

</div>
<script>
const dates={dates_j}, eq={eq_j}, bq={bq_j}, dd={dd_j};
const yrs={yrs_j}, strat={strat_j}, bench={bench_j};
new Chart(document.getElementById('eqC'),{{type:'line',data:{{labels:dates,datasets:[
  {{label:'Strategy',data:eq,borderColor:'#0d3b6e',backgroundColor:'rgba(13,59,110,.06)',borderWidth:2,pointRadius:0,tension:.2}},
  {{label:'SPY',data:bq,borderColor:'#9ca3af',borderDash:[4,2],borderWidth:1.5,pointRadius:0,tension:.2}}
]}},options:{{plugins:{{legend:{{position:'top'}}}},scales:{{x:{{ticks:{{maxTicksLimit:12}}}}}}}}}});
new Chart(document.getElementById('annC'),{{type:'bar',data:{{labels:yrs,datasets:[
  {{label:'Strategy',data:strat,backgroundColor:strat.map(v=>v>=0?'rgba(13,59,110,.75)':'rgba(231,76,60,.75)')}},
  {{label:'SPY',data:bench,backgroundColor:'rgba(156,163,175,.5)'}}
]}},options:{{plugins:{{legend:{{position:'top'}}}},scales:{{y:{{ticks:{{callback:v=>v+'%'}}}}}}}}}});
new Chart(document.getElementById('ddC'),{{type:'line',data:{{labels:dates,datasets:[
  {{label:'Drawdown %',data:dd,borderColor:'#e74c3c',backgroundColor:'rgba(231,76,60,.1)',borderWidth:1.5,pointRadius:0,fill:true}}
]}},options:{{plugins:{{legend:{{position:'top'}}}},scales:{{y:{{ticks:{{callback:v=>v+'%'}}}}}}}}}});
</script>
</body></html>"""

    path = os.path.join(REPORTS_DIR, 'strategy_backtest.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [ok] HTML chart -> {path}")
    return path


def print_results(r: dict):
    if r.get('error'):
        print(f"  [error] {r['error']}"); return
    print(f"\n{'='*68}")
    print(f"  STRATEGY BACKTEST — Full-Rule Momentum")
    print(f"  {r['n_names']} names, {r['n_months']} months "
          f"({r['period_start']} -> {r['period_end']})")
    print(f"  Top {r['top_pct']*100:.0f}% per month | TC {r['tc_per_side']*100:.2f}%/side")
    print('='*68)

    def pct(v): return f"{v*100:+.1f}%" if v is not None else "—"
    def f2(v):  return f"{v:.2f}" if v is not None else "—"

    rows = [
        ('CAGR',              pct(r['cagr']),    f"SPY: {pct(r['benchmark_cagr'])}"),
        ('Sharpe (ann)',       f2(r['sharpe']),   f"SPY: {f2(r['benchmark_sharpe'])}"),
        ('Sortino',            f2(r['sortino']),  ''),
        ('Max Drawdown',       pct(r['max_drawdown']), f"SPY: {pct(r['benchmark_max_dd'])}"),
        ('Calmar Ratio',       f2(r['calmar']),   ''),
        ('Win Rate (monthly)', f"{r['win_rate_monthly']*100:.1f}%", ''),
        ('Best Year',          pct(r['best_year']),  ''),
        ('Worst Year',         pct(r['worst_year']), ''),
        ('Active Return',      pct(r['active_return']), ''),
    ]
    for lbl, val, note in rows:
        print(f"  {lbl:<26} {val:<12} {note}")

    print(f"\n  Max DD period: {r['max_dd_peak']} peak → {r['max_dd_trough']} trough")
    print(f"  Recovery: {r['max_dd_recovery']}")
    print('\n  Annual Returns:')
    print(f"  {'Year':<6} {'Strategy':>9} {'SPY':>9}  Bar")
    print("  " + "-"*40)
    for yr in sorted(r['annual_returns']):
        st = r['annual_returns'][yr]
        bn = r.get('benchmark_annual', {}).get(yr)
        bar_len = int(abs(st) * 100)
        bar = ('▲' if st > 0 else '▼') + '█' * min(bar_len, 25)
        print(f"  {yr:<6} {pct(st):>9} {pct(bn):>9}  {bar}")
    print('='*68)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', default='6y', help='e.g. 6y, 10y')
    parser.add_argument('--top',    type=float, default=0.20, help='Top fraction (0.20 = top 20%%)')
    parser.add_argument('--tc',     type=float, default=TC_PER_SIDE, help='TC per side (0.0015)')
    parser.add_argument('tickers',  nargs='*', help='Custom universe (optional)')
    args = parser.parse_args()

    universe = [t.upper() for t in args.tickers] if args.tickers else UNIVERSE
    results = run_strategy(universe, period=args.period, top_pct=args.top, tc_per_side=args.tc)
    print_results(results)
    if not results.get('error'):
        save_json(results)
        html_path = generate_html(results)
        if html_path:
            print(f"\nOpen: file:///{html_path.replace(os.sep, '/')}")
