"""
Fear→Greed Cycle Swing  (使用者的論點:買在全體投降、抱住復甦、賣在全民狂歡 — 能打敗 DCA 嗎?)
═══════════════════════════════════════════════════════════════════════════════
The earlier panic test BOUGHT into fear but SOLD at the 20dma bounce (a few days) — so it
threw away the whole recovery and lost to DCA. The user's actual thesis is different and
better: buy CAPITULATION, HOLD through the entire recovery, and only sell at EUPHORIA
(everyone optimistic). That captures the full fear→greed expansion AND sidesteps the next
crash — which is the one version of contrarian trading with a real shot at beating DCA.

This back-tests it FAIRLY against DCA, same monthly contributions, on SPY (cash earns rf):
    DCA          — invest every month, never sell. (benchmark)
    PANIC_HOLD   — hold dry powder, deploy on capitulation, NEVER sell (pure buy-timing).
    CYCLE        — buy capitulation, sell to cash at euphoria, rebuy next capitulation.
    CYCLE_TRAIL  — buy capitulation; after euphoria flagged, exit on a trailing stop
                   (let winners run instead of calling the exact top); rebuy capitulation.

Signals (deliberately simple, to avoid curve-fitting):
    CAPITULATION : VIX ≥ 28  OR  SPY ≤ 0.85 × its trailing-252d high   (rearm when calm)
    EUPHORIA     : VIX ≤ 14  AND  SPY ≥ 1.10 × its 200dma              (calm + extended)

Usage:  python fear_greed_cycle.py --period 16y --monthly 1000 --rf 0.02
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')


def _load(period, ticker='SPY'):
    spy = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(spy, 'columns'): spy = spy.iloc[:, 0]
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'): vix = vix.iloc[:, 0]
    df = pd.DataFrame({'spy': spy, 'vix': vix}).dropna()
    df['hi252'] = df['spy'].rolling(252, min_periods=60).max()
    df['ma200'] = df['spy'].rolling(200, min_periods=60).mean()
    return df.dropna()


def _max_dd(equity):
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def _irr(contribs, final, n_months):
    flows = [(m, -a) for m, a in contribs] + [(n_months, final)]
    def npv(r): return sum(cf / (1 + r) ** m for m, cf in flows)
    lo, hi = -0.9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0: lo = mid
        else: hi = mid
    return (1 + (lo + hi) / 2) ** 12 - 1


def simulate(df, monthly, rf, mode, cap_vix=28, cap_dd=0.85, euph_vix=14, euph_ext=1.10,
             trail=0.10, reserve=0.20, lever=0.50, borrow_spread=0.03):
    rf_d = (1 + rf) ** (1 / 252) - 1
    bor_d = (1 + rf + borrow_spread) ** (1 / 252) - 1
    dates = df.index
    mkeys = pd.Series(dates).dt.to_period('M').values
    shares = cash = contributed = 0.0
    debt = 0.0                          # for LEVER_DIP
    lever_entry = None
    contribs = []; mi = -1; prev = None
    buy_armed = True; euph_seen = False; peak_since = 0.0
    eq = []
    for k, dt in enumerate(dates):
        px = float(df['spy'].iloc[k]); v = float(df['vix'].iloc[k])
        hi = float(df['hi252'].iloc[k]); ma = float(df['ma200'].iloc[k])
        cash *= (1 + rf_d); debt *= (1 + bor_d)
        mk = mkeys[k]
        if mk != prev:
            cash += monthly; contributed += monthly; mi += 1
            contribs.append((mi, monthly)); prev = mk
            if mode in ('DCA', 'LEVER_DIP'):
                shares += cash / px; cash = 0.0          # stay fully invested
            elif mode == 'LEAN_RESERVE':
                invest = cash * (1 - reserve)            # invest most, keep a small reserve
                shares += invest / px; cash -= invest
        capit = (v >= cap_vix) or (px <= cap_dd * hi)
        euph = (v <= euph_vix) and (px >= euph_ext * ma)
        if not capit and v < 18:
            buy_armed = True
        # ── buy-the-capitulation behaviours ──
        if mode in ('PANIC_HOLD', 'CYCLE', 'CYCLE_TRAIL', 'LEAN_RESERVE') and capit and buy_armed and cash > 0:
            shares += cash / px; cash = 0.0; buy_armed = False; euph_seen = False; peak_since = px
        if mode == 'LEVER_DIP' and capit and buy_armed and debt == 0:
            borrow = lever * (shares * px - debt)        # borrow vs current equity
            if borrow > 0:
                shares += borrow / px; debt += borrow; lever_entry = px; buy_armed = False
        if shares > 0:
            peak_since = max(peak_since, px)
        # ── exits ──
        if mode == 'CYCLE' and euph and shares > 0:
            cash += shares * px; shares = 0.0
        if mode == 'CYCLE_TRAIL':
            if euph: euph_seen = True
            if euph_seen and shares > 0 and px <= peak_since * (1 - trail):
                cash += shares * px; shares = 0.0; euph_seen = False
        if mode == 'LEVER_DIP' and debt > 0 and lever_entry and px >= lever_entry * 1.15:
            repay = min(debt, shares * px)               # recovered → delever (lock the lean-in gain)
            shares -= repay / px; debt -= repay; lever_entry = None; buy_armed = True
        eq.append(shares * px + cash - debt)
    eqs = pd.Series(eq, index=dates)
    n_months = mi + 1
    final = float(eqs.iloc[-1])
    rd = eqs.pct_change().dropna()
    sharpe = float(rd.mean() / rd.std() * np.sqrt(252)) if rd.std() > 0 else 0.0
    invested = (eqs - pd.Series([c for c in eqs], index=dates))  # not used
    pct_in = float(((pd.Series(eq, index=dates).diff().abs() >= 0)).mean())  # placeholder
    return {
        'final': round(final, 0), 'contributed': round(contributed, 0),
        'profit_pct': round((final / contributed - 1) * 100, 1) if contributed else 0,
        'irr': round(_irr(contribs, final, n_months) * 100, 2),
        'sharpe': round(sharpe, 2), 'max_dd': round(_max_dd(eqs) * 100, 1),
    }


def run(period, monthly, rf, ticker='SPY'):
    df = _load(period, ticker)
    modes = ['DCA', 'PANIC_HOLD', 'CYCLE', 'CYCLE_TRAIL', 'LEAN_RESERVE', 'LEVER_DIP']
    rows = {m: simulate(df, monthly, rf, m) for m in modes}
    # Buy-and-hold lump sum (invest 1 unit at start, hold) for return/drawdown context
    px = df['spy']
    bh_mult = float(px.iloc[-1] / px.iloc[0])
    bh_cagr = bh_mult ** (252 / len(px)) - 1
    bh_dd = _max_dd(px)
    return {'ticker': ticker, 'period': period, 'monthly': monthly, 'rf': rf,
            'start': str(df.index[0].date()), 'end': str(df.index[-1].date()),
            'buyhold': {'mult': round(bh_mult, 2), 'cagr': round(bh_cagr * 100, 2),
                        'max_dd': round(bh_dd * 100, 1)},
            'rows': rows}


def print_report(o):
    print("\n" + "=" * 84)
    print(f"  恐懼→貪婪 完整週期 vs 定期定額/長抱 — 標的:{o['ticker']}")
    print(f"  {o['start']} → {o['end']} · 每月 ${o['monthly']:,.0f} · 現金 {o['rf']*100:.1f}%/年")
    bh = o['buyhold']
    print(f"  [買進持有 lump-sum 參考] 總報酬 ×{bh['mult']} · CAGR {bh['cagr']}% · 最大回撤 {bh['max_dd']}%")
    print("=" * 84)
    print(f"  {'策略':<14}{'最終資產':>14}{'獲利%':>9}{'年化IRR':>10}{'Sharpe':>9}{'最大回撤':>10}")
    print("  " + "-" * 80)
    desc = {'DCA': '定期定額(基準)', 'PANIC_HOLD': '只買投降·不賣',
            'CYCLE': '買投降·賣狂歡', 'CYCLE_TRAIL': '買投降·狂歡後移動停利',
            'LEAN_RESERVE': '幾乎全投+小銀彈砸崩盤', 'LEVER_DIP': '崩盤用槓桿加重(50%)'}
    for m, r in o['rows'].items():
        print(f"  {desc[m]:<14}{r['final']:>14,.0f}{r['profit_pct']:>8.1f}%"
              f"{r['irr']:>9.2f}%{r['sharpe']:>9.2f}{r['max_dd']:>9.1f}%")
    print("=" * 84)
    dca = o['rows']['DCA']
    winners = [(m, r) for m, r in o['rows'].items() if r['irr'] > dca['irr']]
    if winners:
        for m, r in winners:
            print(f"  ✓ {desc[m]} 打敗 DCA:IRR {r['irr']}% vs {dca['irr']}% · 回撤 {r['max_dd']}% vs {dca['max_dd']}%")
    else:
        print(f"  ✗ 沒有任何週期版本在 IRR 上打敗 DCA({dca['irr']}%)。")
    print("  公正提醒:此期間若仍是強多頭,擇時等跌天生吃虧;真正關鍵看『最大回撤』有沒有顯著變淺")
    print("           (用較少報酬換睡得著覺,也是一種勝利)。")
    print("=" * 84)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='16y')
    ap.add_argument('--monthly', type=float, default=1000.0)
    ap.add_argument('--rf', type=float, default=0.02)
    ap.add_argument('--ticker', default='SPY')
    args = ap.parse_args()
    res = run(args.period, args.monthly, args.rf, args.ticker.upper())
    print_report(res)
    fn = f"fear_greed_cycle_{args.ticker.upper()}.json"
    with open(os.path.join(REPORTS_DIR, fn), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
