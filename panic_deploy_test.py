"""
Panic-Deployment Test  (投資人視角:在恐慌加碼,到底比定期定額多賺嗎?— 公正版)
═══════════════════════════════════════════════════════════════════════════════
Phase 0/1 measured the strategy as an always-on book. But an INVESTOR doesn't run a book —
an investor has cash arriving over time (salary, contributions, dry powder) and the only
real decision is WHEN to deploy it. So the fair, investor-relevant question is NOT
"strategy vs buy-and-hold SPY" (unfair — the tool sits in cash a lot). It is:

    "If I add the same $ every month, does HOLDING CASH AND DEPLOYING INTO PANIC (VIX spike)
     actually end me with more money — and less pain — than just dollar-cost-averaging?"

This tests exactly that, on SPY (isolating the TIMING edge of 'buy the fear', without
single-stock survivorship noise). Cash earns a modest money-market yield so the comparison
is fair to the panic strategy (dry powder isn't dead money).

Strategies (identical monthly contribution):
    DCA        — invest each month's contribution immediately into SPY.
    PANIC(T)   — park each contribution in cash (earning rf); when VIX first crosses ABOVE
                 T (re-armed after it drops back below), deploy ALL accumulated cash into SPY.
    PANIC_TIER — deploy in tranches at VIX>22 / >28 / >35 (graduated fear).

Reports: final wealth, profit on identical contributions, money-weighted IRR, equity-curve
Sharpe, and MAX DRAWDOWN (the pain). Honest verdict printed.

Usage:  python panic_deploy_test.py --period 15y --monthly 1000
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')


def _load(period):
    spy = yf.download('SPY', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(spy, 'columns'): spy = spy.iloc[:, 0]
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'): vix = vix.iloc[:, 0]
    df = pd.DataFrame({'spy': spy, 'vix': vix}).dropna()
    return df


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _irr_monthly(cashflows: list, final_value: float, n_months: int):
    """Money-weighted IRR (annualised). cashflows: list of (month_index, amount_out) contributions
    (negative = invested). final_value received at n_months. Solve NPV=0 over monthly rate."""
    flows = [(m, -amt) for m, amt in cashflows]      # contributions are outflows
    flows.append((n_months, final_value))            # terminal inflow
    def npv(r):
        return sum(cf / (1 + r) ** m for m, cf in flows)
    lo, hi = -0.9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0: lo = mid
        else: hi = mid
    monthly = (lo + hi) / 2
    return (1 + monthly) ** 12 - 1


def simulate(df, monthly, rf_annual, mode, thresholds):
    """Walk daily. On the first trading day of each month, add `monthly` to cash.
    Deploy per the mode. Returns (equity_series, total_contributed, contributions_list)."""
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    dates = df.index
    month_keys = pd.Series(dates).dt.to_period('M').values
    shares = 0.0
    cash = 0.0
    contributed = 0.0
    contribs = []           # (month_index, amount) for IRR
    armed = {t: True for t in thresholds}   # re-arm flags for PANIC tiers
    equity = []
    prev_mk = None
    mi = -1
    for k, dt in enumerate(dates):
        mk = month_keys[k]
        price = float(df['spy'].iloc[k]); v = float(df['vix'].iloc[k])
        cash *= (1 + rf_daily)                      # cash earns money-market
        if mk != prev_mk:                            # first trading day of a new month → contribute
            cash += monthly; contributed += monthly; mi += 1
            contribs.append((mi, monthly)); prev_mk = mk
            if mode == 'DCA':
                shares += cash / price; cash = 0.0
        if mode == 'PANIC':
            T = thresholds[0]
            if v < T:
                armed[T] = True
            if v >= T and armed[T] and cash > 0:
                shares += cash / price; cash = 0.0; armed[T] = False
        elif mode == 'PANIC_TIER':
            # deploy a fraction of CURRENT cash at each tier when crossed (re-armed below tier)
            for T, frac in zip(thresholds, (0.5, 0.75, 1.0)):
                if v < T:
                    armed[T] = True
                if v >= T and armed[T] and cash > 0:
                    deploy = cash * frac
                    shares += deploy / price; cash -= deploy; armed[T] = False
        equity.append(shares * price + cash)
    eq = pd.Series(equity, index=dates)
    return eq, contributed, contribs


def run(period, monthly, rf):
    df = _load(period)
    n_days = len(df); n_months = int(np.ceil(n_days / 21))
    configs = {
        'DCA':            ('DCA', [0]),
        'PANIC(VIX>22)':  ('PANIC', [22.0]),
        'PANIC(VIX>28)':  ('PANIC', [28.0]),
        'PANIC(VIX>35)':  ('PANIC', [35.0]),
        'PANIC_TIER':     ('PANIC_TIER', [22.0, 28.0, 35.0]),
    }
    rows = {}
    for name, (mode, thr) in configs.items():
        eq, contrib, contribs = simulate(df, monthly, rf, mode, thr)
        final = float(eq.iloc[-1])
        ret_daily = eq.pct_change().dropna()
        sharpe = float(ret_daily.mean() / ret_daily.std() * np.sqrt(252)) if ret_daily.std() > 0 else 0.0
        irr = _irr_monthly(contribs, final, n_months)
        rows[name] = {
            'final': round(final, 0), 'contributed': round(contrib, 0),
            'profit': round(final - contrib, 0),
            'profit_pct': round((final / contrib - 1) * 100, 1) if contrib else 0,
            'irr_annual': round(irr * 100, 2),
            'sharpe': round(sharpe, 2),
            'max_dd': round(_max_drawdown(eq) * 100, 1),
            'end_invested_pct': round(float(eq.iloc[-1] and (1 - 0)) and 100, 0),  # placeholder
        }
    return {'period': period, 'monthly': monthly, 'rf': rf,
            'n_days': n_days, 'n_months': n_months, 'rows': rows,
            'start': str(df.index[0].date()), 'end': str(df.index[-1].date())}


def print_report(o):
    print("\n" + "=" * 88)
    print("  在恐慌加碼 vs 定期定額 — 投資人視角(同樣每月投入,投 SPY,公正對照)")
    print(f"  期間 {o['start']} → {o['end']} ({o['n_months']} 個月) · 每月 ${o['monthly']:,} · 現金利率 {o['rf']*100:.1f}%/年")
    print("=" * 88)
    print(f"  {'策略':<16}{'最終資產':>13}{'累計投入':>12}{'獲利':>12}{'獲利%':>8}{'年化IRR':>9}{'Sharpe':>8}{'最大回撤':>9}")
    print("  " + "-" * 84)
    for k, m in o['rows'].items():
        print(f"  {k:<16}{m['final']:>13,.0f}{m['contributed']:>12,.0f}{m['profit']:>12,.0f}"
              f"{m['profit_pct']:>7.1f}%{m['irr_annual']:>8.2f}%{m['sharpe']:>8.2f}{m['max_dd']:>8.1f}%")
    print("=" * 88)
    dca = o['rows']['DCA']
    best = max(o['rows'].items(), key=lambda kv: kv[1]['irr_annual'])
    print(f"  基準 DCA: IRR {dca['irr_annual']}% · 最大回撤 {dca['max_dd']}%")
    print(f"  最佳:{best[0]} IRR {best[1]['irr_annual']}% · 最大回撤 {best[1]['max_dd']}%")
    print("  讀法:IRR 是錢加權報酬(投資人實拿)。看恐慌加碼有沒有『同時』IRR 更高且回撤更淺;")
    print("       若 IRR 沒贏,代表『等恐慌』的現金拖累 > 撿便宜的好處(這是公正該揭露的)。")
    print("=" * 88)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='15y')
    ap.add_argument('--monthly', type=float, default=1000.0)
    ap.add_argument('--rf', type=float, default=0.02)
    args = ap.parse_args()
    res = run(args.period, args.monthly, args.rf)
    print_report(res)
    with open(os.path.join(REPORTS_DIR, 'panic_deploy_test.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
