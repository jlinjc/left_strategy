"""
Dip-Buy & HOLD vs DCA  (只選進場點、在大跌恐慌時勇敢買、永不出場 — 進場點好就贏得了 DCA 嗎?)
═══════════════════════════════════════════════════════════════════════════════
User's refined (and sharp) hypothesis: the reason left-side lost before was SELLING (capping
winners / calling tops). So remove selling entirely — ONLY optimise the ENTRY: hold cash,
deploy it into big drawdowns (panic), and NEVER sell. A better average entry price *should*
beat dollar-cost-averaging. Test it honestly, indices only (QQQ / SPY / SOXX).

Fair setup: identical $X/month contributions for every strategy; cash earns a money-market
yield (so dry powder isn't dead money — fair to the dip-buyer); nobody ever sells.
    DCA          — invest each month's cash immediately, hold forever.
    DIP(-d%)     — park contributions in cash; when price is ≥ d% below its all-time high,
                   deploy ALL accumulated cash (rearm after it recovers halfway); hold forever.

Reports per index: final wealth, IRR (money-weighted), max drawdown, and avg % held in cash
(the drag). Verdict: did ANY dip threshold beat DCA's IRR?

Usage:  python dip_hold_test.py --monthly 1000 --rf 0.02
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
TICKERS = ['QQQ', 'SPY', 'SOXX']
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]


def _irr(contribs, final, n_months):
    flows = [(m, -a) for m, a in contribs] + [(n_months, final)]
    def npv(r): return sum(cf / (1 + r) ** m for m, cf in flows)
    lo, hi = -0.95, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0: lo = mid
        else: hi = mid
    return (1 + (lo + hi) / 2) ** 12 - 1


def _maxdd(eq):
    return float((eq / eq.cummax() - 1).min())


def simulate(px, monthly, rf, mode, thr=0.0):
    rf_d = (1 + rf) ** (1 / 252) - 1
    dates = px.index
    mkeys = pd.Series(dates).dt.to_period('M').values
    dd = px / px.cummax() - 1.0
    shares = cash = contributed = 0.0
    contribs = []; mi = -1; prev = None; armed = True
    cash_w = []; eq = []
    for k in range(len(dates)):
        p = float(px.iloc[k]); cash *= (1 + rf_d)
        mk = mkeys[k]
        if mk != prev:
            cash += monthly; contributed += monthly; mi += 1
            contribs.append((mi, monthly)); prev = mk
            if mode == 'DCA':
                shares += cash / p; cash = 0.0
        if mode == 'DIP':
            d = float(dd.iloc[k])
            if d > -thr / 2:        # recovered halfway → re-arm for the next distinct dip
                armed = True
            if d <= -thr and armed and cash > 0:
                shares += cash / p; cash = 0.0; armed = False
        val = shares * p + cash
        eq.append(val); cash_w.append(cash / val if val > 0 else 0)
    eqs = pd.Series(eq, index=dates)
    final = float(eqs.iloc[-1])
    return {'final': round(final, 0), 'contributed': round(contributed, 0),
            'irr': round(_irr(contribs, final, mi + 1) * 100, 2),
            'maxdd': round(_maxdd(eqs) * 100, 1),
            'avg_cash_pct': round(float(np.mean(cash_w)) * 100, 1)}


def run(monthly, rf):
    out = {}
    for tk in TICKERS:
        px = yf.download(tk, period='max', interval='1d', auto_adjust=True, progress=False)['Close']
        if hasattr(px, 'columns'): px = px.iloc[:, 0]
        px = px.dropna()
        rows = {'DCA(每月即投)': simulate(px, monthly, rf, 'DCA')}
        for thr in THRESHOLDS:
            rows[f'DIP −{int(thr*100)}%'] = simulate(px, monthly, rf, 'DIP', thr)
        bh_mult = float(px.iloc[-1] / px.iloc[0])
        out[tk] = {'start': str(px.index[0].date()), 'end': str(px.index[-1].date()),
                   'buyhold_mult': round(bh_mult, 1), 'rows': rows}
    return {'monthly': monthly, 'rf': rf, 'tickers': out}


def print_report(o):
    for tk, d in o['tickers'].items():
        print("\n" + "=" * 78)
        print(f"  {tk} — 只選進場點、永不賣 vs 定期定額   ({d['start']} → {d['end']})")
        print(f"  每月 ${o['monthly']:,.0f} · 現金 {o['rf']*100:.1f}%/年 · 一次買進長抱總倍數 ×{d['buyhold_mult']}")
        print("=" * 78)
        print(f"  {'策略':<16}{'最終資產':>14}{'年化IRR':>10}{'最大回撤':>10}{'平均持現金':>11}")
        print("  " + "-" * 74)
        dca = d['rows']['DCA(每月即投)']['irr']
        for k, m in d['rows'].items():
            flag = ''
            if k != 'DCA(每月即投)':
                flag = '  ✓贏DCA' if m['irr'] > dca else '  ✗輸DCA'
            print(f"  {k:<16}{m['final']:>14,.0f}{m['irr']:>9.2f}%{m['maxdd']:>9.1f}%{m['avg_cash_pct']:>10.1f}%{flag}")
        print("=" * 78)
    print("\n  讀法:看有沒有任何『DIP −d%』的年化IRR > DCA。若沒有 → 即使『不賣、只優化進場』,")
    print("       等大跌的現金拖累仍 > 進場折扣的好處(這就是為什麼擇時難贏長抱)。")
    print("       同時看『最大回撤』與『平均持現金』:等越深、抱越多現金 = 報酬越低但波動略緩。")


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--monthly', type=float, default=1000.0)
    ap.add_argument('--rf', type=float, default=0.02)
    args = ap.parse_args()
    res = run(args.monthly, args.rf)
    print_report(res)
    with open(os.path.join(REPORTS_DIR, 'dip_hold_test.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
