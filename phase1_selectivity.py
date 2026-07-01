"""
Phase 1 — Selectivity Test  (砍掉 90% 的交易,能不能把負 alpha 救成正 alpha?)
═══════════════════════════════════════════════════════════════════════════════
Phase 0 proved the COMMODITY core (gated RSI(2) above a rising 200dma) has NEGATIVE alpha
after costs — it is a high-beta long book, and the only place real edge showed up was
high-VIX entries (liquidity provision into fear). So the real question for the strategy's
survival is NOT "add more filters", it is:

    If we trade FAR less — only when (a) the market is genuinely fearful, AND/OR (b) the
    name fell on its OWN account (idiosyncratic, not dragged by the market), AND/OR (c) the
    oversold is EXTREME — does the after-cost alpha turn positive and significant?

All conditions here are POINT-IN-TIME (VIX at signal, beta-adjusted residual from trailing
data, RSI(2)/Bollinger from past prices) so this backtest does NOT cheat. Fundamental gates
are still excluded (they need PIT fundamentals).

Variants (each adds selectivity on top of the same gated RSI(2) core):
    baseline      — commodity core (Phase 0)                          [the trap]
    +fear         — only entries with VIX ≥ 25 at signal
    +idio         — only entries where 20d beta-adjusted residual < −8%
    +extreme      — RSI(2) < 5 AND close below lower Bollinger band
    SELECTIVE     — fear OR idio, AND extreme  (the disciplined book)

Output: per-variant trade stats + a CAPM alpha/beta on the daily book (cash when flat).

Usage:  python phase1_selectivity.py --period 8y --cost-bps 20
"""
from __future__ import annotations
import os, json, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from bottom_fishing_backtest import rsi, atr_series
from attribution import ols_nw, UNIVERSE, REPORTS_DIR

PARAMS = dict(rsi_thr=10.0, exit_ma=20, max_hold=25, stop_atr=3.0)

# Variant entry-condition specs (all point-in-time). None = no constraint on that axis.
VARIANTS = {
    'baseline':  dict(vix_min=None, resid_max=None, rsi_hard=None, below_lband=False, combine='and'),
    '+fear':     dict(vix_min=25.0, resid_max=None, rsi_hard=None, below_lband=False, combine='and'),
    '+idio':     dict(vix_min=None, resid_max=-0.08, rsi_hard=None, below_lband=False, combine='and'),
    '+extreme':  dict(vix_min=None, resid_max=None, rsi_hard=5.0, below_lband=True, combine='and'),
    'SELECTIVE': dict(vix_min=25.0, resid_max=-0.05, rsi_hard=5.0, below_lband=True, combine='fear_or_idio'),
}


def _bollinger_lower(close, period=20, k=2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return ma - k * sd


def simulate_variant(df, spy_close, vix_close, p, cost_bps, cond):
    """Gated RSI(2) core + extra point-in-time entry conditions; returns (trades, daily_ret, held)."""
    if df is None or len(df) < 260:
        return [], pd.Series(dtype=float), pd.Series(dtype=float)
    close = df['Close']; openp = df.get('Open', close); low = df.get('Low', close)
    r2 = rsi(close, 2)
    ma_exit = close.rolling(p['exit_ma']).mean()
    ma200 = close.rolling(200).mean()
    rising = ma200 > ma200.shift(21)
    atr = atr_series(df, 14)
    lband = _bollinger_lower(close)
    idx = df.index; n = len(df)
    cost = cost_bps / 1e4

    # Align SPY + VIX to this name's index (point-in-time, ffill)
    spy = spy_close.reindex(idx).ffill()
    spy_ret = spy.pct_change()
    name_ret = close.pct_change()
    vix = vix_close.reindex(idx).ffill()

    def residual_at(i, L=20, B=120):
        if i < B + 1:
            return None
        r = name_ret.iloc[i - B:i]; b = spy_ret.iloc[i - B:i]
        v = float(b.var())
        if v <= 0 or r.isna().all():
            return None
        beta = float(((r - r.mean()) * (b - b.mean())).mean() / v)
        beta = max(0.0, min(3.0, beta))
        if i - L < 0:
            return None
        sret = float(close.iloc[i] / close.iloc[i - L] - 1)
        bret = float(spy.iloc[i] / spy.iloc[i - L] - 1)
        return sret - beta * bret

    def passes(i):
        # axis flags
        fear = (cond['vix_min'] is None) or (not pd.isna(vix.iloc[i]) and float(vix.iloc[i]) >= cond['vix_min'])
        if cond['resid_max'] is not None:
            rv = residual_at(i)
            idio = (rv is not None and rv < cond['resid_max'])
        else:
            idio = True
        extreme = True
        if cond['rsi_hard'] is not None:
            extreme = extreme and (r2.iloc[i] < cond['rsi_hard'])
        if cond['below_lband']:
            extreme = extreme and (not pd.isna(lband.iloc[i]) and close.iloc[i] < lband.iloc[i])
        if cond['combine'] == 'fear_or_idio':
            # need (fear OR idio) AND extreme
            base = (fear or idio)
            if cond['vix_min'] is None and cond['resid_max'] is None:
                base = True
            return base and extreme
        return fear and idio and extreme

    dret = pd.Series(0.0, index=idx); held = pd.Series(0.0, index=idx)
    trades = []
    i = 200
    while i < n - 1:
        if not (r2.iloc[i] < p['rsi_thr']):
            i += 1; continue
        if not (close.iloc[i] > ma200.iloc[i]) or not bool(rising.iloc[i]) or pd.isna(atr.iloc[i]):
            i += 1; continue
        if not passes(i):
            i += 1; continue
        entry_i = i + 1
        entry_open = float(openp.iloc[entry_i])
        if not entry_open or entry_open <= 0:
            i += 1; continue
        hard_stop = entry_open - p['stop_atr'] * float(atr.iloc[i])
        v_entry = float(vix.iloc[i]) if not pd.isna(vix.iloc[i]) else None

        exit_i = exit_px = reason = None
        mae = 0.0
        for j in range(entry_i, min(entry_i + p['max_hold'] + 1, n)):
            mae = min(mae, float(low.iloc[j]) / entry_open - 1)
            if float(low.iloc[j]) <= hard_stop:
                exit_i, exit_px, reason = j, hard_stop, 'stop'; break
            if not pd.isna(ma_exit.iloc[j]) and float(close.iloc[j]) > float(ma_exit.iloc[j]):
                exit_i, exit_px, reason = j, float(close.iloc[j]), 'target'; break
        if exit_i is None:
            exit_i = min(entry_i + p['max_hold'], n - 1)
            exit_px = float(close.iloc[exit_i]); reason = 'time'

        for j in range(entry_i, exit_i + 1):
            if j == entry_i:
                r = float(close.iloc[j]) / entry_open - 1 - cost
            elif j == exit_i:
                r = exit_px / float(close.iloc[j - 1]) - 1 - cost
            else:
                r = float(close.iloc[j]) / float(close.iloc[j - 1]) - 1
            dret.iloc[j] = r; held.iloc[j] = 1.0

        trades.append({'entry_date': idx[entry_i], 'ret': exit_px / entry_open - 1 - 2 * cost,
                       'mae': mae, 'bars': exit_i - entry_i, 'vix': v_entry})
        i = exit_i + 1
    return trades, dret, held


def run(universe, period, cost_bps):
    print(f"  [phase1] downloading {len(universe)} names + SPY + ^VIX ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True,
                      progress=False, group_by='ticker')
    spy = yf.download('SPY', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(spy, 'columns'): spy = spy.iloc[:, 0]
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'): vix = vix.iloc[:, 0]
    spy_ret = spy.pct_change()

    frames = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 260:
                frames[tk] = df
        except Exception:
            continue
    print(f"  [phase1] {len(frames)} names usable. Simulating {len(VARIANTS)} variants...")

    results = {}
    for vname, cond in VARIANTS.items():
        port_num = defaultdict(float); port_den = defaultdict(float)
        trades = []
        for df in frames.values():
            tr, dret, held = simulate_variant(df, spy, vix, PARAMS, cost_bps, cond)
            trades.extend(tr)
            for dt, h in held.items():
                if h > 0:
                    port_num[dt] += dret.loc[dt]; port_den[dt] += 1.0
        # daily book over ALL trading days (cash=0 when flat)
        strat = pd.Series({d: (port_num[d] / port_den[d]) for d in port_den}).sort_index()
        full = pd.DataFrame({'strat': strat, 'spy': spy_ret}).reindex(spy_ret.index)
        full['strat'] = full['strat'].fillna(0.0)
        full = full.dropna(subset=['spy'])
        y = full['strat'].values
        X = np.column_stack([np.ones(len(y)), full['spy'].values])
        b, se, t, r2 = ols_nw(y, X)
        rr = np.array([tr['ret'] for tr in trades]) if trades else np.array([0.0])
        ann = float(np.mean(y) * 252); vol = float(np.std(y) * np.sqrt(252))
        results[vname] = {
            'n_trades': len(trades),
            'win_rate': round(float((rr > 0).mean()), 3),
            'avg_trade': round(float(rr.mean()), 4),
            'pct_in_mkt': round(float((full['strat'] != 0).mean()), 3),
            'ann_return': round(ann, 4), 'ann_vol': round(vol, 4),
            'sharpe': round(ann / vol, 2) if vol > 0 else 0.0,
            'alpha_annual': round(float(b[0] * 252), 4), 'alpha_t': round(float(t[0]), 2),
            'beta': round(float(b[1]), 2),
        }
    spy_ann = float(spy_ret.mean() * 252); spy_vol = float(spy_ret.std() * np.sqrt(252))
    return {'params': {**PARAMS, 'period': period, 'cost_bps': cost_bps, 'n_names': len(frames)},
            'spy_sharpe': round(spy_ann / spy_vol, 2) if spy_vol > 0 else 0.0,
            'variants': results}


def print_report(o):
    p = o['params']
    print("\n" + "=" * 96)
    print("  Phase 1 — 選擇性測試:砍交易能不能救回 alpha?(PIT-乾淨,扣成本)")
    print(f"  {p['n_names']} 檔 · {p['period']} · 成本 {p['cost_bps']}bps/邊 · SPY 同期 Sharpe {o['spy_sharpe']}")
    print("=" * 96)
    print(f"  {'變體':<12}{'交易數':>7}{'勝率':>7}{'均報酬':>8}{'在場%':>7}"
          f"{'年化':>8}{'波動':>7}{'Sharpe':>8}{'α年化':>9}{'α t值':>7}{'β':>6}")
    print("  " + "-" * 92)
    for k, m in o['variants'].items():
        print(f"  {k:<12}{m['n_trades']:>7}{m['win_rate']*100:>6.1f}%{m['avg_trade']*100:>7.2f}%"
              f"{m['pct_in_mkt']*100:>6.0f}%{m['ann_return']*100:>7.1f}%{m['ann_vol']*100:>6.1f}%"
              f"{m['sharpe']:>8.2f}{m['alpha_annual']*100:>8.1f}%{m['alpha_t']:>7.2f}{m['beta']:>6.2f}")
    print("=" * 96)
    print("  讀法:α t值 > 2 = 統計顯著的正 alpha。看 SELECTIVE 的 α 與 Sharpe 有沒有"
          "明顯優於 baseline,且 β 是否下降(在場時間變少=不再是偽裝的 beta)。")
    print("=" * 96)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='8y')
    ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.cost_bps)
    print_report(res)
    path = os.path.join(REPORTS_DIR, 'phase1_selectivity.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [ok] -> {path}")


if __name__ == '__main__':
    main()
