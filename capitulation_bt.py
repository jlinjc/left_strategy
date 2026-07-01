"""
Capitulation Engine Backtest  (重做:抓「真投降」而非「天天oversold」,抱到滿足而非小反彈就跑)
═══════════════════════════════════════════════════════════════════════════════
User's (correct) critique: the live engine buys RSI(2)<10 — which fires on EVERY minor dip
(3584 times in 8y) and exits at the 20dma bounce. That is "too casual, too early" on BOTH
ends. It captures 'oversold', not CAPITULATION, and scalps the bounce instead of riding the
fear→greed recovery. Our own data agrees: the edge lived in the RARE deep-fear events.

So this rebuilds the thesis rigorously and BACKTESTS it (no more "sounds right"):

  ENTRY = TRUE CAPITULATION CONFLUENCE (rare by construction):
    1. deep drawdown   : price ≤ 80% of its 252d high        (≥ −20%, real damage)
    2. extreme oversold: RSI(2) < 5  on the prior bar
    3. selling climax  : a down day with volume ≥ 2.5× its 50d avg in the last 5 bars
    4. THE TURN        : today closes UP and above yesterday's high (knife slowing)
                         → enter next open. (Don't catch the falling knife — buy the turn.)

  EXIT = HOLD TO SATISFACTION (ride, don't scalp):
    • initial stop   : below the capitulation low (entry − 3·ATR), wide on purpose
    • chandelier trail: stop ratchets to highest_close − 3·ATR (let winners run)
    • satisfaction   : recovered to ~prior 252d high → take profit (optimism/target met)
    • time stop      : 252 bars (a year) — these are position trades, not scalps

Variants:
    CAPIT        — the confluence above (per-stock capitulation)
    CAPIT+FEAR   — also require market VIX ≥ 25 at entry (whole-market panic too)
    OLD_CORE     — the current live core (RSI2<10 + rising 200dma, exit 20dma) for contrast
Plus a buy-and-hold per-name average for context.

Usage:  python capitulation_bt.py --period 15y --cost-bps 20
"""
from __future__ import annotations
import os, json, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from bottom_fishing_backtest import rsi, atr_series, summarize
from attribution import UNIVERSE, REPORTS_DIR


def _capit_trades(df, vix, cost, dd=-0.20, rsi2_thr=5.0, climax_x=2.5, require_turn=True,
                  vix_min=None, atr_mult=3.0, max_hold=252):
    """Parameterised capitulation entry so we can sweep how much loosening costs us.
    dd: drawdown-from-252d-high threshold (negative). rsi2_thr / climax_x / require_turn /
    vix_min are the crowd-emotion knobs. Exit fixed (chandelier + prior-high) to isolate entry."""
    if df is None or len(df) < 300:
        return []
    close = df['Close']; openp = df.get('Open', close)
    high = df.get('High', close); low = df.get('Low', close); vol = df.get('Volume')
    r2 = rsi(close, 2)
    atr = atr_series(df, 14)
    hi252 = high.rolling(252, min_periods=120).max()
    avg50v = vol.rolling(50).mean() if vol is not None else None
    idx = df.index; n = len(df)
    vix_al = vix.reindex(idx).ffill() if vix is not None else None
    dd_factor = 1.0 + dd

    # selling-climax flag: any down day in last 5 with volume ≥ climax_x × 50d avg
    climax = pd.Series(False, index=idx)
    if vol is not None and avg50v is not None:
        downday = close < close.shift(1)
        spike = (vol >= climax_x * avg50v) & downday
        climax = spike.rolling(5, min_periods=1).max().astype(bool)
    elif climax_x <= 1.0:
        climax = pd.Series(True, index=idx)   # climax disabled

    trades = []
    i = 252
    while i < n - 1:
        try:
            cond_dd = close.iloc[i] <= dd_factor * hi252.iloc[i]
            cond_os = r2.iloc[i - 1] < rsi2_thr
            cond_cl = bool(climax.iloc[i])
            cond_turn = (not require_turn) or ((close.iloc[i] > close.iloc[i - 1]) and (close.iloc[i] > high.iloc[i - 1]))
            cond_fear = (vix_min is None) or (vix_al is not None and not pd.isna(vix_al.iloc[i]) and float(vix_al.iloc[i]) >= vix_min)
        except Exception:
            i += 1; continue
        if not (cond_dd and cond_os and cond_cl and cond_turn and cond_fear and not pd.isna(atr.iloc[i])):
            i += 1; continue

        entry_i = i + 1
        entry = float(openp.iloc[entry_i])
        if not entry or entry <= 0:
            i += 1; continue
        atr_e = float(atr.iloc[i]); target = float(hi252.iloc[i]) * 0.97
        init_stop = entry - atr_mult * atr_e
        highest = entry; exit_i = exit_px = reason = None; mae = 0.0
        for j in range(entry_i, min(entry_i + max_hold + 1, n)):
            mae = min(mae, float(low.iloc[j]) / entry - 1)
            highest = max(highest, float(close.iloc[j]))
            stop_j = max(init_stop, highest - atr_mult * atr_e)
            if float(low.iloc[j]) <= stop_j:
                exit_i, exit_px, reason = j, stop_j, 'trail/stop'; break
            if float(close.iloc[j]) >= target:
                exit_i, exit_px, reason = j, float(close.iloc[j]), 'satisfaction'; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold, n - 1)
            exit_px = float(close.iloc[exit_i]); reason = 'time'
        ret = exit_px / entry - 1 - 2 * cost
        trades.append({'entry_date': idx[entry_i], 'ret': ret, 'mae': mae,
                       'bars': exit_i - entry_i, 'reason': reason})
        i = exit_i + 1
    return trades


def _old_core_trades(df, cost, max_hold=25):
    """Current live core for contrast: RSI2<10 + rising 200dma, exit close>20dma / stop -3ATR."""
    if df is None or len(df) < 260:
        return []
    close = df['Close']; openp = df.get('Open', close); low = df.get('Low', close)
    r2 = rsi(close, 2); ma20 = close.rolling(20).mean()
    ma200 = close.rolling(200).mean(); rising = ma200 > ma200.shift(21)
    atr = atr_series(df, 14); idx = df.index; n = len(df)
    trades = []; i = 200
    while i < n - 1:
        if not (r2.iloc[i] < 10 and close.iloc[i] > ma200.iloc[i] and bool(rising.iloc[i]) and not pd.isna(atr.iloc[i])):
            i += 1; continue
        entry_i = i + 1; entry = float(openp.iloc[entry_i])
        if not entry or entry <= 0:
            i += 1; continue
        hard = entry - 3.0 * float(atr.iloc[i]); exit_i = exit_px = reason = None; mae = 0.0
        for j in range(entry_i, min(entry_i + max_hold + 1, n)):
            mae = min(mae, float(low.iloc[j]) / entry - 1)
            if float(low.iloc[j]) <= hard:
                exit_i, exit_px, reason = j, hard, 'stop'; break
            if not pd.isna(ma20.iloc[j]) and float(close.iloc[j]) > float(ma20.iloc[j]):
                exit_i, exit_px, reason = j, float(close.iloc[j]), 'target'; break
        if exit_i is None:
            exit_i = min(entry_i + max_hold, n - 1); exit_px = float(close.iloc[exit_i]); reason = 'time'
        trades.append({'entry_date': idx[entry_i], 'ret': exit_px / entry - 1 - 2 * cost,
                       'mae': mae, 'bars': exit_i - entry_i, 'reason': reason})
        i = exit_i + 1
    return trades


# Loosening ladder — from MOST extreme to LOOSER, each a coherent crowd-emotion stance.
CONFIGS = {
    '①極端+全市場恐慌': dict(dd=-0.20, rsi2_thr=5.0,  climax_x=2.5, require_turn=True, vix_min=25),
    '②極端(個股)':     dict(dd=-0.20, rsi2_thr=5.0,  climax_x=2.5, require_turn=True, vix_min=None),
    '③深+市場恐慌':     dict(dd=-0.18, rsi2_thr=5.0,  climax_x=2.2, require_turn=True, vix_min=20),
    '④中度':           dict(dd=-0.15, rsi2_thr=5.0,  climax_x=2.0, require_turn=True, vix_min=None),
    '⑤略寬':           dict(dd=-0.12, rsi2_thr=10.0, climax_x=1.8, require_turn=True, vix_min=None),
    '⑥寬鬆':           dict(dd=-0.10, rsi2_thr=10.0, climax_x=1.5, require_turn=True, vix_min=None),
    '⑦中度·不等轉折':   dict(dd=-0.15, rsi2_thr=5.0,  climax_x=2.0, require_turn=False, vix_min=None),
}


def run(universe, period, cost_bps):
    print(f"  [capit] downloading {len(universe)} names + ^VIX ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True, progress=False, group_by='ticker')
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'): vix = vix.iloc[:, 0]
    cost = cost_bps / 1e4
    frames = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 300:
                frames[tk] = df
        except Exception:
            continue
    n_names = len(frames)
    yrs = max(1, int(period.rstrip('y') or 8))
    variants = {}
    items = list(CONFIGS.items()) + [('舊核心(對照)', None)]
    for name, kw in items:
        trades = []
        for df in frames.values():
            trades.extend(_old_core_trades(df, cost) if kw is None else _capit_trades(df, vix, cost, **kw))
        m = summarize(trades)
        m['trades_per_name_per_yr'] = round(len(trades) / max(n_names, 1) / yrs, 2)
        reasons = defaultdict(int)
        for t in trades:
            reasons[t['reason']] += 1
        m['exit_mix'] = dict(reasons)
        variants[name] = m
    bh = []
    for df in frames.values():
        c = df['Close'].dropna()
        if len(c) > 252:
            bh.append(float(c.iloc[-1] / c.iloc[0] - 1))
    return {'period': period, 'n_names': n_names, 'cost_bps': cost_bps,
            'buyhold_avg_total': round(float(np.mean(bh)), 3) if bh else None,
            'variants': variants}


def print_report(o):
    print("\n" + "=" * 100)
    print("  投降強度階梯回測 — 從「最極端」到「較寬鬆」,看放寬 vs edge 衰減的頻譜(扣成本,抱到滿足)")
    print(f"  {o['n_names']} 檔 · {o['period']} · {o['cost_bps']}bps/邊 · 買進持有每檔平均總報酬 {o['buyhold_avg_total']*100:+.0f}%")
    print("=" * 100)
    print(f"  {'設定':<20}{'交易數':>7}{'每檔每年':>9}{'勝率':>7}{'均報酬':>9}{'中位數':>9}"
          f"{'盈虧比':>7}{'持有天':>7}{'最深MAE':>9}")
    print("  " + "-" * 96)
    for k, m in o['variants'].items():
        if not m.get('n_trades'):
            print(f"  {k:<20}{'—':>7}"); continue
        print(f"  {k:<20}{m['n_trades']:>7}{m['trades_per_name_per_yr']:>9.2f}{m['win_rate']*100:>6.1f}%"
              f"{m['avg_trade']*100:>8.2f}%{m['median_trade']*100:>8.2f}%{(m['profit_factor'] or 0):>7.2f}"
              f"{m['avg_bars_held']:>7.0f}{m['worst_mae']*100:>8.1f}%")
    print("=" * 100)
    print("  讀法:由上而下逐步放寬。交易數會變多,但要盯『均報酬/盈虧比』衰減多少 —")
    print("       甜蜜點 = 在 edge 還沒明顯崩壞前、交易數已實用的那一列。⑦對照可看『等轉折』值多少。")
    print("=" * 100)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='15y')
    ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.cost_bps)
    print_report(res)
    with open(os.path.join(REPORTS_DIR, 'capitulation_bt.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [ok] -> reports/capitulation_bt.json")


if __name__ == '__main__':
    main()
