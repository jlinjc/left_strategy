"""
Bottom-Fishing Parameter Sweep  (找最適合的抄底參數)
═══════════════════════════════════════════════════════════════════════════════
The backtest (bottom_fishing_backtest.py) proves the EDGE exists; this finds the best
KNOBS for it. It downloads the universe ONCE, then sweeps a grid of entry/exit/stop
parameters for the live rule (strict 200dma gate + ATR stop), ranking every config by
a robustness-aware objective:

    objective = expectancy_per_trade × √n_trades × profit_factor_clip
                penalised by worst-MAE (the hole you must sit in)

We optimise the LIVE rule (gated_stopped), not the naive one — there is no point tuning
a strategy you would never trade. Output is a ranked table + the single recommended
config, written to reports/bottom_fishing_sweep.json/.html.

Usage:
    python bottom_fishing_sweep.py                 # default grid, 8y
    python bottom_fishing_sweep.py --period 10y
"""
from __future__ import annotations
import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from bottom_fishing_backtest import UNIVERSE, simulate_name, summarize, REPORTS_DIR

# Grid — kept deliberately small so the sweep finishes in a couple of minutes.
GRID = {
    'rsi_thr':  [5.0, 10.0, 15.0],
    'exit_ma':  [5, 10, 20],
    'max_hold': [10, 15, 25],
    'stop_atr': [2.0, 3.0, 4.0],
}


def _objective(m: dict) -> float:
    """Reward expectancy and sample size and profit factor; punish deep MAE."""
    if not m or not m.get('n_trades') or m['n_trades'] < 100:
        return -1e9
    exp = m['expectancy']
    pf = min(m.get('profit_factor') or 0, 3.0)
    n = m['n_trades']
    worst_mae = abs(m.get('worst_mae') or 0)          # e.g. 0.24 = −24%
    mae_penalty = max(0.0, worst_mae - 0.20) * 0.5    # only punish tails beyond −20%
    return exp * (n ** 0.5) * pf - mae_penalty


def run_sweep(universe, period):
    print(f"  [sweep] downloading {len(universe)} names ({period}) once...")
    raw = yf.download(universe, period=period, interval='1d',
                      auto_adjust=True, progress=False, group_by='ticker')
    frames = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 260:
                frames[tk] = df
        except Exception:
            continue
    print(f"  [sweep] {len(frames)} names usable. Sweeping {np.prod([len(v) for v in GRID.values()])} configs...")

    combos = list(itertools.product(GRID['rsi_thr'], GRID['exit_ma'],
                                    GRID['max_hold'], GRID['stop_atr']))
    results = []
    for k, (rsi_thr, exit_ma, max_hold, stop_atr) in enumerate(combos, 1):
        trades = []
        for df in frames.values():
            trades.extend(simulate_name(df, rsi_thr=rsi_thr, exit_ma=exit_ma,
                                        max_hold=max_hold, gate='above_rising',
                                        use_stop=True, stop_atr=stop_atr))
        m = summarize(trades)
        obj = _objective(m)
        results.append({'rsi_thr': rsi_thr, 'exit_ma': exit_ma, 'max_hold': max_hold,
                        'stop_atr': stop_atr, 'objective': round(obj, 4), **m})
        if k % 9 == 0:
            print(f"    ... {k}/{len(combos)}")
    results.sort(key=lambda r: r['objective'], reverse=True)
    return {'period': period, 'n_names': len(frames), 'results': results}


def print_top(res, n=12):
    print("\n" + "=" * 104)
    print(f"  抄底參數掃描 — 實戰規則(站上上升200dma + ATR硬停損)· {res['n_names']} 檔 · {res['period']}")
    print("=" * 104)
    print(f"  {'#':<3}{'RSI<':>6}{'出場MA':>7}{'時停':>6}{'ATR停':>7}"
          f"{'交易數':>7}{'勝率':>7}{'均報酬':>8}{'盈虧比':>7}{'最深MAE':>9}{'目標分':>9}")
    print("  " + "-" * 100)
    for i, r in enumerate(res['results'][:n], 1):
        star = ' ★' if i == 1 else ''
        print(f"  {i:<3}{r['rsi_thr']:>6.0f}{r['exit_ma']:>7}{r['max_hold']:>6}{r['stop_atr']:>7.1f}"
              f"{r['n_trades']:>7}{r['win_rate']*100:>6.1f}%{r['avg_trade']*100:>7.2f}%"
              f"{(r['profit_factor'] or 0):>7.2f}{r['worst_mae']*100:>8.1f}%{r['objective']:>9.3f}{star}")
    print("=" * 104)
    best = res['results'][0]
    print(f"  ★ 建議參數:RSI(2)<{best['rsi_thr']:.0f} · 出場 收盤>{best['exit_ma']}日均 · "
          f"時間停損 {best['max_hold']}日 · 硬停損 −{best['stop_atr']:.1f}·ATR")
    print(f"    → 勝率 {best['win_rate']*100:.1f}% · 盈虧比 {best['profit_factor']:.2f} · "
          f"每筆期望 {best['expectancy']*100:+.2f}% · 最深MAE {best['worst_mae']*100:.1f}%")
    print(f"  指令:python bottom_fishing_backtest.py --rsi {best['rsi_thr']:.0f} "
          f"--exit-ma {best['exit_ma']} --max-hold {best['max_hold']} --stop-atr {best['stop_atr']:.1f}")
    print("=" * 104)


def save(res):
    path = os.path.join(REPORTS_DIR, 'bottom_fishing_sweep.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"  [ok] -> {path}")


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='8y')
    ap.add_argument('tickers', nargs='*')
    args = ap.parse_args()
    uni = [t.upper() for t in args.tickers] or UNIVERSE
    res = run_sweep(uni, args.period)
    print_top(res)
    save(res)


if __name__ == '__main__':
    main()
