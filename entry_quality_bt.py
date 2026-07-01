"""
Entry-Quality Backtest  (在同一個投降 setup 上,K線/技術「進場確認」哪個最準?越精細真的越好嗎?)
═══════════════════════════════════════════════════════════════════════════════
The user's goal: catch the BEST dip-entry using candlesticks / technicals. Honest framing —
this can't make left-side beat DCA (that's structural cash-drag), but it CAN make each
capitulation deployment higher-quality. AND "more sophisticated = better" must be PROVEN,
because in markets simple often beats complex (overfitting). So we hold the capitulation
SETUP fixed (deep drawdown + extreme RSI2 + volume climax) and the EXIT fixed (chandelier +
prior-high satisfaction), then vary ONLY the entry CONFIRMATION and measure per-trade quality.

Confirmations tested:
   NONE        — buy the setup immediately (catch the knife, no confirmation)
   TURN        — close up AND above yesterday's high (current engine's rule)
   HAMMER      — bullish hammer / long-lower-wick reversal candle
   ENGULF      — bullish engulfing
   DIVERGENCE  — RSI(14) bullish divergence (price lower-low, RSI higher-low)
   DRYUP       — volume dried up after the climax (selling exhausted)
   CONFLUENCE  — TURN and (HAMMER or ENGULF or DIVERGENCE)

Metric that matters: per-trade win%, avg, PF, and worst-MAE (did the confirmation reduce the
pain of catching it too early?). Best entry = highest PF / shallowest MAE for enough trades.

Usage:  python entry_quality_bt.py --period 15y --cost-bps 20
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

SETUP_DD = -0.15; SETUP_RSI2 = 5.0; SETUP_CLIMAX = 2.0


# ── candlestick / divergence detectors (operate on bar i) ────────────────────
def _hammer(o, h, l, c):
    body = abs(c - o); rng = h - l
    if rng <= 0: return False
    lower = min(o, c) - l
    return (lower >= 2 * body) and (body <= 0.4 * rng) and ((h - max(o, c)) <= 0.35 * rng)

def _engulf(o, h, l, c, po, pc):
    return (c > o) and (pc < po) and (c >= po) and (o <= pc)   # green engulfs prior red body

def _divergence(close, r14, i, look=20):
    try:
        cc = close.iloc[i-look:i+1].values; rr = r14.iloc[i-look:i+1].values
        half = len(cc)//2
        i1 = int(np.argmin(cc[:half])); i2 = int(np.argmin(cc[half:]))+half
        return bool(cc[i2] < cc[i1] and rr[i2] > rr[i1])
    except Exception:
        return False


def _trades(df, cost, confirm, atr_mult=3.0, max_hold=252):
    if df is None or len(df) < 300: return []
    close = df['Close']; openp = df.get('Open', close); high = df.get('High', close)
    low = df.get('Low', close); vol = df.get('Volume')
    r2 = rsi(close, 2); r14 = rsi(close, 14); atr = atr_series(df, 14)
    hi252 = high.rolling(252, min_periods=120).max()
    avg50v = vol.rolling(50).mean() if vol is not None else None
    idx = df.index; n = len(df)
    climax = pd.Series(False, index=idx)
    if vol is not None and avg50v is not None:
        spike = (vol >= SETUP_CLIMAX * avg50v) & (close < close.shift(1))
        climax = spike.rolling(5, min_periods=1).max().astype(bool)
    r2min5 = r2.rolling(5).min()

    trades = []; i = 252
    while i < n - 1:
        setup = (close.iloc[i] <= (1+SETUP_DD)*hi252.iloc[i]
                 and r2min5.iloc[i] < SETUP_RSI2 and bool(climax.iloc[i]))
        if not setup or pd.isna(atr.iloc[i]):
            i += 1; continue
        o,h,l,c = float(openp.iloc[i]),float(high.iloc[i]),float(low.iloc[i]),float(close.iloc[i])
        pc,po,ph = float(close.iloc[i-1]),float(openp.iloc[i-1]),float(high.iloc[i-1])
        turn = (c > pc) and (c > ph)
        ok = {
            'NONE': True,
            'TURN': turn,
            'HAMMER': _hammer(o,h,l,c),
            'ENGULF': _engulf(o,h,l,c,po,pc),
            'DIVERGENCE': _divergence(close, r14, i),
            'DRYUP': (vol is not None and float(vol.iloc[i-2:i+1].mean()) < 0.6*float(vol.iloc[i-9:i+1].max())),
            'CONFLUENCE': turn and (_hammer(o,h,l,c) or _engulf(o,h,l,c,po,pc) or _divergence(close,r14,i)),
        }[confirm]
        if not ok:
            i += 1; continue

        ei = i+1; entry = float(openp.iloc[ei])
        if not entry or entry <= 0: i += 1; continue
        cap_low = float(low.iloc[max(0,i-10):i+1].min())
        init_stop = min(entry - atr_mult*float(atr.iloc[i]), cap_low - 0.5*float(atr.iloc[i]))
        target = float(hi252.iloc[i])*0.97
        highest = entry; xi=xp=reason=None; mae=0.0
        for j in range(ei, min(ei+max_hold+1, n)):
            mae = min(mae, float(low.iloc[j])/entry - 1)
            highest = max(highest, float(close.iloc[j]))
            stop_j = max(init_stop, highest - atr_mult*float(atr.iloc[i]))
            if float(low.iloc[j]) <= stop_j: xi,xp,reason=j,stop_j,'trail/stop'; break
            if float(close.iloc[j]) >= target: xi,xp,reason=j,float(close.iloc[j]),'satisfaction'; break
        if xi is None:
            xi = min(ei+max_hold, n-1); xp=float(close.iloc[xi]); reason='time'
        trades.append({'ret': xp/entry-1-2*cost, 'mae': mae, 'bars': xi-ei, 'reason': reason})
        i = xi+1
    return trades


def run(universe, period, cost_bps):
    print(f"  [entryQ] downloading {len(universe)} ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True, progress=False, group_by='ticker')
    cost = cost_bps/1e4
    frames = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 300: frames[tk] = df
        except Exception: continue
    yrs = max(1, int(period.rstrip('y') or 8))
    confirms = ['NONE','TURN','HAMMER','ENGULF','DIVERGENCE','DRYUP','CONFLUENCE']
    out = {}
    for cf in confirms:
        trs = []
        for df in frames.values():
            trs.extend(_trades(df, cost, cf))
        m = summarize(trs)
        m['per_name_yr'] = round(len(trs)/max(len(frames),1)/yrs, 2)
        out[cf] = m
    return {'n_names': len(frames), 'period': period, 'cost_bps': cost_bps, 'variants': out}


def print_report(o):
    print("\n" + "="*92)
    print("  進場確認品質比較 — 同一投降setup,只換K線/技術確認(誰讓每筆抄底最好?)")
    print(f"  {o['n_names']} 檔 · {o['period']} · 扣{o['cost_bps']}bps · setup: 距高{int(SETUP_DD*100)}% + RSI2<5 + 量能投降")
    print("="*92)
    print(f"  {'進場確認':<14}{'交易數':>7}{'每檔每年':>9}{'勝率':>7}{'均報酬':>9}{'盈虧比':>7}{'最深MAE':>9}{'持有天':>7}")
    print("  "+"-"*88)
    for k,m in o['variants'].items():
        if not m.get('n_trades'):
            print(f"  {k:<14}{'—':>7}"); continue
        print(f"  {k:<14}{m['n_trades']:>7}{m['per_name_yr']:>9.2f}{m['win_rate']*100:>6.1f}%"
              f"{m['avg_trade']*100:>8.2f}%{(m['profit_factor'] or 0):>7.2f}{m['worst_mae']*100:>8.1f}%{m['avg_bars_held']:>7.0f}")
    print("="*92)
    print("  讀法:比 NONE(裸接刀)→ 各確認,看『盈虧比↑、最深MAE↓』改善多少。")
    print("       若某個複雜確認沒比 TURN 好 → 簡單就夠,別過度精細(過度配適)。")
    print("="*92)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='15y'); ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.cost_bps)
    print_report(res)
    with open(os.path.join(REPORTS_DIR, 'entry_quality_bt.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    main()
