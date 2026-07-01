"""
Naked-K Capitulation  (純裸K讀人性:群眾何時崩潰、哪一根K可以買 — 不用任何指標)
═══════════════════════════════════════════════════════════════════════════════
Pure PRICE ACTION only — no RSI, no volume, no VIX. The candles ARE the footprints of crowd
fear and greed. We detect a capitulation BOTTOM purely from OHLC structure and backtest which
naked-K reversal bar gives the best entry.

THE STORY a capitulation bottom tells in candles:
  1. acceleration  — a run of big red bars / range expansion (fear escalating to panic)
  2. climax bar    — an abnormally WIDE-range down bar (the puke; the emotional peak)
  3. the turn      — buyers seize control while the crowd is still terrified:
        HAMMER        long lower wick, close in upper half (sellers exhausted)
        ENGULF        a green bar fully engulfs the prior red (buyers overwhelm)
        OUTSIDE_UP    makes a NEW LOW then closes ABOVE the prior high (key reversal — strongest)
        RECLAIM       closes back above the climax bar's high (the puke is reclaimed)
        ANY           any of the above

SETUP (pure price): ≥15% below the 40-day high AND a wide-range down climax bar in the last 5
bars. Range normalised by average daily range (pure OHLC, not a momentum indicator).
EXIT: chandelier (highest close − 3·ATR) + recover to the pre-drop high (satisfaction); stop
below the climax low. ATR used only for risk sizing, not for the signal.

Usage:  python naked_k_bt.py --period 15y --cost-bps 20
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from bottom_fishing_backtest import atr_series, summarize
from attribution import UNIVERSE, REPORTS_DIR

SETUP_DD = -0.15          # ≥15% below the 40-day high (pure price drawdown)
CLIMAX_WIDE = 1.8         # climax bar range ≥ 1.8× average daily range
LOOK_HIGH = 40


def _candle(o, h, l, c):
    body = abs(c-o); rng = h-l
    lower = min(o, c)-l; upper = h-max(o, c)
    return body, rng, lower, upper


def _trades(df, cost, trigger, atr_mult=3.0, max_hold=252):
    if df is None or len(df) < 120: return []
    O,H,L,C = df.get('Open',df['Close']),df.get('High',df['Close']),df.get('Low',df['Close']),df['Close']
    rng = (H-L); avgrng = rng.rolling(20).mean()
    hi40 = H.rolling(LOOK_HIGH, min_periods=20).max()
    atr = atr_series(df, 14)
    idx = df.index; n = len(df); i = 60
    trades = []
    while i < n-1:
        try:
            # SETUP: deep off the 40d high + a wide-range down climax bar in the last 5 bars
            decline = C.iloc[i] <= (1+SETUP_DD)*hi40.iloc[i]
            climax_j = None
            for j in range(max(0, i-4), i+1):
                if (C.iloc[j] < O.iloc[j] and avgrng.iloc[j] and
                        rng.iloc[j] >= CLIMAX_WIDE*avgrng.iloc[j] and
                        rng.iloc[j] >= rng.iloc[max(0,j-9):j+1].max()):
                    climax_j = j
            setup = decline and climax_j is not None
        except Exception:
            i += 1; continue
        if not setup or pd.isna(atr.iloc[i]):
            i += 1; continue

        o,h,l,c = float(O.iloc[i]),float(H.iloc[i]),float(L.iloc[i]),float(C.iloc[i])
        po,ph,pl,pc = float(O.iloc[i-1]),float(H.iloc[i-1]),float(L.iloc[i-1]),float(C.iloc[i-1])
        body,r,lower,upper = _candle(o,h,l,c)
        hammer = (r > 0 and lower >= 2*body and body <= 0.4*r and (h-c) <= 0.35*r)
        engulf = (c > o and pc < po and c >= po and o <= pc)
        outside_up = (l < pl and c > ph)
        reclaim = (c > float(H.iloc[climax_j]))
        trig = {'HAMMER':hammer, 'ENGULF':engulf, 'OUTSIDE_UP':outside_up, 'RECLAIM':reclaim,
                'ANY': hammer or engulf or outside_up or reclaim}[trigger]
        if not trig:
            i += 1; continue

        ei = i+1; entry = float(O.iloc[ei])
        if not entry or entry <= 0: i += 1; continue
        climax_low = float(L.iloc[max(0,i-5):i+1].min())
        init_stop = climax_low - 0.5*float(atr.iloc[i])
        target = float(hi40.iloc[i])*0.97
        highest = entry; xi=xp=None; mae=0.0; reason=None
        for k in range(ei, min(ei+max_hold+1, n)):
            mae = min(mae, float(L.iloc[k])/entry-1)
            highest = max(highest, float(C.iloc[k]))
            stop_k = max(init_stop, highest - atr_mult*float(atr.iloc[i]))
            if float(L.iloc[k]) <= stop_k: xi,xp,reason=k,stop_k,'trail/stop'; break
            if float(C.iloc[k]) >= target: xi,xp,reason=k,float(C.iloc[k]),'satisfaction'; break
        if xi is None:
            xi=min(ei+max_hold,n-1); xp=float(C.iloc[xi]); reason='time'
        trades.append({'ret':xp/entry-1-2*cost,'mae':mae,'bars':xi-ei,'reason':reason})
        i = xi+1
    return trades


def run(universe, period, cost_bps):
    print(f"  [nakedK] downloading {len(universe)} ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True, progress=False, group_by='ticker')
    cost = cost_bps/1e4; frames = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 300: frames[tk] = df
        except Exception: continue
    yrs = max(1, int(period.rstrip('y') or 8))
    out = {}
    for tg in ['HAMMER','ENGULF','OUTSIDE_UP','RECLAIM','ANY']:
        trs = []
        for df in frames.values(): trs.extend(_trades(df, cost, tg))
        m = summarize(trs); m['per_name_yr'] = round(len(trs)/max(len(frames),1)/yrs, 2)
        out[tg] = m
    return {'n_names':len(frames),'period':period,'cost_bps':cost_bps,'variants':out}


def print_report(o):
    print("\n"+"="*90)
    print("  純裸K投降反轉 — 哪一根K可以買?(無RSI/無量/無VIX,只看價格行為)")
    print(f"  {o['n_names']} 檔 · {o['period']} · 扣{o['cost_bps']}bps · setup:距40日高{int(SETUP_DD*100)}% + 寬幅climax棒")
    print("="*90)
    print(f"  {'裸K轉折棒':<14}{'交易數':>7}{'每檔每年':>9}{'勝率':>7}{'均報酬':>9}{'盈虧比':>7}{'最深MAE':>9}{'持有天':>7}")
    print("  "+"-"*86)
    names={'HAMMER':'錘子(長下影)','ENGULF':'多頭吞噬','OUTSIDE_UP':'外包反轉(創低收高)',
           'RECLAIM':'收復climax棒','ANY':'任一反轉棒'}
    for k,m in o['variants'].items():
        if not m.get('n_trades'): print(f"  {names[k]:<14}{'—':>7}"); continue
        print(f"  {names[k]:<14}{m['n_trades']:>7}{m['per_name_yr']:>9.2f}{m['win_rate']*100:>6.1f}%"
              f"{m['avg_trade']*100:>8.2f}%{(m['profit_factor'] or 0):>7.2f}{m['worst_mae']*100:>8.1f}%{m['avg_bars_held']:>7.0f}")
    print("="*90)
    print("  讀法:純價格行為能不能抓到投降底?看盈虧比與最深MAE。對照之前『有量版多頭吞噬』PF≈3.1,")
    print("       若裸K版接近 → 光看K棒就夠;若明顯較差 → 量能(成交量)確實多帶了資訊。")
    print("="*90)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='15y'); ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.cost_bps); print_report(res)
    with open(os.path.join(REPORTS_DIR,'naked_k_bt.json'),'w',encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    main()
