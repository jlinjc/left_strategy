"""
Champion Optimisation  (把跨回測的洞察套到冠軍訊號:寬幅climax + 多頭吞噬,還能再優化嗎?)
═══════════════════════════════════════════════════════════════════════════════
Cross-backtest insights, applied to the champion entry (a wide-range climax bar + a bullish
ENGULFING reversal, hold-to-satisfaction). Tests the highest-value opportunities:

  A. VIX conditioning — market-wide fear was the strongest quality knob in EVERY prior test,
     yet the champion never used it. Does engulfing-capitulation get even better when the
     whole crowd is also panicking?               → champion + VIX≥25 / ≥30
  B. Confluence for the tail — divergence had the tightest worst-MAE.  → champion + RSI divergence
  F. Frequency — the good reversals OR'd together (engulf/outside-up/reclaim; NO hammer).
  D. Exit sweep — chandelier trail multiple (2 / 3 / 4 ·ATR) on the champion.

Pure-price setup (≥15% off 40d high + wide climax bar). VIX only as a conditioning overlay.
Exit: chandelier + recover-to-prior-high satisfaction; stop below climax low.

Usage:  python champion_opt_bt.py --period 15y --cost-bps 20
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from bottom_fishing_backtest import atr_series, summarize, rsi
from attribution import UNIVERSE, REPORTS_DIR

SETUP_DD = -0.15; CLIMAX_WIDE = 1.8; LOOK_HIGH = 40


def _div(close, r14, i, look=20):
    try:
        cc = close.iloc[i-look:i+1].values; rr = r14.iloc[i-look:i+1].values
        half = len(cc)//2
        i1 = int(np.argmin(cc[:half])); i2 = int(np.argmin(cc[half:]))+half
        return bool(cc[i2] < cc[i1] and rr[i2] > rr[i1])
    except Exception:
        return False


def _trades(df, vix, cost, mode='ENGULF', vix_min=None, atr_mult=3.0, max_hold=252):
    if df is None or len(df) < 120: return []
    O,H,L,C = df.get('Open',df['Close']),df.get('High',df['Close']),df.get('Low',df['Close']),df['Close']
    rng = (H-L); avgrng = rng.rolling(20).mean(); hi40 = H.rolling(LOOK_HIGH, min_periods=20).max()
    atr = atr_series(df, 14); r14 = rsi(C, 14)
    idx = df.index; n = len(df); vix_al = vix.reindex(idx).ffill() if vix is not None else None
    i = 60; trades = []
    while i < n-1:
        try:
            decline = C.iloc[i] <= (1+SETUP_DD)*hi40.iloc[i]
            climax_j = None
            for j in range(max(0, i-4), i+1):
                if (C.iloc[j] < O.iloc[j] and avgrng.iloc[j] and rng.iloc[j] >= CLIMAX_WIDE*avgrng.iloc[j]
                        and rng.iloc[j] >= rng.iloc[max(0,j-9):j+1].max()):
                    climax_j = j
            setup = decline and climax_j is not None
        except Exception:
            i += 1; continue
        if not setup or pd.isna(atr.iloc[i]):
            i += 1; continue
        o,h,l,c = float(O.iloc[i]),float(H.iloc[i]),float(L.iloc[i]),float(C.iloc[i])
        po,ph,pl,pc = float(O.iloc[i-1]),float(H.iloc[i-1]),float(L.iloc[i-1]),float(C.iloc[i-1])
        engulf = (c > o and pc < po and c >= po and o <= pc)
        outside = (l < pl and c > ph)
        reclaim = (c > float(H.iloc[climax_j]))
        diverg = _div(C, r14, i)
        if mode == 'ENGULF': trig = engulf
        elif mode == 'GOODOR': trig = engulf or outside or reclaim
        elif mode == 'ENGULF+DIV': trig = engulf and diverg
        else: trig = engulf
        if vix_min is not None:
            v = vix_al.iloc[i] if vix_al is not None else None
            if v is None or pd.isna(v) or float(v) < vix_min: trig = False
        if not trig:
            i += 1; continue
        ei = i+1; entry = float(O.iloc[ei])
        if not entry or entry <= 0: i += 1; continue
        clow = float(L.iloc[max(0,i-5):i+1].min()); init_stop = clow - 0.5*float(atr.iloc[i])
        target = float(hi40.iloc[i])*0.97; highest = entry; xi=xp=None; mae=0.0; reason=None
        for k in range(ei, min(ei+max_hold+1, n)):
            mae = min(mae, float(L.iloc[k])/entry-1); highest = max(highest, float(C.iloc[k]))
            stop_k = max(init_stop, highest - atr_mult*float(atr.iloc[i]))
            if float(L.iloc[k]) <= stop_k: xi,xp,reason=k,stop_k,'trail'; break
            if float(C.iloc[k]) >= target: xi,xp,reason=k,float(C.iloc[k]),'satis'; break
        if xi is None: xi=min(ei+max_hold,n-1); xp=float(C.iloc[xi]); reason='time'
        trades.append({'ret':xp/entry-1-2*cost,'mae':mae,'bars':xi-ei,'reason':reason})
        i = xi+1
    return trades


CONFIGS = [
    ('冠軍:寬climax+吞噬',        dict(mode='ENGULF', vix_min=None, atr_mult=3.0)),
    ('A) 冠軍 + VIX≥25',          dict(mode='ENGULF', vix_min=25, atr_mult=3.0)),
    ('A) 冠軍 + VIX≥30',          dict(mode='ENGULF', vix_min=30, atr_mult=3.0)),
    ('B) 冠軍 + 背離(合流)',       dict(mode='ENGULF+DIV', vix_min=None, atr_mult=3.0)),
    ('F) 好訊號OR(增頻)',         dict(mode='GOODOR', vix_min=None, atr_mult=3.0)),
    ('F) 好訊號OR + VIX≥25',      dict(mode='GOODOR', vix_min=25, atr_mult=3.0)),
    ('D) 冠軍 吊燈2ATR(緊)',      dict(mode='ENGULF', vix_min=None, atr_mult=2.0)),
    ('D) 冠軍 吊燈4ATR(鬆)',      dict(mode='ENGULF', vix_min=None, atr_mult=4.0)),
]


def run(universe, period, cost_bps):
    print(f"  [champ] downloading {len(universe)} + ^VIX ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True, progress=False, group_by='ticker')
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'): vix = vix.iloc[:, 0]
    cost = cost_bps/1e4; frames = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 300: frames[tk] = df
        except Exception: continue
    yrs = max(1, int(period.rstrip('y') or 8)); out = []
    for name, kw in CONFIGS:
        trs = []
        for df in frames.values(): trs.extend(_trades(df, vix, cost, **kw))
        m = summarize(trs); m['name'] = name; m['per_name_yr'] = round(len(trs)/max(len(frames),1)/yrs, 2)
        out.append(m)
    return {'n_names': len(frames), 'period': period, 'cost_bps': cost_bps, 'rows': out}


def print_report(o):
    print("\n"+"="*96)
    print("  冠軍訊號優化 — 把跨回測洞察套上去(寬climax+多頭吞噬 為基準)")
    print(f"  {o['n_names']} 檔 · {o['period']} · 扣{o['cost_bps']}bps · 抱到滿足")
    print("="*96)
    print(f"  {'設定':<22}{'交易數':>7}{'每檔每年':>9}{'勝率':>7}{'均報酬':>9}{'盈虧比':>7}{'最深MAE':>9}{'持有天':>7}")
    print("  "+"-"*92)
    for m in o['rows']:
        if not m.get('n_trades'): print(f"  {m['name']:<22}{'—':>7}"); continue
        print(f"  {m['name']:<22}{m['n_trades']:>7}{m['per_name_yr']:>9.2f}{m['win_rate']*100:>6.1f}%"
              f"{m['avg_trade']*100:>8.2f}%{(m['profit_factor'] or 0):>7.2f}{m['worst_mae']*100:>8.1f}%{m['avg_bars_held']:>7.0f}")
    print("="*96)
    print("  讀法:看 A)VIX 過濾有沒有把冠軍的盈虧比/均報酬再推高(最大機會點);")
    print("       B)背離合流是否縮尾部;F)增頻後品質掉多少;D)吊燈鬆緊對報酬的影響。")
    print("="*96)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='15y'); ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.cost_bps); print_report(res)
    with open(os.path.join(REPORTS_DIR,'champion_opt_bt.json'),'w',encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    main()
