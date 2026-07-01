"""
Factor Add-on Test  (公正評估:把「有名有據」的波動率/廣度因子加到冠軍上,有沒有再加值?)
═══════════════════════════════════════════════════════════════════════════════
The champion (wide-climax + bullish-engulfing capitulation) already uses VIX LEVEL. This tests
two famous, grounded, ORTHOGONAL factors the strategy has NOT used:

  1. VIX TERM-STRUCTURE BACKWARDATION (VIX > VIX3M) — VIX level = "how much fear"; backwardation
     = "how ACUTE" (near-term fear above longer-term). Well-documented short-term-bottom signal,
     orthogonal to the level.
  2. MARKET BREADTH WASHOUT (% of the universe below its 200dma) — Zweig/new-lows school: when
     most of the market is below the year line, the selling is indiscriminate = the deepest
     mispricing. Complements VIX (emotion) with actual participation.

Honest test: do these IMPROVE per-trade quality on top of the champion + VIX≥25, or is it
over-engineering (like divergence-stacking was)? Self-contained breadth from the universe.

Usage:  python factor_addon_bt.py --period 15y --cost-bps 20
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

SETUP_DD = -0.15; CLIMAX_WIDE = 1.8; LOOK_HIGH = 40


def _trades(df, vix, vix3m, breadth, cost, vix_min=25, backwardation=False, breadth_min=None,
            atr_mult=3.0, max_hold=252):
    if df is None or len(df) < 120: return []
    O,H,L,C = df.get('Open',df['Close']),df.get('High',df['Close']),df.get('Low',df['Close']),df['Close']
    rng = (H-L); avgrng = rng.rolling(20).mean(); hi40 = H.rolling(LOOK_HIGH, min_periods=20).max()
    atr = atr_series(df, 14); idx = df.index; n = len(df)
    vix_al = vix.reindex(idx).ffill() if vix is not None else None
    v3_al = vix3m.reindex(idx).ffill() if vix3m is not None else None
    br_al = breadth.reindex(idx).ffill() if breadth is not None else None
    i = 60; trades = []
    while i < n-1:
        try:
            decline = C.iloc[i] <= (1+SETUP_DD)*hi40.iloc[i]
            cj = None
            for j in range(max(0,i-4), i+1):
                if (C.iloc[j] < O.iloc[j] and avgrng.iloc[j] and rng.iloc[j] >= CLIMAX_WIDE*avgrng.iloc[j]
                        and rng.iloc[j] >= rng.iloc[max(0,j-9):j+1].max()):
                    cj = j
            setup = decline and cj is not None
        except Exception:
            i += 1; continue
        if not setup or pd.isna(atr.iloc[i]):
            i += 1; continue
        o,h,l,c = float(O.iloc[i]),float(H.iloc[i]),float(L.iloc[i]),float(C.iloc[i])
        po,ph,pl,pc = float(O.iloc[i-1]),float(H.iloc[i-1]),float(L.iloc[i-1]),float(C.iloc[i-1])
        engulf = (c > o and pc < po and c >= po and o <= pc)
        trig = engulf
        # base VIX level gate
        if vix_min is not None:
            v = vix_al.iloc[i] if vix_al is not None else None
            if v is None or pd.isna(v) or float(v) < vix_min: trig = False
        # VIX backwardation (acute fear)
        if trig and backwardation:
            v = vix_al.iloc[i] if vix_al is not None else None
            v3 = v3_al.iloc[i] if v3_al is not None else None
            if v is None or v3 is None or pd.isna(v) or pd.isna(v3) or float(v) <= float(v3): trig = False
        # breadth washout
        if trig and breadth_min is not None:
            b = br_al.iloc[i] if br_al is not None else None
            if b is None or pd.isna(b) or float(b) < breadth_min: trig = False
        if not trig:
            i += 1; continue
        ei = i+1; entry = float(O.iloc[ei])
        if not entry or entry <= 0: i += 1; continue
        clow = float(L.iloc[max(0,i-5):i+1].min()); init_stop = clow - 0.5*float(atr.iloc[i])
        target = float(hi40.iloc[i])*0.97; highest=entry; xi=xp=None; mae=0.0; reason=None
        for k in range(ei, min(ei+max_hold+1, n)):
            mae = min(mae, float(L.iloc[k])/entry-1); highest = max(highest, float(C.iloc[k]))
            stop_k = max(init_stop, highest - atr_mult*float(atr.iloc[i]))
            if float(L.iloc[k]) <= stop_k: xi,xp,reason=k,stop_k,'trail'; break
            if float(C.iloc[k]) >= target: xi,xp,reason=k,float(C.iloc[k]),'satis'; break
        if xi is None: xi=min(ei+max_hold,n-1); xp=float(C.iloc[xi]); reason='time'
        trades.append({'ret':xp/entry-1-2*cost,'mae':mae,'bars':xi-ei,'reason':reason})
        i = xi+1
    return trades


def run(universe, period, cost_bps):
    print(f"  [addon] downloading {len(universe)} + ^VIX + ^VIX3M ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True, progress=False, group_by='ticker')
    def _dl(sym):
        try:
            s = yf.download(sym, period=period, interval='1d', auto_adjust=True, progress=False)['Close']
            return s.iloc[:, 0] if hasattr(s, 'columns') else s
        except Exception:
            return None
    vix = _dl('^VIX'); vix3m = _dl('^VIX3M')
    cost = cost_bps/1e4; frames = {}; closes = {}
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) >= 300: frames[tk] = df; closes[tk] = df['Close']
        except Exception: continue
    # breadth = fraction of universe below its 200dma
    px = pd.DataFrame(closes).sort_index()
    below = (px < px.rolling(200, min_periods=100).mean())
    breadth = below.mean(axis=1)
    print(f"  [addon] {len(frames)} names · VIX3M {'OK' if vix3m is not None else 'MISSING(跳過倒掛)'} · "
          f"breadth 範圍 {breadth.min():.0%}-{breadth.max():.0%}")
    yrs = max(1, int(period.rstrip('y') or 8))
    configs = [
        ('基準:冠軍 + VIX≥25',        dict(vix_min=25)),
        ('+ VIX倒掛(VIX>VIX3M)',     dict(vix_min=25, backwardation=True)),
        ('+ 廣度洗盤(>50%破年線)',     dict(vix_min=25, breadth_min=0.50)),
        ('+ 廣度洗盤(>65%破年線)',     dict(vix_min=25, breadth_min=0.65)),
        ('+ 倒掛 且 廣度>50%',        dict(vix_min=25, backwardation=True, breadth_min=0.50)),
    ]
    out = []
    for name, kw in configs:
        trs = []
        for df in frames.values():
            trs.extend(_trades(df, vix, vix3m, breadth, cost, **kw))
        m = summarize(trs); m['name'] = name; m['per_name_yr'] = round(len(trs)/max(len(frames),1)/yrs, 2)
        out.append(m)
    return {'n_names': len(frames), 'period': period, 'cost_bps': cost_bps,
            'vix3m': vix3m is not None, 'rows': out}


def print_report(o):
    print("\n"+"="*94)
    print("  加入波動率形狀 / 市場廣度因子 — 在冠軍(+VIX≥25)上還能再加值嗎?")
    print(f"  {o['n_names']} 檔 · {o['period']} · 扣{o['cost_bps']}bps · VIX3M:{'有' if o['vix3m'] else '缺(倒掛跳過)'}")
    print("="*94)
    print(f"  {'設定':<22}{'交易數':>7}{'每檔每年':>9}{'勝率':>7}{'均報酬':>9}{'盈虧比':>7}{'最深MAE':>9}")
    print("  "+"-"*90)
    for m in o['rows']:
        if not m.get('n_trades'): print(f"  {m['name']:<22}{'—':>7}"); continue
        print(f"  {m['name']:<22}{m['n_trades']:>7}{m['per_name_yr']:>9.2f}{m['win_rate']*100:>6.1f}%"
              f"{m['avg_trade']*100:>8.2f}%{(m['profit_factor'] or 0):>7.2f}{m['worst_mae']*100:>8.1f}%")
    print("="*94)
    print("  讀法:盈虧比/勝率有沒有比『基準』再升、且樣本還夠?升 → 該因子真有正交資訊;")
    print("       沒升或砍光樣本 → 過度配適,別加(像之前的背離疊加)。")
    print("="*94)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='15y'); ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.cost_bps); print_report(res)
    with open(os.path.join(REPORTS_DIR,'factor_addon_bt.json'),'w',encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)


if __name__ == '__main__':
    main()
