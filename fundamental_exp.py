"""
基本面實驗 A — 財報驅動的投降 vs 大盤驅動的投降  (乾淨可測,無 look-ahead)
═══════════════════════════════════════════════════════════════════════════════
直接測試核心洞見:「跌的是情緒,還是公司真的變差?」

  • 財報驅動投降 = 投降發生在剛出財報之後(miss / 砍 guidance)→ 基本面可能真變差 → 接刀?
  • 大盤/情緒驅動投降 = 沒有財報、純粹被市場情緒拖下水 → 錯殺 → 該買?

方法:用已驗證的投降進場(深跌 + RSI2<5 + 量能投降 + 轉折)產生所有交易,
      每筆標記「進場前 N 個交易日內是否有財報日」,分兩組比 PF / 勝率 / MAE / 均報酬。

為什麼乾淨:財報「日期」是歷史上真實、不會變的(不像 yfinance 的財報數字是現在的快照)。
           所以這個實驗沒有 look-ahead —— B/C/D 那些用財報數字的實驗才有偏差。

假設:若「財報組」PF 明顯低於「大盤組」→ 證明該加「避開剛出財報的投降」濾網。
      若兩組差不多 → 投降訊號本身已隱含足夠資訊,不需要這道濾網。

Usage:  python fundamental_exp.py --period 12y --window 5 --cost-bps 20
"""
from __future__ import annotations
import os, json, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from capitulation_bt import _capit_trades
from bottom_fishing_backtest import summarize
from attribution import UNIVERSE, REPORTS_DIR

# 用「④中度」設定 — 樣本最大、統計力最強(進場條件已驗證)
ENTRY = dict(dd=-0.15, rsi2_thr=5.0, climax_x=2.0, require_turn=True, vix_min=None)


def _earnings_dates(ticker):
    """歷史財報日(date 集合)。yfinance 大約回溯到 2014。"""
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=60)
        if ed is None or not len(ed):
            return set()
        return {pd.Timestamp(d).normalize().tz_localize(None) for d in ed.index}
    except Exception:
        return set()


def _near_earnings(entry_date, earn_set, window_days):
    """進場日前 window_days 個「日曆日」內有無財報(投降的 climax→turn 通常 2-5 天內)。"""
    if not earn_set:
        return None  # 無財報資料 → 無法歸類
    e = pd.Timestamp(entry_date).normalize().tz_localize(None)
    lo = e - pd.Timedelta(days=window_days + 2)  # +2 緩衝週末
    return any(lo <= d <= e for d in earn_set)


def run(universe, period, window, cost_bps):
    print(f"  [expA] downloading {len(universe)} names + ^VIX ({period})...")
    raw = yf.download(universe, period=period, interval='1d', auto_adjust=True,
                      progress=False, group_by='ticker')
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'): vix = vix.iloc[:, 0]
    cost = cost_bps / 1e4

    earn_hit = earn_miss_data = 0
    grp = {'財報驅動投降': [], '大盤/情緒驅動投降': [], '無財報資料(排除)': []}
    n_names = 0
    print("  [expA] tagging trades by earnings proximity...")
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
        except Exception:
            continue
        if len(df) < 300:
            continue
        n_names += 1
        trades = _capit_trades(df, vix, cost, **ENTRY)
        if not trades:
            continue
        earn = _earnings_dates(tk)
        if not earn:
            earn_miss_data += 1
        for t in trades:
            flag = _near_earnings(t['entry_date'], earn, window)
            if flag is None:
                grp['無財報資料(排除)'].append(t)
            elif flag:
                grp['財報驅動投降'].append(t)
            else:
                grp['大盤/情緒驅動投降'].append(t)

    out = {'period': period, 'window_days': window, 'cost_bps': cost_bps,
           'n_names': n_names, 'names_without_earnings': earn_miss_data, 'groups': {}}
    for name, trs in grp.items():
        m = summarize(trs)
        m['name'] = name
        out['groups'][name] = m
    return out


def print_report(o):
    print("\n" + "=" * 96)
    print("  實驗 A:財報驅動投降 vs 大盤/情緒驅動投降(進場前 %d 日內有無財報)" % o['window_days'])
    print(f"  {o['n_names']} 檔 · {o['period']} · 扣{o['cost_bps']}bps · "
          f"{o['names_without_earnings']} 檔無財報日期資料")
    print("=" * 96)
    print(f"  {'分組':<22}{'交易數':>7}{'勝率':>8}{'均報酬':>9}{'中位數':>9}{'盈虧比':>8}{'最深MAE':>9}")
    print("  " + "-" * 92)
    for name in ['大盤/情緒驅動投降', '財報驅動投降', '無財報資料(排除)']:
        m = o['groups'].get(name, {})
        if not m.get('n_trades'):
            print(f"  {name:<22}{'—':>7}"); continue
        print(f"  {name:<22}{m['n_trades']:>7}{m['win_rate']*100:>7.1f}%"
              f"{m['avg_trade']*100:>8.2f}%{m['median_trade']*100:>8.2f}%"
              f"{(m['profit_factor'] or 0):>8.2f}{m['worst_mae']*100:>8.1f}%")
    print("=" * 96)
    # 結論
    macro = o['groups'].get('大盤/情緒驅動投降', {})
    earn = o['groups'].get('財報驅動投降', {})
    if macro.get('n_trades') and earn.get('n_trades'):
        pf_m = macro.get('profit_factor') or 0; pf_e = earn.get('profit_factor') or 0
        av_m = macro.get('avg_trade') or 0; av_e = earn.get('avg_trade') or 0
        print("  結論:")
        if pf_m > pf_e * 1.3 or av_m > av_e + 0.03:
            print(f"    → 大盤/情緒組明顯較優(PF {pf_m:.2f} vs {pf_e:.2f}, 均報酬 {av_m*100:+.1f}% vs {av_e*100:+.1f}%)")
            print(f"    → 假設成立:剛出財報的投降較可能是『真變差』。值得加濾網:避開進場前{o['window_days']}日內有財報者。")
        elif pf_e > pf_m * 1.3:
            print(f"    → 反而財報組較優(PF {pf_e:.2f} vs {pf_m:.2f}) — 財報後恐慌超賣反彈更兇,不該避開。")
        else:
            print(f"    → 兩組差異不大(PF {pf_m:.2f} vs {pf_e:.2f})。投降訊號本身已含足夠資訊,加這道濾網屬過度配適,不加。")
    print("=" * 96)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='12y')
    ap.add_argument('--window', type=int, default=5, help='進場前幾個日曆日內算「財報驅動」')
    ap.add_argument('--cost-bps', type=float, default=20.0)
    args = ap.parse_args()
    res = run(UNIVERSE, args.period, args.window, args.cost_bps)
    print_report(res)
    with open(os.path.join(REPORTS_DIR, 'fundamental_exp_A.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [ok] -> reports/fundamental_exp_A.json")


if __name__ == '__main__':
    main()
