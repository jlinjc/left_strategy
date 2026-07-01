"""
Attribution & Honest Validation  (驗明正身 — 這策略賺的是 alpha 還是偽裝的 beta?)
═══════════════════════════════════════════════════════════════════════════════
Phase 0 of the takeover. Everything else (more filters, more polish) is premature until
we know whether the strategy has any edge that is NOT just market exposure. So this module
answers the only question that matters to a sophisticated investor:

    "After costs, and after stripping out market beta (and generic short-term reversal),
     is there a positive, statistically-significant alpha — and is it stable over time?"

What it does, HONESTLY:
  1. Simulates the strategy's PIT-CLEAN CORE (RSI(2) oversold + above a RISING 200dma +
     ATR stop + MA-reversion/time exit) as a daily-marked, equal-weight-of-open-positions
     book WITH transaction costs. (The fundamental gates — Piotroski/value-trap/revisions —
     are deliberately EXCLUDED here because back-testing them on today's fundamentals is
     look-ahead; they need point-in-time data to validate. We test only what we can test
     without cheating, and say so.)
  2. Regresses the daily book return on SPY  →  alpha (annualised), beta, R², Newey-West
     t-stats.  Then ADDS a generic short-term-reversal control basket — if alpha survives
     that too, the edge is not merely "buy recent losers".
  3. Decay check: win-rate / expectancy by entry-year bucket.
  4. Liquidity-premium check: does entering at higher VIX actually earn more (the empirical
     justification for the left_multiplier "scale into fear" dial)?

Residual caveat we do NOT hide: the universe is still drawn from names that exist today, so
some survivorship bias remains (delisted disasters are missing). The numbers here are an
optimistic-but-much-more-honest read than the original backtest, not ground truth.

Usage:
    python attribution.py                 # 8y, default broadened universe
    python attribution.py --period 12y --cost-bps 25
"""
from __future__ import annotations
import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from bottom_fishing_backtest import rsi, atr_series

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Broadened, deliberately LESS curated universe: mega-caps PLUS multi-year laggards and
# names that had deep, ugly drawdowns (INTC, PYPL, WBA, PARA, MRNA, ENPH, BABA, F, NKE,
# PFE, T, VZ, DIS, SBUX, TGT, KHC, MMM, BA…). Still not survivorship-free (no delisted
# names via yfinance) but far less rosy than "today's winners only".
UNIVERSE = [
    # mega / winners
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','AVGO','TSM','AMD','QCOM','ORCL','CRM','ADBE',
    'JPM','BAC','GS','V','MA','UNH','LLY','COST','HD','WMT','PG','KO','PEP','MCD','NFLX','CAT','HON',
    # cyclicals / energy / industrials
    'XOM','CVX','COP','SLB','GE','DE','MMM','BA','EMR','DOW','FCX','GM','F',
    # laggards / deep-drawdown names (the anti-survivorship leaning)
    'INTC','CSCO','PYPL','WBA','PARA','WBD','MRNA','ENPH','BABA','NKE','SBUX','TGT','KHC',
    'PFE','BMY','MRK','CVS','T','VZ','DIS','INTU','TXN','MU','AMAT','WDC','STX','GLW','NOK','COHR',
    'GIS','KMB','HSY','EL','CL','MO','D','DUK','SO','LMT','RTX','NOC','UPS','FDX',
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))

PARAMS = dict(rsi_thr=10.0, exit_ma=20, max_hold=25, stop_atr=3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-name simulation that ALSO marks daily held-state and daily strategy return
# ─────────────────────────────────────────────────────────────────────────────

def simulate_daily(df: pd.DataFrame, p: dict, cost_bps: float):
    """Run the PIT-clean gated_stopped rule on one name; return (trades, daily_ret_series,
    held_series). daily_ret is the strategy's realised daily return for THIS name on days it
    is held (entry day measured from the fill open; costs charged on entry & exit days)."""
    if df is None or len(df) < 260:
        return [], pd.Series(dtype=float), pd.Series(dtype=float)
    close = df['Close']; openp = df.get('Open', close); low = df.get('Low', close)
    r2 = rsi(close, 2)
    ma_exit = close.rolling(p['exit_ma']).mean()
    ma200 = close.rolling(200).mean()
    rising = ma200 > ma200.shift(21)
    atr = atr_series(df, 14)
    idx = df.index; n = len(df)
    cost = cost_bps / 1e4

    dret = pd.Series(0.0, index=idx)
    held = pd.Series(0.0, index=idx)
    trades = []
    i = 200
    while i < n - 1:
        if not (r2.iloc[i] < p['rsi_thr']):
            i += 1; continue
        if not (close.iloc[i] > ma200.iloc[i]) or not bool(rising.iloc[i]) or pd.isna(atr.iloc[i]):
            i += 1; continue
        entry_i = i + 1
        entry_open = float(openp.iloc[entry_i])
        if not entry_open or entry_open <= 0:
            i += 1; continue
        hard_stop = entry_open - p['stop_atr'] * float(atr.iloc[i])

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

        # Mark daily returns across the hold (entry from fill open; costs on entry & exit)
        for j in range(entry_i, exit_i + 1):
            if j == entry_i:
                r = float(close.iloc[j]) / entry_open - 1 - cost
            elif j == exit_i:
                base = exit_px / float(close.iloc[j - 1]) - 1
                r = base - cost
            else:
                r = float(close.iloc[j]) / float(close.iloc[j - 1]) - 1
            dret.iloc[j] = r; held.iloc[j] = 1.0

        trades.append({
            'entry_date': idx[entry_i], 'ret': exit_px / entry_open - 1 - 2 * cost,
            'mae': mae, 'reason': reason, 'bars': exit_i - entry_i,
        })
        i = exit_i + 1
    return trades, dret, held


# ─────────────────────────────────────────────────────────────────────────────
# OLS with Newey-West (HAC) standard errors — overlapping daily returns need it
# ─────────────────────────────────────────────────────────────────────────────

def ols_nw(y: np.ndarray, X: np.ndarray, lags: int = 5):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    r2 = 1 - np.var(resid) / np.var(y) if np.var(y) > 0 else 0.0
    return beta, se, t, r2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(universe, period, cost_bps):
    print(f"  [attr] downloading {len(universe)} names + SPY + ^VIX ({period})...")
    raw = yf.download(universe, period=period, interval='1d',
                      auto_adjust=True, progress=False, group_by='ticker')
    spy = yf.download('SPY', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(spy, 'columns'):
        spy = spy.iloc[:, 0]
    vix = yf.download('^VIX', period=period, interval='1d', auto_adjust=True, progress=False)['Close']
    if hasattr(vix, 'columns'):
        vix = vix.iloc[:, 0]
    spy_ret = spy.pct_change()

    # Reversal control basket: each day long the worst trailing-5d-return names (bottom quintile),
    # realised next day. A generic short-term-reversal factor to control the strategy against.
    closes = {}
    port_num = defaultdict(float); port_den = defaultdict(float)   # date -> sum/count of held name daily ret
    all_trades = []
    n_names = 0
    for tk in universe:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=['Close'])
            if len(df) < 260:
                continue
            n_names += 1
            closes[tk] = df['Close']
            trades, dret, held = simulate_daily(df, PARAMS, cost_bps)
            all_trades.extend(trades)
            for dt, h in held.items():
                if h > 0:
                    port_num[dt] += dret.loc[dt]; port_den[dt] += 1.0
        except Exception:
            continue

    # Daily strategy return (equal weight across open positions; 0 = flat/cash)
    dates = sorted(port_den.keys())
    strat = pd.Series({d: port_num[d] / port_den[d] for d in dates}).sort_index()
    n_open = pd.Series({d: port_den[d] for d in dates}).sort_index()

    # Build reversal-factor daily series on the same universe
    px = pd.DataFrame(closes).sort_index()
    r5 = px.pct_change(5)
    fwd1 = px.pct_change().shift(-1)            # next-day return
    rev_factor = {}
    for d in px.index:
        row = r5.loc[d].dropna()
        if len(row) < 10:
            continue
        losers = row.nsmallest(max(3, len(row) // 5)).index
        nxt = fwd1.loc[d, losers].dropna()
        if len(nxt):
            rev_factor[d] = float(nxt.mean())
    rev = pd.Series(rev_factor).sort_index()

    # ── Align for regression on the days the strategy is actually IN the market ──
    df_reg = pd.DataFrame({'strat': strat, 'spy': spy_ret, 'rev': rev}).dropna()
    df_in = df_reg[df_reg['strat'] != 0.0] if False else df_reg   # keep all aligned days
    y = df_reg['strat'].values
    n = len(y)

    out = {'params': {**PARAMS, 'period': period, 'cost_bps': cost_bps,
                      'n_names': n_names, 'n_days': int(n),
                      'n_trades': len(all_trades)}}

    # CAPM: strat ~ a + b*SPY
    X1 = np.column_stack([np.ones(n), df_reg['spy'].values])
    b1, se1, t1, r2_1 = ols_nw(y, X1)
    out['capm'] = {
        'alpha_daily': float(b1[0]), 'alpha_annual': float(b1[0] * 252),
        'alpha_t': float(t1[0]), 'beta': float(b1[1]), 'beta_t': float(t1[1]),
        'r2': float(r2_1),
    }
    # + reversal control: strat ~ a + b*SPY + c*REV
    X2 = np.column_stack([np.ones(n), df_reg['spy'].values, df_reg['rev'].values])
    b2, se2, t2, r2_2 = ols_nw(y, X2)
    out['capm_plus_rev'] = {
        'alpha_daily': float(b2[0]), 'alpha_annual': float(b2[0] * 252),
        'alpha_t': float(t2[0]), 'beta_spy': float(b2[1]), 'beta_spy_t': float(t2[1]),
        'beta_rev': float(b2[2]), 'beta_rev_t': float(t2[2]), 'r2': float(r2_2),
    }

    # Performance of the book itself (on aligned days)
    ann_ret = float(np.mean(y) * 252)
    ann_vol = float(np.std(y) * np.sqrt(252))
    out['book'] = {
        'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        'pct_days_in_market': float((n_open.reindex(df_reg.index).fillna(0) > 0).mean()),
        'avg_positions_when_in': float(n_open[n_open > 0].mean()) if (n_open > 0).any() else 0.0,
        'spy_ann_return': float(df_reg['spy'].mean() * 252),
        'spy_sharpe': float((df_reg['spy'].mean() * 252) / (df_reg['spy'].std() * np.sqrt(252)))
                      if df_reg['spy'].std() > 0 else 0.0,
    }

    # ── Decay by entry-year ──
    by_year = defaultdict(list)
    for t in all_trades:
        by_year[t['entry_date'].year].append(t['ret'])
    decay = {}
    for yr in sorted(by_year):
        rr = np.array(by_year[yr])
        decay[int(yr)] = {'n': int(len(rr)), 'win_rate': round(float((rr > 0).mean()), 3),
                          'avg': round(float(rr.mean()), 4)}
    out['decay_by_year'] = decay

    # ── Liquidity-premium: trade outcome vs VIX at entry ──
    vix_aligned = vix.reindex(px.index).ffill()
    buckets = {'<15': [], '15-20': [], '20-25': [], '25-30': [], '>30': []}
    for t in all_trades:
        v = vix_aligned.get(t['entry_date'], np.nan)
        if pd.isna(v):
            continue
        v = float(v)
        key = ('<15' if v < 15 else '15-20' if v < 20 else '20-25' if v < 25
               else '25-30' if v < 30 else '>30')
        buckets[key].append(t['ret'])
    vixtab = {}
    for k, rr in buckets.items():
        if rr:
            a = np.array(rr)
            vixtab[k] = {'n': int(len(a)), 'win_rate': round(float((a > 0).mean()), 3),
                         'avg': round(float(a.mean()), 4)}
        else:
            vixtab[k] = {'n': 0, 'win_rate': None, 'avg': None}
    out['vix_premium'] = vixtab

    return out


def print_report(o: dict):
    p = o['params']; c = o['capm']; cr = o['capm_plus_rev']; bk = o['book']
    print("\n" + "=" * 80)
    print("  驗明正身 — Alpha vs Beta 歸因(PIT-乾淨核心,扣成本)")
    print(f"  {p['n_names']} 檔 · {p['period']} · {p['n_trades']} 筆交易 · {p['n_days']} 交易日 · 成本 {p['cost_bps']}bps/邊")
    print("=" * 80)
    print(f"  [策略本身]  年化報酬 {bk['ann_return']*100:+.1f}%  波動 {bk['ann_vol']*100:.1f}%  "
          f"Sharpe {bk['sharpe']:.2f}   (SPY 同期 Sharpe {bk['spy_sharpe']:.2f})")
    print(f"              在場時間 {bk['pct_days_in_market']*100:.0f}%  在場時平均持倉 {bk['avg_positions_when_in']:.1f} 檔")
    print("  " + "-" * 76)
    print("  [CAPM: 策略 ~ α + β·SPY]")
    print(f"     α(年化) {c['alpha_annual']*100:+.2f}%   t={c['alpha_t']:.2f}   "
          f"β={c['beta']:.2f} (t={c['beta_t']:.1f})   R²={c['r2']:.2f}")
    verdict_a = ('✓ 顯著正 alpha' if c['alpha_t'] > 2 else
                 '~ 正但不顯著' if c['alpha_annual'] > 0 else '✗ 無 alpha')
    print(f"     → {verdict_a}  (t>2 才算統計顯著)")
    print("  [+ 短線反轉對照: 策略 ~ α + β·SPY + c·REV]")
    print(f"     α(年化) {cr['alpha_annual']*100:+.2f}%   t={cr['alpha_t']:.2f}   "
          f"β_spy={cr['beta_spy']:.2f}   β_rev={cr['beta_rev']:.2f} (t={cr['beta_rev_t']:.1f})   R²={cr['r2']:.2f}")
    surv = ('✓ alpha 在控制反轉後仍存活 = 不只是「買跌」' if cr['alpha_t'] > 2 else
            '✗ 控制反轉後 alpha 消失 = 本質就是通用短線反轉' if cr['alpha_annual'] <= 0 or cr['alpha_t'] < 1 else
            '~ 邊際,反轉解釋了大部分')
    print(f"     → {surv}")
    print("  " + "-" * 76)
    print("  [edge 衰減 — 逐年]")
    for yr, m in o['decay_by_year'].items():
        print(f"     {yr}:  n={m['n']:>4}  勝率 {m['win_rate']*100:>5.1f}%  均報酬 {m['avg']*100:+.2f}%")
    print("  [流動性溢酬 — 進場 VIX vs 事後報酬](驗證 scale-into-fear)")
    for k, m in o['vix_premium'].items():
        if m['n']:
            print(f"     VIX {k:<6} n={m['n']:>4}  勝率 {m['win_rate']*100:>5.1f}%  均報酬 {m['avg']*100:+.2f}%")
    print("=" * 80)
    print("  注:基本面濾網(Piotroski/價值陷阱/分析師)未納入此回測 — 用今天的財報測過去=偷看未來。")
    print("      這裡只誠實驗證『不需要未來資訊』的核心。基本面那層需 point-in-time 資料才能驗。")
    print("=" * 80)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='8y')
    ap.add_argument('--cost-bps', type=float, default=20.0)
    ap.add_argument('tickers', nargs='*')
    args = ap.parse_args()
    uni = [t.upper() for t in args.tickers] or UNIVERSE
    res = run(uni, args.period, args.cost_bps)
    print_report(res)
    path = os.path.join(REPORTS_DIR, 'attribution.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [ok] -> {path}")


if __name__ == '__main__':
    main()
