"""
Universe Screener
─────────────────
Fast sweep of 150+ liquid names — rank by momentum composite to surface
the BEST candidates BEFORE running the full analysis pipeline.

This solves the "hand-picking 8 stocks" problem: instead of analyzing
what you already know, the screener tells you what the signals recommend.
The top 20 become the input to batch_run.py for full valuation + signals.

Score = 0.6 × mom_12_1  +  0.4 × mom_6m
(price-based only; no DCF/options/revisions — those run in batch_run.py on the top N)

Usage:
  python screener.py                        # top 20 from default universe
  python screener.py --top 30               # top 30
  python screener.py --out candidates.txt   # save for batch_run --watchlist
  python screener.py --out c.txt --run      # screen + run batch on top 10
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ~150 liquid names across sectors (S&P 500 / Nasdaq 100 core)
UNIVERSE = [
    # Mega-cap tech / semiconductors
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','ORCL','ADBE',
    'CRM','AMD','INTC','QCOM','TXN','MU','AMAT','LRCX','KLAC','MRVL',
    'TSM','ASML','SNPS','CDNS','ON','MCHP','STX','WDC','COHR','GLW',
    # Financials
    'JPM','BAC','GS','MS','WFC','C','AXP','V','MA','BLK','SCHW','COF',
    # Healthcare
    'UNH','JNJ','LLY','PFE','MRK','ABBV','TMO','ABT','DHR','ISRG','VRTX','REGN',
    # Consumer staples / discretionary
    'HD','MCD','NKE','SBUX','TGT','WMT','COST','LULU','AMZN',
    # Industrials / defense
    'CAT','HON','GE','RTX','LMT','NOC','DE','ETN','MMM','EMR',
    # Energy
    'XOM','CVX','COP','SLB','EOG',
    # Communications / media
    'DIS','NFLX','CMCSA','T','VZ','TMUS',
    # Materials / utilities
    'LIN','APD','NEM','FCX','NEE','DUK',
    # Small/mid tech targets
    'NOK','WOLF','PENG','CIEN','VIAV','LITE','IIVI',
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))  # deduplicate, preserve order


def _score_universe(tickers: list, period: str = '2y') -> pd.DataFrame:
    print(f"  [screener] Downloading {len(tickers)} tickers ({period})...")
    raw = yf.download(tickers, period=period, interval='1d',
                      auto_adjust=True, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    monthly = close.resample('ME').last()
    monthly = monthly.dropna(axis=1, thresh=int(len(monthly) * 0.5))
    print(f"  [screener] {monthly.shape[1]} tickers with usable history")

    if len(monthly) < 13:
        return pd.DataFrame()

    p_now = monthly.iloc[-2]                               # 1m ago (skip last month)
    p_6m  = monthly.iloc[-7]  if len(monthly) >= 7  else monthly.iloc[0]
    p_12m = monthly.iloc[-13] if len(monthly) >= 13 else monthly.iloc[0]
    p_cur = monthly.iloc[-1]
    high  = monthly.tail(12).max()

    mom_12_1 = p_now / p_12m - 1
    mom_6m   = p_now / p_6m  - 1
    mom_1m   = p_cur / p_now - 1
    dist_52w = p_cur / high  - 1

    common = mom_12_1.dropna().index.intersection(mom_6m.dropna().index)
    composite = (0.6 * mom_12_1 + 0.4 * mom_6m).reindex(common)

    df = pd.DataFrame({
        'price':         p_cur.reindex(common).round(2),
        'mom_12_1':      (mom_12_1.reindex(common) * 100).round(1),
        'mom_6m':        (mom_6m.reindex(common)   * 100).round(1),
        'mom_1m':        (mom_1m.reindex(common)   * 100).round(1),
        'dist_52w_high': (dist_52w.reindex(common)  * 100).round(1),
        'composite':     composite.round(4),
    }).dropna(subset=['composite'])

    return df.sort_values('composite', ascending=False)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(al != 0, 100.0)


def _scale01(x, lo, hi):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _score_oversold(tickers: list, period: str = '2y') -> pd.DataFrame:
    """
    Bottom-fishing screen: find names that are OVERSOLD but still in an UPTREND —
    the only place left-side (buy-the-dip) trading has positive expectancy (see
    bottom_fishing_backtest.py). This is the correct input to the抄底 engine; the
    momentum screen (_score_universe) would feed it exactly the wrong names.

    Score rewards: deep RSI(2)/RSI(14), below the lower Bollinger band, stretched
    below the 50dma, and a meaningful pullback from the 52w high — but ONLY for names
    trading above their 200dma (dips, not falling knives). Names below the 200dma are
    kept but heavily penalised so the user can still see them flagged as knives.
    """
    print(f"  [oversold] Downloading {len(tickers)} tickers ({period})...")
    raw = yf.download(tickers, period=period, interval='1d',
                      auto_adjust=True, progress=False, group_by='ticker')
    rows = {}
    for tk in tickers:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            close = df['Close'].dropna()
            if len(close) < 210:
                continue
            price = float(close.iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma200_series = close.rolling(200).mean()
            ma200 = float(ma200_series.iloc[-1])
            ma200_prev = float(ma200_series.iloc[-22])
            rsi14 = float(_rsi(close, 14).iloc[-1])
            rsi2 = float(_rsi(close, 2).iloc[-1])
            mid = close.rolling(20).mean()
            sd = close.rolling(20).std(ddof=0)
            lower = mid - 2 * sd
            width = (mid + 2 * sd) - lower
            pctb = float(((close - lower) / width.replace(0, np.nan)).iloc[-1])
            hi52 = float(close.tail(252).max())
            dist50 = price / ma50 - 1
            dist200 = price / ma200 - 1
            ddh = price / hi52 - 1
            in_uptrend = dist200 > 0
            rising = ma200 > ma200_prev

            # oversold composite 0..1
            os_score = (0.26 * _scale01(rsi2, 30, 2) +
                        0.22 * _scale01(rsi14, 45, 20) +
                        0.18 * _scale01(pctb, 0.2, -0.2) +
                        0.18 * _scale01(dist50, -0.02, -0.15) +
                        0.16 * _scale01(ddh, -0.08, -0.30))
            # trend gate multiplier: full credit in a rising uptrend, heavy penalty below 200dma
            trend_mult = 1.0 if (in_uptrend and rising) else (0.8 if in_uptrend else 0.25)
            score = os_score * trend_mult

            rows[tk] = {
                'price': round(price, 2),
                'rsi14': round(rsi14, 1), 'rsi2': round(rsi2, 1),
                'pctb': round(pctb, 2),
                'dist_50dma': round(dist50 * 100, 1),
                'dist_200dma': round(dist200 * 100, 1),
                'drawdown': round(ddh * 100, 1),
                'trend': ('UPTREND' if (in_uptrend and rising) else
                          'WEAK_UP' if in_uptrend else 'DOWNTREND'),
                'oversold': round(os_score * 100, 1),
                'score': round(score, 4),
            }
        except Exception:
            continue
    df = pd.DataFrame.from_dict(rows, orient='index')
    if df.empty:
        return df
    return df.sort_values('score', ascending=False)


def _enrich(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    top = df.head(top_n).copy()
    names, sectors, short_pcts, fwd_pes = [], [], [], []
    print(f"  [screener] Fetching info for top {len(top)} candidates...")
    for ticker in top.index:
        try:
            info = yf.Ticker(ticker).info
            names.append((info.get('shortName') or ticker)[:24])
            sectors.append(info.get('sector') or '—')
            sp = info.get('shortPercentOfFloat') or info.get('sharesPercentSharesOut')
            short_pcts.append(round(float(sp) * 100, 1) if sp else None)
            fpe = info.get('forwardPE')
            fwd_pes.append(round(float(fpe), 1) if fpe and float(fpe) < 999 else None)
        except Exception:
            names.append(ticker); sectors.append('—')
            short_pcts.append(None); fwd_pes.append(None)
    top['name']      = names
    top['sector']    = sectors
    top['short_pct'] = short_pcts
    top['fwd_pe']    = fwd_pes
    return top


def run_screener(universe: list = None, top_n: int = 20,
                 out_file: str = None, mode: str = 'momentum') -> pd.DataFrame:
    universe = universe or UNIVERSE
    scores = _score_oversold(universe) if mode == 'oversold' else _score_universe(universe)
    if scores.empty:
        print("  [screener] No data returned."); return pd.DataFrame()

    top = _enrich(scores, top_n=max(top_n, 25)).head(top_n)

    if out_file:
        out_path = os.path.join(REPORTS_DIR, out_file) if not os.path.isabs(out_file) else out_file
        with open(out_path, 'w') as f:
            for t in top.index:
                f.write(t + '\n')
        print(f"  [ok] Candidates -> {out_path}")

    return top


def print_screener(df: pd.DataFrame):
    if df.empty:
        print("  No results."); return
    print(f"\n{'='*82}")
    print(f"  SCREENER — Top {len(df)} Momentum Candidates")
    print(f"  Score = 0.6 x mom_12_1 + 0.4 x mom_6m  (price-based, point-in-time)")
    print('='*82)
    hdr = (f"  {'#':<3} {'Ticker':<7} {'Name':<22} {'Sector':<20} "
           f"{'Price':>7} {'12-1%':>7} {'6m%':>6} {'1m%':>6} "
           f"{'vs52w':>6} {'Short%':>7} {'FwdPE':>6} {'Score':>8}")
    print(hdr)
    print("  " + "-"*79)
    for i, (tk, r) in enumerate(df.iterrows(), 1):
        sp  = f"{r['short_pct']:.1f}" if pd.notna(r.get('short_pct')) and r['short_pct'] else "—"
        fpe = f"{r['fwd_pe']:.0f}"    if pd.notna(r.get('fwd_pe')) and r['fwd_pe'] else "—"
        print(f"  {i:<3} {tk:<7} {str(r.get('name','')):<22} "
              f"{str(r.get('sector','—'))[:19]:<20} "
              f"${r['price']:>6.2f} "
              f"{r['mom_12_1']:>+6.1f}% "
              f"{r['mom_6m']:>+5.1f}% "
              f"{r['mom_1m']:>+5.1f}% "
              f"{r['dist_52w_high']:>+5.1f}% "
              f"{sp:>7} "
              f"{fpe:>6} "
              f"{r['composite']:>+8.4f}")
    print('='*82)
    top10 = ' '.join(df.head(10).index.tolist())
    print(f"\n  Full analysis on top 10:")
    print(f"    python batch_run.py {top10}")
    print('='*82)


def print_oversold(df: pd.DataFrame):
    if df.empty:
        print("  No results."); return
    print(f"\n{'='*92}")
    print(f"  抄底 SCREENER — Top {len(df)} Oversold-in-Uptrend Candidates")
    print(f"  Score = oversold composite × trend gate  (只有上升趨勢中的超賣才有左側優勢)")
    print('='*92)
    hdr = (f"  {'#':<3} {'Ticker':<7} {'Price':>9} {'Trend':<10} "
           f"{'RSI14':>6} {'RSI2':>6} {'%B':>6} {'vs50':>7} {'vs200':>7} {'回撤':>7} "
           f"{'超賣':>6} {'Score':>7}")
    print(hdr)
    print("  " + "-"*89)
    for i, (tk, r) in enumerate(df.iterrows(), 1):
        print(f"  {i:<3} {tk:<7} ${r['price']:>8.2f} {r['trend']:<10} "
              f"{r['rsi14']:>6.1f} {r['rsi2']:>6.1f} {r['pctb']:>6.2f} "
              f"{r['dist_50dma']:>+6.1f}% {r['dist_200dma']:>+6.1f}% {r['drawdown']:>+6.1f}% "
              f"{r['oversold']:>6.1f} {r['score']:>7.4f}")
    print('='*92)
    top10 = ' '.join(df.head(10).index.tolist())
    print(f"\n  抄底完整分析(估值+分批劇本)on top 10:")
    print(f"    python batch_run.py {top10}")
    print('='*92)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universe screener (momentum or oversold/bottom-fishing)')
    parser.add_argument('--top',  type=int, default=20, help='Number of top candidates')
    parser.add_argument('--out',  type=str, default=None, help='Output file (tickers list)')
    parser.add_argument('--mode', choices=['momentum', 'oversold'], default='momentum',
                        help='momentum = buy strength (right-side); oversold = buy dips (left-side/抄底)')
    parser.add_argument('--run',  action='store_true', help='Run batch_run on top 10 after screening')
    args = parser.parse_args()

    df = run_screener(top_n=args.top, out_file=args.out, mode=args.mode)
    if args.mode == 'oversold':
        print_oversold(df)
    else:
        print_screener(df)

    if args.run and not df.empty:
        tickers = ' '.join(df.head(10).index.tolist())
        print(f"\n  Running batch analysis on top 10...")
        os.system(f'"{sys.executable}" batch_run.py {tickers}')
