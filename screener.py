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
                 out_file: str = None) -> pd.DataFrame:
    universe = universe or UNIVERSE
    scores = _score_universe(universe)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universe momentum screener')
    parser.add_argument('--top',  type=int, default=20, help='Number of top candidates')
    parser.add_argument('--out',  type=str, default=None, help='Output file (tickers list)')
    parser.add_argument('--run',  action='store_true', help='Run batch_run on top 10 after screening')
    args = parser.parse_args()

    df = run_screener(top_n=args.top, out_file=args.out)
    print_screener(df)

    if args.run and not df.empty:
        tickers = ' '.join(df.head(10).index.tolist())
        print(f"\n  Running batch analysis on top 10...")
        os.system(f'"{sys.executable}" batch_run.py {tickers}')
