"""
Factor Backtest Harness
───────────────────────
The honest core of a return predictor: does the signal actually forecast returns?

This runs TODAY (no need to wait months for live snapshots to mature) by reconstructing
price-based factors point-in-time from historical prices and measuring their predictive
power out-of-sample. For each month-end it computes a factor cross-section using only
data available at that date, then correlates it with the SUBSEQUENT forward return.

Metrics (per factor × horizon):
  • IC (mean Spearman rank corr of factor vs forward return, by date)
  • IC t-stat and % of months with positive IC  (consistency, not luck)
  • Top-minus-bottom tercile forward return (the tradable long/short spread)

Scope note: revision-momentum and earnings-surprise CANNOT be reconstructed
point-in-time from free data without lookahead bias, so they are validated LIVE via
snapshot.py's track record. Here we rigorously validate the price/vol factors that the
composite leans on (≈37% of weight) — momentum, reversal, low-vol.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# A reasonable liquid universe spanning sectors (so cross-sectional IC is meaningful).
DEFAULT_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'TSM', 'AMD', 'QCOM',
    'JPM', 'BAC', 'GS', 'V', 'MA', 'UNH', 'JNJ', 'LLY', 'PFE', 'MRK',
    'XOM', 'CVX', 'CAT', 'HON', 'GE', 'PG', 'KO', 'PEP', 'WMT', 'COST',
    'HD', 'MCD', 'NKE', 'DIS', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC', 'CSCO',
    'GLW', 'NOK', 'WDC', 'STX', 'COHR', 'T', 'VZ', 'TXN', 'MU', 'AMAT',
]


# ─────────────────────────────────────────────
# FACTOR DEFINITIONS (operate on a monthly price panel)
# Each returns a DataFrame [dates × tickers] of factor values, point-in-time valid
# at each row's date (i.e. uses only data up to and including that month-end).
# ─────────────────────────────────────────────
def f_momentum_12_1(monthly: pd.DataFrame) -> pd.DataFrame:
    """12-month return excluding the most recent month (Jegadeesh-Titman)."""
    return monthly.shift(1) / monthly.shift(12) - 1


def f_reversal_1m(monthly: pd.DataFrame) -> pd.DataFrame:
    """Last-month return. Expected NEGATIVE IC (short-term mean reversion)."""
    return monthly.pct_change()


def f_low_vol(monthly: pd.DataFrame) -> pd.DataFrame:
    """Negative of trailing 12m return volatility (low-vol anomaly → higher = lower vol)."""
    return -monthly.pct_change().rolling(12).std()


def f_momentum_6m(monthly: pd.DataFrame) -> pd.DataFrame:
    return monthly.shift(1) / monthly.shift(6) - 1


FACTORS = {
    'momentum_12_1': f_momentum_12_1,
    'momentum_6m': f_momentum_6m,
    'reversal_1m': f_reversal_1m,
    'low_vol': f_low_vol,
}


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def _download_monthly(universe: list, period: str = '6y') -> pd.DataFrame:
    print(f"  [bt] Downloading {len(universe)} tickers ({period})...")
    raw = yf.download(universe, period=period, interval='1d',
                      auto_adjust=True, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    # Month-end last price; drop tickers with too little history
    monthly = close.resample('ME').last()
    monthly = monthly.dropna(axis=1, thresh=int(len(monthly) * 0.6))
    print(f"  [bt] {monthly.shape[1]} tickers with usable history, "
          f"{monthly.shape[0]} months")
    return monthly


# ─────────────────────────────────────────────
# CORE EVALUATION
# ─────────────────────────────────────────────
def evaluate_factor(monthly: pd.DataFrame, factor_df: pd.DataFrame,
                    horizon_m: int = 1, min_names: int = 8) -> dict:
    """
    Cross-sectional IC and tercile spread for one factor at one horizon.
    Forward return is from month t to t+horizon_m, aligned to the factor at t.
    """
    fwd = monthly.shift(-horizon_m) / monthly - 1

    ics, spreads = [], []
    for date in factor_df.index:
        if date not in fwd.index:
            continue
        f = factor_df.loc[date]
        r = fwd.loc[date]
        pair = pd.concat([f, r], axis=1, keys=['f', 'r']).dropna()
        if len(pair) < min_names:
            continue
        ic = pair['f'].corr(pair['r'], method='spearman')
        if pd.notna(ic):
            ics.append(ic)
        # tercile spread: top third minus bottom third forward return
        try:
            q = pd.qcut(pair['f'], 3, labels=['lo', 'mid', 'hi'], duplicates='drop')
            top = pair['r'][q == 'hi'].mean()
            bot = pair['r'][q == 'lo'].mean()
            if pd.notna(top) and pd.notna(bot):
                spreads.append(top - bot)
        except Exception:
            pass

    if not ics:
        return {'n_periods': 0}

    ics = np.array(ics)
    mean_ic = ics.mean()
    ic_std = ics.std(ddof=1) if len(ics) > 1 else np.nan
    t_stat = (mean_ic / (ic_std / np.sqrt(len(ics)))) if ic_std and ic_std > 0 else np.nan
    return {
        'n_periods': len(ics),
        'mean_ic': round(float(mean_ic), 4),
        'ic_std': round(float(ic_std), 4) if not np.isnan(ic_std) else None,
        'ic_t_stat': round(float(t_stat), 2) if not np.isnan(t_stat) else None,
        'pct_positive': round(float((ics > 0).mean()), 3),
        'tercile_spread_avg': round(float(np.mean(spreads)), 4) if spreads else None,
        'tercile_spread_ann': round(float(np.mean(spreads)) * (12 / horizon_m), 4) if spreads else None,
    }


def run_backtest(universe: list = None, horizons=(1, 3, 6), period: str = '6y') -> dict:
    universe = universe or DEFAULT_UNIVERSE
    monthly = _download_monthly(universe, period=period)
    results = {}
    for fname, ffn in FACTORS.items():
        fdf = ffn(monthly)
        results[fname] = {}
        for h in horizons:
            results[fname][f'{h}m'] = evaluate_factor(monthly, fdf, horizon_m=h)
    return {'universe_size': monthly.shape[1], 'months': monthly.shape[0],
            'horizons': list(horizons), 'factors': results}


def print_backtest(res: dict):
    print("\n" + "=" * 74)
    print("  FACTOR BACKTEST — Out-of-sample predictive power")
    print(f"  Universe: {res['universe_size']} names · {res['months']} months")
    print("=" * 74)
    print(f"  {'Factor':<16}{'Horizon':<9}{'IC':>8}{'t-stat':>8}{'%pos':>7}"
          f"{'T-B sprd':>10}{'ann':>9}")
    print("  " + "-" * 70)
    for fname, byh in res['factors'].items():
        for h, m in byh.items():
            if not m.get('n_periods'):
                continue
            print(f"  {fname:<16}{h:<9}{m['mean_ic']:>8.3f}"
                  f"{(m['ic_t_stat'] if m['ic_t_stat'] is not None else 0):>8.2f}"
                  f"{m['pct_positive']*100:>6.0f}%"
                  f"{(m['tercile_spread_avg'] or 0)*100:>9.2f}%"
                  f"{(m['tercile_spread_ann'] or 0)*100:>8.1f}%")
    print("=" * 74)
    print("  IC > 0.03 with t-stat > 2 is a genuinely useful signal. Reversal_1m IC")
    print("  is expected NEGATIVE (buy losers / sell winners over ~1 month).")
    print("=" * 74)


if __name__ == '__main__':
    import sys
    uni = sys.argv[1:] if len(sys.argv) > 1 else None
    print_backtest(run_backtest(uni))
