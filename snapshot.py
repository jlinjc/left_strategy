"""
Snapshot Persistence
────────────────────
A predictor that keeps no record can never be held accountable. Every run appends a
timestamped row — the valuation call, the signal scores, and the price AT THAT MOMENT
— to a panel. Months later, `evaluate_track_record()` joins those rows to realized
forward prices and tells you whether the calls actually worked (IC, hit rate, and the
average return of BUYs minus SELLs).

This is the difference between "a model that outputs opinions" and "a model with a
measurable edge." The file is plain CSV so it is trivially inspectable and portable.
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

SNAP_DIR = os.path.join(os.path.dirname(__file__), 'reports')
SNAP_FILE = os.path.join(SNAP_DIR, 'snapshots.csv')

COLUMNS = [
    'date', 'ticker', 'price', 'price_target', 'upside_pct', 'recommendation',
    'data_quality', 'signal_score', 'conviction', 'mom_12_1', 'revision_score',
    'surprise_last_pct', 'quality_f', 'short_pct_float', 'short_score',
    'put_skew', 'options_score', 'value_quadrant',
]


def record_snapshot(data: dict, pt_data: dict) -> dict:
    """Build + append one snapshot row from a completed analysis. Returns the row."""
    info = data['info']
    sp = data.get('signal_profile') or {}
    dq = data.get('data_quality') or {}
    comps = sp.get('components', {})
    price = (info.get('currentPrice') or info.get('regularMarketPrice')
             or info.get('previousClose') or 0)

    upside = pt_data.get('upside', 0)
    comp_score = sp.get('composite_score', 0.0)
    cheap, expensive = upside > 10, upside < -10
    pos, neg = comp_score > 0.15, comp_score < -0.15
    quad = ('cheap_improving' if cheap and pos else 'value_trap' if cheap and neg
            else 'momentum_growth' if expensive and pos else 'avoid' if expensive and neg
            else 'neutral')

    row = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'ticker': data['symbol'],
        'price': round(price, 2),
        'price_target': pt_data.get('price_target'),
        'upside_pct': upside,
        'recommendation': pt_data.get('recommendation'),
        'data_quality': dq.get('verdict', 'OK'),
        'signal_score': comp_score,
        'conviction': sp.get('conviction'),
        'mom_12_1': comps.get('price_momentum', {}).get('raw_12_1'),
        'revision_score': comps.get('revision_momentum', {}).get('score'),
        'surprise_last_pct': comps.get('earnings_surprise', {}).get('last_surprise_pct'),
        'quality_f': comps.get('quality', {}).get('f_score'),
        'short_pct_float': comps.get('short_interest', {}).get('short_pct_float'),
        'short_score': comps.get('short_interest', {}).get('score'),
        'put_skew': comps.get('options_skew', {}).get('put_skew'),
        'options_score': comps.get('options_skew', {}).get('score'),
        'value_quadrant': quad,
    }
    _append(row)
    return row


def _append(row: dict):
    os.makedirs(SNAP_DIR, exist_ok=True)
    df_row = pd.DataFrame([{c: row.get(c) for c in COLUMNS}])
    if os.path.exists(SNAP_FILE):
        df_row.to_csv(SNAP_FILE, mode='a', header=False, index=False)
    else:
        df_row.to_csv(SNAP_FILE, mode='w', header=True, index=False)


def load_snapshots() -> pd.DataFrame:
    if not os.path.exists(SNAP_FILE):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(SNAP_FILE, parse_dates=['date'])


# ─────────────────────────────────────────────
# TRACK RECORD — did past calls work?
# ─────────────────────────────────────────────
def evaluate_track_record(min_age_days: int = 21) -> dict:
    """
    For every snapshot older than `min_age_days`, fetch the realized price as of
    today (or the latest available) and measure whether the calls predicted returns.

    Reports:
      • IC of signal_score vs forward return (Spearman)
      • IC of valuation upside vs forward return
      • Hit rate of BUY (up) and SELL (down) recommendations
      • Average forward return: BUY minus SELL (the long/short spread)
      • Forward return by value × momentum quadrant
    """
    snaps = load_snapshots()
    if snaps.empty:
        return {'status': 'no_data', 'note': 'No snapshots recorded yet.'}

    today = pd.Timestamp(datetime.now().date())
    snaps['age_days'] = (today - snaps['date']).dt.days
    mature = snaps[(snaps['age_days'] >= min_age_days) &
                   (snaps['data_quality'] != 'UNRELIABLE')].copy()
    if mature.empty:
        return {'status': 'too_young',
                'note': f'No snapshots older than {min_age_days}d yet. '
                        f'Track record builds as time passes.',
                'pending': int(len(snaps))}

    # Fetch current prices once per ticker
    tickers = sorted(mature['ticker'].unique())
    cur_px = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period='5d')
            if not h.empty:
                cur_px[t] = float(h['Close'].iloc[-1])
        except Exception:
            continue

    mature['fwd_price'] = mature['ticker'].map(cur_px)
    mature = mature.dropna(subset=['fwd_price', 'price'])
    mature = mature[mature['price'] > 0]
    if mature.empty:
        return {'status': 'no_prices', 'note': 'Could not fetch forward prices.'}
    mature['fwd_return'] = mature['fwd_price'] / mature['price'] - 1

    def _ic(col):
        sub = mature.dropna(subset=[col, 'fwd_return'])
        if len(sub) < 4:
            return None
        return round(float(sub[col].corr(sub['fwd_return'], method='spearman')), 3)

    buys = mature[mature['recommendation'] == 'BUY']
    sells = mature[mature['recommendation'] == 'SELL']
    buy_hit = round(float((buys['fwd_return'] > 0).mean()), 3) if len(buys) else None
    sell_hit = round(float((sells['fwd_return'] < 0).mean()), 3) if len(sells) else None
    ls_spread = None
    if len(buys) and len(sells):
        ls_spread = round(float(buys['fwd_return'].mean() - sells['fwd_return'].mean()), 4)

    quad_perf = (mature.groupby('value_quadrant')['fwd_return']
                 .agg(['mean', 'count']).round(4).to_dict('index'))

    return {
        'status': 'ok',
        'n_evaluated': int(len(mature)),
        'avg_age_days': int(mature['age_days'].mean()),
        'ic_signal_vs_return': _ic('signal_score'),
        'ic_conviction_vs_return': _ic('conviction'),
        'ic_valuation_upside_vs_return': _ic('upside_pct'),
        'ic_momentum_vs_return': _ic('mom_12_1'),
        'ic_short_interest_vs_return': _ic('short_score'),
        'buy_hit_rate': buy_hit,
        'sell_hit_rate': sell_hit,
        'buy_minus_sell_return': ls_spread,
        'quadrant_forward_return': quad_perf,
    }


def print_track_record(tr: dict):
    print("\n" + "=" * 60)
    print("  TRACK RECORD — Do our calls predict returns?")
    print("=" * 60)
    if tr.get('status') != 'ok':
        print(f"  {tr.get('note', tr.get('status'))}")
        return
    print(f"  Evaluated {tr['n_evaluated']} snapshots (avg age {tr['avg_age_days']}d)\n")
    print(f"  IC  signal score    vs fwd return : {tr['ic_signal_vs_return']}")
    print(f"  IC  conviction       vs fwd return : {tr['ic_conviction_vs_return']}")
    print(f"  IC  valuation upside vs fwd return : {tr['ic_valuation_upside_vs_return']}")
    print(f"  IC  12-1 momentum    vs fwd return : {tr['ic_momentum_vs_return']}")
    print(f"\n  BUY hit rate  : {tr['buy_hit_rate']}")
    print(f"  SELL hit rate : {tr['sell_hit_rate']}")
    print(f"  BUY − SELL avg forward return : {tr['buy_minus_sell_return']}")
    print("\n  Forward return by value × momentum quadrant:")
    for q, d in (tr.get('quadrant_forward_return') or {}).items():
        print(f"    {q:18} {d['mean']:+.2%}  (n={int(d['count'])})")
    print("=" * 60)


if __name__ == '__main__':
    print_track_record(evaluate_track_record())
