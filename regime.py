"""
Market Regime Filter
────────────────────
SPY 200dma + VIX → risk_on / caution / risk_off regime.
Returns an exposure multiplier applied to ALL trade-plan position sizes.

This is the portfolio-level dial: in a bear tape, automatically reduce total
risk BEFORE sizing individual names. No individual-name analysis matters if
the market is in free-fall — this is what stops the whole portfolio from blowing up.

Regime rules:
  RISK ON   — SPY > 200dma AND VIX < 20          → 100% of normal sizing
  CAUTION   — SPY > 200dma AND VIX 20-25          →  80%
  CAUTION   — SPY > 200dma AND VIX 25-30          →  65%
  RISK OFF  — SPY > 200dma AND VIX > 30 (fear spike) →  40%
  CAUTION   — SPY < 200dma AND VIX < 20 (trend broken)→  55%
  RISK OFF  — SPY < 200dma AND VIX 20-30          →  35%
  RISK OFF  — SPY < 200dma AND VIX > 30 (bear)    →  20%

Usage:
  from regime import get_regime
  r = get_regime()
  # r = {'regime': 'RISK ON', 'multiplier': 1.0, 'vix': 18.3, ...}

  python regime.py   — print current reading
"""
from __future__ import annotations
import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)
REGIME_STATE_FILE = os.path.join(REPORTS_DIR, 'regime_state.json')

# (vix_ceiling, spy_above_200, label, multiplier, color, description)
_RULES = [
    (20,  True,  'RISK ON',  1.00, '#27ae60', 'SPY > 200dma & VIX<20 — full deployment'),
    (25,  True,  'CAUTION',  0.80, '#f1c40f', 'SPY > 200dma, VIX 20-25 — trim sizes 20%'),
    (30,  True,  'CAUTION',  0.65, '#e67e22', 'SPY > 200dma, VIX 25-30 — notable stress, cut 35%'),
    (999, True,  'RISK OFF', 0.40, '#e74c3c', 'SPY > 200dma, VIX>30 — fear spike, cut 60%'),
    (20,  False, 'CAUTION',  0.55, '#e67e22', 'SPY < 200dma, VIX<20 — trend broken, defensive'),
    (30,  False, 'RISK OFF', 0.35, '#e74c3c', 'SPY < 200dma + elevated VIX — bear mode'),
    (999, False, 'RISK OFF', 0.20, '#c0392b', 'SPY < 200dma + VIX>30 — full bear market'),
]


def get_regime(lookback: str = '1y') -> dict:
    """
    Fetch SPY + VIX and classify current market regime.
    Falls back to CAUTION / 0.70x if data unavailable.
    """
    try:
        spy_raw = yf.download('SPY', period=lookback, interval='1d',
                              auto_adjust=True, progress=False)
        vix_raw = yf.download('^VIX', period='5d', interval='1d',
                              auto_adjust=True, progress=False)

        spy_close = spy_raw['Close']
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close = spy_close.dropna()

        vix_close = vix_raw['Close']
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
        vix_close = vix_close.dropna()

        spy_price = float(spy_close.iloc[-1])
        spy_200   = float(spy_close.rolling(200).mean().iloc[-1])
        spy_above = spy_price > spy_200
        spy_vs_200 = round((spy_price / spy_200 - 1) * 100, 2)
        vix_val   = round(float(vix_close.iloc[-1]), 1)

    except Exception as exc:
        result = {
            'regime': 'CAUTION', 'multiplier': 0.70,
            'spy_price': None, 'spy_200dma': None, 'spy_vs_200_pct': None,
            'spy_above_200': None, 'vix': None, 'color': '#e67e22',
            'description': f'Regime data unavailable ({exc}) — default 0.70x sizing',
            'as_of': datetime.now().strftime('%Y-%m-%d'),
        }
        _save_state(result)
        return result

    for vix_ceil, above, label, mult, color, desc in _RULES:
        if vix_val <= vix_ceil and spy_above == above:
            result = {
                'regime': label, 'multiplier': mult,
                'spy_price': round(spy_price, 2), 'spy_200dma': round(spy_200, 2),
                'spy_vs_200_pct': spy_vs_200, 'spy_above_200': spy_above,
                'vix': vix_val, 'color': color, 'description': desc,
                'as_of': datetime.now().strftime('%Y-%m-%d'),
            }
            _save_state(result)
            return result

    result = {
        'regime': 'CAUTION', 'multiplier': 0.65,
        'spy_price': round(spy_price, 2), 'spy_200dma': round(spy_200, 2),
        'spy_vs_200_pct': spy_vs_200, 'spy_above_200': spy_above,
        'vix': vix_val, 'color': '#e67e22',
        'description': 'Unclassified — conservative 0.65x sizing',
        'as_of': datetime.now().strftime('%Y-%m-%d'),
    }
    _save_state(result)
    return result


def _save_state(r: dict):
    try:
        with open(REGIME_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(r, f, indent=2)
    except Exception:
        pass


def load_last_regime() -> dict | None:
    if os.path.exists(REGIME_STATE_FILE):
        with open(REGIME_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def print_regime(r: dict):
    mult_bar = '█' * int(r['multiplier'] * 10) + '░' * (10 - int(r['multiplier'] * 10))
    print(f"\n{'='*62}")
    print(f"  MARKET REGIME:  {r['regime']:12}  sizing ×{r['multiplier']:.0%}  [{mult_bar}]")
    spy_p = r.get('spy_price')
    spy_d = r.get('spy_200dma')
    spy_v = r.get('spy_vs_200_pct', 0)
    vix   = r.get('vix')
    if spy_p:
        arrow = '▲' if r.get('spy_above_200') else '▼'
        print(f"  SPY  ${spy_p}  {arrow} 200dma ${spy_d}  ({spy_v:+.1f}%)")
    if vix:
        vix_desc = 'low' if vix < 20 else 'elevated' if vix < 30 else 'HIGH'
        print(f"  VIX  {vix}  [{vix_desc}]")
    print(f"  {r['description']}")
    print(f"  As of: {r['as_of']}")
    print('='*62)


if __name__ == '__main__':
    print_regime(get_regime())
