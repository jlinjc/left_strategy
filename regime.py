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


def _fast_crash(spy_above, spy_vs_50, spy_ret_20d, spy_ret_5d, vix, vix_5d_chg):
    """Is the tape in a FAST, disorderly decline — the regime where scaling INTO fear
    is how funds die rather than how they get paid?

    The contrarian dial used to scale up to 1.5x on any VIX>30 spike while SPY was still
    above its 200dma. But the 200dma is a LAGGING line: in 2008/Mar-2020 SPY gapped down
    through it, so for days price was 'still above 200dma' on the way to −35%. Sizing UP
    there catches the falling knife with MAXIMUM size, exactly when correlations go to 1.

    So before we allow any >1.0 amplification we demand the decline be ORDERLY. A fast
    crash is flagged when price action confirms what the lagging gate can't yet see:
      • price has already lost the 50dma (faster structure break), OR
      • a sharp drop velocity (−8% in ~1m or −5% in a week), OR
      • a violent VIX spike (>+10 pts in a week) — vol shock, not a graded selloff.
    """
    if spy_vs_50 is not None and spy_vs_50 < -0.02:
        return True
    if spy_ret_20d is not None and spy_ret_20d < -0.08:
        return True
    if spy_ret_5d is not None and spy_ret_5d < -0.05:
        return True
    if vix_5d_chg is not None and vix_5d_chg > 10:
        return True
    return False


def _left_multiplier(spy_above, vix, fast_crash=False):
    """Contrarian (LEFT-side) exposure dial — the OPPOSITE of the momentum dial during
    an ORDERLY fear spike, but explicitly NOT during a disorderly crash.

    Evidence: when VIX is high, expected stock returns over the next month are high
    (Martin 2017's lower bound; replicated by Lou, Polk & Skouras 2024; Aug-2024 was a
    textbook confirmation). For a strategy that BUYS weakness, a fear spike INSIDE an
    intact, ORDERLY uptrend (SPY > 200dma, price not knifing) is the best window — so we
    scale IN. BUT that average masks a brutal left tail: the same VIX>30 reading covers
    both 'graded pullback that bounces' and '2008-style cascade'. We refuse to size up
    into the second kind.

    Two guard rails vs the old version:
      1. fast_crash (see _fast_crash) bars ALL >1.0 amplification — in a disorderly
         decline we DEFEND even while the lagging 200dma still says 'uptrend'.
      2. the orderly-fear cap is lowered 1.50 → 1.25 until the tail behaviour of the
         scale-in is understood out-of-sample. Brave, not reckless.

    This keeps the engine's 'act when it feels worst' discipline literally true for
    graded pullbacks, without amplifying exactly the tail that kills the book."""
    if vix is None:
        return 0.80
    if not spy_above:
        # Trend itself is broken — a VIX spike here is genuine knife risk. Defend.
        if vix < 20:   return 0.60       # trend broken, calm — cautious
        if vix < 30:   return 0.40
        return 0.25                      # true bear (SPY<200 + VIX>30) — smallest
    # SPY above 200dma:
    if fast_crash:
        # Lagging gate says uptrend but price/vol action says cascade. NEVER add here —
        # reduce, and reduce more the hotter the vol. This is the line that prevents the
        # 'caught the knife with max size' failure mode.
        if vix >= 30:  return 0.55
        if vix >= 25:  return 0.70
        return 0.85
    # Orderly fear spike inside an intact uptrend — scale IN, capped at 1.25.
    if vix < 20:   return 1.00
    if vix < 25:   return 1.10
    if vix < 30:   return 1.20
    return 1.25                          # graded fear spike in an uptrend = scale INTO it (capped)


def get_taiwan_regime(lookback: str = '1y') -> dict:
    """
    台股恐慌指標 — 用 TAIEX(^TWII)跌幅取代 VIX。
    VIX 是美股恐懼計,對台股毫無意義;台灣自己的市場崩盤要用台灣指數衡量。

    恐慌對照表(校準自 2008/2015/2020/2022 台股大跌):
      距20日高 > -5%                 → 平靜     (合成 VIX ≈ 16)
      距20日高 -5% ~ -10%            → 偏恐懼   (合成 VIX ≈ 21)
      距20日高 -10% ~ -15%           → 全市場恐慌(合成 VIX ≈ 27) ← 可全倉
      距20日高 < -15% 或距52週高<-20% → 極端恐慌  (合成 VIX ≈ 33) ← 世紀級
    """
    try:
        raw = yf.download('^TWII', period=lookback, interval='1d',
                          auto_adjust=True, progress=False)
        close = raw['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 20:
            raise ValueError('insufficient TAIEX data')

        price   = float(close.iloc[-1])
        hi20    = float(close.rolling(20).max().iloc[-1])
        hi52    = float(close.tail(252).max())
        dd_20d  = price / hi20 - 1
        dd_52w  = price / hi52 - 1

        # 合成 VIX:讓下游 verdict() / build_plan() 完全不需要改
        if dd_20d <= -0.15 or dd_52w <= -0.20:
            synthetic_vix = 33.0; label = '極端恐慌'; left_mult = 1.25
        elif dd_20d <= -0.10:
            synthetic_vix = 27.0; label = '全市場恐慌'; left_mult = 1.10
        elif dd_20d <= -0.05:
            synthetic_vix = 21.0; label = '偏恐懼'; left_mult = 0.90
        else:
            synthetic_vix = 16.0; label = '平靜'; left_mult = 1.00

        above_ma120 = price > float(close.rolling(120).mean().iloc[-1]) if len(close) >= 120 else True

        return {
            'regime': f'TW-{label}',
            'multiplier': left_mult,
            'left_multiplier': left_mult,
            'vix': synthetic_vix,           # 合成值,讓 verdict() 的 VIX 條件直接適用
            'vix3m': None, 'backwardation': False,
            'fast_crash': dd_20d < -0.12,   # 短期急跌 → 關閉加碼
            'tw_taiex': round(price, 0),
            'tw_dd_20d_pct': round(dd_20d * 100, 1),
            'tw_dd_52w_pct': round(dd_52w * 100, 1),
            'tw_above_ma120': above_ma120,
            'is_taiwan_regime': True,
            'spy_price': None, 'spy_200dma': None, 'spy_vs_200_pct': None,
            'spy_above_200': above_ma120,
            'color': ('#b52a2a' if synthetic_vix >= 30 else
                      '#e67e22' if synthetic_vix >= 25 else '#27ae60'),
            'description': (f'TAIEX {price:.0f} · 距20日高 {dd_20d*100:+.1f}% · '
                            f'距52週高 {dd_52w*100:+.1f}% → {label}(合成VIX {synthetic_vix})'),
            'as_of': datetime.now().strftime('%Y-%m-%d'),
        }
    except Exception as exc:
        return {
            'regime': 'TW-UNKNOWN', 'multiplier': 0.70, 'left_multiplier': 0.70,
            'vix': 20.0, 'vix3m': None, 'backwardation': False, 'fast_crash': False,
            'is_taiwan_regime': True,
            'spy_price': None, 'spy_200dma': None, 'spy_vs_200_pct': None,
            'spy_above_200': None, 'color': '#e67e22',
            'description': f'TAIEX 資料取得失敗({exc}) — 預設謹慎',
            'as_of': datetime.now().strftime('%Y-%m-%d'),
        }


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

        # ── Fast-crash inputs (so the left-side dial doesn't add into a cascade) ──
        spy_50 = float(spy_close.rolling(50).mean().iloc[-1]) if len(spy_close) >= 50 else None
        spy_vs_50 = (spy_price / spy_50 - 1) if spy_50 else None
        spy_ret_20d = (spy_price / float(spy_close.iloc[-21]) - 1) if len(spy_close) > 21 else None
        spy_ret_5d  = (spy_price / float(spy_close.iloc[-6]) - 1) if len(spy_close) > 6 else None
        try:
            vix_5d_chg = float(vix_close.iloc[-1]) - float(vix_close.iloc[0])
        except Exception:
            vix_5d_chg = None
        fast_crash = _fast_crash(spy_above, spy_vs_50, spy_ret_20d, spy_ret_5d,
                                 vix_val, vix_5d_chg)

        # ── VIX term structure (backwardation = acute fear) ──────────────────────
        # Backtest (factor_addon_bt.py): adding backwardation to the capitulation champion
        # lifts PF 6.15→6.95 / win 56%→63% — orthogonal to the VIX LEVEL. (Breadth tested
        # as redundant with VIX, so deliberately NOT added.)
        vix3m_val = None; backwardation = False
        try:
            v3_raw = yf.download('^VIX3M', period='5d', interval='1d',
                                 auto_adjust=True, progress=False)['Close']
            if isinstance(v3_raw, pd.DataFrame):
                v3_raw = v3_raw.iloc[:, 0]
            v3_raw = v3_raw.dropna()
            if len(v3_raw):
                vix3m_val = round(float(v3_raw.iloc[-1]), 1)
                backwardation = bool(vix_val > vix3m_val)   # near-term fear > longer-term = acute
        except Exception:
            pass

    except Exception as exc:
        result = {
            'regime': 'CAUTION', 'multiplier': 0.70, 'left_multiplier': 0.70,
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
                'left_multiplier': _left_multiplier(spy_above, vix_val, fast_crash),
                'spy_price': round(spy_price, 2), 'spy_200dma': round(spy_200, 2),
                'spy_vs_200_pct': spy_vs_200, 'spy_above_200': spy_above,
                'spy_vs_50_pct': round(spy_vs_50 * 100, 2) if spy_vs_50 is not None else None,
                'spy_ret_20d_pct': round(spy_ret_20d * 100, 2) if spy_ret_20d is not None else None,
                'spy_ret_5d_pct': round(spy_ret_5d * 100, 2) if spy_ret_5d is not None else None,
                'vix_5d_chg': round(vix_5d_chg, 1) if vix_5d_chg is not None else None,
                'fast_crash': fast_crash,
        'vix3m': vix3m_val, 'backwardation': backwardation,
                'vix3m': vix3m_val, 'backwardation': backwardation,
                'vix': vix_val, 'color': color, 'description': desc,
                'as_of': datetime.now().strftime('%Y-%m-%d'),
            }
            _save_state(result)
            return result

    result = {
        'regime': 'CAUTION', 'multiplier': 0.65,
        'left_multiplier': _left_multiplier(spy_above, vix_val, fast_crash),
        'spy_price': round(spy_price, 2), 'spy_200dma': round(spy_200, 2),
        'spy_vs_200_pct': spy_vs_200, 'spy_above_200': spy_above,
        'spy_vs_50_pct': round(spy_vs_50 * 100, 2) if spy_vs_50 is not None else None,
        'spy_ret_20d_pct': round(spy_ret_20d * 100, 2) if spy_ret_20d is not None else None,
        'spy_ret_5d_pct': round(spy_ret_5d * 100, 2) if spy_ret_5d is not None else None,
        'vix_5d_chg': round(vix_5d_chg, 1) if vix_5d_chg is not None else None,
        'fast_crash': fast_crash,
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
    lm = r.get('left_multiplier')
    if lm is not None:
        tag = '加碼進恐慌' if lm > 1.0 else ('防禦' if lm < 0.7 else '常規')
        print(f"  LEFT-SIDE (抄底) sizing ×{lm:.0%}  — {tag}"
              f"  (年線上的『有序』恐慌尖刺=最佳左側窗口,放大但封頂1.25x)")
    if r.get('fast_crash'):
        print(f"  ⚠ 快速崩跌偵測:ON — 雖然 SPY 仍在 200dma 之上,但價格/波動顯示為瀑布式下跌,"
              f"左側加碼已被關閉(改為防禦)。")
        det = []
        if r.get('spy_vs_50_pct') is not None: det.append(f"vs50dma {r['spy_vs_50_pct']:+.1f}%")
        if r.get('spy_ret_20d_pct') is not None: det.append(f"20日 {r['spy_ret_20d_pct']:+.1f}%")
        if r.get('spy_ret_5d_pct') is not None: det.append(f"5日 {r['spy_ret_5d_pct']:+.1f}%")
        if r.get('vix_5d_chg') is not None: det.append(f"VIX 5日 {r['vix_5d_chg']:+.1f}")
        if det:
            print(f"    觸發:{' · '.join(det)}")
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
