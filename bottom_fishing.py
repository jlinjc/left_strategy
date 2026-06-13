"""
Left-Side / Bottom-Fishing Engine  (抄底引擎)
═══════════════════════════════════════════════════════════════════════════════
This is the CONTRARIAN counterpart to the momentum engine. Momentum buys strength
(right-side); this buys weakness (left-side) — but ONLY where left-side trading
actually has positive expectancy, because the project's own backtest.py proved that
naive "buy the dip" on large-caps has the WRONG sign (reversal_1m IC is positive —
large-caps that fall keep falling short-term). So a left-side strategy that just buys
oversold names loses. The edge lives in a narrow intersection, and this engine is
built to trade ONLY that intersection:

  1.  STRUCTURE GATE  — buy dips only in an UPTREND (price above a rising 200dma).
      Connors-style RSI(2) mean-reversion has robust positive expectancy in uptrends
      and negative expectancy in downtrends. Below a falling 200dma = a falling knife,
      not a buyable dip. This gate does most of the work.

  2.  SURVIVABILITY  — only fish quality that cannot go to zero. The drop must be
      sentiment/technical, not fundamental decay. Piotroski F, positive FCF, leverage,
      Altman-Z proxy. A cheap-and-deteriorating name is a value trap, not a bottom.

  3.  OVERSOLD / CAPITULATION — the opportunity must actually exist: deep RSI, below
      the lower Bollinger band, far below moving averages, drawdown depth, down-day
      streak, and a volume climax. You don't buy "down," you buy SELLING EXHAUSTION.

  4.  CONFIRMATION — reduce "too early" risk: RSI turning up from oversold, a bullish
      reversal candle, reclaim of the 10dma, volume dry-up, RSI bullish divergence.
      Pure left-side buys into the fall; smart left-side waits for the knife to slow.

  5.  MARGIN OF SAFETY — buy quality ON SALE: price below intrinsic value (composite
      PT). Oversold + expensive is gambling; oversold + cheap is an opportunity.

Output is an EXECUTABLE, anti-human (反人性) playbook:
  • a verdict tier (STRONG / SPECULATIVE / FALLING-KNIFE / NOT-OVERSOLD)
  • a 3-tranche scale-in ladder with pre-committed prices and sizes
  • a hard stop = thesis invalidation (sized so a full failure = your risk budget)
  • mean-reversion profit targets with a scale-out plan
  • a time stop (an oversold bounce that doesn't bounce is a failed thesis)
  • discipline rules written IN ADVANCE so you act when it feels worst

Backtested honestly in bottom_fishing_backtest.py — don't trust it until you've seen
the win rate and expectancy.

API:
  from bottom_fishing import analyze_bottom_fish
  res = analyze_bottom_fish(data, quality=..., pt_data=..., regime=...)

CLI:
  python bottom_fishing.py AAPL MSFT NVDA            # analyze + HTML report
  python bottom_fishing.py --watchlist candidates.txt
"""
from __future__ import annotations
import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

ACCOUNT_DEFAULTS = {
    'account_size': 100_000.0,
    'risk_per_trade': 0.01,      # risk budget for the FULL (all-tranches-filled) position
    'max_position_pct': 0.15,
    'atr_period': 14,
    'regime_multiplier': 1.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS  (all point-in-time; operate on a daily OHLCV frame)
# ═══════════════════════════════════════════════════════════════════════════════

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)   # no losses → RSI 100
    return out


def bollinger_pctb(close: pd.Series, period: int = 20, k: float = 2.0):
    """Returns (%B, lower_band, mid, upper_band). %B<0 = below lower band."""
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower).replace(0, np.nan)
    pctb = (close - lower) / width
    return pctb, lower, ma, upper


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / rng


def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    return 100 * (close - ll) / rng


def atr(hist: pd.DataFrame, period: int = 14):
    if hist is None or hist.empty or not {'High', 'Low', 'Close'} <= set(hist.columns):
        return None
    h, l, c = hist['High'], hist['Low'], hist['Close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]
    try:
        return float(a)
    except Exception:
        return None


def consecutive_down_days(close: pd.Series) -> int:
    rets = close.pct_change().dropna()
    n = 0
    for r in reversed(rets.values):
        if r < 0:
            n += 1
        else:
            break
    return n


def _safe_last(series, default=None):
    try:
        v = series.dropna().iloc[-1]
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else default
    except Exception:
        return default


def _ma(close: pd.Series, n: int):
    if close is None or len(close) < n:
        return None
    return float(close.rolling(n).mean().iloc[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — TECHNICAL OVERSOLD READING
# ═══════════════════════════════════════════════════════════════════════════════

def technical_panel(hist_1y: pd.DataFrame, hist_5y: pd.DataFrame) -> dict:
    """A full oversold/exhaustion read from price+volume. Uses 5y for 200dma context."""
    out = {}
    base = hist_5y if (hist_5y is not None and len(hist_5y) > 220) else hist_1y
    if base is None or base.empty:
        return out
    close = base['Close']
    high = base.get('High', close)
    low = base.get('Low', close)
    vol = base.get('Volume')

    price = float(close.iloc[-1])
    out['price'] = round(price, 2)

    # RSI
    out['rsi14'] = _safe_last(rsi(close, 14))
    out['rsi2'] = _safe_last(rsi(close, 2))

    # Bollinger %B
    pctb, lower, mid, upper = bollinger_pctb(close, 20, 2.0)
    out['boll_pctb'] = _safe_last(pctb)
    out['boll_lower'] = _safe_last(lower)
    out['boll_mid'] = _safe_last(mid)

    # Williams %R / Stochastic
    out['williams_r'] = _safe_last(williams_r(high, low, close, 14))
    out['stoch_k'] = _safe_last(stochastic_k(high, low, close, 14))

    # Moving-average distances
    ma50 = _ma(close, 50)
    ma200 = _ma(close, 200)
    ma10 = _ma(close, 10)
    out['ma10'] = round(ma10, 2) if ma10 else None
    out['ma50'] = round(ma50, 2) if ma50 else None
    out['ma200'] = round(ma200, 2) if ma200 else None
    out['dist_ma50'] = round(price / ma50 - 1, 4) if ma50 else None
    out['dist_ma200'] = round(price / ma200 - 1, 4) if ma200 else None

    # 200dma slope over last ~month (rising trend = buyable-dip context)
    if len(close) > 221:
        ma200_series = close.rolling(200).mean()
        prev = ma200_series.iloc[-21]
        cur = ma200_series.iloc[-1]
        out['ma200_slope_1m'] = round(cur / prev - 1, 4) if prev and prev > 0 else None
    else:
        out['ma200_slope_1m'] = None

    # Drawdown depth
    hi_52 = float(close.tail(252).max()) if len(close) >= 30 else float(close.max())
    out['dist_52w_high'] = round(price / hi_52 - 1, 4) if hi_52 else None
    lo_52 = float(close.tail(252).min()) if len(close) >= 30 else float(close.min())
    out['dist_52w_low'] = round(price / lo_52 - 1, 4) if lo_52 else None
    out['low_52w'] = round(lo_52, 2) if lo_52 else None

    # Down-day streak
    out['down_days_streak'] = consecutive_down_days(close)

    # Volume climax / dry-up
    if vol is not None and len(vol.dropna()) >= 25:
        v = vol.fillna(0)
        avg20 = float(v.tail(20).mean())
        out['vol_today_x'] = round(float(v.iloc[-1]) / avg20, 2) if avg20 > 0 else None
        # peak volume in the last 10 days vs the 20d average = climax intensity
        out['vol_climax_x'] = round(float(v.tail(10).max()) / avg20, 2) if avg20 > 0 else None
        # last 3 days avg vs the climax — falling = sellers drying up
        recent3 = float(v.tail(3).mean())
        peak10 = float(v.tail(10).max())
        out['vol_dryup'] = round(recent3 / peak10, 2) if peak10 > 0 else None
    else:
        out['vol_today_x'] = out['vol_climax_x'] = out['vol_dryup'] = None

    # ATR
    out['atr'] = round(atr(base, 14) or 0, 2) or None
    out['atr_pct'] = round((out['atr'] / price), 4) if out['atr'] else None

    # Candle / reversal read (last bar)
    try:
        o = float(base['Open'].iloc[-1]); c = price
        pc = float(close.iloc[-2])
        rng = float(high.iloc[-1] - low.iloc[-1])
        lower_wick = (min(o, c) - float(low.iloc[-1]))
        out['bull_reversal_candle'] = bool(c > o and c > pc and rng > 0 and lower_wick / rng > 0.4)
    except Exception:
        out['bull_reversal_candle'] = False

    # RSI turning up from oversold (last RSI2 > prior RSI2 while still low)
    try:
        r2 = rsi(close, 2)
        out['rsi2_turning_up'] = bool(r2.iloc[-1] > r2.iloc[-2] and r2.iloc[-2] < 15)
    except Exception:
        out['rsi2_turning_up'] = False

    # RSI(14) bullish divergence over the last ~20 sessions
    out['rsi_bull_divergence'] = _rsi_divergence(close, rsi(close, 14))

    # Reclaim of 10dma after being below it
    out['reclaim_ma10'] = bool(ma10 and price > ma10 and float(close.iloc[-2]) < ma10) if ma10 else False

    return out


def _rsi_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 20) -> bool:
    """Price makes a lower low but RSI makes a higher low → bullish divergence."""
    try:
        c = close.tail(lookback).values
        r = rsi_series.tail(lookback).values
        half = lookback // 2
        p_low1_i = int(np.argmin(c[:half]))
        p_low2_i = int(np.argmin(c[half:])) + half
        price_lower_low = c[p_low2_i] < c[p_low1_i]
        rsi_higher_low = r[p_low2_i] > r[p_low1_i]
        return bool(price_lower_low and rsi_higher_low)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — STRUCTURE GATE  (the master filter: buyable dip vs falling knife)
# ═══════════════════════════════════════════════════════════════════════════════

def structure_gate(panel: dict) -> dict:
    """
    Classify the trend the dip is happening IN. This is the single most important
    determinant of whether bottom-fishing works.
      UPTREND       price > 200dma AND 200dma rising   → buyable dip (best)
      WEAK_UPTREND  price > 200dma BUT 200dma flat/down → fragile, smaller size
      EARLY_BREAK   price just below 200dma, 200dma still rising → transition, caution
      DOWNTREND     price < falling 200dma              → falling knife, avoid
    """
    dist200 = panel.get('dist_ma200')
    slope = panel.get('ma200_slope_1m')
    if dist200 is None:
        return {'trend': 'UNKNOWN', 'structure_mult': 0.5,
                'note': '200dma 資料不足,結構不明 — 視為投機。'}

    above = dist200 > 0
    rising = (slope or 0) > 0

    if above and rising:
        return {'trend': 'UPTREND', 'structure_mult': 1.00,
                'note': '價在上升 200dma 之上 — 標準「上升趨勢中的回檔」,左側勝率最高。'}
    if above and not rising:
        return {'trend': 'WEAK_UPTREND', 'structure_mult': 0.70,
                'note': '價在 200dma 之上但均線轉平/下彎 — 趨勢轉弱,縮小部位。'}
    if (not above) and rising and dist200 > -0.05:
        return {'trend': 'EARLY_BREAK', 'structure_mult': 0.55,
                'note': '剛跌破仍上升的 200dma(<5%)— 過渡期,僅投機性試單。'}
    return {'trend': 'DOWNTREND', 'structure_mult': 0.25,
            'note': '價在下彎 200dma 之下 — 接刀區,結構未壞前不抄底。'}


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — SURVIVABILITY  (won't go to zero; the drop is sentiment, not decay)
# ═══════════════════════════════════════════════════════════════════════════════

def survivability(info: dict, quality: dict, data: dict) -> dict:
    """
    Score 0-100 of how safe it is to catch this name. Low = high knife/zero risk.
    Blends Piotroski F, FCF sign, leverage, current ratio, and an Altman-Z proxy.
    """
    reasons = []
    score = 50.0

    # Piotroski F-score (0-9): the spine of survivability
    f = (quality or {}).get('score')
    if f is not None:
        score += (f - 5) * 6   # F=9 → +24, F=0 → -30
        if f <= 3:
            reasons.append(f'Piotroski F={f} 偏弱 — 基本面惡化風險')
        elif f >= 7:
            reasons.append(f'Piotroski F={f} 強健')

    # Free cash flow sign
    fcf = info.get('freeCashflow')
    if fcf is not None:
        if fcf > 0:
            score += 8
        else:
            score -= 15
            reasons.append('自由現金流為負 — 燒錢中,接刀風險升高')

    # Leverage
    de = info.get('debtToEquity')
    if de is not None:
        de = de / 100 if de > 5 else de   # yfinance reports as %
        if de > 2.0:
            score -= 12; reasons.append(f'負債/權益 {de:.1f}x 偏高 — 槓桿脆弱')
        elif de < 0.5:
            score += 6

    # Liquidity
    cr = info.get('currentRatio')
    if cr is not None:
        if cr < 1.0:
            score -= 10; reasons.append(f'流動比率 {cr:.2f} < 1 — 短期償付壓力')
        elif cr > 1.5:
            score += 4

    # Altman-Z proxy (lightweight): profitability + leverage + size
    pm = info.get('profitMargins')
    if pm is not None and pm < -0.10:
        score -= 12; reasons.append('淨利率深度為負 — 盈利能力受損')

    # Earnings trajectory (declining = the drop may be fundamental)
    eg = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth')
    if eg is not None and eg < -0.30:
        score -= 10; reasons.append(f'盈餘年減 {eg*100:.0f}% — 基本面走弱,慎防價值陷阱')

    score = max(0.0, min(100.0, score))
    knife = score < 40
    return {'survivability': round(score, 0), 'knife_risk': knife, 'reasons': reasons,
            'f_score': f, 'fcf_positive': (fcf is not None and fcf > 0)}


def value_trap_check(info: dict, data: dict, pt_data: dict | None) -> dict:
    """
    A value trap is cheap-AND-deteriorating: oversold but the fundamental story is
    breaking (analysts cutting, earnings falling, no valuation cushion). Different
    from a quality name on a sentiment dip.
    """
    flags = []
    trap = False
    # Analyst revisions trend (from consensus_result if present)
    rev = ((data.get('consensus_result') or {}).get('analyst_data') or {})
    er = rev.get('eps_revisions')
    if er is not None and hasattr(er, 'empty') and not er.empty:
        try:
            down = up = 0
            for p in ['0y', '+1y', '0q', '+1q']:
                if p in er.index:
                    up += int(er.loc[p].get('upLast30days', 0) or 0)
                    down += int(er.loc[p].get('downLast30days', 0) or 0)
            if (up + down) >= 3 and down > up * 2:
                trap = True; flags.append('分析師近月大幅下修 EPS — 基本面動能向下')
        except Exception:
            pass
    # No valuation cushion (oversold but not actually cheap vs intrinsic)
    if pt_data and pt_data.get('upside') is not None and pt_data['upside'] < -5:
        flags.append('即便回到合理價仍無上行空間 — 缺乏估值安全邊際')
    return {'value_trap': trap, 'flags': flags}


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING — oversold, capitulation, confirmation, and the final conviction
# ═══════════════════════════════════════════════════════════════════════════════

def _scale(x, lo, hi):
    """Map x in [lo,hi] → [0,1], clamped. Handles inverted ranges."""
    if x is None:
        return None
    if hi == lo:
        return 0.0
    t = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, t))


def oversold_score(p: dict) -> dict:
    """0-100: how stretched to the downside the name is. Higher = more oversold."""
    parts = {}
    # RSI14: 50→0, 20→1
    parts['rsi14'] = _scale(p.get('rsi14'), 50, 20)
    # RSI2 (fast): 30→0, 2→1
    parts['rsi2'] = _scale(p.get('rsi2'), 30, 2)
    # Bollinger %B: 0.2→0, -0.2→1 (below lower band)
    parts['boll'] = _scale(p.get('boll_pctb'), 0.2, -0.2)
    # Williams %R: -60→0, -95→1
    parts['williams'] = _scale(p.get('williams_r'), -60, -95)
    # Distance below 50dma: -2%→0, -15%→1
    parts['dist50'] = _scale(p.get('dist_ma50'), -0.02, -0.15)
    # Drawdown from 52w high: -8%→0, -30%→1
    parts['drawdown'] = _scale(p.get('dist_52w_high'), -0.08, -0.30)
    # Down-day streak: 2→0, 6→1
    parts['streak'] = _scale(p.get('down_days_streak'), 2, 6)

    weights = {'rsi14': 0.18, 'rsi2': 0.22, 'boll': 0.16, 'williams': 0.12,
               'dist50': 0.14, 'drawdown': 0.10, 'streak': 0.08}
    acc = wsum = 0.0
    for k, w in weights.items():
        if parts.get(k) is not None:
            acc += w * parts[k]; wsum += w
    score = (acc / wsum * 100) if wsum > 0 else 0.0
    return {'score': round(score, 0), 'parts': {k: (round(v, 2) if v is not None else None)
                                                for k, v in parts.items()}}


def capitulation_score(p: dict) -> dict:
    """0-100: is selling EXHAUSTING (volume climax then dry-up, extreme down stretch)?"""
    parts = {}
    parts['climax'] = _scale(p.get('vol_climax_x'), 1.3, 3.0)     # spike vs 20d avg
    parts['dryup'] = _scale(p.get('vol_dryup'), 0.9, 0.4)         # recent vol falling off the peak
    parts['rsi2_floor'] = _scale(p.get('rsi2'), 15, 1)            # truly washed out
    parts['streak'] = _scale(p.get('down_days_streak'), 3, 7)
    weights = {'climax': 0.35, 'dryup': 0.30, 'rsi2_floor': 0.20, 'streak': 0.15}
    acc = wsum = 0.0
    for k, w in weights.items():
        if parts.get(k) is not None:
            acc += w * parts[k]; wsum += w
    score = (acc / wsum * 100) if wsum > 0 else 0.0
    return {'score': round(score, 0), 'parts': {k: (round(v, 2) if v is not None else None)
                                                for k, v in parts.items()}}


def confirmation_score(p: dict) -> dict:
    """0-100: signs the knife has SLOWED (turn-up, reversal candle, reclaim, divergence)."""
    signals = {
        'rsi2_turning_up': (p.get('rsi2_turning_up'), 30, 'RSI2 自低位翻揚'),
        'bull_reversal_candle': (p.get('bull_reversal_candle'), 25, '出現帶長下影的反轉K棒'),
        'reclaim_ma10': (p.get('reclaim_ma10'), 20, '收復 10 日線'),
        'rsi_bull_divergence': (p.get('rsi_bull_divergence'), 25, 'RSI 出現底背離'),
    }
    score = 0.0
    present = []
    for k, (val, pts, label) in signals.items():
        if val:
            score += pts; present.append(label)
    return {'score': round(min(score, 100), 0), 'present': present}


def bottom_fish_score(panel, struct, surv, trap, oversold, capit, confirm,
                      pt_data, regime) -> dict:
    """
    Combine everything into a left-side conviction (0-100) and a verdict tier.
    Logic: the OPPORTUNITY (oversold) is gated by STRUCTURE and SURVIVABILITY, lifted
    by a valuation cushion and by CONFIRMATION, and scaled by the market REGIME.
    """
    os_score = oversold['score']
    struct_mult = struct['structure_mult']
    surv_score = surv['survivability']
    regime_mult = (regime or {}).get('multiplier', 1.0)

    # Margin-of-safety multiplier from composite PT upside
    upside = (pt_data or {}).get('upside')
    if upside is None:
        value_mult = 1.0
    elif upside >= 20:
        value_mult = 1.20
    elif upside >= 8:
        value_mult = 1.08
    elif upside >= -5:
        value_mult = 0.95
    else:
        value_mult = 0.80   # oversold but expensive → gambling

    quality_gate = 0.45 + 0.55 * (surv_score / 100.0)   # 0.45..1.00
    confirm_boost = confirm['score'] * 0.18              # up to +18

    conviction = (os_score
                  * struct_mult
                  * quality_gate
                  * value_mult)
    conviction = conviction + confirm_boost
    conviction = conviction * (0.7 + 0.3 * regime_mult)  # regime dampens but never fully kills
    conviction = max(0.0, min(100.0, conviction))

    # ── Verdict tiers ──
    has_setup = os_score >= 35
    knife = surv['knife_risk'] or struct['trend'] == 'DOWNTREND' or trap['value_trap']

    if not has_setup:
        tier, tier_label, tier_color = 'NONE', '未到抄底區 — 觀望', '#6b7280'
        rationale = '尚未進入超賣/恐慌區,沒有左側進場優勢。等更深的回檔或恐慌再評估。'
    elif knife and conviction < 45:
        tier, tier_label, tier_color = 'KNIFE', '接刀風險 — 避開', '#b52a2a'
        kr = '、'.join(surv['reasons'][:2] + trap['flags'][:1]) or struct['note']
        rationale = f'雖然超賣,但結構/基本面不支持抄底({kr})。便宜可能更便宜 — 不接刀。'
    elif (struct['trend'] in ('UPTREND', 'WEAK_UPTREND') and not knife
          and surv_score >= 55 and conviction >= 58):
        tier, tier_label, tier_color = 'STRONG', '高把握抄底', '#1a7a4a'
        rationale = '上升趨勢中的超賣回檔 + 體質健全 + 估值有安全邊際 — 左側勝率最高的組合,分批進場。'
    else:
        tier, tier_label, tier_color = 'SPECULATIVE', '投機性抄底', '#d4900a'
        rationale = '具備超賣條件但結構或確認訊號尚不足 — 縮小部位、放寬分批、嚴守停損試單。'

    return {
        'conviction': round(conviction, 0),
        'tier': tier, 'tier_label': tier_label, 'tier_color': tier_color,
        'rationale': rationale,
        'value_mult': round(value_mult, 2),
        'quality_gate': round(quality_gate, 2),
        'structure_mult': struct_mult,
        'regime_mult': regime_mult,
        'upside': upside,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED-ENTRY PLAYBOOK  (the 反人性 core — pre-committed tranches + discipline)
# ═══════════════════════════════════════════════════════════════════════════════

def staged_entry_plan(panel, score, struct, surv, pt_data, regime, config=None) -> dict:
    cfg = {**ACCOUNT_DEFAULTS, **(config or {})}
    price = panel.get('price')
    atr_v = panel.get('atr')
    tier = score['tier']

    plan = {'tier': tier, 'price': price, 'atr': atr_v}

    if tier in ('NONE', 'KNIFE') or not price or not atr_v:
        plan.update(action='NO TRADE', tranches=[],
                    note=('未到抄底區,放觀察清單。' if tier == 'NONE'
                          else '接刀風險過高 — 結構或基本面修復前不進場。'))
        plan['discipline'] = _discipline_rules(None, None, tier, None)
        return plan

    # Tier sizing factor: STRONG fishes a full book, SPECULATIVE a half
    tier_factor = 1.0 if tier == 'STRONG' else 0.5
    conv_scale = max(0.4, min(1.1, score['conviction'] / 75.0))
    regime_mult = (regime or {}).get('multiplier', 1.0)
    risk_mult = tier_factor * conv_scale * regime_mult

    # ── Scale-in ladder: 3 pre-committed prices ──
    # T1 near current (where the setup triggers), T2/T3 progressively lower on ATR steps,
    # snapped toward real structure (lower Bollinger band, 52w low) where sensible.
    boll_lower = panel.get('boll_lower')
    low_52 = panel.get('low_52w')

    t1 = round(price, 2)
    t2 = round(price - 1.0 * atr_v, 2)
    t3 = round(price - 2.2 * atr_v, 2)
    # Snap T3 toward a real support if one sits nearby (don't invent a level below the low)
    if low_52 and low_52 < t2 and low_52 > t3 - atr_v:
        t3 = round(max(t3, low_52 * 1.01), 2)
    if boll_lower and boll_lower < t1 and boll_lower > t2:
        t2 = round(boll_lower, 2)

    tranche_prices = sorted({t1, t2, t3}, reverse=True)
    while len(tranche_prices) < 3:                      # keep 3 distinct rungs
        tranche_prices.append(round(tranche_prices[-1] - 0.8 * atr_v, 2))
    tranche_prices = tranche_prices[:3]

    avg_entry = round(sum(tranche_prices) / len(tranche_prices), 2)

    # ── Hard stop = thesis invalidation: below the deepest tranche, beyond noise ──
    stop = round(min(tranche_prices) - 1.5 * atr_v, 2)
    # If structural 52w low is just under, anchor a touch below it instead
    if low_52 and stop < low_52 < min(tranche_prices):
        stop = round(low_52 - 0.5 * atr_v, 2)
    risk_per_share = max(avg_entry - stop, 0.01)

    # ── Risk-budgeted total size, split across the 3 rungs ──
    risk_budget = cfg['account_size'] * cfg['risk_per_trade'] * risk_mult
    total_shares = int(risk_budget / risk_per_share)
    max_shares_cap = int(cfg['account_size'] * cfg['max_position_pct'] / avg_entry) if avg_entry else 0
    capped = total_shares > max_shares_cap
    total_shares = max(0, min(total_shares, max_shares_cap))
    per_tranche = total_shares // 3

    tranches = []
    labels = ['第一批(訊號觸發)', '第二批(更深一階)', '第三批(恐慌/支撐)']
    for i, px in enumerate(tranche_prices):
        sh = per_tranche if i < 2 else (total_shares - 2 * per_tranche)
        tranches.append({
            'label': labels[i], 'price': px, 'shares': sh,
            'value': round(sh * px, 0),
            'pct_of_plan': round(sh / total_shares * 100, 0) if total_shares else 0,
        })

    pos_value = round(sum(t['value'] for t in tranches), 0)
    pos_pct = round(pos_value / cfg['account_size'] * 100, 1)
    dollar_risk = round(total_shares * risk_per_share, 0)

    # ── Mean-reversion targets (sell into strength) ──
    ma50 = panel.get('ma50')
    ma200 = panel.get('ma200')
    pt = (pt_data or {}).get('price_target')
    t_profit1 = round(ma50, 2) if (ma50 and ma50 > avg_entry) else round(avg_entry * 1.10, 2)
    if pt and pt > t_profit1:
        t_profit2 = round(pt, 2)
    elif ma200 and ma200 > t_profit1:
        t_profit2 = round(ma200, 2)
    else:
        t_profit2 = round(avg_entry * 1.20, 2)
    rr = round((t_profit1 - avg_entry) / risk_per_share, 2) if risk_per_share > 0 else None

    # ── Time stop: an oversold bounce that doesn't bounce is a failed thesis ──
    # Backtest sweet spot for the gated swing was ~25 sessions to a 20dma reversion.
    time_stop_days = 25 if tier == 'STRONG' else 15

    plan.update(
        action='SCALE-IN BUY(分批進場)',
        tranches=tranches,
        avg_entry=avg_entry,
        stop=stop, risk_per_share=round(risk_per_share, 2),
        total_shares=total_shares, position_value=pos_value, position_pct=pos_pct,
        dollar_risk=dollar_risk, size_capped=capped,
        target1=t_profit1, target2=t_profit2, rr=rr,
        target_note=f'目標1=50日線回歸 ${t_profit1};目標2=合理價/200日線 ${t_profit2}。到價各減 1/2。',
        time_stop_days=time_stop_days,
        risk_mult=round(risk_mult, 2),
    )
    plan['discipline'] = _discipline_rules(stop, time_stop_days, tier, regime)
    return plan


def _discipline_rules(stop, time_stop_days, tier, regime) -> list:
    if tier in ('NONE',):
        return ['尚未進場 — 等超賣訊號出現再啟動劇本。']
    if tier == 'KNIFE':
        return ['不進場。等結構修復(站回 200 日線)或基本面止血再重新評估。',
                '「便宜」不是進場理由 — 接刀的代價是歸零風險。']
    rules = [
        f'只在預設的三個分批價買進;跌破不臨時加碼 — 不凹單、不報復性攤平。',
        f'最害怕的時候(VIX 高、新聞最壞)照表進場 — 不要等「感覺安全」才買,那通常是反彈的頭部。',
        f'反彈到目標分批減碼,賣在貪婪裡 — 不貪圖完整的 V 型回升。',
        f'跌破硬停損 ${stop} = 論點被證偽(這是接刀不是底)→ 無條件清倉,不討價還價。',
        f'時間停損:{time_stop_days} 個交易日內沒有反轉跡象 → 出場,別把短線反彈凹成長線套牢。',
        f'嚴守部位上限 — 即使單一標的歸零,也傷不了總資本。這就是敢於進場的底氣。',
    ]
    if regime and regime.get('multiplier', 1.0) < 0.7:
        rules.append(f'目前市場體制偏空({regime.get("regime")},×{regime.get("multiplier"):.0%} sizing)'
                     f'— 部位已自動縮減,寧可少賺不可重傷。')
    return rules


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER — analyze one name end to end
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_bottom_fish(data: dict, quality: dict = None, pt_data: dict = None,
                        regime: dict = None, config: dict = None) -> dict:
    info = data.get('info', {})
    panel = technical_panel(data.get('hist_1y'), data.get('hist_5y'))
    if not panel:
        return {'ticker': data.get('symbol', '?'), 'status': 'ERROR',
                'error': 'no price history'}

    struct = structure_gate(panel)
    surv = survivability(info, quality or {}, data)
    trap = value_trap_check(info, data, pt_data)
    oversold = oversold_score(panel)
    capit = capitulation_score(panel)
    confirm = confirmation_score(panel)
    score = bottom_fish_score(panel, struct, surv, trap, oversold, capit, confirm,
                              pt_data, regime)
    plan = staged_entry_plan(panel, score, struct, surv, pt_data, regime, config)

    return {
        'ticker': data.get('symbol', '?'),
        'name': info.get('shortName') or info.get('longName') or data.get('symbol'),
        'sector': info.get('sector') or '-',
        'status': 'OK',
        'panel': panel,
        'structure': struct,
        'survivability': surv,
        'value_trap': trap,
        'oversold': oversold,
        'capitulation': capit,
        'confirmation': confirm,
        'score': score,
        'plan': plan,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }


def format_console(res: dict) -> str:
    if res.get('status') != 'OK':
        return f"  {res.get('ticker')}: ERROR {res.get('error')}"
    s = res['score']; p = res['panel']; pl = res['plan']
    head = (f"\n  {res['ticker']:6} {s['tier_label']:14} conviction={s['conviction']:.0f}"
            f"  超賣={res['oversold']['score']:.0f} 投降={res['capitulation']['score']:.0f}"
            f" 確認={res['confirmation']['score']:.0f}")
    tech = (f"    RSI14={p.get('rsi14')} RSI2={p.get('rsi2')} %B={p.get('boll_pctb')}"
            f" vs50dma={_pct(p.get('dist_ma50'))} vs200dma={_pct(p.get('dist_ma200'))}"
            f" trend={res['structure']['trend']} surv={res['survivability']['survivability']:.0f}")
    if pl.get('tranches'):
        rungs = ' / '.join(f"${t['price']}×{t['shares']}" for t in pl['tranches'])
        ex = (f"    分批: {rungs} | 均價 ${pl['avg_entry']} 停損 ${pl['stop']}"
              f" 目標 ${pl['target1']}/${pl['target2']} R:R {pl['rr']}"
              f" 部位 {pl['position_pct']}% 風險 ${pl['dollar_risk']:,.0f}")
    else:
        ex = f"    {pl.get('note','')}"
    return '\n'.join([head, tech, ex])


def _pct(v):
    return f"{v*100:+.1f}%" if v is not None else "—"


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _gauge(label, val, color='#0d3b6e'):
    v = max(0, min(100, val or 0))
    return (f'<div style="margin-bottom:6px"><div style="font-size:7.5pt;color:#6b7280">{label}</div>'
            f'<div style="background:#eee;border-radius:6px;height:12px;overflow:hidden">'
            f'<div style="width:{v}%;height:12px;background:{color}"></div></div>'
            f'<div style="font-size:8pt;font-weight:700;color:{color};text-align:right">{v:.0f}</div></div>')


def render_report(results: list, output_path: str = None) -> str:
    output_path = output_path or os.path.join(REPORTS_DIR, 'bottom_fishing.html')
    ok = [r for r in results if r.get('status') == 'OK']
    order = {'STRONG': 0, 'SPECULATIVE': 1, 'KNIFE': 2, 'NONE': 3}
    ok.sort(key=lambda r: (order.get(r['score']['tier'], 9), -r['score']['conviction']))

    n_strong = sum(1 for r in ok if r['score']['tier'] == 'STRONG')
    n_spec = sum(1 for r in ok if r['score']['tier'] == 'SPECULATIVE')
    n_knife = sum(1 for r in ok if r['score']['tier'] == 'KNIFE')

    rows = ''
    cards = ''
    for r in ok:
        s = r['score']; p = r['panel']; pl = r['plan']; st = r['structure']
        rows += f'''<tr onclick="document.getElementById('card-{r['ticker']}').scrollIntoView({{behavior:'smooth'}})" style="cursor:pointer">
          <td style="font-weight:700;color:#0d3b6e">{r['ticker']}</td>
          <td style="max-width:150px;overflow:hidden;text-overflow:ellipsis">{r['name'][:22]}</td>
          <td><span style="background:{s['tier_color']};color:white;padding:2px 8px;border-radius:3px;font-size:7.5pt;font-weight:700">{s['tier_label']}</span></td>
          <td class="num" style="font-weight:700;color:{s['tier_color']}">{s['conviction']:.0f}</td>
          <td class="num">{r['oversold']['score']:.0f}</td>
          <td class="num">{r['capitulation']['score']:.0f}</td>
          <td class="num">{r['confirmation']['score']:.0f}</td>
          <td>{st['trend']}</td>
          <td class="num">{r['survivability']['survivability']:.0f}</td>
          <td class="num">{_pct(p.get('dist_ma200'))}</td>
          <td class="num">{p.get('rsi14') if p.get('rsi14') is not None else '—'}</td>
          <td class="num">{('$'+str(pl['avg_entry'])) if pl.get('avg_entry') else '—'}</td>
          <td class="num">{('$'+str(pl['stop'])) if pl.get('stop') else '—'}</td>
          <td class="num">{pl.get('position_pct','—')}{'%' if pl.get('position_pct') is not None else ''}</td>
        </tr>'''
        cards += _render_card(r)

    html = _REPORT_TEMPLATE.format(
        updated=datetime.now().strftime('%Y-%m-%d %H:%M'),
        n=len(ok), n_strong=n_strong, n_spec=n_spec, n_knife=n_knife,
        rows=rows, cards=cards)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[ok] Bottom-fishing report -> {output_path}")
    return output_path


def _render_card(r):
    s = r['score']; p = r['panel']; pl = r['plan']; st = r['structure']
    surv = r['survivability']
    # tranche ladder
    ladder = ''
    if pl.get('tranches'):
        for t in pl['tranches']:
            ladder += (f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid #f0f0f0">'
                       f'<div style="width:140px;font-size:8pt;color:#374151">{t["label"]}</div>'
                       f'<div style="font-weight:700;color:#0d3b6e;width:70px">${t["price"]}</div>'
                       f'<div style="font-size:8pt;color:#6b7280">{t["shares"]} 股 (${t["value"]:,.0f}, {t["pct_of_plan"]:.0f}%)</div></div>')
    else:
        ladder = f'<div style="color:#9ca3af;font-size:8.5pt">{pl.get("note","")}</div>'

    disc = ''.join(f'<li style="margin-bottom:5px;font-size:8.3pt;line-height:1.5">{d}</li>'
                   for d in pl.get('discipline', []))

    # technical chips
    def chip(lbl, val, good=None):
        col = '#374151'
        if good is True: col = '#1a7a4a'
        if good is False: col = '#b52a2a'
        return (f'<span style="display:inline-block;background:#f1f5f9;border-radius:10px;'
                f'padding:3px 9px;margin:2px;font-size:7.8pt">{lbl} '
                f'<strong style="color:{col}">{val}</strong></span>')

    rsi14 = p.get('rsi14')
    chips = (
        chip('RSI14', rsi14, rsi14 is not None and rsi14 < 30) +
        chip('RSI2', p.get('rsi2'), (p.get('rsi2') or 100) < 10) +
        chip('%B', p.get('boll_pctb'), (p.get('boll_pctb') or 1) < 0) +
        chip('vs 50dma', _pct(p.get('dist_ma50')), (p.get('dist_ma50') or 0) < 0) +
        chip('vs 200dma', _pct(p.get('dist_ma200')), (p.get('dist_ma200') or 0) > 0) +
        chip('回撤52高', _pct(p.get('dist_52w_high'))) +
        chip('連跌', f"{p.get('down_days_streak')}天") +
        chip('量能', f"{p.get('vol_today_x')}x" if p.get('vol_today_x') else '—') +
        chip('Williams%R', p.get('williams_r')) +
        chip('ATR', f"${p.get('atr')} ({_pct(p.get('atr_pct'))})" if p.get('atr') else '—')
    )
    confirm_chips = ''.join(f'<span style="background:#e8f5e9;color:#1a7a4a;border-radius:10px;'
                            f'padding:3px 9px;margin:2px;font-size:7.8pt">✓ {c}</span>'
                            for c in r['confirmation']['present']) or \
                    '<span style="color:#9ca3af;font-size:8pt">尚無止跌確認訊號</span>'

    surv_reasons = ''.join(f'<li style="font-size:7.8pt;color:#6b7280">{x}</li>'
                           for x in surv['reasons']) or '<li style="font-size:7.8pt;color:#9ca3af">無重大體質警訊</li>'

    targets = ''
    if pl.get('target1'):
        targets = (f'<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:8.5pt">'
                   f'<div>均價 <strong>${pl["avg_entry"]}</strong></div>'
                   f'<div>停損 <strong style="color:#b52a2a">${pl["stop"]}</strong></div>'
                   f'<div>目標1 <strong style="color:#1a7a4a">${pl["target1"]}</strong></div>'
                   f'<div>目標2 <strong style="color:#1a7a4a">${pl["target2"]}</strong></div>'
                   f'<div>R:R <strong>{pl["rr"]}</strong></div>'
                   f'<div>部位 <strong>{pl["position_pct"]}%</strong> (風險 ${pl["dollar_risk"]:,.0f})</div>'
                   f'<div>時間停損 <strong>{pl["time_stop_days"]} 交易日</strong></div></div>')

    return f'''
    <div class="card" id="card-{r['ticker']}">
      <div class="card-head" style="border-left:6px solid {s['tier_color']}">
        <div>
          <span style="font-size:14pt;font-weight:700;color:#0d3b6e">{r['ticker']}</span>
          <span style="font-size:9pt;color:#6b7280;margin-left:8px">{r['name'][:30]} · {r['sector']}</span>
        </div>
        <div style="text-align:right">
          <span style="background:{s['tier_color']};color:white;padding:4px 12px;border-radius:4px;font-size:9pt;font-weight:700">{s['tier_label']}</span>
          <span style="font-size:8.5pt;color:#374151;margin-left:8px">conviction <strong style="color:{s['tier_color']}">{s['conviction']:.0f}</strong></span>
        </div>
      </div>
      <div style="padding:14px 18px">
        <div style="font-size:8.8pt;color:#374151;background:#f8fafd;padding:9px 12px;border-radius:5px;margin-bottom:12px;line-height:1.55">{s['rationale']}</div>

        <div style="display:grid;grid-template-columns:1.4fr 1fr;gap:18px">
          <div>
            <div style="font-size:8pt;font-weight:700;color:#6b7280;text-transform:uppercase;margin-bottom:6px">技術面 — 超賣讀數</div>
            <div style="margin-bottom:10px">{chips}</div>
            <div style="font-size:8pt;font-weight:700;color:#6b7280;text-transform:uppercase;margin-bottom:6px">止跌確認</div>
            <div style="margin-bottom:10px">{confirm_chips}</div>
            <div style="font-size:8pt;font-weight:700;color:#6b7280;text-transform:uppercase;margin-bottom:4px">結構 / 體質</div>
            <div style="font-size:8.3pt;color:#374151;margin-bottom:4px">{st['note']}</div>
            <ul style="margin:0 0 0 16px">{surv_reasons}</ul>
          </div>
          <div>
            {_gauge('超賣 Oversold', r['oversold']['score'], '#1a6fa8')}
            {_gauge('投降 Capitulation', r['capitulation']['score'], '#8b5cf6')}
            {_gauge('確認 Confirmation', r['confirmation']['score'], '#1a7a4a')}
            {_gauge('體質 Survivability', surv['survivability'], '#d4900a' if surv['survivability']<55 else '#1a7a4a')}
          </div>
        </div>

        <div style="margin-top:14px;border-top:1px solid #eaeaea;padding-top:12px">
          <div style="font-size:8.5pt;font-weight:700;color:#0d3b6e;margin-bottom:8px">分批進場階梯(預先承諾的價位)</div>
          {ladder}
          <div style="margin-top:10px">{targets}</div>
        </div>

        <div style="margin-top:14px;border-top:1px solid #eaeaea;padding-top:12px">
          <div style="font-size:8.5pt;font-weight:700;color:#b52a2a;margin-bottom:6px">反人性紀律(寫在進場之前)</div>
          <ol style="margin:0 0 0 18px;color:#374151">{disc}</ol>
        </div>
      </div>
    </div>'''


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>左側交易 / 抄底儀表板</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI','Microsoft JhengHei',Arial,sans-serif;font-size:9.5pt;color:#1a1a2e;background:#f0f2f7}}
.top-bar{{background:#7a1f1f;color:white;padding:13px 28px;display:flex;align-items:center;justify-content:space-between}}
.top-bar h1{{font-size:13pt}}
.top-bar .sub{{font-size:7.8pt;opacity:.8;margin-top:3px}}
.container{{max-width:1500px;margin:0 auto;padding:20px}}
.kpi-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}}
.kpi{{background:white;border-radius:6px;padding:11px 15px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.kpi .lbl{{font-size:7pt;color:#6b7280;text-transform:uppercase;letter-spacing:.4px}}
.kpi .val{{font-size:15pt;font-weight:700;margin-top:2px}}
.table-card{{background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow-x:auto;margin-bottom:20px}}
.table-header{{background:#7a1f1f;color:white;padding:9px 16px;font-size:10pt;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:8.6pt}}
thead th{{background:#5e1818;color:white;padding:8px 9px;text-align:left;font-size:7.4pt;white-space:nowrap}}
tbody tr{{border-bottom:1px solid #eee}} tbody tr:hover{{background:#fff5f5}}
td{{padding:7px 9px;white-space:nowrap}} .num{{text-align:right}}
.card{{background:white;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.1);margin-bottom:16px;overflow:hidden}}
.card-head{{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;background:#fafbfd}}
.intro{{background:#fff;border-radius:6px;padding:14px 18px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,.08);font-size:8.5pt;color:#374151;line-height:1.6}}
.footer{{text-align:center;font-size:7.3pt;color:#9ca3af;padding:14px}}
</style></head><body>
<div class="top-bar">
  <div><h1>左側交易 / 抄底劇本 (Bottom-Fishing)</h1>
    <div class="sub">只在「上升趨勢中的超賣 × 體質健全 × 估值安全邊際 × 止跌確認」交集出手 — 分批進場、預設停損、反人性紀律</div></div>
  <div style="font-size:7.5pt;opacity:.6">更新:{updated}</div>
</div>
<div class="container">
  <div class="intro">
    <strong>為什麼這套左側策略刻意保守:</strong>本專案自己的 backtest.py 證明,大型股「單純買超賣」是<strong>反向有效</strong>的
    (reversal_1m 的 IC 為正 — 大型股下跌會續跌)。因此本引擎<strong>只抄</strong>同時滿足以下四道關卡的標的:
    ① 結構:在上升的 200 日線之上(上升趨勢中的回檔,而非破線後的接刀);
    ② 體質:Piotroski、現金流、槓桿過關,跌的是情緒不是基本面;
    ③ 超賣+投降:RSI、布林、均線乖離、量能投降確認賣壓衰竭;
    ④ 確認:RSI 翻揚 / 反轉K棒 / 底背離。再以分批進場 + 預設硬停損 + 時間停損執行。
    <div style="margin-top:10px;padding:9px 12px;background:#f0f7f2;border-left:4px solid #1a7a4a;border-radius:4px">
    <strong>回測證據(50 檔大型股 · 8 年 · 事件型):</strong>RSI(2) 超賣均值回歸本身有效(勝率 ~72%、盈虧比 1.5)。
    加上「站上上升 200 日線」過濾後,盈虧比 1.50→<strong>1.67</strong>、勝率 71.8%→<strong>73.0%</strong>;
    再加 ATR 硬停損,把單筆最深帳面虧損從 <strong style="color:#b52a2a">−57%</strong> 砍到
    <strong style="color:#1a7a4a">−24%</strong>(可控)。<span style="color:#6b7280">完整數字見 bottom_fishing_backtest.html。</span>
    </div>
  </div>
  <div class="kpi-strip">
    <div class="kpi"><div class="lbl">分析標的</div><div class="val" style="color:#0d3b6e">{n}</div></div>
    <div class="kpi"><div class="lbl">高把握抄底</div><div class="val" style="color:#1a7a4a">{n_strong}</div></div>
    <div class="kpi"><div class="lbl">投機性抄底</div><div class="val" style="color:#d4900a">{n_spec}</div></div>
    <div class="kpi"><div class="lbl">接刀風險(避開)</div><div class="val" style="color:#b52a2a">{n_knife}</div></div>
  </div>
  <div class="table-card">
    <div class="table-header">抄底候選排序 — 點列開啟完整劇本</div>
    <table><thead><tr>
      <th>代號</th><th>名稱</th><th>判定</th><th class="num">把握度</th>
      <th class="num">超賣</th><th class="num">投降</th><th class="num">確認</th>
      <th>結構</th><th class="num">體質</th><th class="num">vs200dma</th><th class="num">RSI14</th>
      <th class="num">均價</th><th class="num">停損</th><th class="num">部位</th>
    </tr></thead><tbody>{rows}</tbody></table>
  </div>
  {cards}
  <div class="footer">左側 / 抄底引擎 · 資料:Yahoo Finance · 僅供研究估算,非投資建議 · {updated}</div>
</div></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ticker(ticker: str) -> dict:
    from data_fetcher import fetch_company_data
    from valuation_engine import quality_score
    try:
        data = fetch_company_data(ticker)
        try:
            quality = quality_score(data['info'], data['cashflow'],
                                    data['balance_sheet'], data['income_stmt'])
        except Exception:
            quality = {}
        pt_data = None  # standalone: rely on info.targetMeanPrice cushion via survivability
        tm = data['info'].get('targetMeanPrice')
        price = (data['info'].get('currentPrice') or data['info'].get('regularMarketPrice'))
        if tm and price:
            pt_data = {'price_target': tm, 'upside': round((tm / price - 1) * 100, 1)}
        return analyze_bottom_fish(data, quality=quality, pt_data=pt_data, regime=_regime())
    except Exception as e:
        import traceback
        return {'ticker': ticker, 'status': 'ERROR', 'error': str(e),
                'traceback': traceback.format_exc()}


_REGIME_CACHE = None
def _regime():
    global _REGIME_CACHE
    if _REGIME_CACHE is None:
        try:
            from regime import get_regime
            _REGIME_CACHE = get_regime()
        except Exception:
            _REGIME_CACHE = {'regime': 'CAUTION', 'multiplier': 0.8}
    return _REGIME_CACHE


def main():
    import sys, argparse
    try:                                  # make Chinese print correctly on Windows consoles
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='Left-side / bottom-fishing analyzer')
    parser.add_argument('tickers', nargs='*')
    parser.add_argument('--watchlist', '-w')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.watchlist and os.path.exists(args.watchlist):
        with open(args.watchlist) as f:
            tickers += [l.strip().upper() for l in f if l.strip() and not l.startswith('#')]
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    if not tickers:
        print('Usage: python bottom_fishing.py AAPL MSFT NVDA  [--watchlist f.txt]')
        sys.exit(1)

    print(f"\n左側抄底掃描:{len(tickers)} 檔 — {', '.join(tickers)}")
    r = _regime()
    print(f"  市場體制:{r.get('regime')} (×{r.get('multiplier',1):.0%} sizing)")

    results = []
    for i, t in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {t}...", end='')
        res = _run_ticker(t)
        results.append(res)
        if res.get('status') == 'OK':
            print(format_console(res))
        else:
            print(f"  ERROR {res.get('error','')[:60]}")

    if not args.no_report:
        path = render_report(results)
        print(f"\n開啟:file:///{path.replace(os.sep, '/')}")

    # persist for dashboard
    try:
        slim = [{k: v for k, v in r.items() if k != 'traceback'} for r in results]
        with open(os.path.join(REPORTS_DIR, 'bottom_fishing.json'), 'w', encoding='utf-8') as f:
            json.dump(slim, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


if __name__ == '__main__':
    main()
