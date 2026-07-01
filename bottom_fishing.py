"""
⚠ DEPRECATED (2026-06) — superseded by capitulation_engine.py (the clean rewrite).
This file grew into a palimpsest; its casual-oversold core was proven to have NO edge
(see attribution.py). The validated logic now lives, clean, in `capitulation_engine.py`.
Kept only for reference / historical diff. Use:  python capitulation_engine.py NVDA ...
═══════════════════════════════════════════════════════════════════════════════

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

# Scale-in tranche WEIGHTS across the 3 price rungs (ordered high→low price).
# STRONG fishes HEAVIER into the deeper, more-oversold rungs — Connors/Alvarez: the deeper
# the RSI(2) washout the higher the reversion expectancy, AND a lower weighted avg_entry
# lifts the reward to a fixed 20dma target, which lets the structural stop sit wider without
# breaking R:R≥1 (fewer noise stop-outs → better win rate too). This is only justified for
# STRONG because it sits in an uptrend behind the hard stop. A SPECULATIVE probe stays
# balanced (you are less sure, so don't commit your biggest clip into the deepest hole).
TRANCHE_WEIGHTS_STRONG = [0.25, 0.35, 0.40]
TRANCHE_WEIGHTS_PROBE  = [0.34, 0.33, 0.33]

# Exit knobs
SCALEOUT_T1 = 2.0 / 3.0       # fraction sold at target1 (edge concentrates in the first
                             # reversion; bank the bulk, leave a 1/3 runner to target2)
TIME_STOP_STRONG = 12        # RSI(2) reversion completes in ~3-10 sessions; 25 let failures rot
TIME_STOP_PROBE  = 7
# Signal-based early exit: the cleanest RSI(2) exit is the signal RE-NORMALISING, not a fixed
# price. If price hasn't reached target1 but momentum has clearly turned, bank the target1 clip.
SIGNAL_EXIT_RSI2 = 65.0

# ── CAPITULATION thresholds (the crowd-fear entry; tuned to the backtest sweet spot) ──
# The loosening-ladder sweep (capitulation_bt.py) showed the whole family beats the old core,
# degrading gracefully. We sit at the 'sweet spot': deep (not most-extreme) drawdown + extreme
# RSI(2) + a real volume climax, and require the TURN (cheap, but cuts worst-MAE −46%→−31%).
# STRONG (full size) additionally needs whole-market fear (VIX≥CAPIT_FEAR_VIX) — the single
# strongest quality knob (PF jumps to ~4+ when the whole crowd panics together).
CAPIT_DD       = -0.15    # ≥15% off the 252d high (was −20% extreme; this is the practical ③/④ point)
CAPIT_RSI2     = 5.0      # extreme oversold in the last 5 sessions (kept strict — it's cheap)
CAPIT_CLIMAX   = 2.0      # a down day with ≥2.0× its 50d avg volume (panic puke)
CAPIT_FEAR_VIX = 20.0     # market-wide fear that upgrades an individual capitulation to full size


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


# ── Benchmark (for residual / idiosyncratic oversold) ────────────────────────
# Short-term reversal is far cleaner on the STOCK-SPECIFIC component (residual
# reversal) than on raw returns: a name oversold because it fell on its OWN news
# mean-reverts reliably; a name merely dragged down by a falling market (pure
# beta) does not bounce until the market does. We fetch SPY once per process and
# beta-adjust the recent drop so the engine fishes idiosyncratic washouts.
_BENCH_CACHE = None
_BENCH_LOADED = False
def _benchmark_returns():
    global _BENCH_CACHE, _BENCH_LOADED
    if not _BENCH_LOADED:
        _BENCH_LOADED = True
        try:
            import yfinance as yf
            s = yf.download('SPY', period='1y', interval='1d',
                            auto_adjust=True, progress=False)['Close']
            if hasattr(s, 'columns'):
                s = s.iloc[:, 0]
            _BENCH_CACHE = s.pct_change().dropna()
        except Exception:
            _BENCH_CACHE = None
    return _BENCH_CACHE


def _residual_drop(close: pd.Series, bench_ret: pd.Series,
                   lookback: int = 20, beta_win: int = 120):
    """Stock's lookback return MINUS beta×benchmark return = idiosyncratic drop.
    Very negative = the name fell hard on its OWN account (clean reversion fuel)."""
    if bench_ret is None or close is None:
        return None, None
    try:
        def _naive(s):                     # align tz-aware stock vs tz-naive benchmark
            s = s.copy()
            idx = s.index
            if getattr(idx, 'tz', None) is not None:
                idx = idx.tz_localize(None)
            s.index = idx.normalize()
            return s
        r = _naive(close.pct_change().dropna())
        b = _naive(bench_ret.dropna())
        b = b.reindex(r.index).dropna()
        r = r.reindex(b.index).dropna()
        if len(r) < beta_win // 2:
            return None, None
        rw, bw = r.tail(beta_win), b.tail(beta_win)
        var = float(bw.var())
        if var <= 0:
            return None, None
        beta = float(((rw - rw.mean()) * (bw - bw.mean())).mean() / var)
        beta = max(0.0, min(3.0, beta))
        stock_ret = float((1 + r.tail(lookback)).prod() - 1)
        bench_r = float((1 + b.tail(lookback)).prod() - 1)
        return round(stock_ret - beta * bench_r, 4), round(beta, 2)
    except Exception:
        return None, None


def _drawdown_part(ddh):
    """Win-rate-shaped drawdown credit (inverted-U, not monotone).
    RSI(2) reversion hits highest in SHALLOW pullbacks of strong names; a −30%+
    'dip' inside a nominal uptrend is usually a regime change starting, with a low
    bounce rate. So: full credit in the −8%..−18% sweet spot, decaying for deeper
    holes — but never to zero (deep washouts keep payoff, just not win-rate)."""
    if ddh is None:
        return None
    d = -ddh  # positive depth
    if d < 0.08:
        return (_scale(d, 0.03, 0.08) or 0.0) * 0.7   # too shallow → partial
    if d <= 0.18:
        return 1.0                                     # sweet spot
    if d <= 0.30:
        return 1.0 - (_scale(d, 0.18, 0.30) or 0.0) * 0.55   # 1.0 → ~0.45
    return 0.45                                         # very deep: keep some (賠率)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — TECHNICAL OVERSOLD READING
# ═══════════════════════════════════════════════════════════════════════════════

def technical_panel(hist_1y: pd.DataFrame, hist_5y: pd.DataFrame,
                    bench_ret: pd.Series = None) -> dict:
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
    out['high_52w'] = round(hi_52, 2) if hi_52 else None   # the "pain started here" satisfaction target
    lo_52 = float(close.tail(252).min()) if len(close) >= 30 else float(close.min())
    out['dist_52w_low'] = round(price / lo_52 - 1, 4) if lo_52 else None
    out['low_52w'] = round(lo_52, 2) if lo_52 else None
    # Recent swing low (~3m) — a nearer, more relevant support to anchor deep tranches to.
    try:
        out['swing_low'] = round(float(low.tail(60).min()), 2)
    except Exception:
        out['swing_low'] = None

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

    # IBS — Internal Bar Strength = (close-low)/(high-low). One of the most robust
    # single-stock mean-reversion signals (Alvarez Quant et al.): low IBS (<0.2) =
    # close near the bottom of its range = bullish for next-day reversion.
    try:
        h1, l1, c1 = float(high.iloc[-1]), float(low.iloc[-1]), price
        rng1 = h1 - l1
        out['ibs'] = round((c1 - l1) / rng1, 2) if rng1 > 0 else 0.5
    except Exception:
        out['ibs'] = None

    # Connors "Double 7s" — is today a new 7-day low? Simple, well-documented
    # short-term oversold trigger, independent of RSI/Bollinger.
    try:
        out['new_7d_low'] = bool(price <= float(close.tail(7).min()))
    except Exception:
        out['new_7d_low'] = False

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

    # ═══ TRUE CAPITULATION DETECTION (the heart of the rebuilt thesis) ═══════════
    # Psychology: people do NOT capitulate at a −8% dip; they capitulate after DEEP,
    # sustained pain (−20%+), in a VOLUME CLIMAX (the last weak hands puke), with the
    # oscillator pinned to the floor (relentless waterfall). That is maximum FEAR.
    #   capit_setup  = deep drawdown  ×  extreme oversold (recent)  ×  selling climax
    # But you do NOT buy while fear is still ACCELERATING (a falling knife). You wait for
    # THE TURN — the first bar where buyers step in (close up AND back above yesterday's
    # high) — when the crowd is still terrified and disbelieving. That disbelief is the edge.
    try:
        r2s = rsi(close, 2)
        out['rsi2_min_5d'] = round(float(r2s.tail(5).min()), 1)
    except Exception:
        out['rsi2_min_5d'] = out.get('rsi2')
    dd = out.get('dist_52w_high')
    climax_x = out.get('vol_climax_x')
    out['capit_setup'] = bool(
        (dd is not None and dd <= CAPIT_DD)                    # deep drawdown = real damage
        and ((out.get('rsi2_min_5d') if out.get('rsi2_min_5d') is not None else 99) < CAPIT_RSI2)  # extreme oversold
        and ((climax_x or 0) >= CAPIT_CLIMAX)                  # a panic volume climax occurred
    )
    try:
        prev_high = float(high.iloc[-2])
        out['the_turn'] = bool(price > float(close.iloc[-2]) and price > prev_high)  # buyers stepped in
    except Exception:
        out['the_turn'] = False
    # also accept a strong reversal candle / 10dma reclaim as a valid turn
    out['the_turn'] = bool(out['the_turn'] or out.get('bull_reversal_candle') or out.get('reclaim_ma10'))
    out['capitulation_buy'] = bool(out['capit_setup'] and out['the_turn'])

    # ── Residual / idiosyncratic drop (beta-adjusted) ────────────────────────
    # Trade the part of the fall that is the STOCK'S OWN, not market beta.
    resid, beta = _residual_drop(close, bench_ret, lookback=20, beta_win=120)
    out['resid_drop_20d'] = resid
    out['beta'] = beta

    # ── Gap / news-knife detection ───────────────────────────────────────────
    # Grind-down (陰跌) = sentiment → reverts. Gap-down on news (利空跳水) = the
    # drop ratified by fundamentals → weak reversion, value-trap odds. Separate them.
    try:
        opens = base['Open']
        gap = (opens / close.shift(1) - 1)
        out['gap_today'] = round(float(gap.iloc[-1]), 4)
        out['recent_gap_down'] = round(float(gap.tail(3).min()), 4)   # worst gap in trigger window
        out['news_gap'] = bool((out['recent_gap_down'] or 0) <= -0.07)
    except Exception:
        out['gap_today'] = out['recent_gap_down'] = None
        out['news_gap'] = False

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
# PANEL 4 — SENTIMENT  (情緒面: how the crowd/analysts feel — and which way it's turning)
# ═══════════════════════════════════════════════════════════════════════════════

def sentiment_panel(data: dict, info: dict, pt_data: dict | None) -> dict:
    """
    Contrarian sentiment read. Two things matter and they point opposite ways:
      • LEVEL  — extreme pessimism (everyone bearish) is contrarian FUEL for a bounce.
      • TREND  — analysts actively CUTTING numbers while it falls = the drop is being
                 ratified by fundamentals → value-trap risk, not a sentiment dip.
    We surface both so the scorecard can reward washed-out fear but penalise fresh cuts.
    """
    out = {}
    # Street recommendation level (1=Strong Buy … 5=Sell). High mean = hated = contrarian +.
    rec_mean = info.get('recommendationMean')
    out['rec_mean'] = round(float(rec_mean), 2) if rec_mean else None
    out['rec_key'] = info.get('recommendationKey')
    out['n_analysts'] = info.get('numberOfAnalystOpinions')

    # Target-price upside (analyst mean) — a cushion read independent of our own PT.
    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
    tm = info.get('targetMeanPrice')
    out['target_mean'] = round(float(tm), 2) if tm else None
    out['target_upside'] = round((tm / price - 1) * 100, 1) if (tm and price) else None

    # EPS revision trend (last 30d up vs down) — the single best "value-trap" tell.
    up = down = 0
    er = ((data.get('consensus_result') or {}).get('analyst_data') or {}).get('eps_revisions')
    if er is None:
        er = data.get('earnings_estimate')
    try:
        if er is not None and hasattr(er, 'index') and not er.empty:
            for p in ['0y', '+1y', '0q', '+1q']:
                if p in er.index:
                    up += int(er.loc[p].get('upLast30days', 0) or 0)
                    down += int(er.loc[p].get('downLast30days', 0) or 0)
    except Exception:
        pass
    out['rev_up_30d'] = up
    out['rev_down_30d'] = down
    if (up + down) >= 2:
        out['rev_net'] = up - down
        out['rev_trend'] = ('cutting' if down > up * 1.5 else
                            'raising' if up > down * 1.5 else 'mixed')
    else:
        out['rev_net'] = None
        out['rev_trend'] = 'n/a'

    # Composite sentiment posture for the contrarian: washed-out fear is good, but only
    # if numbers are NOT being cut (no fundamental ratification of the fall).
    pessimism = 0.0  # 0..1 how hated
    if out['rec_mean'] is not None:
        pessimism = _scale(out['rec_mean'], 2.5, 4.5) or 0.0   # 2.5→0, 4.5→1
    cutting = (out['rev_trend'] == 'cutting')
    out['contrarian_fuel'] = round(pessimism * (0.4 if cutting else 1.0), 2)
    out['numbers_cut'] = bool(cutting)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — POSITIONING  (籌碼面: short interest, days-to-cover, institutional float)
# ═══════════════════════════════════════════════════════════════════════════════

def positioning_panel(info: dict, data: dict) -> dict:
    """
    Who holds the shares and how crowded is the short side. For a bottom-fish:
      • Elevated short % + days-to-cover + an oversold reversal = squeeze fuel (bullish).
      • But a VERY high short on a broken name can be smart money being right (bearish);
        we only treat shorts as fuel when structure/survivability already pass.
      • Heavy institutional ownership = a name that gets defended on dips (support).
    """
    out = {}
    sp = info.get('shortPercentOfFloat')
    out['short_pct_float'] = round(float(sp) * 100, 1) if sp else None
    out['short_ratio'] = round(float(info.get('shortRatio')), 1) if info.get('shortRatio') else None
    inst = info.get('heldPercentInstitutions')
    out['inst_pct'] = round(float(inst) * 100, 1) if inst else None
    insd = info.get('heldPercentInsiders')
    out['insider_pct'] = round(float(insd) * 100, 1) if insd else None

    # Squeeze-fuel score 0..1: crowded short + slow to cover = more fuel if it turns.
    fuel = 0.0
    if out['short_pct_float'] is not None:
        fuel += (_scale(out['short_pct_float'], 4, 20) or 0.0) * 0.6   # 4%→0, 20%→full
    if out['short_ratio'] is not None:
        fuel += (_scale(out['short_ratio'], 2, 8) or 0.0) * 0.4        # 2d→0, 8d→full
    out['squeeze_fuel'] = round(min(fuel, 1.0), 2)
    out['crowded_short'] = bool(out['short_pct_float'] and out['short_pct_float'] >= 10)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TRADEABILITY GATE  (可交易性: liquidity, spread, earnings-event proximity)
# ═══════════════════════════════════════════════════════════════════════════════

def tradeability(info: dict, panel: dict, data: dict, config: dict | None = None) -> dict:
    """
    A signal you cannot execute cleanly is not an edge — it is a way to donate to the
    market maker. Bottom-fishing makes this acute: you buy oversold names IN A PANIC, when
    spreads gap wide and you (taking the first tranche MOC on the worst day) ARE the
    liquidity. And buying an 'oversold' name two days before earnings is not mean reversion
    — it is a coin flip that the engine's price/volume signals cannot see.

    This gate enforces three executability checks the live engine was missing:
      • LIQUIDITY  — dollar ADV must clear a floor (severe → NO TRADE) and a higher bar
                     to qualify for a full-size STRONG (soft → cap to SPECULATIVE).
      • SPREAD     — quoted spread (best-effort from info bid/ask) must be tight.
      • EARNINGS   — no fresh full-size entry inside the pre-earnings blackout window;
                     hold the dip thesis until the binary event is out of the way.
    """
    cfg = {**ACCOUNT_DEFAULTS, **(config or {})}
    min_adv = cfg.get('min_dollar_adv', 5e6)         # below → untradeable for a real book
    strong_adv = cfg.get('strong_dollar_adv', 20e6)  # below → no full-size STRONG
    max_spread = cfg.get('max_spread_pct', 0.012)    # 1.2% quoted spread ceiling
    blackout_days = cfg.get('earnings_blackout_days', 5)

    out = {'reasons': [], 'block': False, 'cap_tier': None}
    price = panel.get('price') or info.get('currentPrice') or info.get('regularMarketPrice')

    # ── Dollar ADV (20d avg volume × price), hist preferred, info as fallback ──
    avg_vol = None
    try:
        hv = (data.get('hist_1y') or {})
        vol = hv['Volume'] if hasattr(hv, '__getitem__') and 'Volume' in getattr(hv, 'columns', []) else None
        if vol is not None and len(vol.dropna()) >= 20:
            avg_vol = float(vol.dropna().tail(20).mean())
    except Exception:
        avg_vol = None
    if avg_vol is None:
        avg_vol = info.get('averageVolume10days') or info.get('averageVolume') or info.get('averageDailyVolume10Day')
    dollar_adv = (float(avg_vol) * float(price)) if (avg_vol and price) else None
    out['dollar_adv'] = round(dollar_adv, 0) if dollar_adv is not None else None

    if dollar_adv is None:
        out['reasons'].append('無法取得成交量 — 流動性未知,視為不可全倉')
        out['cap_tier'] = 'SPECULATIVE'
    elif dollar_adv < min_adv:
        out['reasons'].append(f'日均成交額 ${dollar_adv/1e6:.1f}M < ${min_adv/1e6:.0f}M 流動性下限 — 不可交易(滑價吃掉邊際)')
        out['block'] = True
    elif dollar_adv < strong_adv:
        out['reasons'].append(f'日均成交額 ${dollar_adv/1e6:.1f}M < ${strong_adv/1e6:.0f}M — 流動性偏薄,降為投機(縮小部位)')
        out['cap_tier'] = 'SPECULATIVE'

    # ── Quoted spread (best-effort; single-stock quotes are often 0 after hours) ──
    bid = info.get('bid'); ask = info.get('ask')
    spread_pct = None
    try:
        if bid and ask and ask > 0 and bid > 0 and ask >= bid:
            mid = (ask + bid) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else None
    except Exception:
        spread_pct = None
    out['spread_pct'] = round(spread_pct, 4) if spread_pct is not None else None
    if spread_pct is not None and spread_pct > max_spread:
        out['reasons'].append(f'買賣價差 {spread_pct*100:.1f}% > {max_spread*100:.1f}% — 進出成本過高,降為投機')
        if not out['block']:
            out['cap_tier'] = 'SPECULATIVE'

    # ── Earnings-date proximity (binary event the price signals can't see) ──
    days_to_earn = None
    try:
        from datetime import datetime as _dt
        now = _dt.now()
        ts = info.get('earningsTimestampStart') or info.get('earningsTimestamp')
        if ts:
            ed = _dt.fromtimestamp(float(ts))
            d = (ed - now).days
            # only treat as upcoming if it's in the future (info also carries last print)
            if d >= 0:
                days_to_earn = d
    except Exception:
        days_to_earn = None
    out['days_to_earnings'] = days_to_earn
    out['earnings_blackout'] = bool(days_to_earn is not None and days_to_earn <= blackout_days)
    if out['earnings_blackout']:
        out['reasons'].append(f'距財報僅 {days_to_earn} 天 — 進入財報空窗期,不開新全倉(等二元事件過後再評估);若已持有,留意跳空風險')
        if not out['block']:
            out['cap_tier'] = 'SPECULATIVE'

    out['tradeable'] = not out['block']
    return out


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
    """0-100: how stretched to the downside the name is — measured along ORTHOGONAL axes.

    The naive version summed 9 raw indicators (RSI14, RSI2, %B, Williams, Stoch, dist50,
    drawdown, resid, streak, IBS) with hand weights. The problem: most of those are the
    SAME signal. RSI14/RSI2/Williams/Stoch are one oscillator measured four ways; %B and
    dist50 are both 'distance below the mean'. Summing them gives the ILLUSION of nine
    independent confirmations ('9/9 agree!') when it is really ~4 facts counted nine times,
    so a single oversold oscillator can dominate the whole score.

    Fix: group the collinear indicators into a few near-orthogonal AXES, AVERAGE within each
    axis (so redundancy inside an axis can't double-count), then weight ACROSS the axes.
    Now 'agreement' means agreement across genuinely different evidence:
      • oscillator   — short-term overbought/oversold (RSI14, RSI2, Williams%R, Stoch)
      • mean_dist    — how far price sits below its mean band (Bollinger %B, dist 50dma)
      • drawdown     — peak-to-trough depth (inverted-U; longer horizon than the oscillator)
      • idiosyncratic— the STOCK-SPECIFIC drop after removing market beta (resid) — the one
                       axis that is cross-sectionally distinct and the cleanest reversion fuel
      • micro        — intraday/persistence (down-day streak, IBS close-on-low)
    """
    def _avg(vals):
        v = [x for x in vals if x is not None]
        return (sum(v) / len(v)) if v else None

    axes = {}
    # Oscillator family — four readings of one fact → average them, don't add them.
    axes['oscillator'] = _avg([
        _scale(p.get('rsi14'), 50, 20),       # 50→0, 20→1
        _scale(p.get('rsi2'), 30, 2),         # fast: 30→0, 2→1
        _scale(p.get('williams_r'), -60, -95),
        _scale(p.get('stoch_k'), 30, 5),      # 30→0, 5→1
    ])
    # Mean-distance family — %B and dist-50dma are the same 'below the mean' fact.
    axes['mean_dist'] = _avg([
        _scale(p.get('boll_pctb'), 0.2, -0.2),
        _scale(p.get('dist_ma50'), -0.02, -0.15),
    ])
    # Drawdown depth — inverted-U (sweet spot −8%..−18%; deep holes de-weighted).
    axes['drawdown'] = _drawdown_part(p.get('dist_52w_high'))
    # Idiosyncratic / residual drop — down on its OWN account = clean reversion fuel.
    axes['idiosyncratic'] = _scale(p.get('resid_drop_20d'), -0.03, -0.12)
    # Microstructure — persistence + close pinned to the low.
    axes['micro'] = _avg([
        _scale(p.get('down_days_streak'), 2, 6),
        _scale(p.get('ibs'), 0.5, 0.0),
    ])

    # Weights ACROSS orthogonal axes (sum≈1). Idiosyncratic axis is up-weighted because it
    # is the one with cross-sectionally distinct, academically robust reversion expectancy.
    weights = {'oscillator': 0.30, 'mean_dist': 0.22, 'drawdown': 0.12,
               'idiosyncratic': 0.24, 'micro': 0.12}
    acc = wsum = 0.0
    for k, w in weights.items():
        if axes.get(k) is not None:
            acc += w * axes[k]; wsum += w
    score = (acc / wsum * 100) if wsum > 0 else 0.0

    # Count how many DISTINCT axes are firing (≥0.5) — this is real confirmation breadth,
    # not the inflated raw-indicator count the old version implied.
    axes_firing = sum(1 for v in axes.values() if v is not None and v >= 0.5)

    return {'score': round(score, 0),
            'axes': {k: (round(v, 2) if v is not None else None) for k, v in axes.items()},
            'axes_firing': axes_firing,
            # back-compat: keep a flat 'parts' view so older readers don't break
            'parts': {k: (round(v, 2) if v is not None else None) for k, v in axes.items()}}


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


def _f(label, value, rule, status, kind='support', blocking=False):
    """Build one scorecard factor row. status ∈ pass|partial|fail|info."""
    return {'label': label, 'value': value, 'rule': rule,
            'status': status, 'kind': kind, 'blocking': blocking}


def factor_scorecard(panel, struct, surv, trap, oversold, capit, confirm,
                     sentiment, positioning, pt_data, trade=None) -> dict:
    """
    The auditable heart of the report. Lays out EVERY factor across the six dimensions
    (結構/基本/技術/人性/情緒/籌碼/估值) with its live value, the rule it is judged against,
    and a PASS / PARTIAL / FAIL verdict — and flags which of the GATING factors are
    currently BLOCKING a high-conviction entry. This is what turns "0 strong" from a
    dead end into an answer: it tells you exactly which factors are missing and what
    would have to change for the name to become buyable.

    kind:  gate     — must pass or there is no trade (structure, survivability)
           trigger  — defines whether the opportunity exists (oversold)
           timing   — reduces "too early" risk (confirmation)
           support  — informs size/conviction (capitulation, sentiment, positioning, value)
    """
    groups = []

    # ── 結構面 (GATE) ───────────────────────────────────────────────────────────
    d200 = panel.get('dist_ma200'); slope = panel.get('ma200_slope_1m')
    trend = struct['trend']
    above = (d200 or -1) > 0
    rising = (slope or 0) > 0
    groups.append(('結構面 — 是回檔還是接刀(硬性關卡)', [
        _f('價 vs 200日線', _pct(d200), '需 > 0(站在年線之上才是回檔)',
           'pass' if above else 'fail', 'gate', blocking=not above),
        _f('200日線方向', (_pct(slope) if slope is not None else '—'),
           '需上升(>0);走平偏弱、下彎為接刀',
           'pass' if rising else ('partial' if above else 'fail'), 'gate',
           blocking=(trend == 'DOWNTREND')),
        _f('趨勢判定', trend, 'UPTREND 最佳 / DOWNTREND 不抄',
           'pass' if trend == 'UPTREND' else ('partial' if trend in ('WEAK_UPTREND', 'EARLY_BREAK') else 'fail'),
           'gate', blocking=(trend == 'DOWNTREND')),
    ]))

    # ── 基本面 / 體質 (GATE) ─────────────────────────────────────────────────────
    f = surv.get('f_score'); knife = surv.get('knife_risk')
    de = None
    try:
        de = panel.get('_de')
    except Exception:
        pass
    groups.append(('基本面 / 體質 — 跌的是情緒不是基本面(硬性關卡)', [
        _f('Piotroski F 分數', (f if f is not None else '—'),
           '需 ≥ 4(≥7 強);≤3 為基本面惡化',
           'pass' if (f or 0) >= 7 else ('partial' if (f or 0) >= 4 else 'fail'),
           'gate', blocking=((f is not None) and f <= 3)),
        _f('自由現金流', ('正' if surv.get('fcf_positive') else '負/未知'),
           '需為正(不燒錢才撐得住)',
           'pass' if surv.get('fcf_positive') else 'fail', 'gate'),
        _f('體質總分', f"{surv.get('survivability'):.0f}", '需 ≥ 55(越高越能接)',
           'pass' if surv.get('survivability', 0) >= 55 else ('partial' if surv.get('survivability', 0) >= 40 else 'fail'),
           'gate', blocking=bool(knife)),
        _f('價值陷阱', ('是' if trap.get('value_trap') else '否'),
           '分析師大幅下修 + 無估值墊 = 陷阱',
           'fail' if trap.get('value_trap') else 'pass', 'gate', blocking=bool(trap.get('value_trap'))),
        _f('近期利空跳空', (_pct(panel.get('recent_gap_down')) if panel.get('recent_gap_down') is not None else '—'),
           '近3日跳空 < −7% = 利空跳水,反轉率低(降為投機,不給高把握)',
           'fail' if panel.get('news_gap') else 'pass', 'gate', blocking=False),
    ]))

    # ── 技術面 — 超賣 (TRIGGER) ──────────────────────────────────────────────────
    rsi14 = panel.get('rsi14'); rsi2 = panel.get('rsi2'); pctb = panel.get('boll_pctb')
    d50 = panel.get('dist_ma50'); ddh = panel.get('dist_52w_high'); wr = panel.get('williams_r')
    os_ok = oversold['score'] >= 35
    groups.append(('技術面 — 超賣程度(機會是否存在)', [
        _f('RSI(14)', (round(rsi14, 1) if rsi14 is not None else '—'), '< 30 超賣;30–40 偏弱',
           'pass' if (rsi14 or 99) < 30 else ('partial' if (rsi14 or 99) < 40 else 'fail'), 'trigger'),
        _f('RSI(2) 快線', (round(rsi2, 1) if rsi2 is not None else '—'), '< 10 極度washed-out',
           'pass' if (rsi2 or 99) < 10 else ('partial' if (rsi2 or 99) < 30 else 'fail'), 'trigger'),
        _f('布林 %B', (round(pctb, 2) if pctb is not None else '—'), '< 0 跌破下軌',
           'pass' if (pctb if pctb is not None else 1) < 0 else ('partial' if (pctb if pctb is not None else 1) < 0.2 else 'fail'), 'trigger'),
        _f('vs 50日線乖離', _pct(d50), '< −8% 深度乖離',
           'pass' if (d50 or 0) < -0.08 else ('partial' if (d50 or 0) < -0.02 else 'fail'), 'trigger'),
        _f('距52週高回撤', _pct(ddh), '< −15% 顯著回檔',
           'pass' if (ddh or 0) < -0.15 else ('partial' if (ddh or 0) < -0.08 else 'fail'), 'trigger'),
        _f('Williams %R', (round(wr, 0) if wr is not None else '—'), '< −90 觸底',
           'pass' if (wr if wr is not None else 0) < -90 else ('partial' if (wr if wr is not None else 0) < -80 else 'fail'), 'trigger'),
        _f('IBS(收盤位於當日區間)', (panel.get('ibs') if panel.get('ibs') is not None else '—'),
           '< 0.2 收在當日最低區(極弱收)',
           'pass' if (panel.get('ibs') if panel.get('ibs') is not None else 1) < 0.2
           else ('partial' if (panel.get('ibs') if panel.get('ibs') is not None else 1) < 0.4 else 'fail'), 'trigger'),
        _f('創7日新低(Double 7s)', ('✓ 是' if panel.get('new_7d_low') else '✗ 否'),
           '今日為近7日新低=經典逢低買進觸發點',
           'pass' if panel.get('new_7d_low') else 'info', 'trigger'),
        _f('個股殘差超賣(剔除大盤beta)', (_pct(panel.get('resid_drop_20d')) if panel.get('resid_drop_20d') is not None else '—'),
           '< −8% 是「自己跌」的乾淨反轉燃料;接近0僅被大盤拖累',
           'pass' if (panel.get('resid_drop_20d') if panel.get('resid_drop_20d') is not None else 0) < -0.08
           else ('partial' if (panel.get('resid_drop_20d') if panel.get('resid_drop_20d') is not None else 0) < -0.03 else 'info'),
           'trigger'),
        _f('超賣軸向一致(5獨立軸)', f"{oversold.get('axes_firing', 0)}/5 軸",
           '計算的是「振盪/均線乖離/回撤/個股殘差/微結構」5個近正交軸各自是否觸發 — '
           '而非同一個指標數8次造成的假確認;≥3軸才算真正多面向超賣',
           'pass' if oversold.get('axes_firing', 0) >= 3 else ('partial' if oversold.get('axes_firing', 0) >= 2 else 'fail'),
           'trigger'),
        _f('超賣總分', f"{oversold['score']:.0f}", '需 ≥ 35 才有左側進場優勢',
           'pass' if os_ok else 'fail', 'trigger', blocking=not os_ok),
    ]))

    # ── 人性 / 投降面 (SUPPORT) ──────────────────────────────────────────────────
    groups.append(('人性 / 投降面 — 是否賣壓衰竭(不是只買「下跌」)', [
        _f('連續下跌天數', f"{panel.get('down_days_streak')} 天", '≥ 3 天賣壓堆積',
           'pass' if (panel.get('down_days_streak') or 0) >= 3 else 'partial', 'support'),
        _f('量能投降(climax)', (f"{panel.get('vol_climax_x')}x" if panel.get('vol_climax_x') else '—'),
           '≥ 1.5x 出現恐慌放量',
           'pass' if (panel.get('vol_climax_x') or 0) >= 1.5 else 'partial', 'support'),
        _f('量能枯竭(dry-up)', (panel.get('vol_dryup') if panel.get('vol_dryup') is not None else '—'),
           '< 0.7 賣方力竭',
           'pass' if (panel.get('vol_dryup') if panel.get('vol_dryup') is not None else 1) < 0.7 else 'partial', 'support'),
        _f('投降總分', f"{capit['score']:.0f}", '越高代表賣壓越接近衰竭', 'info', 'support'),
    ]))

    # ── 確認面 (TIMING) ──────────────────────────────────────────────────────────
    present = set(confirm.get('present', []))
    def _cf(name):
        return 'pass' if name in present else 'fail'
    groups.append(('確認面 — 刀子是否變慢(降低「太早」風險)', [
        _f('RSI2 自低位翻揚', ('✓' if 'RSI2 自低位翻揚' in present else '✗'), '低位翻揚=動能轉向', _cf('RSI2 自低位翻揚'), 'timing'),
        _f('反轉K棒(長下影)', ('✓' if '出現帶長下影的反轉K棒' in present else '✗'), '買方接手的足跡', _cf('出現帶長下影的反轉K棒'), 'timing'),
        _f('收復10日線', ('✓' if '收復 10 日線' in present else '✗'), '短均收復=止跌', _cf('收復 10 日線'), 'timing'),
        _f('RSI 底背離', ('✓' if 'RSI 出現底背離' in present else '✗'), '價創低RSI不創低', _cf('RSI 出現底背離'), 'timing'),
        _f('確認總分', f"{confirm['score']:.0f}", '0=完全沒有止跌訊號,寧可等',
           'pass' if confirm['score'] >= 40 else ('partial' if confirm['score'] > 0 else 'fail'), 'timing'),
    ]))

    # ── 情緒面 (SUPPORT) ─────────────────────────────────────────────────────────
    rt = sentiment.get('rev_trend')
    groups.append(('情緒面 — 恐慌是燃料,但下修是警訊', [
        _f('分析師評等(1強買-5賣)', (sentiment.get('rec_mean') if sentiment.get('rec_mean') is not None else '—'),
           '越接近賣出=越被嫌棄(反向燃料)',
           'pass' if (sentiment.get('rec_mean') or 0) >= 3.0 else 'info', 'support'),
        _f('EPS 修正趨勢(近30天)', (f"↑{sentiment.get('rev_up_30d')} / ↓{sentiment.get('rev_down_30d')}"),
           '下修為主=基本面被坐實(警訊)',
           'fail' if rt == 'cutting' else ('pass' if rt == 'raising' else 'info'),
           'support', blocking=False),
        _f('分析師目標價上行', (_pct((sentiment.get('target_upside') or 0) / 100) if sentiment.get('target_upside') is not None else '—'),
           '> 0 街上認為有空間', 'pass' if (sentiment.get('target_upside') or -1) > 0 else 'partial', 'support'),
    ]))

    # ── 籌碼面 (SUPPORT) ─────────────────────────────────────────────────────────
    groups.append(('籌碼面 — 軋空燃料與持股結構', [
        _f('空單佔流通比', (f"{positioning.get('short_pct_float')}%" if positioning.get('short_pct_float') is not None else '—'),
           '偏高(>10%)+ 反轉 = 軋空燃料',
           'pass' if (positioning.get('short_pct_float') or 0) >= 10 else 'info', 'support'),
        _f('回補天數(short ratio)', (positioning.get('short_ratio') if positioning.get('short_ratio') is not None else '—'),
           '越高軋空越猛', 'pass' if (positioning.get('short_ratio') or 0) >= 5 else 'info', 'support'),
        _f('法人持股比', (f"{positioning.get('inst_pct')}%" if positioning.get('inst_pct') is not None else '—'),
           '高=回檔有人接(支撐)', 'pass' if (positioning.get('inst_pct') or 0) >= 60 else 'info', 'support'),
    ]))

    # ── 可交易性 (GATE) ──────────────────────────────────────────────────────────
    if trade is not None:
        adv = trade.get('dollar_adv'); spr = trade.get('spread_pct')
        dte = trade.get('days_to_earnings')
        adv_status = ('fail' if trade.get('block') else
                      ('partial' if (trade.get('cap_tier') == 'SPECULATIVE' and adv is not None) else 'pass'))
        groups.append(('可交易性 — 能否乾淨進出(硬性關卡)', [
            _f('日均成交額(ADV)', (f"${adv/1e6:.1f}M" if adv is not None else '—'),
               '≥ $20M 可全倉;< $5M 不可交易(滑價吃掉邊際)',
               adv_status, 'gate', blocking=bool(trade.get('block'))),
            _f('買賣價差', (f"{spr*100:.2f}%" if spr is not None else '—'),
               '≤ 1.2%;過寬代表進出成本高',
               'fail' if (spr is not None and spr > 0.012) else ('pass' if spr is not None else 'info'),
               'gate'),
            _f('距下次財報', (f"{dte} 天" if dte is not None else '—'),
               '> 5 天才開新全倉;空窗期內為二元事件,降為投機',
               'fail' if trade.get('earnings_blackout') else ('pass' if dte is not None else 'info'),
               'gate', blocking=False),
        ]))

    # ── 估值面 (MULTIPLIER) ──────────────────────────────────────────────────────
    up = (pt_data or {}).get('upside')
    groups.append(('估值面 — 便宜貨還是貴超賣', [
        _f('合理價上行空間', (_pct(up / 100) if up is not None else '—'),
           '≥ +8% 才有安全邊際;< −5% 是貴的超賣',
           'pass' if (up or -99) >= 8 else ('partial' if (up or -99) >= -5 else 'fail'), 'support'),
    ]))

    # ── Blocking analysis: which GATES are failing right now ─────────────────────
    blocking = []
    for _, rows in groups:
        for row in rows:
            if row['blocking'] and row['status'] == 'fail':
                blocking.append(row['label'])
    # If oversold trigger not met, that's the "no setup yet" reason
    if not os_ok and '超賣總分' not in blocking:
        pass

    # ── What needs to change to reach STRONG ─────────────────────────────────────
    needs = []
    if trend == 'DOWNTREND':
        needs.append('站回上升的 200 日線(目前在下彎年線之下 = 接刀區)')
    elif trend in ('WEAK_UPTREND', 'EARLY_BREAK'):
        needs.append('200 日線重新上彎、趨勢轉強(目前結構偏弱,僅能投機試單)')
    if knife or (surv.get('survivability', 0) < 55):
        needs.append('體質改善:F分數/現金流/槓桿過關(目前接刀/歸零風險偏高)')
    if trap.get('value_trap'):
        needs.append('分析師停止下修 EPS(目前數字仍被坐實向下 = 價值陷阱)')
    if not os_ok:
        needs.append('更深的超賣/恐慌:RSI、布林、乖離尚未進入抄底區(再等回檔)')
    if confirm['score'] == 0 and os_ok:
        needs.append('等待止跌確認訊號(RSI翻揚/反轉K棒/收復10日線)再進場,避免接在半山腰')
    if (up is not None) and up < -5:
        needs.append('估值修復:即使回到合理價仍無空間,目前是「貴的超賣」')
    if panel.get('news_gap'):
        needs.append('避開利空跳空:近期是 gap-down 觸發(下跌被基本面坐實),反轉率偏低 — 等陰跌型回檔或跳空被填補再評估')
    if trade is not None:
        if trade.get('block'):
            needs.append('流動性不足以乾淨進出 — 對真實資金而言此單不可交易(等成交量放大或換更深標的)')
        elif trade.get('earnings_blackout'):
            needs.append(f"距財報僅 {trade.get('days_to_earnings')} 天 — 等二元事件過後再開新全倉(現在進場是賭財報,不是賭均值回歸)")
        elif trade.get('cap_tier') == 'SPECULATIVE':
            needs.append('可交易性受限(流動性偏薄或價差偏寬)— 只能投機性縮小部位')

    return {'groups': groups, 'blocking': blocking, 'needs': needs}


def crowd_phase(panel: dict, regime: dict | None) -> dict:
    """Map price/vol/vix into the CROWD EMOTION cycle and say which zone we're in.
    This is the soul of a contrarian strategy: buy maximum fear (capitulation, after the
    turn), hold through hope→optimism, sell into greed (euphoria). It drives the verdict
    AND is shown to the user so every action is tied to a human-nature rationale."""
    vix = (regime or {}).get('vix')
    dd = panel.get('dist_52w_high')           # how far below the high (pain)
    d200 = panel.get('dist_ma200')            # extension vs the year line
    rsi14 = panel.get('rsi14')

    # EUPHORIA — everyone greedy: calm vol + far extended + overbought + near the highs.
    euphoria = ((vix is not None and vix < 14) and (d200 is not None and d200 > 0.15)
                and (rsi14 is not None and rsi14 > 70) and (dd is not None and dd > -0.03))
    if panel.get('capitulation_buy'):
        return {'phase': 'CAPITULATION', 'label': '投降後轉折 — 群眾極度恐懼、最後一批人吐貨、買方剛站出來',
                'zone': 'buy', 'color': '#1a7a4a',
                'note': '別人恐懼我貪婪:賣壓力竭+轉折確認,這是左側唯一該重手的時刻。'}
    if panel.get('capit_setup'):
        return {'phase': 'PANIC', 'label': '恐慌投降中 — 還在落刀,賣壓尚未力竭',
                'zone': 'wait', 'color': '#b52a2a',
                'note': '投降成形但「轉折」未現 — 不接落下的刀,等買方第一次站出來(收紅站回昨高)再進。'}
    if euphoria:
        return {'phase': 'EUPHORIA', 'label': '全民狂歡 — 群眾極度貪婪、價格遠離均線、人人看多',
                'zone': 'sell', 'color': '#d4900a',
                'note': '別人貪婪我恐懼:這是把貨交給貪婪者的區域 — 持倉者收割,空手者別追。'}
    if (dd is not None and dd <= -0.15) or (d200 is not None and d200 < 0):
        return {'phase': 'FEAR', 'label': '恐懼下跌 — 趨勢轉弱/深度回檔,群眾焦慮否認',
                'zone': 'wait', 'color': '#e67e22',
                'note': '恐懼尚未到投降極端 — 觀察,等真正的恐慌climax+轉折。'}
    if (d200 is not None and d200 > 0):
        return {'phase': 'OPTIMISM', 'label': '樂觀復甦 — 站上均線、群眾轉為樂觀',
                'zone': 'hold', 'color': '#1a6fa8',
                'note': '若持有投降買進的核心 → 抱住讓它跑;空手 → 此處非左側進場區。'}
    return {'phase': 'NEUTRAL', 'label': '中性 — 無明顯情緒極端', 'zone': 'wait', 'color': '#6b7280',
            'note': '沒有可交易的情緒極端 — 等待。'}


def bottom_fish_score(panel, struct, surv, trap, oversold, capit, confirm,
                      pt_data, regime, trade=None) -> dict:
    """
    Combine everything into a left-side conviction (0-100) and a verdict tier.
    Logic: the OPPORTUNITY (oversold) is gated by STRUCTURE and SURVIVABILITY, lifted
    by a valuation cushion and by CONFIRMATION, and scaled by the market REGIME.
    A final TRADEABILITY gate can block an untradeable name or cap it out of STRONG.
    """
    os_score = oversold['score']
    struct_mult = struct['structure_mult']
    surv_score = surv['survivability']
    # Left-side book uses the CONTRARIAN regime dial (scale into fear when the uptrend
    # is intact); fall back to the generic multiplier if not present.
    regime_mult = (regime or {}).get('left_multiplier',
                                     (regime or {}).get('multiplier', 1.0))

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

    # News-gap knife: the trigger fired on a fundamental gap-down (利空跳水), not a
    # grind. Reversion odds are much worse — penalise hard and bar it from STRONG.
    news_gap = bool(panel.get('news_gap'))
    gap_mult = 0.65 if news_gap else 1.0

    conviction = (os_score
                  * struct_mult
                  * quality_gate
                  * value_mult
                  * gap_mult)
    conviction = conviction + confirm_boost
    conviction = conviction * (0.7 + 0.3 * regime_mult)  # regime dampens but never fully kills
    conviction = max(0.0, min(100.0, conviction))

    # ══ Verdict — rebuilt around TRUE CAPITULATION + the crowd's emotion ════════════
    # The old engine bought RSI(2)<10 (every minor dip — "too casual, too early") and is
    # replaced. Backtest-validated rule: only act on a genuine CAPITULATION confluence
    # (deep drawdown × volume climax × extreme oversold) AND only AFTER the TURN (buyers
    # step in) — never into a still-falling knife. Market-wide fear upgrades it to the
    # table-pounding tier (the +14%/trade, PF 5.39 events). No capitulation → no buy.
    crowd = crowd_phase(panel, regime)
    capit_buy = bool(panel.get('capitulation_buy'))
    capit_setup = bool(panel.get('capit_setup'))
    regime_vix = (regime or {}).get('vix')
    market_fear = (regime_vix is not None and regime_vix >= CAPIT_FEAR_VIX)
    # Fundamental knife = a company that can go to ZERO / value trap. A mere price downtrend
    # is EXPECTED in capitulation, so it is NOT itself a knife — the TURN + quality are what
    # separate a 'capitulation bottom' from a 'falling knife to zero'. Quality is also what
    # gives you the PSYCHOLOGICAL ability to hold through the −30% intratrade pain.
    knife = surv['knife_risk'] or trap['value_trap']
    quality_ok = (surv_score >= 50 and not knife and not news_gap)
    # kept for downstream (cycle hold-mode + scorecard + score dict)
    resid_v = panel.get('resid_drop_20d')
    edge_context = bool(market_fear or capit_buy)
    hard_trigger = capit_buy

    if knife:
        tier, tier_label, tier_color = 'KNIFE', '接刀風險 — 體質可能歸零', '#b52a2a'
        kr = '、'.join(surv['reasons'][:2] + trap['flags'][:1]) or '基本面惡化 / 價值陷阱'
        rationale = (f'即使深跌投降也不接 — 體質可能歸零({kr})。'
                     '投降買進的前提是「跌的是情緒、不是這家公司」;這檔不符合。')
    elif capit_buy and quality_ok and market_fear:
        tier, tier_label, tier_color = 'STRONG', '真投降 × 全市場恐慌 — 重手區', '#1a7a4a'
        rationale = (f'深度投降(距高 {_pct(panel.get("dist_52w_high"))})+ 量能力竭 + 買方「轉折」確認,'
                     f'且全市場同步恐慌(VIX {regime_vix})、體質撐得住。'
                     '這是「別人恐懼我貪婪」最該重手的時刻 — 全倉分批、抱到滿足/狂歡,不在第一個反彈賣。')
    elif capit_buy and quality_ok:
        tier, tier_label, tier_color = 'SPECULATIVE', '個股真投降(全市場尚未恐慌)', '#d4900a'
        rationale = ('個股深度投降 + 轉折確認、體質好,但全市場還沒一起恐慌(VIX 不高)— '
                     '縮小部位試單、一樣抱到滿足;等全市場也投降時才是重手機會。')
    elif capit_setup and not bool(panel.get('the_turn')):
        tier, tier_label, tier_color = 'WATCH', '投降成形 — 等轉折(別接刀)', '#e67e22'
        rationale = ('已深跌 + 量能投降,但「轉折」尚未出現(買方還沒站出來)。'
                     '此刻恐懼仍在加速 — 不接落下的刀。等收紅站回昨高 / 反轉K / 收復10日線再進。')
    elif crowd['phase'] == 'EUPHORIA':
        tier, tier_label, tier_color = 'NONE', '全民狂歡 — 非進場區', '#d4900a'
        rationale = '群眾極度貪婪、價格遠離均線 — 這是「持倉者收割」區,不是左側進場區。空手別追高。'
    else:
        tier, tier_label, tier_color = 'NONE', '無投降訊號 — 觀望', '#6b7280'
        rationale = ('沒有「深度投降 × 量能力竭 × 轉折」的合流 — 不為了出手而出手(避免太隨便)。'
                     '左側只在真恐慌出手,耐心等。')

    # ── Tradeability override (executability beats analytics) ──
    tradeable = True
    trade_reasons = []
    if trade:
        trade_reasons = list(trade.get('reasons', []))
        if trade.get('block') and tier not in ('NONE',):
            # Can't execute cleanly → there is no trade, regardless of how good the setup is.
            tradeable = False
            tier, tier_label, tier_color = 'UNTRADEABLE', '訊號成立但不可交易', '#6b21a8'
            rationale = '抄底訊號可能成立,但流動性不足以乾淨進出 — 對真實資金而言滑價會吃掉邊際,不交易。' \
                        + ('（' + '；'.join(trade_reasons[:2]) + '）' if trade_reasons else '')
        elif trade.get('cap_tier') == 'SPECULATIVE' and tier == 'STRONG':
            # Setup is STRONG but not full-size-able (thin liquidity / wide spread / pre-earnings).
            tier, tier_label, tier_color = 'SPECULATIVE', '投機性抄底(可交易性受限)', '#d4900a'
            rationale = '結構/體質/超賣本可列高把握,但可交易性受限(流動性、價差或財報空窗)— 降為投機、縮小部位。' \
                        + ('（' + '；'.join(trade_reasons[:2]) + '）' if trade_reasons else '')

    return {
        'conviction': round(conviction, 0),
        'tier': tier, 'tier_label': tier_label, 'tier_color': tier_color,
        'rationale': rationale,
        'hard_trigger': bool(hard_trigger),
        'edge_context': bool(edge_context),
        'capitulation': bool(capit_buy),
        'crowd': crowd,
        'tradeable': tradeable,
        'trade_reasons': trade_reasons,
        'value_mult': round(value_mult, 2),
        'quality_gate': round(quality_gate, 2),
        'structure_mult': struct_mult,
        'regime_mult': regime_mult,
        'upside': upside,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED-ENTRY PLAYBOOK  (the 反人性 core — pre-committed tranches + discipline)
# ═══════════════════════════════════════════════════════════════════════════════

def _round_level(price: float):
    """Nearest psychologically-round price. Limit orders and support/resistance cluster at
    round numbers, so a rung snapped to one is more likely to actually fill on a wick."""
    if not price or price <= 0:
        return None
    # Coarse, psychological grid (NOT the price itself) so a round level is a real magnet
    # that can win over a raw ATR step only when it is genuinely close.
    if price >= 500:   step = 50.0
    elif price >= 100: step = 10.0
    elif price >= 20:  step = 5.0
    elif price >= 5:   step = 1.0
    else:              step = 0.5
    return round(round(price / step) * step, 2)


def _snap_to_support(mech_price: float, levels: list, atr: float, tol: float = 0.5):
    """Snap a mechanical ATR-step rung to the NEAREST real level (Bollinger lower band, a
    recent swing low, the 52w low, or a round number) when one sits within tol×ATR. Anchoring
    to structure beats hanging a limit order in mid-air where nobody defends the price."""
    best, best_d = mech_price, tol * atr
    for L in levels:
        if L is None or L <= 0:
            continue
        d = abs(L - mech_price)
        if d <= best_d:
            best, best_d = L, d
    return best


def staged_entry_plan(panel, score, struct, surv, pt_data, regime, config=None) -> dict:
    cfg = {**ACCOUNT_DEFAULTS, **(config or {})}
    price = panel.get('price')
    atr_v = panel.get('atr')
    tier = score['tier']

    plan = {'tier': tier, 'price': price, 'atr': atr_v}

    if tier in ('NONE', 'KNIFE', 'UNTRADEABLE', 'WATCH') or not price or not atr_v:
        note = {'NONE': '未到投降區,放觀察清單 — 左側只在真恐慌出手。',
                'WATCH': '投降已成形但「轉折」未現 — 不接落下的刀,等買方站出來再進。',
                'KNIFE': '接刀風險過高 — 體質可能歸零,投降也不接。',
                'UNTRADEABLE': '可交易性不足(流動性/價差)— 對真實資金無法乾淨進出,不進場。'}.get(tier,
                '接刀風險過高 — 結構或基本面修復前不進場。')
        plan.update(action='NO TRADE', tranches=[], note=note)
        plan['discipline'] = _discipline_rules(None, None, tier, None)
        return plan

    # Tier sizing factor: STRONG fishes a full book, SPECULATIVE a half
    tier_factor = 1.0 if tier == 'STRONG' else 0.5
    conv_scale = max(0.4, min(1.1, score['conviction'] / 75.0))
    # Contrarian sizing dial: in a fear spike within an uptrend this is >1 (scale into
    # fear); capped so a single name still respects max_position_pct downstream.
    regime_mult = (regime or {}).get('left_multiplier',
                                     (regime or {}).get('multiplier', 1.0))
    risk_mult = tier_factor * conv_scale * regime_mult

    # ── Scale-in ladder: 3 pre-committed prices, ANCHORED to real structure ──
    # T1 near current (where the setup triggers); T2/T3 step down on ATR but each is SNAPPED
    # to the nearest real support (Bollinger lower band, recent swing low, 52w low, or a round
    # number) when one is within 0.5×ATR — so the resting limit orders sit where price is
    # actually defended, not hung in mid-air on an arbitrary ATR multiple.
    boll_lower = panel.get('boll_lower')
    low_52 = panel.get('low_52w')
    swing_low = panel.get('swing_low')
    supports = [boll_lower, swing_low, low_52]

    t1 = round(price, 2)
    t2_mech = price - 1.0 * atr_v
    t3_mech = price - 2.2 * atr_v
    t2 = round(_snap_to_support(t2_mech, supports + [_round_level(t2_mech)], atr_v), 2)
    t3 = round(_snap_to_support(t3_mech, supports + [_round_level(t3_mech)], atr_v), 2)

    tranche_prices = sorted({t1, t2, t3}, reverse=True)
    while len(tranche_prices) < 3:                      # keep 3 distinct rungs
        tranche_prices.append(round(tranche_prices[-1] - 0.8 * atr_v, 2))
    tranche_prices = tranche_prices[:3]

    # ── Weighted average entry (pyramid down for STRONG; balanced for a probe) ──
    weights = TRANCHE_WEIGHTS_STRONG if tier == 'STRONG' else TRANCHE_WEIGHTS_PROBE
    avg_entry = round(sum(p * w for p, w in zip(tranche_prices, weights)), 2)

    # ── Mean-reversion TARGET1 = the SHORT mean we reverted from (20dma / Bollinger
    #    mid), NOT the far 50dma. A deeply oversold name's 50dma is often +8~12% away,
    #    so it rarely gets hit and the bounce is given back waiting. The reversion the
    #    edge actually delivers is back to the 20dma — that is what the backtest measured.
    #    Floor it at avg+1.2ATR so target1 is always a meaningful move; cap it at the
    #    50dma so we never aim PAST the next mean. (Computed BEFORE the stop because the
    #    stop is now sized to guarantee a payoff vs this target.) ──
    mid20 = panel.get('boll_mid')          # 20-day SMA = Bollinger middle band
    ma50 = panel.get('ma50'); ma200 = panel.get('ma200')
    t1_cands = [x for x in (mid20, round(avg_entry + 1.2 * atr_v, 2)) if x and x > avg_entry]
    t_profit1 = max(t1_cands) if t1_cands else round(avg_entry * 1.06, 2)
    if ma50 and ma50 > avg_entry:
        t_profit1 = min(t_profit1, round(ma50, 2))     # don't aim past the next mean up
    t_profit1 = round(t_profit1, 2)
    reward1 = t_profit1 - avg_entry

    # ── Stop = thesis invalidation, BUT payoff is a HARD constraint (R:R ≥ 1 to target1) ──
    # Old behaviour put the stop a fixed 1.5ATR below the deepest tranche → with a near
    # target1 that produced R:R≈0.4 (risk $1 to make $0.40), violating the payoff≥1 mandate.
    # Now: start from the structural stop, and if it does not pay ≥1:1, TIGHTEN it to exactly
    # 1:1 — provided that still leaves ≥1 ATR of room (else the stop sits in the noise and
    # gets swept). If even a 1:1 stop would be inside the noise, we cannot honour payoff
    # cleanly → flag it and cut the size to a probe (payoff_ok=False).
    # The stop must ALWAYS sit below the DEEPEST buy rung — otherwise you'd be stopped out
    # before your last tranche ever fills. Within that hard ceiling we pick the stop that
    # honours R:R≥1: a roomy structural stop if it already pays ≥1:1, else tighten toward the
    # exact 1:1 point — but never above the rungs and never inside the noise.
    deepest = min(tranche_prices)
    max_valid_stop = round(deepest - 0.25 * atr_v, 2)   # tightest stop still below all rungs
    struct_stop = round(deepest - 1.5 * atr_v, 2)       # roomy structural stop
    if low_52 and struct_stop < low_52 < deepest:
        struct_stop = round(low_52 - 0.5 * atr_v, 2)    # anchor just below a real structural low
    payoff_ok = True
    rr_struct = reward1 / (avg_entry - struct_stop) if (avg_entry - struct_stop) > 0 else 0.0
    if rr_struct >= 1.0:
        stop = struct_stop                              # structural stop already pays ≥1:1 (most room)
    else:
        stop_for_rr = round(avg_entry - reward1, 2)     # the stop that makes R:R exactly 1:1
        if stop_for_rr <= max_valid_stop and (avg_entry - stop_for_rr) >= 0.5 * atr_v:
            stop = stop_for_rr                          # 1:1 reachable with a valid, non-noise stop
        else:
            stop = max_valid_stop                       # can't reach 1:1 below the rungs → tightest valid
            payoff_ok = (reward1 / (avg_entry - stop)) >= 0.999 if (avg_entry - stop) > 0 else False
    risk_per_share = max(avg_entry - stop, 0.01)
    if not payoff_ok:
        risk_mult *= 0.5                                # payoff can't clear 1:1 → half-size probe

    # ── Risk-budgeted total size, split across the 3 rungs BY WEIGHT (pyramid) ──
    risk_budget = cfg['account_size'] * cfg['risk_per_trade'] * risk_mult
    total_shares = int(risk_budget / risk_per_share)
    max_shares_cap = int(cfg['account_size'] * cfg['max_position_pct'] / avg_entry) if avg_entry else 0
    capped = total_shares > max_shares_cap
    total_shares = max(0, min(total_shares, max_shares_cap))

    tranches = []
    # Entry timing: take the FIRST tranche market-on-close on the signal day, not at
    # next-day open. 2025 reversal research (Della Corte & Kosowski; Baltussen-Da-Soebhag
    # End-of-Day Reversal) shows the rebound concentrates close→open — buying the next
    # open hands that overnight pop to someone else. T2/T3 are resting limit orders lower.
    labels = ['第一批(收盤MOC進場)', '第二批(更深一階·掛限價)', '第三批(恐慌/支撐·掛限價)']
    allocated = 0
    for i, px in enumerate(tranche_prices):
        sh = (total_shares - allocated) if i == len(tranche_prices) - 1 else int(total_shares * weights[i])
        allocated += sh
        tranches.append({
            'label': labels[i], 'price': px, 'shares': sh,
            'value': round(sh * px, 0),
            'pct_of_plan': round(sh / total_shares * 100, 0) if total_shares else 0,
        })

    pos_value = round(sum(t['value'] for t in tranches), 0)
    pos_pct = round(pos_value / cfg['account_size'] * 100, 1)
    dollar_risk = round(total_shares * risk_per_share, 0)

    # ── TARGET2 = the NEXT mean up / intrinsic value (this is the OLD target1: 50dma,
    #    then composite PT, then 200dma). Pick the nearest level that sits above target1. ──
    pt = (pt_data or {}).get('price_target')
    t2_cands = [round(x, 2) for x in (ma50, pt, ma200) if x and x > t_profit1]
    t_profit2 = min(t2_cands) if t2_cands else round(t_profit1 * 1.10, 2)
    rr = round(reward1 / risk_per_share, 2) if risk_per_share > 0 else None

    # ── HOLD MODE: swing vs cycle ────────────────────────────────────────────────
    # Backtests settled a real debate. The short-term 'swing' (sell 2/3 at the 20dma in
    # ~12 days) THROWS AWAY the recovery — exactly why buying March-2020 and bailing at the
    # bounce loses. For a GENUINE CAPITULATION buy (STRONG in real fear/idiosyncratic
    # washout) the money is in RIDING the whole fear→greed expansion. AND a fear-to-greed
    # swinger who exits at euphoria sidesteps the next crash (the only contrarian variant that
    # cut drawdown −33%→−28% vs DCA). So such positions switch to 'cycle': trim only a third
    # at the first bounce, then RIDE — exit the core on euphoria-with-rollover or a 200dma
    # break (let winners run; do NOT sell the first sign of greed, which just leaves money on
    # the table — the top is unidentifiable, so use a trailing/structure exit).
    # ALL true-capitulation buys ride the fear→greed recovery (cycle); only weak/legacy
    # entries scalp (swing). Cycle = the validated "hold to satisfaction" behaviour.
    cycle = bool(score.get('capitulation')) or (tier == 'STRONG' and bool(score.get('edge_context')))
    ma200 = panel.get('ma200'); high_52 = panel.get('high_52w')
    if cycle:
        scaleout1_pct = 33                                  # bank only a third at the relief bounce…
        time_stop_days = 252                                # …then ride the whole recovery (up to a year)
        satis = f'${round(high_52,2)}' if high_52 else '前波高點'
        signal_exit = ('核心倉「抱住整段復甦」,絕不在第一個 relief 反彈賣(那是賣給懷疑者)。出場條件(擇一):'
                       f'① 滿足點:收復痛苦起點 {satis}(前52週高)→ 先了結一部分;'
                       f'② 全民狂歡轉彎:VIX<14 且 價遠高於年線「之後、跌破10週均」(賣在貪婪、但等它真轉);'
                       f'③ 吊燈移動停損:自最高收盤回落 3×ATR;④ 跌破年線 ${round(ma200,2) if ma200 else "—"}(趨勢結束)。')
        hold_note = ('【週期模式·真投降買點】目標吃完「恐懼→貪婪」整段。'
                     '反彈只減 1/3(舒緩心理),核心抱到滿足點/狂歡轉彎 — 不被早期反彈騙下車。')
    else:
        scaleout1_pct = round(SCALEOUT_T1 * 100)            # swing: bank 2/3 at the bounce
        time_stop_days = TIME_STOP_STRONG if tier == 'STRONG' else TIME_STOP_PROBE
        signal_exit = (f'若未達目標1但 RSI(2) 回升 > {SIGNAL_EXIT_RSI2:.0f} 或 收盤站回10日線 '
                       f'→ 先了結目標1那一份({scaleout1_pct}%)鎖住反轉,不必死等價位。')
        hold_note = None

    plan.update(
        action='SCALE-IN BUY(分批進場)',
        tranches=tranches,
        avg_entry=avg_entry,
        stop=stop, risk_per_share=round(risk_per_share, 2),
        total_shares=total_shares, position_value=pos_value, position_pct=pos_pct,
        dollar_risk=dollar_risk, size_capped=capped,
        target1=t_profit1, target2=t_profit2, rr=rr,
        payoff_ok=payoff_ok, breakeven_after_t1=True,
        hold_mode=('cycle' if cycle else 'swing'), hold_note=hold_note,
        scaleout1_pct=scaleout1_pct, signal_exit=signal_exit,
        target_note=((hold_note + ' ') if hold_note else '') +
                    (f'目標1=短均回歸(20日線/中軌)${t_profit1};目標2=50日線/合理價 ${t_profit2}。'
                     f'到目標1減 {scaleout1_pct}% 並『將停損抬至成本價 ${avg_entry}』,'
                     + ('剩餘核心抱住整段復甦,賣在狂歡轉彎或破年線。' if cycle else '剩餘到目標2出清。')),
        time_stop_days=time_stop_days,
        risk_mult=round(risk_mult, 2),
        entry_timing='第一批今日尾盤(MOC)進場吃隔夜反轉;第二、三批在更低價掛限價單等待。',
    )
    plan['discipline'] = _discipline_rules(stop, time_stop_days, tier, regime,
                                           avg_entry=avg_entry, payoff_ok=payoff_ok,
                                           scaleout1_pct=scaleout1_pct, signal_exit=signal_exit,
                                           cycle=cycle)
    return plan


def _discipline_rules(stop, time_stop_days, tier, regime, avg_entry=None, payoff_ok=True,
                      scaleout1_pct=50, signal_exit=None, cycle=False) -> list:
    if tier in ('NONE',):
        return ['尚未進場 — 等超賣訊號出現再啟動劇本。']
    if tier in ('KNIFE', 'UNTRADEABLE'):
        return ['不進場。等結構修復(站回 200 日線)或基本面止血再重新評估。',
                '「便宜」不是進場理由 — 接刀的代價是歸零風險。']
    # ── PRECEDENCE (解決「加碼進恐慌」與「停損清倉」的表面矛盾)───────────────────
    # 兩條規則從不衝突,因為它們治理「不同的價格區間」:
    #   • 停損價之上 = 階梯區:這裡才有「在最害怕時照表買 T2/T3」的勇氣紀律。
    #   • 停損價之下 = 出場區:論點已被證偽,紀律從「買恐慌」翻轉為「無條件清倉」。
    # 把這個層級寫在最前面,當價格同時穿過 T3 與停損時,規則明確:停損優先,清倉。
    rules = [
        f'【最高原則】硬停損 ${stop} 是唯一的總開關。價在停損之上 → 只在預設階梯買進(見下);'
        f'價跌破 ${stop} → 立刻無條件清倉,「加碼進恐慌」的勇氣紀律到此失效,不再攤平、不再凹單。',
        f'【階梯區·停損之上】最害怕的時候(VIX 高、新聞最壞)有勇氣執行『預設的』T2/T3 分批 —— '
        f'「加碼進恐慌」= 把事先掛好的低接單吃滿,絕不是在停損之下臨時加碼。不追價、不報復性攤平。',
        f'部位規模的「加碼」只發生在『進場前』(由市場體制 left_multiplier 決定大小);'
        f'一旦建倉,唯一向下的動作就是停損,沒有第四批。',
        (f'【達標即鎖利】價格觸及目標1 → 先減 {scaleout1_pct}%,並『將停損抬至成本價 ${avg_entry}』。'
         f'已實現的單不再由綠翻紅 — 把一筆贏的單拗成輸的單是左側最常見的死法。'
         if avg_entry is not None else
         f'【達標即鎖利】價格觸及目標1 → 先減 {scaleout1_pct}%,並將停損抬至成本價,已實現的單不再由綠翻紅。'),
        (signal_exit or '若 RSI(2) 回升 > 65 或收盤站回10日線 → 先了結目標1那一份鎖住反轉,不必死等價位。'),
        ('【週期模式·核心抱住整段復甦】減碼後剩餘核心『不』在小反彈賣 —— 這是真投降買點,'
         '目標是吃完恐懼→貪婪整段。只在(全民狂歡且動能轉彎 / 跌破年線 / 自高點回落>15%)才出核心。'
         '回測:賣在狂歡能把最大回撤 −33%→−28%,但別「猜頭」太早賣(頂部認不出來,會踏空)。'
         if cycle else
         '剩餘 runner 到目標2(50日線/合理價)出清 — 賣在貪婪裡,不貪圖完整 V 型回升。'),
        f'時間停損:{time_stop_days} 個交易日內沒有反轉跡象 → 出場,'
        + ('別把死錢凹著佔資金。' if cycle else '別把短線反彈凹成長線套牢。'),
        f'嚴守部位上限 — 即使單一標的歸零,也傷不了總資本。這就是敢於進場的底氣。',
    ]
    if not payoff_ok:
        rules.insert(1, '⚠【賠率警告】此設定無法在「可存活的停損」下達到 1:1 賠率 — 目標太近、結構停損太寬。'
                        '已自動砍半為試單;若不願接受 <1 的賠率,寧可不做、等更深的回檔把目標space拉開。')
    if regime and regime.get('fast_crash'):
        rules.append('目前偵測到「快速崩跌」— 雖然大盤仍在年線上,但屬瀑布式下跌,左側加碼已自動關閉'
                     '(改為防禦)。此時寧可錯過,不可接刀。')
    elif regime and regime.get('multiplier', 1.0) < 0.7:
        rules.append(f'目前市場體制偏空({regime.get("regime")},×{regime.get("multiplier"):.0%} sizing)'
                     f'— 部位已自動縮減,寧可少賺不可重傷。')
    return rules


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER — analyze one name end to end
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_bottom_fish(data: dict, quality: dict = None, pt_data: dict = None,
                        regime: dict = None, config: dict = None,
                        benchmark: pd.Series = None) -> dict:
    info = data.get('info', {})
    # Resolve benchmark for residual/idiosyncratic oversold (explicit > data > cached SPY).
    bench_ret = benchmark
    if bench_ret is None:
        bench_ret = data.get('benchmark_returns')
    if bench_ret is None:
        bench_ret = _benchmark_returns()
    panel = technical_panel(data.get('hist_1y'), data.get('hist_5y'), bench_ret)
    if not panel:
        return {'ticker': data.get('symbol', '?'), 'status': 'ERROR',
                'error': 'no price history'}

    struct = structure_gate(panel)
    surv = survivability(info, quality or {}, data)
    trap = value_trap_check(info, data, pt_data)
    oversold = oversold_score(panel)
    capit = capitulation_score(panel)
    confirm = confirmation_score(panel)
    sentiment = sentiment_panel(data, info, pt_data)
    positioning = positioning_panel(info, data)
    trade = tradeability(info, panel, data, config)
    score = bottom_fish_score(panel, struct, surv, trap, oversold, capit, confirm,
                              pt_data, regime, trade)
    scorecard = factor_scorecard(panel, struct, surv, trap, oversold, capit, confirm,
                                 sentiment, positioning, pt_data, trade)
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
        'sentiment': sentiment,
        'positioning': positioning,
        'tradeability': trade,
        'scorecard': scorecard,
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


_STATUS_STYLE = {
    'pass':    ('✓', '#1a7a4a', '#e8f5e9'),
    'partial': ('~', '#b8860b', '#fff8e1'),
    'fail':    ('✗', '#b52a2a', '#fdecea'),
    'info':    ('·', '#6b7280', '#f1f5f9'),
}


def _render_scorecard(sc: dict) -> str:
    """Full six-dimension factor checklist: value · rule · PASS/PARTIAL/FAIL per factor."""
    if not sc or not sc.get('groups'):
        return ''
    blocks = ''
    for title, rows in sc['groups']:
        is_gate = '硬性關卡' in title
        head_bg = '#5e1818' if is_gate else '#34506b'
        rrows = ''
        for r in rows:
            mark, col, bg = _STATUS_STYLE.get(r['status'], _STATUS_STYLE['info'])
            block_tag = ('<span style="color:#b52a2a;font-weight:700;font-size:7pt"> ⛔阻擋</span>'
                         if r.get('blocking') and r['status'] == 'fail' else '')
            rrows += (f'<tr style="background:{bg}">'
                      f'<td style="padding:4px 8px;font-size:8pt;color:#374151">{r["label"]}{block_tag}</td>'
                      f'<td style="padding:4px 8px;font-size:8pt;font-weight:700;color:{col};text-align:right;white-space:nowrap">{r["value"]}</td>'
                      f'<td style="padding:4px 8px;font-size:7.5pt;color:#6b7280">{r["rule"]}</td>'
                      f'<td style="padding:4px 8px;text-align:center"><span style="display:inline-block;width:16px;height:16px;line-height:16px;border-radius:50%;background:{col};color:white;font-size:8pt;font-weight:700">{mark}</span></td>'
                      f'</tr>')
        blocks += (f'<div style="margin-bottom:10px;border:1px solid #e5e7eb;border-radius:6px;overflow:hidden">'
                   f'<div style="background:{head_bg};color:white;padding:5px 10px;font-size:8pt;font-weight:700">{title}</div>'
                   f'<table style="width:100%;border-collapse:collapse">{rrows}</table></div>')
    return blocks


def _render_gate_callout(sc: dict, tier: str) -> str:
    """The 'why not / what would have to change' box — answers 哪些因子沒做到導致不能抄底."""
    if not sc:
        return ''
    blocking = sc.get('blocking', [])
    needs = sc.get('needs', [])
    if tier == 'STRONG' and not blocking:
        return ('<div style="background:#f0f7f2;border-left:4px solid #1a7a4a;border-radius:5px;padding:10px 13px;margin-bottom:12px">'
                '<strong style="color:#1a7a4a;font-size:8.8pt">✓ 四道關卡全過 — 可依下方分批劇本進場</strong>'
                '<div style="font-size:8pt;color:#374151;margin-top:3px">結構、體質、超賣、確認、估值安全邊際同時滿足,這是左側勝率最高的交集。</div></div>')
    block_html = ''
    if blocking:
        block_html = ('<div style="margin-bottom:7px"><span style="font-size:8pt;font-weight:700;color:#b52a2a">⛔ 目前阻擋進場的關卡:</span> '
                      + '、'.join(f'<span style="background:#fdecea;color:#b52a2a;border-radius:3px;padding:1px 7px;font-size:7.8pt;margin:0 2px">{b}</span>' for b in blocking)
                      + '</div>')
    needs_html = ''
    if needs:
        needs_html = ('<div style="font-size:8pt;color:#374151;font-weight:700;margin-bottom:3px">要變成「可抄底」,需要看到:</div>'
                      '<ul style="margin:0 0 0 16px">'
                      + ''.join(f'<li style="font-size:8pt;color:#374151;line-height:1.5;margin-bottom:2px">{n}</li>' for n in needs)
                      + '</ul>')
    return (f'<div style="background:#fffaf5;border-left:4px solid #d4900a;border-radius:5px;padding:10px 13px;margin-bottom:12px">'
            f'<div style="font-size:8.8pt;font-weight:700;color:#92400e;margin-bottom:6px">為什麼還不能(或只能投機)抄底</div>'
            f'{block_html}{needs_html}</div>')


def _render_portfolio_section(pf: dict) -> str:
    """Portfolio-layer panel: heat/gross vs budget + correlated clusters + sector caps.
    Answers the question the per-name cards can't: are these candidates really one bet?"""
    if not pf or not pf.get('applied'):
        return ''
    acct = pf.get('account_size', 1) or 1
    def bar(before, after, budget, label, unit_pct=True):
        b_pct = before / acct * 100
        a_pct = after / acct * 100
        bud_pct = budget / acct * 100
        over = before > budget
        col = '#b52a2a' if over else '#1a7a4a'
        return (f'<div style="margin-bottom:8px">'
                f'<div style="font-size:8pt;color:#374151"><strong>{label}</strong> '
                f'單股加總 {b_pct:.1f}% → 組合後 <strong style="color:{col}">{a_pct:.1f}%</strong> '
                f'<span style="color:#9ca3af">(預算 {bud_pct:.1f}%)</span></div>'
                f'<div style="background:#eee;border-radius:5px;height:10px;overflow:hidden;position:relative">'
                f'<div style="width:{min(a_pct/max(bud_pct,0.1)*100,100):.0f}%;height:10px;background:{col}"></div>'
                f'</div></div>')

    heat = bar(pf.get('heat_before', 0), pf.get('heat_after', 0), pf.get('heat_budget', 1), '組合總風險 Heat')
    gross = bar(pf.get('gross_before', 0), pf.get('gross_after', 0), pf.get('gross_budget', 1), '總曝險 Gross')

    clusters = ''
    multi = [c for c in pf.get('clusters', []) if len(c.get('tickers', [])) >= 2]
    if multi:
        rows = ''
        for c in multi:
            flag = '<span style="color:#b52a2a;font-weight:700">⚠超預算</span>' if c['over_budget'] else '<span style="color:#1a7a4a">✓</span>'
            cc = f"ρ≈{c['avg_corr']}" if c.get('avg_corr') is not None else '—'
            rows += (f'<tr><td style="padding:3px 8px;font-size:8pt">{", ".join(c["tickers"])}</td>'
                     f'<td style="padding:3px 8px;font-size:8pt;text-align:center">{cc}</td>'
                     f'<td style="padding:3px 8px;font-size:8pt;text-align:right">${c["risk_before"]:,.0f}→${c["risk_after"]:,.0f}</td>'
                     f'<td style="padding:3px 8px;text-align:center">{flag}</td></tr>')
        clusters = ('<div style="margin-top:10px"><div style="font-size:8pt;font-weight:700;color:#6b21a8;margin-bottom:4px">'
                    '相關性群組(同群=同一個賭注,共用風險預算)</div>'
                    '<table style="width:100%;border-collapse:collapse;background:#faf8ff;border-radius:5px">'
                    '<tr style="color:#6b7280;font-size:7pt"><th style="text-align:left;padding:3px 8px">標的</th>'
                    '<th style="padding:3px 8px">平均相關</th><th style="text-align:right;padding:3px 8px">風險(縮減前→後)</th>'
                    '<th style="padding:3px 8px">狀態</th></tr>' + rows + '</table></div>')
    elif pf.get('n_eligible', 0) >= 2 and not pf.get('corr_available'):
        clusters = ('<div style="margin-top:8px;font-size:7.5pt;color:#b8860b">'
                    '相關性資料不可用 — 已退化為僅套用產業/總風險上限(每檔自成一群)。</div>')

    return (f'<div style="background:#fff;border-radius:6px;padding:14px 18px;margin-bottom:18px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,.08);border-left:5px solid #6b21a8">'
            f'<div style="font-size:10pt;font-weight:700;color:#6b21a8;margin-bottom:4px">組合層風險控管 — 這些候選「合起來」是不是同一個賭注?</div>'
            f'<div style="font-size:7.8pt;color:#6b7280;margin-bottom:10px">'
            f'單股各自風控 1% 並不代表組合安全:左側洗盤時高把握名單常是同一族群,一起漲跌。'
            f'本層把相關/同產業的部位納入共用風險預算,超標就等比例縮小 — 體制係數 ×{pf.get("regime_mult",1):.2f}。</div>'
            f'{heat}{gross}{clusters}</div>')


def render_report(results: list, output_path: str = None, portfolio: dict = None) -> str:
    output_path = output_path or os.path.join(REPORTS_DIR, 'bottom_fishing.html')
    ok = [r for r in results if r.get('status') == 'OK']
    order = {'STRONG': 0, 'SPECULATIVE': 1, 'WATCH': 2, 'UNTRADEABLE': 3, 'KNIFE': 4, 'NONE': 5}
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
        portfolio=_render_portfolio_section(portfolio),
        rows=rows, cards=cards)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[ok] Bottom-fishing report -> {output_path}")
    return output_path


_ZONE_TXT = {'buy': '進場區', 'hold': '持有/抱住', 'sell': '收割區', 'wait': '等待', 'watch': '觀察'}


def _render_crowd_banner(crowd: dict) -> str:
    """Show WHERE in the fear↔greed cycle this name is — every action tied to crowd emotion."""
    if not crowd:
        return ''
    col = crowd.get('color', '#6b7280')
    zone = _ZONE_TXT.get(crowd.get('zone'), crowd.get('zone', ''))
    return (f'<div style="display:flex;align-items:center;gap:10px;background:{col}12;'
            f'border-left:4px solid {col};border-radius:5px;padding:8px 12px;margin-bottom:10px">'
            f'<span style="background:{col};color:#fff;font-size:7.5pt;font-weight:700;'
            f'padding:2px 9px;border-radius:10px;white-space:nowrap">群眾情緒 · {crowd.get("phase","")} · {zone}</span>'
            f'<span style="font-size:8pt;color:#374151">{crowd.get("label","")} — {crowd.get("note","")}</span></div>')


def _render_card(r):
    s = r['score']; p = r['panel']; pl = r['plan']; st = r['structure']
    surv = r['survivability']
    gate_callout = _render_gate_callout(r.get('scorecard'), s['tier'])
    scorecard_html = _render_scorecard(r.get('scorecard'))
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

    # Portfolio-layer trim banner (shown when the book-level caps reduced this name)
    pf = r.get('portfolio') or {}
    pf_note = ''
    if pf.get('scale') is not None and pf['scale'] < 0.999:
        pf_note = (f'<div style="background:#faf8ff;border-left:4px solid #6b21a8;border-radius:5px;'
                   f'padding:8px 12px;margin-bottom:10px;font-size:8pt;color:#374151">'
                   f'<strong style="color:#6b21a8">組合層已縮減至 {pf["scale"]*100:.0f}%</strong> — {pf.get("note","")}</div>')

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
    rules_block = ''
    if pl.get('target1'):
        targets = (f'<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:8.5pt">'
                   f'<div>均價 <strong>${pl["avg_entry"]}</strong></div>'
                   f'<div>停損 <strong style="color:#b52a2a">${pl["stop"]}</strong></div>'
                   f'<div>目標1 <strong style="color:#1a7a4a">${pl["target1"]}</strong></div>'
                   f'<div>目標2 <strong style="color:#1a7a4a">${pl["target2"]}</strong></div>'
                   f'<div>R:R <strong>{pl["rr"]}</strong></div>'
                   f'<div>部位 <strong>{pl["position_pct"]}%</strong> (風險 ${pl["dollar_risk"]:,.0f})</div>'
                   f'<div>時間停損 <strong>{pl["time_stop_days"]} 交易日</strong></div></div>')
        # Explicit, copy-able rule card: ENTRY / TAKE-PROFIT / STOP — no ambiguity.
        t = pl['tranches']
        entry_rule = ' → '.join(f'${x["price"]} 買 {x["shares"]}股' for x in t)
        loss_per_share = round(pl['avg_entry'] - pl['stop'], 2)
        loss_pct = round((pl['stop'] / pl['avg_entry'] - 1) * 100, 1) if pl['avg_entry'] else 0
        gain1_pct = round((pl['target1'] / pl['avg_entry'] - 1) * 100, 1) if pl['avg_entry'] else 0
        gain2_pct = round((pl['target2'] / pl['avg_entry'] - 1) * 100, 1) if pl['avg_entry'] else 0
        rules_block = (
            '<div style="margin-top:12px;display:grid;grid-template-columns:repeat(3,1fr);gap:10px">'
            f'<div style="background:#eef5ff;border-radius:6px;padding:9px 11px">'
            f'<div style="font-size:7.5pt;font-weight:700;color:#1a4d80;text-transform:uppercase;margin-bottom:3px">進場規則 ENTRY</div>'
            f'<div style="font-size:8pt;color:#374151;line-height:1.5">分三批掛單,不追價:<br><strong>{entry_rule}</strong><br>均價 <strong>${pl["avg_entry"]}</strong>、部位 {pl["position_pct"]}%</div></div>'
            f'<div style="background:#f0f7f2;border-radius:6px;padding:9px 11px">'
            f'<div style="font-size:7.5pt;font-weight:700;color:#1a7a4a;text-transform:uppercase;margin-bottom:3px">停利規則 TAKE-PROFIT</div>'
            f'<div style="font-size:8pt;color:#374151;line-height:1.5">到 <strong style="color:#1a7a4a">${pl["target1"]}</strong>(+{gain1_pct}%,20日線/中軌)減 {pl.get("scaleout1_pct",50)}%<br>→ 停損抬至成本價 <strong>${pl["avg_entry"]}</strong><br>到 <strong style="color:#1a7a4a">${pl["target2"]}</strong>(+{gain2_pct}%,50日線/合理價)出清剩餘<br>訊號出場:RSI(2)&gt;65 或 收復10日線 先了結首份<br>R:R = <strong>{pl["rr"]}</strong></div></div>'
            f'<div style="background:#fdecea;border-radius:6px;padding:9px 11px">'
            f'<div style="font-size:7.5pt;font-weight:700;color:#b52a2a;text-transform:uppercase;margin-bottom:3px">停損規則 STOP</div>'
            f'<div style="font-size:8pt;color:#374151;line-height:1.5">硬停損 <strong style="color:#b52a2a">${pl["stop"]}</strong>(−${loss_per_share}/股,{loss_pct}%)<br>時間停損 <strong>{pl["time_stop_days"]} 交易日</strong>無反彈<br>單筆最大風險 ${pl["dollar_risk"]:,.0f}</div></div>'
            '</div>'
            '<div style="margin-top:8px;font-size:7pt;color:#9ca3af;line-height:1.4">'
            '註:Larry Connors 的研究發現,對 RSI(2) 類型的均值回歸策略加上固定停損,'
            '會略微降低每筆期望值(因部分停損後仍會反彈)。本系統的回測也證實這點'
            '(無停損版本盈虧比 1.50 vs 加 −3×ATR 停損版本 1.43),'
            '但加停損把最深浮虧從 −56.9% 收斂到 −24.4%——'
            '我們選擇用較小的期望值換取可承受的尾部風險,因此上方仍列出硬停損。'
            '</div>')

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
        {_render_crowd_banner(s.get('crowd'))}
        <div style="font-size:8.8pt;color:#374151;background:#f8fafd;padding:9px 12px;border-radius:5px;margin-bottom:12px;line-height:1.55">{s['rationale']}</div>

        {gate_callout}

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
          <div style="font-size:8.5pt;font-weight:700;color:#0d3b6e;margin-bottom:8px">因子總表 — 六面向逐項打分(✓過關 / ~部分 / ✗未過 / ⛔阻擋進場)</div>
          {scorecard_html}
        </div>

        <div style="margin-top:14px;border-top:1px solid #eaeaea;padding-top:12px">
          <div style="font-size:8.5pt;font-weight:700;color:#0d3b6e;margin-bottom:8px">分批進場階梯(預先承諾的價位)</div>
          {pf_note}
          {ladder}
          <div style="margin-top:10px">{targets}</div>
          {rules_block}
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
    <div style="margin-top:10px;padding:10px 13px;background:#fffaf5;border-left:4px solid #d4900a;border-radius:4px">
    <strong style="color:#92400e">誠實的證據(別把它當打敗大盤的 alpha 引擎):</strong>
    對 84 檔、8 年、扣交易成本做了 point-in-time 歸因(只測不需未來資訊的核心)。結論:
    在流動性好的大型股上,<strong>策略對 SPY 沒有統計顯著的 alpha</strong>(α≈−3~−8%/年、t 不顯著),
    且風險調整後 <strong>Sharpe 低於直接持有 SPY</strong>。它<strong>不是</strong>選股 alpha 機器。
    <br><span style="color:#374151">真正被量到的 edge 集中在「恐慌」:進場 VIX&gt;30 的單均報酬 <strong style="color:#1a7a4a">+1.1%/筆</strong>、
    VIX 20–25 <strong style="color:#1a7a4a">+0.25%</strong>;而 VIX&lt;20 的日常抄底 ≈ <strong>0(已被套利)</strong>。</span>
    <br><strong style="color:#1a7a4a">所以本工具的定位:在恐慌與個股錯殺時、有紀律地把現金投出去的「控回撤加碼工具」。</strong>
    平靜盤它會自動降為觀察/試單,不裝有 edge。它的護城河是<strong>紀律與風控(不接刀、賠率≥1、組合不爆倉)</strong>,不是預測力。
    <span style="color:#6b7280">完整歸因見 attribution.json / phase1_selectivity.json。</span>
    </div>
  </div>
  <div class="kpi-strip">
    <div class="kpi"><div class="lbl">分析標的</div><div class="val" style="color:#0d3b6e">{n}</div></div>
    <div class="kpi"><div class="lbl">高把握抄底</div><div class="val" style="color:#1a7a4a">{n_strong}</div></div>
    <div class="kpi"><div class="lbl">投機性抄底</div><div class="val" style="color:#d4900a">{n_spec}</div></div>
    <div class="kpi"><div class="lbl">接刀風險(避開)</div><div class="val" style="color:#b52a2a">{n_knife}</div></div>
  </div>
  {portfolio}
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

    # ── Portfolio construction layer: size the candidates AS A BOOK, not in isolation ──
    portfolio_summary = None
    try:
        from portfolio_construct import construct_portfolio, print_portfolio
        portfolio_summary = construct_portfolio(results, regime=r)
        print_portfolio(portfolio_summary)
    except Exception as e:
        print(f"  [portfolio] 組合層略過:{e}")

    if not args.no_report:
        path = render_report(results, portfolio=portfolio_summary)
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
