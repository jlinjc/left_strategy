"""
Predictive Signal Layer
───────────────────────
Valuation tells you whether a stock is cheap. It does NOT tell you whether it will
go up — cheap stocks can stay cheap for years. The signals here target *forward
returns* directly, using the alpha sources with the most robust out-of-sample
evidence in public equities:

  • Price momentum (12-1)        — Jegadeesh-Titman; 3-12m horizon
  • Short-term reversal (1m)     — negative predictor over ~1m
  • Estimate-revision momentum   — among the most reliable public signals
  • Earnings surprise / PEAD     — post-earnings-announcement drift, 1-3m
  • Quality (Piotroski)          — quality earns a premium
  • Insider net buying           — mild positive signal
  • Realized volatility / risk   — low-vol anomaly + risk scaling

Each sub-signal is computed point-in-time and mapped to a bounded score in [-1, +1]
(via absolute, literature-informed transforms) so a single name is interpretable on
its own. For a universe, `cross_sectional_scores()` converts raw values to
percentile ranks — because return prediction is fundamentally RELATIVE.

The composite is a deliberate, transparent weighted blend — not a black box — so it
can be inspected, overridden, and (critically) BACKTESTED. See backtest.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# Composite weights — CALIBRATED against backtest.py, not guessed.
# Backtest (50 large-caps, 73 months) found:
#   • momentum_12_1 / 6m : IC +0.05..+0.07, ~64% months positive  → trade it, top weight
#   • reversal_1m        : IC was POSITIVE (large-caps continue, not revert) — my signal
#                          bets on reversal, i.e. WRONG SIGN here → zero-weighted
#   • low_vol            : IC −0.05..−0.14 (t=−3.3); high-vol won this regime → zero-weighted
# Revision-momentum, PEAD and quality carry strong out-of-sample evidence in the
# literature and are validated LIVE via snapshot.py (can't be reconstructed PIT here).
# The two zero-weighted factors are still COMPUTED and shown — we measure 7, trade the 5
# that pay. Re-run backtest.py periodically and revisit, especially the low-vol regime call.
SIGNAL_WEIGHTS = {
    'price_momentum': 0.28,
    'revision_momentum': 0.21,
    'earnings_surprise': 0.17,
    'quality': 0.14,
    'short_interest': 0.10,        # 籌碼面: high short interest → negative (live-validated)
    'options_skew': 0.06,          # put skew / IV positioning → negative (live-validated)
    'insider': 0.04,
    'short_term_reversal': 0.00,   # wrong sign for large-caps (backtest); kept for display
    'low_volatility': 0.00,        # negative IC this regime (backtest); kept for display
}


def _clip(x, lo=-1.0, hi=1.0):
    return max(min(x, hi), lo)


def _ret(series: pd.Series, lookback: int, skip: int = 0):
    """Total return over [t-lookback-skip, t-skip] in trading days."""
    if series is None or len(series) <= lookback + skip:
        return None
    end = series.iloc[-1 - skip]
    start = series.iloc[-1 - skip - lookback]
    if start and start > 0:
        return float(end / start - 1)
    return None


# ─────────────────────────────────────────────
# PRICE MOMENTUM (12-1) + reversal + 52w position
# ─────────────────────────────────────────────
def price_momentum(hist_1y: pd.DataFrame, hist_5y: pd.DataFrame) -> dict:
    out = {'raw_12_1': None, 'ret_1m': None, 'ret_3m': None, 'ret_6m': None,
           'ret_12m': None, 'dist_52w_high': None, 'score': 0.0}
    close5 = hist_5y['Close'] if (hist_5y is not None and not hist_5y.empty) else None
    close1 = hist_1y['Close'] if (hist_1y is not None and not hist_1y.empty) else None

    if close5 is not None and len(close5) > 260:
        r12 = _ret(close5, 252)
        r1 = _ret(close5, 21)
        # 12-1 momentum: 12m return excluding the most recent month
        mom = _ret(close5, 252 - 21, skip=21)
        out['raw_12_1'] = round(mom, 4) if mom is not None else None
        out['ret_12m'] = round(r12, 4) if r12 is not None else None
        out['ret_1m'] = round(r1, 4) if r1 is not None else None
        out['ret_3m'] = round(_ret(close5, 63), 4) if _ret(close5, 63) is not None else None
        out['ret_6m'] = round(_ret(close5, 126), 4) if _ret(close5, 126) is not None else None

    if close1 is not None and not close1.empty:
        hi = float(close1.max())
        cur = float(close1.iloc[-1])
        if hi > 0:
            out['dist_52w_high'] = round(cur / hi - 1, 4)  # negative = below high

    # Score: tanh-squash 12-1 momentum (~+30% maps near +0.6). Stocks near 52w high
    # get a small anchoring bonus.
    if out['raw_12_1'] is not None:
        s = np.tanh(out['raw_12_1'] * 2.0)
        if out['dist_52w_high'] is not None and out['dist_52w_high'] > -0.05:
            s += 0.15
        out['score'] = round(_clip(s), 3)
    return out


def short_term_reversal(mom: dict) -> dict:
    """Last-1m return is a NEGATIVE predictor at ~1m horizon (mean reversion)."""
    r1 = mom.get('ret_1m')
    if r1 is None:
        return {'ret_1m': None, 'score': 0.0}
    return {'ret_1m': r1, 'score': round(_clip(-np.tanh(r1 * 3.0)), 3)}


# ─────────────────────────────────────────────
# REVISION MOMENTUM  (analysts raising/cutting estimates)
# ─────────────────────────────────────────────
def revision_momentum(analyst_data: dict, info: dict) -> dict:
    """
    Net up-vs-down EPS revisions over the last 30 days across the current and next
    fiscal year. Also folds in earningsGrowth direction as a fallback.
    """
    out = {'up': 0, 'down': 0, 'net_ratio': None, 'score': 0.0, 'source': 'none'}
    er = (analyst_data or {}).get('eps_revisions')
    if er is not None and not er.empty:
        up = down = 0
        for period in ['0y', '+1y', '0q', '+1q']:
            if period in er.index:
                row = er.loc[period]
                up += int(row.get('upLast30days', 0) or 0)
                down += int(row.get('downLast30days', 0) or 0)
        total = up + down
        if total > 0:
            net = (up - down) / total
            out.update(up=up, down=down, net_ratio=round(net, 3),
                       score=round(_clip(net), 3), source='eps_revisions')
            return out

    # Fallback: directional earnings growth sign (weak)
    eg = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth')
    if eg is not None:
        out.update(net_ratio=None, score=round(_clip(np.tanh(eg * 1.5)) * 0.4, 3),
                   source='earnings_growth_fallback')
    return out


# ─────────────────────────────────────────────
# EARNINGS SURPRISE / PEAD
# ─────────────────────────────────────────────
def earnings_surprise(ticker_symbol: str) -> dict:
    """
    Recent EPS surprise (%) drives post-earnings-announcement drift over 1-3 months.
    Uses the trailing 4 prints: weights the latest most.
    """
    out = {'last_surprise_pct': None, 'avg_surprise_pct': None,
           'beat_rate': None, 'score': 0.0}
    try:
        t = yf.Ticker(ticker_symbol)
        eh = t.earnings_history
        if eh is None or eh.empty:
            return out
        # columns typically: epsActual, epsEstimate, surprisePercent
        if 'epsActual' in eh.columns and 'epsEstimate' in eh.columns:
            df = eh.dropna(subset=['epsActual', 'epsEstimate']).tail(4)
            if df.empty:
                return out
            surp = []
            beats = 0
            for _, r in df.iterrows():
                est = float(r['epsEstimate'])
                act = float(r['epsActual'])
                if est != 0:
                    s = (act - est) / abs(est)
                    surp.append(s)
                    if act > est:
                        beats += 1
            if surp:
                last = surp[-1]
                avg = float(np.mean(surp))
                out['last_surprise_pct'] = round(last * 100, 1)
                out['avg_surprise_pct'] = round(avg * 100, 1)
                out['beat_rate'] = round(beats / len(surp), 2)
                # Weight latest surprise 60%, trailing average 40%
                blended = 0.6 * last + 0.4 * avg
                out['score'] = round(_clip(np.tanh(blended * 5.0)), 3)
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────
# QUALITY (reuse Piotroski F-score)
# ─────────────────────────────────────────────
def quality_signal(quality: dict) -> dict:
    if not quality:
        return {'f_score': None, 'score': 0.0}
    f = quality.get('score', 0)
    mx = quality.get('max', 9) or 9
    # Map 0..9 to -1..+1 centered at ~5
    s = (f - mx / 2) / (mx / 2)
    return {'f_score': f, 'score': round(_clip(s), 3)}


# ─────────────────────────────────────────────
# INSIDER NET BUYING
# ─────────────────────────────────────────────
def insider_signal(insider: pd.DataFrame) -> dict:
    out = {'net_shares': None, 'buys': 0, 'sells': 0, 'score': 0.0}
    if insider is None or insider.empty:
        return out
    try:
        buy_sh = sell_sh = 0.0
        buys = sells = 0
        for _, r in insider.iterrows():
            txt = str(r.get('Transaction', '') or r.get('Text', '')).lower()
            shares = r.get('Shares', 0) or 0
            try:
                shares = float(shares)
            except Exception:
                shares = 0
            if 'buy' in txt or 'purchase' in txt:
                buy_sh += shares; buys += 1
            elif 'sale' in txt or 'sell' in txt:
                sell_sh += shares; sells += 1
        net = buy_sh - sell_sh
        total = buy_sh + sell_sh
        out.update(net_shares=net, buys=buys, sells=sells)
        if total > 0:
            out['score'] = round(_clip(net / total), 3)
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────
# SHORT INTEREST / POSITIONING (籌碼面)
# ─────────────────────────────────────────────
def short_interest_signal(info: dict) -> dict:
    """
    Short interest is a robust NEGATIVE cross-sectional return predictor: heavily
    shorted names underperform on average (smart-money / borrow-cost signal). We also
    read the month-over-month change (shorts piling in = bearish; covering = bullish)
    and surface days-to-cover as squeeze context (high = squeeze risk on good news).

    NOTE: yfinance only exposes current + prior-month short interest, so this factor
    CANNOT be reconstructed point-in-time in backtest.py — it is validated LIVE via the
    snapshot track record (like revisions / PEAD).
    """
    out = {'short_pct_float': None, 'short_ratio_days': None, 'short_change_mom': None,
           'squeeze_risk': None, 'score': 0.0, 'available': False}

    short_pct = info.get('shortPercentOfFloat')
    if short_pct is None:
        sps = info.get('sharesPercentSharesOut')
        short_pct = sps
    days = info.get('shortRatio')  # days-to-cover
    cur = info.get('sharesShort')
    prior = info.get('sharesShortPriorMonth')

    if short_pct is None and cur is None:
        return out  # no positioning data

    out['available'] = True
    out['short_pct_float'] = round(short_pct, 4) if short_pct is not None else None
    out['short_ratio_days'] = round(days, 1) if days else None

    # Level: high short % → negative. ~5% float shorted ≈ neutral pivot.
    level_score = 0.0
    if short_pct is not None:
        level_score = -np.tanh((short_pct - 0.05) * 12.0)  # 20% float → ~ -0.9

    # Change: rising short interest = bearish, covering = bullish.
    change_score = 0.0
    if cur and prior and prior > 0:
        chg = cur / prior - 1
        out['short_change_mom'] = round(chg, 3)
        change_score = -np.tanh(chg * 2.5)

    # Squeeze context (informational, doesn't flip the sign): high days-to-cover +
    # recent covering is the classic squeeze setup.
    if days and short_pct is not None:
        out['squeeze_risk'] = 'high' if (days > 5 and short_pct > 0.10) else \
                              'moderate' if (days > 3 and short_pct > 0.05) else 'low'

    out['score'] = round(_clip(0.6 * level_score + 0.4 * change_score), 3)
    return out


# ─────────────────────────────────────────────
# LOW VOLATILITY (low-vol anomaly)
# ─────────────────────────────────────────────
def volatility_signal(hist_1y: pd.DataFrame) -> dict:
    out = {'realized_vol': None, 'score': 0.0}
    if hist_1y is None or hist_1y.empty:
        return out
    rets = hist_1y['Close'].pct_change().dropna()
    if len(rets) < 30:
        return out
    vol = float(rets.std() * np.sqrt(252))
    out['realized_vol'] = round(vol, 3)
    # Lower vol → mild positive. Center near 30% annualized.
    out['score'] = round(_clip((0.30 - vol) * 2.0), 3)
    return out


# ─────────────────────────────────────────────
# OPTIONS-IMPLIED (skew / IV vs realized / term structure)
# ─────────────────────────────────────────────
def options_signal(options_metrics: dict, realized_vol: float = None) -> dict:
    """
    Reads option-market positioning:
      • Put skew (OTM put IV − OTM call IV): elevated skew = downside hedging / informed
        put demand → NEGATIVE forward returns (Xing-Zhang-Zhao 2010). Primary driver.
      • Inverted term structure (near IV > far): near-term stress → mild negative.
      • IV richness vs realized vol: large premium = elevated fear/event risk → mild negative.
    Single-stock option data is noisy, hence a modest composite weight.
    NOTE: like short interest, not point-in-time reconstructable → validated LIVE.
    """
    out = {'atm_iv': None, 'put_skew': None, 'term_structure': None,
           'iv_vs_realized': None, 'available': False, 'score': 0.0}
    om = options_metrics or {}
    skew = om.get('put_skew')
    atm_iv = om.get('atm_iv')
    term = om.get('term_structure')

    # PLAUSIBILITY GATE — yfinance single-stock IV is often junk/placeholder in this
    # environment. An equity ATM IV below ~10% annualized is effectively impossible, and
    # an exactly-0.0 skew means no real skew data. Reject junk so the factor drops out of
    # the composite (better no signal than a fake one). Works when fed clean option data.
    if atm_iv is None or atm_iv < 0.10:
        # Implausible ATM IV ⇒ the whole chain's IV is junk ⇒ distrust skew too.
        return out  # available stays False → excluded from composite
    if skew is not None and abs(skew) < 1e-6:
        skew = None

    out['available'] = True
    out['atm_iv'] = atm_iv
    out['put_skew'] = skew
    out['term_structure'] = term

    score = 0.0
    n = 0
    if skew is not None:
        # ~0.03 skew is normal; higher = fear. Negative score for elevated skew.
        score += -np.tanh((skew - 0.03) * 15.0); n += 1
    if term is not None:
        # inverted (negative) term structure = near-term stress → negative
        score += np.tanh(term * 12.0) * 0.5; n += 1
    if atm_iv is not None and realized_vol:
        vrp = atm_iv - realized_vol
        out['iv_vs_realized'] = round(vrp, 3)
        # very rich IV vs realized = elevated fear → mild negative
        score += -np.tanh((vrp - 0.05) * 4.0) * 0.4; n += 1

    out['score'] = round(_clip(score / max(n, 1) * 1.3), 3) if n else 0.0
    return out


# ─────────────────────────────────────────────
# MASTER: build full signal profile for one name
# ─────────────────────────────────────────────
def build_signal_profile(data: dict, quality: dict) -> dict:
    info = data['info']
    ticker = data['symbol']
    analyst_data = (data.get('consensus_result') or {}).get('analyst_data') or {}

    mom = price_momentum(data.get('hist_1y'), data.get('hist_5y'))
    rev = revision_momentum(analyst_data, info)
    srev = short_term_reversal(mom)
    surp = earnings_surprise(ticker)
    qual = quality_signal(quality)
    si = short_interest_signal(info)
    ins = insider_signal(data.get('insider'))
    vol = volatility_signal(data.get('hist_1y'))
    opt = options_signal(data.get('options_metrics'), vol.get('realized_vol'))

    components = {
        'price_momentum': mom,
        'revision_momentum': rev,
        'earnings_surprise': surp,
        'quality': qual,
        'short_interest': si,
        'options_skew': opt,
        'short_term_reversal': srev,
        'insider': ins,
        'low_volatility': vol,
    }

    # Weighted composite over AVAILABLE signals (re-normalize weights for missing ones)
    used, wsum, acc = {}, 0.0, 0.0
    for key, w in SIGNAL_WEIGHTS.items():
        comp = components.get(key, {})
        sc = comp.get('score')
        # treat a hard-zero from a 'none'/unavailable source as missing
        unavailable = (
            (key == 'revision_momentum' and comp.get('source') == 'none') or
            (key == 'earnings_surprise' and comp.get('last_surprise_pct') is None) or
            (key == 'insider' and comp.get('net_shares') is None) or
            (key == 'short_interest' and not comp.get('available')) or
            (key == 'options_skew' and not comp.get('available')) or
            (key == 'price_momentum' and comp.get('raw_12_1') is None)
        )
        if unavailable:
            continue
        used[key] = sc
        acc += w * sc
        wsum += w
    composite = (acc / wsum) if wsum > 0 else 0.0
    composite = _clip(composite)

    # 0-100 conviction (50 = neutral)
    conviction = round(50 + composite * 50)
    label = ('Strong Positive' if composite > 0.4 else 'Positive' if composite > 0.15
             else 'Neutral' if composite > -0.15 else 'Negative' if composite > -0.4
             else 'Strong Negative')

    return {
        'components': components,
        'used_scores': used,
        'composite_score': round(composite, 3),   # -1..+1
        'conviction': conviction,                  # 0..100
        'label': label,
        'weights': SIGNAL_WEIGHTS,
    }


# ─────────────────────────────────────────────
# CROSS-SECTIONAL RANKING (for a universe / batch)
# ─────────────────────────────────────────────
def cross_sectional_scores(profiles: dict) -> dict:
    """
    profiles: {ticker: signal_profile}
    Converts each raw sub-signal to a within-universe percentile rank (0-100) and
    rebuilds a cross-sectional composite — the correct frame for return prediction.
    Returns {ticker: {'cs_composite': float, 'cs_rank_pct': float, 'percentiles': {...}}}.
    """
    if not profiles:
        return {}
    keys = list(SIGNAL_WEIGHTS.keys())
    # Collect raw composite contributions per signal
    raw = {k: {} for k in keys}
    for tk, prof in profiles.items():
        for k in keys:
            sc = (prof.get('components', {}).get(k, {}) or {}).get('score')
            if sc is not None:
                raw[k][tk] = sc

    out = {tk: {'percentiles': {}} for tk in profiles}
    for k in keys:
        vals = raw[k]
        if len(vals) < 2:
            for tk in vals:
                out[tk]['percentiles'][k] = 50.0
            continue
        s = pd.Series(vals)
        pct = s.rank(pct=True) * 100
        for tk, p in pct.items():
            out[tk]['percentiles'][k] = round(float(p), 1)

    for tk in profiles:
        acc = wsum = 0.0
        for k, w in SIGNAL_WEIGHTS.items():
            p = out[tk]['percentiles'].get(k)
            if p is not None:
                acc += w * (p - 50) / 50.0   # -1..+1
                wsum += w
        comp = (acc / wsum) if wsum else 0.0
        out[tk]['cs_composite'] = round(comp, 3)

    # Rank the cross-sectional composite itself
    comps = pd.Series({tk: out[tk]['cs_composite'] for tk in profiles})
    rnk = comps.rank(pct=True) * 100
    for tk in profiles:
        out[tk]['cs_rank_pct'] = round(float(rnk[tk]), 1)
    return out
