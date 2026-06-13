"""
Data Fetcher: pulls all available financial data via yfinance
Includes sector multiples lookup and macro context
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SECTOR DEFAULT MULTIPLES (Damodaran Jan 2025)
# EV/EBITDA | P/E | EV/Revenue
# ─────────────────────────────────────────────
SECTOR_MULTIPLES = {
    'Technology':          {'ev_ebitda': 22.0,  'pe': 30.0,  'ev_revenue': 6.5},
    'Software':            {'ev_ebitda': 28.0,  'pe': 40.0,  'ev_revenue': 9.0},
    'Semiconductors':      {'ev_ebitda': 20.0,  'pe': 28.0,  'ev_revenue': 5.5},
    'Healthcare':          {'ev_ebitda': 16.0,  'pe': 22.0,  'ev_revenue': 3.0},
    'Biotechnology':       {'ev_ebitda': 18.0,  'pe': 35.0,  'ev_revenue': 8.0},
    'Pharmaceuticals':     {'ev_ebitda': 14.0,  'pe': 18.0,  'ev_revenue': 4.0},
    'Financial Services':  {'ev_ebitda': 14.0,  'pe': 15.0,  'ev_revenue': 3.5},
    'Banks':               {'ev_ebitda': 10.0,  'pe': 12.0,  'ev_revenue': 2.5},
    'Consumer Cyclical':   {'ev_ebitda': 13.0,  'pe': 18.0,  'ev_revenue': 1.8},
    'Consumer Defensive':  {'ev_ebitda': 15.0,  'pe': 20.0,  'ev_revenue': 2.5},
    'Energy':              {'ev_ebitda': 7.0,   'pe': 12.0,  'ev_revenue': 1.5},
    'Utilities':           {'ev_ebitda': 12.0,  'pe': 16.0,  'ev_revenue': 2.8},
    'Industrials':         {'ev_ebitda': 14.0,  'pe': 20.0,  'ev_revenue': 2.2},
    'Materials':           {'ev_ebitda': 11.0,  'pe': 16.0,  'ev_revenue': 2.0},
    'Real Estate':         {'ev_ebitda': 18.0,  'pe': 30.0,  'ev_revenue': 8.0},
    'Communication Services': {'ev_ebitda': 14.0, 'pe': 20.0, 'ev_revenue': 3.0},
    'default':             {'ev_ebitda': 14.0,  'pe': 20.0,  'ev_revenue': 2.5},
}


def get_sector_multiples(sector: str) -> dict:
    for key in SECTOR_MULTIPLES:
        if key.lower() in (sector or '').lower() or (sector or '').lower() in key.lower():
            return SECTOR_MULTIPLES[key]
    return SECTOR_MULTIPLES['default']


# ─────────────────────────────────────────────
# COUNTRY RISK PREMIUM (Damodaran-style, additive to ERP)
# Used to lift cost of equity for non-US domiciled names / ADRs
# ─────────────────────────────────────────────
COUNTRY_RISK_PREMIUM = {
    'United States': 0.000, 'Canada': 0.000, 'United Kingdom': 0.006,
    'Germany': 0.000, 'France': 0.006, 'Switzerland': 0.000,
    'Netherlands': 0.000, 'Ireland': 0.010, 'Finland': 0.006,
    'Sweden': 0.000, 'Denmark': 0.000, 'Japan': 0.006,
    'Korea': 0.009, 'South Korea': 0.009, 'Taiwan': 0.009,
    'China': 0.028, 'Hong Kong': 0.012, 'India': 0.030,
    'Brazil': 0.044, 'Mexico': 0.034, 'Israel': 0.012,
    'Singapore': 0.000, 'Australia': 0.000,
}


def get_country_risk_premium(country: str) -> float:
    if not country:
        return 0.0
    return COUNTRY_RISK_PREMIUM.get(country.strip(), 0.012)  # 1.2% default for unlisted


def get_fx_rate(from_ccy: str, to_ccy: str = 'USD') -> float:
    """
    Spot FX so that  amount_in_from_ccy × rate = amount_in_to_ccy.
    Needed for ADRs whose financials are reported in a foreign currency (e.g. TSM
    files in TWD) while the price/shares are in USD — otherwise absolute FCF, revenue
    and net debt are in the wrong unit and the DCF per share is nonsense.
    Returns 1.0 on same currency or any failure.
    """
    if not from_ccy or from_ccy == to_ccy:
        return 1.0
    for sym in (f'{from_ccy}{to_ccy}=X', f'{to_ccy}{from_ccy}=X'):
        try:
            h = yf.Ticker(sym).history(period='5d')
            if not h.empty:
                px = float(h['Close'].iloc[-1])
                if px > 0:
                    return px if sym.startswith(from_ccy) else 1.0 / px
        except Exception:
            continue
    return 1.0


# ─────────────────────────────────────────────
# MACRO DATA
# ─────────────────────────────────────────────
def get_macro_context() -> dict:
    """Pull current 10-yr Treasury yield as risk-free rate proxy"""
    try:
        tnx = yf.Ticker('^TNX')
        hist = tnx.history(period='5d')
        rf_rate = hist['Close'].iloc[-1] / 100 if not hist.empty else 0.0435
    except Exception:
        rf_rate = 0.0435

    try:
        vix = yf.Ticker('^VIX')
        vix_hist = vix.history(period='5d')
        vix_level = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20.0
    except Exception:
        vix_level = 20.0

    return {
        'risk_free_rate': rf_rate,
        'vix': vix_level,
        'fetch_date': datetime.now().strftime('%Y-%m-%d'),
    }


# ─────────────────────────────────────────────
# MAIN COMPANY DATA FETCH
# ─────────────────────────────────────────────
def fetch_company_data(ticker_symbol: str) -> dict:
    print(f"  [+] Fetching data for {ticker_symbol.upper()}...")

    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info or {}

    # Price history
    hist_1y = ticker.history(period='1y')
    hist_5y = ticker.history(period='5y')

    # Financial statements
    try:
        income_stmt = ticker.income_stmt  # annual
    except Exception:
        income_stmt = pd.DataFrame()

    try:
        cashflow = ticker.cash_flow  # annual
    except Exception:
        cashflow = pd.DataFrame()

    try:
        balance_sheet = ticker.balance_sheet  # annual
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        quarterly_income = ticker.quarterly_income_stmt
    except Exception:
        quarterly_income = pd.DataFrame()

    try:
        quarterly_cashflow = ticker.quarterly_cash_flow
    except Exception:
        quarterly_cashflow = pd.DataFrame()

    # Analyst estimates
    try:
        analyst_price_targets = ticker.analyst_price_targets
    except Exception:
        analyst_price_targets = {}

    try:
        earnings_estimate = ticker.earnings_estimate
    except Exception:
        earnings_estimate = pd.DataFrame()

    try:
        revenue_estimate = ticker.revenue_estimate
    except Exception:
        revenue_estimate = pd.DataFrame()

    # Institutional holders
    try:
        institutional = ticker.institutional_holders
    except Exception:
        institutional = pd.DataFrame()

    # Insider transactions
    try:
        insider = ticker.insider_transactions
    except Exception:
        insider = pd.DataFrame()

    # Options-implied metrics (ATM IV, put-call skew, term structure)
    spot = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0
    options_metrics = _compute_options_metrics(ticker, spot)
    implied_vol = options_metrics.get('atm_iv')  # kept for backward compatibility

    # Compute beta from regression if not provided
    beta = info.get('beta') or _compute_beta(hist_1y)

    # Revenue series from income statement
    revenue_series = _extract_revenue_series(income_stmt)

    return {
        'symbol': ticker_symbol.upper(),
        'info': info,
        'hist_1y': hist_1y,
        'hist_5y': hist_5y,
        'income_stmt': income_stmt,
        'cashflow': cashflow,
        'balance_sheet': balance_sheet,
        'quarterly_income': quarterly_income,
        'quarterly_cashflow': quarterly_cashflow,
        'analyst_price_targets': analyst_price_targets,
        'earnings_estimate': earnings_estimate,
        'revenue_estimate': revenue_estimate,
        'institutional': institutional,
        'insider': insider,
        'implied_vol': implied_vol,
        'options_metrics': options_metrics,
        'beta': beta,
        'revenue_series': revenue_series,
    }


def _clean_chain(chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter an option chain to QUOTABLE strikes. yfinance pads illiquid/deep-ITM strikes
    with junk IV (e.g. 0.00001, or tiny placeholders) — using those produces fake
    signals, so we require a sane IV band AND some liquidity (OI or volume).
    """
    if chain_df is None or chain_df.empty or 'impliedVolatility' not in chain_df.columns:
        return pd.DataFrame()
    df = chain_df.copy()
    df = df[(df['impliedVolatility'] >= 0.03) & (df['impliedVolatility'] <= 3.0)]
    oi = df['openInterest'].fillna(0) if 'openInterest' in df.columns else 0
    vol = df['volume'].fillna(0) if 'volume' in df.columns else 0
    liq = (oi > 0) | (vol > 0)
    if hasattr(liq, 'any') and liq.any():
        df = df[liq]
    return df


def _atm_iv(chain_df: pd.DataFrame, spot: float):
    """Median IV of strikes within ±7% of spot (robust to single junk quotes)."""
    df = _clean_chain(chain_df)
    if df.empty:
        return None
    band = df[(df['strike'] >= spot * 0.93) & (df['strike'] <= spot * 1.07)]
    use = band if len(band) >= 2 else df.iloc[(df['strike'] - spot).abs().argsort()[:3]]
    if use.empty:
        return None
    return float(use['impliedVolatility'].median())


def _otm_iv(chain_df: pd.DataFrame, spot: float, moneyness: float, is_put: bool):
    """Median IV of OTM strikes in a band around `moneyness` away from spot."""
    df = _clean_chain(chain_df)
    if df.empty:
        return None
    target = spot * (1 - moneyness) if is_put else spot * (1 + moneyness)
    lo, hi = min(spot, target) * 0.97, max(spot, target) * 1.03
    band = df[(df['strike'] >= lo) & (df['strike'] <= hi)]
    if is_put:
        band = band[band['strike'] <= spot]
    else:
        band = band[band['strike'] >= spot]
    if band.empty:
        idx = (df['strike'] - target).abs().idxmin()
        return float(df.loc[idx, 'impliedVolatility'])
    return float(band['impliedVolatility'].median())


def _compute_options_metrics(ticker, spot: float) -> dict:
    """
    ATM implied vol, put-call skew (OTM put IV − OTM call IV), and term structure
    (near vs ~3-month IV). Skew > 0 means downside protection is bid up (fear).
    Heavily guarded — single-stock option chains are often sparse/illiquid.
    """
    out = {'atm_iv': None, 'put_skew': None, 'term_structure': None,
           'put_call_iv_ratio': None, 'expiry_days': None}
    if not spot or spot <= 0:
        return out
    try:
        import pandas as _pd
        from datetime import datetime as _dt
        expirations = ticker.options
        if not expirations:
            return out
        # Pick the nearest expiry that is at least ~25 days out (avoid 0-DTE noise)
        today = _dt.now().date()
        chosen = None
        for e in expirations:
            try:
                d = (_dt.strptime(e, '%Y-%m-%d').date() - today).days
            except Exception:
                continue
            if d >= 25:
                chosen = (e, d)
                break
        if chosen is None and expirations:
            e = expirations[0]
            try:
                d = (_dt.strptime(e, '%Y-%m-%d').date() - today).days
            except Exception:
                d = None
            chosen = (e, d)

        e, d = chosen
        out['expiry_days'] = d
        oc = ticker.option_chain(e)
        atm_call = _atm_iv(oc.calls, spot)
        atm_put = _atm_iv(oc.puts, spot)
        atm = np.nanmean([v for v in (atm_call, atm_put) if v is not None]) if (atm_call or atm_put) else None
        out['atm_iv'] = round(float(atm), 4) if atm is not None else None

        otm_put = _otm_iv(oc.puts, spot, 0.10, is_put=True)
        otm_call = _otm_iv(oc.calls, spot, 0.10, is_put=False)
        if otm_put is not None and otm_call is not None:
            out['put_skew'] = round(otm_put - otm_call, 4)
        if atm_put is not None and atm_call is not None and atm_call > 0:
            out['put_call_iv_ratio'] = round(atm_put / atm_call, 3)

        # Term structure: ~3-month expiry IV minus near IV (positive = upward sloping/normal)
        far = None
        for ee in expirations:
            try:
                dd = (_dt.strptime(ee, '%Y-%m-%d').date() - today).days
            except Exception:
                continue
            if dd >= 80:
                far_oc = ticker.option_chain(ee)
                far = _atm_iv(far_oc.calls, spot)
                break
        if far is not None and out['atm_iv'] is not None:
            out['term_structure'] = round(far - out['atm_iv'], 4)
    except Exception:
        pass
    return out


def _compute_beta(hist: pd.DataFrame, market_period: str = '1y') -> float:
    """Compute beta via OLS regression against SPY"""
    try:
        spy = yf.Ticker('SPY').history(period=market_period)
        if hist.empty or spy.empty:
            return 1.0

        stock_ret = hist['Close'].pct_change().dropna()
        market_ret = spy['Close'].pct_change().dropna()

        aligned = pd.concat([stock_ret, market_ret], axis=1, join='inner')
        aligned.columns = ['stock', 'market']
        aligned = aligned.dropna()

        if len(aligned) < 20:
            return 1.0

        slope, _, _, _, _ = stats.linregress(aligned['market'], aligned['stock'])
        return round(max(min(slope, 3.0), 0.1), 2)
    except Exception:
        return 1.0


def _extract_revenue_series(income_stmt: pd.DataFrame) -> pd.Series:
    if income_stmt is None or income_stmt.empty:
        return pd.Series(dtype=float)
    for label in ['Total Revenue', 'Revenue', 'Net Revenue']:
        if label in income_stmt.index:
            return income_stmt.loc[label].dropna()
    return pd.Series(dtype=float)


def _extract_fcf_series(cashflow: pd.DataFrame) -> pd.Series:
    if cashflow is None or cashflow.empty:
        return pd.Series(dtype=float)

    ocf = None
    for label in ['Operating Cash Flow', 'Cash From Operating Activities']:
        if label in cashflow.index:
            ocf = cashflow.loc[label]
            break

    capex = None
    for label in ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of PPE']:
        if label in cashflow.index:
            capex = cashflow.loc[label]
            break

    if ocf is not None:
        if capex is not None:
            return (ocf + capex).dropna()
        return ocf.dropna()

    return pd.Series(dtype=float)


def _extract_sbc_series(cashflow: pd.DataFrame) -> pd.Series:
    """Stock-based compensation by year (positive numbers). Empty if not reported."""
    if cashflow is None or cashflow.empty:
        return pd.Series(dtype=float)
    for label in ['Stock Based Compensation', 'Stock-Based Compensation',
                  'Share Based Compensation']:
        if label in cashflow.index:
            return cashflow.loc[label].dropna().abs()
    return pd.Series(dtype=float)


def normalized_base_fcf(cashflow: pd.DataFrame, income: pd.DataFrame,
                        ttm_fcf: float, deduct_sbc: bool = True,
                        years: int = 3) -> dict:
    """
    Build a normalized base-year FCF instead of trusting a single noisy TTM point.

    Steps:
      1. Pull the last `years` of FCF (OCF - CapEx) from filings.
      2. Compute FCF margin each year; take the median margin (robust to one-offs).
      3. Apply median margin to the latest revenue -> margin-normalized FCF.
      4. Blend 50/50 with the latest reported FCF so we don't drift from reality.
      5. Optionally deduct stock-based compensation (yfinance/OCF adds SBC back,
         which flatters FCF while shareholders are diluted — treat SBC as a real cost).

    Returns dict with base_fcf, components, and the SBC drag so the report can show it.
    """
    fcf_hist = _extract_fcf_series(cashflow)
    rev = _extract_revenue_series(income)
    sbc_hist = _extract_sbc_series(cashflow)

    latest_fcf = None
    if not fcf_hist.empty:
        latest_fcf = float(fcf_hist.iloc[0])
    elif ttm_fcf and ttm_fcf > 0:
        latest_fcf = float(ttm_fcf)

    # Median FCF margin over the window
    margin_norm_fcf = None
    median_margin = None
    if not fcf_hist.empty and not rev.empty:
        common = fcf_hist.index.intersection(rev.index)[:years]
        margins = []
        for idx in common:
            r = float(rev[idx])
            if r > 0:
                margins.append(float(fcf_hist[idx]) / r)
        if margins:
            median_margin = float(np.median(margins))
            latest_rev = float(rev.iloc[0])
            if latest_rev > 0:
                margin_norm_fcf = median_margin * latest_rev

    # Blend reported latest with margin-normalized
    if latest_fcf is not None and margin_norm_fcf is not None:
        base = 0.5 * latest_fcf + 0.5 * margin_norm_fcf
    elif latest_fcf is not None:
        base = latest_fcf
    elif margin_norm_fcf is not None:
        base = margin_norm_fcf
    else:
        base = float(ttm_fcf) if ttm_fcf else 0.0

    # SBC drag (latest year)
    sbc_latest = float(sbc_hist.iloc[0]) if not sbc_hist.empty else 0.0
    sbc_pct_of_fcf = (sbc_latest / base) if base > 0 else 0.0
    base_after_sbc = base - sbc_latest if deduct_sbc else base

    return {
        'base_fcf': round(base_after_sbc, 0),
        'base_fcf_pre_sbc': round(base, 0),
        'reported_latest_fcf': round(latest_fcf, 0) if latest_fcf is not None else None,
        'margin_normalized_fcf': round(margin_norm_fcf, 0) if margin_norm_fcf is not None else None,
        'median_fcf_margin': round(median_margin, 4) if median_margin is not None else None,
        'sbc_latest': round(sbc_latest, 0),
        'sbc_pct_of_fcf': round(sbc_pct_of_fcf, 4),
        'sbc_deducted': deduct_sbc,
    }


def estimate_sbc_dilution(cashflow: pd.DataFrame, info: dict) -> float:
    """
    Annual share-count growth from net stock-based dilution.
    Approx: SBC / market cap, netted against buybacks if visible.
    Clamped to [0, 4%]/yr. Used to grow share count through the DCF horizon.
    """
    try:
        sbc_hist = _extract_sbc_series(cashflow)
        if sbc_hist.empty:
            return 0.0
        sbc = float(sbc_hist.iloc[0])
        mcap = info.get('marketCap') or 0
        if mcap <= 0:
            return 0.0
        gross_dilution = sbc / mcap

        # Net out buybacks if reported
        buyback = 0.0
        for label in ['Repurchase Of Capital Stock', 'Repurchase Of Stock',
                      'Common Stock Payments']:
            if label in cashflow.index:
                buyback = abs(float(cashflow.loc[label].iloc[0]))
                break
        net = gross_dilution - (buyback / mcap)
        return float(max(min(net, 0.04), 0.0))
    except Exception:
        return 0.0


def get_price_performance(hist_1y: pd.DataFrame, hist_5y: pd.DataFrame) -> dict:
    results = {}
    for label, hist, days in [('1m', hist_1y, 21), ('3m', hist_1y, 63),
                               ('6m', hist_1y, 126), ('1y', hist_1y, 252),
                               ('3y', hist_5y, 756), ('5y', hist_5y, 1260)]:
        try:
            if len(hist) > days:
                ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[-days] - 1) * 100
                results[label] = round(ret, 1)
        except Exception:
            pass
    return results


def get_52w_stats(hist_1y: pd.DataFrame) -> dict:
    if hist_1y.empty:
        return {}
    return {
        'high_52w': round(hist_1y['High'].max(), 2),
        'low_52w': round(hist_1y['Low'].min(), 2),
        'avg_volume_30d': int(hist_1y['Volume'].tail(30).mean()) if len(hist_1y) >= 30 else 0,
    }
