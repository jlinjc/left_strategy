"""
Peer-Based Forward P/E Valuation Model
Three-layer approach:
  1. PEG-adjusted P/E  — normalize for growth differential vs. peers
  2. Historical P/E band — mean-reversion anchor for this stock
  3. Football field PT   — Bear / Base / Bull scenarios
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SECTOR PEER UNIVERSE (手動維護最佳，以下為預設)
# Format: sector → [(ticker, weight), ...]
# Weight = market cap proxy; set equal if unsure
# ─────────────────────────────────────────────
SECTOR_PEERS = {
    'Technology': [
        ('MSFT', 1.0), ('GOOGL', 1.0), ('META', 1.0),
        ('AAPL', 1.0), ('AMZN', 0.8), ('NVDA', 0.8),
    ],
    'Semiconductors': [
        ('NVDA', 1.0), ('AMD', 1.0), ('INTC', 1.0),
        ('QCOM', 0.8), ('AVGO', 0.8), ('TXN', 0.8),
    ],
    'Software': [
        ('MSFT', 1.0), ('ADBE', 1.0), ('CRM', 1.0),
        ('NOW', 0.8), ('WDAY', 0.8), ('ORCL', 0.8),
    ],
    # NB: 'Consumer Electronics' is overridden by the more specific entry below.
    'Healthcare': [
        ('JNJ', 1.0), ('UNH', 1.0), ('ABT', 0.8),
        ('MDT', 0.8), ('TMO', 0.8),
    ],
    'Pharmaceuticals': [
        ('LLY', 1.0), ('PFE', 1.0), ('MRK', 1.0),
        ('ABBV', 0.8), ('BMY', 0.8),
    ],
    'Biotechnology': [
        ('AMGN', 1.0), ('GILD', 1.0), ('REGN', 0.8),
        ('VRTX', 0.8), ('BIIB', 0.7),
    ],
    'Financial Services': [
        ('JPM', 1.0), ('BAC', 1.0), ('GS', 0.8),
        ('MS', 0.8), ('BRK-B', 0.8),
    ],
    'Consumer Cyclical': [
        ('AMZN', 1.0), ('TSLA', 0.8), ('HD', 0.8),
        ('MCD', 0.8), ('NKE', 0.7),
    ],
    'Consumer Defensive': [
        ('PG', 1.0), ('KO', 1.0), ('PEP', 1.0),
        ('WMT', 0.8), ('COST', 0.8),
    ],
    'Communication Services': [
        ('GOOGL', 1.0), ('META', 1.0), ('NFLX', 0.8),
        ('T', 0.7), ('VZ', 0.7),
    ],
    'Industrials': [
        ('HON', 1.0), ('MMM', 0.8), ('GE', 0.8),
        ('CAT', 0.8), ('UPS', 0.8),
    ],
    'Energy': [
        ('XOM', 1.0), ('CVX', 1.0), ('COP', 0.8),
        ('SLB', 0.7), ('EOG', 0.7),
    ],
    # Industry-level overrides (more specific than sector)
    # AAPL trades as a mega-cap quality compounder, not alongside low-multiple
    # hardware OEMs — value it against the names the market actually brackets it with.
    'Consumer Electronics': [
        ('MSFT', 1.0), ('GOOGL', 1.0), ('META', 0.9),
        ('AMZN', 0.8), ('SONY', 0.5),
    ],
    'Semiconductor Equipment': [
        ('AMAT', 1.0), ('LRCX', 1.0), ('KLAC', 0.9), ('ASML', 0.8), ('TER', 0.7),
    ],
    'Electronic Components': [
        ('GLW', 1.0), ('TE', 0.8), ('APH', 0.8), ('FLEX', 0.7), ('JBL', 0.7),
    ],
    'Telecom Equipment': [
        ('NOK', 1.0), ('ERIC', 1.0), ('JNPR', 0.8), ('CSCO', 0.8), ('ZBRA', 0.6),
    ],
    'Optical Components': [
        ('COHR', 1.0), ('IIVI', 0.9), ('LITE', 0.9), ('FNSR', 0.7), ('NPTN', 0.7),
    ],
    'Data Storage': [
        ('WDC', 1.0), ('STX', 1.0), ('NTAP', 0.8), ('PSTG', 0.8), ('SNDK', 0.7),
    ],
    'default': [
        ('SPY', 1.0),
    ],
}


def _get_peer_list(sector: str, industry: str, subject_ticker: str) -> list[str]:
    """Return peer tickers, excluding the subject itself. Industry takes priority over sector."""
    industry_str = (industry or '').lower()
    sector_str = (sector or '').lower()

    # 1. Try exact industry match first (most specific)
    for key in SECTOR_PEERS:
        if key.lower() == industry_str:
            return [t for t, _ in SECTOR_PEERS[key] if t.upper() != subject_ticker.upper()]

    # 2. Try partial industry match
    for key in SECTOR_PEERS:
        kl = key.lower()
        if kl in industry_str or industry_str in kl:
            return [t for t, _ in SECTOR_PEERS[key] if t.upper() != subject_ticker.upper()]

    # 3. Try sector match
    sector_key = None
    for key in SECTOR_PEERS:
        kl = key.lower()
        if kl in sector_str or sector_str in kl:
            sector_key = key
            break

    peers_raw = SECTOR_PEERS.get(sector_key) or SECTOR_PEERS['default']
    return [t for t, _ in peers_raw if t.upper() != subject_ticker.upper()]


# ─────────────────────────────────────────────
# FETCH PEER MULTIPLES
# ─────────────────────────────────────────────
def fetch_peer_multiples(peer_tickers: list[str]) -> pd.DataFrame:
    """
    Pull forward P/E, PEG, P/S, EPS growth for each peer.
    Returns a clean DataFrame; skips tickers that fail.
    """
    rows = []
    for t in peer_tickers:
        try:
            info = yf.Ticker(t).info or {}
            fwd_pe = info.get('forwardPE')
            trail_pe = info.get('trailingPE')
            peg = info.get('trailingPegRatio') or info.get('pegRatio')
            eps_fwd = info.get('forwardEps')
            eps_trail = info.get('trailingEps')
            rev_growth = info.get('revenueGrowth') or 0
            earn_growth = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth') or 0
            op_margin = info.get('operatingMargins') or 0
            mc = info.get('marketCap') or 0
            name = info.get('shortName') or t

            if fwd_pe and fwd_pe > 0:
                rows.append({
                    'ticker': t,
                    'name': name,
                    'market_cap_b': mc / 1e9,
                    'fwd_pe': fwd_pe,
                    'trail_pe': trail_pe,
                    'peg': peg,
                    'eps_fwd': eps_fwd,
                    'eps_trail': eps_trail,
                    'rev_growth_pct': rev_growth * 100,
                    'earn_growth_pct': earn_growth * 100,
                    'op_margin_pct': op_margin * 100,
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Remove outliers: P/E > 200 or < 0
    df = df[(df['fwd_pe'] > 0) & (df['fwd_pe'] < 200)]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# PEG-ADJUSTED FAIR P/E
# Core logic: P/E ∝ growth rate (PEG normalization)
# ─────────────────────────────────────────────
def peg_adjusted_pe(peer_df: pd.DataFrame, subject_growth_pct: float,
                    subject_quality_premium: float = 0.0) -> dict:
    """
    1. Compute sector median PEG ratio from peers
    2. Apply: Fair P/E = sector_median_PEG × subject_growth
    3. Apply quality premium/discount (e.g., +15% for best-in-class margins)

    subject_growth_pct: forward EPS growth estimate (annualized %)
    subject_quality_premium: -20 to +30 (percentage adjustment for quality)
    """
    if peer_df.empty or subject_growth_pct <= 0:
        return None

    valid_peg = peer_df.dropna(subset=['peg', 'fwd_pe'])
    valid_peg = valid_peg[(valid_peg['peg'] > 0) & (valid_peg['peg'] < 5)]

    if valid_peg.empty:
        # Fallback: derive PEG from P/E and growth
        valid_pe_g = peer_df.dropna(subset=['fwd_pe', 'earn_growth_pct'])
        valid_pe_g = valid_pe_g[valid_pe_g['earn_growth_pct'] > 0]
        if valid_pe_g.empty:
            return None
        implied_pegs = valid_pe_g['fwd_pe'] / valid_pe_g['earn_growth_pct']
        implied_pegs = implied_pegs[(implied_pegs > 0) & (implied_pegs < 5)]
        median_peg = implied_pegs.median()
    else:
        median_peg = valid_peg['peg'].median()

    raw_fair_pe = median_peg * subject_growth_pct
    adjusted_fair_pe = raw_fair_pe * (1 + subject_quality_premium / 100)

    # Sanity floor/cap
    adjusted_fair_pe = max(min(adjusted_fair_pe, 80), 5)

    return {
        'method': 'PEG-Adjusted',
        'sector_median_peg': round(median_peg, 2),
        'subject_growth_pct': subject_growth_pct,
        'raw_fair_pe': round(raw_fair_pe, 1),
        'quality_adjustment_pct': subject_quality_premium,
        'fair_pe': round(adjusted_fair_pe, 1),
        'n_peers_used': len(valid_peg) if not valid_peg.empty else len(peer_df),
    }


# ─────────────────────────────────────────────
# HISTORICAL P/E BAND (5-year)
# ─────────────────────────────────────────────
def historical_pe_band(ticker_symbol: str, income_stmt: pd.DataFrame) -> dict:
    """
    Reconstruct trailing P/E from historical price and EPS.
    Returns: min, 25th pct, median, 75th pct, max over 5 years.
    """
    try:
        hist = yf.Ticker(ticker_symbol).history(period='5y')['Close']
        if hist.empty:
            return {}

        # Get annual EPS from income statement
        net_income = None
        shares = None
        for label in ['Net Income', 'Net Income Common Stockholders']:
            if label in income_stmt.index:
                net_income = income_stmt.loc[label].dropna()
                break

        if net_income is None or net_income.empty:
            return {}

        # Build quarterly EPS proxy: annual NI / shares from info
        info = yf.Ticker(ticker_symbol).fast_info
        shares_out = getattr(info, 'shares', None) or 1

        # Map annual EPS to daily price history
        eps_by_year = {}
        for date, val in net_income.items():
            yr = date.year
            eps_by_year[yr] = val / shares_out

        pe_series = []
        for date, price in hist.items():
            yr = date.year
            eps = eps_by_year.get(yr) or eps_by_year.get(yr - 1)
            if eps and eps > 0:
                pe_series.append(price / eps)

        if not pe_series:
            return {}

        pe_arr = np.array(pe_series)
        pe_arr = pe_arr[(pe_arr > 5) & (pe_arr < 200)]  # remove noise

        return {
            'min': round(np.percentile(pe_arr, 5), 1),
            'p25': round(np.percentile(pe_arr, 25), 1),
            'median': round(np.median(pe_arr), 1),
            'p75': round(np.percentile(pe_arr, 75), 1),
            'max': round(np.percentile(pe_arr, 95), 1),
            'current': round(pe_arr[-1], 1) if pe_arr.size else None,
            'n_obs': len(pe_arr),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
# FORWARD EPS BUILDER
# Construct 1yr and 2yr forward EPS from available data
# ─────────────────────────────────────────────
def build_forward_eps(info: dict, earnings_estimate: pd.DataFrame) -> dict:
    """
    Priority order:
    1. Analyst consensus (earnings_estimate from yfinance)
    2. Self-constructed: trailing EPS × (1 + earnings_growth)
    3. Forward EPS from info dict
    """
    eps_sources = {}

    # Source 1: yfinance earnings estimate table
    if earnings_estimate is not None and not earnings_estimate.empty:
        try:
            for period in ['0q', '+1q', '0y', '+1y']:
                if period in earnings_estimate.columns:
                    avg = earnings_estimate.loc['avg', period] if 'avg' in earnings_estimate.index else None
                    low = earnings_estimate.loc['low', period] if 'low' in earnings_estimate.index else None
                    high = earnings_estimate.loc['high', period] if 'high' in earnings_estimate.index else None
                    n = earnings_estimate.loc['numberOfAnalysts', period] if 'numberOfAnalysts' in earnings_estimate.index else None
                    if avg:
                        eps_sources[period] = {'avg': avg, 'low': low, 'high': high, 'n': n}
        except Exception:
            pass

    # Source 2: info dict
    fwd_eps = info.get('forwardEps')
    trail_eps = info.get('trailingEps')
    earn_growth = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth') or 0

    result = {
        'eps_ttm': trail_eps,
        'eps_fwd_1y': fwd_eps,
        'eps_fwd_2y': fwd_eps * (1 + earn_growth) if fwd_eps and earn_growth else None,
        'eps_growth_assumed': earn_growth,
        'consensus_available': bool(eps_sources),
        'consensus_detail': eps_sources,
    }

    # If no forward EPS from yfinance, construct from trailing + growth
    if not fwd_eps and trail_eps and earn_growth:
        result['eps_fwd_1y'] = trail_eps * (1 + earn_growth)
        result['eps_fwd_2y'] = trail_eps * (1 + earn_growth) ** 2
        result['eps_source'] = 'constructed'
    elif fwd_eps:
        result['eps_source'] = 'yfinance_info'
    else:
        result['eps_source'] = 'unavailable'

    return result


# ─────────────────────────────────────────────
# FOOTBALL FIELD — Bear/Base/Bull PTs
# ─────────────────────────────────────────────
def football_field(eps_fwd: float, peer_pe_stats: dict, hist_pe: dict,
                   peg_pe: dict, info: dict) -> dict:
    """
    Build 3-scenario price targets using different P/E anchors:
    - Bear: peer P/E 25th pctile or hist P/E 25th pctile (whichever lower)
    - Base: peer median P/E or PEG-adjusted P/E (weighted avg)
    - Bull: peer P/E 75th pctile or hist P/E 75th pctile (whichever higher)
    """
    if not eps_fwd or eps_fwd <= 0:
        return {}

    peer_p25 = peer_pe_stats.get('p25') or 0
    peer_med = peer_pe_stats.get('median') or 0
    peer_p75 = peer_pe_stats.get('p75') or 0

    hist_p25 = hist_pe.get('p25') or peer_p25
    hist_med = hist_pe.get('median') or peer_med
    hist_p75 = hist_pe.get('p75') or peer_p75

    peg_pe_val = (peg_pe or {}).get('fair_pe') or peer_med

    # Scenario multiples
    bear_pe = min(peer_p25, hist_p25) if peer_p25 and hist_p25 else (peer_p25 or hist_p25 or peer_med * 0.7)
    bull_pe = max(peer_p75, hist_p75) if peer_p75 and hist_p75 else (peer_p75 or hist_p75 or peer_med * 1.3)

    # Base: weight peer median 50% + PEG-adjusted 50%
    base_pe = (peer_med * 0.5 + peg_pe_val * 0.5) if peer_med and peg_pe_val else (peer_med or peg_pe_val)

    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0

    def _pt(pe):
        if not pe or pe <= 0:
            return None
        return round(pe * eps_fwd, 2)

    bear_pt = _pt(bear_pe)
    base_pt = _pt(base_pe)
    bull_pt = _pt(bull_pe)

    def _upside(pt):
        if not pt or not current_price:
            return None
        return round((pt / current_price - 1) * 100, 1)

    return {
        'eps_fwd_used': round(eps_fwd, 2),
        'bear': {'pe': round(bear_pe, 1), 'pt': bear_pt, 'upside': _upside(bear_pt)},
        'base': {'pe': round(base_pe, 1), 'pt': base_pt, 'upside': _upside(base_pt)},
        'bull': {'pe': round(bull_pe, 1), 'pt': bull_pt, 'upside': _upside(bull_pt)},
        'peer_pe_range': f"{peer_p25:.0f}x – {peer_p75:.0f}x" if peer_p25 and peer_p75 else 'N/A',
        'hist_pe_range': f"{hist_p25:.0f}x – {hist_p75:.0f}x" if hist_p25 and hist_p75 else 'N/A',
    }


# ─────────────────────────────────────────────
# PEER COMPARISON TABLE (clean summary)
# ─────────────────────────────────────────────
def peer_comparison_table(subject_ticker: str, subject_info: dict,
                          peer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side table: subject vs. each peer on key metrics.
    Highlights where subject trades at premium/discount.
    """
    rows = []

    # Subject row
    rows.append({
        'Ticker': subject_ticker + ' *',
        'Name': (subject_info.get('shortName') or subject_ticker)[:20],
        'Mkt Cap ($B)': round((subject_info.get('marketCap') or 0) / 1e9, 1),
        'Fwd P/E': subject_info.get('forwardPE') or 'N/A',
        'Trail P/E': subject_info.get('trailingPE') or 'N/A',
        'PEG': subject_info.get('pegRatio') or subject_info.get('trailingPegRatio') or 'N/A',
        'Rev Growth %': round((subject_info.get('revenueGrowth') or 0) * 100, 1),
        'EPS Growth %': round((subject_info.get('earningsGrowth') or 0) * 100, 1),
        'Op Margin %': round((subject_info.get('operatingMargins') or 0) * 100, 1),
    })

    if not peer_df.empty:
        for _, row in peer_df.iterrows():
            rows.append({
                'Ticker': row['ticker'],
                'Name': str(row['name'])[:20],
                'Mkt Cap ($B)': round(row.get('market_cap_b', 0), 1),
                'Fwd P/E': round(row.get('fwd_pe', 0), 1) if row.get('fwd_pe') else 'N/A',
                'Trail P/E': round(row.get('trail_pe', 0), 1) if row.get('trail_pe') else 'N/A',
                'PEG': round(row.get('peg', 0), 2) if row.get('peg') else 'N/A',
                'Rev Growth %': round(row.get('rev_growth_pct', 0), 1),
                'EPS Growth %': round(row.get('earn_growth_pct', 0), 1),
                'Op Margin %': round(row.get('op_margin_pct', 0), 1),
            })

        # Sector summary row
        num_cols = ['Fwd P/E', 'Trail P/E', 'PEG', 'Rev Growth %', 'EPS Growth %', 'Op Margin %']
        summary = {'Ticker': 'SECTOR MED', 'Name': '—', 'Mkt Cap ($B)': '—'}
        for col in num_cols:
            vals = [r[col] for r in rows[1:] if isinstance(r[col], (int, float))]
            summary[col] = round(np.median(vals), 1) if vals else 'N/A'
        rows.append(summary)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────
def run_peer_pe_model(data: dict) -> dict:
    info = data['info']
    ticker = data['symbol']
    sector = info.get('sector') or ''
    industry = info.get('industry') or ''

    print(f"  [PE] Fetching peer universe for {ticker}...")
    peer_tickers = _get_peer_list(sector, industry, ticker)
    peer_df = fetch_peer_multiples(peer_tickers[:8])  # cap at 8 peers
    print(f"  [PE] {len(peer_df)} peers with valid data: {list(peer_df['ticker']) if not peer_df.empty else []}")

    # Peer P/E statistics
    peer_pe_stats = {}
    if not peer_df.empty:
        fwd_pes = peer_df['fwd_pe'].dropna()
        fwd_pes = fwd_pes[(fwd_pes > 0) & (fwd_pes < 200)]
        if not fwd_pes.empty:
            peer_pe_stats = {
                'median': round(fwd_pes.median(), 1),
                'mean': round(fwd_pes.mean(), 1),
                'p25': round(fwd_pes.quantile(0.25), 1),
                'p75': round(fwd_pes.quantile(0.75), 1),
                'min': round(fwd_pes.min(), 1),
                'max': round(fwd_pes.max(), 1),
            }

    # Forward EPS
    eps_data = build_forward_eps(info, data.get('earnings_estimate'))
    eps_fwd = eps_data.get('eps_fwd_1y')

    # Quality premium: compare vs sector median margins
    subject_op_margin = (info.get('operatingMargins') or 0) * 100
    peer_median_margin = peer_df['op_margin_pct'].median() if not peer_df.empty else subject_op_margin
    quality_premium = min(max((subject_op_margin - peer_median_margin) * 0.5, -20), 30)

    # PEG-adjusted P/E
    subject_growth = (info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth') or 0) * 100
    if subject_growth <= 0:
        eps_trail = info.get('trailingEps') or 0
        eps_fwd_v = info.get('forwardEps') or 0
        if eps_trail > 0 and eps_fwd_v > eps_trail:
            subject_growth = (eps_fwd_v / eps_trail - 1) * 100
        else:
            subject_growth = 5.0  # fallback

    peg_result = peg_adjusted_pe(peer_df, subject_growth, quality_premium)
    print(f"  [PE] PEG-adjusted fair P/E: {peg_result.get('fair_pe', 'N/A') if peg_result else 'N/A'}x")

    # Historical P/E band
    print(f"  [PE] Computing 5-year historical P/E band...")
    hist_pe = historical_pe_band(ticker, data['income_stmt'])
    if hist_pe:
        print(f"  [PE] Historical P/E range: {hist_pe.get('p25')}x – {hist_pe.get('p75')}x (median {hist_pe.get('median')}x)")

    # Football field
    ff = {}
    if eps_fwd and eps_fwd > 0:
        ff = football_field(eps_fwd, peer_pe_stats, hist_pe, peg_result, info)
        if ff:
            print(f"  [PE] Bear/Base/Bull PT: ${ff['bear']['pt']} / ${ff['base']['pt']} / ${ff['bull']['pt']}")

    # Peer comparison table
    comp_table = peer_comparison_table(ticker, info, peer_df)

    return {
        'peer_df': peer_df,
        'peer_pe_stats': peer_pe_stats,
        'eps_data': eps_data,
        'peg_result': peg_result,
        'hist_pe': hist_pe,
        'football_field': ff,
        'comp_table': comp_table,
        'quality_premium': round(quality_premium, 1),
        'subject_growth_pct': round(subject_growth, 1),
    }
