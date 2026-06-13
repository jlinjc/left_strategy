"""
Consensus-Driven Valuation Model
─────────────────────────────────
Architecture:
  Stage 1 (Y1-Y2): analyst consensus revenue growth → implied FCF
  Stage 1.5 (Y3-Y5): blend toward company's own 3yr avg FCF margin
  Stage 2 (Y6-Y10): linear fade to terminal growth rate
  Terminal: Gordon Growth at 2.5%

Also provides:
  - Full analyst landscape (PT distribution, rec counts, upgrades)
  - Consensus EPS-driven Forward P/E valuation
  - Composite price target blending our model + street consensus anchor
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# FETCH FULL ANALYST LANDSCAPE
# ─────────────────────────────────────────────
def fetch_analyst_landscape(ticker_symbol: str) -> dict:
    t = yf.Ticker(ticker_symbol)

    result = {
        'price_targets': {},
        'earnings_estimate': None,
        'revenue_estimate': None,
        'recommendations_summary': None,
        'upgrades_downgrades': None,
        'eps_revisions': None,
    }

    try:
        result['price_targets'] = t.analyst_price_targets or {}
    except Exception:
        pass

    try:
        ee = t.earnings_estimate
        result['earnings_estimate'] = ee if (ee is not None and not ee.empty) else None
    except Exception:
        pass

    try:
        re = t.revenue_estimate
        result['revenue_estimate'] = re if (re is not None and not re.empty) else None
    except Exception:
        pass

    try:
        rs = t.recommendations_summary
        result['recommendations_summary'] = rs if (rs is not None and not rs.empty) else None
    except Exception:
        pass

    try:
        ud = t.upgrades_downgrades
        result['upgrades_downgrades'] = ud.head(15) if (ud is not None and not ud.empty) else None
    except Exception:
        pass

    try:
        er = t.eps_revisions
        result['eps_revisions'] = er if (er is not None and not er.empty) else None
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────
# EXTRACT CONSENSUS GROWTH RATES
# ─────────────────────────────────────────────
def extract_consensus_growth(analyst_data: dict, info: dict) -> dict:
    """
    Returns consensus revenue growth and EPS for Y1 and Y2.
    Handles currency for foreign ADRs by using EPS-based approach.
    """
    ee = analyst_data.get('earnings_estimate')
    re = analyst_data.get('revenue_estimate')

    result = {
        'rev_growth_y1': None,
        'rev_growth_y2': None,
        'eps_y1': None,
        'eps_y2': None,
        'eps_growth_y1': None,
        'eps_growth_y2': None,
        'n_analysts_eps': None,
        'n_analysts_rev': None,
        'source': 'none',
    }

    # EPS consensus (always USD for US-listed tickers including ADRs)
    if ee is not None:
        try:
            if '0y' in ee.index:
                result['eps_y1'] = float(ee.loc['0y', 'avg']) if 'avg' in ee.columns else None
                result['eps_growth_y1'] = float(ee.loc['0y', 'growth']) if 'growth' in ee.columns else None
                result['n_analysts_eps'] = int(ee.loc['0y', 'numberOfAnalysts']) if 'numberOfAnalysts' in ee.columns else None
            if '+1y' in ee.index:
                result['eps_y2'] = float(ee.loc['+1y', 'avg']) if 'avg' in ee.columns else None
                result['eps_growth_y2'] = float(ee.loc['+1y', 'growth']) if 'growth' in ee.columns else None
        except Exception:
            pass

    # Revenue consensus — detect currency issue for foreign ADRs
    if re is not None:
        try:
            currency = info.get('currency') or 'USD'
            country = info.get('country') or ''
            is_foreign_currency = currency not in ('USD', 'CAD') or country in ('Taiwan', 'Finland', 'Japan', 'Korea')

            if '0y' in re.index and 'growth' in re.columns:
                rev_g1 = float(re.loc['0y', 'growth'])
                result['rev_growth_y1'] = rev_g1

            if '+1y' in re.index and 'growth' in re.columns:
                rev_g2 = float(re.loc['+1y', 'growth'])
                result['rev_growth_y2'] = rev_g2

            # If foreign currency, revenue growth % is still valid (ratio)
            # but absolute revenue numbers cannot be used directly
            result['is_foreign_revenue'] = is_foreign_currency
            result['source'] = 'consensus'
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────
# FCF MARGIN HISTORY
# ─────────────────────────────────────────────
def _compute_fcf_margin(cashflow: pd.DataFrame, income: pd.DataFrame) -> float:
    """Trailing 3-year average FCF / Revenue margin."""
    try:
        ocf = None
        for label in ['Operating Cash Flow', 'Cash From Operating Activities']:
            if label in cashflow.index:
                ocf = cashflow.loc[label]
                break

        capex = None
        for label in ['Capital Expenditure', 'Capital Expenditures']:
            if label in cashflow.index:
                capex = cashflow.loc[label]
                break

        rev = None
        for label in ['Total Revenue', 'Revenue']:
            if label in income.index:
                rev = income.loc[label]
                break

        if ocf is None or rev is None:
            return 0.12  # fallback

        fcf = ocf + (capex if capex is not None else pd.Series(0, index=ocf.index))
        common_idx = fcf.index.intersection(rev.index)[:3]  # last 3 years

        margins = []
        for idx in common_idx:
            r = float(rev[idx])
            f = float(fcf[idx])
            if r > 0:
                margins.append(f / r)

        if margins:
            avg = np.mean(margins)
            return max(min(avg, 0.50), -0.10)  # clamp
        return 0.12
    except Exception:
        return 0.12


# ─────────────────────────────────────────────
# CONSENSUS-DRIVEN DCF
# ─────────────────────────────────────────────
def _fcf_margin_trend(cashflow: pd.DataFrame, income: pd.DataFrame) -> float:
    """
    Average annual change in FCF margin over recent years (pp/yr, as a decimal).
    Positive = margins expanding. Clamped to ±2pp/yr to avoid extrapolating noise.
    """
    try:
        from data_fetcher import _extract_fcf_series, _extract_revenue_series
        fcf = _extract_fcf_series(cashflow)
        rev = _extract_revenue_series(income)
        common = fcf.index.intersection(rev.index)[:4]
        margins = []
        for idx in common:
            r = float(rev[idx])
            if r > 0:
                margins.append(float(fcf[idx]) / r)
        if len(margins) >= 2:
            # margins[0] is latest; compute mean step (latest - oldest)/(n-1)
            chrono = margins[::-1]
            step = (chrono[-1] - chrono[0]) / (len(chrono) - 1)
            return float(max(min(step, 0.02), -0.02))
    except Exception:
        pass
    return 0.0


def consensus_dcf(info: dict, cashflow: pd.DataFrame, income: pd.DataFrame,
                  consensus_growth: dict, wacc: float, shares: int,
                  net_debt: float, base_fcf_data: dict = None,
                  sbc_dilution: float = 0.0,
                  growth_delta: float = 0.0, margin_delta: float = 0.0,
                  scenario: str = 'base', fx_to_usd: float = 1.0) -> dict | None:
    """
    Revenue × margin driven DCF.

    Revenue path:
      Y1-Y2  : analyst consensus growth (revenue preferred, EPS fallback)
      Y3-Y5  : fade from Y2 growth toward historical revenue CAGR
      Y6-Y10 : fade toward terminal growth (2.5%)
      NB: growth is NO LONGER floored at terminal — a fading/declining business
          (cyclical, structurally challenged) can show contracting revenue/FCF.

    Margin path:
      FCF margin starts at the normalized base margin and drifts linearly toward a
      terminal margin = base + (historical trend × horizon), clamped. This captures
      operating leverage / margin compression instead of freezing margin flat.

    Adjustments:
      base_fcf_data : normalized, SBC-deducted base FCF (from data_fetcher).
      sbc_dilution  : annual share-count growth from net stock comp; dilutes per-share.
      growth_delta / margin_delta : scenario shifts (bear/bull) on the driver path.
    """
    # Normalized base FCF (SBC-adjusted) — fall back to raw TTM only if missing.
    # fx_to_usd converts reporting-currency financials (e.g. TWD for the TSM ADR)
    # into the USD price/share basis. Applied to FCF, revenue and net debt alike, so
    # margins (ratios) are unaffected and only the per-share scale is corrected.
    if base_fcf_data and base_fcf_data.get('base_fcf', 0) > 0:
        base_fcf = float(base_fcf_data['base_fcf'])
    else:
        base_fcf = info.get('freeCashflow') or 0
    if base_fcf <= 0:
        return None
    base_fcf *= fx_to_usd

    revenue = info.get('totalRevenue') or 0
    if revenue <= 0:
        from data_fetcher import _extract_revenue_series
        rs = _extract_revenue_series(income)
        revenue = float(rs.iloc[0]) if not rs.empty else 0
    revenue *= fx_to_usd
    net_debt = net_debt * fx_to_usd
    base_margin = (base_fcf / revenue) if revenue > 0 else None

    rev_g1 = consensus_growth.get('rev_growth_y1')
    rev_g2 = consensus_growth.get('rev_growth_y2')
    eps_g1 = consensus_growth.get('eps_growth_y1')
    eps_g2 = consensus_growth.get('eps_growth_y2')

    # Determine Stage 1 growth — consensus revenue > consensus EPS > fallback
    if rev_g1 is not None and -0.5 < rev_g1 < 1.5:
        g_y1, growth_source = rev_g1, 'consensus_revenue'
    elif eps_g1 is not None and -0.5 < eps_g1 < 1.5:
        g_y1, growth_source = eps_g1, 'consensus_eps'
    else:
        g_y1, growth_source = 0.05, 'fallback'

    if rev_g2 is not None and -0.5 < rev_g2 < 1.5:
        g_y2 = rev_g2
    elif eps_g2 is not None and -0.5 < eps_g2 < 1.5:
        g_y2 = eps_g2
    else:
        g_y2 = g_y1 * 0.8

    # Apply scenario shift to growth
    g_y1 += growth_delta
    g_y2 += growth_delta

    # Historical revenue CAGR (company-specific reality check)
    hist_cagr = 0.04
    try:
        from data_fetcher import _extract_revenue_series
        rev_hist = _extract_revenue_series(income)
        if len(rev_hist) >= 3:
            valid = rev_hist.values[::-1]
            valid = valid[valid > 0]
            if len(valid) >= 2:
                hist_cagr = (valid[-1] / valid[0]) ** (1 / (len(valid) - 1)) - 1
                hist_cagr = max(min(hist_cagr, 0.35), -0.15)
    except Exception:
        pass

    terminal_growth = 0.025

    # Medium-term "mature growth" anchor for years 3-5. We blend consensus-implied
    # momentum with trailing reality, but DON'T yank growth all the way to a depressed
    # trailing CAGR (that conflicts with a consensus that says the business is
    # re-accelerating). Floor at ~nominal GDP so durable franchises aren't assumed to
    # grow below the economy mid-cycle; cap so we don't extrapolate hyper-growth.
    mature_g = 0.5 * hist_cagr + 0.5 * max(g_y2, terminal_growth)
    mature_g = max(min(mature_g, 0.10), 0.04)

    # Margin path
    if base_margin is None:
        base_margin = _compute_fcf_margin(cashflow, income)
    margin_trend = _fcf_margin_trend(cashflow, income)
    # Structural drift from the historical trend is DAMPENED and tightly capped —
    # a 2-4yr margin delta is mostly noise and must not drive a 10pp terminal swing.
    structural_drift = max(min(margin_trend * 2.0, 0.03), -0.03)  # ±3pp cap
    terminal_margin = base_margin + structural_drift + margin_delta
    # Absolute sanity band (wider, lets scenario deltas through)
    terminal_margin = max(min(terminal_margin, base_margin + 0.08, 0.50),
                          base_margin - 0.06, 0.0)

    # Build 10-year revenue and FCF projection
    projected_fcfs = []
    projected_revs = []
    growth_path = []
    rev_t = revenue if revenue > 0 else base_fcf / max(base_margin, 0.01)

    for t in range(1, 11):
        if t == 1:
            g = g_y1
        elif t == 2:
            g = g_y2
        elif t <= 5:
            # Years 3-5: fade Y2 growth toward the mature-growth anchor
            blend = (t - 2) / 3
            g = g_y2 * (1 - blend) + mature_g * blend
        else:
            # Years 6-10: fade mature growth toward terminal
            fade = (t - 5) / 5
            g = mature_g * (1 - fade) + terminal_growth * fade
        # NO floor at terminal: allow g below terminal (decline) but cap extreme negatives
        g = max(g, -0.30)
        growth_path.append(g)

        rev_t = rev_t * (1 + g)
        projected_revs.append(rev_t)
        # Linear margin glide from base to terminal over the 10-year horizon
        margin_t = base_margin + (terminal_margin - base_margin) * (t / 10)
        projected_fcfs.append(rev_t * margin_t)

    pv_explicit = sum(f / (1 + wacc) ** i for i, f in enumerate(projected_fcfs, 1))

    if wacc <= terminal_growth:
        wacc = terminal_growth + 0.02
    terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 10

    total_equity = max(pv_explicit + pv_terminal - net_debt, 0)

    # SBC is already expensed inside FCF (base_fcf is SBC-deducted), so we divide by
    # the CURRENT diluted share count — NOT a grown one. Growing shares on top of
    # expensing SBC would double-count the cost (Damodaran). sbc_dilution is kept as
    # an informational metric only.
    diluted_shares = shares
    per_share = total_equity / diluted_shares if diluted_shares > 0 else 0

    tv_pct = pv_terminal / (pv_explicit + pv_terminal) if (pv_explicit + pv_terminal) > 0 else 0

    return {
        'method': 'Consensus-Driven DCF (rev × margin)',
        'scenario': scenario,
        'base_fcf': base_fcf,
        'base_revenue': round(revenue, 0),
        'base_margin': round(base_margin, 4),
        'terminal_margin': round(terminal_margin, 4),
        'margin_trend_pp': round(margin_trend * 100, 2),
        'fcf_margin': round(base_margin, 3),
        'g_y1': round(g_y1, 4),
        'g_y2': round(g_y2, 4),
        'hist_cagr': round(hist_cagr, 4),
        'mature_growth': round(mature_g, 4),
        'terminal_growth': terminal_growth,
        'growth_source': growth_source,
        'growth_path': [round(g, 4) for g in growth_path],
        'projected_fcfs': projected_fcfs,
        'projected_revs': projected_revs,
        'pv_stage12': round(pv_explicit, 0),
        'pv_terminal': round(pv_terminal, 0),
        'total_equity_value': round(total_equity, 0),
        'intrinsic_per_share': round(per_share, 2),
        'tv_pct_of_total': round(tv_pct, 3),
        'stage1_growth': round(g_y1, 4),
        'historical_cagr': round(hist_cagr, 4),
        'sbc_dilution_pct': round(sbc_dilution * 100, 2),
        'diluted_shares': round(diluted_shares, 0),
    }


def consensus_dcf_scenarios(info, cashflow, income, consensus_growth, wacc,
                            shares, net_debt, base_fcf_data, sbc_dilution,
                            fx_to_usd: float = 1.0) -> dict:
    """
    Bear / Base / Bull intrinsic values driven by explicit shifts to the
    fundamental drivers (growth & FCF margin) — NOT by P/E percentiles.

      Bear : growth -4pp/yr, terminal margin -3pp
      Base : as modelled
      Bull : growth +4pp/yr, terminal margin +3pp
    """
    def run(gd, md, name):
        return consensus_dcf(info, cashflow, income, consensus_growth, wacc,
                             shares, net_debt, base_fcf_data, sbc_dilution,
                             growth_delta=gd, margin_delta=md, scenario=name,
                             fx_to_usd=fx_to_usd)

    bear = run(-0.04, -0.03, 'bear')
    base = run(0.0, 0.0, 'base')
    bull = run(+0.04, +0.03, 'bull')
    return {'bear': bear, 'base': base, 'bull': bull}


def dcf_sensitivity_grid(info, cashflow, income, consensus_growth, wacc,
                         shares, net_debt, base_fcf_data, sbc_dilution,
                         fx_to_usd: float = 1.0) -> pd.DataFrame:
    """
    2D sensitivity on the drivers that actually move the value: Stage-1 revenue
    growth (rows) × terminal FCF margin (columns). Each cell is intrinsic $/share,
    computed with the SAME engine as the base case (no separate, drifting formula).
    """
    growth_deltas = [-0.06, -0.03, 0.0, 0.03, 0.06]   # pp/yr shift on growth path
    margin_deltas = [-0.04, -0.02, 0.0, 0.02, 0.04]   # pp shift on terminal margin

    rows, row_labels = [], []
    base = consensus_dcf(info, cashflow, income, consensus_growth, wacc,
                         shares, net_debt, base_fcf_data, sbc_dilution,
                         fx_to_usd=fx_to_usd)
    g_base = base['g_y1'] if base else 0.0
    m_base = base['terminal_margin'] if base else 0.0

    for gd in growth_deltas:
        row = []
        for md in margin_deltas:
            r = consensus_dcf(info, cashflow, income, consensus_growth, wacc,
                              shares, net_debt, base_fcf_data, sbc_dilution,
                              growth_delta=gd, margin_delta=md, fx_to_usd=fx_to_usd)
            row.append(round(r['intrinsic_per_share'], 2) if r else None)
        rows.append(row)
        row_labels.append(f"{(g_base + gd)*100:.1f}%")

    col_labels = [f"{(m_base + md)*100:.1f}%" for md in margin_deltas]
    return pd.DataFrame(rows, index=row_labels, columns=col_labels)


# ─────────────────────────────────────────────
# CONSENSUS EPS FORWARD P/E VALUATION
# Use consensus EPS (not trailing) × fair multiple
# ─────────────────────────────────────────────
def consensus_pe_valuation(consensus_growth: dict, info: dict,
                           sector_multiples: dict) -> dict | None:
    """
    Simple and direct: Fair P/E × Consensus Forward EPS = PT
    Fair P/E derived from: sector median + PEG adjustment
    """
    eps_y1 = consensus_growth.get('eps_y1')
    eps_y2 = consensus_growth.get('eps_y2')
    eps_g1 = consensus_growth.get('eps_growth_y1') or 0
    eps_g2 = consensus_growth.get('eps_growth_y2') or 0

    if not eps_y1 or eps_y1 <= 0:
        return None

    sector_pe = sector_multiples.get('pe') or 20

    # Forward EPS growth (%), averaged over the two estimate years where available
    g1 = eps_g1 * 100 if eps_g1 else 0
    g2 = eps_g2 * 100 if eps_g2 else 0
    growth_pct = np.mean([g for g in (g1, g2) if g]) if (g1 or g2) else 10.0
    growth_pct = max(growth_pct, 0)

    # ── Single, principled fair-P/E rule: PEG anchoring ──
    # Fair P/E = FAIR_PEG × growth%, where FAIR_PEG ~ 1.5 reflects what the market
    # pays per point of durable EPS growth. We then blend 50/50 with the sector
    # median as a mean-reversion anchor, and clamp. No tiered magic numbers.
    FAIR_PEG = 1.5
    peg_based_pe = FAIR_PEG * growth_pct if growth_pct > 0 else sector_pe * 0.7
    fair_pe = 0.5 * sector_pe + 0.5 * peg_based_pe
    fair_pe = max(min(fair_pe, sector_pe * 2.5, 80), max(sector_pe * 0.5, 8))

    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    current_pe = (current_price / eps_y1) if eps_y1 > 0 else None

    return {
        'method': 'Consensus Forward P/E (PEG-anchored)',
        'eps_y1': round(eps_y1, 2),
        'eps_y2': round(eps_y2, 2) if eps_y2 else None,
        'eps_growth_y1': round(eps_g1 * 100, 1),
        'fair_peg': FAIR_PEG,
        'growth_used_pct': round(growth_pct, 1),
        'fair_pe': round(fair_pe, 1),
        'sector_median_pe': sector_pe,
        'implied_price': round(fair_pe * eps_y1, 2),
        'current_pe': round(current_pe, 1) if current_pe else None,
        'n_analysts': consensus_growth.get('n_analysts_eps'),
    }


# ─────────────────────────────────────────────
# RECOMMENDATION DISTRIBUTION
# ─────────────────────────────────────────────
def analyst_rec_summary(analyst_data: dict) -> dict:
    rs = analyst_data.get('recommendations_summary')
    if rs is None or rs.empty:
        return {}

    try:
        latest = rs.iloc[0]
        sb = int(latest.get('strongBuy', 0))
        b = int(latest.get('buy', 0))
        h = int(latest.get('hold', 0))
        s = int(latest.get('sell', 0))
        ss = int(latest.get('strongSell', 0))
        total = sb + b + h + s + ss

        if total == 0:
            return {}

        bull_pct = (sb + b) / total * 100
        bear_pct = (s + ss) / total * 100

        street_view = 'Bullish' if bull_pct > 60 else 'Neutral' if bull_pct > 40 else 'Bearish'

        return {
            'strong_buy': sb, 'buy': b, 'hold': h,
            'sell': s, 'strong_sell': ss, 'total': total,
            'bull_pct': round(bull_pct, 1),
            'bear_pct': round(bear_pct, 1),
            'street_view': street_view,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
# RECENT UPGRADES / DOWNGRADES SUMMARY
# ─────────────────────────────────────────────
def parse_upgrades(analyst_data: dict) -> list[dict]:
    ud = analyst_data.get('upgrades_downgrades')
    if ud is None or ud.empty:
        return []

    rows = []
    for date, row in ud.iterrows():
        try:
            action = str(row.get('Action', '')).strip()
            to_grade = str(row.get('ToGrade', '')).strip()
            from_grade = str(row.get('FromGrade', '')).strip()
            firm = str(row.get('Firm', '')).strip()
            new_pt = row.get('priceTarget') if hasattr(row, 'get') else None
            rows.append({
                'date': str(date)[:10],
                'firm': firm[:22],
                'action': action,
                'to': to_grade,
                'from': from_grade,
                'new_pt': round(float(new_pt), 2) if new_pt and float(new_pt) > 0 else None,
            })
        except Exception:
            continue
    return rows[:12]


# ─────────────────────────────────────────────
# COMPOSITE PRICE TARGET
# Blend our model with street consensus
# ─────────────────────────────────────────────
def composite_price_target(our_dcf_pt: float, our_comps_pt: float,
                            consensus_pe_pt: float, street_mean_pt: float,
                            current_price: float, has_earnings: bool) -> dict:
    """
    INDEPENDENT price target — our own view, deliberately NOT blended with the
    street mean PT. Anchoring to consensus would (a) double-count consensus, since
    our DCF/PE already run off consensus estimates, and (b) erase the variant view
    that is the whole point of buy-side work. The street mean is carried alongside
    (composite['street_mean']) and the gap is explained in the Consensus Bridge,
    rather than being averaged into our number.

    Weights (our methods only):
      Consensus-driven DCF : 45%
      Consensus Fwd P/E    : 30%
      Sector/Peer Comps    : 25%
    If company has no earnings, drop DCF and re-weight onto comps + fwd P/E.
    """
    components = {}

    if has_earnings and our_dcf_pt and our_dcf_pt > 0:
        components['consensus_dcf'] = (our_dcf_pt, 0.45)
    if consensus_pe_pt and consensus_pe_pt > 0:
        components['consensus_fwd_pe'] = (consensus_pe_pt, 0.30)
    if our_comps_pt and our_comps_pt > 0:
        components['sector_comps'] = (our_comps_pt, 0.25)

    def _street_block(pt_val):
        spread = None
        if street_mean_pt and street_mean_pt > 0 and pt_val:
            spread = round((pt_val / street_mean_pt - 1) * 100, 1)
        return {
            'street_mean': round(street_mean_pt, 2) if street_mean_pt else None,
            'spread_vs_street_pct': spread,
            'stance': (None if spread is None else
                       'ABOVE street' if spread > 5 else
                       'BELOW street' if spread < -5 else 'IN LINE with street'),
        }

    if not components:
        out = {
            'price_target': current_price,
            'upside': 0.0,
            'recommendation': 'HOLD',
            'components': {},
        }
        out.update(_street_block(current_price))
        return out

    # Normalize weights (in case some components missing)
    total_w = sum(w for _, w in components.values())
    norm = {k: (v, w / total_w) for k, (v, w) in components.items()}

    # Sanity filter: exclude any single component > 5x or < 0.1x current price
    if current_price > 0:
        sane = {k: (v, w) for k, (v, w) in norm.items()
                if 0.1 * current_price <= v <= 5.0 * current_price}
        if sane:
            tw2 = sum(w for _, w in sane.values())
            norm = {k: (v, w / tw2) for k, (v, w) in sane.items()}

    pt = sum(v * w for _, (v, w) in norm.items())
    upside = (pt / current_price - 1) * 100 if current_price > 0 else 0

    rec = 'BUY' if upside > 15 else 'HOLD' if upside > -5 else 'SELL'

    out = {
        'price_target': round(pt, 2),
        'upside': round(upside, 1),
        'recommendation': rec,
        'components': {k: {'value': round(v, 2), 'weight': round(w, 2)}
                       for k, (v, w) in norm.items()},
    }
    out.update(_street_block(pt))
    return out


# ─────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────
def run_consensus_model(data: dict, wacc_data: dict,
                        sector_multiples: dict,
                        live_peer_multiples: dict = None) -> dict:
    info = data['info']
    ticker = data['symbol']

    print(f"  [cns] Fetching analyst landscape for {ticker}...")
    analyst_data = fetch_analyst_landscape(ticker)

    # Consensus growth extraction
    consensus_growth = extract_consensus_growth(analyst_data, info)
    g1 = consensus_growth.get('rev_growth_y1') or consensus_growth.get('eps_growth_y1') or 0
    print(f"  [cns] Consensus Y1/Y2 growth: "
          f"{g1*100:.1f}% / "
          f"{(consensus_growth.get('rev_growth_y2') or consensus_growth.get('eps_growth_y2') or 0)*100:.1f}%"
          f"  [{consensus_growth.get('source')}]")

    # Consensus-driven DCF
    shares = info.get('sharesOutstanding') or 1
    total_debt = info.get('totalDebt') or 0
    cash = info.get('totalCash') or 0
    net_debt = total_debt - cash

    # Normalized, SBC-adjusted base FCF + share dilution from net stock comp
    from data_fetcher import normalized_base_fcf, estimate_sbc_dilution, get_fx_rate
    base_fcf_data = normalized_base_fcf(
        data['cashflow'], data['income_stmt'], info.get('freeCashflow') or 0)
    sbc_dilution = estimate_sbc_dilution(data['cashflow'], info)

    # FX: reporting currency (financials) → price currency. Critical for ADRs whose
    # statements are in a foreign currency (e.g. TSM files TWD, trades in USD).
    report_ccy = info.get('financialCurrency') or 'USD'
    price_ccy = info.get('currency') or 'USD'
    fx_to_usd = get_fx_rate(report_ccy, price_ccy) if report_ccy != price_ccy else 1.0
    if fx_to_usd != 1.0:
        print(f"  [cns] FX {report_ccy}->{price_ccy}: {fx_to_usd:.5f} "
              f"(converting financials to price basis)")
    data['fx_to_usd'] = fx_to_usd

    if base_fcf_data.get('sbc_pct_of_fcf'):
        print(f"  [cns] Normalized base FCF ${base_fcf_data['base_fcf']*fx_to_usd/1e9:.1f}B "
              f"(SBC drag {base_fcf_data['sbc_pct_of_fcf']*100:.0f}% of FCF, "
              f"dilution {sbc_dilution*100:.1f}%/yr)")

    dcf_result = consensus_dcf(
        info, data['cashflow'], data['income_stmt'],
        consensus_growth, wacc_data['wacc'], shares, net_debt,
        base_fcf_data=base_fcf_data, sbc_dilution=sbc_dilution, fx_to_usd=fx_to_usd)

    # Bear / Base / Bull intrinsic value driven by growth & margin
    dcf_scenarios = consensus_dcf_scenarios(
        info, data['cashflow'], data['income_stmt'],
        consensus_growth, wacc_data['wacc'], shares, net_debt,
        base_fcf_data, sbc_dilution, fx_to_usd=fx_to_usd)

    if dcf_result:
        print(f"  [cns] Consensus DCF: ${dcf_result['intrinsic_per_share']:.2f}/share"
              f"  (margin {dcf_result['base_margin']*100:.1f}%→{dcf_result['terminal_margin']*100:.1f}%)")
        if dcf_scenarios.get('bear') and dcf_scenarios.get('bull'):
            print(f"  [cns] Scenarios  Bear ${dcf_scenarios['bear']['intrinsic_per_share']:.2f}"
                  f" / Base ${dcf_scenarios['base']['intrinsic_per_share']:.2f}"
                  f" / Bull ${dcf_scenarios['bull']['intrinsic_per_share']:.2f}")

    # Consensus Forward P/E valuation
    eps_pe_result = consensus_pe_valuation(consensus_growth, info, sector_multiples)
    if eps_pe_result:
        print(f"  [cns] Consensus Fwd P/E: {eps_pe_result['fair_pe']:.1f}x × ${eps_pe_result['eps_y1']:.2f} = ${eps_pe_result['implied_price']:.2f}")

    # Comps — prefer live peer multiples over the stale sector table
    from valuation_engine import comps_valuation
    comps = comps_valuation(info, sector_multiples, live_peer_multiples=live_peer_multiples,
                            fx_to_usd=fx_to_usd)
    comps_ev_ebitda_pt = (comps.get('ev_ebitda') or {}).get('implied_price')
    comps_pe_pt = (comps.get('pe') or {}).get('implied_price')
    comps_avg = None
    comp_pts = [v for v in [comps_ev_ebitda_pt, comps_pe_pt] if v and v > 0]
    if comp_pts:
        comps_avg = np.mean(comp_pts)

    # Street consensus mean PT
    street_pt = (analyst_data.get('price_targets') or {}).get('mean')

    # Recommendation distribution
    rec_summary = analyst_rec_summary(analyst_data)
    if rec_summary:
        print(f"  [cns] Street: {rec_summary.get('street_view')} "
              f"({rec_summary.get('bull_pct')}% bull, {rec_summary.get('total')} analysts)")

    # Recent upgrades/downgrades
    upgrades = parse_upgrades(analyst_data)

    # EPS revisions (positive = analysts raising estimates = bullish signal)
    eps_revision_signal = _eps_revision_signal(analyst_data)

    current_price = (info.get('currentPrice') or info.get('regularMarketPrice')
                     or info.get('previousClose') or 0)
    has_positive_earnings = (info.get('trailingEps') or 0) > 0

    # Composite PT
    composite_pt = composite_price_target(
        our_dcf_pt=dcf_result['intrinsic_per_share'] if dcf_result else None,
        our_comps_pt=comps_avg,
        consensus_pe_pt=eps_pe_result['implied_price'] if eps_pe_result else None,
        street_mean_pt=street_pt,
        current_price=current_price,
        has_earnings=has_positive_earnings,
    )

    print(f"  [cns] Composite PT: ${composite_pt['price_target']:.2f} "
          f"({composite_pt['upside']:+.1f}%) → {composite_pt['recommendation']}")

    return {
        'analyst_data': analyst_data,
        'consensus_growth': consensus_growth,
        'consensus_dcf': dcf_result,
        'consensus_dcf_scenarios': dcf_scenarios,
        'base_fcf_data': base_fcf_data,
        'sbc_dilution': sbc_dilution,
        'consensus_pe': eps_pe_result,
        'comps': comps,
        'rec_summary': rec_summary,
        'upgrades': upgrades,
        'eps_revision_signal': eps_revision_signal,
        'composite_pt': composite_pt,
        'street_pt': street_pt,
    }


def _eps_revision_signal(analyst_data: dict) -> str:
    """Detect if analysts are revising EPS up or down recently."""
    er = analyst_data.get('eps_revisions')
    if er is None or er.empty:
        return 'No data'
    try:
        # Look at current quarter: upLast7days vs downLast7days
        if '0q' in er.index:
            row = er.loc['0q']
            up = int(row.get('upLast7days', 0) or 0)
            down = int(row.get('downLast7days', 0) or 0)
            if up > down * 1.5:
                return f'Positive ({up} up vs {down} down, last 7d)'
            elif down > up * 1.5:
                return f'Negative ({down} down vs {up} up, last 7d)'
            return f'Neutral ({up} up / {down} down, last 7d)'
    except Exception:
        pass
    return 'No data'
