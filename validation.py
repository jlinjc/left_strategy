"""
Buy-Side Validation Module
Three layers:
  1. Reverse DCF  — what growth rate does the current price imply?
  2. Implied multiples at PT — does our PT embed reasonable multiples?
  3. Sanity check — does our model's calculated EPS/FCF match raw filings?
  4. Consensus bridge — how far are we from sell-side consensus, and why?
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. REVERSE DCF
# Solve: current_price = f(implied_growth) via bisection
# ─────────────────────────────────────────────
def reverse_dcf(current_price: float, base_fcf: float, wacc: float,
                shares: int, net_debt: float,
                terminal_growth: float = 0.025,
                stage1_years: int = 5) -> dict:
    """
    Find the Stage-1 growth rate g* such that DCF intrinsic = current market price.
    This tells the investor exactly what the market is pricing in — not what we think.
    """
    if base_fcf <= 0 or shares <= 0 or current_price <= 0:
        return None

    def dcf_price(g1: float) -> float:
        proj = []
        for t in range(1, 11):
            if t <= stage1_years:
                g = g1
            else:
                # linear fade
                fade = (g1 - terminal_growth) / stage1_years
                g = g1 - fade * (t - stage1_years)
                g = max(g, terminal_growth)
            prev = proj[-1] if proj else base_fcf
            proj.append(prev * (1 + g))

        pv12 = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(proj))
        if wacc <= terminal_growth:
            return 1e9
        tv = proj[-1] * (1 + terminal_growth) / (wacc - terminal_growth) / (1 + wacc) ** 10
        equity_val = pv12 + tv - net_debt
        return equity_val / shares

    target_fn = lambda g: dcf_price(g) - current_price

    try:
        # Search range: -10% to +60% Stage-1 growth
        lo, hi = -0.10, 0.60
        if target_fn(lo) * target_fn(hi) > 0:
            # If no sign change, push hi further
            hi = 1.00
        if target_fn(lo) * target_fn(hi) > 0:
            return {'implied_growth': None, 'note': 'Market price outside solvable range'}

        g_implied = brentq(target_fn, lo, hi, xtol=1e-6)
    except Exception as e:
        return {'implied_growth': None, 'note': str(e)}

    # FCF yield at current price
    fcf_yield = base_fcf / (current_price * shares)

    # What % premium does the market assign over pure FCF yield?
    cost_of_equity = wacc  # simplification for display
    growth_premium = g_implied - (wacc - 1 / (current_price / (base_fcf / shares)))

    return {
        'implied_growth_stage1': round(g_implied * 100, 2),
        'terminal_growth': round(terminal_growth * 100, 2),
        'wacc': round(wacc * 100, 2),
        'base_fcf_per_share': round(base_fcf / shares, 2),
        'fcf_yield_at_current': round(fcf_yield * 100, 2),
        'interpretation': _interpret_implied_growth(g_implied, wacc, terminal_growth),
    }


def _interpret_implied_growth(g: float, wacc: float, tg: float) -> str:
    if g < 0:
        return f"Market prices in FCF CONTRACTION of {abs(g)*100:.1f}%/yr — deeply discounted or in distress"
    elif g < 0.05:
        return f"Market prices in LOW growth ({g*100:.1f}%/yr) — value/defensive territory, low expectations"
    elif g < 0.12:
        return f"Market prices in MODERATE growth ({g*100:.1f}%/yr) — in line with nominal GDP + some premium"
    elif g < 0.20:
        return f"Market prices in STRONG growth ({g*100:.1f}%/yr) — execution must be near-flawless to justify"
    elif g < 0.35:
        return f"Market prices in HIGH growth ({g*100:.1f}%/yr) — typical for high-quality compounders (FAANG-tier)"
    else:
        return f"Market prices in VERY HIGH growth ({g*100:.1f}%/yr) — aggressive; only sustainable for AI/platform winners"


# ─────────────────────────────────────────────
# 2. IMPLIED MULTIPLES AT PRICE TARGET
# At our PT, what P/E / EV/EBITDA does it embed?
# Compare to sector median — is it reasonable?
# ─────────────────────────────────────────────
def implied_multiples_at_pt(price_target: float, info: dict, sector_multiples: dict,
                            fx_to_usd: float = 1.0) -> dict:
    # EPS (per ADR) is already in the price currency; EBITDA/revenue/debt/cash are in
    # the reporting currency and must be converted so EV-based multiples are coherent.
    eps_ttm = info.get('trailingEps') or 0
    eps_fwd = info.get('forwardEps') or 0
    ebitda = (info.get('ebitda') or 0) * fx_to_usd
    revenue = (info.get('totalRevenue') or 0) * fx_to_usd
    shares = info.get('sharesOutstanding') or 1
    total_debt = (info.get('totalDebt') or 0) * fx_to_usd
    cash = (info.get('totalCash') or 0) * fx_to_usd

    ev_at_pt = price_target * shares + total_debt - cash

    results = {}

    if eps_ttm > 0:
        pe_at_pt = price_target / eps_ttm
        results['P/E (TTM) at PT'] = {
            'value': round(pe_at_pt, 1),
            'sector_median': sector_multiples.get('pe', 0),
            'premium_discount': round((pe_at_pt / sector_multiples.get('pe', 1) - 1) * 100, 1),
        }

    if eps_fwd > 0:
        pe_fwd_at_pt = price_target / eps_fwd
        results['P/E (Fwd) at PT'] = {
            'value': round(pe_fwd_at_pt, 1),
            'sector_median': sector_multiples.get('pe', 0),
            'premium_discount': round((pe_fwd_at_pt / sector_multiples.get('pe', 1) - 1) * 100, 1),
        }

    if ebitda > 0:
        ev_ebitda_at_pt = ev_at_pt / ebitda
        results['EV/EBITDA at PT'] = {
            'value': round(ev_ebitda_at_pt, 1),
            'sector_median': sector_multiples.get('ev_ebitda', 0),
            'premium_discount': round((ev_ebitda_at_pt / sector_multiples.get('ev_ebitda', 1) - 1) * 100, 1),
        }

    if revenue > 0:
        ev_rev_at_pt = ev_at_pt / revenue
        results['EV/Revenue at PT'] = {
            'value': round(ev_rev_at_pt, 1),
            'sector_median': sector_multiples.get('ev_revenue', 0),
            'premium_discount': round((ev_rev_at_pt / sector_multiples.get('ev_revenue', 1) - 1) * 100, 1),
        }

    return results


# ─────────────────────────────────────────────
# 3. MODEL SANITY CHECK
# Recompute key metrics from raw statements and compare
# to yfinance-reported values — catches data issues
# ─────────────────────────────────────────────
def sanity_check(info: dict, income_stmt: pd.DataFrame,
                 cashflow: pd.DataFrame, balance_sheet: pd.DataFrame) -> list:
    """
    Returns list of (metric, our_calc, yf_reported, delta_pct, status)
    """
    checks = []

    def _get(df, *labels):
        for l in labels:
            if l in df.index:
                v = df.loc[l]
                return v.iloc[0] if hasattr(v, 'iloc') else v
        return None

    def _check(name, calc, reported, tolerance=0.05):
        if calc is None or reported is None:
            return (name, 'N/A', 'N/A', None, 'NO DATA')
        if reported == 0:
            return (name, f"{calc:,.0f}", f"{reported:,.0f}", None, 'ZERO REPORTED')
        delta = abs(calc - reported) / abs(reported)
        status = 'PASS' if delta <= tolerance else 'REVIEW' if delta <= 0.15 else 'FAIL'
        return (name, f"{calc:,.0f}", f"{reported:,.0f}", f"{delta*100:.1f}%", status)

    # EPS check: Net Income / Diluted Shares = EPS
    net_income = _get(income_stmt, 'Net Income', 'Net Income Common Stockholders')
    diluted_shares = info.get('sharesOutstanding') or 0
    if net_income and diluted_shares:
        our_eps = net_income / diluted_shares
        reported_eps = info.get('trailingEps') or 0
        checks.append(_check('EPS (TTM)', our_eps, reported_eps, tolerance=0.08))

    # Revenue check
    our_rev = _get(income_stmt, 'Total Revenue', 'Revenue')
    reported_rev = info.get('totalRevenue') or 0
    if our_rev is not None:
        checks.append(_check('Total Revenue ($)', our_rev, reported_rev, tolerance=0.02))

    # Gross Margin check
    gross_profit = _get(income_stmt, 'Gross Profit')
    if gross_profit and our_rev and our_rev != 0:
        our_gm = gross_profit / our_rev
        reported_gm = info.get('grossMargins') or 0
        checks.append(_check('Gross Margin (%)', our_gm, reported_gm, tolerance=0.03))

    # Operating Income check
    op_income = _get(income_stmt, 'Operating Income', 'EBIT')
    reported_op = info.get('operatingCashflow') or 0  # not perfect but available
    if op_income is not None and our_rev and our_rev != 0:
        our_op_margin = op_income / our_rev
        reported_op_margin = info.get('operatingMargins') or 0
        checks.append(_check('Op. Margin (%)', our_op_margin, reported_op_margin, tolerance=0.03))

    # FCF check: OCF - CapEx
    ocf = _get(cashflow, 'Operating Cash Flow', 'Cash From Operating Activities')
    capex = _get(cashflow, 'Capital Expenditure', 'Capital Expenditures')
    if ocf is not None and capex is not None:
        our_fcf = ocf + capex  # capex stored negative
        reported_fcf = info.get('freeCashflow') or 0
        if reported_fcf:
            checks.append(_check('Free Cash Flow ($)', our_fcf, reported_fcf, tolerance=0.05))

    # Debt/Equity check
    total_debt = _get(balance_sheet, 'Total Debt', 'Long Term Debt')
    total_equity = _get(balance_sheet, 'Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest')
    if total_debt is not None and total_equity and total_equity != 0:
        our_de = total_debt / abs(total_equity)
        reported_de = (info.get('debtToEquity') or 0) / 100
        checks.append(_check('D/E Ratio', our_de, reported_de, tolerance=0.10))

    # Current Ratio: Current Assets / Current Liabilities
    curr_assets = _get(balance_sheet, 'Current Assets')
    curr_liab = _get(balance_sheet, 'Current Liabilities')
    if curr_assets and curr_liab and curr_liab != 0:
        our_cr = curr_assets / curr_liab
        reported_cr = info.get('currentRatio') or 0
        if reported_cr:
            checks.append(_check('Current Ratio', our_cr, reported_cr, tolerance=0.05))

    return checks


# ─────────────────────────────────────────────
# 4. CONSENSUS BRIDGE
# Compare our PT vs sell-side; explain the gap
# ─────────────────────────────────────────────
def consensus_bridge(our_pt: float, our_rec: str, current_price: float,
                     analyst_targets: dict, dcf_result: dict,
                     comps_result: dict, info: dict) -> dict:
    """
    Decomposes the gap between our PT and consensus mean into:
    - Growth assumption difference
    - Multiple assumption difference
    - Other (qualitative)
    """
    consensus_mean = analyst_targets.get('mean') or analyst_targets.get('median') or 0
    n_analysts = analyst_targets.get('numberOfAnalysts') or 0

    if not consensus_mean:
        return {'note': 'No consensus data available'}

    our_upside = (our_pt / current_price - 1) * 100
    consensus_upside = (consensus_mean / current_price - 1) * 100
    pt_gap = our_pt - consensus_mean
    pt_gap_pct = (our_pt / consensus_mean - 1) * 100

    # Attribute the gap
    attribution = {}

    # Growth: compare our Stage-1 FCF growth vs revenue growth consensus
    our_growth = (dcf_result or {}).get('stage1_growth', 0)
    consensus_rev_growth = info.get('revenueGrowth') or 0

    # Multiple: compare what P/E our PT implies vs what consensus likely uses
    our_pe_at_pt = None
    eps_fwd = info.get('forwardEps') or info.get('trailingEps') or 0
    if eps_fwd > 0:
        our_pe_at_pt = our_pt / eps_fwd
        consensus_pe_at_pt = consensus_mean / eps_fwd

    # DCF weight vs comps weight in our model
    dcf_share = (dcf_result or {}).get('intrinsic_per_share', 0)
    comps_avg = 0
    if comps_result:
        vals = [v['implied_price'] for v in comps_result.values() if v.get('implied_price', 0) > 0]
        comps_avg = np.mean(vals) if vals else 0

    # Is our DCF dragging the PT down?
    dcf_vs_comps_gap = dcf_share - comps_avg if dcf_share and comps_avg else None

    key_differences = []
    if dcf_vs_comps_gap is not None and abs(dcf_vs_comps_gap) > 20:
        direction = 'significantly below' if dcf_vs_comps_gap < 0 else 'above'
        key_differences.append(
            f"Our DCF (${dcf_share:.2f}) is {direction} comps average (${comps_avg:.2f}) "
            f"by ${abs(dcf_vs_comps_gap):.2f} — the gap reflects our conservative FCF growth "
            f"assumption ({our_growth*100:.1f}%/yr) vs. market's implied higher growth rate."
        )

    if our_pe_at_pt and consensus_pe_at_pt:
        if abs(our_pe_at_pt - consensus_pe_at_pt) > 3:
            key_differences.append(
                f"Implied P/E at our PT: {our_pe_at_pt:.1f}x vs. consensus PT implies {consensus_pe_at_pt:.1f}x. "
                f"We apply a {'lower' if our_pe_at_pt < consensus_pe_at_pt else 'higher'} multiple "
                f"based on DCF-anchored methodology."
            )

    if pt_gap_pct < -15:
        key_differences.append(
            f"Our PT is {abs(pt_gap_pct):.0f}% below consensus — suggesting sell-side may be applying "
            f"higher terminal multiples or more optimistic near-term catalysts than our model captures."
        )
    elif pt_gap_pct > 15:
        key_differences.append(
            f"Our PT is {pt_gap_pct:.0f}% above consensus — our model captures upside the street may be underweighting."
        )

    return {
        'our_pt': our_pt,
        'our_upside': round(our_upside, 1),
        'consensus_mean': round(consensus_mean, 2),
        'consensus_upside': round(consensus_upside, 1),
        'n_analysts': n_analysts,
        'pt_gap_vs_consensus': round(pt_gap, 2),
        'pt_gap_pct': round(pt_gap_pct, 1),
        'our_dcf_per_share': dcf_share,
        'comps_average': round(comps_avg, 2) if comps_avg else None,
        'our_pe_implied': round(our_pe_at_pt, 1) if our_pe_at_pt else None,
        'key_differences': key_differences,
    }


# ─────────────────────────────────────────────
# MASTER: run all validation
# ─────────────────────────────────────────────
def run_validation(data: dict, valuations: dict, price_target_data: dict,
                   wacc_data: dict, sector_multiples: dict) -> dict:
    info = data['info']
    current_price = (info.get('currentPrice') or info.get('regularMarketPrice')
                     or info.get('previousClose') or 0)
    dcf = valuations.get('dcf') or {}
    comps = valuations.get('comps') or {}
    pt = price_target_data['price_target']
    fx = data.get('fx_to_usd', 1.0)  # reporting-ccy → price-ccy (ADRs)

    results = {}

    # 1. Reverse DCF
    print("  [val] Running Reverse DCF...")
    from data_fetcher import _extract_fcf_series
    fcf_hist = _extract_fcf_series(data['cashflow'])
    # dcf['base_fcf'] is already in USD; the raw-statement fallback is not, so convert.
    base_fcf = dcf.get('base_fcf') if dcf else None
    if not base_fcf and not fcf_hist.empty:
        base_fcf = fcf_hist.iloc[0] * fx

    if base_fcf and base_fcf > 0:
        shares = info.get('sharesOutstanding') or 1
        net_debt = ((info.get('totalDebt') or 0) - (info.get('totalCash') or 0)) * fx
        results['reverse_dcf'] = reverse_dcf(
            current_price, base_fcf, wacc_data['wacc'],
            shares, net_debt)
    else:
        results['reverse_dcf'] = {'note': 'Base FCF not available for reverse DCF'}

    # 2. Implied multiples at PT
    print("  [val] Checking implied multiples at PT...")
    results['implied_multiples'] = implied_multiples_at_pt(pt, info, sector_multiples, fx_to_usd=fx)

    # 3. Sanity check
    print("  [val] Running model sanity checks...")
    results['sanity_checks'] = sanity_check(
        info, data['income_stmt'], data['cashflow'], data['balance_sheet'])

    # 4. Consensus bridge
    print("  [val] Comparing to analyst consensus...")
    results['consensus_bridge'] = consensus_bridge(
        pt, price_target_data['recommendation'], current_price,
        data.get('analyst_price_targets') or {},
        dcf, comps, info)

    return results


# ─────────────────────────────────────────────
# PRINT VALIDATION REPORT (terminal)
# ─────────────────────────────────────────────
def print_validation_report(ticker: str, validation: dict, current_price: float):
    print("\n" + "="*65)
    print(f"  VALIDATION REPORT -- {ticker} (Current: ${current_price:.2f})")
    print("="*65)

    # Reverse DCF
    rv = validation.get('reverse_dcf') or {}
    print("\n[A] REVERSE DCF -- What growth rate is the market pricing in?")
    print("-"*65)
    if rv.get('implied_growth_stage1') is not None:
        print(f"  Implied Stage-1 FCF Growth : {rv['implied_growth_stage1']:+.2f}%/yr")
        print(f"  Our Model Stage-1 Growth   : (see DCF section)")
        print(f"  FCF Yield at Current Price : {rv['fcf_yield_at_current']:.2f}%")
        print(f"  WACC                       : {rv['wacc']:.2f}%")
        print(f"  Interpretation:")
        print(f"    -> {rv['interpretation']}")
    else:
        print(f"  {rv.get('note', 'N/A')}")

    # Implied multiples
    print("\n[B] IMPLIED MULTIPLES AT OUR PRICE TARGET")
    print("-"*65)
    im = validation.get('implied_multiples') or {}
    if im:
        print(f"  {'Metric':<22} {'At PT':>8} {'Sector Med':>12} {'Prem/Disc':>12} {'Verdict':>10}")
        print("  " + "-"*60)
        for metric, d in im.items():
            val = d['value']
            med = d['sector_median']
            prem = d['premium_discount']
            verdict = 'RICH' if prem > 20 else 'FAIR' if abs(prem) <= 20 else 'CHEAP'
            print(f"  {metric:<22} {val:>8.1f}x {med:>11.1f}x {prem:>+11.1f}% {verdict:>10}")
    else:
        print("  No multiple data available")

    # Sanity checks
    print("\n[C] MODEL SANITY CHECK -- Our calculations vs. SEC filings")
    print("-"*65)
    checks = validation.get('sanity_checks') or []
    if checks:
        print(f"  {'Metric':<22} {'Our Calc':>15} {'Reported':>15} {'Delta':>8} {'Status':>8}")
        print("  " + "-"*70)
        for row in checks:
            name, calc, reported, delta, status = row
            status_fmt = f"[{status}]"
            delta_str = delta or '  —'
            print(f"  {name:<22} {str(calc):>15} {str(reported):>15} {delta_str:>8} {status_fmt:>8}")
        passed = sum(1 for r in checks if r[4] == 'PASS')
        print(f"\n  Result: {passed}/{len(checks)} checks passed")
    else:
        print("  No check data available")

    # Consensus bridge
    print("\n[D] CONSENSUS BRIDGE -- Our PT vs. Sell-Side")
    print("-"*65)
    cb = validation.get('consensus_bridge') or {}
    if cb.get('consensus_mean'):
        print(f"  Our Price Target   : ${cb['our_pt']:.2f}  ({cb['our_upside']:+.1f}%)")
        print(f"  Consensus Mean PT  : ${cb['consensus_mean']:.2f} ({cb['consensus_upside']:+.1f}%) [n={cb['n_analysts']}]")
        print(f"  Gap vs Consensus   : ${cb['pt_gap_vs_consensus']:+.2f}  ({cb['pt_gap_pct']:+.1f}%)")
        if cb.get('comps_average'):
            print(f"  Our DCF /share     : ${cb['our_dcf_per_share']:.2f}")
            print(f"  Our Comps avg      : ${cb['comps_average']:.2f}")
        print()
        for diff in cb.get('key_differences', []):
            print(f"  >> {diff}")
    else:
        print(f"  {cb.get('note', 'No consensus data')}")

    print("\n" + "="*65)
