"""
Buy-Side Equity Research Valuation Engine
Methodology: DCF + Comparable Companies + Dividend Discount Model
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# WACC CALCULATOR
# ─────────────────────────────────────────────
def calculate_wacc(ticker_info: dict, beta: float, risk_free_rate: float = 0.0435,
                   equity_risk_premium: float = 0.046,
                   country_risk_premium: float = 0.0) -> dict:
    """
    WACC = (E/V)*Re + (D/V)*Rd*(1-T)
    Risk-free rate: 10-yr US Treasury yield
    ERP: Damodaran long-run average
    Cost of Equity = Rf + beta*ERP + Country Risk Premium (for foreign/ADR names)
    """
    total_debt = ticker_info.get('totalDebt', 0) or 0
    market_cap = ticker_info.get('marketCap', 1) or 1
    total_equity = market_cap
    total_capital = total_equity + total_debt

    # Blume-adjusted beta: betas mean-revert toward 1.0, and a 1yr daily raw beta is
    # noisy / momentum-inflated (e.g. NVDA ~2.2). adj = 0.67*raw + 0.33*1.0.
    raw_beta = beta
    beta = 0.67 * beta + 0.33 * 1.0

    # Cost of Equity — CAPM + country risk premium
    cost_of_equity = risk_free_rate + beta * equity_risk_premium + (country_risk_premium or 0.0)

    # Cost of Debt — interest expense / total debt
    interest_expense = abs(ticker_info.get('interestExpense', 0) or 0)
    if total_debt > 0 and interest_expense > 0:
        cost_of_debt = interest_expense / total_debt
        cost_of_debt = max(min(cost_of_debt, 0.15), 0.02)  # sanity clamp
    else:
        cost_of_debt = risk_free_rate + 0.015  # default spread

    tax_rate = ticker_info.get('effectiveTaxRate', 0.21) or 0.21
    tax_rate = max(min(tax_rate, 0.35), 0.05)

    w_equity = total_equity / total_capital if total_capital > 0 else 1.0
    w_debt = total_debt / total_capital if total_capital > 0 else 0.0

    wacc = w_equity * cost_of_equity + w_debt * cost_of_debt * (1 - tax_rate)
    wacc = max(wacc, 0.06)  # floor at 6%

    return {
        'wacc': wacc,
        'cost_of_equity': cost_of_equity,
        'cost_of_debt': cost_of_debt,
        'tax_rate': tax_rate,
        'w_equity': w_equity,
        'w_debt': w_debt,
        'beta': beta,
        'raw_beta': raw_beta,
        'risk_free_rate': risk_free_rate,
        'erp': equity_risk_premium,
        'country_risk_premium': country_risk_premium or 0.0,
    }


# ─────────────────────────────────────────────
# FREE CASH FLOW NORMALIZER
# ─────────────────────────────────────────────
def calculate_fcf_history(cashflow_df: pd.DataFrame, income_df: pd.DataFrame) -> pd.Series:
    """
    FCFF = EBIT*(1-T) + D&A - ΔWorkingCapital - CapEx
    Use operating cash flow - capex as primary, fallback to EBIT method
    """
    try:
        # Primary: OCF - CapEx
        ocf = cashflow_df.loc['Operating Cash Flow'] if 'Operating Cash Flow' in cashflow_df.index else None
        capex_row = None
        for label in ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of PPE']:
            if label in cashflow_df.index:
                capex_row = cashflow_df.loc[label]
                break

        if ocf is not None:
            capex = capex_row if capex_row is not None else pd.Series(0, index=ocf.index)
            fcf = ocf + capex  # capex is stored negative
            return fcf.dropna()
    except Exception:
        pass

    return pd.Series(dtype=float)


def project_fcf(historical_fcf: pd.Series, revenue_series: pd.Series,
                wacc: float, shares_outstanding: int) -> dict:
    """
    Stage 1: Analyst-estimated growth for 5 years
    Stage 2: Mean-reverting fade to terminal growth
    Terminal value: Gordon Growth Model
    """
    if len(historical_fcf) < 2:
        return None

    fcf_vals = historical_fcf.values[::-1]  # chronological order
    rev_vals = revenue_series.values[::-1] if revenue_series is not None and len(revenue_series) > 0 else None

    # FCF margin stability
    if rev_vals is not None and len(rev_vals) == len(fcf_vals):
        fcf_margins = fcf_vals / rev_vals
        avg_fcf_margin = np.nanmean(fcf_margins[-3:])  # last 3 years
    else:
        avg_fcf_margin = None

    # Historical FCF CAGR
    valid = fcf_vals[fcf_vals > 0]
    if len(valid) >= 2:
        cagr = (valid[-1] / valid[0]) ** (1 / (len(valid) - 1)) - 1
        cagr = max(min(cagr, 0.40), -0.10)
    else:
        cagr = 0.05

    base_fcf = fcf_vals[-1] if fcf_vals[-1] > 0 else abs(fcf_vals[-1]) * 0.5

    # Stage 1: 5-year explicit forecast (regress toward industry mean)
    stage1_growth = min(cagr * 0.8, 0.30)  # haircut for conservatism
    terminal_growth = 0.025  # long-run nominal GDP

    # Stage 2: 5-year fade from stage1 to terminal
    stage2_start = stage1_growth
    stage2_end = terminal_growth

    projected = []
    discount_factors = []

    for t in range(1, 11):
        if t <= 5:
            g = stage1_growth
        else:
            # linear fade
            fade_step = (stage2_start - stage2_end) / 5
            g = stage2_start - fade_step * (t - 5)

        if t == 1:
            fcf_t = base_fcf * (1 + g)
        else:
            fcf_t = projected[-1] * (1 + g)

        projected.append(fcf_t)
        discount_factors.append(1 / (1 + wacc) ** t)

    pv_stage12 = sum(f * d for f, d in zip(projected, discount_factors))

    # Terminal Value
    terminal_fcf = projected[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 10

    total_equity_value = pv_stage12 + pv_terminal
    intrinsic_per_share = total_equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'base_fcf': base_fcf,
        'stage1_growth': stage1_growth,
        'terminal_growth': terminal_growth,
        'projected_fcfs': projected,
        'pv_stage12': pv_stage12,
        'pv_terminal': pv_terminal,
        'total_equity_value': total_equity_value,
        'intrinsic_per_share': intrinsic_per_share,
        'tv_pct_of_total': pv_terminal / (pv_stage12 + pv_terminal) if (pv_stage12 + pv_terminal) > 0 else 0,
        'historical_cagr': cagr,
    }


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
def dcf_sensitivity(base_fcf: float, wacc_base: float, g1_base: float,
                    shares: int, debt: float, cash: float) -> pd.DataFrame:
    """2D sensitivity: WACC vs terminal growth"""
    wacc_range = [wacc_base - 0.02, wacc_base - 0.01, wacc_base,
                  wacc_base + 0.01, wacc_base + 0.02]
    tg_range = [0.015, 0.020, 0.025, 0.030, 0.035]

    rows = []
    for wacc in wacc_range:
        row = []
        for tg in tg_range:
            proj = []
            for t in range(1, 11):
                g = g1_base if t <= 5 else g1_base - (g1_base - tg) / 5 * (t - 5)
                proj.append((proj[-1] if proj else base_fcf) * (1 + g))
            pv12 = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(proj))
            tv = proj[-1] * (1 + tg) / (wacc - tg) / (1 + wacc) ** 10
            total = (pv12 + tv - debt + cash) / shares
            row.append(round(total, 2))
        rows.append(row)

    return pd.DataFrame(rows,
                        index=[f"{w*100:.1f}%" for w in wacc_range],
                        columns=[f"{t*100:.1f}%" for t in tg_range])


# ─────────────────────────────────────────────
# COMPARABLE COMPANY MULTIPLES
# ─────────────────────────────────────────────
def comps_valuation(ticker_info: dict, sector_multiples: dict,
                    live_peer_multiples: dict = None, fx_to_usd: float = 1.0) -> dict:
    """
    Value using EV/EBITDA, P/E, EV/Revenue multiples.

    Multiple source priority:
      1. live_peer_multiples — current median multiples computed from the actual
         peer set this run (passed in from peer_pe_model). Preferred because the
         hardcoded table goes stale.
      2. sector_multiples — static Damodaran fallback table.

    fx_to_usd converts reporting-currency fundamentals (EBITDA/revenue/income/debt)
    onto the USD per-share basis for ADRs.
    """
    ebitda = (ticker_info.get('ebitda') or 0) * fx_to_usd
    net_income = (ticker_info.get('netIncomeToCommon') or 0) * fx_to_usd
    revenue = (ticker_info.get('totalRevenue') or 0) * fx_to_usd
    shares = ticker_info.get('sharesOutstanding') or 1
    total_debt = (ticker_info.get('totalDebt') or 0) * fx_to_usd
    cash = (ticker_info.get('totalCash') or 0) * fx_to_usd
    market_cap = ticker_info.get('marketCap') or 0
    enterprise_value = market_cap + total_debt - cash

    # Merge: live peer medians override the static table where available
    lpm = live_peer_multiples or {}
    mult = dict(sector_multiples)
    mult_source = {'ev_ebitda': 'sector', 'pe': 'sector', 'ev_revenue': 'sector'}
    for k in ('ev_ebitda', 'pe', 'ev_revenue'):
        if lpm.get(k) and lpm[k] > 0:
            mult[k] = lpm[k]
            mult_source[k] = 'live_peers'
    sector_multiples = mult

    results = {}

    # EV/EBITDA
    if ebitda and ebitda > 0 and sector_multiples.get('ev_ebitda'):
        implied_ev = ebitda * sector_multiples['ev_ebitda']
        implied_equity = implied_ev - total_debt + cash
        results['ev_ebitda'] = {
            'multiple': sector_multiples['ev_ebitda'],
            'implied_price': implied_equity / shares,
            'metric': ebitda,
            'source': mult_source['ev_ebitda'],
        }

    # P/E
    if net_income and net_income > 0 and sector_multiples.get('pe'):
        eps = net_income / shares
        results['pe'] = {
            'multiple': sector_multiples['pe'],
            'implied_price': eps * sector_multiples['pe'],
            'metric': eps,
            'source': mult_source['pe'],
        }

    # EV/Revenue (especially for growth companies)
    if revenue and revenue > 0 and sector_multiples.get('ev_revenue'):
        implied_ev = revenue * sector_multiples['ev_revenue']
        implied_equity = implied_ev - total_debt + cash
        results['ev_revenue'] = {
            'multiple': sector_multiples['ev_revenue'],
            'implied_price': implied_equity / shares,
            'metric': revenue,
            'source': mult_source['ev_revenue'],
        }

    return results


# ─────────────────────────────────────────────
# DIVIDEND DISCOUNT MODEL
# ─────────────────────────────────────────────
def ddm_valuation(ticker_info: dict, wacc: float) -> dict | None:
    """H-Model DDM: blend high near-term + long-run stable growth"""
    div_rate = ticker_info.get('dividendRate') or 0
    if div_rate <= 0:
        return None

    payout = ticker_info.get('payoutRatio') or 0.3
    roe = ticker_info.get('returnOnEquity') or 0.12
    if roe < 0:
        return None

    # Sustainable growth = ROE * retention
    g_sustainable = roe * (1 - payout)
    g_stable = 0.025
    half_life = 5  # H-Model half-life

    # H-Model: P = D0 * (1+g_stable) / (r - g_stable)  +  D0 * H * (g_sustainable - g_stable) / (r - g_stable)
    if wacc <= g_sustainable:
        wacc = g_sustainable + 0.02  # prevent division by zero

    p_stable = div_rate * (1 + g_stable) / (wacc - g_stable)
    p_growth = div_rate * half_life * (g_sustainable - g_stable) / (wacc - g_stable)
    value = p_stable + p_growth

    return {
        'model': 'H-Model DDM',
        'annual_dividend': div_rate,
        'sustainable_growth': g_sustainable,
        'stable_growth': g_stable,
        'implied_price': value,
    }


# ─────────────────────────────────────────────
# QUALITY SCORING (Piotroski-inspired F-Score)
# ─────────────────────────────────────────────
def quality_score(info: dict, cashflow_df: pd.DataFrame, balance_df: pd.DataFrame,
                  income_df: pd.DataFrame) -> dict:
    """
    Piotroski F-Score (9 points) — the real definition, using year-over-year
    CHANGES from the filings (not single-period level checks).

    Profitability (4): positive ROA, positive OCF, rising ROA, accruals (OCF>NI)
    Leverage/Liquidity (3): falling leverage, rising current ratio, no dilution
    Operating Efficiency (2): rising gross margin, rising asset turnover

    Falls back to a level-based proxy for any factor whose prior-year data
    is missing, so the score is still populated for thin-history names.
    """
    def col(df, *labels, idx=0):
        """Value at column `idx` (0=latest, 1=prior year) for first matching row."""
        if df is None or df.empty:
            return None
        for l in labels:
            if l in df.index:
                row = df.loc[l]
                try:
                    if hasattr(row, 'iloc') and len(row) > idx:
                        v = row.iloc[idx]
                        return float(v) if pd.notna(v) else None
                except Exception:
                    return None
        return None

    # Latest (t) and prior (t-1) raw inputs
    ni_t   = col(income_df, 'Net Income', 'Net Income Common Stockholders', idx=0)
    ni_p   = col(income_df, 'Net Income', 'Net Income Common Stockholders', idx=1)
    ta_t   = col(balance_df, 'Total Assets', idx=0)
    ta_p   = col(balance_df, 'Total Assets', idx=1)
    ocf_t  = col(cashflow_df, 'Operating Cash Flow', 'Cash From Operating Activities', idx=0)
    ltd_t  = col(balance_df, 'Long Term Debt', idx=0)
    ltd_p  = col(balance_df, 'Long Term Debt', idx=1)
    ca_t   = col(balance_df, 'Current Assets', idx=0)
    ca_p   = col(balance_df, 'Current Assets', idx=1)
    cl_t   = col(balance_df, 'Current Liabilities', idx=0)
    cl_p   = col(balance_df, 'Current Liabilities', idx=1)
    sh_t   = col(balance_df, 'Share Issued', 'Ordinary Shares Number', 'Common Stock', idx=0)
    sh_p   = col(balance_df, 'Share Issued', 'Ordinary Shares Number', 'Common Stock', idx=1)
    rev_t  = col(income_df, 'Total Revenue', 'Revenue', idx=0)
    rev_p  = col(income_df, 'Total Revenue', 'Revenue', idx=1)
    gp_t   = col(income_df, 'Gross Profit', idx=0)
    gp_p   = col(income_df, 'Gross Profit', idx=1)

    score = 0
    details = {}

    def award(label, condition):
        nonlocal score
        ok = bool(condition)
        if ok:
            score += 1
        details[label] = ok

    # ── Profitability ──
    roa_t = (ni_t / ta_t) if (ni_t is not None and ta_t) else (info.get('returnOnAssets') or None)
    roa_p = (ni_p / ta_p) if (ni_p is not None and ta_p) else None
    award('Positive ROA', (roa_t or 0) > 0)
    award('Positive Operating CF', (ocf_t if ocf_t is not None else (info.get('operatingCashflow') or 0)) > 0)
    if roa_t is not None and roa_p is not None:
        award('ROA rising (YoY)', roa_t > roa_p)
    else:
        award('ROA rising (YoY)', (info.get('revenueGrowth') or 0) > 0)  # proxy
    ocf_for_accrual = ocf_t if ocf_t is not None else (info.get('operatingCashflow') or 0)
    ni_for_accrual = ni_t if ni_t is not None else (info.get('netIncomeToCommon') or 0)
    award('Accruals OK (OCF > NI)', ni_for_accrual != 0 and ocf_for_accrual > ni_for_accrual)

    # ── Leverage / Liquidity ──
    if ltd_t is not None and ltd_p is not None and ta_t and ta_p:
        award('Leverage falling (YoY)', (ltd_t / ta_t) < (ltd_p / ta_p))
    else:
        award('Leverage falling (YoY)', ((info.get('debtToEquity') or 100) / 100) < 1.0)  # proxy
    if ca_t and cl_t and ca_p and cl_p:
        award('Current ratio rising (YoY)', (ca_t / cl_t) > (ca_p / cl_p))
    else:
        award('Current ratio rising (YoY)', (info.get('currentRatio') or 0) > 1)  # proxy
    if sh_t is not None and sh_p is not None:
        award('No share dilution (YoY)', sh_t <= sh_p * 1.005)  # allow rounding
    else:
        award('No share dilution (YoY)', True)  # unknown → benefit of doubt

    # ── Operating Efficiency ──
    if gp_t is not None and gp_p is not None and rev_t and rev_p:
        award('Gross margin rising (YoY)', (gp_t / rev_t) > (gp_p / rev_p))
    else:
        award('Gross margin rising (YoY)', (info.get('grossMargins') or 0) > 0.30)  # proxy
    if rev_t and rev_p and ta_t and ta_p:
        award('Asset turnover rising (YoY)', (rev_t / ta_t) > (rev_p / ta_p))
    else:
        award('Asset turnover rising (YoY)', (info.get('revenueGrowth') or 0) > 0)  # proxy

    label = 'Strong' if score >= 7 else 'Average' if score >= 4 else 'Weak'
    return {'score': score, 'max': 9, 'label': label, 'details': details}


# ─────────────────────────────────────────────
# PRICE TARGET & RECOMMENDATION
# ─────────────────────────────────────────────
def derive_price_target(valuations: dict, current_price: float) -> dict:
    """
    Weighted average of DCF (50%), Comps EV/EBITDA (30%), Comps P/E (20%)
    Adjust weights if metrics unavailable
    """
    weights = {'dcf': 0.50, 'ev_ebitda': 0.25, 'pe': 0.15, 'ev_revenue': 0.10}

    available_vals = {}
    if valuations.get('dcf') and valuations['dcf'].get('intrinsic_per_share', 0) > 0:
        available_vals['dcf'] = valuations['dcf']['intrinsic_per_share']
    if valuations.get('comps', {}).get('ev_ebitda'):
        available_vals['ev_ebitda'] = valuations['comps']['ev_ebitda']['implied_price']
    if valuations.get('comps', {}).get('pe'):
        available_vals['pe'] = valuations['comps']['pe']['implied_price']
    if valuations.get('comps', {}).get('ev_revenue') and not available_vals.get('ev_ebitda'):
        available_vals['ev_revenue'] = valuations['comps']['ev_revenue']['implied_price']

    if not available_vals:
        return {'price_target': current_price, 'upside': 0, 'recommendation': 'HOLD'}

    # Normalize weights
    active_weights = {k: weights.get(k, 0.1) for k in available_vals}
    total_w = sum(active_weights.values())
    norm_weights = {k: v / total_w for k, v in active_weights.items()}

    price_target = sum(available_vals[k] * norm_weights[k] for k in available_vals)
    price_target = max(price_target, 0.01)

    # Sanity cap: PT cannot exceed 5x current price (catches ADR currency bugs, bad data)
    # If any single method gives >5x, down-weight it and recalculate without it
    MAX_PT_MULTIPLE = 5.0
    if current_price > 0 and price_target > current_price * MAX_PT_MULTIPLE:
        sane_vals = {k: v for k, v in available_vals.items()
                     if v <= current_price * MAX_PT_MULTIPLE}
        if sane_vals:
            sane_w = {k: weights.get(k, 0.1) for k in sane_vals}
            tw = sum(sane_w.values())
            sane_norm = {k: v / tw for k, v in sane_w.items()}
            price_target = sum(sane_vals[k] * sane_norm[k] for k in sane_vals)
            available_vals = sane_vals
            norm_weights = sane_norm
        else:
            # All methods blew up (e.g. ADR currency): fall back to current price
            price_target = current_price

    upside = (price_target - current_price) / current_price

    # Recommendation thresholds (buy-side convention)
    if upside > 0.15:
        rec = 'BUY'
    elif upside > -0.05:
        rec = 'HOLD'
    else:
        rec = 'SELL'

    component_breakdown = {k: {'value': round(available_vals[k], 2), 'weight': round(norm_weights[k], 2)}
                           for k in available_vals}

    return {
        'price_target': round(price_target, 2),
        'upside': round(upside * 100, 1),
        'recommendation': rec,
        'components': component_breakdown,
    }
