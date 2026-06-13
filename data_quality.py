"""
Data Quality Guard
──────────────────
A valuation is only as good as its inputs. yfinance occasionally serves corrupted
data (e.g. SNDK reported a $1,559 price / $230B market cap / $177 forward EPS for
Sandisk — all wrong). Feeding that into a DCF produces a confident, nonsensical
"BUY +53%". This module runs cheap internal-consistency checks and assigns a
reliability verdict so the pipeline can FLAG suspect names instead of pretending.

The checks are deliberately source-agnostic and self-referential — they don't need
a second data feed, they just test whether the numbers a single source returned are
internally coherent.
"""
from __future__ import annotations


# Severity → how much it should erode trust
_SEVERITY_WEIGHT = {'critical': 1.0, 'high': 0.6, 'medium': 0.3, 'low': 0.1}


def _flag(checks, name, severity, detail):
    checks.append({'check': name, 'severity': severity, 'detail': detail})


def assess_data_quality(info: dict, *, fx_to_usd: float = 1.0) -> dict:
    """
    Returns:
      {
        'reliable': bool,            # False -> do not trust the valuation
        'verdict': 'OK'|'REVIEW'|'UNRELIABLE',
        'score': 0..100,             # 100 = clean
        'flags': [ {check, severity, detail}, ... ],
      }
    """
    checks = []

    price = (info.get('currentPrice') or info.get('regularMarketPrice')
             or info.get('previousClose') or 0)
    prev = info.get('previousClose') or 0
    shares = info.get('sharesOutstanding') or 0
    mktcap = info.get('marketCap') or 0
    eps_ttm = info.get('trailingEps')
    eps_fwd = info.get('forwardEps')
    revenue = (info.get('totalRevenue') or 0) * fx_to_usd
    fcf = (info.get('freeCashflow') or 0) * fx_to_usd

    # 1. price × shares should ≈ market cap (the SNDK signature). The two come from
    #    different fields, so a large mismatch means at least one is corrupt.
    if price and shares and mktcap:
        implied = price * shares
        ratio = implied / mktcap if mktcap else 0
        if ratio < 0.5 or ratio > 2.0:
            _flag(checks, 'price×shares vs marketCap', 'critical',
                  f"price×shares = {implied/1e9:.1f}B but marketCap = {mktcap/1e9:.1f}B "
                  f"(ratio {ratio:.2f}; expect ~1.0)")

    # 2. P/E sanity — an absurd trailing P/E usually means a bad price or bad EPS.
    if eps_ttm and eps_ttm > 0 and price:
        pe = price / eps_ttm
        if pe > 300:
            _flag(checks, 'extreme P/E', 'high', f"trailing P/E = {pe:.0f}x")
        elif pe > 150:
            _flag(checks, 'high P/E', 'medium', f"trailing P/E = {pe:.0f}x — verify price/EPS")

    # 3. forward vs trailing EPS — a >6x jump is almost always a data error.
    if eps_ttm and eps_fwd and eps_ttm > 0 and eps_fwd > 0:
        jump = eps_fwd / eps_ttm
        if jump > 6 or jump < 1 / 6:
            _flag(checks, 'EPS fwd/ttm discontinuity', 'high',
                  f"forward EPS {eps_fwd:.2f} vs trailing {eps_ttm:.2f} ({jump:.1f}x)")

    # 3b. ACCOUNTING IDENTITY: implied net income (EPS × shares) cannot exceed revenue.
    #     This is what nails SNDK — fwd EPS 177.8 × 148M shares = $26B "earnings" on
    #     $13B revenue is impossible. A near-bulletproof corruption detector.
    #     Financials (banks/insurers) report revenue oddly, so flag CRITICAL only when
    #     implied earnings exceed 1.5× revenue (unambiguous), HIGH between 1.05–1.5×.
    sector = (info.get('sector') or '').lower()
    is_financial = 'financial' in sector or 'bank' in sector or 'insurance' in sector
    if shares and revenue and revenue > 0 and not is_financial:
        for label, eps in (('trailing', eps_ttm), ('forward', eps_fwd)):
            if eps and eps > 0:
                implied_ni = eps * shares  # price currency; revenue fx-adjusted above
                ratio = implied_ni / revenue
                if ratio > 1.5:
                    _flag(checks, f'implied earnings > revenue ({label})', 'critical',
                          f"{label} EPS×shares = {implied_ni/1e9:.1f}B = {ratio:.1f}× revenue "
                          f"{revenue/1e9:.1f}B — net margin >100% is impossible")
                elif ratio > 1.05:
                    _flag(checks, f'implied earnings near/above revenue ({label})', 'high',
                          f"{label} EPS×shares = {implied_ni/1e9:.1f}B = {ratio:.1f}× revenue "
                          f"{revenue/1e9:.1f}B — verify EPS/share count")

    # 4. implied P/E on market cap vs revenue — a memory/hardware name at >40x sales
    #    with thin margins is a red flag for a corrupted price (SNDK: 230B cap on 13B rev).
    if mktcap and revenue and revenue > 0:
        ps = mktcap / revenue
        margin = (info.get('operatingMargins') or 0)
        if ps > 40 and margin < 0.35:
            _flag(checks, 'price/sales vs margin', 'high',
                  f"P/S = {ps:.0f}x on {margin*100:.0f}% op margin — implausible")

    # 5. daily move sanity — currentPrice vs previousClose. A >35% gap with no obvious
    #    halt usually means stale/wrong intraday data.
    if price and prev and prev > 0:
        move = abs(price / prev - 1)
        if move > 0.35:
            _flag(checks, 'price vs previousClose gap', 'medium',
                  f"{move*100:.0f}% gap vs previous close — verify price feed")

    # 6. FCF vs market cap — yield outside a sane band is suspicious (negative is fine,
    #    but a >40% FCF yield or implausibly tiny one can flag unit/currency errors).
    if mktcap and fcf:
        fcf_yield = fcf / mktcap
        if fcf_yield > 0.40:
            _flag(checks, 'implausible FCF yield', 'medium',
                  f"FCF yield = {fcf_yield*100:.0f}% — check currency/units")

    # 7. missing essentials
    if not price:
        _flag(checks, 'missing price', 'critical', 'no usable price field')
    if not shares:
        _flag(checks, 'missing share count', 'high', 'no sharesOutstanding')

    # Aggregate
    erosion = sum(_SEVERITY_WEIGHT.get(c['severity'], 0.2) for c in checks)
    score = max(0, round(100 * (1 - min(erosion, 1.0))))
    has_critical = any(c['severity'] == 'critical' for c in checks)

    if has_critical or score < 40:
        verdict, reliable = 'UNRELIABLE', False
    elif score < 75:
        verdict, reliable = 'REVIEW', True
    else:
        verdict, reliable = 'OK', True

    return {'reliable': reliable, 'verdict': verdict, 'score': score, 'flags': checks}


def format_quality_banner(dq: dict) -> str:
    """One-line console banner."""
    icon = {'OK': '[ok]', 'REVIEW': '[!]', 'UNRELIABLE': '[XX]'}.get(dq['verdict'], '[?]')
    line = f"{icon} Data quality: {dq['verdict']} ({dq['score']}/100)"
    if dq['flags']:
        top = dq['flags'][0]
        line += f" — {top['check']}: {top['detail']}"
    return line
