"""
Portfolio Construction Layer  (組合層 — 把「一籃子單股評分」變成「能在最壞那天活下來的組合」)
═══════════════════════════════════════════════════════════════════════════════
PRE-TRADE sizing, distinct from portfolio.py (which TRACKS positions you already hold).
This takes the per-name bottom-fishing candidates (each with a sized staged_entry_plan)
and decides how big each one is ALLOWED to be once you look at them together.

Why it exists: the single-name engine sizes each candidate to risk ~1% in isolation. In a
real left-side washout the 8-10 names that screen as "STRONG 抄底" are almost always the
SAME bet — high-beta, oversold, clustered in one or two sectors. They fell together and
they bounce (or keep falling) together. Ten "independent" 1% positions in correlated names
is not a diversified book; it is one ~10% directional bet in a diversification costume —
the classic way a contrarian book blows up.

This layer scales position sizes DOWN (never up) until these hold:

  • PORTFOLIO HEAT      — total $-risk across all candidates ≤ a fixed budget.
  • SECTOR CAPS         — $-risk per sector ≤ cap (no single theme dominates).
  • CORRELATION CLUSTERS— names whose recent returns move together SHARE one risk budget,
                          so "5 semis that all gap with SOX" count as ~1 bet, not 5.
  • GROSS EXPOSURE      — total deployed $ ≤ cap.

Each affected plan is rescaled end-to-end (tranche shares, position value, $-risk) and
annotated with which constraint bound it, so the dashboard can show WHY a name was trimmed.

API:
  from portfolio_construct import construct_portfolio
  summary = construct_portfolio(results, regime=regime, config=cfg)   # mutates results in place
"""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

PORTFOLIO_DEFAULTS = {
    'account_size':        100_000.0,
    'max_portfolio_heat':  0.06,    # total risk budget across the whole book (6% of equity)
    'max_sector_risk':     0.025,   # max $-risk concentrated in any one sector (2.5%)
    'max_cluster_risk':    0.025,   # max $-risk in any one correlated cluster (2.5%)
    'max_gross_exposure':  0.60,    # max fraction of equity deployed at once
    'corr_threshold':      0.65,    # pairwise return corr ≥ this → same cluster
    'corr_lookback':       90,      # trading days used for the correlation estimate
}


# ─────────────────────────────────────────────────────────────────────────────
# Returns + correlation clustering
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_returns(tickers: list, lookback: int) -> pd.DataFrame:
    """Daily returns for the candidate set (one download). Empty frame on failure —
    callers degrade gracefully to 'every name is its own cluster'."""
    if not tickers:
        return pd.DataFrame()
    try:
        import yfinance as yf
        period = f"{max(lookback + 40, 130)}d"
        raw = yf.download(tickers, period=period, interval='1d',
                          auto_adjust=True, progress=False, group_by='ticker')
        closes = {}
        for tk in tickers:
            try:
                df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
                c = df['Close'].dropna()
                if len(c) >= 30:
                    closes[tk] = c
            except Exception:
                continue
        if not closes:
            return pd.DataFrame()
        px = pd.DataFrame(closes).tail(lookback + 1)
        return px.pct_change().dropna(how='all')
    except Exception:
        return pd.DataFrame()


def _clusters_from_corr(corr: pd.DataFrame, threshold: float) -> dict:
    """Union-find clustering: any two names with |corr| ≥ threshold land in one cluster.
    Returns {ticker: cluster_id}. Names absent from corr each get their own cluster."""
    tickers = list(corr.columns)
    parent = {t: t for t in tickers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            try:
                if abs(float(corr.loc[a, b])) >= threshold:
                    union(a, b)
            except Exception:
                continue
    roots = {}
    out = {}
    for t in tickers:
        r = find(t)
        roots.setdefault(r, len(roots))
        out[t] = roots[r]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plan rescaling
# ─────────────────────────────────────────────────────────────────────────────

def _rescale_plan(plan: dict, factor: float, account_size: float) -> dict:
    """Scale a staged-entry plan's size by `factor` (0..1), recomputing tranche shares,
    position value, position %, and $-risk consistently. Prices/stop/targets unchanged."""
    if factor >= 0.999 or not plan.get('tranches'):
        return plan
    rps = plan.get('risk_per_share') or 0.0
    total_shares = int(plan.get('total_shares', 0) * factor)
    tranches = plan['tranches']
    n = len(tranches)
    per = total_shares // n if n else 0
    new_total = 0
    for i, t in enumerate(tranches):
        sh = per if i < n - 1 else (total_shares - per * (n - 1))
        sh = max(0, sh)
        t['shares'] = sh
        t['value'] = round(sh * t['price'], 0)
        new_total += sh
    for t in tranches:
        t['pct_of_plan'] = round(t['shares'] / new_total * 100, 0) if new_total else 0
    plan['total_shares'] = new_total
    plan['position_value'] = round(sum(t['value'] for t in tranches), 0)
    plan['position_pct'] = round(plan['position_value'] / account_size * 100, 1) if account_size else 0
    plan['dollar_risk'] = round(new_total * rps, 0)
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def construct_portfolio(results: list, regime: dict | None = None,
                        config: dict | None = None) -> dict:
    """
    Apply portfolio-level risk controls to a list of analyze_bottom_fish results.
    Mutates each tradeable result in place (rescaled plan + an attached `portfolio` dict)
    and returns a summary for the dashboard.
    """
    cfg = {**PORTFOLIO_DEFAULTS, **(config or {})}
    acct = cfg['account_size']

    # Portfolio heat itself flexes with the market regime: in a defensive regime the whole
    # book runs cooler. The left-side dial scaling INTO an orderly fear spike is already a
    # per-name effect; here we keep the TOTAL bounded so scaling-in can't blow the budget.
    regime_mult = 1.0
    if regime is not None:
        regime_mult = regime.get('left_multiplier', regime.get('multiplier', 1.0)) or 1.0
        regime_mult = max(0.4, min(1.25, regime_mult))
    heat_budget    = acct * cfg['max_portfolio_heat'] * regime_mult
    sector_budget  = acct * cfg['max_sector_risk']
    cluster_budget = acct * cfg['max_cluster_risk']
    gross_budget   = acct * cfg['max_gross_exposure']

    # Eligible = tradeable names with an actual sized entry plan.
    elig = []
    for r in results:
        if r.get('status') != 'OK':
            continue
        plan = r.get('plan') or {}
        if not plan.get('tranches') or not plan.get('dollar_risk'):
            continue
        elig.append(r)

    summary = {
        'account_size': acct, 'regime_mult': round(regime_mult, 2),
        'heat_budget': round(heat_budget, 0), 'gross_budget': round(gross_budget, 0),
        'sector_budget': round(sector_budget, 0), 'cluster_budget': round(cluster_budget, 0),
        'n_eligible': len(elig), 'clusters': [], 'sectors': [],
        'applied': False,
    }
    if not elig:
        return summary

    tickers = [r['ticker'] for r in elig]
    risk0   = {r['ticker']: float(r['plan']['dollar_risk']) for r in elig}
    val0    = {r['ticker']: float(r['plan'].get('position_value', 0)) for r in elig}
    sector  = {r['ticker']: (r.get('sector') or '—') for r in elig}
    factor  = {t: 1.0 for t in tickers}
    binding = {t: [] for t in tickers}

    summary['heat_before'] = round(sum(risk0.values()), 0)
    summary['gross_before'] = round(sum(val0.values()), 0)

    # ── Correlation clustering (degrade to singletons if data unavailable) ──
    rets = _fetch_returns(tickers, cfg['corr_lookback'])
    if not rets.empty and rets.shape[1] >= 2:
        corr = rets.corr()
        cluster = _clusters_from_corr(corr, cfg['corr_threshold'])
    else:
        corr = None
        cluster = {t: i for i, t in enumerate(tickers)}
    for r in elig:
        r.setdefault('portfolio', {})['cluster'] = cluster.get(r['ticker'])

    def cur_risk(t): return risk0[t] * factor[t]
    def cur_val(t):  return val0[t] * factor[t]

    def _apply_group_cap(groups: dict, budget: float, tag: str):
        """groups: {key: [tickers]}. Scale each over-budget group's members down to budget."""
        for key, members in groups.items():
            s = sum(cur_risk(t) for t in members)
            if s > budget and s > 0:
                f = budget / s
                for t in members:
                    factor[t] *= f
                    binding[t].append(tag)

    # 1) Sector caps
    sec_groups = {}
    for t in tickers:
        sec_groups.setdefault(sector[t], []).append(t)
    _apply_group_cap(sec_groups, sector_budget, 'sector')

    # 2) Correlation-cluster caps
    cl_groups = {}
    for t in tickers:
        cl_groups.setdefault(cluster[t], []).append(t)
    _apply_group_cap(cl_groups, cluster_budget, 'cluster')

    # 3) Portfolio heat (total risk)
    total_risk = sum(cur_risk(t) for t in tickers)
    if total_risk > heat_budget and total_risk > 0:
        g = heat_budget / total_risk
        for t in tickers:
            factor[t] *= g
            binding[t].append('heat')

    # 4) Gross exposure (total deployed $)
    total_val = sum(cur_val(t) for t in tickers)
    if total_val > gross_budget and total_val > 0:
        g = gross_budget / total_val
        for t in tickers:
            factor[t] *= g
            binding[t].append('gross')

    # ── Apply scaling to the plans + annotate ──
    for r in elig:
        t = r['ticker']
        f = max(0.0, min(1.0, factor[t]))
        _rescale_plan(r['plan'], f, acct)
        binds = sorted(set(binding[t]))
        r['portfolio'].update({
            'scale': round(f, 2),
            'binding': binds,
            'risk_before': round(risk0[t], 0),
            'risk_after': round(cur_risk(t), 0),
            'sector': sector[t],
            'note': _binding_note(binds, f),
        })

    # ── Summary tables ──
    summary['applied'] = True
    summary['heat_after'] = round(sum(cur_risk(t) for t in tickers), 0)
    summary['gross_after'] = round(sum(cur_val(t) for t in tickers), 0)
    summary['corr_available'] = corr is not None

    for sec, members in sorted(sec_groups.items(), key=lambda kv: -sum(cur_risk(t) for t in kv[1])):
        summary['sectors'].append({
            'sector': sec, 'tickers': members,
            'risk_before': round(sum(risk0[t] for t in members), 0),
            'risk_after': round(sum(cur_risk(t) for t in members), 0),
            'over_budget': sum(risk0[t] for t in members) > sector_budget,
        })
    for cid, members in sorted(cl_groups.items(), key=lambda kv: -sum(cur_risk(t) for t in kv[1])):
        if len(members) < 2 and corr is None:
            continue
        avg_corr = None
        if corr is not None and len(members) >= 2:
            vals = [abs(float(corr.loc[a, b])) for i, a in enumerate(members)
                    for b in members[i + 1:] if a in corr.columns and b in corr.columns]
            avg_corr = round(float(np.mean(vals)), 2) if vals else None
        summary['clusters'].append({
            'id': cid, 'tickers': members, 'avg_corr': avg_corr,
            'risk_before': round(sum(risk0[t] for t in members), 0),
            'risk_after': round(sum(cur_risk(t) for t in members), 0),
            'over_budget': sum(risk0[t] for t in members) > cluster_budget,
        })
    return summary


def _binding_note(binds: list, factor: float) -> str:
    if not binds or factor >= 0.999:
        return '組合層未縮減(獨立風險已在預算內)。'
    names = {'sector': '產業集中度上限', 'cluster': '相關性群組上限',
             'heat': '組合總風險上限', 'gross': '總曝險上限'}
    hit = '、'.join(names.get(b, b) for b in binds)
    return f'組合層縮減至 {factor*100:.0f}%(觸發:{hit})— 與其他相關部位共用風險預算,避免「十個 1% 其實是一個 10% 賭注」。'


def print_portfolio(summary: dict):
    if not summary or not summary.get('applied'):
        print("  [portfolio] 無可組合的部位(沒有 STRONG/SPECULATIVE 帶倉位的標的)。")
        return
    acct = summary['account_size']
    print("\n" + "=" * 78)
    print("  組合層風險控管 (PORTFOLIO RISK LAYER)")
    print("=" * 78)
    print(f"  帳戶 ${acct:,.0f} · 體制係數 ×{summary['regime_mult']:.2f}")
    hb, ha = summary.get('heat_before', 0), summary.get('heat_after', 0)
    gb, ga = summary.get('gross_before', 0), summary.get('gross_after', 0)
    print(f"  組合總風險 heat:  ${hb:,.0f} -> ${ha:,.0f}  (預算 ${summary['heat_budget']:,.0f}"
          f" = {summary['heat_budget']/acct*100:.1f}%)")
    print(f"  總曝險 gross:     ${gb:,.0f} -> ${ga:,.0f}  (預算 ${summary['gross_budget']:,.0f}"
          f" = {summary['gross_budget']/acct*100:.0f}%)")
    if summary.get('n_eligible', 0) >= 2 and not summary.get('corr_available'):
        print("  [warn] 相關性資料不可用 — 已退化為『每檔自成一群』,僅套用產業/總風險上限。")
    if summary.get('clusters'):
        print("  -- 相關性群組(同群=同一個賭注,共用風險預算)--")
        for c in summary['clusters']:
            if len(c['tickers']) < 2:
                continue
            flag = ' [超預算]' if c['over_budget'] else ''
            cc = f"avg rho={c['avg_corr']}" if c['avg_corr'] is not None else ''
            print(f"    群#{c['id']}: {', '.join(c['tickers'])}  {cc}"
                  f"  風險 ${c['risk_before']:,.0f}->${c['risk_after']:,.0f}{flag}")
    print("  -- 產業集中度 --")
    for s in summary['sectors']:
        flag = ' [超預算]' if s['over_budget'] else ''
        print(f"    {s['sector'][:22]:24} {', '.join(s['tickers'])[:40]:42}"
              f"  ${s['risk_before']:,.0f}->${s['risk_after']:,.0f}{flag}")
    print("=" * 78)
