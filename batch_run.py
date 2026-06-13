"""
Batch runner — generate reports for multiple tickers + dashboard
Usage:
  python batch_run.py AVGO COHR SNDK PENG NOK WOLF TSM GLW
  python batch_run.py --watchlist my_watchlist.txt
  python batch_run.py AAPL MSFT
"""
import sys
import os
import json
import argparse
import traceback
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(REPORTS_DIR, '_summary.json')


_REGIME_CACHE: dict | None = None   # fetch once per batch run


def _get_regime_cached() -> dict:
    global _REGIME_CACHE
    if _REGIME_CACHE is None:
        try:
            from regime import get_regime
            _REGIME_CACHE = get_regime()
            print(f"\n  [regime] {_REGIME_CACHE['regime']}  "
                  f"VIX={_REGIME_CACHE.get('vix')}  "
                  f"SPY vs 200dma {_REGIME_CACHE.get('spy_vs_200_pct',0):+.1f}%  "
                  f"sizing x{_REGIME_CACHE['multiplier']:.0%}")
        except Exception as e:
            _REGIME_CACHE = {'regime': 'CAUTION', 'multiplier': 0.80,
                             'description': f'Regime fetch failed: {e}'}
    return _REGIME_CACHE


def _run_single(ticker: str) -> dict:
    from data_fetcher import (fetch_company_data, get_macro_context,
                              get_sector_multiples, get_country_risk_premium)
    from valuation_engine import calculate_wacc, quality_score, comps_valuation, ddm_valuation
    from report_generator import render_report
    from validation import run_validation
    from peer_pe_model import run_peer_pe_model
    from consensus_model import run_consensus_model, dcf_sensitivity_grid

    output_path = os.path.join(REPORTS_DIR, f'{ticker}_equity_report.html')

    try:
        macro = get_macro_context()
        data = fetch_company_data(ticker)
        info = data['info']

        current_price = (info.get('currentPrice') or info.get('regularMarketPrice')
                         or info.get('previousClose') or 0)
        if not current_price:
            return {'ticker': ticker, 'status': 'ERROR', 'error': 'No price data'}

        sector = info.get('sector') or 'default'
        sector_mults = get_sector_multiples(sector)

        # WACC (with country-risk premium for ADRs) + Blume-adjusted beta inside
        beta = data['beta']
        crp = get_country_risk_premium(info.get('country'))
        wacc_data = calculate_wacc(info, beta=beta, risk_free_rate=macro['risk_free_rate'],
                                   country_risk_premium=crp)

        # ── Peer P/E first — its live multiples feed comps ─────
        peer_model = run_peer_pe_model(data)
        data['peer_model'] = peer_model
        peer_stats = peer_model.get('peer_pe_stats') or {}
        live_peer_multiples = {'pe': peer_stats['median']} if peer_stats.get('median') else {}

        # ── CONSENSUS MODEL (primary) ──────────────────────────
        consensus_result = run_consensus_model(data, wacc_data, sector_mults,
                                               live_peer_multiples=live_peer_multiples)
        pt_data = consensus_result['composite_pt']
        data['consensus_result'] = consensus_result

        # Data-quality gate
        from data_quality import assess_data_quality
        dq = assess_data_quality(info, fx_to_usd=data.get('fx_to_usd', 1.0))
        data['data_quality'] = dq
        if not dq['reliable']:
            pt_data['rec_before_dq'] = pt_data['recommendation']
            pt_data['recommendation'] = 'NO RATING'

        shares = info.get('sharesOutstanding') or 1
        primary_dcf = consensus_result.get('consensus_dcf')
        base_fcf_data = consensus_result.get('base_fcf_data')
        sbc_dilution = consensus_result.get('sbc_dilution', 0.0)
        fx = data.get('fx_to_usd', 1.0)

        comps_result = consensus_result.get('comps') or comps_valuation(
            info, sector_mults, live_peer_multiples=live_peer_multiples, fx_to_usd=fx)
        ddm_result = ddm_valuation(info, wacc_data['wacc'])

        # Driver-based sensitivity (growth × terminal margin)
        sensitivity_df = None
        if primary_dcf:
            net_debt = (info.get('totalDebt') or 0) - (info.get('totalCash') or 0)
            try:
                sensitivity_df = dcf_sensitivity_grid(
                    info, data['cashflow'], data['income_stmt'],
                    consensus_result['consensus_growth'], wacc_data['wacc'],
                    shares, net_debt, base_fcf_data, sbc_dilution, fx_to_usd=fx)
            except Exception:
                pass

        # Quality (Piotroski F-Score) + predictive signals
        quality = quality_score(info, data['cashflow'], data['balance_sheet'], data['income_stmt'])
        from signals import build_signal_profile
        signal_profile = build_signal_profile(data, quality)
        data['signal_profile'] = signal_profile

        all_valuations = {
            'dcf': primary_dcf,
            'comps': comps_result,
            'ddm': ddm_result,
        }

        # Validation
        validation_results = run_validation(data, all_valuations, pt_data, wacc_data, sector_mults)

        # Executable trade plan (regime-aware sizing)
        from trade_plan import build_trade_plan
        regime = _get_regime_cached()
        plan_cfg = {'regime_multiplier': regime.get('multiplier', 1.0)}
        trade_plan = build_trade_plan(data, pt_data, config=plan_cfg)
        data['trade_plan'] = trade_plan

        # Left-side / bottom-fishing analysis (reuses already-fetched data — no extra calls)
        bf_result = None
        try:
            from bottom_fishing import analyze_bottom_fish
            bf_result = analyze_bottom_fish(data, quality=quality, pt_data=pt_data,
                                            regime=regime)
        except Exception:
            bf_result = None

        # Persist snapshot for track-record / backtesting
        try:
            from snapshot import record_snapshot
            record_snapshot(data, pt_data)
        except Exception:
            pass

        # Render
        render_report(
            data=data, valuations=all_valuations, quality=quality,
            price_target_data=pt_data, wacc_data=wacc_data, macro=macro,
            sensitivity_df=sensitivity_df, validation=validation_results,
            output_path=output_path,
        )

        # Dashboard summary
        ff = (peer_model.get('football_field') or {})
        cg = consensus_result.get('consensus_growth') or {}
        rs = consensus_result.get('rec_summary') or {}
        cpe = consensus_result.get('consensus_pe') or {}

        return {
            'ticker': ticker,
            'status': 'OK',
            'name': info.get('shortName') or info.get('longName') or ticker,
            'sector': info.get('sector') or '-',
            'industry': info.get('industry') or '-',
            'current_price': round(current_price, 2),
            'market_cap_b': round((info.get('marketCap') or 0) / 1e9, 1),
            'recommendation': pt_data['recommendation'],
            'price_target': pt_data['price_target'],
            'upside': pt_data['upside'],
            'consensus_dcf_pt': round(primary_dcf['intrinsic_per_share'], 2) if primary_dcf else None,
            'consensus_pe_pt': cpe.get('implied_price'),
            'street_mean_pt': consensus_result.get('street_pt'),
            'peer_pe_base_pt': (ff.get('base') or {}).get('pt'),
            'wacc': round(wacc_data['wacc'] * 100, 2),
            'beta': round(beta, 2),
            'fwd_pe': info.get('forwardPE'),
            'pe_ttm': info.get('trailingPE'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'consensus_eps_y1': cg.get('eps_y1'),
            'consensus_eps_growth_y1': round(cg.get('eps_growth_y1', 0) * 100, 1) if cg.get('eps_growth_y1') else None,
            'consensus_rev_growth_y1': round(cg.get('rev_growth_y1', 0) * 100, 1) if cg.get('rev_growth_y1') else None,
            'rev_growth': round((info.get('revenueGrowth') or 0) * 100, 1),
            'gross_margin': round((info.get('grossMargins') or 0) * 100, 1),
            'op_margin': round((info.get('operatingMargins') or 0) * 100, 1),
            'roe': round((info.get('returnOnEquity') or 0) * 100, 1),
            'debt_to_equity': round((info.get('debtToEquity') or 0) / 100, 2),
            'quality_score': quality['score'],
            'quality_label': quality['label'],
            'data_quality_verdict': dq['verdict'],
            'data_quality_score': dq['score'],
            'data_quality_flags': [f"{f['check']}: {f['detail']}" for f in dq['flags']],
            'signal_conviction': signal_profile['conviction'],
            'signal_score': signal_profile['composite_score'],
            'signal_label': signal_profile['label'],
            'mom_12_1': signal_profile['components']['price_momentum'].get('raw_12_1'),
            'revision_score': signal_profile['components']['revision_momentum'].get('score'),
            'surprise_last_pct': signal_profile['components']['earnings_surprise'].get('last_surprise_pct'),
            'short_pct_float': signal_profile['components']['short_interest'].get('short_pct_float'),
            'short_score': signal_profile['components']['short_interest'].get('score'),
            'options_available': signal_profile['components']['options_skew'].get('available'),
            'plan_action': trade_plan.get('action'),
            'plan_mode': trade_plan.get('mode'),
            'plan_entry_low': trade_plan.get('entry_low'),
            'plan_entry_high': trade_plan.get('entry_high'),
            'plan_stop': trade_plan.get('stop') or trade_plan.get('holder_stop'),
            'plan_target1': trade_plan.get('target1'),
            'plan_rr': trade_plan.get('rr'),
            'plan_position_pct': trade_plan.get('position_pct'),
            'plan_shares': trade_plan.get('shares'),
            'street_bull_pct': rs.get('bull_pct'),
            'street_n_analysts': rs.get('total'),
            'street_view': rs.get('street_view'),
            'div_yield': round((info.get('dividendYield') or 0) * 100, 2),
            'report_file': os.path.basename(output_path),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'risk_free_rate': round(macro['risk_free_rate'] * 100, 2),
            'vix': round(macro['vix'], 1),
            'regime': regime.get('regime'),
            'regime_multiplier': regime.get('multiplier'),
            'regime_description': regime.get('description'),
            '_bf': bf_result,   # left-side analysis, extracted by main() then stripped
        }

    except Exception as e:
        return {
            'ticker': ticker,
            'status': 'ERROR',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def run_batch(tickers: list) -> list:
    global _REGIME_CACHE
    _REGIME_CACHE = None   # force fresh fetch at start of each batch
    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        ticker = ticker.upper().strip()
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] {ticker}")
        print('='*60)
        result = _run_single(ticker)
        if result['status'] == 'OK':
            print(f"  [done] {ticker}: {result['recommendation']} "
                  f"PT=${result['price_target']} ({result['upside']:+.1f}%)")
        else:
            print(f"  [fail] {ticker}: {result.get('error','')[:80]}")
        results.append(result)
    return results


def save_summary(results: list):
    existing = {}
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            existing = {r['ticker']: r for r in json.load(f)}
    for r in results:
        # never persist the heavy left-side payload into _summary.json
        existing[r['ticker']] = {k: v for k, v in r.items() if k != '_bf'}
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(existing.values()), f, indent=2, ensure_ascii=False)
    print(f"\n[ok] Summary saved -> {SUMMARY_FILE}")


def save_bottom_fishing(results: list):
    """Collect the per-name left-side analyses, write JSON + render the report."""
    bf = [r['_bf'] for r in results if r.get('_bf')]
    if not bf:
        return
    bf_path = os.path.join(REPORTS_DIR, 'bottom_fishing.json')
    try:
        with open(bf_path, 'w', encoding='utf-8') as f:
            json.dump(bf, f, ensure_ascii=False, indent=2, default=str)
        from bottom_fishing import render_report
        render_report(bf)
        n_strong = sum(1 for r in bf if r.get('score', {}).get('tier') == 'STRONG')
        n_spec = sum(1 for r in bf if r.get('score', {}).get('tier') == 'SPECULATIVE')
        print(f"[ok] Bottom-fishing: {n_strong} strong / {n_spec} speculative -> {bf_path}")
    except Exception as e:
        print(f"[warn] bottom-fishing save failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Batch equity research report generator')
    parser.add_argument('tickers', nargs='*')
    parser.add_argument('--watchlist', '-w', type=str)
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.watchlist:
        with open(args.watchlist) as f:
            tickers += [l.strip().upper() for l in f if l.strip() and not l.startswith('#')]
    if not tickers:
        parser.print_help(); sys.exit(1)

    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    print(f"\nBatch: {len(tickers)} tickers: {', '.join(tickers)}")

    results = run_batch(tickers)
    save_summary(results)
    save_bottom_fishing(results)

    from dashboard_generator import generate_dashboard
    dashboard_path = generate_dashboard()
    print(f"\n[ok] Dashboard -> {dashboard_path}")
    print(f"Open: file:///{dashboard_path.replace(os.sep, '/')}")

    ok = sum(1 for r in results if r['status'] == 'OK')
    print(f"\nDone: {ok}/{len(results)} success")


if __name__ == '__main__':
    main()
