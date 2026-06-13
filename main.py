"""
Buy-Side Equity Research Report Generator
Usage: python main.py <TICKER>
       python main.py AAPL
       python main.py MSFT --output reports/MSFT.html
       python main.py NVDA --validate
"""
import sys
import os
import argparse
import traceback

from data_fetcher import (
    fetch_company_data, get_macro_context, get_sector_multiples,
    get_country_risk_premium,
)
from valuation_engine import (
    calculate_wacc, comps_valuation, ddm_valuation, quality_score,
)
from consensus_model import run_consensus_model, dcf_sensitivity_grid
from report_generator import render_report
from validation import run_validation, print_validation_report
from peer_pe_model import run_peer_pe_model


def run_analysis(ticker_symbol: str, output_path: str = None,
                 show_validation: bool = False) -> str:
    ticker_symbol = ticker_symbol.upper().strip()
    output_path = output_path or f"{ticker_symbol}_equity_report.html"

    print("\n" + "="*60)
    print(f"  Buy-Side Equity Research -- {ticker_symbol}")
    print("="*60)

    # 1. Fetch data
    print("\n[1/5] Fetching market data & financials...")
    macro = get_macro_context()
    data = fetch_company_data(ticker_symbol)
    info = data['info']

    if not info or (not info.get('regularMarketPrice')
                    and not info.get('currentPrice')
                    and not info.get('previousClose')):
        raise ValueError(f"No data for '{ticker_symbol}'")

    current_price = (info.get('currentPrice') or info.get('regularMarketPrice')
                     or info.get('previousClose') or 0)
    sector = info.get('sector') or 'default'
    sector_mults = get_sector_multiples(sector)

    print(f"  [ok] {info.get('longName', ticker_symbol)} -- ${current_price:.2f}")
    print(f"  [ok] Sector: {sector} | Industry: {info.get('industry', '-')}")
    print(f"  [ok] RF: {macro['risk_free_rate']*100:.2f}% | VIX: {macro['vix']:.1f}")

    # 2. WACC (with country risk premium for foreign / ADR names)
    print("\n[2/6] Computing WACC...")
    beta = data['beta']
    crp = get_country_risk_premium(info.get('country'))
    wacc_data = calculate_wacc(info, beta=beta, risk_free_rate=macro['risk_free_rate'],
                               country_risk_premium=crp)
    print(f"  [ok] WACC: {wacc_data['wacc']*100:.2f}% | Beta: {beta:.2f}"
          f" | Country risk +{crp*100:.1f}%")

    # 3. Peer P/E model FIRST — its live peer multiples feed the comps below
    print("\n[3/6] Peer universe & live multiples...")
    peer_model = run_peer_pe_model(data)
    data['peer_model'] = peer_model
    peer_stats = peer_model.get('peer_pe_stats') or {}
    live_peer_multiples = {}
    if peer_stats.get('median'):
        live_peer_multiples['pe'] = peer_stats['median']  # live forward P/E median

    # 4. Consensus model (primary) — uses live peer multiples for comps
    print("\n[4/6] Running consensus-driven valuation...")
    consensus_result = run_consensus_model(data, wacc_data, sector_mults,
                                           live_peer_multiples=live_peer_multiples)
    pt_data = consensus_result['composite_pt']
    data['consensus_result'] = consensus_result

    # Data-quality gate — don't issue a rating on corrupted source data (e.g. SNDK)
    from data_quality import assess_data_quality, format_quality_banner
    dq = assess_data_quality(info, fx_to_usd=data.get('fx_to_usd', 1.0))
    data['data_quality'] = dq
    print("  " + format_quality_banner(dq))
    if not dq['reliable']:
        pt_data['rec_before_dq'] = pt_data['recommendation']
        pt_data['recommendation'] = 'NO RATING'
        print("  [!] Source data failed quality checks — rating suppressed.")

    shares = info.get('sharesOutstanding') or 1
    primary_dcf = consensus_result.get('consensus_dcf')
    base_fcf_data = consensus_result.get('base_fcf_data')
    sbc_dilution = consensus_result.get('sbc_dilution', 0.0)

    comps_result = consensus_result.get('comps') or comps_valuation(
        info, sector_mults, live_peer_multiples=live_peer_multiples,
        fx_to_usd=data.get('fx_to_usd', 1.0))
    ddm_result = ddm_valuation(info, wacc_data['wacc'])

    # Driver-based sensitivity (Stage-1 growth × terminal FCF margin)
    sensitivity_df = None
    if primary_dcf:
        net_debt = (info.get('totalDebt') or 0) - (info.get('totalCash') or 0)
        try:
            sensitivity_df = dcf_sensitivity_grid(
                info, data['cashflow'], data['income_stmt'],
                consensus_result['consensus_growth'], wacc_data['wacc'],
                shares, net_debt, base_fcf_data, sbc_dilution,
                fx_to_usd=data.get('fx_to_usd', 1.0))
        except Exception as e:
            print(f"  [warn] sensitivity grid failed: {e}")

    all_valuations = {'dcf': primary_dcf, 'comps': comps_result, 'ddm': ddm_result}

    # 5. Quality + predictive signals
    print("\n[5/6] Quality & predictive signals...")
    quality = quality_score(info, data['cashflow'], data['balance_sheet'], data['income_stmt'])
    print(f"  [ok] Quality (Piotroski F-Score): {quality['score']}/9 -- {quality['label']}")

    from signals import build_signal_profile
    signal_profile = build_signal_profile(data, quality)
    data['signal_profile'] = signal_profile
    print(f"  [ok] Signal conviction: {signal_profile['conviction']}/100 "
          f"({signal_profile['label']}) | composite {signal_profile['composite_score']:+.2f}")

    # 6. Validation
    print("\n[6/6] Validation...")
    validation_results = run_validation(data, all_valuations, pt_data, wacc_data, sector_mults)
    if show_validation:
        print_validation_report(ticker_symbol, validation_results, current_price)

    # Executable trade plan (action / entry / ATR stop / targets / risk-based size)
    from trade_plan import build_trade_plan, format_plan_console
    trade_plan = build_trade_plan(data, pt_data)
    data['trade_plan'] = trade_plan
    print(format_plan_console(trade_plan))

    # Persist a timestamped snapshot for track-record / backtesting
    try:
        from snapshot import record_snapshot
        record_snapshot(data, pt_data)
    except Exception as e:
        print(f"  [warn] snapshot not saved: {e}")

    print(f"\n  [**] RECOMMENDATION : {pt_data['recommendation']}")
    print(f"  [**] Price Target    : ${pt_data['price_target']:.2f}")
    print(f"  [**] Upside/Downside : {pt_data['upside']:+.1f}%")

    # Render
    print("\n[Rendering]...")
    render_report(
        data=data, valuations=all_valuations, quality=quality,
        price_target_data=pt_data, wacc_data=wacc_data, macro=macro,
        sensitivity_df=sensitivity_df, validation=validation_results,
        output_path=output_path,
    )

    print("\n" + "="*60)
    print(f"  Done: {os.path.abspath(output_path)}")
    print("="*60 + "\n")
    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description='Buy-Side Equity Research')
    parser.add_argument('ticker')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    try:
        path = run_analysis(args.ticker, args.output, args.validate)
        print(f"file:///{path.replace(os.sep, '/')}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
