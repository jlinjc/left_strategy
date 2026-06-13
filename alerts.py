"""
Alert Monitor — Daily Discipline Enforcement
────────────────────────────────────────────
Run every day (or before each session) to surface any trades requiring action.
This is what makes the stop-loss discipline REAL: without daily checking,
"跌破停損出場" is just text in a report.

Alert types:
  STOP_BREACH     — position price <= stop → EXIT NOW (critical)
  NEAR_STOP       — price within 3% above stop → WARNING, prepare to exit
  CONVICTION_EXIT — signal conviction < 35 → EXIT (rule-based, no discretion)
  CONVICTION_TRIM — conviction 35-45 → reduce by 50%
  ENTRY_ZONE      — watchlist name is inside its computed entry band → ACT
  EARNINGS_SOON   — earnings in next 7 days → review thesis before
  REGIME_CHANGE   — market regime changed since last check → adjust all sizes
  DATA_DEGRADE    — data quality downgraded → pause new adds

Reads: reports/portfolio.json + reports/_summary.json + reports/regime_state.json
Writes: reports/alerts.json (dashboard integration)

Usage:
  python alerts.py              # run full check
  python alerts.py --no-regime  # skip regime check (faster)
  python alerts.py --watchlist  # also check non-position names in _summary.json
"""
from __future__ import annotations
import json
import os
import argparse
from datetime import datetime, timezone
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR       = os.path.join(os.path.dirname(__file__), 'reports')
PORTFOLIO_FILE    = os.path.join(REPORTS_DIR, 'portfolio.json')
SUMMARY_FILE      = os.path.join(REPORTS_DIR, '_summary.json')
ALERTS_FILE       = os.path.join(REPORTS_DIR, 'alerts.json')
REGIME_STATE_FILE = os.path.join(REPORTS_DIR, 'regime_state.json')
os.makedirs(REPORTS_DIR, exist_ok=True)

SEV_ORDER = {'critical': 0, 'warning': 1, 'opportunity': 2, 'info': 3}


def _load(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _price(ticker: str) -> float | None:
    try:
        fi = yf.Ticker(ticker).fast_info
        p = getattr(fi, 'last_price', None) or getattr(fi, 'regular_market_price', None)
        return round(float(p), 2) if p else None
    except Exception:
        return None


# ─────────────────────────────────────────────
def check_positions(portfolio: dict, summary: dict) -> list:
    """Check each open position for stop breaches, conviction drops, earnings."""
    alerts = []
    positions = (portfolio or {}).get('positions', {})
    sum_map   = {r.get('ticker'): r for r in (summary or []) if r.get('status') == 'OK'}

    for ticker, pos in positions.items():
        cp = _price(ticker)
        if cp is None:
            alerts.append({
                'type': 'DATA', 'ticker': ticker, 'severity': 'info',
                'message': f'{ticker}: Could not fetch current price'
            })
            continue

        stop  = pos.get('stop')
        entry = pos.get('entry_price', cp)

        # STOP BREACH
        if stop and cp <= stop:
            loss = round((cp - entry) * pos.get('shares', 0), 0)
            alerts.append({
                'type': 'STOP_BREACH', 'ticker': ticker, 'severity': 'critical',
                'message': (f'{ticker}: *** STOP BREACHED *** price ${cp} <= stop ${stop} | '
                            f'Loss: ${loss:+,.0f} | EXIT NOW — no discretion'),
                'current_price': cp, 'stop': stop, 'pnl': loss,
            })
        # NEAR STOP (within 3%)
        elif stop and cp <= stop * 1.03:
            gap_pct = round((cp / stop - 1) * 100, 1)
            alerts.append({
                'type': 'NEAR_STOP', 'ticker': ticker, 'severity': 'warning',
                'message': (f'{ticker}: Near stop — price ${cp} vs stop ${stop} '
                            f'({gap_pct:+.1f}%). Prepare exit or tighten stop.'),
                'current_price': cp, 'stop': stop,
            })

        # CONVICTION SIGNALS from latest analysis
        sr = sum_map.get(ticker)
        if sr:
            conv = sr.get('signal_conviction')
            if conv is not None:
                if conv < 35:
                    alerts.append({
                        'type': 'CONVICTION_EXIT', 'ticker': ticker, 'severity': 'critical',
                        'message': f'{ticker}: Conviction {conv} < 35 — EXIT POSITION (rule-based)',
                        'conviction': conv,
                    })
                elif conv < 45:
                    alerts.append({
                        'type': 'CONVICTION_TRIM', 'ticker': ticker, 'severity': 'warning',
                        'message': f'{ticker}: Conviction {conv} < 45 — REDUCE by 50% (rule-based)',
                        'conviction': conv,
                    })

            # DATA QUALITY DOWNGRADE
            dq = sr.get('data_quality_verdict')
            if dq in ('REVIEW', 'UNRELIABLE'):
                alerts.append({
                    'type': 'DATA_DEGRADE', 'ticker': ticker, 'severity': 'warning',
                    'message': f'{ticker}: Data quality = {dq} — pause new adds, verify inputs',
                    'data_quality': dq,
                })

        # EARNINGS WITHIN 7 DAYS
        try:
            info = yf.Ticker(ticker).info
            et   = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
            if et:
                ed   = datetime.fromtimestamp(float(et), tz=timezone.utc)
                days = (ed - datetime.now(tz=timezone.utc)).days
                if 0 <= days <= 7:
                    alerts.append({
                        'type': 'EARNINGS_SOON', 'ticker': ticker, 'severity': 'info',
                        'message': (f'{ticker}: Earnings in {days} day(s) ({ed.strftime("%Y-%m-%d")}) — '
                                    f'review thesis, consider sizing down before print'),
                        'days_to_earnings': days,
                    })
        except Exception:
            pass

    return alerts


# ─────────────────────────────────────────────
def check_entry_zones(summary: dict) -> list:
    """Check watchlist / coverage names for entry zone triggers."""
    alerts = []
    portfolio = (_load(PORTFOLIO_FILE) or {}).get('positions', {})

    for r in (summary or []):
        if r.get('status') != 'OK':
            continue
        ticker = r.get('ticker')
        if ticker in portfolio:
            continue  # already in portfolio, skip
        elo  = r.get('plan_entry_low')
        ehi  = r.get('plan_entry_high')
        mode = r.get('plan_mode')
        if not elo or not ehi or mode not in ('value', 'momentum'):
            continue
        cp = _price(ticker)
        if cp and elo <= cp <= ehi:
            conv  = r.get('signal_conviction', '?')
            act   = r.get('plan_action', '?')
            size  = r.get('plan_position_pct', '?')
            alerts.append({
                'type': 'ENTRY_ZONE', 'ticker': ticker, 'severity': 'opportunity',
                'message': (f'{ticker}: IN ENTRY ZONE — ${cp} in [${elo}-${ehi}] | '
                            f'Action: {act} | Conviction: {conv} | Size: {size}%'),
                'current_price': cp, 'entry_low': elo, 'entry_high': ehi,
                'conviction': conv, 'suggested_action': act,
            })
    return alerts


# ─────────────────────────────────────────────
def check_regime(current_regime: dict | None = None) -> dict | None:
    """Detect regime change vs last saved state."""
    prior = _load(REGIME_STATE_FILE)

    if current_regime is None:
        try:
            from regime import get_regime
            current_regime = get_regime()
        except Exception:
            return None

    if prior and prior.get('regime') != current_regime.get('regime'):
        old_mult = prior.get('multiplier', 1.0)
        new_mult = current_regime.get('multiplier', 1.0)
        direction = 'DETERIORATED' if new_mult < old_mult else 'IMPROVED'
        sev = 'warning' if new_mult < old_mult else 'info'
        msg = (f"REGIME {direction}: {prior['regime']} -> {current_regime['regime']} "
               f"(sizing x{old_mult:.0%} -> x{new_mult:.0%})")
        if new_mult < old_mult:
            msg += f" | Reduce ALL new position sizes to {new_mult:.0%} of plan"
        else:
            msg += " | Normal position sizing restored"
        return {
            'type': 'REGIME_CHANGE', 'severity': sev, 'message': msg,
            'old_regime': prior['regime'], 'new_regime': current_regime['regime'],
            'old_mult': old_mult, 'new_mult': new_mult,
        }
    return None


# ─────────────────────────────────────────────
def run_alerts(check_regime_flag: bool = True,
               check_watchlist: bool = True) -> list:
    portfolio = _load(PORTFOLIO_FILE)
    summary   = _load(SUMMARY_FILE)
    all_alerts = []

    # Regime first (portfolio-wide impact)
    if check_regime_flag:
        try:
            from regime import get_regime
            regime = get_regime()
            rc = check_regime(regime)
            if rc:
                all_alerts.append(rc)
        except Exception as e:
            all_alerts.append({'type': 'REGIME_ERROR', 'severity': 'info',
                               'message': f'Could not check regime: {e}'})

    # Position-level checks
    all_alerts += check_positions(portfolio, summary)

    # Entry zone / watchlist
    if check_watchlist:
        all_alerts += check_entry_zones(summary)

    # Sort by severity
    all_alerts.sort(key=lambda x: SEV_ORDER.get(x.get('severity', 'info'), 4))

    # Save to JSON for dashboard integration
    out = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'n_critical':    sum(1 for a in all_alerts if a.get('severity') == 'critical'),
        'n_warnings':    sum(1 for a in all_alerts if a.get('severity') == 'warning'),
        'n_opportunities': sum(1 for a in all_alerts if a.get('severity') == 'opportunity'),
        'alerts': all_alerts,
    }
    with open(ALERTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return all_alerts


def print_alerts(alerts: list):
    SEV_LABEL = {
        'critical':    '[!!!] CRITICAL',
        'warning':     '[ ! ] WARNING',
        'opportunity': '[ + ] OPPORTUNITY',
        'info':        '[ i ] INFO',
    }
    nc = sum(1 for a in alerts if a.get('severity') == 'critical')
    nw = sum(1 for a in alerts if a.get('severity') == 'warning')
    no = sum(1 for a in alerts if a.get('severity') == 'opportunity')

    print(f"\n{'='*72}")
    print(f"  ALERTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Critical: {nc}  |  Warnings: {nw}  |  Opportunities: {no}")
    print('='*72)

    if not alerts:
        print("  No alerts — all clear.\n")
    else:
        for a in alerts:
            lbl = SEV_LABEL.get(a.get('severity', 'info'), '[ i ]')
            print(f"\n  {lbl} [{a.get('type','?')}]")
            print(f"  {a['message']}")

    print('='*72)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alert monitor')
    parser.add_argument('--no-regime',   action='store_true', help='Skip regime check')
    parser.add_argument('--no-watchlist', action='store_true', help='Skip watchlist entry check')
    args = parser.parse_args()

    alerts = run_alerts(
        check_regime_flag=not args.no_regime,
        check_watchlist=not args.no_watchlist,
    )
    print_alerts(alerts)
