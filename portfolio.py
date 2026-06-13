"""
Portfolio Manager
─────────────────
Tracks live positions with entry, stop, target, P&L, and computes portfolio-level
risk: total exposure, sector concentration, correlation matrix, and a Bear scenario
stress test (aggregate loss if all positions hit their Bear intrinsic value).

State persists in: reports/portfolio.json

This is the 組合層視角 missing piece: each individual trade is fine, but $1M across
6 correlated semiconductor names is NOT 6 independent 1% risks — it's one large bet.
Portfolio.py surfaces that reality before you're surprised by it.

Usage:
  python portfolio.py                          # show current holdings + P&L
  python portfolio.py --update                 # refresh current prices
  python portfolio.py --add TSM 15 427 394 493 Technology
  python portfolio.py --close TSM 450 "Target hit"
  python portfolio.py --corr                  # correlation matrix
  python portfolio.py --stress                # Bear scenario stress test
  python portfolio.py --account 1000000       # specify account size

  from portfolio import Portfolio
  p = Portfolio(account_size=1_000_000)
  p.add_position('TSM', 15, 427, 394, 493, 'Technology', 'momentum', 89)
"""
from __future__ import annotations
import json
import os
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

REPORTS_DIR     = os.path.join(os.path.dirname(__file__), 'reports')
PORTFOLIO_FILE  = os.path.join(REPORTS_DIR, 'portfolio.json')
SUMMARY_FILE    = os.path.join(REPORTS_DIR, '_summary.json')
os.makedirs(REPORTS_DIR, exist_ok=True)

SECTOR_LIMIT        = 0.30   # max sector concentration
SINGLE_NAME_LIMIT   = 0.20   # max single name
TOTAL_EXPOSURE_LIMIT = 0.90  # keep at least 10% cash


class Portfolio:
    def __init__(self, file: str = PORTFOLIO_FILE, account_size: float = 1_000_000.0):
        self.file = file
        self.account_size = account_size
        self._data = self._load()
        if self._data.get('account_size'):
            self.account_size = self._data['account_size']

    # ── Persistence ────────────────────────────────────────────────────
    def _load(self) -> dict:
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'account_size': self.account_size, 'positions': {}, 'closed': []}

    def _save(self):
        self._data['account_size'] = self.account_size
        with open(self.file, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    @property
    def positions(self) -> dict:
        return self._data.setdefault('positions', {})

    @property
    def closed(self) -> list:
        return self._data.setdefault('closed', [])

    # ── Add / Close ─────────────────────────────────────────────────────
    def add_position(self, ticker: str, shares: int, entry_price: float,
                     stop: float, target1: float, target2: float = None,
                     sector: str = 'Unknown', plan_mode: str = 'momentum',
                     conviction: int = 50, notes: str = '') -> dict:
        pos_value    = shares * entry_price
        pos_pct      = pos_value / self.account_size
        risk_per_sh  = entry_price - stop
        dollar_risk  = shares * risk_per_sh
        rr_estimate  = (target1 - entry_price) / risk_per_sh if risk_per_sh > 0 else None

        warns = []
        # Sector concentration
        sector_val = sum(
            p['shares'] * p['entry_price']
            for p in self.positions.values() if p.get('sector') == sector
        )
        if (sector_val + pos_value) / self.account_size > SECTOR_LIMIT:
            warns.append(f"SECTOR: {sector} would be >{SECTOR_LIMIT*100:.0f}% of portfolio")
        # Single-name limit
        if pos_pct > SINGLE_NAME_LIMIT:
            warns.append(f"SIZE: {ticker} = {pos_pct*100:.1f}% > {SINGLE_NAME_LIMIT*100:.0f}% limit")
        # Total exposure
        deployed = sum(p['shares'] * p['entry_price'] for p in self.positions.values())
        if (deployed + pos_value) / self.account_size > TOTAL_EXPOSURE_LIMIT:
            warns.append(f"EXPOSURE: total would be >{TOTAL_EXPOSURE_LIMIT*100:.0f}%")

        pos = {
            'ticker': ticker, 'shares': shares,
            'entry_price': round(entry_price, 2), 'stop': round(stop, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2) if target2 else None,
            'sector': sector, 'plan_mode': plan_mode,
            'conviction': conviction, 'notes': notes,
            'pos_value':  round(pos_value, 0),
            'pos_pct':    round(pos_pct * 100, 1),
            'dollar_risk': round(dollar_risk, 0),
            'rr_estimate': round(rr_estimate, 2) if rr_estimate else None,
            'date_opened': datetime.now().strftime('%Y-%m-%d'),
            'current_price': None, 'unrealized_pnl': None, 'unrealized_pct': None,
        }
        if warns:
            pos['warnings'] = warns
        self.positions[ticker] = pos
        self._save()
        return {'position': pos, 'warnings': warns}

    def close_position(self, ticker: str, exit_price: float, reason: str = '') -> dict:
        if ticker not in self.positions:
            return {'error': f'{ticker} not in portfolio'}
        pos = self.positions.pop(ticker)
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        pos.update({
            'exit_price': round(exit_price, 2),
            'pnl': round(pnl, 0),
            'pnl_pct': round((exit_price / pos['entry_price'] - 1) * 100, 2),
            'date_closed': datetime.now().strftime('%Y-%m-%d'),
            'close_reason': reason,
        })
        self.closed.append(pos)
        self._save()
        return pos

    # ── Price update ────────────────────────────────────────────────────
    def update_prices(self) -> dict:
        tickers = list(self.positions.keys())
        if not tickers:
            return {}
        prices = {}
        for t in tickers:
            try:
                fi = yf.Ticker(t).fast_info
                p = (getattr(fi, 'last_price', None)
                     or getattr(fi, 'regular_market_price', None))
                if p:
                    prices[t] = round(float(p), 2)
            except Exception:
                pass
        for ticker, pos in self.positions.items():
            if ticker in prices:
                cp = prices[ticker]
                pos['current_price']  = cp
                pos['unrealized_pnl'] = round((cp - pos['entry_price']) * pos['shares'], 0)
                pos['unrealized_pct'] = round((cp / pos['entry_price'] - 1) * 100, 2)
        self._save()
        return prices

    # ── Analytics ───────────────────────────────────────────────────────
    def get_summary(self) -> dict:
        if not self.positions:
            return {'n_positions': 0, 'total_deployed': 0, 'total_pnl': 0,
                    'deployed_pct': 0, 'cash_pct': 100,
                    'sector_pct': {}, 'warnings': [], 'total_risk_dollar': 0}

        deployed  = sum(p['shares'] * p['entry_price'] for p in self.positions.values())
        total_pnl = sum(p.get('unrealized_pnl') or 0 for p in self.positions.values())
        total_risk = sum(p.get('dollar_risk') or 0 for p in self.positions.values())

        sec_val = {}
        for pos in self.positions.values():
            s = pos.get('sector', 'Unknown')
            sec_val[s] = sec_val.get(s, 0) + pos['shares'] * pos['entry_price']
        sec_pct = {s: round(v / self.account_size * 100, 1) for s, v in sec_val.items()}

        warns = []
        for sec, pct in sec_pct.items():
            if pct > SECTOR_LIMIT * 100:
                warns.append(f"SECTOR CONCENTRATION: {sec} = {pct:.1f}% (>{SECTOR_LIMIT*100:.0f}%)")
        if deployed / self.account_size > TOTAL_EXPOSURE_LIMIT:
            warns.append(f"HIGH EXPOSURE: {deployed/self.account_size*100:.1f}% deployed")

        return {
            'n_positions':      len(self.positions),
            'total_deployed':   round(deployed, 0),
            'deployed_pct':     round(deployed / self.account_size * 100, 1),
            'cash_pct':         round((1 - deployed / self.account_size) * 100, 1),
            'total_pnl':        round(total_pnl, 0),
            'total_risk_dollar': round(total_risk, 0),
            'total_risk_pct':   round(total_risk / self.account_size * 100, 2),
            'sector_pct':       sec_pct,
            'warnings':         warns,
        }

    def get_correlation_matrix(self, period: str = '1y') -> pd.DataFrame | None:
        tickers = list(self.positions.keys())
        if len(tickers) < 2:
            return None
        try:
            raw = yf.download(tickers, period=period, interval='1d',
                              auto_adjust=True, progress=False)
            close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close.pct_change().dropna().corr().round(2)
        except Exception:
            return None

    def stress_test(self) -> dict:
        """
        Bear scenario: for each position, find Bear intrinsic value from _summary.json.
        Returns aggregate loss if all positions simultaneously hit their Bear PT.
        """
        if not os.path.exists(SUMMARY_FILE):
            return {'error': 'No _summary.json found — run batch_run.py first'}

        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        sum_map = {r.get('ticker'): r for r in summary if r.get('status') == 'OK'}

        results = {}
        total_bear_loss = 0.0
        for ticker, pos in self.positions.items():
            sr = sum_map.get(ticker, {})
            entry = pos['entry_price']
            shares = pos['shares']
            bear_pt = (sr.get('consensus_dcf_scenarios') or {})
            # Try to get bear PT from summary (may not be present directly)
            # Fall back to PT * 0.70 as a rough stress
            pt = sr.get('price_target')
            if pt:
                bear_est = round(pt * 0.70, 2)
            else:
                bear_est = round(entry * 0.75, 2)
            bear_loss = (bear_est - entry) * shares
            total_bear_loss += bear_loss
            results[ticker] = {
                'entry': entry, 'shares': shares,
                'bear_estimate': bear_est,
                'bear_loss': round(bear_loss, 0),
                'bear_loss_pct': round((bear_est / entry - 1) * 100, 1),
            }

        return {
            'positions': results,
            'total_bear_loss': round(total_bear_loss, 0),
            'total_bear_loss_pct_account': round(total_bear_loss / self.account_size * 100, 2),
            'note': 'Bear = PT x 0.70 approximation. Use full report Bear DCF for exact values.',
        }

    def get_closed_stats(self) -> dict:
        cl = self.closed
        if not cl:
            return {'n_trades': 0}
        pnls = [t.get('pnl', 0) for t in cl]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            'n_trades': len(cl),
            'total_pnl': round(sum(pnls), 0),
            'win_rate': round(len(wins) / len(pnls), 3) if pnls else 0,
            'avg_win': round(sum(wins) / len(wins), 0) if wins else 0,
            'avg_loss': round(sum(losses) / len(losses), 0) if losses else 0,
            'profit_factor': round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else None,
        }

    # ── Display ─────────────────────────────────────────────────────────
    def print_summary(self, show_stress: bool = False):
        summ  = self.get_summary()
        stats = self.get_closed_stats()

        print(f"\n{'='*76}")
        print(f"  PORTFOLIO  ({summ['n_positions']} positions | "
              f"${self.account_size:,.0f} account)")
        print('='*76)

        if not self.positions:
            print("  No open positions."); print('='*76); return

        print(f"  Deployed: ${summ['total_deployed']:,.0f} ({summ['deployed_pct']:.1f}%)  "
              f"Cash: {summ['cash_pct']:.1f}%  "
              f"P&L: ${summ['total_pnl']:+,.0f}  "
              f"Total Risk: ${summ['total_risk_dollar']:,.0f} ({summ['total_risk_pct']:.2f}%)")

        # Sector breakdown
        print("\n  Sector Exposure:")
        for sec, pct in sorted(summ['sector_pct'].items(), key=lambda x: -x[1]):
            flag = '  *** OVER LIMIT' if pct > SECTOR_LIMIT * 100 else ''
            bar = '█' * int(pct / 5)
            print(f"    {sec:<28} {pct:>5.1f}%  {bar}{flag}")

        # Positions table
        print(f"\n  {'Ticker':<8} {'Shrs':>5} {'Entry':>8} {'Stop':>8} "
              f"{'Target':>8} {'Cur':>8} {'P&L':>9} {'%':>7} {'Mode'}")
        print("  " + "-"*74)
        for tk, p in sorted(self.positions.items()):
            cp    = p.get('current_price')
            pnl   = p.get('unrealized_pnl')
            pnlp  = p.get('unrealized_pct')
            cp_s  = f"${cp:.2f}"  if cp   is not None else "—"
            pnl_s = f"${pnl:+,.0f}" if pnl  is not None else "—"
            pnlp_s = f"{pnlp:+.1f}%" if pnlp is not None else "—"
            stop_flag = ' *** BREACH' if cp and p.get('stop') and cp <= p['stop'] else ''
            print(f"  {tk:<8} {p['shares']:>5} "
                  f"${p['entry_price']:>7.2f} "
                  f"${p['stop']:>7.2f} "
                  f"${p['target1']:>7.2f} "
                  f"{cp_s:>8} "
                  f"{pnl_s:>9} "
                  f"{pnlp_s:>7} "
                  f"{p.get('plan_mode','')}{stop_flag}")

        # Warnings
        if summ['warnings']:
            print("\n  WARNINGS:")
            for w in summ['warnings']:
                print(f"    *** {w}")

        # Closed trade stats
        if stats['n_trades'] > 0:
            pf = stats.get('profit_factor')
            pf_str = f"{pf:.2f}" if pf else "—"
            print(f"\n  Closed: {stats['n_trades']} trades  "
                  f"Win rate: {stats['win_rate']*100:.1f}%  "
                  f"Total P&L: ${stats['total_pnl']:+,.0f}  "
                  f"PF: {pf_str}")

        # Stress test
        if show_stress:
            st = self.stress_test()
            if st.get('error'):
                print(f"\n  Stress test: {st['error']}")
            else:
                print(f"\n  Bear Stress Test (all positions hit Bear PT):")
                for tk, r in st['positions'].items():
                    print(f"    {tk:<8} Bear est: ${r['bear_estimate']}  "
                          f"Loss: ${r['bear_loss']:+,.0f} ({r['bear_loss_pct']:+.1f}%)")
                print(f"    TOTAL BEAR LOSS: ${st['total_bear_loss']:+,.0f} "
                      f"({st['total_bear_loss_pct_account']:+.2f}% of account)  "
                      f"[{st['note']}]")

        print('='*76)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio tracker')
    parser.add_argument('--account', type=float, default=1_000_000.0)
    parser.add_argument('--add',    nargs='+', metavar='ARG',
                        help='TICKER SHARES ENTRY STOP TARGET1 [TARGET2] [SECTOR]')
    parser.add_argument('--close',  nargs='+', metavar='ARG',
                        help='TICKER EXIT_PRICE [REASON...]')
    parser.add_argument('--corr',   action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--stress', action='store_true')
    args = parser.parse_args()

    p = Portfolio(account_size=args.account)

    if args.add:
        a = args.add
        ticker = a[0].upper()
        shares = int(a[1])
        entry  = float(a[2])
        stop   = float(a[3])
        t1     = float(a[4]) if len(a) > 4 else round(entry * 1.15, 2)
        t2     = float(a[5]) if len(a) > 5 and a[5].replace('.','').isdigit() else None
        sector = a[6] if len(a) > 6 else (a[5] if len(a) > 5 and not a[5].replace('.','').isdigit() else 'Unknown')
        res = p.add_position(ticker, shares, entry, stop, t1, t2, sector=sector)
        print(f"  [ok] Added {ticker}: {shares} shares @ ${entry}")
        for w in res.get('warnings', []):
            print(f"  [warn] {w}")

    if args.close:
        c = args.close
        ticker = c[0].upper()
        exit_p = float(c[1])
        reason = ' '.join(c[2:]) if len(c) > 2 else 'Manual close'
        res = p.close_position(ticker, exit_p, reason)
        print(f"  [ok] Closed {ticker} @ ${exit_p} | P&L: ${res.get('pnl', 0):+,.0f}")

    if args.update:
        prices = p.update_prices()
        print(f"  Updated {len(prices)} prices")

    if args.corr:
        corr = p.get_correlation_matrix()
        if corr is not None:
            print("\n  Correlation Matrix (1y daily returns):")
            print(corr.to_string())
        else:
            print("  Need >=2 positions for correlation matrix")

    p.print_summary(show_stress=args.stress)
