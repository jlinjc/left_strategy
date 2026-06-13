"""
Unified Dashboard Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-page tabbed dashboard:
  Tab 1 — Overview   : regime · alerts · portfolio · KPI cards · value×momentum matrix
  Tab 2 — Coverage   : sortable/filterable table + stock detail modal (no new tab)
  Tab 3 — Backtest   : inline Chart.js equity curve + annual bars + drawdown chart

Sources:
  reports/_summary.json            (batch_run.py output)
  reports/regime_state.json        (regime.py)
  reports/alerts.json              (alerts.py)
  reports/portfolio.json           (portfolio.py)
  reports/strategy_backtest.json   (strategy_backtest.py stats)
  reports/strategy_backtest_curves.json  (strategy_backtest.py curves)
"""
import os
import json
from datetime import datetime

REPORTS_DIR    = os.path.join(os.path.dirname(__file__), 'reports')
SUMMARY_FILE   = os.path.join(REPORTS_DIR, '_summary.json')
DASHBOARD_FILE = os.path.join(REPORTS_DIR, 'dashboard.html')
PORTFOLIO_FILE = os.path.join(REPORTS_DIR, 'portfolio.json')
ALERTS_FILE    = os.path.join(REPORTS_DIR, 'alerts.json')
REGIME_FILE    = os.path.join(REPORTS_DIR, 'regime_state.json')
BACKTEST_FILE  = os.path.join(REPORTS_DIR, 'strategy_backtest.json')
CURVES_FILE    = os.path.join(REPORTS_DIR, 'strategy_backtest_curves.json')
BOTTOMFISH_FILE = os.path.join(REPORTS_DIR, 'bottom_fishing.json')
BF_BACKTEST_FILE = os.path.join(REPORTS_DIR, 'bottom_fishing_backtest.json')


# ─── colour helpers ───────────────────────────────────────────────────────────

def _rec_color(rec):
    return {'BUY': '#1a7a4a', 'HOLD': '#d4900a', 'SELL': '#b52a2a',
            'NO RATING': '#6b7280'}.get(rec, '#6b7280')

def _upside_color(u):
    if u is None: return '#6b7280'
    return '#1a7a4a' if u > 10 else '#d4900a' if u > -5 else '#b52a2a'

def _conv_color(c):
    if c is None: return '#6b7280'
    return '#1a7a4a' if c >= 65 else '#d4900a' if c >= 45 else '#b52a2a'

def _dq_color(v):
    return {'OK': '#1a7a4a', 'REVIEW': '#d4900a', 'UNRELIABLE': '#b52a2a'}.get(v, '#6b7280')

def _q_color(label):
    return {'Strong': '#1a7a4a', 'Average': '#d4900a', 'Weak': '#b52a2a'}.get(label, '#6b7280')

def _load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ─── quadrant logic ───────────────────────────────────────────────────────────

QUADRANTS = {
    'cheap_improving': ('Cheap + Improving', '#1a7a4a',
                        'Undervalued AND signals turning up — highest-conviction long'),
    'value_trap':      ('Value-Trap Risk',   '#d4900a',
                        'Cheap but signals weak — may stay cheap; wait for a catalyst'),
    'momentum_growth': ('Momentum / Growth',  '#1a6fa8',
                        'Expensive but strongly bid — momentum can persist, valuation risk'),
    'avoid':           ('Avoid',              '#b52a2a',
                        'Overvalued AND signals rolling over — highest-conviction avoid/short'),
    'neutral':         ('Mixed / Neutral',    '#6b7280',
                        'No strong edge from the value × momentum intersection'),
}

def _quadrant(upside, sig):
    if upside is None or sig is None:
        return 'neutral'
    cheap, expensive = upside > 10, upside < -10
    pos, neg = sig > 0.15, sig < -0.15
    if cheap and pos:     return 'cheap_improving'
    if cheap and neg:     return 'value_trap'
    if expensive and pos: return 'momentum_growth'
    if expensive and neg: return 'avoid'
    return 'neutral'


# ─── HTML section builders ────────────────────────────────────────────────────

def _regime_html(regime):
    if not regime:
        return '<div style="background:#f5f5f5;border-left:5px solid #9ca3af;padding:10px 20px;margin-bottom:14px;border-radius:4px;color:#9ca3af;font-size:8.5pt">Regime data not available — run <code>python regime.py</code></div>'
    label  = regime.get('regime', '—')
    mult   = regime.get('multiplier', 1.0)
    vix    = regime.get('vix', '—')
    spy_p  = regime.get('spy_price', '—')
    spy_d  = regime.get('spy_200dma', '—')
    spy_v  = regime.get('spy_vs_200_pct', 0) or 0
    color  = regime.get('color', '#6b7280')
    desc   = regime.get('description', '')
    as_of  = regime.get('as_of', '')
    arrow  = '▲' if regime.get('spy_above_200') else '▼'
    filled = int(mult * 10)
    mult_bar = '█' * filled + '░' * (10 - filled)
    bg = {'RISK ON': '#e8f5e9', 'CAUTION': '#fff8e1', 'RISK OFF': '#fce4ec'}.get(label, '#f5f5f5')
    spy_v_color = '#27ae60' if spy_v >= 0 else '#e74c3c'
    return f'''
<div class="strip" style="background:{bg};border-left-color:{color}">
  <div style="font-size:13pt;font-weight:700;color:{color};min-width:90px">{label}</div>
  <div class="strip-item">Sizing multiplier: <strong style="color:{color}">×{mult:.0%}</strong>
    <span style="font-family:monospace;color:{color};font-size:9pt">[{mult_bar}]</span></div>
  <div class="strip-item">SPY <strong>${spy_p}</strong> {arrow} 200dma <strong>${spy_d}</strong>
    (<span style="color:{spy_v_color}">{spy_v:+.1f}%</span>)</div>
  <div class="strip-item">VIX <strong>{vix}</strong></div>
  <div style="font-size:7.8pt;color:#6b7280;flex:1">{desc}</div>
  <div style="font-size:7pt;color:#9ca3af">{as_of}</div>
</div>'''


def _alerts_html(alerts_data):
    if not alerts_data or not alerts_data.get('alerts'):
        return ''
    nc  = alerts_data.get('n_critical', 0)
    nw  = alerts_data.get('n_warnings', 0)
    no  = alerts_data.get('n_opportunities', 0)
    ts  = alerts_data.get('generated_at', '')
    bg  = '#fce4ec' if nc else '#fff8e1' if nw else '#e8f5e9'
    bdr = '#e74c3c' if nc else '#e67e22' if nw else '#27ae60'

    badges = ''
    if nc: badges += f'<span class="alert-chip" style="background:#e74c3c">{nc} Critical</span>'
    if nw: badges += f'<span class="alert-chip" style="background:#e67e22">{nw} Warning</span>'
    if no: badges += f'<span class="alert-chip" style="background:#27ae60">{no} Entry</span>'
    if not badges: badges = '<span style="font-size:8pt;color:#9ca3af">No alerts</span>'

    msgs = ''
    ICONS = {'critical': '🚨', 'warning': '⚠', 'opportunity': '✓', 'info': 'i'}
    COLS  = {'critical': '#e74c3c', 'warning': '#e67e22', 'opportunity': '#27ae60', 'info': '#6b7280'}
    for a in alerts_data['alerts'][:6]:
        sev = a.get('severity', 'info')
        msgs += (f'<div style="font-size:7.8pt;color:{COLS.get(sev,"#6b7280")};margin-bottom:2px">'
                 f'<strong>{ICONS.get(sev,"·")}</strong> {a["message"][:110]}</div>')

    return f'''
<div class="strip" style="background:{bg};border-left-color:{bdr};flex-direction:column;gap:6px">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span style="font-size:9pt;font-weight:700;color:#374151">Alerts</span>
    {badges}
    <span style="font-size:7pt;color:#9ca3af;margin-left:auto">as of {ts} &nbsp;|&nbsp; run <code>python alerts.py</code> to refresh</span>
  </div>
  {msgs}
</div>'''


def _portfolio_html(portfolio):
    if not portfolio or not portfolio.get('positions'):
        return ''
    positions = portfolio['positions']
    account   = portfolio.get('account_size', 1_000_000)
    deployed  = sum(p['shares'] * p['entry_price'] for p in positions.values())
    pnl       = sum(p.get('unrealized_pnl') or 0 for p in positions.values())
    dep_pct   = deployed / account * 100
    cash_pct  = 100 - dep_pct
    pnl_c     = '#27ae60' if pnl >= 0 else '#e74c3c'

    sec_val = {}
    for p in positions.values():
        s = p.get('sector', 'Unknown')
        sec_val[s] = sec_val.get(s, 0) + p['shares'] * p['entry_price']
    sec_pct = {s: v / account * 100 for s, v in sec_val.items()}

    SEC_COLORS = ['#0d3b6e','#1a6fa8','#27ae60','#d4900a','#e74c3c','#8b5cf6','#ec4899']
    sec_bars = ''
    for i, (sec, pct) in enumerate(sorted(sec_pct.items(), key=lambda x: -x[1])):
        c = SEC_COLORS[i % len(SEC_COLORS)]
        sec_bars += (f'<span style="display:inline-block;margin-right:10px;font-size:7.5pt">'
                     f'<span style="background:{c};color:white;padding:2px 7px;border-radius:10px">{sec[:14]}</span>'
                     f' <strong>{pct:.1f}%</strong></span>')

    pos_chips = ''
    for tk, pos in sorted(positions.items()):
        cp = pos.get('current_price', pos['entry_price'])
        pnl_p = pos.get('unrealized_pct')
        cl = '#27ae60' if (pnl_p or 0) >= 0 else '#e74c3c'
        stop_hit = cp and pos.get('stop') and float(cp) <= pos['stop']
        bg_chip  = '#fce4ec' if stop_hit else '#eef2f8'
        pnl_str  = f"{pnl_p:+.1f}%" if pnl_p is not None else "—"
        pos_chips += (f'<span style="background:{bg_chip};border-radius:12px;padding:3px 9px;'
                      f'margin:2px;display:inline-block;font-size:8pt">'
                      f'<strong>{tk}</strong> <span style="color:{cl}">{pnl_str}</span>'
                      f'{" ⚠STOP" if stop_hit else ""}</span>')

    return f'''
<div style="background:white;border-radius:6px;padding:12px 18px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
  <div style="display:flex;align-items:center;gap:20px;margin-bottom:8px;flex-wrap:wrap">
    <span style="font-size:9pt;font-weight:700;color:#0d3b6e">Portfolio</span>
    <span style="font-size:8.5pt">Deployed: <strong>{dep_pct:.1f}%</strong> (${deployed:,.0f})</span>
    <span style="font-size:8.5pt">Cash: <strong>{cash_pct:.1f}%</strong></span>
    <span style="font-size:8.5pt;color:{pnl_c}">Unrealized P&amp;L: <strong>${pnl:+,.0f}</strong></span>
    <span style="font-size:7.5pt;color:#9ca3af;margin-left:auto">
      <code>python portfolio.py --update</code> to refresh P&amp;L
    </span>
  </div>
  <div style="margin-bottom:6px">{sec_bars}</div>
  <div>{pos_chips}</div>
</div>'''


def _matrix_html(ok_records):
    matrix_order = ['cheap_improving', 'momentum_growth', 'value_trap', 'avoid']
    cells = ''
    for q in matrix_order:
        label, color, desc = QUADRANTS[q]
        members = [r for r in ok_records if r.get('_quad') == q]
        chips = ''
        for r in sorted(members, key=lambda x: -(x.get('signal_conviction') or 0)):
            up = r.get('upside')
            up_str = ("%+.0f%%" % up) if up is not None else ''
            chips += (f'<span class="chip" onclick="filterQuad(\'{label}\')">'
                      f'{r["ticker"]} <span class="chip-pct">{up_str}</span></span>')
        if not chips:
            chips = '<span style="font-size:8pt;color:#ccc">—</span>'
        cells += (f'<div class="quad" style="border-top-color:{color}">'
                  f'<h4 style="color:{color}">{label}</h4>'
                  f'<div class="quad-desc">{desc}</div>{chips}</div>')
    return cells


def _backtest_stats_html(bt):
    if not bt:
        return '<div style="color:#9ca3af;font-size:8.5pt;padding:20px">No backtest data — run <code>python strategy_backtest.py</code></div>'
    def pct(v): return f"{v*100:+.1f}%" if v is not None else "—"
    def f2(v):  return f"{v:.2f}" if v is not None else "—"
    cagr_c   = '#27ae60' if (bt.get('cagr') or 0) > 0 else '#e74c3c'
    dd_c     = '#e74c3c' if (bt.get('max_drawdown') or 0) < -0.15 else '#e67e22' if (bt.get('max_drawdown') or 0) < -0.08 else '#27ae60'
    sharpe_c = '#27ae60' if (bt.get('sharpe') or 0) > 1.0 else '#e67e22' if (bt.get('sharpe') or 0) > 0.5 else '#e74c3c'
    act_c    = '#27ae60' if (bt.get('active_return') or 0) > 0 else '#e74c3c'
    stats = [
        ('CAGR',             pct(bt.get('cagr')),         cagr_c),
        ('Max Drawdown',     pct(bt.get('max_drawdown')), dd_c),
        ('Sharpe',           f2(bt.get('sharpe')),        sharpe_c),
        ('Sortino',          f2(bt.get('sortino')),       '#374151'),
        ('Calmar',           f2(bt.get('calmar')),        '#374151'),
        ('vs SPY (active)',  pct(bt.get('active_return')),act_c),
        ('SPY CAGR',         pct(bt.get('benchmark_cagr')),'#374151'),
        ('Win Rate/Month',   f"{(bt.get('win_rate_monthly') or 0)*100:.0f}%", '#374151'),
        ('Worst Year',       pct(bt.get('worst_year')),   '#e74c3c'),
        ('Best Year',        pct(bt.get('best_year')),    '#27ae60'),
        ('Worst 12m Rolling',pct(bt.get('worst_12m_rolling')),'#e74c3c'),
        ('Period',           f"{bt.get('period_start','—')} → {bt.get('period_end','—')}", '#374151'),
    ]
    cards = ''
    for lbl, val, clr in stats:
        cards += (f'<div class="bt-stat"><div class="bt-lbl">{lbl}</div>'
                  f'<div class="bt-val" style="color:{clr}">{val}</div></div>')
    dd_info = (f"Peak→Trough: {bt.get('max_dd_peak','—')} → {bt.get('max_dd_trough','—')} | "
               f"Recovery: {bt.get('max_dd_recovery','—')}")
    n_months = bt.get('n_months', '—')
    n_names  = bt.get('n_names', '—')
    scope = (f"Top {int((bt.get('top_pct') or 0.2)*100)}% by composite momentum · "
             f"TC {(bt.get('tc_per_side') or 0)*100:.2f}%/side · "
             f"Monthly rebalance · {n_names} names · {n_months} months")
    return f'''
<div class="bt-stats-grid">{cards}</div>
<div style="font-size:7.5pt;color:#9ca3af;margin-bottom:4px">{dd_info}</div>
<div style="font-size:7.5pt;color:#9ca3af;margin-bottom:18px">{scope}</div>'''


def _table_rows_html(records):
    rows = ''
    for r in sorted(records, key=lambda x: (-(x.get('signal_conviction') or 0), x.get('ticker', ''))):
        ticker = r.get('ticker', '?')
        if r.get('status') != 'OK':
            rows += (f'<tr data-rec="" data-quad=""><td class="ticker-cell">{ticker}</td>'
                     f'<td colspan="16" style="color:#b52a2a;font-style:italic">'
                     f'Error: {str(r.get("error",""))[:80]}</td></tr>')
            continue

        rec   = r.get('recommendation', '-')
        up    = r.get('upside')
        conv  = r.get('signal_conviction')
        quad  = r.get('_quad', 'neutral')
        qlabel, qcolor, _ = QUADRANTS[quad]
        dq    = r.get('data_quality_verdict', 'OK')
        shortf = r.get('short_pct_float')
        p_action = r.get('plan_action', '—')
        p_mode   = r.get('plan_mode', '')
        elo, ehi = r.get('plan_entry_low'), r.get('plan_entry_high')
        entry_str = f"${elo}–${ehi}" if (elo and ehi) else '—'
        act_color = {'value': '#1a7a4a', 'momentum': '#1a6fa8', 'wait': '#d4900a',
                     'exit': '#b52a2a', 'blocked': '#6b7280'}.get(p_mode, '#6b7280')

        def _num(v, pre='', suf='', dec=1):
            if v is None or v == '': return '<td class="num">—</td>'
            try:
                fv = float(v)
                return f'<td class="num" data-val="{fv}">{pre}{fv:.{dec}f}{suf}</td>'
            except Exception:
                return f'<td class="num">{v}</td>'

        conv_cell = '<td class="num">—</td>'
        if conv is not None:
            conv_cell = (f'<td class="num" data-val="{conv}">'
                         f'<span class="conv-bar-bg"><span class="conv-bar" '
                         f'style="width:{conv*0.46:.0f}px;background:{_conv_color(conv)}"></span></span>'
                         f'<span style="color:{_conv_color(conv)};font-weight:700">{conv}</span></td>')

        q_score = r.get('quality_score', '-')
        q_label = r.get('quality_label', '-')
        p_stop  = r.get('plan_stop')
        p_t1    = r.get('plan_target1')
        p_rr    = r.get('plan_rr')
        p_size  = r.get('plan_position_pct')

        rows += f'''<tr data-rec="{rec}" data-quad="{qlabel}" data-ticker="{ticker}"
          onclick="openModal('{ticker}')">
          <td class="ticker-cell">{ticker}</td>
          <td class="name-cell">{str(r.get("name",""))[:20]}</td>
          <td class="num" data-val="{r.get('current_price',0)}">${r.get('current_price','—')}</td>
          <td><span class="badge" style="background:{_rec_color(rec)}">{rec}</span></td>
          <td class="num" data-val="{r.get('price_target',0)}" style="font-weight:700">${r.get('price_target','—')}</td>
          <td class="num" data-val="{up or 0}" style="color:{_upside_color(up)};font-weight:700">{("%+.1f%%"%up) if up is not None else "—"}</td>
          {conv_cell}
          <td><span class="tag" style="background:{qcolor}22;color:{qcolor}">{qlabel}</span></td>
          <td style="font-size:7.8pt;color:{act_color};font-weight:600;max-width:150px;white-space:normal">{p_action}</td>
          <td class="num" style="font-size:8pt">{entry_str}</td>
          {_num(p_stop, pre='$', dec=2) if p_stop else '<td class="num">—</td>'}
          {_num(p_t1, pre='$', dec=2) if p_t1 else '<td class="num">—</td>'}
          {_num(p_rr, dec=2) if p_rr else '<td class="num">—</td>'}
          {_num(p_size, suf='%') if p_size else '<td class="num">—</td>'}
          {_num(shortf*100 if shortf is not None else None, suf='%', dec=1)}
          <td style="color:{_q_color(q_label)};font-weight:600">{q_score}/9</td>
          <td><span class="badge" style="background:{_dq_color(dq)}">{dq}</span></td>
        </tr>'''
    return rows


# ─── bottom-fishing (left-side) tab ───────────────────────────────────────────

_BF_TIER_COLOR = {'STRONG': '#1a7a4a', 'SPECULATIVE': '#d4900a',
                  'KNIFE': '#b52a2a', 'NONE': '#6b7280'}

def _bottomfish_html(bf_records, bf_bt):
    """Build the left-side / bottom-fishing tab body."""
    if not bf_records:
        return ('<div style="color:#9ca3af;font-size:9pt;padding:30px;text-align:center">'
                '尚未產生左側抄底資料 — 執行 <code>python bottom_fishing.py AAPL MSFT ...</code></div>')

    ok = [r for r in bf_records if r.get('status') == 'OK']
    order = {'STRONG': 0, 'SPECULATIVE': 1, 'KNIFE': 2, 'NONE': 3}
    ok.sort(key=lambda r: (order.get(r.get('score', {}).get('tier'), 9),
                           -(r.get('score', {}).get('conviction') or 0)))
    n_strong = sum(1 for r in ok if r.get('score', {}).get('tier') == 'STRONG')
    n_spec   = sum(1 for r in ok if r.get('score', {}).get('tier') == 'SPECULATIVE')
    n_knife  = sum(1 for r in ok if r.get('score', {}).get('tier') == 'KNIFE')

    # backtest evidence strip
    bt_html = ''
    if bf_bt and bf_bt.get('variants'):
        v = bf_bt['variants']
        g = v.get('above_200', {})
        nv = v.get('naive', {})
        gs = v.get('gated_stopped', {})
        if g.get('n_trades'):
            bt_html = (
                '<div style="background:#f0f7f2;border-left:4px solid #1a7a4a;border-radius:4px;'
                'padding:10px 16px;margin-bottom:16px;font-size:8.3pt;color:#374151;line-height:1.6">'
                '<strong>回測證據</strong>（50檔·8年·事件型）：超賣均值回歸有效；加「站上上升200日線」'
                f'過濾後盈虧比 {nv.get("profit_factor","—")}→<strong>{g.get("profit_factor","—")}</strong>、'
                f'勝率 {nv.get("win_rate",0)*100:.1f}%→<strong>{g.get("win_rate",0)*100:.1f}%</strong>；'
                f'加 ATR 硬停損把單筆最深帳面虧損從 <strong style="color:#b52a2a">{nv.get("worst_mae",0)*100:.0f}%</strong>'
                f' 砍到 <strong style="color:#1a7a4a">{gs.get("worst_mae",0)*100:.0f}%</strong>。'
                '<a href="bottom_fishing_backtest.html" target="_blank" style="margin-left:8px;color:#1a7a4a">完整回測 →</a></div>')

    rows = ''
    for r in ok:
        s = r.get('score', {}); p = r.get('panel', {}); pl = r.get('plan', {})
        st = r.get('structure', {}); surv = r.get('survivability', {})
        tier = s.get('tier', 'NONE'); col = _BF_TIER_COLOR.get(tier, '#6b7280')
        dist200 = p.get('dist_ma200')
        avg_entry = pl.get('avg_entry'); stop = pl.get('stop')
        rows += f'''<tr>
          <td style="font-weight:700;color:#0d3b6e">{r.get('ticker','?')}</td>
          <td class="name-cell">{str(r.get('name',''))[:20]}</td>
          <td><span class="badge" style="background:{col}">{s.get('tier_label','—')}</span></td>
          <td class="num" style="font-weight:700;color:{col}">{s.get('conviction','—')}</td>
          <td class="num">{r.get('oversold',{}).get('score','—')}</td>
          <td class="num">{r.get('capitulation',{}).get('score','—')}</td>
          <td class="num">{r.get('confirmation',{}).get('score','—')}</td>
          <td><span class="tag">{st.get('trend','—')}</span></td>
          <td class="num">{surv.get('survivability','—')}</td>
          <td class="num">{('%+.1f%%'%(dist200*100)) if dist200 is not None else '—'}</td>
          <td class="num">{p.get('rsi14') if p.get('rsi14') is not None else '—'}</td>
          <td class="num">{('$'+str(avg_entry)) if avg_entry else '—'}</td>
          <td class="num" style="color:#b52a2a">{('$'+str(stop)) if stop else '—'}</td>
          <td class="num">{(str(pl.get('position_pct'))+'%') if pl.get('position_pct') is not None else '—'}</td>
        </tr>'''

    intro = (
        '<div style="background:white;border-radius:6px;padding:12px 16px;margin-bottom:14px;'
        'box-shadow:0 1px 4px rgba(0,0,0,.08);font-size:8.3pt;color:#374151;line-height:1.6">'
        '<strong>左側 / 抄底引擎：</strong>動能引擎買強勢(右側),這裡買弱勢(左側)。但只在'
        '<strong>「上升趨勢中的超賣 × 體質健全 × 估值安全邊際 × 止跌確認」</strong>的交集出手，'
        '以分批進場 + 預設硬停損 + 時間停損 + 反人性紀律執行。完整劇本見 '
        '<a href="bottom_fishing.html" target="_blank" style="color:#7a1f1f">bottom_fishing.html →</a></div>')

    kpis = (
        '<div class="kpi-strip" style="grid-template-columns:repeat(4,1fr)">'
        f'<div class="kpi"><div class="lbl">分析標的</div><div class="val">{len(ok)}</div></div>'
        f'<div class="kpi"><div class="lbl">高把握抄底</div><div class="val" style="color:#1a7a4a">{n_strong}</div></div>'
        f'<div class="kpi"><div class="lbl">投機性抄底</div><div class="val" style="color:#d4900a">{n_spec}</div></div>'
        f'<div class="kpi"><div class="lbl">接刀風險(避開)</div><div class="val" style="color:#b52a2a">{n_knife}</div></div>'
        '</div>')

    return f'''{intro}{bt_html}{kpis}
    <div class="table-card">
      <div class="table-header"><h2>左側抄底候選排序</h2>
        <span style="font-size:7.8pt;opacity:.75">點 bottom_fishing.html 看完整分批劇本</span></div>
      <table><thead><tr>
        <th>代號</th><th>名稱</th><th>判定</th><th class="num">把握度</th>
        <th class="num">超賣</th><th class="num">投降</th><th class="num">確認</th>
        <th>結構</th><th class="num">體質</th><th class="num">vs200dma</th><th class="num">RSI14</th>
        <th class="num">均價</th><th class="num">停損</th><th class="num">部位</th>
      </tr></thead><tbody>{rows}</tbody></table>
    </div>'''


# ─── main generator ───────────────────────────────────────────────────────────

def generate_dashboard(summary_file=None, output_file=None):
    summary_file = summary_file or SUMMARY_FILE
    output_file  = output_file  or DASHBOARD_FILE
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary not found: {summary_file}. Run batch_run.py first.")

    with open(summary_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    ok = [r for r in records if r.get('status') == 'OK']
    recs = [r.get('recommendation', '') for r in ok]
    n_buy, n_hold, n_sell = recs.count('BUY'), recs.count('HOLD'), recs.count('SELL')
    convs = [r.get('signal_conviction') for r in ok if r.get('signal_conviction') is not None]
    avg_conv = round(sum(convs) / len(convs)) if convs else None
    n_dq_flags = sum(1 for r in ok if r.get('data_quality_verdict') in ('REVIEW', 'UNRELIABLE'))
    rf_rate = ok[0].get('risk_free_rate', '—') if ok else '—'
    vix     = ok[0].get('vix', '—') if ok else '—'

    for r in ok:
        r['_quad'] = _quadrant(r.get('upside'), r.get('signal_score'))

    regime_data    = _load_json(REGIME_FILE)
    alerts_data    = _load_json(ALERTS_FILE)
    portfolio_data = _load_json(PORTFOLIO_FILE)
    bt_stats       = _load_json(BACKTEST_FILE)
    bt_curves      = _load_json(CURVES_FILE)
    bf_records     = _load_json(BOTTOMFISH_FILE)
    bf_backtest    = _load_json(BF_BACKTEST_FILE)

    # Build embedded stock data for modal (JS object keyed by ticker)
    stock_data_js = json.dumps({r['ticker']: r for r in records if r.get('status') == 'OK'},
                               ensure_ascii=False)

    # Build backtest chart data for inline Chart.js
    if bt_curves:
        bt_dates_js   = json.dumps(bt_curves.get('dates', []))
        bt_eq_js      = json.dumps(bt_curves.get('equity', []))
        bt_spy_js     = json.dumps(bt_curves.get('benchmark', []))
        bt_dd_js      = json.dumps(bt_curves.get('drawdown', []))
        ann           = bt_curves.get('annual_returns', {})
        b_ann         = bt_curves.get('benchmark_annual', {})
        bt_yrs_js     = json.dumps([str(y) for y in sorted(ann.keys())])
        bt_strat_js   = json.dumps([round(ann.get(y, 0) * 100, 1) for y in sorted(ann.keys())])
        bt_bench_js   = json.dumps([round(b_ann.get(y, 0) * 100, 1) for y in sorted(ann.keys())])
        has_curves    = 'true'
    else:
        bt_dates_js = bt_eq_js = bt_spy_js = bt_dd_js = '[]'
        bt_yrs_js = bt_strat_js = bt_bench_js = '[]'
        has_curves = 'false'

    # ── assemble sections ──
    regime_html    = _regime_html(regime_data)
    alerts_html    = _alerts_html(alerts_data)
    portfolio_html = _portfolio_html(portfolio_data)
    matrix_cells   = _matrix_html(ok)
    bt_stats_html  = _backtest_stats_html(bt_stats)
    bf_html        = _bottomfish_html(bf_records, bf_backtest)
    n_bf_strong    = sum(1 for r in (bf_records or [])
                         if r.get('status') == 'OK'
                         and r.get('score', {}).get('tier') == 'STRONG')
    table_rows     = _table_rows_html(records)
    updated        = datetime.now().strftime('%Y-%m-%d %H:%M')

    avg_conv_str   = str(avg_conv) if avg_conv is not None else '—'
    avg_conv_color = _conv_color(avg_conv)
    dq_color       = '#b52a2a' if n_dq_flags else '#1a7a4a'

    html = _HTML_TEMPLATE.format(
        updated=updated,
        n_total=len(records), n_ok=len(ok),
        n_buy=n_buy, n_hold=n_hold, n_sell=n_sell,
        avg_conv=avg_conv_str, avg_conv_color=avg_conv_color,
        n_dq_flags=n_dq_flags, dq_color=dq_color,
        rf_rate=rf_rate, vix=vix,
        regime_html=regime_html,
        alerts_html=alerts_html,
        portfolio_html=portfolio_html,
        matrix_cells=matrix_cells,
        bt_stats_html=bt_stats_html,
        bf_html=bf_html,
        n_bf_strong=n_bf_strong,
        table_rows=table_rows,
        stock_data_js=stock_data_js,
        bt_dates_js=bt_dates_js, bt_eq_js=bt_eq_js,
        bt_spy_js=bt_spy_js, bt_dd_js=bt_dd_js,
        bt_yrs_js=bt_yrs_js, bt_strat_js=bt_strat_js,
        bt_bench_js=bt_bench_js, has_curves=has_curves,
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[ok] Unified dashboard -> {output_file}")
    return output_file


# ─── HTML template ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Equity Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;font-size:9.5pt;color:#1a1a2e;background:#f0f2f7}}

/* ── top bar ── */
.top-bar{{background:#0d3b6e;color:white;padding:12px 28px;display:flex;align-items:center;justify-content:space-between}}
.top-bar h1{{font-size:13pt;letter-spacing:.4px}}
.top-bar .sub{{font-size:7.8pt;opacity:.75;margin-top:3px}}
.top-bar .updated{{font-size:7.5pt;opacity:.55}}

/* ── tab nav ── */
.tab-nav{{background:#0a2e58;display:flex;gap:0;padding:0 28px}}
.tab-btn{{background:none;border:none;color:rgba(255,255,255,.55);padding:10px 22px;font-size:9pt;
          font-weight:600;cursor:pointer;border-bottom:3px solid transparent;transition:.15s;letter-spacing:.3px}}
.tab-btn:hover{{color:white;background:rgba(255,255,255,.06)}}
.tab-btn.active{{color:white;border-bottom-color:#4da6ff}}
.tab-count{{font-size:7.5pt;background:rgba(255,255,255,.18);padding:1px 7px;border-radius:10px;margin-left:6px}}

/* ── container ── */
.container{{max-width:1520px;margin:0 auto;padding:20px}}
.tab-content{{display:none}}.tab-content.active{{display:block}}

/* ── strips (regime / alerts) ── */
.strip{{display:flex;align-items:center;gap:24px;padding:10px 20px;margin-bottom:14px;
        border-radius:4px;border-left:5px solid #ccc;flex-wrap:wrap}}
.strip-item{{font-size:8.5pt;color:#374151}}
.alert-chip{{padding:3px 10px;border-radius:12px;font-weight:700;font-size:7.8pt;color:white}}

/* ── KPI cards ── */
.kpi-strip{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}}
.kpi{{background:white;border-radius:6px;padding:11px 15px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.kpi .lbl{{font-size:7pt;color:#6b7280;text-transform:uppercase;letter-spacing:.4px}}
.kpi .val{{font-size:15pt;font-weight:700;color:#0d3b6e;margin-top:2px}}

/* ── value × momentum matrix ── */
.matrix-title{{font-weight:600;color:#0d3b6e;margin-bottom:10px;font-size:10.5pt}}
.matrix{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:22px}}
.quad{{background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);
       padding:13px 15px;border-top:4px solid #ccc;min-height:88px}}
.quad h4{{font-size:9.5pt;margin-bottom:3px}}
.quad-desc{{font-size:7.5pt;color:#6b7280;margin-bottom:8px;line-height:1.4}}
.chip{{display:inline-block;padding:3px 9px;margin:2px 3px 2px 0;border-radius:12px;
       background:#eef2f8;font-size:8.3pt;font-weight:600;cursor:pointer}}
.chip:hover{{background:#dbe6f5}}.chip-pct{{font-weight:400;color:#6b7280;font-size:7.3pt}}

/* ── coverage table ── */
.table-card{{background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.08);
             overflow-x:auto;margin-bottom:20px}}
.table-header{{background:#0d3b6e;color:white;padding:9px 16px;display:flex;
               align-items:center;justify-content:space-between}}
.table-header h2{{font-size:10pt}}
table{{width:100%;border-collapse:collapse;font-size:8.8pt}}
thead th{{background:#1a2e4e;color:white;padding:8px 10px;text-align:left;
          font-weight:600;font-size:7.6pt;cursor:pointer;white-space:nowrap}}
thead th:hover{{background:#243c5e}}
thead th::after{{content:' ↕';opacity:.4;font-size:6.5pt}}
tbody tr{{border-bottom:1px solid #eaeaea;cursor:pointer}}
tbody tr:hover{{background:#f0f7ff}}
td{{padding:7px 10px;white-space:nowrap}}
.ticker-cell{{font-weight:700;color:#0d3b6e;font-size:9.5pt}}
.name-cell{{color:#374151;max-width:145px;overflow:hidden;text-overflow:ellipsis}}
.badge{{display:inline-block;padding:2px 8px;border-radius:3px;color:white;
        font-weight:700;font-size:7.3pt;letter-spacing:.3px}}
.num{{text-align:right}}.tag{{display:inline-block;padding:2px 7px;background:#e8f0fe;
       border-radius:10px;font-size:7.1pt;color:#1a6fa8}}
.conv-bar-bg{{display:inline-block;width:46px;height:8px;background:#eee;border-radius:4px;
              overflow:hidden;vertical-align:middle;margin-right:4px}}
.conv-bar{{height:8px}}

/* ── controls ── */
.controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}
.controls label{{font-size:8.3pt;color:#6b7280}}
.filter-btn{{padding:4px 12px;border-radius:4px;border:1px solid #dce3ec;background:white;
             font-size:8.3pt;cursor:pointer;color:#374151}}
.filter-btn.active{{background:#0d3b6e;color:white;border-color:#0d3b6e}}
.filter-btn:hover{{background:#f0f7ff}}.filter-btn.active:hover{{background:#1a4d82}}
.search-box{{padding:4px 12px;border-radius:4px;border:1px solid #dce3ec;font-size:8.3pt;width:170px}}

/* ── backtest section ── */
.bt-stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}}
.bt-stat{{background:white;border-radius:6px;padding:12px 16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.bt-lbl{{font-size:7pt;color:#6b7280;text-transform:uppercase;letter-spacing:.5px}}
.bt-val{{font-size:14pt;font-weight:700;margin-top:2px}}
.chart-card{{background:white;border-radius:6px;padding:16px 20px;margin-bottom:16px;
             box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.chart-card h3{{font-size:9.5pt;font-weight:600;color:#0d3b6e;margin-bottom:12px}}

/* ── modal ── */
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:1000;
                overflow-y:auto;padding:40px 20px}}
.modal{{background:white;border-radius:8px;max-width:820px;margin:0 auto;
        box-shadow:0 8px 32px rgba(0,0,0,.28);overflow:hidden}}
.modal-head{{background:#0d3b6e;color:white;padding:16px 22px;display:flex;
             align-items:flex-start;justify-content:space-between}}
.modal-close{{background:none;border:none;color:white;font-size:18pt;cursor:pointer;
              opacity:.7;line-height:1;padding:0 4px}}
.modal-close:hover{{opacity:1}}
.modal-body{{padding:20px 22px}}
.modal-section{{margin-bottom:18px}}
.modal-section h4{{font-size:8.5pt;font-weight:700;color:#6b7280;text-transform:uppercase;
                   letter-spacing:.5px;margin-bottom:10px;border-bottom:1px solid #eaeaea;padding-bottom:4px}}
.modal-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.modal-card{{background:#f8fafd;border-radius:5px;padding:9px 12px}}
.modal-card .lbl{{font-size:7pt;color:#9ca3af;text-transform:uppercase;letter-spacing:.4px}}
.modal-card .val{{font-size:11pt;font-weight:700;color:#1a1a2e;margin-top:1px}}
.modal-card .sub{{font-size:7.5pt;color:#6b7280;margin-top:1px}}
.signal-row{{display:flex;align-items:center;gap:8px;padding:4px 0;
             border-bottom:1px solid #f0f0f0;font-size:8.3pt}}
.signal-lbl{{width:160px;color:#374151}}
.signal-bar-bg{{flex:1;height:7px;background:#eee;border-radius:3px;overflow:hidden}}
.signal-bar{{height:7px;border-radius:3px}}
.signal-val{{width:50px;text-align:right;font-weight:600;font-size:8pt}}
.flag{{font-size:7.8pt;color:#e67e22;padding:3px 0;border-bottom:1px solid #fef3c7}}

.footer{{text-align:center;font-size:7.3pt;color:#9ca3af;padding:14px}}
</style></head><body>

<div class="top-bar">
  <div>
    <h1>Buy-Side Equity Research Dashboard</h1>
    <div class="sub">Dual engine — Valuation (DCF · Peer P/E · Comps) × Predictive Signals (momentum · revisions · PEAD · 籌碼)</div>
  </div>
  <div class="updated">Last updated: {updated}</div>
</div>

<div class="tab-nav">
  <button class="tab-btn active" onclick="showTab('overview',this)">
    Overview</button>
  <button class="tab-btn" onclick="showTab('coverage',this)">
    Coverage <span class="tab-count">{n_total}</span></button>
  <button class="tab-btn" onclick="showTab('bottomfish',this)">
    左側抄底 <span class="tab-count">{n_bf_strong}</span></button>
  <button class="tab-btn" onclick="showTab('backtest',this)">
    Strategy Backtest</button>
</div>

<!-- ═══════════════════════════════════════════════════════ TAB: OVERVIEW -->
<div id="tab-overview" class="tab-content active">
<div class="container">

  {regime_html}
  {alerts_html}
  {portfolio_html}

  <div class="kpi-strip">
    <div class="kpi"><div class="lbl">Coverage</div><div class="val">{n_total}</div></div>
    <div class="kpi"><div class="lbl">BUY</div><div class="val" style="color:#1a7a4a">{n_buy}</div></div>
    <div class="kpi"><div class="lbl">HOLD</div><div class="val" style="color:#d4900a">{n_hold}</div></div>
    <div class="kpi"><div class="lbl">SELL</div><div class="val" style="color:#b52a2a">{n_sell}</div></div>
    <div class="kpi"><div class="lbl">Avg Conviction</div>
      <div class="val" style="color:{avg_conv_color}">{avg_conv}</div></div>
    <div class="kpi"><div class="lbl">Data Flags</div>
      <div class="val" style="color:{dq_color}">{n_dq_flags}</div></div>
  </div>

  <div class="matrix-title">Value × Momentum Matrix
    <span style="font-size:7.8pt;color:#9ca3af;font-weight:400"> — click a name to filter the Coverage table</span>
  </div>
  <div class="matrix">{matrix_cells}</div>

</div>
</div>

<!-- ═══════════════════════════════════════════════════════ TAB: COVERAGE -->
<div id="tab-coverage" class="tab-content">
<div class="container">

  <div class="controls">
    <label>Rec:</label>
    <button class="filter-btn active" onclick="filterRec('ALL',event)">All</button>
    <button class="filter-btn" onclick="filterRec('BUY',event)" style="border-color:#1a7a4a;color:#1a7a4a">BUY</button>
    <button class="filter-btn" onclick="filterRec('HOLD',event)" style="border-color:#d4900a;color:#d4900a">HOLD</button>
    <button class="filter-btn" onclick="filterRec('SELL',event)" style="border-color:#b52a2a;color:#b52a2a">SELL</button>
    <input class="search-box" id="searchBox" type="text" placeholder="Search ticker / name…" oninput="filterSearch(this.value)"/>
    <span id="quadFilterLabel" style="font-size:7.8pt;color:#1a6fa8"></span>
    <span style="margin-left:auto;font-size:7.8pt;color:#9ca3af">Click header to sort · click row for detail</span>
  </div>

  <div class="table-card">
    <div class="table-header">
      <h2>Coverage Universe</h2>
      <span style="font-size:7.8pt;opacity:.75">{n_ok} reports · RF {rf_rate}% · VIX {vix}</span>
    </div>
    <table id="mainTable"><thead><tr>
      <th onclick="sortTable(0)">Ticker</th>
      <th onclick="sortTable(1)">Company</th>
      <th onclick="sortTable(2)" class="num">Price</th>
      <th onclick="sortTable(3)">Rec</th>
      <th onclick="sortTable(4)" class="num">PT</th>
      <th onclick="sortTable(5)" class="num">Upside</th>
      <th onclick="sortTable(6)" class="num">Conviction</th>
      <th onclick="sortTable(7)">Quadrant</th>
      <th onclick="sortTable(8)">Action</th>
      <th onclick="sortTable(9)">Entry Zone</th>
      <th onclick="sortTable(10)" class="num">Stop</th>
      <th onclick="sortTable(11)" class="num">Target</th>
      <th onclick="sortTable(12)" class="num">R:R</th>
      <th onclick="sortTable(13)" class="num">Size%</th>
      <th onclick="sortTable(14)" class="num">Short%Flt</th>
      <th onclick="sortTable(15)">Quality</th>
      <th onclick="sortTable(16)">Data</th>
    </tr></thead><tbody id="tableBody">{table_rows}</tbody></table>
  </div>

</div>
</div>

<!-- ═══════════════════════════════════════════════════════ TAB: 左側抄底 -->
<div id="tab-bottomfish" class="tab-content">
<div class="container">
  {bf_html}
</div>
</div>

<!-- ═══════════════════════════════════════════════════════ TAB: BACKTEST -->
<div id="tab-backtest" class="tab-content">
<div class="container">

  {bt_stats_html}

  <div class="chart-card">
    <h3>Equity Curve (indexed to 100)</h3>
    <canvas id="eqChart" height="90"></canvas>
  </div>

  <div style="display:grid;grid-template-columns:2fr 1fr;gap:16px">
    <div class="chart-card">
      <h3>Drawdown (%)</h3>
      <canvas id="ddChart" height="110"></canvas>
    </div>
    <div class="chart-card">
      <h3>Annual Returns vs SPY</h3>
      <canvas id="annChart" height="110"></canvas>
    </div>
  </div>

  <div style="font-size:7.5pt;color:#9ca3af;padding:4px 0">
    Price-momentum signals only (12-1 + 6m). Full strategy adds revisions / PEAD / quality / 籌碼 —
    validated via live snapshots.csv track record.
  </div>

</div>
</div>

<!-- ═══════════════════════════════════════════════════════ STOCK MODAL -->
<div class="modal-overlay" id="stockModal" onclick="modalOverlayClick(event)">
  <div class="modal" id="modalInner">
    <div class="modal-head" id="modalHead">
      <div id="modalTitle"></div>
      <button class="modal-close" onclick="closeModal()">×</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<div class="footer">
  Unified dashboard · Data: Yahoo Finance · Signals backtest-validated · Estimates only, not investment advice · {updated}
</div>

<script>
// ── embedded data ────────────────────────────────────────────────────────────
const STOCK_DATA = {stock_data_js};
const BT_HAS_CURVES = {has_curves};
const BT_DATES  = {bt_dates_js};
const BT_EQ     = {bt_eq_js};
const BT_SPY    = {bt_spy_js};
const BT_DD     = {bt_dd_js};
const BT_YRS    = {bt_yrs_js};
const BT_STRAT  = {bt_strat_js};
const BT_BENCH  = {bt_bench_js};

// ── tabs ─────────────────────────────────────────────────────────────────────
let btChartsInit = false;
function showTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'backtest' && !btChartsInit) {{ initBtCharts(); btChartsInit = true; }}
}}

// ── matrix → coverage filter ─────────────────────────────────────────────────
let currentFilter='ALL', currentSearch='', currentQuad='';
function filterQuad(q) {{
  currentQuad = (currentQuad === q ? '' : q);
  document.getElementById('quadFilterLabel').textContent =
    currentQuad ? ('Quadrant: ' + currentQuad + '  ×') : '';
  // switch to coverage tab
  showTab('coverage', document.querySelectorAll('.tab-btn')[1]);
  apply();
}}
function filterRec(rec, e) {{
  currentFilter = rec;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  if (e) e.target.classList.add('active');
  apply();
}}
function filterSearch(v) {{ currentSearch = v.toLowerCase(); apply(); }}
function apply() {{
  document.querySelectorAll('#tableBody tr').forEach(row => {{
    const rec  = row.getAttribute('data-rec') || '';
    const quad = row.getAttribute('data-quad') || '';
    const text = row.textContent.toLowerCase();
    const okRec    = (currentFilter==='ALL') || (rec===currentFilter);
    const okSearch = !currentSearch || text.includes(currentSearch);
    const okQuad   = !currentQuad   || (quad===currentQuad);
    row.style.display = (okRec && okSearch && okQuad) ? '' : 'none';
  }});
}}

// ── sort ──────────────────────────────────────────────────────────────────────
let sortDir = {{}};
function sortTable(col) {{
  const tb = document.getElementById('tableBody');
  const rows = Array.from(tb.querySelectorAll('tr'));
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    const av = a.cells[col] ? (a.cells[col].getAttribute('data-val') || a.cells[col].textContent.trim()) : '';
    const bv = b.cells[col] ? (b.cells[col].getAttribute('data-val') || b.cells[col].textContent.trim()) : '';
    const an = parseFloat(String(av).replace(/[^0-9.\-]/g,'')),
          bn = parseFloat(String(bv).replace(/[^0-9.\-]/g,''));
    if (!isNaN(an) && !isNaN(bn)) return sortDir[col] ? an - bn : bn - an;
    return sortDir[col] ? String(av).localeCompare(bv) : String(bv).localeCompare(av);
  }});
  rows.forEach(r => tb.appendChild(r));
}}

// ── modal ─────────────────────────────────────────────────────────────────────
function openModal(ticker) {{
  const d = STOCK_DATA[ticker];
  if (!d) return;
  const REC_COL = {{BUY:'#1a7a4a', HOLD:'#d4900a', SELL:'#b52a2a', 'NO RATING':'#6b7280'}};
  const rec = d.recommendation || '—';
  const recC = REC_COL[rec] || '#6b7280';
  const up = d.upside;
  const upStr = up != null ? (up >= 0 ? '+' : '') + up.toFixed(1) + '%' : '—';
  const upC = up == null ? '#6b7280' : up > 10 ? '#1a7a4a' : up > -5 ? '#d4900a' : '#b52a2a';

  document.getElementById('modalTitle').innerHTML = `
    <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
      <span style="font-size:14pt;font-weight:700">${{ticker}}</span>
      <span style="font-size:9pt;opacity:.8">${{d.name || ''}}</span>
      <span style="background:${{recC}};color:white;padding:3px 10px;border-radius:3px;font-size:8.5pt;font-weight:700">${{rec}}</span>
      <span style="font-size:10pt;color:white;opacity:.9">${{d.current_price != null ? '$'+d.current_price : ''}}</span>
      <span style="font-size:9pt;color:rgba(255,255,255,.65)">→ PT</span>
      <span style="font-size:10pt;font-weight:700;color:white">${{d.price_target != null ? '$'+d.price_target : '—'}}</span>
      <span style="font-size:9.5pt;color:rgba(255,255,255,.9);font-weight:700">(
        <span style="color:${{upC}}">${{upStr}}</span>)</span>
    </div>
    <div style="font-size:7.5pt;opacity:.55;margin-top:4px">${{d.sector||''}} · ${{d.industry||''}} · Cap ${{d.market_cap_b != null ? '$'+d.market_cap_b+'B' : '—'}}</div>`;

  // ── valuation anchors ──────────────────────────────────────────────────────
  const va = [
    ['Composite PT',    d.price_target,    null,   'Our weighted valuation'],
    ['DCF (Consensus)', d.consensus_dcf_pt, null,  `WACC ${{d.wacc}}%`],
    ['Peer P/E PT',     d.peer_pe_base_pt,  null,  `Fwd P/E ${{d.fwd_pe != null ? d.fwd_pe.toFixed(1) : '—'}}`],
    ['Street Mean PT',  d.street_mean_pt,   null,  `${{d.street_n_analysts||'—'}} analysts · ${{d.street_view||'—'}}`],
  ];
  let valHtml = '<div class="modal-grid">';
  for (const [lbl, val, col, sub] of va) {{
    const vStr = val != null ? '$' + (typeof val === 'number' ? val.toFixed(2) : val) : '—';
    valHtml += `<div class="modal-card"><div class="lbl">${{lbl}}</div>
      <div class="val" style="color:${{col||'#0d3b6e'}}">${{vStr}}</div>
      <div class="sub">${{sub||''}}</div></div>`;
  }}
  valHtml += '</div>';

  // ── trade plan ────────────────────────────────────────────────────────────
  const elo = d.plan_entry_low, ehi = d.plan_entry_high;
  const entryStr = (elo && ehi) ? `$${{elo}}–$${{ehi}}` : '—';
  const MODE_COL = {{value:'#1a7a4a',momentum:'#1a6fa8',wait:'#d4900a',exit:'#b52a2a',blocked:'#6b7280'}};
  const actC = MODE_COL[d.plan_mode] || '#6b7280';
  const plan = [
    ['Action',       `<strong style="color:${{actC}}">${{d.plan_action||'—'}}</strong>`, true],
    ['Entry Zone',   entryStr, false],
    ['Stop',         d.plan_stop   != null ? '$'+d.plan_stop   : '—', false],
    ['Target 1',     d.plan_target1 != null ? '$'+d.plan_target1 : '—', false],
    ['R:R',          d.plan_rr != null ? d.plan_rr.toFixed(2)+'×' : '—', false],
    ['Position Size',d.plan_position_pct != null ? d.plan_position_pct.toFixed(1)+'%' : '—', false],
    ['Shares',       d.plan_shares != null ? d.plan_shares : '—', false],
    ['Regime Mult',  d.regime_multiplier != null ? '×'+(d.regime_multiplier*100).toFixed(0)+'%' : '—', false],
    ['Regime',       d.regime || '—', false],
  ];
  let planHtml = '<div class="modal-grid">';
  for (const [lbl, val, raw] of plan) {{
    planHtml += `<div class="modal-card"><div class="lbl">${{lbl}}</div>
      <div class="val" style="font-size:9.5pt">${{raw ? val : val}}</div></div>`;
  }}
  planHtml += '</div>';

  // ── signals ───────────────────────────────────────────────────────────────
  const conv = d.signal_conviction;
  const convC = conv == null ? '#6b7280' : conv >= 65 ? '#1a7a4a' : conv >= 45 ? '#d4900a' : '#b52a2a';
  const sigRows = [
    ['Price Momentum (12-1)', d.mom_12_1, 100, 'raw return'],
    ['Revision Momentum',     d.revision_score, 1, 'score'],
    ['Earnings Surprise',     d.surprise_last_pct, 100, '% surprise'],
    ['Short Interest',        d.short_pct_float != null ? d.short_pct_float*100 : null, 100, '% float'],
  ];
  let sigHtml = `<div class="signal-row" style="border-bottom:2px solid #0d3b6e;margin-bottom:4px">
    <div class="signal-lbl" style="font-weight:700">Conviction</div>
    <div class="signal-bar-bg"><div class="signal-bar"
      style="width:${{conv||0}}%;background:${{convC}}"></div></div>
    <div class="signal-val" style="color:${{convC}}">${{conv != null ? conv : '—'}}</div>
  </div>`;
  for (const [lbl, raw, scale, unit] of sigRows) {{
    const norm = raw != null ? Math.min(Math.abs(raw / scale * 100), 100) : 0;
    const barC = raw == null ? '#ccc' : raw > 0 ? '#1a6fa8' : '#e74c3c';
    const rawStr = raw != null ? (typeof raw === 'number' ? raw.toFixed(2) : raw) : '—';
    sigHtml += `<div class="signal-row">
      <div class="signal-lbl">${{lbl}}</div>
      <div class="signal-bar-bg"><div class="signal-bar"
        style="width:${{norm}}%;background:${{barC}}"></div></div>
      <div class="signal-val" style="color:${{barC}}">${{rawStr}}</div>
    </div>`;
  }}

  // ── fundamentals ──────────────────────────────────────────────────────────
  const fundItems = [
    ['Fwd P/E',     d.fwd_pe        != null ? d.fwd_pe.toFixed(1)       : '—'],
    ['P/E TTM',     d.pe_ttm        != null ? d.pe_ttm.toFixed(1)       : '—'],
    ['EV/EBITDA',   d.ev_ebitda     != null ? d.ev_ebitda.toFixed(1)    : '—'],
    ['Rev Growth',  d.rev_growth    != null ? d.rev_growth.toFixed(1)+'%': '—'],
    ['Gross Margin',d.gross_margin  != null ? d.gross_margin.toFixed(1)+'%':'—'],
    ['Op Margin',   d.op_margin     != null ? d.op_margin.toFixed(1)+'%' : '—'],
    ['ROE',         d.roe           != null ? d.roe.toFixed(1)+'%'      : '—'],
    ['D/E',         d.debt_to_equity!= null ? d.debt_to_equity.toFixed(2): '—'],
    ['Beta',        d.beta          != null ? d.beta.toFixed(2)          : '—'],
  ];
  let fundHtml = '<div class="modal-grid">';
  for (const [lbl, val] of fundItems) {{
    fundHtml += `<div class="modal-card"><div class="lbl">${{lbl}}</div>
      <div class="val" style="font-size:9.5pt">${{val}}</div></div>`;
  }}
  fundHtml += '</div>';

  // ── data quality flags ────────────────────────────────────────────────────
  const dq    = d.data_quality_verdict || 'OK';
  const dqC   = {{OK:'#1a7a4a',REVIEW:'#d4900a',UNRELIABLE:'#b52a2a'}}[dq] || '#6b7280';
  const flags = d.data_quality_flags || [];
  let flagHtml = `<div style="margin-bottom:8px">
    <span style="background:${{dqC}};color:white;padding:2px 10px;border-radius:3px;
      font-size:8pt;font-weight:700">${{dq}}</span>
    <span style="font-size:7.5pt;color:#9ca3af;margin-left:8px">score ${{d.data_quality_score||'—'}}/100</span>
  </div>`;
  if (flags.length) {{
    for (const f of flags) flagHtml += `<div class="flag">⚠ ${{f}}</div>`;
  }} else {{
    flagHtml += '<div style="font-size:8pt;color:#9ca3af">No data quality flags</div>';
  }}

  document.getElementById('modalBody').innerHTML = `
    <div class="modal-section"><h4>Valuation Anchors</h4>${{valHtml}}</div>
    <div class="modal-section"><h4>Trade Plan</h4>${{planHtml}}</div>
    <div class="modal-section"><h4>Predictive Signals</h4>${{sigHtml}}</div>
    <div class="modal-section"><h4>Fundamentals</h4>${{fundHtml}}</div>
    <div class="modal-section"><h4>Data Quality</h4>${{flagHtml}}</div>`;

  document.getElementById('stockModal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}}
function closeModal() {{
  document.getElementById('stockModal').style.display = 'none';
  document.body.style.overflow = '';
}}
function modalOverlayClick(e) {{
  if (e.target === document.getElementById('stockModal')) closeModal();
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

// ── backtest charts ───────────────────────────────────────────────────────────
function initBtCharts() {{
  if (!BT_HAS_CURVES || !BT_DATES.length) {{
    document.getElementById('eqChart').parentElement.innerHTML +=
      '<div style="color:#9ca3af;font-size:8.5pt;padding:20px 0">No curve data — run <code>python strategy_backtest.py</code></div>';
    return;
  }}

  // Equity curve
  new Chart(document.getElementById('eqChart'), {{
    type: 'line',
    data: {{
      labels: BT_DATES,
      datasets: [
        {{ label: 'Momentum Strategy', data: BT_EQ, borderColor:'#0d3b6e', borderWidth:2,
           pointRadius:0, fill: false, tension:0.1 }},
        {{ label: 'SPY (B&H)', data: BT_SPY, borderColor:'#9ca3af', borderWidth:1.5,
           pointRadius:0, fill: false, tension:0.1, borderDash:[4,4] }},
      ]
    }},
    options: {{
      responsive:true, interaction:{{mode:'index',intersect:false}},
      plugins:{{ legend:{{position:'top',labels:{{font:{{size:9}}}}}},
                 tooltip:{{callbacks:{{label: ctx => ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)}}}} }},
      scales:{{ x:{{ticks:{{maxTicksLimit:12,font:{{size:8}}}},grid:{{display:false}}}},
                y:{{ticks:{{font:{{size:8}}}},title:{{display:true,text:'Value (base 100)',font:{{size:8}}}}}} }}
    }}
  }});

  // Drawdown
  new Chart(document.getElementById('ddChart'), {{
    type: 'line',
    data: {{
      labels: BT_DATES,
      datasets: [
        {{ label: 'Drawdown %', data: BT_DD, borderColor:'#e74c3c', borderWidth:1.5,
           backgroundColor:'rgba(231,76,60,.1)', pointRadius:0, fill:true, tension:0.1 }}
      ]
    }},
    options: {{
      responsive:true, interaction:{{mode:'index',intersect:false}},
      plugins:{{ legend:{{position:'top',labels:{{font:{{size:9}}}}}},
                 tooltip:{{callbacks:{{label: ctx => ctx.parsed.y.toFixed(1)+'%'}}}} }},
      scales:{{ x:{{ticks:{{maxTicksLimit:12,font:{{size:8}}}},grid:{{display:false}}}},
                y:{{ticks:{{font:{{size:8}},callback: v => v+'%'}}}} }}
    }}
  }});

  // Annual bars
  new Chart(document.getElementById('annChart'), {{
    type: 'bar',
    data: {{
      labels: BT_YRS,
      datasets: [
        {{ label: 'Strategy', data: BT_STRAT, backgroundColor: BT_STRAT.map(v => v>=0?'rgba(13,59,110,.7)':'rgba(231,76,60,.7)') }},
        {{ label: 'SPY',      data: BT_BENCH, backgroundColor: 'rgba(156,163,175,.45)', borderRadius:2 }},
      ]
    }},
    options: {{
      responsive:true, interaction:{{mode:'index',intersect:false}},
      plugins:{{ legend:{{position:'top',labels:{{font:{{size:9}}}}}},
                 tooltip:{{callbacks:{{label: ctx => ctx.parsed.y.toFixed(1)+'%'}}}} }},
      scales:{{ x:{{ticks:{{font:{{size:8}}}},grid:{{display:false}}}},
                y:{{ticks:{{font:{{size:8}},callback: v => v+'%'}}}} }}
    }}
  }});
}}

// auto-init backtest if landing on that tab (unlikely but safe)
if (document.getElementById('tab-backtest').classList.contains('active')) {{
  initBtCharts(); btChartsInit = true;
}}
</script>
</body></html>"""


if __name__ == '__main__':
    path = generate_dashboard()
    print(f"Open: file:///{path.replace(os.sep, '/')}")
