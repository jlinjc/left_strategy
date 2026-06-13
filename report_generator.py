"""
HTML Report Generator — Buy-Side Equity Research Style
Outputs a standalone HTML file with embedded CSS and charts
"""
import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
})

COLORS = {
    'primary': '#0d3b6e',
    'secondary': '#1a6fa8',
    'accent': '#e8a020',
    'buy': '#1a7a4a',
    'hold': '#d4900a',
    'sell': '#b52a2a',
    'light_bg': '#f5f7fa',
    'border': '#dce3ec',
}

REC_COLOR = {'BUY': COLORS['buy'], 'HOLD': COLORS['hold'], 'SELL': COLORS['sell']}


# ─────────────────────────────────────────────
# CHART GENERATORS
# ─────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def chart_price_history(hist_1y: pd.DataFrame, company_name: str, ticker: str,
                        price_target: float, current_price: float) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    dates = hist_1y.index
    closes = hist_1y['Close']

    ax.fill_between(dates, closes, alpha=0.12, color=COLORS['secondary'])
    ax.plot(dates, closes, color=COLORS['secondary'], linewidth=1.5, label='Price')
    ax.axhline(y=price_target, color=COLORS['accent'], linewidth=1.5,
               linestyle='--', label=f'PT: ${price_target:.2f}')
    ax.axhline(y=current_price, color='#888', linewidth=1, linestyle=':', alpha=0.7,
               label=f'Current: ${current_price:.2f}')

    ax.set_title(f'{ticker} — 1-Year Price History', fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax.set_ylabel('Price (USD)')
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_fcf_waterfall(fcf_history: pd.Series, projected_fcfs: list,
                        base_year: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.5))

    hist_years = list(range(base_year - len(fcf_history) + 1, base_year + 1))
    proj_years = list(range(base_year + 1, base_year + 11))

    hist_vals = [v / 1e9 for v in fcf_history.values[::-1]]
    proj_vals = [v / 1e9 for v in projected_fcfs]

    bars1 = ax.bar(hist_years, hist_vals, color=COLORS['secondary'], alpha=0.85, label='Historical FCF')
    bars2 = ax.bar(proj_years[:5], proj_vals[:5], color=COLORS['accent'], alpha=0.85, label='Stage 1 Forecast')
    bars3 = ax.bar(proj_years[5:], proj_vals[5:], color='#adb5bd', alpha=0.85, label='Stage 2 Fade')

    ax.axvline(x=base_year + 0.5, color='#888', linestyle='--', linewidth=0.8)
    ax.set_title('Free Cash Flow — Historical vs. Forecast ($B)', fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax.set_ylabel('FCF ($B)')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}B'))
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_sensitivity_heatmap(sensitivity_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    data = sensitivity_df.values.astype(float)

    vmin, vmax = data.min(), data.max()
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sensitivity_df.columns)))
    ax.set_xticklabels(sensitivity_df.columns, fontsize=8)
    ax.set_yticks(range(len(sensitivity_df.index)))
    ax.set_yticklabels(sensitivity_df.index, fontsize=8)
    ax.set_xlabel('Terminal FCF Margin', fontsize=9)
    ax.set_ylabel('Stage-1 Revenue Growth', fontsize=9)
    ax.set_title('DCF Sensitivity: Implied Price ($) — Growth × Margin', fontsize=11, fontweight='bold', color=COLORS['primary'])

    for i in range(len(sensitivity_df.index)):
        for j in range(len(sensitivity_df.columns)):
            val = data[i, j]
            text_color = 'white' if (val < vmin + (vmax - vmin) * 0.3 or val > vmax - (vmax - vmin) * 0.3) else 'black'
            ax.text(j, i, f'${val:.0f}', ha='center', va='center', fontsize=7.5,
                    color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Implied Price ($)')
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_valuation_bridge(components: dict, current_price: float, price_target: float) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    labels = [c.replace('_', '/').upper() for c in components] + ['Price Target']
    values = [components[c]['value'] for c in components] + [price_target]
    weights = [f"({components[c]['weight']*100:.0f}%)" for c in components] + ['']
    colors = [COLORS['secondary']] * len(components) + [COLORS['accent']]

    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)
    ax.axvline(x=current_price, color='#888', linestyle='--', linewidth=1.2,
               label=f'Current ${current_price:.2f}')

    for bar, val, w in zip(bars, values, weights):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'${val:.2f} {w}', va='center', fontsize=8, color=COLORS['primary'])

    ax.set_title('Valuation Summary — Method Comparison', fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax.set_xlabel('Implied Price (USD)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_quality_radar(quality: dict) -> str:
    labels = list(quality['details'].keys())
    values = [1 if v else 0 for v in quality['details'].values()]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values_plot, color=COLORS['secondary'], linewidth=2)
    ax.fill(angles, values_plot, color=COLORS['secondary'], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    ax.set_title(f'Quality Score: {quality["score"]}/{quality["max"]} — {quality["label"]}',
                 fontsize=10, fontweight='bold', color=COLORS['primary'], pad=15)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_revenue_trend(income_stmt: pd.DataFrame, quarterly_income: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Annual revenue
    try:
        rev_label = next((l for l in ['Total Revenue', 'Revenue'] if l in income_stmt.index), None)
        if rev_label:
            rev = income_stmt.loc[rev_label].dropna().sort_index()
            years = [str(d.year) for d in rev.index]
            vals = [v / 1e9 for v in rev.values]
            axes[0].bar(years, vals, color=COLORS['primary'], alpha=0.8)
            axes[0].set_title('Annual Revenue ($B)', fontsize=10, fontweight='bold', color=COLORS['primary'])
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}B'))
    except Exception:
        axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0].transAxes)

    # Operating margin trend
    try:
        op_income_label = next((l for l in ['Operating Income', 'EBIT'] if l in income_stmt.index), None)
        if rev_label and op_income_label:
            rev = income_stmt.loc[rev_label].dropna().sort_index()
            op = income_stmt.loc[op_income_label].dropna().sort_index()
            common_idx = rev.index.intersection(op.index)
            margins = (op[common_idx] / rev[common_idx] * 100).values
            years = [str(d.year) for d in common_idx]
            axes[1].plot(years, margins, marker='o', color=COLORS['accent'], linewidth=2)
            axes[1].fill_between(years, margins, alpha=0.15, color=COLORS['accent'])
            axes[1].set_title('Operating Margin (%)', fontsize=10, fontweight='bold', color=COLORS['primary'])
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    except Exception:
        axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1].transAxes)

    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────
# PEER P/E CHARTS
# ─────────────────────────────────────────────
def chart_football_field(ff: dict, current_price: float, dcf_pt: float = None) -> str:
    """Horizontal bar chart showing Bear/Base/Bull P/E scenarios + DCF."""
    fig, ax = plt.subplots(figsize=(10, 3.0))

    scenarios = []
    if ff.get('bear', {}).get('pt'):
        scenarios.append(('Bear', ff['bear']['pt'], ff['bear']['pe'], '#b52a2a'))
    if ff.get('base', {}).get('pt'):
        scenarios.append(('Base (Peer P/E)', ff['base']['pt'], ff['base']['pe'], COLORS['secondary']))
    if ff.get('bull', {}).get('pt'):
        scenarios.append(('Bull', ff['bull']['pt'], ff['bull']['pe'], '#1a7a4a'))
    if dcf_pt and dcf_pt > 0:
        scenarios.append(('DCF Intrinsic', dcf_pt, None, COLORS['primary']))

    labels = [s[0] for s in scenarios]
    values = [s[1] for s in scenarios]
    colors = [s[3] for s in scenarios]

    bars = ax.barh(labels, values, color=colors, alpha=0.82, height=0.5)
    ax.axvline(x=current_price, color='black', linewidth=1.5, linestyle='--', label=f'Current ${current_price:.2f}')

    for bar, val, scenario in zip(bars, values, scenarios):
        pe_str = f" ({scenario[2]:.1f}x)" if scenario[2] else ''
        ax.text(bar.get_width() + (max(values) * 0.01), bar.get_y() + bar.get_height() / 2,
                f'${val:.2f}{pe_str}', va='center', fontsize=9, fontweight='bold')

    ax.set_title('Football Field — Price Target Scenarios (Forward P/E vs DCF)',
                 fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax.set_xlabel('Implied Price (USD)')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(values) * 1.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_peer_pe_scatter(peer_df, subject_info: dict, subject_ticker: str) -> str:
    """Scatter plot: EPS growth vs. Forward P/E for all peers + subject."""
    if peer_df is None or peer_df.empty:
        return ''

    fig, ax = plt.subplots(figsize=(8, 4))

    valid = peer_df.dropna(subset=['fwd_pe', 'earn_growth_pct'])
    valid = valid[(valid['fwd_pe'] > 0) & (valid['fwd_pe'] < 150)]

    if not valid.empty:
        ax.scatter(valid['earn_growth_pct'], valid['fwd_pe'],
                   s=valid['market_cap_b'].clip(10, 3000) / 10,
                   alpha=0.7, color=COLORS['secondary'], label='Peers')

        for _, row in valid.iterrows():
            ax.annotate(row['ticker'], (row['earn_growth_pct'], row['fwd_pe']),
                        fontsize=7.5, ha='center', va='bottom', color='#374151')

    # Subject
    subj_pe = subject_info.get('forwardPE')
    subj_growth = (subject_info.get('earningsGrowth') or 0) * 100
    if subj_pe and subj_pe > 0:
        ax.scatter([subj_growth], [subj_pe], s=200, color=COLORS['accent'],
                   zorder=5, label=subject_ticker)
        ax.annotate(subject_ticker, (subj_growth, subj_pe),
                    fontsize=9, fontweight='bold', color=COLORS['accent'],
                    ha='center', va='bottom')

    # PEG=1 and PEG=2 reference lines
    x_range = np.linspace(max(ax.get_xlim()[0], 0), ax.get_xlim()[1], 100)
    ax.plot(x_range, x_range * 1, '--', color='gray', linewidth=0.8, alpha=0.6, label='PEG=1')
    ax.plot(x_range, x_range * 2, ':', color='gray', linewidth=0.8, alpha=0.6, label='PEG=2')

    ax.set_xlabel('EPS Growth (YoY %)')
    ax.set_ylabel('Forward P/E')
    ax.set_title('Peer Valuation Map — Forward P/E vs. EPS Growth',
                 fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_hist_pe_band(hist_pe: dict, current_pe: float, ticker: str) -> str:
    """Box-plot style visualization of 5yr P/E range."""
    if not hist_pe:
        return ''

    fig, ax = plt.subplots(figsize=(6, 2.5))

    lo = hist_pe.get('min', hist_pe.get('p25', 0))
    p25 = hist_pe.get('p25', 0)
    med = hist_pe.get('median', 0)
    p75 = hist_pe.get('p75', 0)
    hi = hist_pe.get('max', hist_pe.get('p75', 0))

    # Draw band
    ax.barh([0], [hi - lo], left=lo, height=0.4, color='#e8f0fe', alpha=0.8)
    ax.barh([0], [p75 - p25], left=p25, height=0.4, color=COLORS['secondary'], alpha=0.5, label='IQR (25-75%)')
    ax.axvline(x=med, color=COLORS['primary'], linewidth=2, label=f'Median {med:.1f}x')
    if current_pe:
        ax.axvline(x=current_pe, color=COLORS['accent'], linewidth=2,
                   linestyle='--', label=f'Current {current_pe:.1f}x')

    ax.set_yticks([])
    ax.set_xlabel('P/E Multiple')
    ax.set_title(f'{ticker} — 5-Year Historical P/E Range', fontsize=10,
                 fontweight='bold', color=COLORS['primary'])
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return _fig_to_b64(fig)


def _build_peer_pe_section(peer_model: dict, info: dict, ticker: str, dcf_pt: float = None) -> str:
    if not peer_model:
        return ''

    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0
    ff = peer_model.get('football_field') or {}
    peer_df = peer_model.get('peer_df')
    hist_pe = peer_model.get('hist_pe') or {}
    eps_data = peer_model.get('eps_data') or {}
    peg_result = peer_model.get('peg_result') or {}
    comp_table = peer_model.get('comp_table')
    peer_pe_stats = peer_model.get('peer_pe_stats') or {}

    html = '<div class="section"><div class="section-title">Peer-Based Forward P/E Valuation</div>'
    html += '<div style="font-size:8.5pt;color:#4b5563;margin-bottom:14px">Relative valuation anchored to peer multiples, PEG normalization, and 5-year historical P/E band. This method answers: <em>what multiple is the market willing to pay, given comparable companies?</em></div>'

    # EPS & Growth inputs
    eps_fwd = eps_data.get('eps_fwd_1y')
    eps_ttm = eps_data.get('eps_ttm')
    growth = peer_model.get('subject_growth_pct', 0)
    q_prem = peer_model.get('quality_premium', 0)
    eps_fwd_str = f"${eps_fwd:.2f}" if eps_fwd else "N/A"
    eps_ttm_str = f"${eps_ttm:.2f}" if eps_ttm else "N/A"
    html += f'''<div class="grid-3" style="margin-bottom:16px">
      <div class="kpi-card">
        <div class="kpi-label">Forward EPS (1yr)</div>
        <div class="kpi-value">{eps_fwd_str}</div>
        <div class="kpi-sub">Trailing EPS: {eps_ttm_str} &nbsp;|&nbsp; Source: {eps_data.get("eps_source","--")}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">EPS Growth Assumed</div>
        <div class="kpi-value">{growth:+.1f}%</div>
        <div class="kpi-sub">Quality premium: {q_prem:+.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Sector Median Fwd P/E</div>
        <div class="kpi-value">{peer_pe_stats.get("median","N/A")}x</div>
        <div class="kpi-sub">Range: {peer_pe_stats.get("p25","?")}x – {peer_pe_stats.get("p75","?")}x</div>
      </div>
    </div>'''

    # Football field chart
    if ff and eps_fwd:
        chart_ff = chart_football_field(ff, current_price, dcf_pt)
        html += f'<div class="chart-box"><img src="data:image/png;base64,{chart_ff}"/>'
        html += '<div class="chart-caption">Bear = peer P/E 25th pctile; Base = peer median + PEG-adjusted blend; Bull = peer P/E 75th pctile. DCF shown for reference.</div></div>'

        # Scenario table
        html += '<div style="margin-top:14px"><table><thead><tr><th>Scenario</th><th>P/E Multiple</th><th>Fwd EPS</th><th>Price Target</th><th>Upside / (Downside)</th><th>Rationale</th></tr></thead><tbody>'
        rationales = {
            'bear': 'Compression to bottom-quartile peer multiple; sector de-rating or earnings miss',
            'base': 'Blend of peer median (50%) + PEG-adjusted fair value (50%); base execution',
            'bull': 'Expansion to top-quartile peer multiple; beats estimates + re-rating catalyst',
        }
        colors_s = {'bear': '#b52a2a', 'base': COLORS['secondary'], 'bull': '#1a7a4a'}
        for s in ['bear', 'base', 'bull']:
            d = ff.get(s, {})
            pt = d.get('pt')
            up = d.get('upside')
            pe = d.get('pe')
            if pt:
                up_color = '#1a7a4a' if (up or 0) > 0 else '#b52a2a'
                html += (f'<tr><td style="color:{colors_s[s]};font-weight:700">{s.capitalize()}</td>'
                         f'<td>{pe:.1f}x</td><td>${eps_fwd:.2f}</td>'
                         f'<td><strong>${pt:.2f}</strong></td>'
                         f'<td style="color:{up_color}">{up:+.1f}%</td>'
                         f'<td style="font-size:8pt">{rationales[s]}</td></tr>')
        html += '</tbody></table></div>'

    # P/E scatter chart + hist band
    if peer_df is not None and not peer_df.empty:
        chart_scatter = chart_peer_pe_scatter(peer_df, info, ticker)
        current_pe = info.get('forwardPE')
        chart_band = chart_hist_pe_band(hist_pe, current_pe, ticker)

        html += '<div class="grid-2" style="margin-top:16px">'
        if chart_scatter:
            html += f'<div class="chart-box"><img src="data:image/png;base64,{chart_scatter}"/><div class="chart-caption">Bubble size = market cap. PEG=1 line: P/E equals growth rate (fair value). Above PEG=2: expensive relative to growth.</div></div>'
        if chart_band:
            html += f'<div class="chart-box"><img src="data:image/png;base64,{chart_band}"/><div class="chart-caption">5-year historical P/E band. Current P/E vs. own history shows expansion / compression vs. norms.</div></div>'
        html += '</div>'

    # PEG detail
    if peg_result:
        html += f'''<div style="margin-top:14px;padding:12px;background:#f5f7fa;border:1px solid #dce3ec;border-radius:5px;font-size:9pt;line-height:1.8">
          <strong>PEG Normalization Logic:</strong><br/>
          Sector median PEG: <strong>{peg_result.get("sector_median_peg","N/A")}x</strong> (from {peg_result.get("n_peers_used","?")} peers)<br/>
          Subject EPS growth: <strong>{peg_result.get("subject_growth_pct","?")}%</strong><br/>
          Raw fair P/E = PEG × growth = {peg_result.get("sector_median_peg","?")} × {peg_result.get("subject_growth_pct","?")} = <strong>{peg_result.get("raw_fair_pe","?")}x</strong><br/>
          Quality adjustment: <strong>{q_prem:+.1f}%</strong> (margin {(info.get("operatingMargins") or 0)*100:.1f}% vs. peer median)<br/>
          <strong>Adjusted fair P/E: {peg_result.get("fair_pe","?")}x</strong>
        </div>'''

    # Peer comparison table
    if comp_table is not None and not comp_table.empty:
        html += '<div style="margin-top:16px"><div style="font-weight:600;color:#0d3b6e;margin-bottom:6px">Peer Comparison Table</div>'
        html += '<table><thead><tr>'
        for col in comp_table.columns:
            html += f'<th>{col}</th>'
        html += '</tr></thead><tbody>'
        for i, row in comp_table.iterrows():
            is_subject = str(row.get('Ticker', '')).endswith('*')
            is_median = str(row.get('Ticker', '')) == 'SECTOR MED'
            style = ' style="background:#fff3cd;font-weight:700"' if is_subject else ' style="background:#f0f7ff"' if is_median else ''
            html += f'<tr{style}>'
            for val in row:
                html += f'<td>{val}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        html += '<div style="margin-top:4px;font-size:7.5pt;color:#9ca3af">* Subject company. Sector Med = median of peers. Source: Yahoo Finance.</div>'
        html += '</div>'

    html += '</div>'
    return html


# ─────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{ticker} — Equity Research Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; color: #1a1a2e; background: #fff; }}
  .page-wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}

  /* HEADER */
  .report-header {{ background: {primary}; color: white; padding: 28px 32px; border-radius: 6px 6px 0 0; margin-bottom: 0; }}
  .report-header .ticker {{ font-size: 28pt; font-weight: 700; letter-spacing: 1px; }}
  .report-header .company-name {{ font-size: 14pt; opacity: 0.88; margin-top: 2px; }}
  .report-header .meta {{ margin-top: 12px; font-size: 9pt; opacity: 0.75; }}
  .header-right {{ float: right; text-align: right; }}
  .rec-badge {{ display: inline-block; padding: 8px 20px; border-radius: 4px;
                font-size: 15pt; font-weight: 800; letter-spacing: 2px;
                background: {rec_color}; color: white; margin-bottom: 6px; }}
  .pt-label {{ font-size: 11pt; color: rgba(255,255,255,0.85); }}
  .pt-value {{ font-size: 20pt; font-weight: 700; color: white; }}
  .upside {{ font-size: 10pt; color: rgba(255,255,255,0.8); }}

  /* SECTION */
  .section {{ margin-top: 28px; }}
  .section-title {{ font-size: 12pt; font-weight: 700; color: {primary};
                    border-bottom: 2px solid {accent}; padding-bottom: 4px; margin-bottom: 14px; }}

  /* GRID */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }}

  /* KPI CARDS */
  .kpi-card {{ background: {light_bg}; border: 1px solid {border};
               border-radius: 5px; padding: 12px 16px; }}
  .kpi-label {{ font-size: 8pt; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 14pt; font-weight: 700; color: {primary}; margin-top: 2px; }}
  .kpi-sub {{ font-size: 8pt; color: #9ca3af; margin-top: 1px; }}

  /* TABLES */
  table {{ width: 100%; border-collapse: collapse; font-size: 9pt; }}
  thead {{ background: {primary}; color: white; }}
  th {{ padding: 8px 12px; text-align: left; font-weight: 600; font-size: 8.5pt; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #eaeaea; }}
  tr:nth-child(even) {{ background: {light_bg}; }}
  tr:last-child td {{ border-bottom: none; }}

  /* CHARTS */
  .chart-box img {{ width: 100%; border: 1px solid {border}; border-radius: 4px; }}
  .chart-caption {{ font-size: 8pt; color: #6b7280; margin-top: 4px; text-align: center; }}

  /* RISK TABLE */
  .risk-high {{ color: #b52a2a; font-weight: 600; }}
  .risk-med  {{ color: #d4900a; font-weight: 600; }}
  .risk-low  {{ color: #1a7a4a; font-weight: 600; }}

  /* INVESTMENT THESIS */
  .thesis-box {{ background: #f0f7ff; border-left: 4px solid {secondary}; padding: 14px 18px; border-radius: 0 5px 5px 0; }}
  .thesis-box p {{ margin: 6px 0; line-height: 1.6; }}
  .bull {{ color: {buy}; font-weight: 600; }}
  .bear {{ color: {sell}; font-weight: 600; }}

  /* DISCLAIMER */
  .disclaimer {{ margin-top: 32px; padding: 12px; background: #f9fafb; border-top: 1px solid {border};
                 font-size: 7.5pt; color: #9ca3af; line-height: 1.5; border-radius: 0 0 6px 6px; }}

  /* SCORE */
  .score-badge {{ display: inline-block; width: 44px; height: 44px; border-radius: 50%;
                  background: {secondary}; color: white; font-size: 13pt; font-weight: 700;
                  text-align: center; line-height: 44px; }}
  .clearfix::after {{ content: ''; display: table; clear: both; }}
  .tag {{ display: inline-block; padding: 2px 8px; background: {light_bg};
          border: 1px solid {border}; border-radius: 12px; font-size: 8pt; margin: 2px; }}
  .page-break {{ margin: 28px 0; border: none; border-top: 1px solid {border}; }}
  @media print {{
    body {{ font-size: 9pt; }}
    .page-wrap {{ padding: 12px; }}
  }}
</style>
</head>
<body>
<div class="page-wrap">

<!-- ════════════════════ HEADER ════════════════════ -->
<div class="report-header clearfix">
  <div style="float:left">
    <div class="ticker">{ticker}</div>
    <div class="company-name">{company_name}</div>
    <div style="margin-top:10px">
      <span class="tag" style="background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.3);color:white">{sector}</span>
      <span class="tag" style="background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.3);color:white">{industry}</span>
      <span class="tag" style="background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.3);color:white">{exchange}</span>
    </div>
    <div class="meta">Report Date: {report_date} &nbsp;|&nbsp; Data as of market close &nbsp;|&nbsp; Risk-Free Rate: {rf_rate}% &nbsp;|&nbsp; VIX: {vix:.1f}</div>
  </div>
  <div class="header-right">
    <div class="rec-badge">{recommendation}</div><br/>
    <div class="pt-label">12-Month Price Target</div>
    <div class="pt-value">${price_target:.2f}</div>
    <div class="upside">vs Current ${current_price:.2f} &nbsp; ({upside:+.1f}%)</div>
  </div>
</div>

{data_quality_banner}

<!-- ════════════════════ TRADE PLAN ════════════════════ -->
{trade_plan_section}

<!-- ════════════════════ KEY METRICS ════════════════════ -->
<div class="section">
  <div class="section-title">Key Financial Metrics</div>
  <div class="grid-3">
    {kpi_cards}
  </div>
</div>

<!-- ════════════════════ PRICE CHART ════════════════════ -->
<div class="section">
  <div class="chart-box">
    <img src="data:image/png;base64,{chart_price}"/>
    <div class="chart-caption">Source: Yahoo Finance. Price target represented by dashed gold line.</div>
  </div>
</div>

<!-- ════════════════════ INVESTMENT THESIS ════════════════════ -->
<div class="section">
  <div class="section-title">Investment Thesis</div>
  <div class="grid-2">
    <div>
      <div style="font-weight:600;color:{buy};margin-bottom:8px">▲ Bull Case</div>
      <div class="thesis-box">
        {bull_points}
      </div>
    </div>
    <div>
      <div style="font-weight:600;color:{sell};margin-bottom:8px">▼ Bear Case</div>
      <div class="thesis-box" style="border-left-color:{accent}">
        {bear_points}
      </div>
    </div>
  </div>
</div>

<hr class="page-break"/>

<!-- ════════════════════ DCF VALUATION ════════════════════ -->
<div class="section">
  <div class="section-title">Discounted Cash Flow Analysis</div>
  <div class="grid-2">
    <div>
      <table>
        <thead><tr><th>DCF Parameter</th><th>Value</th></tr></thead>
        <tbody>{dcf_table_rows}</tbody>
      </table>
    </div>
    <div>
      <table>
        <thead><tr><th>Value Bridge</th><th>Amount</th></tr></thead>
        <tbody>{dcf_bridge_rows}</tbody>
      </table>
    </div>
  </div>
  <div style="margin-top:16px" class="chart-box">
    <img src="data:image/png;base64,{chart_fcf}"/>
    <div class="chart-caption">Free Cash Flow: historical actuals and 10-year projection (Stage 1: high-growth; Stage 2: fade to terminal)</div>
  </div>
</div>

<!-- ════════════════════ PREDICTIVE SIGNALS ════════════════════ -->
{signals_section}

<!-- ════════════════════ SCENARIOS ════════════════════ -->
{scenario_section}

<!-- ════════════════════ DCF SENSITIVITY ════════════════════ -->
<div class="section">
  <div class="section-title">DCF Sensitivity — Driver-Based (Implied $/share)</div>
  <div class="grid-2">
    <div class="chart-box">
      <img src="data:image/png;base64,{chart_sensitivity}"/>
      <div class="chart-caption">Rows = Stage-1 revenue growth · Columns = terminal FCF margin. Green = higher implied value.</div>
    </div>
    <div>
      <p style="font-size:9pt;color:#4b5563;line-height:1.7;padding:8px">
        Unlike a generic WACC × terminal-growth grid, this table flexes the two drivers that
        actually move equity value for an operating business: <strong>near-term revenue growth</strong>
        and the <strong>terminal free-cash-flow margin</strong>. Each cell is recomputed with the same
        DCF engine as the base case, so the table is internally consistent with the headline number.
        <br/><br/>
        <strong>Base case WACC: {wacc_pct:.1f}%</strong> — CAPM (Blume-adjusted β={beta:.2f},
        Rf={rf_rate}%, ERP={erp_pct:.1f}%{crp_note}) blended with after-tax cost of debt.
      </p>
    </div>
  </div>
</div>

<hr class="page-break"/>

<!-- ════════════════════ COMPS ════════════════════ -->
<div class="section">
  <div class="section-title">Comparable Companies Analysis</div>
  <div class="grid-2">
    <div class="chart-box">
      <img src="data:image/png;base64,{chart_valuation_bridge}"/>
      <div class="chart-caption">Weighted average of valuation methodologies to derive 12-month price target</div>
    </div>
    <div>
      <table>
        <thead><tr><th>Method</th><th>Multiple</th><th>Implied Price</th><th>Weight</th></tr></thead>
        <tbody>{comps_rows}</tbody>
      </table>
      <div style="margin-top:12px;font-size:8.5pt;color:#4b5563">
        Sector: <strong>{sector}</strong><br/>
        {comps_source_note}
      </div>
    </div>
  </div>
</div>

<!-- ════════════════════ PEER P/E MODEL ════════════════════ -->
{peer_pe_section}

<hr class="page-break"/>

<!-- ════════════════════ REVENUE & MARGINS ════════════════════ -->
<div class="section">
  <div class="section-title">Revenue & Profitability Trends</div>
  <div class="chart-box">
    <img src="data:image/png;base64,{chart_revenue}"/>
    <div class="chart-caption">Annual revenue ($B) and operating margin trend. Source: SEC Filings via Yahoo Finance.</div>
  </div>
</div>

<hr class="page-break"/>

<!-- ════════════════════ FINANCIAL SUMMARY TABLE ════════════════════ -->
<div class="section">
  <div class="section-title">Financial Summary (Annual)</div>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        {financial_header_years}
        <th>LTM</th>
      </tr>
    </thead>
    <tbody>
      {financial_rows}
    </tbody>
  </table>
</div>

<!-- ════════════════════ WACC DECOMPOSITION ════════════════════ -->
<div class="section">
  <div class="section-title">WACC Decomposition</div>
  <div class="grid-3">
    {wacc_cards}
  </div>
</div>

<!-- ════════════════════ QUALITY SCORE ════════════════════ -->
<div class="section">
  <div class="section-title">Business Quality Assessment (Modified Piotroski)</div>
  <div class="grid-2">
    <div class="chart-box">
      <img src="data:image/png;base64,{chart_quality}"/>
      <div class="chart-caption">9-factor quality scoring: profitability, leverage, efficiency, growth</div>
    </div>
    <div>
      <table>
        <thead><tr><th>Quality Factor</th><th>Pass</th></tr></thead>
        <tbody>{quality_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- ════════════════════ RISK FACTORS ════════════════════ -->
<div class="section">
  <div class="section-title">Risk Factor Matrix</div>
  <table>
    <thead><tr><th>Risk Factor</th><th>Category</th><th>Severity</th><th>Description</th></tr></thead>
    <tbody>{risk_rows}</tbody>
  </table>
</div>

<!-- ════════════════════ ANALYST LANDSCAPE ════════════════════ -->
{analyst_landscape_section}

<!-- ════════════════════ OWNERSHIP ════════════════════ -->
{ownership_section}

<!-- ════════════════════ VALIDATION ════════════════════ -->
{validation_section}

<!-- ════════════════════ DISCLAIMER ════════════════════ -->
<div class="disclaimer">
  <strong>Important Disclosures:</strong> This report is generated for informational and educational purposes only. It does not constitute investment advice,
  a solicitation, or an offer to buy or sell any security. All valuations are based on publicly available financial data
  and standard financial modeling techniques. Forward-looking statements involve risks and uncertainties. Past performance
  is not indicative of future results. The price target represents a 12-month forward estimate based on the described
  methodologies. Investors should conduct their own due diligence and consult a licensed financial advisor before making
  investment decisions. Data sourced from Yahoo Finance. Report generated: {report_date}.
</div>

</div><!-- /page-wrap -->
</body>
</html>"""


# ─────────────────────────────────────────────
# HELPER: format numbers
# ─────────────────────────────────────────────
def _fmt_billions(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    if abs(v) >= 1e12:
        return f'${v/1e12:.2f}T'
    if abs(v) >= 1e9:
        return f'${v/1e9:.2f}B'
    if abs(v) >= 1e6:
        return f'${v/1e6:.1f}M'
    return f'${v:,.0f}'


def _fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v*100:.{decimals}f}%'


def _fmt_ratio(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.{decimals}f}x'


def _safe(v, fmt='ratio'):
    try:
        if v is None:
            return 'N/A'
        if isinstance(v, float) and np.isnan(v):
            return 'N/A'
        if fmt == 'pct':
            return _fmt_pct(v)
        if fmt == 'billions':
            return _fmt_billions(v)
        return _fmt_ratio(v)
    except Exception:
        return 'N/A'


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
def _build_kpi_cards(info: dict, performance: dict) -> str:
    cards = [
        ('Market Cap', _fmt_billions(info.get('marketCap')), ''),
        ('EV', _fmt_billions(info.get('enterpriseValue')), ''),
        ('P/E (TTM)', _safe(info.get('trailingPE')), ''),
        ('Fwd P/E', _safe(info.get('forwardPE')), ''),
        ('EV/EBITDA', _safe(info.get('enterpriseToEbitda')), ''),
        ('P/S', _safe(info.get('priceToSalesTrailing12Months')), ''),
        ('P/B', _safe(info.get('priceToBook')), ''),
        ('EPS (TTM)', f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else 'N/A', ''),
        ('Revenue (TTM)', _fmt_billions(info.get('totalRevenue')), ''),
        ('Gross Margin', _fmt_pct(info.get('grossMargins')), ''),
        ('Op. Margin', _fmt_pct(info.get('operatingMargins')), ''),
        ('Net Margin', _fmt_pct(info.get('profitMargins')), ''),
        ('ROE', _fmt_pct(info.get('returnOnEquity')), ''),
        ('ROA', _fmt_pct(info.get('returnOnAssets')), ''),
        ('ROIC', _fmt_pct(info.get('returnOnCapital') or info.get('returnOnEquity')), ''),
        ('D/E Ratio', _safe(info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else None), ''),
        ('Current Ratio', _safe(info.get('currentRatio')), ''),
        ('Rev. Growth (YoY)', _fmt_pct(info.get('revenueGrowth')), ''),
        ('Earnings Growth', _fmt_pct(info.get('earningsGrowth')), ''),
        ('Dividend Yield', _fmt_pct(info.get('dividendYield')), ''),
        ('1Y Return', f"{performance.get('1y', 'N/A')}%" if performance.get('1y') else 'N/A', ''),
        ('52W High', f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A', ''),
        ('52W Low', f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A', ''),
        ('Shares Out.', _fmt_billions(info.get('sharesOutstanding')).replace('$', ''), ''),
    ]

    html = ''
    for label, value, sub in cards:
        html += f'''<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
</div>'''
    return html


# ─────────────────────────────────────────────
# INVESTMENT THESIS (rule-based)
# ─────────────────────────────────────────────
def _build_thesis(info: dict, dcf: dict, quality: dict) -> tuple:
    bull = []
    bear = []

    rev_growth = info.get('revenueGrowth') or 0
    gross_margin = info.get('grossMargins') or 0
    op_margin = info.get('operatingMargins') or 0
    roe = info.get('returnOnEquity') or 0
    debt_to_eq = (info.get('debtToEquity') or 0) / 100
    current_ratio = info.get('currentRatio') or 1
    pe = info.get('trailingPE') or 0
    fcf_yield = 0
    mc = info.get('marketCap') or 1
    if dcf and dcf.get('base_fcf'):
        fcf_yield = dcf['base_fcf'] / mc

    if rev_growth > 0.10:
        bull.append(f"Strong revenue trajectory ({rev_growth*100:.1f}% YoY) indicates durable demand drivers")
    if gross_margin > 0.50:
        bull.append(f"High gross margin ({gross_margin*100:.1f}%) reflects strong pricing power and competitive moat")
    if op_margin > 0.15:
        bull.append(f"Solid operating leverage ({op_margin*100:.1f}% op. margin) driving earnings conversion")
    if roe > 0.15:
        bull.append(f"Superior capital allocation (ROE: {roe*100:.1f}%) vs. sector benchmarks")
    if fcf_yield > 0.03:
        bull.append(f"FCF yield of {fcf_yield*100:.1f}% provides valuation support and capital return optionality")
    if quality.get('score', 0) >= 7:
        bull.append("High Piotroski score indicates robust balance sheet and earnings quality")
    if dcf and dcf.get('stage1_growth', 0) > 0.10:
        bull.append(f"Compounding FCF growth ({dcf['stage1_growth']*100:.0f}% Stage 1) drives substantial terminal value creation")
    if not bull:
        bull.append("Company maintains stable cash flows and defensible market position")

    if debt_to_eq > 1.0:
        bear.append(f"Elevated leverage (D/E: {debt_to_eq:.1f}x) increases financial risk in rate-sensitive environment")
    if current_ratio < 1.2:
        bear.append(f"Tight liquidity (current ratio: {current_ratio:.1f}x) limits near-term financial flexibility")
    if pe > 40:
        bear.append(f"Premium valuation (P/E: {pe:.1f}x) embeds high execution expectations; re-rating risk if growth decelerates")
    if rev_growth < 0.03:
        bear.append("Slowing top-line growth may pressure margin expansion thesis")
    if op_margin < 0.05:
        bear.append("Thin operating margins leave limited buffer against cost pressures or demand softness")
    if dcf and dcf.get('tv_pct_of_total', 0) > 0.75:
        bear.append(f"Terminal value constitutes {dcf['tv_pct_of_total']*100:.0f}% of DCF — high sensitivity to long-run growth assumptions")
    if not bear:
        bear.append("Macro headwinds and sector competition may pressure near-term multiples")

    bull_html = ''.join(f'<p>▸ {p}</p>' for p in bull[:5])
    bear_html = ''.join(f'<p>▸ {p}</p>' for p in bear[:5])
    return bull_html, bear_html


# ─────────────────────────────────────────────
# RISK FACTORS (rule-based)
# ─────────────────────────────────────────────
def _build_risks(info: dict, dcf: dict, macro: dict) -> str:
    risks = []
    vix = macro.get('vix', 20)
    debt = info.get('debtToEquity', 0) or 0
    beta = info.get('beta') or 1.0

    risks.append(('Macroeconomic Slowdown', 'Macro', 'Medium', 'GDP deceleration could compress multiples and reduce consumer/enterprise spending'))
    risks.append(('Interest Rate Risk', 'Macro', 'High' if macro.get('risk_free_rate', 0.04) > 0.045 else 'Medium',
                  f"Rising rates (current 10yr: {macro.get('risk_free_rate',0.04)*100:.2f}%) increase discount rate, compressing DCF value"))
    if beta > 1.3:
        risks.append(('Market Volatility', 'Market', 'High', f"Beta {beta:.2f} implies amplified drawdowns vs. market in risk-off environments"))
    if debt / 100 > 1.5:
        risks.append(('Debt / Refinancing Risk', 'Financial', 'High', f"D/E {debt/100:.1f}x may face refinancing pressure in high-rate environment"))
    risks.append(('Competitive Disruption', 'Industry', 'Medium', 'New entrants, technological shifts, or pricing pressure from incumbents'))
    risks.append(('Regulatory / ESG', 'Regulatory', 'Medium', 'Evolving regulation (antitrust, data privacy, ESG mandates) could increase compliance costs'))
    risks.append(('Execution Risk', 'Operational', 'Medium', 'Failure to deliver on growth initiatives, M&A integration, or margin expansion targets'))
    if vix > 25:
        risks.append(('Elevated Market Volatility', 'Market', 'High', f"VIX at {vix:.1f} signals heightened uncertainty; liquidity premium may widen spreads"))
    if dcf and dcf.get('tv_pct_of_total', 0) > 0.75:
        risks.append(('Terminal Value Concentration', 'Valuation', 'High',
                      f">{dcf['tv_pct_of_total']*100:.0f}% of DCF value in terminal — small assumption change has outsized price impact"))

    severity_class = {'High': 'risk-high', 'Medium': 'risk-med', 'Low': 'risk-low'}
    rows = ''
    for risk, cat, sev, desc in risks:
        rows += f'<tr><td><strong>{risk}</strong></td><td>{cat}</td><td class="{severity_class.get(sev, "")}">{sev}</td><td>{desc}</td></tr>'
    return rows


# ─────────────────────────────────────────────
# FINANCIAL SUMMARY TABLE
# ─────────────────────────────────────────────
def _build_financial_table(income: pd.DataFrame, cashflow: pd.DataFrame, balance: pd.DataFrame, info: dict):
    metrics = {
        'Revenue': ('Total Revenue', income, 'billions'),
        'Gross Profit': ('Gross Profit', income, 'billions'),
        'EBIT': ('EBIT', income, 'billions'),
        'Net Income': ('Net Income', income, 'billions'),
        'Operating CF': ('Operating Cash Flow', cashflow, 'billions'),
        'CapEx': ('Capital Expenditure', cashflow, 'billions'),
        'Total Assets': ('Total Assets', balance, 'billions'),
        'Total Debt': ('Total Debt', balance, 'billions'),
    }

    header_years = ''
    all_data = {}
    years = []

    for label, (row_name, df, fmt) in metrics.items():
        if df is None or df.empty:
            continue
        for rn in [row_name, row_name.replace(' ', '_'), row_name.lower()]:
            if rn in df.index:
                series = df.loc[rn].dropna().sort_index(ascending=True)
                years = [str(d.year) for d in series.index]
                all_data[label] = (series.values, fmt)
                break

    if years:
        header_years = ''.join(f'<th>{y}</th>' for y in years[-4:])

    rows_html = ''
    for label, (vals, fmt) in all_data.items():
        row = f'<tr><td><strong>{label}</strong></td>'
        for v in vals[-4:]:
            row += f'<td>{_fmt_billions(v) if fmt == "billions" else _fmt_pct(v)}</td>'
        # LTM (use info dict for latest)
        row += '<td>—</td></tr>'
        rows_html += row

    # Add margin rows
    op_m = info.get('operatingMargins')
    net_m = info.get('profitMargins')
    rows_html += f'<tr><td><strong>Op. Margin</strong></td><td colspan="4" style="text-align:center">{_fmt_pct(op_m)} (TTM)</td></tr>'
    rows_html += f'<tr><td><strong>Net Margin</strong></td><td colspan="4" style="text-align:center">{_fmt_pct(net_m)} (TTM)</td></tr>'

    return header_years, rows_html


# ─────────────────────────────────────────────
# ANALYST LANDSCAPE SECTION (full consensus)
# ─────────────────────────────────────────────
def _build_analyst_landscape_section(consensus_result: dict, current_price: float) -> str:
    if not consensus_result:
        return ''

    analyst_data = consensus_result.get('analyst_data') or {}
    pt = analyst_data.get('price_targets') or {}
    cg = consensus_result.get('consensus_growth') or {}
    rs = consensus_result.get('rec_summary') or {}
    upgrades = consensus_result.get('upgrades') or []
    eps_signal = consensus_result.get('eps_revision_signal') or 'No data'
    cpe = consensus_result.get('consensus_pe') or {}
    cdcf = consensus_result.get('consensus_dcf') or {}
    composite = consensus_result.get('composite_pt') or {}

    html = '<div class="section"><div class="section-title">Sell-Side Analyst Landscape</div>'
    html += '<div style="font-size:8.5pt;color:#4b5563;margin-bottom:14px">Consensus estimates used as Stage 1 input in our DCF model. We do not adopt analyst targets uncritically — they anchor growth assumptions only.</div>'

    # PT distribution + rec distribution
    html += '<div class="grid-2" style="margin-bottom:16px">'

    # Price target range
    if pt:
        low_pt = pt.get('low', 0)
        mean_pt = pt.get('mean', 0)
        high_pt = pt.get('high', 0)
        median_pt = pt.get('median', 0)
        low_str = f"${low_pt:.2f}" if low_pt else '—'
        mean_str = f"${mean_pt:.2f}" if mean_pt else '—'
        high_str = f"${high_pt:.2f}" if high_pt else '—'
        median_str = f"${median_pt:.2f}" if median_pt else '—'
        mean_upside = round((mean_pt / current_price - 1) * 100, 1) if mean_pt and current_price else 0
        up_color = '#1a7a4a' if mean_upside > 10 else '#d4900a' if mean_upside > -5 else '#b52a2a'
        html += f'''<div class="kpi-card">
          <div class="kpi-label">Street Consensus Price Targets</div>
          <div style="margin-top:8px">
            <table><thead><tr><th>Low</th><th>Median</th><th>Mean</th><th>High</th></tr></thead>
            <tbody><tr>
              <td>{low_str}</td><td>{median_str}</td>
              <td style="font-weight:700;color:{up_color}">{mean_str} ({mean_upside:+.1f}%)</td>
              <td>{high_str}</td>
            </tr></tbody></table>
            <div style="margin-top:6px;font-size:8pt;color:#6b7280">vs Current: ${current_price:.2f}</div>
          </div>
        </div>'''

    # Rec distribution
    if rs:
        total = rs.get('total', 0)
        sb = rs.get('strong_buy', 0)
        b = rs.get('buy', 0)
        h = rs.get('hold', 0)
        s = rs.get('sell', 0)
        ss = rs.get('strong_sell', 0)
        bull = rs.get('bull_pct', 0)
        view = rs.get('street_view', '-')
        view_color = '#1a7a4a' if 'Bull' in view else '#d4900a' if 'Neutral' in view else '#b52a2a'
        html += f'''<div class="kpi-card">
          <div class="kpi-label">Analyst Recommendations (n={total})</div>
          <div class="kpi-value" style="color:{view_color}">{view}</div>
          <div style="margin-top:8px">
            <table><thead><tr><th>Str Buy</th><th>Buy</th><th>Hold</th><th>Sell</th><th>Str Sell</th><th>% Bull</th></tr></thead>
            <tbody><tr>
              <td style="color:#1a7a4a;font-weight:700">{sb}</td>
              <td style="color:#2d8a5e">{b}</td>
              <td style="color:#d4900a">{h}</td>
              <td style="color:#c0392b">{s}</td>
              <td style="color:#b52a2a;font-weight:700">{ss}</td>
              <td style="font-weight:700">{bull:.1f}%</td>
            </tr></tbody></table>
          </div>
        </div>'''

    html += '</div>'

    # Consensus EPS & Revenue estimates
    ee = analyst_data.get('earnings_estimate')
    re = analyst_data.get('revenue_estimate')

    if ee is not None and not ee.empty:
        html += '<div style="margin-bottom:16px"><div style="font-weight:600;color:#0d3b6e;margin-bottom:6px">Consensus EPS Estimates</div>'
        html += '<table><thead><tr><th>Period</th><th>Low</th><th>Avg</th><th>High</th><th>YoY Growth</th><th># Analysts</th></tr></thead><tbody>'
        period_labels = {'0q': 'Current Qtr', '+1q': 'Next Qtr', '0y': 'Current FY', '+1y': 'Next FY'}
        for period in ['0q', '+1q', '0y', '+1y']:
            if period in ee.index:
                row = ee.loc[period]
                avg = row.get('avg')
                lo = row.get('low')
                hi = row.get('high')
                gr = row.get('growth')
                n = row.get('numberOfAnalysts')
                gr_str = f"{gr*100:+.1f}%" if gr is not None else '—'
                gr_color = '#1a7a4a' if (gr or 0) > 0 else '#b52a2a'
                html += (f'<tr><td>{period_labels.get(period, period)}</td>'
                         f'<td>${lo:.2f}' if lo else '<td>—'
                         + f'</td><td style="font-weight:700">${avg:.2f}' if avg else '</td><td>—'
                         + f'</td><td>${hi:.2f}' if hi else '</td><td>—'
                         + f'</td><td style="color:{gr_color}">{gr_str}</td>'
                         + f'<td>{int(n) if n else "—"}</td></tr>')
        html += '</tbody></table>'
        html += f'<div style="margin-top:4px;font-size:8pt;color:#6b7280">EPS Revision Trend: <strong>{eps_signal}</strong></div>'
        html += '</div>'

    # Valuation model inputs derived from consensus
    html += '<div style="margin-bottom:16px"><div style="font-weight:600;color:#0d3b6e;margin-bottom:6px">How Consensus Feeds Our Model</div>'
    g1 = cg.get('rev_growth_y1') or cg.get('eps_growth_y1') or 0
    g2 = cg.get('rev_growth_y2') or cg.get('eps_growth_y2') or 0
    source = cg.get('source', '-')
    hist_cagr = cdcf.get('hist_cagr', 0)
    html += f'''<div style="background:#f5f7fa;border:1px solid #dce3ec;border-radius:5px;padding:12px 16px;font-size:9pt;line-height:1.9">
      <strong>Stage 1 growth (Y1):</strong> {g1*100:.1f}% &nbsp;[source: {source}]<br/>
      <strong>Stage 1 growth (Y2):</strong> {g2*100:.1f}%<br/>
      <strong>Blend-down anchor (Y3-Y5):</strong> {hist_cagr*100:.1f}% historical FCF CAGR<br/>
      <strong>Terminal growth:</strong> 2.5% (nominal GDP)<br/>
      <strong>WACC:</strong> see WACC section above<br/>
      <div style="margin-top:6px;font-size:8pt;color:#6b7280">
        We use analyst consensus revenue/EPS growth for Stage 1 only. Stage 2+ reverts to
        company-specific historical trends. We do NOT adopt analyst price targets — they serve as
        a 15% anchor in our composite PT to prevent model outliers, not as a valuation method.
      </div>
    </div>'''
    html += '</div>'

    # Composite PT breakdown
    if composite and composite.get('components'):
        html += '<div style="margin-bottom:16px"><div style="font-weight:600;color:#0d3b6e;margin-bottom:6px">Composite Price Target — Method Breakdown</div>'
        html += '<table><thead><tr><th>Method</th><th>Implied Price</th><th>Weight</th><th>Contribution</th></tr></thead><tbody>'
        for method, d in composite['components'].items():
            v = d['value']
            w = d['weight']
            contrib = round(v * w, 2)
            method_label = {
                'consensus_dcf': 'Consensus-Driven DCF',
                'consensus_fwd_pe': 'Consensus Fwd P/E × EPS',
                'sector_comps': 'Sector EV/EBITDA + P/E Comps',
                'street_consensus': 'Street Mean PT (anchor)',
            }.get(method, method)
            html += f'<tr><td>{method_label}</td><td>${v:.2f}</td><td>{w*100:.0f}%</td><td>${contrib:.2f}</td></tr>'
        html += f'<tr style="background:#e8f0fe;font-weight:700"><td>Composite PT</td><td>${composite["price_target"]:.2f}</td><td>100%</td><td>{composite["upside"]:+.1f}%</td></tr>'
        html += '</tbody></table></div>'

    # Recent upgrades / downgrades
    if upgrades:
        html += '<div><div style="font-weight:600;color:#0d3b6e;margin-bottom:6px">Recent Rating Actions</div>'
        html += '<table><thead><tr><th>Date</th><th>Firm</th><th>Action</th><th>To</th><th>From</th><th>New PT</th></tr></thead><tbody>'
        for u in upgrades:
            action = u.get('action', '')
            action_color = '#1a7a4a' if 'up' in action.lower() or 'init' in action.lower() else '#b52a2a' if 'down' in action.lower() else '#6b7280'
            pt_str = f"${u['new_pt']:.2f}" if u.get('new_pt') else '—'
            html += (f'<tr><td>{u["date"]}</td><td><strong>{u["firm"]}</strong></td>'
                     f'<td style="color:{action_color};font-weight:600">{action}</td>'
                     f'<td>{u["to"]}</td><td style="color:#9ca3af">{u["from"]}</td>'
                     f'<td>{pt_str}</td></tr>')
        html += '</tbody></table></div>'

    html += '</div>'
    return html


# ─────────────────────────────────────────────
# OWNERSHIP SECTION
# ─────────────────────────────────────────────
def _build_ownership_section(institutional: pd.DataFrame) -> str:
    if institutional is None or institutional.empty:
        return ''

    top5 = institutional.head(10)
    rows = ''
    for _, row in top5.iterrows():
        holder = row.get('Holder', row.get('Name', '—'))
        shares = row.get('Shares', row.get('Value', 0))
        pct = row.get('% Out', row.get('pctHeld', None))
        pct_str = f"{pct*100:.2f}%" if pct and not pd.isna(pct) else '—'
        rows += f'<tr><td>{holder}</td><td>{_fmt_billions(shares).replace("$","")}</td><td>{pct_str}</td></tr>'

    return f'''<div class="section">
    <div class="section-title">Institutional Ownership (Top 10)</div>
    <table>
      <thead><tr><th>Holder</th><th>Shares</th><th>% Owned</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>'''


# ─────────────────────────────────────────────
# VALIDATION SECTION HTML BUILDER
# ─────────────────────────────────────────────
def _build_validation_section(validation: dict, current_price: float) -> str:
    if not validation:
        return ''

    html = '<div class="section"><div class="section-title">Model Validation — Buy-Side Checks</div>'
    html += '<div style="font-size:8.5pt;color:#4b5563;margin-bottom:14px">Four independent checks that test whether the model numbers are internally consistent and market-grounded.</div>'

    # A. Reverse DCF
    rv = validation.get('reverse_dcf') or {}
    html += '<div style="margin-bottom:18px">'
    html += '<div style="font-weight:700;color:#0d3b6e;margin-bottom:6px">A. Reverse DCF — Implied Market Growth Rate</div>'
    if rv.get('implied_growth_stage1') is not None:
        g = rv['implied_growth_stage1']
        color = '#1a7a4a' if g < 15 else '#d4900a' if g < 25 else '#b52a2a'
        html += f'''<div class="grid-2">
          <div class="kpi-card">
            <div class="kpi-label">Market-Implied Stage-1 FCF Growth</div>
            <div class="kpi-value" style="color:{color}">{g:+.2f}%/yr</div>
            <div class="kpi-sub">FCF Yield at current price: {rv.get("fcf_yield_at_current", 0):.2f}%</div>
          </div>
          <div style="padding:12px;background:#f5f7fa;border:1px solid #dce3ec;border-radius:5px;font-size:9pt;line-height:1.7">
            <strong>What this means:</strong><br/>
            {rv.get("interpretation", "")}
          </div>
        </div>'''
    else:
        html += f'<div style="color:#6b7280;font-size:9pt">{rv.get("note", "Not available")}</div>'
    html += '</div>'

    # B. Implied multiples at PT
    im = validation.get('implied_multiples') or {}
    if im:
        html += '<div style="margin-bottom:18px">'
        html += '<div style="font-weight:700;color:#0d3b6e;margin-bottom:6px">B. Implied Multiples at Our Price Target</div>'
        html += '<table><thead><tr><th>Metric</th><th>Implied at PT</th><th>Sector Median</th><th>Premium / Discount</th><th>Verdict</th></tr></thead><tbody>'
        for metric, d in im.items():
            prem = d['premium_discount']
            verdict = 'RICH' if prem > 20 else 'FAIR' if abs(prem) <= 20 else 'CHEAP'
            v_color = '#b52a2a' if verdict == 'RICH' else '#1a7a4a' if verdict == 'CHEAP' else '#d4900a'
            prem_str = f"{prem:+.1f}%"
            html += (f'<tr><td>{metric}</td><td>{d["value"]:.1f}x</td>'
                     f'<td>{d["sector_median"]:.1f}x</td><td>{prem_str}</td>'
                     f'<td style="color:{v_color};font-weight:700">{verdict}</td></tr>')
        html += '</tbody></table></div>'

    # C. Sanity checks
    checks = validation.get('sanity_checks') or []
    if checks:
        html += '<div style="margin-bottom:18px">'
        html += '<div style="font-weight:700;color:#0d3b6e;margin-bottom:6px">C. Model Sanity Check — Calculated vs. SEC Filing Data</div>'
        html += '<table><thead><tr><th>Metric</th><th>Model Calculated</th><th>Reported (yfinance)</th><th>Delta</th><th>Status</th></tr></thead><tbody>'
        for row in checks:
            name, calc, reported, delta, status = row
            s_color = '#1a7a4a' if status == 'PASS' else '#d4900a' if status == 'REVIEW' else '#b52a2a'
            html += (f'<tr><td>{name}</td><td>{calc}</td><td>{reported}</td>'
                     f'<td>{delta or "—"}</td>'
                     f'<td style="color:{s_color};font-weight:700">[{status}]</td></tr>')
        passed = sum(1 for r in checks if r[4] == 'PASS')
        html += f'</tbody></table><div style="margin-top:6px;font-size:8.5pt;color:#4b5563">{passed}/{len(checks)} checks passed — confirms data integrity between our model and raw filings.</div>'
        html += '</div>'

    # D. Consensus bridge
    cb = validation.get('consensus_bridge') or {}
    if cb.get('consensus_mean'):
        html += '<div style="margin-bottom:18px">'
        html += '<div style="font-weight:700;color:#0d3b6e;margin-bottom:6px">D. Consensus Bridge — Our PT vs. Sell-Side Street</div>'
        gap = cb['pt_gap_pct']
        gap_color = '#1a7a4a' if gap > 0 else '#b52a2a'
        _stance = 'ABOVE' if gap > 5 else 'BELOW' if gap < -5 else 'IN LINE WITH'
        html += (f'<div style="font-size:9pt;color:#4b5563;margin-bottom:8px;line-height:1.6">'
                 f'Our price target is derived <strong>independently</strong> (DCF + PEG-anchored fwd P/E + comps) '
                 f'and is <strong>deliberately not blended with the street mean</strong> — anchoring to consensus '
                 f'would erase the variant view. We sit <strong style="color:{gap_color}">{_stance}</strong> the '
                 f'street; the attribution below explains why.</div>')
        html += f'''<div class="grid-2">
          <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
              <tr><td>Our Price Target</td><td><strong>${cb["our_pt"]:.2f}</strong> ({cb["our_upside"]:+.1f}%)</td></tr>
              <tr><td>Consensus Mean PT (n={cb["n_analysts"]})</td><td>${cb["consensus_mean"]:.2f} ({cb["consensus_upside"]:+.1f}%)</td></tr>
              <tr><td>Gap vs Consensus</td><td style="color:{gap_color};font-weight:700">${cb["pt_gap_vs_consensus"]:+.2f} ({gap:+.1f}%)</td></tr>
              {"<tr><td>Our DCF / Share</td><td>$" + f'{cb["our_dcf_per_share"]:.2f}' + "</td></tr>" if cb.get("our_dcf_per_share") else ""}
              {"<tr><td>Our Comps Average</td><td>$" + f'{cb["comps_average"]:.2f}' + "</td></tr>" if cb.get("comps_average") else ""}
            </tbody>
          </table>
          <div style="padding:12px;background:#f5f7fa;border:1px solid #dce3ec;border-radius:5px;font-size:9pt;line-height:1.7">
            <strong>Attribution:</strong><br/>
            {"<br/>".join(f"• {d}" for d in cb.get("key_differences", ["No material differences identified"])) or "No differences identified."}
          </div>
        </div>'''
        html += '</div>'

    html += '</div>'
    return html


# ─────────────────────────────────────────────
# MASTER RENDER FUNCTION
# ─────────────────────────────────────────────
def _build_trade_plan_section(plan: dict) -> str:
    """Executable playbook card: action, entry, ATR stop, targets, risk-based size, rules."""
    if not plan:
        return ''
    mode = plan.get('mode', 'wait')
    color = {'value': COLORS['buy'], 'momentum': '#1a6fa8', 'wait': COLORS['hold'],
             'exit': COLORS['sell'], 'blocked': '#6b7280'}.get(mode, '#6b7280')
    action = plan.get('action', '—')
    inval = ''.join(f'<li>{x}</li>' for x in plan.get('invalidation', []))

    # Build the metrics grid depending on whether it's an actionable entry
    if mode in ('value', 'momentum'):
        cap_note = ' (capped)' if plan.get('size_capped') else ''
        review = f"<div style='font-size:8pt;color:#9ca3af;margin-top:4px'>Next review: {plan['review_date']}</div>" if plan.get('review_date') else ''
        grid = f'''
        <div class="grid-3" style="margin-top:10px">
          <div class="kpi-card"><div class="kpi-label">Entry Zone</div>
            <div class="kpi-value" style="font-size:14pt">${plan['entry_low']}–${plan['entry_high']}</div>
            <div class="kpi-sub">不追高;區間內分批進</div></div>
          <div class="kpi-card"><div class="kpi-label">Stop Loss (−{plan['stop_atr_mult']}×ATR)</div>
            <div class="kpi-value" style="font-size:14pt;color:{COLORS['sell']}">${plan['stop']}</div>
            <div class="kpi-sub">每股風險 ${plan['risk_per_share']}</div></div>
          <div class="kpi-card"><div class="kpi-label">Targets</div>
            <div class="kpi-value" style="font-size:14pt;color:{COLORS['buy']}">${plan['target1']} / ${plan['target2']}</div>
            <div class="kpi-sub">R:R {plan.get('rr')} · {plan.get('target_note','')}</div></div>
        </div>
        <div class="grid-3" style="margin-top:10px">
          <div class="kpi-card"><div class="kpi-label">Position Size{cap_note}</div>
            <div class="kpi-value" style="font-size:14pt">{plan['shares']} sh</div>
            <div class="kpi-sub">${plan['position_value']:,.0f} · {plan['position_pct']}% of capital</div></div>
          <div class="kpi-card"><div class="kpi-label">Capital at Risk</div>
            <div class="kpi-value" style="font-size:14pt">${plan['dollar_risk']:,.0f}</div>
            <div class="kpi-sub">{plan['risk_per_trade_pct']:.1f}% of ${plan['account_size']:,.0f} a/c</div></div>
          <div class="kpi-card"><div class="kpi-label">Conviction · ATR</div>
            <div class="kpi-value" style="font-size:14pt">{plan['conviction']}/100</div>
            <div class="kpi-sub">ATR ${plan.get('atr')} ({(plan.get('atr_pct') or 0)*100:.1f}%/day){review}</div></div>
        </div>'''
    elif mode == 'wait':
        grid = f'''<div style="margin-top:10px;padding:12px;background:#fff7e6;border-radius:5px;font-size:9.5pt;color:#4b5563;line-height:1.7">
          <strong>觸發條件:</strong> {plan.get('watch_trigger','—')}<br/>
          {plan.get('reentry_note','')}</div>'''
    elif mode == 'exit':
        grid = f'''<div style="margin-top:10px;padding:12px;background:#fdecec;border-radius:5px;font-size:9.5pt;color:#4b5563;line-height:1.7">
          {plan.get('holder_action','不建立新多單。')}</div>'''
    else:  # blocked
        grid = f'''<div style="margin-top:10px;padding:12px;background:#f3f4f6;border-radius:5px;font-size:9.5pt;color:#4b5563">
          {plan.get('rationale','')}</div>'''

    return f'''<div class="section" style="border:2px solid {color};border-radius:8px;padding:16px;background:#fcfdff">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div class="section-title" style="margin:0;border:none">📋 Trade Plan — Executable Playbook</div>
    <div style="background:{color};color:white;padding:6px 16px;border-radius:5px;font-weight:700;font-size:11pt">{action}</div>
  </div>
  <div style="font-size:9pt;color:#4b5563;margin-top:8px;line-height:1.6">{plan.get('rationale','')}</div>
  {grid}
  <div style="margin-top:12px">
    <div style="font-weight:600;color:#0d3b6e;font-size:9pt;margin-bottom:4px">失效 / 紀律條件 (寫在進場前)</div>
    <ul style="margin:0 0 0 18px;font-size:8.8pt;color:#4b5563;line-height:1.7">{inval}</ul>
  </div>
  <div style="margin-top:8px;font-size:7.5pt;color:#9ca3af">
    部位以「每筆固定風險 = 帳戶 {plan.get('risk_per_trade_pct',1):.1f}%」反推;停損用 ATR 適應個股波動。帳戶規模可於 trade_plan.py 調整。
  </div>
</div>'''


def _build_signals_section(signal_profile: dict, upside: float, recommendation: str) -> str:
    """
    Predictive-signal panel + the value × momentum integration matrix. Valuation says
    cheap/expensive; signals say has-momentum/no-momentum. The intersection is the
    actual actionable view (a cheap stock that is also being upgraded > a cheap stock
    falling with estimate cuts).
    """
    if not signal_profile:
        return ''
    comps = signal_profile.get('components', {})
    conv = signal_profile.get('conviction', 50)
    comp_score = signal_profile.get('composite_score', 0.0)
    label = signal_profile.get('label', 'Neutral')
    weights = signal_profile.get('weights', {})

    pretty = {
        'price_momentum': '12-1 Price Momentum',
        'revision_momentum': 'Estimate-Revision Momentum',
        'earnings_surprise': 'Earnings Surprise (PEAD)',
        'quality': 'Quality (Piotroski)',
        'short_interest': 'Short Interest (籌碼)',
        'options_skew': 'Options Skew / IV',
        'short_term_reversal': 'Short-Term Reversal (1m)',
        'insider': 'Insider Net Buying',
        'low_volatility': 'Low-Volatility',
    }

    def _opt_detail(c):
        if not c.get('available'):
            return 'no data'
        parts = []
        if c.get('atm_iv') is not None:
            parts.append(f"IV {c['atm_iv']*100:.0f}%")
        if c.get('put_skew') is not None:
            parts.append(f"skew {c['put_skew']*100:+.1f}pp")
        if c.get('iv_vs_realized') is not None:
            parts.append(f"IV−RV {c['iv_vs_realized']*100:+.0f}pp")
        return ' · '.join(parts) if parts else 'available'

    def _si_detail(c):
        if not c.get('available'):
            return 'no data'
        parts = []
        if c.get('short_pct_float') is not None:
            parts.append(f"{c['short_pct_float']*100:.1f}% float")
        if c.get('short_ratio_days') is not None:
            parts.append(f"{c['short_ratio_days']:.1f}d cover")
        if c.get('squeeze_risk'):
            parts.append(f"squeeze:{c['squeeze_risk']}")
        return ' · '.join(parts) if parts else 'available'

    detail = {
        'price_momentum': lambda c: f"12-1: {(c.get('raw_12_1') or 0)*100:+.0f}% · 52w dist: {(c.get('dist_52w_high') or 0)*100:+.0f}%",
        'revision_momentum': lambda c: (f"{c.get('up',0)} up / {c.get('down',0)} down (30d)" if c.get('source')=='eps_revisions' else f"src: {c.get('source','—')}"),
        'earnings_surprise': lambda c: (f"last {c.get('last_surprise_pct')}% · beat rate {c.get('beat_rate')}" if c.get('last_surprise_pct') is not None else 'no data'),
        'quality': lambda c: f"F-score {c.get('f_score','—')}/9",
        'short_interest': _si_detail,
        'options_skew': _opt_detail,
        'short_term_reversal': lambda c: f"1m: {(c.get('ret_1m') or 0)*100:+.0f}%",
        'insider': lambda c: (f"{c.get('buys',0)} buys / {c.get('sells',0)} sells" if c.get('net_shares') is not None else 'no data'),
        'low_volatility': lambda c: f"σ {(c.get('realized_vol') or 0)*100:.0f}% ann." ,
    }

    rows = ''
    for k in weights:
        c = comps.get(k, {})
        sc = c.get('score', 0.0)
        bar_color = COLORS['buy'] if sc > 0.1 else COLORS['sell'] if sc < -0.1 else '#9aa3af'
        width = int(abs(sc) * 50)  # 0-50px each side
        left = f'<div style="height:9px;width:{width}px;background:{bar_color};margin-left:auto"></div>' if sc < 0 else ''
        right = f'<div style="height:9px;width:{width}px;background:{bar_color}"></div>' if sc >= 0 else ''
        bar = (f'<div style="display:flex;align-items:center;width:120px">'
               f'<div style="width:60px;display:flex;justify-content:flex-end">{left}</div>'
               f'<div style="width:1px;height:12px;background:#888"></div>'
               f'<div style="width:60px">{right}</div></div>')
        rows += (f'<tr><td>{pretty.get(k,k)}</td><td style="font-size:8pt;color:#6b7280">{detail[k](c)}</td>'
                 f'<td>{bar}</td><td style="text-align:right;font-weight:600;color:{bar_color}">{sc:+.2f}</td>'
                 f'<td style="text-align:right;color:#9aa3af">{weights[k]*100:.0f}%</td></tr>')

    # Value × momentum quadrant
    cheap = upside > 10
    expensive = upside < -10
    pos_mom = comp_score > 0.15
    neg_mom = comp_score < -0.15
    if cheap and pos_mom:
        quad, qcolor, qmsg = 'Cheap + Improving', COLORS['buy'], 'Best setup: undervalued AND momentum/estimates turning up. Highest-conviction long.'
    elif cheap and neg_mom:
        quad, qcolor, qmsg = 'Value Trap Risk', COLORS['hold'], 'Cheap but momentum/estimates falling — could stay cheap. Wait for a catalyst or stabilization.'
    elif expensive and pos_mom:
        quad, qcolor, qmsg = 'Momentum / Priced-for-Growth', COLORS['hold'], 'Expensive on fundamentals but strongly bid — momentum can persist; size carefully, valuation risk on any stumble.'
    elif expensive and neg_mom:
        quad, qcolor, qmsg = 'Avoid', COLORS['sell'], 'Worst setup: overvalued AND momentum/estimates rolling over. Highest-conviction avoid/short.'
    else:
        quad, qcolor, qmsg = 'Mixed / Neutral', '#6b7280', 'No strong edge from the value × momentum intersection.'

    conv_color = COLORS['buy'] if comp_score > 0.15 else COLORS['sell'] if comp_score < -0.15 else '#6b7280'
    return f'''<div class="section">
  <div class="section-title">Predictive Signals — Forward-Return Factors</div>
  <div style="font-size:8.5pt;color:#4b5563;margin-bottom:12px">
    These target <strong>future returns directly</strong> (not intrinsic value). Each factor is
    scored −1…+1; the composite is a weighted blend. Unlike the price target, these are the
    inputs with robust out-of-sample evidence for predicting 1–12 month returns — and they are
    designed to be backtested (see backtest.py).
  </div>
  <div class="grid-2">
    <div>
      <table>
        <thead><tr><th>Factor</th><th>Reading</th><th>Score</th><th style="text-align:right">↕</th><th style="text-align:right">Wt</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div>
      <div class="kpi-card" style="margin-bottom:10px">
        <div class="kpi-label">Signal Conviction (0–100)</div>
        <div class="kpi-value" style="color:{conv_color}">{conv}</div>
        <div class="kpi-sub">Composite {comp_score:+.2f} · {label}</div>
      </div>
      <div style="padding:12px;background:#f5f7fa;border:1px solid #dce3ec;border-radius:5px">
        <div style="font-size:8.5pt;color:#6b7280;margin-bottom:4px">Value × Momentum Intersection</div>
        <div style="font-weight:700;color:{qcolor};font-size:11pt;margin-bottom:4px">{quad}</div>
        <div style="font-size:9pt;color:#4b5563;line-height:1.6">{qmsg}</div>
        <div style="margin-top:8px;font-size:8pt;color:#9aa3af">
          Valuation: {upside:+.1f}% to target ({recommendation}) &nbsp;·&nbsp; Signals: {comp_score:+.2f}
        </div>
      </div>
    </div>
  </div>
</div>'''


def _build_scenario_section(consensus_result: dict, current_price: float) -> str:
    """Bear / Base / Bull intrinsic value, driven by growth & margin (not P/E percentiles)."""
    sc = (consensus_result or {}).get('consensus_dcf_scenarios') or {}
    bear, base, bull = sc.get('bear'), sc.get('base'), sc.get('bull')
    if not (bear and base and bull):
        return ''

    def _up(v):
        if not v or not current_price:
            return '—'
        return f"{(v/current_price - 1)*100:+.1f}%"

    def _row(name, s, color, assume):
        ps = s.get('intrinsic_per_share', 0)
        gm = f"{s.get('g_y1',0)*100:.1f}% → {s.get('terminal_growth',0.025)*100:.1f}%"
        mm = f"{s.get('base_margin',0)*100:.1f}% → {s.get('terminal_margin',0)*100:.1f}%"
        return (f'<tr><td><strong style="color:{color}">{name}</strong></td>'
                f'<td>{assume}</td><td>{gm}</td><td>{mm}</td>'
                f'<td><strong>${ps:.2f}</strong></td><td>{_up(ps)}</td></tr>')

    return f'''<div class="section">
  <div class="section-title">Scenario Analysis — Fundamental Drivers (Bear / Base / Bull)</div>
  <table>
    <thead><tr><th>Scenario</th><th>Driver Assumption</th><th>Rev Growth (Y1→term)</th>
    <th>FCF Margin (base→term)</th><th>Intrinsic $/sh</th><th>Upside</th></tr></thead>
    <tbody>
      {_row('Bear', bear, COLORS['sell'], 'Growth −4pp/yr · margin −3pp')}
      {_row('Base', base, COLORS['primary'], 'Consensus growth · trend margin')}
      {_row('Bull', bull, COLORS['buy'], 'Growth +4pp/yr · margin +3pp')}
    </tbody>
  </table>
  <div style="margin-top:10px;font-size:8.5pt;color:#4b5563">
    Scenarios shift the <strong>operating drivers</strong> (revenue growth and terminal FCF margin),
    then re-run the full DCF — not arbitrary P/E percentiles. This isolates how much of the valuation
    rests on operational execution versus multiple assumptions.
  </div>
</div>'''


def render_report(data: dict, valuations: dict, quality: dict,
                  price_target_data: dict, wacc_data: dict,
                  macro: dict, sensitivity_df: pd.DataFrame,
                  output_path: str, validation: dict = None) -> str:

    info = data['info']
    ticker = data['symbol']
    consensus_result = data.get('consensus_result') or {}
    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0
    price_target = price_target_data['price_target']
    recommendation = price_target_data['recommendation']
    upside = price_target_data['upside']

    print("  [+] Generating charts...")

    # Charts
    chart_price = chart_price_history(data['hist_1y'], info.get('longName', ticker),
                                      ticker, price_target, current_price)
    chart_revenue = chart_revenue_trend(data['income_stmt'], data['quarterly_income'])
    chart_quality = chart_quality_radar(quality)

    dcf = valuations.get('dcf') or {}
    chart_fcf = b''
    if dcf and dcf.get('projected_fcfs') and not data['cashflow'].empty:
        from data_fetcher import _extract_fcf_series
        fcf_hist = _extract_fcf_series(data['cashflow'])
        if not fcf_hist.empty:
            chart_fcf = chart_fcf_waterfall(fcf_hist, dcf['projected_fcfs'],
                                            pd.Timestamp.now().year - 1)
        else:
            chart_fcf = chart_fcf_waterfall(
                pd.Series([dcf.get('base_fcf', 0)]),
                dcf['projected_fcfs'], pd.Timestamp.now().year - 1)

    if not chart_fcf:
        # placeholder
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.text(0.5, 0.5, 'FCF data not available', ha='center', va='center', transform=ax.transAxes)
        chart_fcf = _fig_to_b64(fig)

    chart_sensitivity = chart_sensitivity_heatmap(sensitivity_df) if sensitivity_df is not None else ''

    comps = valuations.get('comps') or {}
    chart_val_bridge = ''
    if price_target_data.get('components'):
        chart_val_bridge = chart_valuation_bridge(
            price_target_data['components'], current_price, price_target)

    # KPI cards
    from data_fetcher import get_price_performance
    performance = get_price_performance(data['hist_1y'], data['hist_5y'])
    kpi_cards = _build_kpi_cards(info, performance)

    # Investment thesis
    bull_html, bear_html = _build_thesis(info, dcf, quality)

    # DCF table
    dcf_rows = ''
    if dcf:
        base_m = dcf.get('base_margin', dcf.get('fcf_margin', 0)) or 0
        term_m = dcf.get('terminal_margin', base_m) or 0
        sbc_pct = (consensus_result.get('base_fcf_data') or {}).get('sbc_pct_of_fcf', 0) or 0
        params = [
            ('Base FCF (normalized, SBC-adj.)', _fmt_billions(dcf.get('base_fcf', 0))),
            ('— SBC drag (% of FCF)', f"{sbc_pct*100:.0f}%"),
            ('Stage 1 Growth (Y1 / Y2)', f"{dcf.get('g_y1', 0)*100:.1f}% / {dcf.get('g_y2', 0)*100:.1f}%"),
            ('Growth source', str(dcf.get('growth_source', '—')).replace('_', ' ')),
            ('Mature Growth (Y3–5 anchor)', f"{dcf.get('mature_growth', 0)*100:.1f}%"),
            ('Terminal Growth Rate', f"{dcf.get('terminal_growth', 0.025)*100:.1f}%"),
            ('FCF Margin path (base → term.)', f"{base_m*100:.1f}% → {term_m*100:.1f}%"),
            ('WACC', f"{wacc_data.get('wacc', 0)*100:.2f}%"),
            ('Beta (Blume-adj. / raw)', f"{wacc_data.get('beta', 1):.2f} / {wacc_data.get('raw_beta', wacc_data.get('beta',1)):.2f}"),
            ('Historical Revenue CAGR', f"{dcf.get('historical_cagr', 0)*100:.1f}%"),
            ('TV as % of Total', f"{dcf.get('tv_pct_of_total', 0)*100:.0f}%"),
        ]
        dcf_rows = ''.join(f'<tr><td>{k}</td><td><strong>{v}</strong></td></tr>' for k, v in params)

    bridge_rows = ''
    if dcf:
        fx = data.get('fx_to_usd', 1.0)  # reporting-ccy net debt → USD for ADRs
        total_debt = (info.get('totalDebt') or 0) * fx
        cash = (info.get('totalCash') or 0) * fx
        shares = info.get('sharesOutstanding') or 1
        bridge = [
            ('PV Stage 1+2 FCF', _fmt_billions(dcf.get('pv_stage12', 0))),
            ('PV Terminal Value', _fmt_billions(dcf.get('pv_terminal', 0))),
            ('Enterprise Value', _fmt_billions((dcf.get('pv_stage12', 0) or 0) + (dcf.get('pv_terminal', 0) or 0))),
            ('Less: Net Debt', _fmt_billions(total_debt - cash)),
            ('Equity Value', _fmt_billions(dcf.get('total_equity_value', 0))),
            ('Shares Outstanding', _fmt_billions(shares).replace('$', '')),
            ('→ DCF Per Share', f"<strong style='color:{COLORS['primary']}'>${dcf.get('intrinsic_per_share', 0):.2f}</strong>"),
        ]
        bridge_rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in bridge)

    # Comps rows
    comps_rows = ''
    components = price_target_data.get('components', {})
    for method, comp_data in comps.items():
        mult = comp_data.get('multiple', 0)
        price = comp_data.get('implied_price', 0)
        w = components.get(method, {}).get('weight', 0) if components else 0
        comps_rows += f'<tr><td>{method.replace("_","/").upper()}</td><td>{mult:.1f}x</td><td>${price:.2f}</td><td>{w*100:.0f}%</td></tr>'

    if dcf and dcf.get('intrinsic_per_share'):
        w = components.get('dcf', {}).get('weight', 0) if components else 0
        comps_rows += f'<tr><td>DCF</td><td>—</td><td>${dcf["intrinsic_per_share"]:.2f}</td><td>{w*100:.0f}%</td></tr>'

    # Financial table
    fin_header, fin_rows = _build_financial_table(
        data['income_stmt'], data['cashflow'], data['balance_sheet'], info)

    # WACC cards
    wacc_card_items = [
        ('WACC', f"{wacc_data.get('wacc', 0)*100:.2f}%", 'Blended cost of capital'),
        ('Cost of Equity (CAPM)', f"{wacc_data.get('cost_of_equity', 0)*100:.2f}%", f"β={wacc_data.get('beta',1):.2f}, Rf={wacc_data.get('risk_free_rate',0.04)*100:.2f}%"),
        ('Cost of Debt (after-tax)', f"{wacc_data.get('cost_of_debt', 0)*(1-wacc_data.get('tax_rate',0.21))*100:.2f}%", f"Pre-tax: {wacc_data.get('cost_of_debt',0)*100:.2f}%"),
        ('Tax Rate', f"{wacc_data.get('tax_rate', 0)*100:.1f}%", 'Effective rate'),
        ('Equity Weight', f"{wacc_data.get('w_equity', 1)*100:.1f}%", 'E/(D+E)'),
        ('Debt Weight', f"{wacc_data.get('w_debt', 0)*100:.1f}%", 'D/(D+E)'),
    ]
    wacc_cards = ''.join(
        f'<div class="kpi-card"><div class="kpi-label">{l}</div>'
        f'<div class="kpi-value">{v}</div><div class="kpi-sub">{s}</div></div>'
        for l, v, s in wacc_card_items)

    # Quality rows
    q_rows = ''
    for factor, passed in quality['details'].items():
        icon = '✓' if passed else '✗'
        color = COLORS['buy'] if passed else COLORS['sell']
        q_rows += f'<tr><td>{factor}</td><td style="color:{color};font-weight:700">{icon}</td></tr>'

    # Risks
    risk_rows = _build_risks(info, dcf, macro)

    # Analyst landscape section (new consensus model)
    analyst_landscape_section = _build_analyst_landscape_section(
        data.get('consensus_result') or {}, current_price)

    # Ownership
    ownership_section = _build_ownership_section(data.get('institutional'))

    # Peer P/E section
    peer_model = data.get('peer_model') or {}
    dcf_pt = (valuations.get('dcf') or {}).get('intrinsic_per_share')
    peer_pe_section = _build_peer_pe_section(peer_model, info, ticker, dcf_pt)

    # Validation section
    validation_section = _build_validation_section(validation or {}, current_price)

    # Scenario section (driver-based bear/base/bull)
    scenario_section = _build_scenario_section(consensus_result, current_price)

    # Predictive signals section (value × momentum)
    signals_section = _build_signals_section(
        data.get('signal_profile') or {}, upside, recommendation)

    # Trade plan / playbook card
    trade_plan_section = _build_trade_plan_section(data.get('trade_plan') or {})

    # Data-quality banner (only shown when not clean)
    dq = data.get('data_quality') or {}
    data_quality_banner = ''
    if dq and dq.get('verdict') and dq['verdict'] != 'OK':
        is_bad = dq['verdict'] == 'UNRELIABLE'
        bg = '#fdecec' if is_bad else '#fff7e6'
        bd = '#b52a2a' if is_bad else '#d4900a'
        title = ('⛔ SOURCE DATA UNRELIABLE — rating suppressed' if is_bad
                 else '⚠️ DATA QUALITY: REVIEW — interpret with caution')
        flag_li = ''.join(f'<li>{f["check"]}: {f["detail"]}</li>' for f in dq.get('flags', []))
        data_quality_banner = (
            f'<div style="margin-top:16px;padding:12px 16px;background:{bg};'
            f'border:1px solid {bd};border-left:5px solid {bd};border-radius:5px">'
            f'<div style="font-weight:700;color:{bd};font-size:10.5pt">{title} '
            f'(quality score {dq.get("score","?")}/100)</div>'
            f'<ul style="margin:6px 0 0 18px;font-size:9pt;color:#4b5563">{flag_li}</ul>'
            f'<div style="margin-top:6px;font-size:8.5pt;color:#6b7280">'
            f'These are internal-consistency checks on the data feed (Yahoo Finance). '
            f'A failure means the inputs are corrupt, so any valuation below is unreliable '
            f'until the data is verified against a second source.</div></div>')

    # ERP / country-risk note + comps source note
    erp_pct = wacc_data.get('erp', 0.046) * 100
    crp = wacc_data.get('country_risk_premium', 0.0) or 0.0
    crp_note = f", +CRP {crp*100:.1f}%" if crp > 0 else ""
    comps = valuations.get('comps') or {}
    _live = any((v.get('source') == 'live_peers') for v in comps.values() if isinstance(v, dict))
    if _live:
        comps_source_note = ("P/E multiple from <strong>live peer median</strong> (current peer set), "
                             "with EV/EBITDA &amp; EV/Revenue from Damodaran sector medians as fallback. "
                             "Adjusted for company-specific growth and margin profile.")
    else:
        comps_source_note = ("Multiples from Damodaran sector medians (live peer data unavailable this run). "
                             "Adjusted for company-specific growth and margin profile.")

    html = HTML_TEMPLATE.format(
        ticker=ticker,
        company_name=info.get('longName') or info.get('shortName') or ticker,
        sector=info.get('sector') or '—',
        industry=info.get('industry') or '—',
        exchange=info.get('exchange') or '—',
        report_date=macro.get('fetch_date', 'N/A'),
        rf_rate=f"{macro.get('risk_free_rate', 0.04)*100:.2f}",
        vix=macro.get('vix', 20),
        recommendation=recommendation,
        rec_color=REC_COLOR.get(recommendation, COLORS['hold']),
        price_target=price_target,
        current_price=current_price,
        upside=upside,
        primary=COLORS['primary'],
        secondary=COLORS['secondary'],
        accent=COLORS['accent'],
        buy=COLORS['buy'],
        sell=COLORS['sell'],
        light_bg=COLORS['light_bg'],
        border=COLORS['border'],
        kpi_cards=kpi_cards,
        chart_price=chart_price,
        bull_points=bull_html,
        bear_points=bear_html,
        dcf_table_rows=dcf_rows,
        dcf_bridge_rows=bridge_rows,
        chart_fcf=chart_fcf if isinstance(chart_fcf, str) else chart_fcf.decode(),
        chart_sensitivity=chart_sensitivity,
        wacc_base=wacc_data.get('wacc', 0.09) * 100,
        wacc_pct=wacc_data.get('wacc', 0.09) * 100,
        beta=wacc_data.get('beta', 1.0),
        erp_pct=erp_pct,
        crp_note=crp_note,
        scenario_section=scenario_section,
        signals_section=signals_section,
        trade_plan_section=trade_plan_section,
        data_quality_banner=data_quality_banner,
        comps_source_note=comps_source_note,
        comps_rows=comps_rows,
        chart_valuation_bridge=chart_val_bridge,
        chart_revenue=chart_revenue,
        financial_header_years=fin_header,
        financial_rows=fin_rows,
        wacc_cards=wacc_cards,
        chart_quality=chart_quality,
        quality_rows=q_rows,
        risk_rows=risk_rows,
        analyst_landscape_section=analyst_landscape_section,
        ownership_section=ownership_section,
        peer_pe_section=peer_pe_section,
        validation_section=validation_section,
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  [+] Report saved → {output_path}")
    return output_path
