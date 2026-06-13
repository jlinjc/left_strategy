"""
Trade Plan / Playbook Generator
───────────────────────────────
Turns the dual-engine view (valuation + signals + data quality) into an EXECUTABLE,
rules-based plan so the investor acts with discipline instead of emotion:

  • Action          — what to do, derived from the value × momentum quadrant + conviction
  • Entry zone      — a price band (don't chase), not "buy at market"
  • Stop loss       — ATR-based, which also DEFINES the per-share risk
  • Targets / trim  — fair value (value trades) or R-multiples (momentum trades)
  • Position size   — RISK-BASED: risk a fixed % of capital per trade, so no single
                      loss can hurt you. This is where "底氣" (conviction to act) comes from.
  • Invalidation    — written-in-advance thesis-break rules (the discipline core)

All stops/sizes use ATR (Average True Range) so they adapt to each name's volatility.
Account assumptions are configurable (ACCOUNT_DEFAULTS).
"""
from __future__ import annotations
import math
import pandas as pd

ACCOUNT_DEFAULTS = {
    'account_size': 100_000.0,   # total investable capital (USD)
    'risk_per_trade': 0.01,      # fraction of capital risked per trade (1%)
    'max_position_pct': 0.15,    # cap any single position at 15% of capital
    'atr_period': 14,
    'regime_multiplier': 1.0,    # set by regime.py (1.0=RISK ON, 0.4=RISK OFF)
}


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float | None:
    """Average True Range in price units (volatility-adaptive stop unit)."""
    if hist is None or hist.empty or not {'High', 'Low', 'Close'} <= set(hist.columns):
        return None
    h, l, c = hist['High'], hist['Low'], hist['Close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    try:
        return float(atr)
    except Exception:
        return None


def _quadrant(upside, sig):
    if upside is None or sig is None:
        return 'neutral'
    cheap, expensive = upside > 10, upside < -10
    pos, neg = sig > 0.15, sig < -0.15
    if cheap and pos: return 'cheap_improving'
    if cheap and neg: return 'value_trap'
    if expensive and pos: return 'momentum_growth'
    if expensive and neg: return 'avoid'
    return 'neutral'


def build_trade_plan(data: dict, pt_data: dict, config: dict = None) -> dict:
    cfg = {**ACCOUNT_DEFAULTS, **(config or {})}
    info = data['info']
    price = (info.get('currentPrice') or info.get('regularMarketPrice')
             or info.get('previousClose') or 0)
    sp = data.get('signal_profile') or {}
    dq = data.get('data_quality') or {}
    conv = sp.get('conviction', 50)
    comp = sp.get('composite_score', 0.0)
    rev_score = (sp.get('components', {}).get('revision_momentum', {}) or {}).get('score')
    rec = pt_data.get('recommendation', 'HOLD')
    upside = pt_data.get('upside')
    pt = pt_data.get('price_target')
    quad = _quadrant(upside, comp)

    scen = (data.get('consensus_result') or {}).get('consensus_dcf_scenarios') or {}
    bull_pt = (scen.get('bull') or {}).get('intrinsic_per_share')

    atr = compute_atr(data.get('hist_1y'), cfg['atr_period'])
    atr_pct = (atr / price) if (atr and price) else None

    plan = {
        'price': round(price, 2) if price else None,
        'atr': round(atr, 2) if atr else None,
        'atr_pct': round(atr_pct, 4) if atr_pct else None,
        'quadrant': quad,
        'conviction': conv,
        'account_size': cfg['account_size'],
        'risk_per_trade_pct': cfg['risk_per_trade'] * 100,
        'regime_multiplier': cfg.get('regime_multiplier', 1.0),
    }

    # ── Gate: no actionable plan without trustworthy data / price / volatility ──
    if not dq.get('reliable', True) or rec == 'NO RATING':
        plan.update(action='NO TRADE', mode='blocked',
                    rationale='資料品質不可信 — 在驗證數據前不出手。',
                    invalidation=['資料品質回到 OK 後才重新評估'])
        return plan
    if not price or not atr:
        plan.update(action='NO TRADE', mode='blocked',
                    rationale='缺價格或波動資料,無法定停損/部位。',
                    invalidation=[])
        return plan

    # ── Action + risk multiplier by quadrant ──
    if quad == 'cheap_improving':
        action, mode, size_factor, k = 'BUY / 加碼（最佳組合）', 'value', 1.0, 2.5
        rationale = '便宜且訊號轉強 — 估值與動能同向,最高把握度的多頭。'
    elif quad == 'momentum_growth':
        action, mode, size_factor, k = '動能買進（縮量・估值偏貴）', 'momentum', 0.5, 2.0
        rationale = '基本面偏貴但被強力買進 — 動能可延續,縮小部位、用移動停損控估值風險。'
    elif quad == 'value_trap':
        action, mode, size_factor, k = '觀望 — 等催化劑', 'wait', 0.0, 2.5
        rationale = '便宜但訊號偏弱,可能繼續便宜。放觀察清單,等訊號轉正再進。'
    elif quad == 'avoid':
        action, mode, size_factor, k = '避開 / 持有者出場', 'exit', 0.0, 2.0
        rationale = '又貴又轉弱 — 最該避開;若持有,沿移動停損出場。'
    else:  # neutral on the value axis (roughly fairly valued)
        if comp > 0.15 and conv >= 58:
            # Strong signal on a fairly-valued name → momentum-style trial position
            action, mode, size_factor, k = '試單（合理價・訊號偏多）', 'momentum', 0.45, 2.0
            rationale = '估值大致合理但訊號明顯偏多 — 小量順勢試單,用移動停損控風險,確認再加。'
        elif comp < -0.15 or conv <= 40:
            action, mode, size_factor, k = '避開', 'exit', 0.0, 2.0
            rationale = '訊號偏弱,無進場優勢。'
        else:
            action, mode, size_factor, k = '中性觀望', 'wait', 0.0, 2.5
            rationale = '價值×動能無明顯優勢,等更清楚的訊號。'

    plan.update(action=action, mode=mode, rationale=rationale)

    # Conviction scales the risk budget; regime_multiplier further scales by market conditions
    conv_scale = max(min(conv / 80.0, 1.2), 0.4)
    regime_mult = cfg.get('regime_multiplier', 1.0)
    risk_mult = size_factor * conv_scale * regime_mult

    # ── Common invalidation / discipline rules (always written in advance) ──
    inval = []

    if mode in ('value', 'momentum'):
        ref = price
        stop = round(ref - k * atr, 2)
        risk_per_share = ref - stop
        entry_low = round(ref - 0.75 * atr, 2)
        entry_high = round(ref, 2)

        # Targets
        if mode == 'value':
            t1 = pt if (pt and pt > ref) else round(ref * 1.15, 2)
            t2 = bull_pt if (bull_pt and bull_pt > t1) else round(ref * 1.30, 2)
            target_note = '目標 = 合理價 / Bull 情境;到價分批減碼。'
        else:  # momentum — fair value is below price, so trade R-multiples + trailing stop
            t1 = round(ref + 2 * risk_per_share, 2)   # +2R
            t2 = round(ref + 4 * risk_per_share, 2)   # +4R
            target_note = '目標用 R 倍數(估值已偏貴);跌破移動停損即出,讓獲利奔跑。'

        rr = round((t1 - ref) / risk_per_share, 2) if risk_per_share > 0 else None

        # Risk-based position size
        risk_budget = cfg['account_size'] * cfg['risk_per_trade'] * risk_mult
        shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
        # Cap by max position value
        max_shares_by_cap = int(cfg['account_size'] * cfg['max_position_pct'] / ref) if ref else 0
        capped = shares > max_shares_by_cap
        shares = min(shares, max_shares_by_cap)
        pos_value = shares * ref
        pos_pct = pos_value / cfg['account_size'] if cfg['account_size'] else 0
        dollar_risk = shares * risk_per_share

        plan.update(
            entry_low=entry_low, entry_high=entry_high,
            stop=stop, stop_atr_mult=k,
            risk_per_share=round(risk_per_share, 2),
            target1=t1, target2=t2, target_note=target_note, rr=rr,
            shares=shares, position_value=round(pos_value, 0),
            position_pct=round(pos_pct * 100, 1),
            dollar_risk=round(dollar_risk, 0),
            size_capped=capped,
        )
        inval.append(f'跌破停損 ${stop}(−{k}×ATR)→ 無條件出場')
        inval.append('conviction 跌破 45 → 部位減半;跌破 35 → 出清')
        if rev_score is not None:
            inval.append('調升動能(分析師上修)轉負 → 立即複查論點')
        if mode == 'value':
            inval.append('上漲空間收斂到 <5%(接近合理價)→ 開始分批減碼')
        else:
            inval.append('跌破 ATR 移動停損 → 出場(動能交易不凹單)')
        inval.append('資料品質轉 REVIEW/UNRELIABLE → 暫停加碼')
        if cfg.get('regime_multiplier', 1.0) < 1.0:
            regime_label = 'RISK OFF' if cfg['regime_multiplier'] <= 0.4 else 'CAUTION'
            inval.append(f'市場體制 {regime_label} ({cfg["regime_multiplier"]:.0%} sizing) — 部位已依體制縮減')

    elif mode == 'wait':
        # Watch trigger to convert to a buy
        trigger_px = round(price * 1.0, 2)
        plan.update(
            watch_trigger='conviction 升破 55 或 調升動能轉正 → 轉為買進候選',
            reentry_note=f'目前不進場;放觀察清單。若便宜(現價 ${round(price,2)})且訊號轉強再依價值劇本進場。',
        )
        inval.append('訊號持續轉弱 / upside 消失 → 移出觀察清單')

    else:  # exit
        stop = round(price - k * atr, 2)
        plan.update(
            exit_note='不建立新多單。',
            holder_stop=stop,
            holder_action=f'若持有:沿移動停損 ${stop}(−{k}×ATR)出場,反彈不加碼。',
        )
        inval.append('若訊號意外轉強(conviction>60 且 upside>10%)→ 重新評估')

    plan['invalidation'] = inval

    # Next review anchor (earnings if available)
    et = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
    if et:
        try:
            from datetime import datetime, timezone
            plan['review_date'] = datetime.fromtimestamp(et, tz=timezone.utc).strftime('%Y-%m-%d') + ' (財報)'
        except Exception:
            pass
    return plan


def format_plan_console(plan: dict) -> str:
    a = plan.get('action', '—')
    if plan.get('mode') in ('value', 'momentum'):
        return (f"  [PLAN] {a} | 進場 ${plan['entry_low']}–${plan['entry_high']} | "
                f"停損 ${plan['stop']} | 目標 ${plan['target1']}/${plan['target2']} | "
                f"R:R {plan.get('rr')} | 部位 {plan['shares']}股 (${plan['position_value']:,.0f}, "
                f"{plan['position_pct']}%) 風險 ${plan['dollar_risk']:,.0f}")
    return f"  [PLAN] {a} — {plan.get('rationale','')}"
