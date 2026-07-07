"""
Capitulation Engine  (投降引擎 — 乾淨重寫版,只保留被回測驗證的邏輯)
═══════════════════════════════════════════════════════════════════════════════
This is the clean rewrite. The old bottom_fishing.py grew into a palimpsest of ideas, most
of which our own attribution proved had NO edge (the RSI(2)<10 "casual oversold" core, the
swing scalp, the many overlapping filters). We threw that away. What SURVIVED brutal,
honest backtesting is one narrow but real thing, and this module is ONLY that:

  THE ONE VALIDATED EDGE — buy TRUE CAPITULATION (not routine oversold), after the TURN
  (never a falling knife), heaviest when the WHOLE crowd panics (VIX), in a company whose
  quality lets you HOLD through the pain; then RIDE the fear→greed recovery to a satisfaction
  point — never scalp the relief bounce.

  Backtest (capitulation_bt.py, 84 names, 15y, after costs):
     deep capit + climax + turn + VIX≥25 → 36 trades, +14.1%/trade, PF 5.39
     deep capit + climax + turn (no VIX) → ~98 trades, +6.9%/trade, PF 2.87
     vs the old core RSI2<10            → 7441 trades, +0.08%/trade, PF 1.06 (no edge)

HONEST POSITIONING (see STRATEGY_TRUTH.md): this is NOT a market-beating alpha engine and
does NOT beat DCA on return in a bull. It is a RARE-EVENT, anti-human discipline tool that
deploys cash into genuine panic with controlled risk. Use it as a satellite overlay on a
DCA core, not as a standalone wealth engine.

Every decision is tied to the CROWD'S EMOTION (fear↔greed). Output is an executable playbook.

CLI:  python capitulation_engine.py NVDA AMD TSM
      python capitulation_engine.py --watchlist candidates.txt
"""
from __future__ import annotations
import os, json, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

TAIPEI = timezone(timedelta(hours=8))   # 網站顯示一律用台北時間(CI 跑在 UTC)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Account / sizing ──
ACCOUNT = {'account_size': 100_000.0, 'risk_per_trade': 0.01, 'max_position_pct': 0.15}

# ── The validated capitulation knobs (sweet spot ③/④ from the loosening sweep) ──
CAPIT_DD       = -0.15    # ≥15% off the 252d high  (real damage, not a routine dip)
CAPIT_RSI2     = 5.0      # extreme oversold in the last 5 sessions (relentless waterfall)
CAPIT_CLIMAX   = 2.0      # a down day with ≥2.0× its 50d-avg volume (the panic puke)
CAPIT_FEAR_VIX = 25.0     # whole-market fear → full size. Backtest (champion_opt_bt.py): the
                          # champion engulfing-capitulation jumps PF 3.7→6.1 at VIX≥25 and
                          # →6.74 / 63% win / +17%/trade at VIX≥30 (the strongest single knob).
CAPIT_FEAR_EXTREME = 30.0 # VIX≥30 + A-grade entry = the "世紀級投降" top tier
# ── Exit (hold to satisfaction) ──
TRIM_AT_BOUNCE = 1.0 / 3  # bank only a third at the relief bounce; ride the rest
ATR_TRAIL      = 3.0      # chandelier: exit core if it falls 3·ATR from its highest close
TIME_STOP_DAYS = 252      # a position trade — give the recovery up to a year
MIN_DOLLAR_ADV = 5e6      # below this you cannot execute cleanly → no trade

# ── Index / broad-ETF mode ──
# Broad indices can't go to zero, so BOTH the fundamental "knife" gate and the hard stop are
# WRONG for them: a dead constituent gets swapped out, and SELLING an index just forfeits its
# guaranteed long-run drift (dip_hold_test.py: on QQQ/SPY/SOXX, buy-the-capitulation-and-NEVER-
# sell beats every stop-and-trim variant). So in index mode we BUY the capitulation, ADD on
# further weakness, and HOLD forever — the position folds into the long-term core. Auto-detected
# for common broad ETFs below; force with --index.
INDEX_TICKERS = {
    # ── 美股寬基指數 ETF ──
    'SPY','QQQ','QQQM','DIA','IWM','VOO','VTI','VT','RSP','SCHB','SCHX','ITOT','IVV','SPLG',
    # ── 美股產業 ETF ──
    'SOXX','SMH','XLK','VGT','IYW',
    'XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC',
    'IBB','XBI',
    # ── 美股國際 ETF ──
    'EEM','EFA','VEA','VWO','IEFA','IEMG','ACWI',
    'KWEB','FXI','MCHI','EWT','EWJ','EWZ','INDA','EWY',
    # ── 美股債/商品 ETF ──
    'TLT','IEF','HYG','LQD','GLD','SLV',
    # ── 台股指數 ETF(.TW) ──
    '0050.TW','0051.TW','0056.TW','006208.TW','00878.TW','00881.TW',
    '00692.TW','00850.TW',
}


def is_index(ticker: str) -> bool:
    return (ticker or '').upper() in INDEX_TICKERS

def is_tw_stock(ticker: str) -> bool:
    """非 ETF 的台股個股(.TW 結尾但不在 INDEX_TICKERS 裡)"""
    t = (ticker or '').upper()
    return t.endswith('.TW') and t not in INDEX_TICKERS


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS  (point-in-time, daily OHLCV)
# ═══════════════════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0.0); l = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100/(1+rs)).where(al != 0, 100.0)


def _atr(df: pd.DataFrame, period: int = 14):
    h, l, c = df['High'], df['Low'], df['Close']; pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().iloc[-1]
    try: return float(a)
    except Exception: return None


def _last(s, d=None):
    try:
        v = s.dropna().iloc[-1]
        return float(v) if not (isinstance(v, float) and math.isnan(v)) else d
    except Exception: return d


def _ma(close, n):
    return float(close.rolling(n).mean().iloc[-1]) if len(close) >= n else None


def _pct(v):
    return f"{v*100:+.1f}%" if v is not None else "—"


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL READ  (only the fields the capitulation logic needs)
# ═══════════════════════════════════════════════════════════════════════════════

def technical_read(hist_1y, hist_5y) -> dict:
    base = hist_5y if (hist_5y is not None and len(hist_5y) > 260) else hist_1y
    if base is None or base.empty:
        return {}
    # 台股(.TW)yfinance 常在尾端塞一列「當日未收盤」的 NaN Close,會讓 price=iloc[-1]=NaN,
    # 連帶距高/RSI/量能全部 NaN(整個台股投降偵測失效)。先去掉 Close 為 NaN 的列再算。
    base = base[base['Close'].notna()]
    if base.empty or len(base) < 20:
        return {}
    close = base['Close']; high = base.get('High', close); low = base.get('Low', close)
    vol = base.get('Volume')
    price = float(close.iloc[-1])
    o = {'price': round(price, 2)}

    o['ma10'] = _ma(close, 10); o['ma50'] = _ma(close, 50); o['ma200'] = _ma(close, 200)
    o['dist_ma200'] = round(price/o['ma200'] - 1, 4) if o['ma200'] else None
    mid = close.rolling(20).mean(); sd = close.rolling(20).std(ddof=0)
    o['boll_mid'] = _last(mid); o['boll_lower'] = _last(mid - 2*sd)

    hi252 = float(close.tail(252).max()); lo252 = float(close.tail(252).min())
    o['high_52w'] = round(hi252, 2); o['low_52w'] = round(lo252, 2)
    o['dist_52w_high'] = round(price/hi252 - 1, 4)
    o['swing_low'] = round(float(low.tail(60).min()), 2)
    o['rsi14'] = _last(_rsi(close, 14))
    r2 = _rsi(close, 2); o['rsi2'] = _last(r2)
    o['rsi2_min_5d'] = round(float(r2.tail(5).min()), 1)
    o['atr'] = round(_atr(base, 14) or 0, 2) or None
    o['atr_pct'] = round(o['atr']/price, 4) if o['atr'] else None

    # volume climax = a down day with ≥ CAPIT_CLIMAX × 50d-avg volume in the last 5 sessions
    o['vol_climax_x'] = None
    if vol is not None and len(vol.dropna()) >= 50:
        avg50 = float(vol.tail(50).mean())
        o['vol_climax_x'] = round(float(vol.tail(10).max())/avg50, 2) if avg50 > 0 else None
        down = close < close.shift(1)
        spike = (vol >= CAPIT_CLIMAX*vol.rolling(50).mean()) & down
        o['climax_5d'] = bool(spike.tail(5).any())
    else:
        o['climax_5d'] = False

    # THE TURN — buyers step in, while the crowd is still terrified. Which confirmation we
    # accept is BACKTEST-VALIDATED (entry_quality_bt.py): bullish ENGULFING (PF 1.99→3.12,
    # MAE −45%→−26%) and RSI DIVERGENCE (MAE −23%) are A-grade; the simple turn / 10dma
    # reclaim are B-grade (PF ~2). Hammer & volume dry-up tested NO better → deliberately NOT
    # used (sophistication only where it earns its keep). entry_grade surfaces the quality.
    try:
        prev_c = float(close.iloc[-2]); prev_h = float(high.iloc[-2]); prev_o = float(base['Open'].iloc[-2])
        op = float(base['Open'].iloc[-1])
        turn = bool(price > prev_c and price > prev_h)
        engulf = bool(price > op and prev_c < prev_o and price >= prev_o and op <= prev_c)
        r14 = _rsi(close, 14); cc = close.tail(20).values; rr = r14.tail(20).values
        diverg = False
        if len(cc) >= 10:
            half = len(cc)//2
            i1 = int(np.argmin(cc[:half])); i2 = int(np.argmin(cc[half:]))+half
            diverg = bool(cc[i2] < cc[i1] and rr[i2] > rr[i1])
        reclaim10 = bool(o['ma10'] and price > o['ma10'] and prev_c < o['ma10'])
        o['bull_engulfing'] = engulf; o['rsi_divergence'] = diverg
        o['the_turn'] = bool(turn or engulf or diverg or reclaim10)
        o['entry_grade'] = ('A' if (engulf or diverg) else ('B' if (turn or reclaim10) else '—'))
        o['turn_kind'] = ('多頭吞噬' if engulf else 'RSI底背離' if diverg else
                          '站回昨高' if turn else '收復10日線' if reclaim10 else '—')
    except Exception:
        o['the_turn'] = False; o['entry_grade'] = '—'; o['turn_kind'] = '—'

    # idiosyncratic context (down-days streak — light, for the narrative)
    rets = close.pct_change().dropna(); n = 0
    for x in reversed(rets.values):
        if x < 0: n += 1
        else: break
    o['down_days_streak'] = n

    # ── The capitulation signal ──
    dd = o['dist_52w_high']; cx = o['vol_climax_x']
    o['capit_setup'] = bool(
        (dd is not None and dd <= CAPIT_DD)
        and (o['rsi2_min_5d'] is not None and o['rsi2_min_5d'] < CAPIT_RSI2)
        and (o.get('climax_5d') or (cx or 0) >= CAPIT_CLIMAX)
    )
    o['capitulation_buy'] = bool(o['capit_setup'] and o['the_turn'])
    return o


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE  (the thing that lets you HOLD through −30% — psychology, not just risk)
# ═══════════════════════════════════════════════════════════════════════════════

def quality_gate(info: dict, quality: dict, index_mode: bool = False,
                 ticker: str = '') -> dict:
    # Indices can't go to zero → no fundamental gate, never a knife.
    if index_mode:
        return {'score': 100, 'knife_risk': False, 'f_score': None,
                'fcf_positive': True, 'reasons': ['指數/廣基ETF — 不會歸零,免基本面關卡']}
    score = 50.0; reasons = []

    # ── 市值門檻:太小流動性差、歸零風險高 ──────────────────────────────
    mcap = info.get('marketCap')
    if mcap is not None:
        is_tw = (ticker or '').upper().endswith('.TW')
        # 台股 marketCap 單位為 TWD,除以 32 換算 USD
        mcap_usd = mcap / 32 if is_tw else mcap
        if mcap_usd < 2e9:
            score -= 25; reasons.append(f'市值 ${mcap_usd/1e9:.1f}B 過小,流動性/倒閉風險高')
    f = (quality or {}).get('score')
    if f is not None:
        score += (f-5)*6
        if f <= 3: reasons.append(f'Piotroski F={f} 偏弱')
        elif f >= 7: reasons.append(f'Piotroski F={f} 強健')
    fcf = info.get('freeCashflow')
    if fcf is not None:
        score += 8 if fcf > 0 else -15
        if fcf <= 0: reasons.append('自由現金流為負 — 燒錢')
    de = info.get('debtToEquity')
    if de is not None:
        de = de/100 if de > 5 else de
        if de > 2.0: score -= 12; reasons.append(f'負債權益 {de:.1f}x 高')
        elif de < 0.5: score += 6
    pm = info.get('profitMargins')
    if pm is not None and pm < -0.10:
        score -= 12; reasons.append('淨利率深度為負')
    eg = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth')
    if eg is not None and eg < -0.30:
        score -= 10; reasons.append(f'盈餘年減 {eg*100:.0f}%')
    score = max(0.0, min(100.0, score))
    return {'score': round(score), 'knife_risk': score < 50, 'f_score': f,
            'fcf_positive': bool(fcf is not None and fcf > 0), 'reasons': reasons}


def tradeable(info: dict, tech: dict, hist_1y) -> dict:
    price = tech.get('price'); avg_vol = None
    try:
        if hist_1y is not None and 'Volume' in hist_1y.columns:
            avg_vol = float(hist_1y['Volume'].dropna().tail(20).mean())
    except Exception:
        avg_vol = None
    avg_vol = avg_vol or info.get('averageVolume10days') or info.get('averageVolume')
    adv = float(avg_vol)*float(price) if (avg_vol and price) else None
    ok = not (adv is not None and adv < MIN_DOLLAR_ADV)
    return {'dollar_adv': round(adv, 0) if adv else None, 'ok': ok,
            'reason': (None if ok else f'日均成交額 ${adv/1e6:.1f}M < ${MIN_DOLLAR_ADV/1e6:.0f}M 下限,無法乾淨進出')}


# ═══════════════════════════════════════════════════════════════════════════════
# CROWD PSYCHOLOGY  (fear↔greed cycle — the soul of the contrarian system)
# ═══════════════════════════════════════════════════════════════════════════════

def crowd_phase(tech: dict, regime: dict | None) -> dict:
    vix = (regime or {}).get('vix')
    dd = tech.get('dist_52w_high'); d200 = tech.get('dist_ma200'); rsi14 = tech.get('rsi14')
    if tech.get('capitulation_buy'):
        return {'phase': 'CAPITULATION', 'zone': 'buy', 'color': '#1a7a4a',
                'label': '投降後轉折 — 群眾極度恐懼、最後一批人吐貨、買方剛站出來',
                'note': '別人恐懼我貪婪:賣壓力竭+轉折確認,左側唯一該出手的時刻。'}
    if tech.get('capit_setup'):
        return {'phase': 'PANIC', 'zone': 'wait', 'color': '#b52a2a',
                'label': '恐慌投降中 — 還在落刀,賣壓尚未力竭',
                'note': '投降成形但「轉折」未現 — 不接落下的刀,等買方第一次站出來再進。'}
    euphoria = ((vix is not None and vix < 14) and (d200 is not None and d200 > 0.15)
                and (rsi14 is not None and rsi14 > 70) and (dd is not None and dd > -0.03))
    if euphoria:
        return {'phase': 'EUPHORIA', 'zone': 'sell', 'color': '#d4900a',
                'label': '全民狂歡 — 群眾極度貪婪、價格遠離均線、人人看多',
                'note': '別人貪婪我恐懼:持倉者收割區,空手別追。'}
    if (dd is not None and dd <= -0.10) or (d200 is not None and d200 < 0):
        return {'phase': 'FEAR', 'zone': 'wait', 'color': '#e67e22',
                'label': '恐懼下跌 — 趨勢轉弱/回檔,群眾焦慮否認',
                'note': '恐懼未到投降極端 — 觀察,等真正的climax+轉折。'}
    if d200 is not None and d200 > 0:
        return {'phase': 'OPTIMISM', 'zone': 'hold', 'color': '#1a6fa8',
                'label': '樂觀復甦 — 站上均線、群眾轉樂觀',
                'note': '持有投降買進的核心→抱住讓它跑;空手→非進場區。'}
    return {'phase': 'NEUTRAL', 'zone': 'wait', 'color': '#6b7280',
            'label': '中性 — 無情緒極端', 'note': '無可交易的情緒極端 — 等待。'}


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT  (tiers — gated on TRUE capitulation, never casual oversold)
# ═══════════════════════════════════════════════════════════════════════════════

def verdict(tech, qual, trade, crowd, regime, index_mode: bool = False) -> dict:
    vix = (regime or {}).get('vix')
    backwardation = bool((regime or {}).get('backwardation'))   # VIX > VIX3M = acute fear
    market_fear = (vix is not None and vix >= CAPIT_FEAR_VIX)
    # Extreme fear = VIX≥30, OR VIX≥25 with term-structure backwardation (validated:
    # factor_addon_bt.py — backwardation lifts the champion PF 6.15→6.95 / win→63%).
    extreme_fear = (vix is not None and (vix >= CAPIT_FEAR_EXTREME or (vix >= CAPIT_FEAR_VIX and backwardation)))
    a_grade = tech.get('entry_grade') == 'A'   # 多頭吞噬 / 外包反轉 / RSI背離
    capit_buy = tech.get('capitulation_buy'); capit_setup = tech.get('capit_setup')
    knife = qual['knife_risk']
    quality_ok = (qual['score'] >= 50 and not knife)

    if not trade['ok']:
        return _v('UNTRADEABLE', '訊號可能成立但不可交易', '#6b21a8',
                  f'流動性不足以乾淨進出 — {trade["reason"]}。對真實資金滑價會吃掉邊際。')
    if knife:
        kr = '、'.join(qual['reasons'][:2]) or '基本面惡化'
        return _v('KNIFE', '接刀風險 — 體質可能歸零', '#b52a2a',
                  f'即使深跌投降也不接 — 體質可能歸零({kr})。投降買的前提是「跌的是情緒不是公司」。')
    # ── Index mode: never a knife (can't go to zero); buy capitulation, HOLD forever ──
    if index_mode:
        if capit_buy and extreme_fear and a_grade:
            return _v('STRONG', '指數·世紀級投降 × 極端恐慌 — 重手部署', '#0f5132',
                      f'指數深度投降 + A級買方轉折({tech.get("turn_kind")})+ 全市場極端恐慌(合成VIX {vix}≥30)。'
                      f'歷史上指數這種時刻(2008/2020/2022)一年後幾乎都是大幅正報酬 — 重手部署儲備、越跌越買、長抱。',
                      n_trades=38, stat_note='歷史38筆(15年) PF≈6.7 勝率63% — 樣本偏少,相信機制邏輯')
        if capit_buy and market_fear:
            return _v('STRONG', '指數·真投降 × 全市場恐慌 — 部署區', '#1a7a4a',
                      f'指數深度投降(距高 {_pct(tech.get("dist_52w_high"))})+ 買方轉折({tech.get("turn_kind")}),'
                      f'且全市場同步恐慌(合成VIX {vix}≥{CAPIT_FEAR_VIX:.0f})。把儲備金有紀律地投出去、長抱當核心。',
                      n_trades=50, stat_note='歷史50筆(15年) PF≈6.1')
        if capit_buy:
            return _v('SPECULATIVE', '指數·投降轉折(全市場尚未恐慌)', '#d4900a',
                      '指數深度投降 + 轉折確認,但全市場還沒一起恐慌 — 先部署一部分儲備、長抱;'
                      '等全市場也投降再加重。',
                      n_trades=98, stat_note='歷史98筆(15年) PF≈2.9 — 無全市場恐慌加持,半倉')
        # falls through to WATCH / EUPHORIA / NONE below
    elif capit_buy and quality_ok and extreme_fear and a_grade:
        return _v('STRONG', '世紀級投降 × 極端恐慌 — 最高把握', '#0f5132',
                  f'深度投降 + A級買方轉折({tech.get("turn_kind")})+ 全市場極端恐慌(VIX {vix}≥30)、體質撐得住。'
                  f'回測 PF≈6.7、勝率63%、每筆+17%。',
                  n_trades=38, stat_note='歷史38筆(15年) — 樣本偏少但機制最強,全倉')
    elif capit_buy and quality_ok and market_fear:
        return _v('STRONG', '真投降 × 全市場恐慌 — 重手區', '#1a7a4a',
                  f'深度投降(距高 {_pct(tech.get("dist_52w_high"))})+ 買方轉折({tech.get("turn_kind")}),'
                  f'全市場同步恐慌(VIX {vix}≥{CAPIT_FEAR_VIX:.0f})、體質撐得住。回測 PF≈6。',
                  n_trades=50, stat_note='歷史50筆(15年) PF≈6.1 — 全倉')
    elif capit_buy and quality_ok:
        return _v('SPECULATIVE', '個股真投降(全市場尚未恐慌)', '#d4900a',
                  '個股深度投降 + 轉折確認、體質好,但全市場還沒一起恐慌 — 半倉試單;'
                  '等全市場也投降才是重手機會。',
                  n_trades=98, stat_note='歷史98筆(15年) PF≈2.9 — 半倉,全市場恐慌前不重壓')
    if capit_setup and not tech.get('the_turn'):
        return _v('WATCH', '投降成形 — 等轉折(別接刀)', '#e67e22',
                  '已深跌 + 量能投降,但「轉折」未現(買方還沒站出來)。恐懼仍在加速 — '
                  '不接落下的刀,等收紅站回昨高/反轉K/收復10日線。')
    if crowd['phase'] == 'EUPHORIA':
        return _v('NONE', '全民狂歡 — 非進場區', '#d4900a',
                  '群眾極度貪婪、價遠離均線 — 持倉者收割區,空手別追高。')
    return _v('NONE', '無投降訊號 — 觀望', '#6b7280',
              '沒有「深度投降 × 量能力竭 × 轉折」的合流 — 不為出手而出手。左側只在真恐慌出手,耐心等。')


def _v(tier, label, color, rationale, n_trades=None, stat_note=None):
    return {'tier': tier, 'tier_label': label, 'tier_color': color,
            'rationale': rationale, 'n_trades_ref': n_trades, 'stat_note': stat_note}


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY/EXIT PLAYBOOK  (buy the turn + a deeper retest add; hold to satisfaction)
# ═══════════════════════════════════════════════════════════════════════════════

def build_plan(tech, v, qual, regime, config=None, index_mode: bool = False) -> dict:
    cfg = {**ACCOUNT, **(config or {})}
    price = tech.get('price'); atr = tech.get('atr'); tier = v['tier']
    plan = {'tier': tier, 'price': price, 'atr': atr, 'index_mode': index_mode}
    if tier not in ('STRONG', 'SPECULATIVE') or not price or not atr:
        plan.update(action='NO TRADE', tranches=[],
                    note={'WATCH': '投降成形但轉折未現 — 等買方站出來再進。',
                          'KNIFE': '體質可能歸零 — 不接。',
                          'UNTRADEABLE': '流動性不足 — 不交易。',
                          'NONE': '無投降訊號 — 觀察清單。'}.get(tier, '觀察清單。'))
        return plan

    if index_mode:
        return _build_index_plan(tech, v, regime, cfg, plan)

    # Size: risk-budget, scaled by tier (STRONG full / SPEC half) and the contrarian regime dial.
    tier_factor = 1.0 if tier == 'STRONG' else 0.5
    regime_mult = (regime or {}).get('left_multiplier', 1.0) or 1.0
    risk_mult = tier_factor * max(0.4, min(1.25, regime_mult))

    # Tranches: T1 at the TURN (today, MOC) — heavier, it's the confirmed buy; T2 a deeper add
    # if it retests the capitulation low (where the stop sits just below).
    cap_low = min(x for x in (tech.get('swing_low'), tech.get('low_52w'), price) if x)
    t1 = round(price, 2)
    t2 = round(max(cap_low * 1.01, price - 1.5*atr), 2)
    if t2 >= t1: t2 = round(price - 1.0*atr, 2)
    weights = [0.6, 0.4]
    rungs = [t1, t2]
    avg_entry = round(sum(p*w for p, w in zip(rungs, weights)), 2)

    # Stop = capitulation invalidation: below the panic low. If it loses the low, the
    # capitulation FAILED (it's a falling knife after all) → out, no debate.
    stop = round(cap_low - 0.5*atr, 2)
    risk_ps = max(avg_entry - stop, 0.01)

    # Targets — psychology of the recovery.
    mid = tech.get('boll_mid'); ma50 = tech.get('ma50'); high52 = tech.get('high_52w')
    t_relief = round(max(mid or 0, avg_entry + 1.5*atr), 2)        # relief bounce: trim 1/3
    t_satis = round(high52, 2) if high52 else round(avg_entry*1.5, 2)  # satisfaction: prior high
    rr = round((t_relief - avg_entry)/risk_ps, 2) if risk_ps > 0 else None

    risk_budget = cfg['account_size'] * cfg['risk_per_trade'] * risk_mult
    total = int(risk_budget / risk_ps)
    cap = int(cfg['account_size'] * cfg['max_position_pct'] / avg_entry) if avg_entry else 0
    total = max(0, min(total, cap))
    tranches = []
    labels = ['第一批(轉折日·收盤MOC)', '第二批(回測投降低點·掛限價)']
    alloc = 0
    for i, (px, w) in enumerate(zip(rungs, weights)):
        sh = (total - alloc) if i == len(rungs)-1 else int(total*w)
        alloc += sh
        tranches.append({'label': labels[i], 'price': px, 'shares': sh,
                         'value': round(sh*px, 0), 'pct': round(sh/total*100) if total else 0})
    pos_val = round(sum(t['value'] for t in tranches), 0)

    plan.update(
        action='CAPITULATION BUY(投降分批進場)', tranches=tranches, avg_entry=avg_entry,
        stop=stop, risk_per_share=round(risk_ps, 2), total_shares=total,
        position_value=pos_val, position_pct=round(pos_val/cfg['account_size']*100, 1),
        dollar_risk=round(total*risk_ps, 0),
        target_relief=t_relief, target_satisfaction=t_satis, rr=rr,
        time_stop_days=TIME_STOP_DAYS, risk_mult=round(risk_mult, 2),
        discipline=_discipline(stop, avg_entry, t_relief, t_satis, tech, regime))
    return plan


def _discipline(stop, avg, t_relief, t_satis, tech, regime):
    ma200 = tech.get('ma200')
    rules = [
        f'【進場】只在「轉折日」用收盤(MOC)進第一批;若回測投降低點再掛限價補第二批。不追、不接落下的刀。'
        f'(盤前快照看到訊號 → 等收盤確認 → 隔日開盤前掛限價單,不要追盤中)',
        f'【🔒 鎖定三個數字 — 今天寫下來,之後不改】'
        f'  停損 ${stop} / relief減1/3 ${t_relief} / 滿足出場 ${t_satis}'
        f'  (ATR 每天變、盤中波動,但這三個錨點從進場起就固定,永遠不重新計算)',
        f'【最高原則】硬停損 ${stop}(投降低點之下)= 投降失敗、確認是接刀 → 無條件清倉,不凹單。',
        f'【別賣relief】第一個反彈是群眾的「鬆一口氣」,不是頂。只先減 1/3(到 ${t_relief}),'
        f'並把停損抬到成本價 ${avg};核心抱住,別被早期反彈騙下車。',
        f'【抱到滿足/狂歡】核心出場(擇一):① 收復痛苦起點 ${t_satis}(前52週高)先了結;'
        f'② 全民狂歡轉彎(VIX<14+遠離年線「之後」跌破10週均);③ 吊燈移動停損(自最高收盤回落 {ATR_TRAIL}×ATR);'
        f'④ 跌破年線 ${round(ma200,2) if ma200 else "—"}。— 賣在貪婪,但等它真轉,別猜頭。',
        f'【時間】{TIME_STOP_DAYS} 交易日仍是死錢且未站上年線 → 出場,別佔資金。',
        f'【部位】嚴守上限,單一標的歸零也傷不了總資本 — 這是你敢在最恐慌時出手的底氣。',
    ]
    if regime and regime.get('fast_crash'):
        rules.insert(2, '⚠ 偵測到「快速崩跌」— 雖在年線上但屬瀑布式下殺,左側加碼自動關閉,寧可錯過不接刀。')
    return rules


def _build_index_plan(tech, v, regime, cfg, plan) -> dict:
    """Index capitulation = deploy reserve cash and HOLD. No stop (can't go to zero), no
    satisfaction target (selling forfeits drift). Size by a target % of the account per event
    rather than by stop-distance, because there is no stop. Keep tranches so you save bullets
    for a deeper leg down (越跌越買)."""
    price = tech.get('price'); atr = tech.get('atr'); tier = v['tier']
    tier_factor = 1.0 if tier == 'STRONG' else 0.5
    regime_mult = (regime or {}).get('left_multiplier', 1.0) or 1.0
    deploy_mult = tier_factor * max(0.4, min(1.25, regime_mult))
    target_value = cfg['account_size'] * cfg['max_position_pct'] * deploy_mult

    cap_low = min(x for x in (tech.get('swing_low'), tech.get('low_52w'), price) if x)
    t1 = round(price, 2)
    t2 = round(max(cap_low * 1.01, price - 1.5*atr), 2)
    if t2 >= t1: t2 = round(price - 1.0*atr, 2)
    weights = [0.6, 0.4]; rungs = [t1, t2]
    avg_entry = round(sum(p*w for p, w in zip(rungs, weights)), 2)
    total = int(target_value / avg_entry) if avg_entry else 0

    tranches = []
    labels = ['第一批(投降轉折日·收盤MOC)', '第二批(更深回測·掛限價加碼)']
    alloc = 0
    for i, (px, w) in enumerate(zip(rungs, weights)):
        sh = (total - alloc) if i == len(rungs)-1 else int(total*w)
        alloc += sh
        tranches.append({'label': labels[i], 'price': px, 'shares': sh,
                         'value': round(sh*px, 0), 'pct': round(sh/total*100) if total else 0})
    pos_val = round(sum(t['value'] for t in tranches), 0)

    plan.update(
        action='指數投降部署(買進長抱·不停損不停利)', tranches=tranches, avg_entry=avg_entry,
        stop=None, risk_per_share=None, total_shares=total,
        position_value=pos_val, position_pct=round(pos_val/cfg['account_size']*100, 1),
        dollar_risk=None, target_relief=None, target_satisfaction=None, rr=None,
        time_stop_days=None, risk_mult=round(deploy_mult, 2),
        discipline=_discipline_index(avg_entry, tech, regime))
    return plan


def _discipline_index(avg, tech, regime):
    rules = [
        '【進場】只在「投降轉折日」用收盤(MOC)進第一批;若指數更深回測再掛限價加碼。不接落下的刀。',
        '【不停損】指數不會歸零 — 沒有硬停損。若續跌,那是「越跌越買」的加碼點,不是認賠點。'
        '前提:只動用儲備金、不上槓桿、不會被迫賣出。',
        '【不停利】指數長期為正(dip_hold_test 證實:擇時買進但永不賣 > 反覆停利)。'
        '這批貨併入長期核心,當成提早佈局的 DCA,不設賣點。',
        '【唯一的賣】只有再平衡需求、或極端全民狂歡(VIX<14 + 遠離年線且月線轉弱)才小幅減碼;預設抱著。',
        '【留彈藥】單次投降部署只用帳戶的少數比例,分批進、保留更深下殺的子彈 — 才能在 -30% 還笑得出來。',
    ]
    if regime and regime.get('fast_crash'):
        rules.insert(1, '⚠ 偵測到「快速崩跌」(瀑布式下殺)— 放慢部署、拉大分批間距,別一次打完子彈。')
    return rules


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(data: dict, quality: dict = None, regime: dict = None, config: dict = None,
            index_mode: bool = False) -> dict:
    info = data.get('info', {})
    tech = technical_read(data.get('hist_1y'), data.get('hist_5y'))
    if not tech:
        return {'ticker': data.get('symbol', '?'), 'status': 'ERROR', 'error': 'no price history'}
    qual = quality_gate(info, quality or {}, index_mode=index_mode,
                        ticker=data.get('symbol', ''))
    trade = tradeable(info, tech, data.get('hist_1y'))
    crowd = crowd_phase(tech, regime)
    v = verdict(tech, qual, trade, crowd, regime, index_mode=index_mode)
    plan = build_plan(tech, v, qual, regime, config, index_mode=index_mode)
    return {
        'ticker': data.get('symbol', '?'),
        'name': info.get('shortName') or info.get('longName') or data.get('symbol'),
        'sector': (info.get('sector') or ('指數/ETF' if index_mode else '-')), 'status': 'OK',
        'index_mode': index_mode,
        'tech': tech, 'quality': qual, 'tradeable': trade, 'crowd': crowd,
        'score': {**v, 'crowd': crowd}, 'plan': plan,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }


def format_console(r: dict) -> str:
    if r.get('status') != 'OK':
        return f"  {r.get('ticker')}: ERROR {r.get('error','')[:50]}"
    s = r['score']; t = r['tech']; pl = r['plan']
    head = f"\n  {r['ticker']:6} {s['tier_label']:26} [{s['crowd']['phase']}]"
    turn_disp = (f"{t.get('turn_kind')}({t.get('entry_grade')}級)" if t.get('the_turn') else '未現')
    line = (f"    距高 {_pct(t.get('dist_52w_high'))} RSI2(5d低) {t.get('rsi2_min_5d')} "
            f"量能投降 {t.get('vol_climax_x')}x 轉折 {turn_disp} "
            f"體質 {r['quality']['score']:.0f}")
    if pl.get('tranches'):
        rungs = ' / '.join(f"${x['price']}×{x['shares']}" for x in pl['tranches'])
        if pl.get('index_mode'):
            ex = (f"    {rungs} | 均價 ${pl['avg_entry']} 長抱·不停損不停利 "
                  f"部位 {pl['position_pct']}%（越跌越買)")
        else:
            ex = (f"    {rungs} | 均價 ${pl['avg_entry']} 停損 ${pl['stop']} "
                  f"滿足 ${pl['target_satisfaction']} R:R {pl['rr']} 部位 {pl['position_pct']}%")
    else:
        ex = f"    {pl.get('note','')}"
    stat = s.get('stat_note')
    stat_line = f"    [{stat}]" if stat else ''
    lines = [head, line, ex]
    if stat_line: lines.append(stat_line)
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _check(label, ok, val):
    c = '#1a7a4a' if ok else '#b52a2a'; mark = '✓' if ok else '✗'
    return (f'<span style="display:inline-block;background:{c}14;border:1px solid {c};color:{c};'
            f'border-radius:10px;padding:2px 9px;margin:2px;font-size:7.8pt">{mark} {label} <b>{val}</b></span>')


def _card(r):
    s = r['score']; t = r['tech']; pl = r['plan']; q = r['quality']; cw = s['crowd']
    col = s['tier_color']
    idx = bool(r.get('index_mode'))
    checklist = (
        _check('距高≥15%', (t.get('dist_52w_high') or 0) <= CAPIT_DD, _pct(t.get('dist_52w_high'))) +
        _check('RSI2(5d低)<5', (t.get('rsi2_min_5d') or 99) < CAPIT_RSI2, t.get('rsi2_min_5d')) +
        _check('量能投降≥2x', bool(t.get('climax_5d')) or (t.get('vol_climax_x') or 0) >= CAPIT_CLIMAX, f"{t.get('vol_climax_x')}x") +
        _check('買方轉折', bool(t.get('the_turn')),
               (f"{t.get('turn_kind')} · {t.get('entry_grade')}級" if t.get('the_turn') else '未現')) +
        _check('指數·不會歸零' if idx else '體質撐得住', not q['knife_risk'],
               '免關卡' if idx else f"{q['score']:.0f}"))
    ladder = ''
    if pl.get('tranches'):
        for x in pl['tranches']:
            ladder += (f'<div style="display:flex;gap:10px;padding:4px 0;border-bottom:1px solid #f0f0f0">'
                       f'<div style="width:190px;font-size:8pt;color:#374151">{x["label"]}</div>'
                       f'<div style="font-weight:700;color:#0d3b6e;width:64px">${x["price"]}</div>'
                       f'<div style="font-size:8pt;color:#6b7280">{x["shares"]} 股 ({x["pct"]}%)</div></div>')
        if idx:
            ladder += (f'<div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;font-size:8.5pt">'
                       f'<div>均價 <b>${pl["avg_entry"]}</b></div>'
                       f'<div><b style="color:#1a7a4a">不停損</b>(指數不會歸零·越跌越買)</div>'
                       f'<div><b style="color:#1a7a4a">不停利</b>(併入長期核心)</div>'
                       f'<div>部位 <b>{pl["position_pct"]}%</b></div></div>')
        else:
            ladder += (f'<div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;font-size:8.5pt">'
                       f'<div>均價 <b>${pl["avg_entry"]}</b></div>'
                       f'<div>停損 <b style="color:#b52a2a">${pl["stop"]}</b></div>'
                       f'<div>relief減1/3 <b style="color:#1a7a4a">${pl["target_relief"]}</b></div>'
                       f'<div>滿足/前高 <b style="color:#1a7a4a">${pl["target_satisfaction"]}</b></div>'
                       f'<div>R:R <b>{pl["rr"]}</b></div>'
                       f'<div>部位 <b>{pl["position_pct"]}%</b>(風險 ${pl["dollar_risk"]:,.0f})</div></div>')
    else:
        ladder = f'<div style="color:#9ca3af;font-size:8.5pt">{pl.get("note","")}</div>'
    disc = ''.join(f'<li style="margin-bottom:5px;font-size:8.3pt;line-height:1.5">{d}</li>'
                   for d in pl.get('discipline', []))
    disc_block = (f'<div style="margin-top:12px;border-top:1px solid #eee;padding-top:10px">'
                  f'<div style="font-size:8.5pt;font-weight:700;color:#b52a2a;margin-bottom:6px">反人性紀律(寫在進場前)</div>'
                  f'<ol style="margin:0 0 0 18px;color:#374151">{disc}</ol></div>') if disc else ''
    return f'''
    <div style="background:white;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.1);margin-bottom:14px;overflow:hidden">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:11px 16px;background:#fafbfd;border-left:6px solid {col}">
        <div><span style="font-size:13pt;font-weight:700;color:#0d3b6e">{r['ticker']}</span>
          <span style="font-size:8.5pt;color:#6b7280;margin-left:8px">{r['name'][:28]} · {r['sector']}</span></div>
        <span style="background:{col};color:#fff;padding:4px 12px;border-radius:4px;font-size:9pt;font-weight:700">{s['tier_label']}</span>
      </div>
      <div style="padding:13px 16px">
        <div style="display:flex;gap:10px;align-items:center;background:{cw['color']}12;border-left:4px solid {cw['color']};border-radius:5px;padding:8px 12px;margin-bottom:10px">
          <span style="background:{cw['color']};color:#fff;font-size:7.5pt;font-weight:700;padding:2px 9px;border-radius:10px;white-space:nowrap">群眾情緒 · {cw['phase']}</span>
          <span style="font-size:8pt;color:#374151">{cw['label']} — {cw['note']}</span></div>
        <div style="font-size:8.8pt;color:#374151;background:#f8fafd;padding:9px 12px;border-radius:5px;margin-bottom:10px;line-height:1.55">{s['rationale']}</div>
        <div style="margin-bottom:10px">{checklist}</div>
        <div style="font-size:8.5pt;font-weight:700;color:#0d3b6e;margin-bottom:6px">{'指數投降部署(買進長抱·不停損不停利)' if idx else '投降分批劇本(抱到滿足,不賣relief)'}</div>
        {ladder}{disc_block}
      </div>
    </div>'''


def render_report(results, regime=None, portfolio=None, output_path=None):
    output_path = output_path or os.path.join(REPORTS_DIR, 'capitulation.html')
    ok = [r for r in results if r.get('status') == 'OK']
    order = {'STRONG': 0, 'SPECULATIVE': 1, 'WATCH': 2, 'UNTRADEABLE': 3, 'KNIFE': 4, 'NONE': 5}
    ok.sort(key=lambda r: order.get(r['score']['tier'], 9))
    n = {k: sum(1 for r in ok if r['score']['tier'] == k) for k in ('STRONG', 'SPECULATIVE', 'WATCH')}
    reg = regime or {}
    vix = reg.get('vix'); lm = reg.get('left_multiplier')

    # ── 分市場:台股(.TW)vs 美股,各自成一區、各自依判決排序 ──
    def _is_tw(r): return str(r.get('ticker', '')).upper().endswith('.TW')
    tw = [r for r in ok if _is_tw(r)]
    us = [r for r in ok if not _is_tw(r)]

    def _section(title, subset):
        if not subset:
            return (f'<div class="mkt">{title} <span>· 名單中無標的</span></div>'
                    f'<div style="color:#9ca3af;font-size:8.5pt;padding:6px 2px 14px">—</div>')
        nS = sum(1 for r in subset if r['score']['tier'] == 'STRONG')
        nSp = sum(1 for r in subset if r['score']['tier'] == 'SPECULATIVE')
        nW = sum(1 for r in subset if r['score']['tier'] == 'WATCH')
        act = f' · <b style="color:#7CFC98">真投降 {nS}</b>' if nS else ''
        spc = f' · 試單 {nSp}' if nSp else ''
        wat = f' · 等轉折 {nW}' if nW else ''
        hdr = (f'<div class="mkt">{title} '
               f'<span>· {len(subset)} 檔{act}{spc}{wat}</span></div>')
        return hdr + ''.join(_card(r) for r in subset)

    cards = _section('🇺🇸 美股', us) + _section('🇹🇼 台股', tw)
    updated = datetime.now(TAIPEI).strftime('%Y-%m-%d %H:%M')
    n_scanned = len(results); n_ok = len(ok); n_us = len(us); n_tw = len(tw)
    fast = ' · ⚠快速崩跌:左側加碼關閉' if reg.get('fast_crash') else ''
    lm_pct = (lm if lm is not None else 1)
    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/><title>投降引擎</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI','Microsoft JhengHei',Arial,sans-serif;font-size:9.5pt;color:#1a1a2e;background:#f0f2f7}}
.bar{{background:#5e1818;color:#fff;padding:13px 26px}}.bar h1{{font-size:13pt}}.bar .s{{font-size:8pt;opacity:.85;margin-top:3px}}
.c{{max-width:1100px;margin:0 auto;padding:20px}}
.note{{background:#fffaf5;border-left:4px solid #d4900a;border-radius:5px;padding:11px 14px;margin-bottom:16px;font-size:8.4pt;color:#374151;line-height:1.6}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}}
.k{{background:#fff;border-radius:6px;padding:10px 14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.k .l{{font-size:7pt;color:#6b7280;text-transform:uppercase}}.k .v{{font-size:15pt;font-weight:700;margin-top:2px}}
.mkt{{background:#0d3b6e;color:#fff;border-radius:6px;padding:9px 16px;margin:22px 0 12px;font-size:11pt;font-weight:700;box-shadow:0 1px 4px rgba(0,0,0,.12)}}
.mkt span{{font-size:8.2pt;font-weight:400;opacity:.85}}
.foot{{text-align:center;font-size:7.3pt;color:#9ca3af;padding:14px}}</style></head><body>
<div class="bar"><h1>投降引擎 — 反人性左側(乾淨重寫版)</h1>
<div class="s">🕐 更新時間 <b>{updated}(台北)</b> · 本次掃描 <b>{n_scanned}</b> 檔全部(美股 {n_us} · 台股 {n_tw})· 成功 {n_ok}</div></div>
<div class="c">
<div class="note"><b>誠實定位:</b>這不是打敗大盤的 alpha 引擎(回測證實左側擇時在多頭輸 DCA)。
它是<b>稀有事件的反人性紀律工具</b> — 在真恐慌時把現金有紀律地投出去、抱到群眾轉貪婪。
回測:真投降+全市場恐慌 PF 5.39、每筆 +14%(但 15 年僅 ~36 次,屬世紀級恐慌)。
<b>正確用法:核心 DCA 大盤 + 本引擎當恐慌 overlay。</b>詳見 STRATEGY_TRUTH.md。</div>
<div style="background:#fff;border-radius:6px;padding:9px 14px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.08);font-size:8.4pt;color:#374151">
  市場體制:<b>{reg.get('regime','—')}</b> · VIX <b>{vix}</b> · 左側部位調節 ×{lm_pct:.0%}{fast}</div>
<div class="kpi">
  <div class="k"><div class="l">分析標的</div><div class="v" style="color:#0d3b6e">{len(ok)}</div></div>
  <div class="k"><div class="l">真投降·重手</div><div class="v" style="color:#1a7a4a">{n['STRONG']}</div></div>
  <div class="k"><div class="l">個股投降·試單</div><div class="v" style="color:#d4900a">{n['SPECULATIVE']}</div></div>
  <div class="k"><div class="l">投降成形·等轉折</div><div class="v" style="color:#e67e22">{n['WATCH']}</div></div>
</div>
{cards}
<div class="foot">投降引擎 · 資料 Yahoo Finance · 僅供研究,非投資建議 · {updated}</div>
</div></body></html>"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[ok] 投降引擎報告 -> {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

_REGIME = None
def _regime():
    global _REGIME
    if _REGIME is None:
        try:
            from regime import get_regime
            _REGIME = get_regime()
        except Exception:
            _REGIME = {'regime': 'CAUTION', 'vix': None, 'left_multiplier': 0.8}
    return _REGIME


def _fetch_index_data(ticker):
    """Light price-only fetch for indices/ETFs — skips the fundamental statement calls that
    just 404 for ETFs (and the expensive options scan). Index mode needs only OHLCV."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    try: info = t.info or {}
    except Exception: info = {}
    return {'symbol': ticker.upper(), 'info': info,
            'hist_1y': t.history(period='1y'), 'hist_5y': t.history(period='5y')}


_TW_REGIME = None
def _tw_regime():
    global _TW_REGIME
    if _TW_REGIME is None:
        try:
            from regime import get_taiwan_regime
            _TW_REGIME = get_taiwan_regime()
        except Exception:
            _TW_REGIME = {'regime': 'TW-UNKNOWN', 'vix': 20.0, 'left_multiplier': 0.70,
                          'backwardation': False, 'fast_crash': False, 'is_taiwan_regime': True,
                          'spy_price': None, 'spy_200dma': None, 'spy_above_200': None}
    return _TW_REGIME


def _run_ticker(ticker, index_mode: bool = False):
    tw_stock = is_tw_stock(ticker)
    tw_idx   = index_mode and (ticker or '').upper().endswith('.TW')

    if index_mode:
        try:
            print(f"  [+] Fetching index data for {ticker.upper()}...")
            regime = _tw_regime() if tw_idx else _regime()
            return analyze(_fetch_index_data(ticker), quality={}, regime=regime, index_mode=True)
        except Exception as e:
            return {'ticker': ticker, 'status': 'ERROR', 'error': str(e)}

    from data_fetcher import fetch_company_data
    try:
        from valuation_engine import quality_score
    except Exception:
        quality_score = None
    try:
        data = fetch_company_data(ticker)
        quality = {}
        if quality_score:
            try:
                quality = quality_score(data['info'], data['cashflow'], data['balance_sheet'], data['income_stmt'])
            except Exception:
                quality = {}
        # 台股個股用 TAIEX 恐慌指標;美股用 VIX
        regime = _tw_regime() if tw_stock else _regime()
        return analyze(data, quality=quality, regime=regime)
    except Exception as e:
        return {'ticker': ticker, 'status': 'ERROR', 'error': str(e)}


def main():
    import sys, argparse
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    ap = argparse.ArgumentParser(description='投降引擎 — 反人性左側')
    ap.add_argument('tickers', nargs='*')
    ap.add_argument('--watchlist', '-w')
    ap.add_argument('--index', action='store_true',
                    help='強制所有標的用「指數模式」(免基本面關卡·不停損·不停利·長抱)')
    ap.add_argument('--no-report', action='store_true')
    args = ap.parse_args()
    tickers = list(args.tickers)
    if args.watchlist and os.path.exists(args.watchlist):
        with open(args.watchlist) as f:
            tickers += [l.strip().upper() for l in f if l.strip() and not l.startswith('#')]
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    if not tickers:
        print('用法: python capitulation_engine.py NVDA AMD TSM  [--watchlist f.txt] [--index]\n'
              '      指數(SPY/QQQ/SOXX…)會自動套用指數模式;--index 可強制全部用指數模式。'); sys.exit(1)

    r = _regime()
    n_idx = sum(1 for t in tickers if args.index or is_index(t))
    n_tw  = sum(1 for t in tickers if is_tw_stock(t) or ((args.index or is_index(t)) and t.upper().endswith('.TW')))
    tw_note = ''
    if n_tw:
        tr = _tw_regime()
        tw_note = f' · 台股恐慌:{tr.get("regime")} 合成VIX {tr.get("vix")}'
    print(f"\n投降引擎掃描:{len(tickers)} 檔(指數 {n_idx} 檔)— 美股:{r.get('regime')} VIX {r.get('vix')}{tw_note}")
    results = []
    for i, t in enumerate(tickers, 1):
        idx = args.index or is_index(t)
        print(f"  [{i}/{len(tickers)}] {t}{' [指數]' if idx else ''}...", end='')
        res = _run_ticker(t, index_mode=idx); results.append(res)
        print(format_console(res) if res.get('status') == 'OK' else f"  ERROR {res.get('error','')[:50]}")

    portfolio = None
    try:
        from portfolio_construct import construct_portfolio, print_portfolio
        portfolio = construct_portfolio(results, regime=r)
        print_portfolio(portfolio)
    except Exception as e:
        print(f"  [portfolio] 略過:{e}")

    if not args.no_report:
        path = render_report(results, regime=r, portfolio=portfolio)
        print(f"開啟:file:///{path.replace(os.sep, '/')}")
    try:
        slim = [dict(x) for x in results]
        for fn in ('capitulation.json', 'bottom_fishing.json'):  # bottom_fishing.json = dashboard compat
            with open(os.path.join(REPORTS_DIR, fn), 'w', encoding='utf-8') as f:
                json.dump(slim, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


if __name__ == '__main__':
    main()
