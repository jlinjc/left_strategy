"""
每日抄底巡邏  (Daily Capitulation Patrol)
═══════════════════════════════════════════════════════════════════════════════
每天跑「一次」→ 立刻知道三件事:
  1. 今天市場恐慌了嗎?(體制 + VIX)
  2. 有沒有該抄底的東西?抄什麼?
  3. 怎麼抄?(分批、停損/長抱、部位)

平靜時它明確告訴你「沒事 — 繼續定投,手放開」;真投降時把標的 + 劇本推到你面前,
並自動打開 HTML 報告。這就是把「紀律」變成「每天一個動作」的地方。

這一個指令做完所有事:① 即時/盤前快照(看清戰場)→ ② 收盤資料的引擎掃描(等轉折的進場判斷)
→ ③ 自動打開儀表板(capitulation.html)。你隨時只要跑這一個。

用法:
  python daily.py                  # ★ 全部:快照 + 掃描 + 開儀表板(平常就跑這個)
  python daily.py SPY QQQ NVDA     # 臨時只看某幾檔
  python daily.py -p               # 只看盤前快照(快,不跑完整引擎、不開報告)
  python daily.py --no-open        # 跑完不自動開儀表板
  python daily.py --quiet          # 只印結論(排程用)

排程(每個交易日早上自動跑)見檔尾說明。
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import yfinance as yf
import capitulation_engine as ce

REPORTS_DIR = ce.REPORTS_DIR
WATCHLIST   = os.path.join(os.path.dirname(__file__), 'watchlist.txt')
DIGEST_FILE = os.path.join(REPORTS_DIR, 'daily_digest.json')

# 哪些判決算「今天要動作」 vs 「盯著」 vs 「觀望」
ACT_TIERS   = ('STRONG', 'SPECULATIVE')
WATCH_TIERS = ('WATCH',)


def load_watchlist(path=WATCHLIST):
    tickers = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.split('#')[0].strip()   # 去掉行內註解
                if line:
                    tickers.append(line.upper())
    return list(dict.fromkeys(tickers))


def _tier_of(res):
    return (res.get('score') or {}).get('tier')


# ═══════════════════════════════════════════════════════════════════════════════
# 盤前快照 (PRE-MARKET) — 心理準備,不是進場訊號
# ───────────────────────────────────────────────────────────────────────────────
# 引擎的買進要等「收盤的爆量+轉折」確認(盤前不可能有轉折)。這裡只回答:
# 現在跌到哪、離投降跌幅區多遠、要抄的話該盯哪個價位 — 讓你開盤前心裡有底、手不抖。
# ═══════════════════════════════════════════════════════════════════════════════

def _quote_levels(ticker):
    """回傳即時/盤前報價 + 抄底要盯的關鍵價位。一次 Ticker 物件、最少呼叫。"""
    tk = yf.Ticker(ticker)
    try: info = tk.info or {}
    except Exception: info = {}
    prev = info.get('previousClose') or info.get('regularMarketPreviousClose')
    live = info.get('preMarketPrice') or info.get('regularMarketPrice')
    src = '盤前' if info.get('preMarketPrice') else '即時'
    if live is None:
        try: live = float(tk.fast_info.last_price)
        except Exception: live = prev
    hi52 = prior_high = swing_low = None
    try:
        h = tk.history(period='1y')
        if len(h):
            hi52 = float(h['Close'].tail(252).max())
            prior_high = float(h['High'].iloc[-1])          # 昨日高 = 轉折觸發參考
            swing_low = float(h['Low'].tail(60).min())       # 近期低 = 投降低點參考
    except Exception:
        pass
    return {'live': live, 'prev': prev, 'src': src, 'hi52': hi52,
            'prior_high': prior_high, 'swing_low': swing_low}


def premarket_scan(tickers):
    # 市場層級:VIX + SPY 盤前
    vinfo = {}; sinfo = {}
    try: vinfo = yf.Ticker('^VIX').info or {}
    except Exception: pass
    try: sinfo = yf.Ticker('SPY').info or {}
    except Exception: pass
    vix = vinfo.get('regularMarketPrice'); vix_prev = vinfo.get('previousClose')
    spy_live = sinfo.get('preMarketPrice') or sinfo.get('regularMarketPrice')
    spy_prev = sinfo.get('previousClose')
    spy_pct = (spy_live/spy_prev - 1) if (spy_live and spy_prev) else None

    print(f"\n盤前快照 · {datetime.now():%Y-%m-%d %H:%M}")
    print("═" * 64)
    mbits = []
    if vix is not None:
        chg = f"(昨 {vix_prev})" if vix_prev else ""
        tag = '🔴 全市場恐慌' if vix >= 25 else '🟠 恐懼升溫' if vix >= 20 else '🟢 仍平靜'
        mbits.append(f"VIX {vix} {chg} {tag}")
    if spy_pct is not None:
        mbits.append(f"SPY 盤前 {spy_pct*100:+.1f}%")
    print("  市場:" + " · ".join(mbits))
    print("  ⚠ 這是【心理準備】,不是進場訊號。真投降要等收盤的『爆量+轉折』確認 —")
    print("     盤前不可能有轉折,現在不出手。")
    print("─" * 64)

    in_zone = []
    for t in tickers:
        q = _quote_levels(t)
        live, prev, hi52 = q['live'], q['prev'], q['hi52']
        if not live or not prev:
            print(f"  {t:6} 報價取得失敗"); continue
        pm_pct = (live/prev - 1) * 100
        dd = (live/hi52 - 1) * 100 if hi52 else None
        deep = (dd is not None and dd <= ce.CAPIT_DD*100)   # 進入投降跌幅區
        zone_tag = '【已進投降跌幅區】' if deep else ''
        idx = ce.is_index(t)
        print(f"  {t:6}{'[指數]' if idx else '      '} 盤前 {pm_pct:+5.1f}%  ${live:.2f} (昨收 ${prev:.2f})  "
              f"距52週高 {dd:+.1f}%{zone_tag}" if dd is not None
              else f"  {t:6} 盤前 {pm_pct:+5.1f}%  ${live:.2f}")
        if deep:
            trig = q['prior_high']; lo = q['swing_low']
            line = "         要抄的話盯:"
            if trig: line += f"轉折觸發≈收盤站回 ${trig:.2f}(昨高)以上 "
            if lo: line += f"· 投降低點參考 ${lo:.2f}"
            print(line)
            in_zone.append(t)

    print("─" * 64)
    if in_zone:
        print(f"  📌 已在投降跌幅區({len(in_zone)} 檔):{' '.join(in_zone)}")
        print("     → 別現在追。等收盤跑 python daily.py,看是否出現『爆量+轉折』。")
        print("       出現 = 紅燈分批抄;沒出現 = 還在落刀,繼續等。")
    else:
        print("  目前沒有標的進入投降跌幅區(距高未達 -15%)— 這只是回檔,不是投降。")
    print("\n  心法:盤前狂跌整天反轉的機率很高。你現在唯一的工作是【看戲+心裡有底】,")
    print("       把子彈準備好,等引擎在收盤喊轉折。跌得越急,越要分批、越要留彈藥。")
    print("═" * 64)


def run(tickers, open_report=True, quiet=False):
    # ① 即時/盤前快照 — 先讓你看清現在戰場(盤前/盤中都適用)
    if not quiet:
        try:
            premarket_scan(tickers)
        except Exception as e:
            print(f"(即時快照略過:{e})")

    # ② 收盤資料的引擎掃描 — 真正的進場判斷(等轉折)
    r = ce._regime()
    vix = r.get('vix'); regime = r.get('regime')
    bw = r.get('backwardation'); fc = r.get('fast_crash')

    n_idx = sum(1 for t in tickers if ce.is_index(t))
    if not quiet:
        print(f"\n投降引擎掃描(收盤資料)· {datetime.now():%Y-%m-%d %H:%M}")
        print(f"掃描 {len(tickers)} 檔(指數 {n_idx} · 個股 {len(tickers)-n_idx})")

    results = []
    for i, t in enumerate(tickers, 1):
        idx = ce.is_index(t)
        res = ce._run_ticker(t, index_mode=idx)
        results.append(res)
        if not quiet and res.get('status') == 'OK':
            print(ce.format_console(res))
        elif not quiet:
            print(f"  {t}: ERROR {res.get('error','')[:40]}")

    ok = [r for r in results if r.get('status') == 'OK']
    act   = [r for r in ok if _tier_of(r) in ACT_TIERS]
    watch = [r for r in ok if _tier_of(r) in WATCH_TIERS]

    # HTML 報告(永遠重新產生,讓你隨時點開都是最新)
    portfolio = None
    try:
        from portfolio_construct import construct_portfolio
        portfolio = construct_portfolio(results, regime=r)
    except Exception:
        pass
    html_path = ce.render_report(results, regime=r, portfolio=portfolio)

    # ── 今日結論 ──
    print("\n" + "═" * 64)
    fear_tag = ('🔴 全市場恐慌' if (vix and vix >= 25) else
                '🟠 偏恐懼' if (vix and vix >= 20) else '🟢 平靜')
    bits = [f"體制 {regime}", f"VIX {vix} {fear_tag}"]
    if bw: bits.append("期限倒掛(急性恐慌)")
    if fc: bits.append("⚠快速崩跌")
    print("  市場:" + " · ".join(bits))

    if act:
        print(f"\n  🔴 今天有 {len(act)} 個抄底訊號 — 該動作:")
        for r0 in act:
            s = r0['score']; pl = r0['plan']; t = r0['tech']
            kind = '指數·長抱' if r0.get('index_mode') else '個股'
            line = (f"    ● {r0['ticker']:5} [{kind}] {s['tier_label']}")
            print(line)
            if pl.get('tranches'):
                rungs = ' / '.join(f"${x['price']}×{x['shares']}" for x in pl['tranches'])
                if r0.get('index_mode'):
                    print(f"        進場 {rungs} | 均價 ${pl['avg_entry']} · 不停損不停利·越跌越買 · 部位 {pl['position_pct']}%")
                else:
                    print(f"        進場 {rungs} | 均價 ${pl['avg_entry']} 停損 ${pl['stop']} 滿足 ${pl['target_satisfaction']} · 部位 {pl['position_pct']}%")
        print(f"\n  → 詳細劇本看報告。")
    elif watch:
        print(f"\n  🟠 沒有可進場訊號,但 {len(watch)} 檔『投降成形·等轉折』(盯著,別接刀):")
        for r0 in watch:
            t = r0['tech']
            print(f"    ○ {r0['ticker']:5} 距高 {ce._pct(t.get('dist_52w_high'))} · 等買方轉折出現")
    else:
        print("\n  🟢 今天沒有抄底訊號 — 沒事,繼續定投你的核心,手放開。")
        print("     (這才是常態:左側只在真恐慌出手,90% 的日子就是等。)")

    print(f"\n  報告:file:///{html_path.replace(os.sep, '/')}")
    print("═" * 64)

    # 存一份精簡 digest(可給排程/通知用)
    digest = {
        'as_of': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'regime': regime, 'vix': vix, 'backwardation': bw, 'fast_crash': fc,
        'n_act': len(act), 'n_watch': len(watch),
        'act': [{'ticker': r0['ticker'], 'tier': _tier_of(r0),
                 'index_mode': r0.get('index_mode'),
                 'label': r0['score']['tier_label']} for r0 in act],
        'watch': [r0['ticker'] for r0 in watch],
        'html': html_path,
    }
    try:
        with open(DIGEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(digest, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 跑完一律打開儀表板(報告)
    if open_report:
        try:
            os.startfile(html_path)   # Windows
        except Exception as e:
            print(f"(無法自動開啟報告:{e} — 手動點上面連結)")

    return digest


def main():
    import argparse
    ap = argparse.ArgumentParser(description='每日抄底巡邏 — 一個指令:盤前快照+收盤掃描+開儀表板')
    ap.add_argument('tickers', nargs='*', help='臨時指定標的(留空=用 watchlist.txt)')
    ap.add_argument('--premarket', '-p', action='store_true',
                    help='只看盤前快照(快·不跑完整引擎·不開報告)')
    ap.add_argument('--no-open', action='store_true', help='跑完不要自動開儀表板')
    ap.add_argument('--quiet', action='store_true', help='只印結論(排程用)')
    args = ap.parse_args()
    tickers = [t.upper() for t in args.tickers] or load_watchlist()
    if not tickers:
        print('watchlist.txt 是空的,且未指定標的。請編輯 watchlist.txt 或:python daily.py SPY QQQ NVDA')
        sys.exit(1)
    if args.premarket:
        premarket_scan(tickers)          # 純盤前快照模式
    else:
        run(tickers, open_report=not args.no_open, quiet=args.quiet)


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# 每天自動執行(Windows 工作排程器)
# ───────────────────────────────────────────────────────────────────────────────
# 開 PowerShell 貼這段(每個交易日 09:35 自動跑,有訊號會自動跳報告):
#
#   $py  = (Get-Command python).Source
#   $dir = "C:\Users\Jason\left_strategy"
#   $act = New-ScheduledTaskAction -Execute $py -Argument "daily.py" -WorkingDirectory $dir
#   $trg = New-ScheduledTaskTrigger -Daily -At 9:35AM
#   Register-ScheduledTask -TaskName "抄底巡邏" -Action $act -Trigger $trg -Description "每日投降引擎掃描"
#
# 想手動跑就 double-check:  python daily.py
# 想移除排程:              Unregister-ScheduledTask -TaskName "抄底巡邏" -Confirm:$false
# ═══════════════════════════════════════════════════════════════════════════════
