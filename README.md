# 左側交易 / 抄底引擎 (Left-Side / Bottom-Fishing Strategy)

一個經過回測驗證的**買賣交易系統**，包含兩個獨立的引擎：

1. **動能引擎** — 買強勢（右側），基於 12-1 月動能、分析師上修、盈餘意外、短佣利率
2. **左側抄底引擎** — 買弱勢（左側），RSI(2) 均值回歸 + 200dma 趨勢過濾 + ATR 硬停損

## 核心邏輯（左側引擎）

只在四道關卡全過的時候抄底：
- **結構** — 價格 > 上升的 200 日線（回檔，不是接刀）
- **體質** — Piotroski F-score、正現金流、低槓桿（基本面健全）
- **超賣** — RSI、布林帶、威廉指標、量能投降確認賣壓衰竭
- **確認** — RSI 翻揚、反轉K棒、底背離（不要太早進場）

## 回測證據（8年、50檔大型股）

| 策略 | 勝率 | 盈虧比 | 最深虧損 |
|------|------|--------|----------|
| Naive超賣 | 71.8% | 1.50 | −57% |
| + 200dma過濾 | 73.0% | **1.67** | −57% |
| + ATR硬停損 | 70.4% | 1.43 | **−24%** |

**關鍵發現**：200dma 過濾的價值是**降低尾部風險**（−57% → −24%），讓單筆虧損可控。

## 使用方式

### 1. 完整掃描 + 統一儀表板
```powershell
python batch_run.py AVGO TSM COHR NOK WOLF GLW
```
輸出：
- `reports/dashboard.html` — 4個Tab（Overview / Coverage / 左側抄底 / Backtest）
- `reports/bottom_fishing.html` — 詳細抄底分批劇本

### 2. 單獨掃描左側抄底
```powershell
python bottom_fishing.py NKE PFE PYPL WOLF
```

### 3. 左側策略回測
```powershell
python bottom_fishing_backtest.py --period 8y
# 變數：--rsi 5 --exit-ma 5 --max-hold 12（快速版）
#      --rsi 10 --exit-ma 20 --max-hold 25（搖擺版，預設）
```

### 4. 動能引擎 + 回測
```powershell
python backtest.py
python strategy_backtest.py
```

## 檔案結構

```
├── bottom_fishing.py          — 左側引擎核心
├── bottom_fishing_backtest.py — 左側策略回測
├── signals.py                 — 動能訊號（12個因子）
├── valuation_engine.py        — DCF + Peer P/E + 同行業估值
├── trade_plan.py              — 可執行的交易計畫（進場/停損/目標）
├── regime.py                  — 市場體制過濾（風險開/謹慎/風險關）
├── portfolio.py               — 持倉追蹤 + 風險限制
├── alerts.py                  — 每日紀律檢查
├── strategy_backtest.py       — 動能策略的 monthly 回測
├── batch_run.py               — 批次執行 + 統一儀表板
├── dashboard_generator.py     — 4-tab HTML 儀表板
├── backtest.py                — 因子 IC 驗證（price/vol/reversal）
└── reports/
    ├── dashboard.html         — 統一儀表板
    ├── bottom_fishing.html    — 左側抄底報告
    ├── strategy_backtest.html — 動能策略 equity curve
    └── bottom_fishing_backtest.html — 左側策略回測結果
```

## 關鍵設計

### 反人性紀律（written in advance）
- 分批進場 — 3個預設價位，不臨時加碼
- 硬停損 — 論點被證偽時無條件清倉
- 時間停損 — 25天內沒反彈就出場（不凹單）
- 位置上限 — 單檔 ≤ 15%、單一行業 ≤ 20%、總部署 ≤ 90%

### 誠實回測
- 不看未來數據
- 交易成本納入（0.15% per side）
- 最大不利偏移（MAE）= 你必須忍受的帳面虧損
- 勝率 / 盈虧比 / 期望值都有

## 備註

本專案完全依賴 **yfinance**（免費但有風險）。已知問題見 `data_quality.py` —— SNDK/AVGO 曾返回爛數據，故有內部一致性檢查 + 資料品質過濾。生產環境應升級到 Bloomberg/FactSet。

## License

教育用途。不是投資建議。自擔風險。
