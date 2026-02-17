# 資產配置 Rebalance Copilot（MVP 架構規劃）

> 目的：提供「再平衡建議」的決策支援框架與流程，不包含任何實際投資邏輯或自動化執行。

## 範圍（Scope）
- 提供可審核的再平衡「建議流程」與架構
- 明確區分：資料 → 規則/政策檢查 → 提案 → 人工批准 → 執行規劃（概念層）
- 所有決策最終由人類審核與批准

## 非目標（Out of Scope）
- 任何實際投資邏輯（資產權重計算、風險報酬評估、交易指令生成）
- 連接真實交易、資金或生產環境
- 自動化下單或自動執行

> 註：本專案仍不做交易/下單；但 CLI Demo 現在支援「可選擇」讀取 FinMind 市場資料做持倉估值（預設仍是本地 mock）。

## 核心模組（最小可行）
1. **Data Ingestion**
   - 來源：使用者輸入/匯入資料（持倉、目標配置、限制條件）
   - 只做結構化整理，不做分析與計算

2. **Validation & Normalization**
   - 檢查資料完整性/格式一致性
   - 輸出標準化資料結構

3. **Policy & Rule Gate**
   - 只定義「可允許/禁止」的政策檢查框架
   - 不執行任何投資決策計算

4. **Proposal Generator（框架）**
   - 產出「待審核提案」的格式模板
   - 不生成具體投資建議內容

5. **Human Approval**
   - 顯示提案、要求人工簽核
   - 支援「覆寫 / 拒絕 / 補充理由」

6. **Execution Planning（概念層）**
   - 僅描述「若被批准，執行計畫會如何被記錄」
   - 不輸出任何交易或指令

## 架構圖（MVP 資料流）

```mermaid
graph TD
  A[資料來源: 使用者輸入/匯入] --> B[資料驗證 & 正規化]
  B --> C[規則/政策檢查 Gate]
  C --> D[提案模板生成 (僅框架)]
  D --> E[人工審核/批准]
  E --> F[執行規劃記錄 (概念層)]

  C -. 失敗 .-> G[回饋: 問題清單]
  E -. 拒絕/覆寫 .-> H[人工理由與備註]
```

## 人工控制點（Human-in-the-loop）
- 所有提案必須人工批准
- 人工可隨時覆寫或拒絕
- 不存在自動化執行路徑

## 可審計性（Auditability）
- 每一步輸出均可被記錄
- 保留輸入與輸出版本與時間戳
- 不隱含任何投資決策計算

## 風險與限制
- 不提供投資建議或交易指令
- 不連接外部金融系統
- 不保證任何投資結果

## 後續可擴充方向（需另行批准）
- 讀取外部資料來源
- 加入風險/成本/稅務的分析模組
- 產出具體再平衡建議（需明確合規與治理）

## 使用 FinMind API（可選）
1. 設定環境變數：
   - `USE_FINMIND_API=1`
   - `FINMIND_API_TOKEN=<你的 Bearer token>`
   - `FINMIND_PORTFOLIO_PATH=<portfolio config 路徑>`
2. portfolio config 可參考 `data/finmind_portfolio.example.json`。
3. 執行：`python logic/run_demo.py`

## 台股年化報酬率 API（新增）
- 端點：`POST /api/v1/tw-stock/annual-return`
- 功能：輸入台股代碼或名稱，回傳近 1 年價格報酬率（年化）
- 驗證：亂輸入、不存在代碼/名稱、非 4 位數字代碼，全部回傳 `422`
- 啟動：`uvicorn logic.api_server:app --reload`

## API 快速參考（v1.2）
啟動方式：
```bash
uvicorn logic.api_server:app --reload
```

## Crypto 新聞摘要 API（新增）

- `GET /api/v1/crypto/news-digest`
  - Query:
    - `lang`: `zh` / `en`（預設 `zh`）
    - `limit`: 3-12（預設 6）
    - `force_refresh`: `0` / `1`（預設 `0`）
  - 回傳：每日新聞重點摘要、重要性分數、風險標籤與來源健康狀態

- `POST /api/v1/crypto/news-digest/refresh`
  - Header:
    - `Authorization: Bearer <CRYPTO_NEWS_REFRESH_TOKEN>`
  - Body:
    - `{ "lang": "zh" | "en" | "both" }`
  - 功能：手動/排程刷新摘要快取

### 新增環境變數
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `CRYPTO_NEWS_REFRESH_TOKEN`
- `CRYPTO_NEWS_DEFAULT_LIMIT`（預設 `6`）
- `CRYPTO_NEWS_CACHE_TTL_HOURS`（預設 `24`）
- `CRYPTOPANIC_API_KEY`（可選；未提供時仍可用 RSS 降級）
- `AI_PROVIDER` / `OPENAI_API_KEY` / `GEMINI_API_KEY`（沿用既有 AI 摘要設定）

### 1) 台股年化報酬
- `POST /api/v1/tw-stock/annual-return`

Request:
```json
{
  "query": "2330"
}
```

Response 200（節錄）:
```json
{
  "query": "2330",
  "resolved_stock_id": "2330",
  "resolved_stock_name": "台積電",
  "price_date_latest": "2026-02-07",
  "price_date_base": "2025-02-07",
  "annual_return": 0.2
}
```

### 2) 美股年化報酬
- `POST /api/v1/us-stock/annual-return`

Request:
```json
{
  "query": "AAPL"
}
```

Response 200（節錄）:
```json
{
  "query": "AAPL",
  "resolved_stock_id": "AAPL",
  "resolved_stock_name": "AAPL",
  "price_date_latest": "2026-02-13",
  "price_date_base": "2025-02-13",
  "annual_return": 0.11
}
```

### 3) 投資組合健康度
- `POST /api/v1/portfolio/health-check`
- 錯誤碼：`INVALID_PORTFOLIO_INPUT`（422）、`RISK_ENGINE_ERROR`（500）

Request:
```json
{
  "profile": {
    "age": 35,
    "riskLevel": "balanced",
    "taxRegion": "TW",
    "horizonYears": 10
  },
  "positions": [
    {
      "ticker": "AAPL",
      "market": "US",
      "amountUsd": 700,
      "expectedReturn": 0.12,
      "volatility": 0.28,
      "weight": 0.7
    },
    {
      "ticker": "BND",
      "market": "US",
      "amountUsd": 300,
      "expectedReturn": 0.04,
      "volatility": 0.12,
      "weight": 0.3
    }
  ],
  "portfolio": {
    "expectedReturn": 0.1,
    "volatility": 0.24,
    "maxDrawdown": 0.31
  }
}
```

Response 200（節錄）:
```json
{
  "health_score": 63,
  "risk_band": "medium",
  "components": {
    "concentration_score": 0,
    "diversification_score": 100,
    "volatility_score": 93,
    "drawdown_score": 98
  },
  "flags": [
    "OVER_CONCENTRATION",
    "HIGH_VOLATILITY",
    "DEEP_DRAWDOWN"
  ],
  "explanations": [
    "單一標的比重超過 35%，集中風險偏高。"
  ]
}
```

### 4) 標準壓力情境測試
- `POST /api/v1/portfolio/stress-test`
- 預設情境：
  - `GLOBAL_EQUITY_-15`
  - `RATE_SHOCK_BOND_-8`
  - `FX_USD_TWD_+5`
  - `COMBINED_RISK_OFF`
- 錯誤碼：`INVALID_PORTFOLIO_INPUT`（422）、`RISK_ENGINE_ERROR`（500）

Request:
```json
{
  "positions": [
    {
      "ticker": "AAPL",
      "market": "US",
      "amountUsd": 700,
      "expectedReturn": 0.12,
      "volatility": 0.28,
      "weight": 0.7
    },
    {
      "ticker": "BND",
      "market": "US",
      "amountUsd": 300,
      "expectedReturn": 0.04,
      "volatility": 0.12,
      "weight": 0.3
    }
  ]
}
```

Response 200（節錄）:
```json
{
  "scenario_results": [
    {
      "scenario_id": "GLOBAL_EQUITY_-15",
      "portfolio_pnl_usd": -105.0,
      "drawdown_est": 0.105,
      "risk_label": "medium"
    }
  ],
  "worst_case": {
    "scenario_id": "GLOBAL_EQUITY_-15",
    "portfolio_pnl_usd": -105.0,
    "drawdown_est": 0.105,
    "risk_label": "medium"
  },
  "survival_days_est": 120
}
```

### 5) AI 建議生成
- `POST /api/v1/advice/generate`
- 錯誤碼：`AI_CONFIG_ERROR`（503）、`AI_RATE_LIMITED`（429）、`AI_UPSTREAM_ERROR`（502）
- 說明：請先設定 `OPENAI_API_KEY` 或 `GEMINI_API_KEY`
