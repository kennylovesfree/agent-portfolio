# Rebalance Copilot（MVP Demo）

## 問題敘述
本 Demo 展示「資產再平衡」的受控、人類審核與可稽核流程。系統僅提出可供人工審查的文字化提案，**不提供投資建議**、不連外部資料、也不會自動執行任何交易。

## 高階架構
- **Data Ingestion**：讀取本地 mock 資料（JSON）
- **Drift Calculation**：純計算目前 vs 目標權重的偏離
- **Policy Checks**：規則閘門（限制/禁止/需要人工審核）
- **Proposal Generator**：產生 1–3 個「文字化提案」
- **Human Approval**：人工審核、批准或覆寫
- **Audit Log**：可回放的輸入/規則/理由/批准狀態

## 工作流程圖（文字）
```
[Mock Portfolio JSON]
        |
        v
[Drift Calculation]
        |
        v
[Policy Checks] --(Needs Human Review)--> [Human Approval]
        |
        v
[Proposal Generator]
        |
        v
[Human Approval]
        |
        v
[Audit Log Snapshot]
```

## 人機責任分工
**Agent 可以做**
- 解析 mock 資料
- 計算偏離（純數學）
- 套用規則檢查（閘門）
- 產生文字化提案與理由

**Agent 不可做**
- 任何投資建議或最佳化計算
- 任何自動化交易/執行
- 在不確定時下結論

**Human 必須做**
- 審核提案與理由
- 決定是否批准/拒絕/覆寫
- 在不確定時給出裁決

## 可稽核性（Auditability）
- 所有輸入、檢查結果、提案、人工決策皆可記錄
- 可重播同一組輸入得到一致的輸出（純決定性）
- 任何不確定性會標記為 **Needs Human Review**

## 重要限制與聲明
- 本專案僅為面試展示用 Demo
- 不提供投資建議，不應用於真實資金操作
- 不連接任何外部 API 或市場資料

## FinMind API（可選，預設仍為 mock）

現在 CLI 支援以 FinMind v4 `data` endpoint 讀取行情，將 `positions` 估值後換算成 `holdings.weight`。

### 環境變數
- `USE_FINMIND_API=1`：啟用 API ingestion
- `FINMIND_API_TOKEN`：FinMind Bearer token
- `FINMIND_PORTFOLIO_PATH`：portfolio 設定 JSON 路徑

### portfolio 設定檔範例
請參考 `data/finmind_portfolio.example.json`。

### 執行
```bash
python logic/run_demo.py
```

若未設定 `USE_FINMIND_API=1`，系統會回退到 `data/sample_portfolio.json`。

## 台股年化報酬率 API（新增）
- 端點：`POST /api/v1/tw-stock/annual-return`
- 請先設定：`FINMIND_API_TOKEN`
- 啟動：`uvicorn logic.api_server:app --reload`

### Request
```json
{
  "query": "2330"
}
```

### Success 200
```json
{
  "query": "2330",
  "resolved_stock_id": "2330",
  "resolved_stock_name": "台積電",
  "price_date_latest": "2026-02-07",
  "price_latest": 600.0,
  "price_date_base": "2025-02-07",
  "price_base": 500.0,
  "annual_return": 0.2
}
```

### Error 422（不通過）
```json
{
  "error_code": "INVALID_QUERY",
  "message": "台股代碼需為 4 位數字。",
  "details": {
    "query": "123"
  }
}
```
