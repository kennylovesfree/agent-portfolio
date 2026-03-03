# Web Frontend (Next.js + TypeScript)

## Entry & Routes

- 首頁唯一入口：`/index.html`
- 主功能入口：`/report`（可直接進入，不需先填問卷）
- 問卷路徑：`/onboarding`（可選流程）
- 問卷後續：`/result` -> `/report`

## UI System (Tailwind)

- 使用 Tailwind CSS + `globals.css` tokens 管理色彩、圓角、陰影與互動狀態。
- 共用 UI 元件位於 `components/ui/`：`PageShell`、`SurfaceCard`、`FormField`、`ProgressHeader`、`PrimaryButton`、`SecondaryButton`、`EmptyStateActions`。
- 目標風格為深色極簡，統一 onboarding/result/report 三頁體驗。

## Setup

```bash
cd web
npm install
npm run dev
```

## Environment Variables

請將 `.env.local.example` 複製為 `.env.local` 後再填入實際值：

- `NEXT_PUBLIC_API_BASE_URL`: 前端對後端 API 的 base URL。
- `GEMINI_API_KEY`: 僅供 server-side 讀取，不應在 client component 暴露。
