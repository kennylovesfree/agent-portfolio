# Web Frontend (Next.js + TypeScript)

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
