export const serverEnv = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
  geminiApiKey: process.env.GEMINI_API_KEY,
};
