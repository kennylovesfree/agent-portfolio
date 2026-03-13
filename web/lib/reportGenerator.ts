import type { RiskAnswer } from "./riskScoring";

export interface InvestmentReportInput {
  investorProfile: string;
  portfolioSummary: string;
  riskPreference: string;
  horizonYears?: number;
  extraContext?: string;
}

export interface InvestmentReportResult {
  ok: boolean;
  report?: string;
  error?: {
    code:
      | "MISSING_API_KEY"
      | "EMPTY_INPUT"
      | "TIMEOUT"
      | "UPSTREAM_HTTP_ERROR"
      | "UPSTREAM_NETWORK_ERROR"
      | "INVALID_RESPONSE"
      | "SAFETY_REWRITE_FAILED";
    message: string;
  };
}

type ValidationReason =
  | "LENGTH_OUT_OF_RANGE"
  | "INVALID_SECTION_TITLES"
  | "SECTION_CONTENT_TOO_SHORT";

export type PortfolioReport = {
  summary: string;
  riskWarnings: string[];
  actionItems: string[];
};

const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL = "gemini-2.0-flash";
const DEFAULT_TIMEOUT_MS = 20_000;
const DEFAULT_TIMEOUT_RETRIES = 1;
const DEFAULT_REWRITE_ATTEMPTS = 2;
const MIN_LENGTH = 300;
const MAX_LENGTH = 600;
const MIN_VALIDATION_LENGTH = 220;
const MAX_VALIDATION_LENGTH = 1200;
const MIN_SECTION_LENGTH = 20;
const SECTION_TITLES = ["推薦理由", "10年情境", "壓力測試", "風險提醒"] as const;

function countCjkChars(text: string): number {
  return Array.from(text).filter((char) => /[\u4E00-\u9FFF]/.test(char)).length;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function parseSections(text: string): Array<{ title: (typeof SECTION_TITLES)[number]; body: string }> {
  const normalized = text.replace(/\r/g, "").trim();
  if (!normalized) return [];

  const titlePattern = SECTION_TITLES.map(escapeRegExp).join("|");
  const headingPattern = new RegExp(
    `^\\s*(?:#{1,4}\\s*|[-*]\\s*|\\d+[.)]\\s*)?(${titlePattern})[：:]?\\s*(.*)$`,
    "gm",
  );
  const matches = Array.from(normalized.matchAll(headingPattern));

  if (matches.length === 0) return [];

  return matches.map((match, idx) => {
    const nextMatch = matches[idx + 1];
    const lineTail = (match[2] ?? "").trim();
    const bodyStart = (match.index ?? 0) + match[0].length;
    const bodyEnd = nextMatch?.index ?? normalized.length;
    const bodyTail = normalized.slice(bodyStart, bodyEnd).trim();
    const body = [lineTail, bodyTail].filter(Boolean).join("\n").trim();

    return {
      title: match[1] as (typeof SECTION_TITLES)[number],
      body,
    };
  });
}

function pickOrderedSections(
  sections: Array<{ title: (typeof SECTION_TITLES)[number]; body: string }>,
): Array<{ title: (typeof SECTION_TITLES)[number]; body: string }> | null {
  const ordered: Array<{ title: (typeof SECTION_TITLES)[number]; body: string }> = [];
  let cursor = -1;

  for (const title of SECTION_TITLES) {
    const foundIdx = sections.findIndex((section, idx) => idx > cursor && section.title === title);
    if (foundIdx === -1) return null;
    ordered.push(sections[foundIdx]);
    cursor = foundIdx;
  }

  return ordered;
}

function buildPrompt(input: InvestmentReportInput): string {
  const horizonYears = input.horizonYears ?? 10;

  return [
    "你是審慎理財顧問，請輸出『繁體中文』投資分析報告。",
    "嚴格規則：",
    `1) 全文約 ${MIN_LENGTH}-${MAX_LENGTH} 字。`,
    "2) 只能有固定四段，且段落標題與順序必須完全一致：",
    "   推薦理由",
    "   10年情境",
    "   壓力測試",
    "   風險提醒",
    "3) 請明確描述波動、回撤、情境風險與不確定性，不要淡化風險。",
    "4) 可以寫出潛在損失、壓力情境與風險來源，但不要使用保證收益或確定上漲的語氣。",
    "5) 每段都要具體、可讀，避免空泛結論。",
    "",
    "使用者資訊：",
    `- 投資人概況：${input.investorProfile}`,
    `- 持倉摘要：${input.portfolioSummary}`,
    `- 風險偏好：${input.riskPreference}`,
    `- 評估年期：${horizonYears} 年`,
    input.extraContext ? `- 補充資訊：${input.extraContext}` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function buildRewritePrompt(draft: string): string {
  return [
    "請重寫以下內容，保留核心觀點，但必須符合所有規則：",
    `- 繁體中文 ${MIN_LENGTH}-${MAX_LENGTH} 字。`,
    "- 僅四段，標題依序固定為：推薦理由、10年情境、壓力測試、風險提醒。",
    "- 必須保留風險分析、壓力測試與不確定性描述，不要淡化風險。",
    "- 可以描述回撤、波動與虧損情境，但不要使用保證收益或確定上漲的語氣。",
    "",
    "原始內容：",
    draft,
  ].join("\n");
}

function validateDraft(text: string): { ok: boolean; reason?: ValidationReason; retryable: boolean } {
  const cjkChars = countCjkChars(text);
  if (cjkChars < MIN_VALIDATION_LENGTH || cjkChars > MAX_VALIDATION_LENGTH) {
    return { ok: false, reason: "LENGTH_OUT_OF_RANGE", retryable: false };
  }

  const sections = parseSections(text);
  const orderedSections = pickOrderedSections(sections);
  if (!orderedSections) {
    return { ok: false, reason: "INVALID_SECTION_TITLES", retryable: true };
  }

  if (orderedSections.some((section) => countCjkChars(section.body) < MIN_SECTION_LENGTH)) {
    return { ok: false, reason: "SECTION_CONTENT_TOO_SHORT", retryable: false };
  }

  return { ok: true, retryable: false };
}

function getValidationFailureMessage(reason?: ValidationReason): string {
  switch (reason) {
    case "INVALID_SECTION_TITLES":
      return "模型輸出段落標題格式不符。";
    case "LENGTH_OUT_OF_RANGE":
      return "模型輸出篇幅不在預期範圍，但內容已可回退顯示。";
    case "SECTION_CONTENT_TOO_SHORT":
      return "模型輸出部分段落過短，但內容已可回退顯示。";
    default:
      return "模型輸出未通過安全規則。";
  }
}

async function callGemini(prompt: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<InvestmentReportResult> {
  if (typeof window !== "undefined") {
    return {
      ok: false,
      error: {
        code: "UPSTREAM_NETWORK_ERROR",
        message: "generateInvestmentReport 必須在 server 端執行。",
      },
    };
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return {
      ok: false,
      error: {
        code: "MISSING_API_KEY",
        message: "伺服器未設定 GEMINI_API_KEY。",
      },
    };
  }

  const model = process.env.GEMINI_MODEL || DEFAULT_MODEL;
  const timeoutRetries = Number.parseInt(process.env.GEMINI_TIMEOUT_RETRIES || "", 10);
  const maxRetries = Number.isFinite(timeoutRetries) && timeoutRetries >= 0 ? timeoutRetries : DEFAULT_TIMEOUT_RETRIES;

  let lastTimeoutError: InvestmentReportResult | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${GEMINI_API_URL}/${model}:generateContent?key=${apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.4,
            topP: 0.9,
          },
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        return {
          ok: false,
          error: {
            code: "UPSTREAM_HTTP_ERROR",
            message: `Gemini API 請求失敗（HTTP ${response.status}）。請稍後再試。`,
          },
        };
      }

      const data = (await response.json()) as {
        candidates?: Array<{
          content?: {
            parts?: Array<{ text?: string }>;
          };
        }>;
      };

      const text = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
      if (!text) {
        return {
          ok: false,
          error: {
            code: "INVALID_RESPONSE",
            message: "模型回傳格式異常，請稍後重試。",
          },
        };
      }

      return { ok: true, report: text };
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        lastTimeoutError = {
          ok: false,
          error: {
            code: "TIMEOUT",
            message: `模型回應逾時（${timeoutMs / 1000} 秒），已自動重試 ${maxRetries} 次；請稍後再試。`,
          },
        };

        if (attempt < maxRetries) {
          continue;
        }

        return lastTimeoutError;
      }

      return {
        ok: false,
        error: {
          code: "UPSTREAM_NETWORK_ERROR",
          message: "無法連線至模型服務，請稍後重試。",
        },
      };
    } finally {
      clearTimeout(timer);
    }
  }

  return (
    lastTimeoutError ?? {
      ok: false,
      error: {
        code: "TIMEOUT",
        message: "模型回應逾時，請稍後再試。",
      },
    }
  );
}

export async function generateInvestmentReport(input: InvestmentReportInput): Promise<InvestmentReportResult> {
  if (!input.investorProfile?.trim() || !input.portfolioSummary?.trim() || !input.riskPreference?.trim()) {
    return {
      ok: false,
      error: {
        code: "EMPTY_INPUT",
        message: "投資人概況、持倉摘要與風險偏好不可為空。",
      },
    };
  }

  const firstPass = await callGemini(buildPrompt(input));
  if (!firstPass.ok || !firstPass.report) {
    return firstPass;
  }

  const firstValidation = validateDraft(firstPass.report);
  if (firstValidation.ok) {
    return firstPass;
  }

  if (!firstValidation.retryable) {
    return firstPass;
  }

  let draft = firstPass.report;
  for (let attempt = 0; attempt < DEFAULT_REWRITE_ATTEMPTS; attempt += 1) {
    const rewriteResult = await callGemini(buildRewritePrompt(draft));
    if (!rewriteResult.ok || !rewriteResult.report) {
      return rewriteResult;
    }

    const rewriteValidation = validateDraft(rewriteResult.report);
    if (rewriteValidation.ok) {
      return rewriteResult;
    }

    if (!rewriteValidation.retryable) {
      return rewriteResult;
    }

    draft = rewriteResult.report;
  }

  return {
    ok: false,
    error: {
      code: "SAFETY_REWRITE_FAILED",
      message: `${getValidationFailureMessage(validateDraft(draft).reason)} 請稍後再試。`,
    },
  };
}

export function generateReport(answer: RiskAnswer, score: number): PortfolioReport {
  const riskTier = score >= 70 ? "積極" : score >= 45 ? "平衡" : "保守";

  return {
    summary: `根據問答資料，您的投資風格偏向${riskTier}，建議以分散配置控制波動。`,
    riskWarnings: [
      `可承受回撤約 ${answer.drawdownTolerancePercent}% ，遇到市場劇烈波動時需避免追高殺低。`,
      "請定期檢查資產相關性，避免單一市場風險過度集中。",
    ],
    actionItems: [
      "每季再平衡一次，維持原始風險目標。",
      "設定停利停損紀律並記錄投資日誌。",
      "高波動資產可搭配防禦型資產降低淨值回撤。",
    ],
  };
}
