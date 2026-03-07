import { NextResponse } from "next/server";
import { buildInvestmentReportInput, buildRiskAnswerFromOnboarding } from "@/lib/mappers";
import { ReportApiResponse } from "@/lib/reportApi";
import { generateInvestmentReport, generateReport, PortfolioReport } from "@/lib/reportGenerator";
import { calculateRiskScore } from "@/lib/riskScoring";
import { OnboardingAnswers } from "@/lib/types";

function isOnboardingAnswers(value: unknown): value is OnboardingAnswers {
  if (!value || typeof value !== "object") return false;
  const raw = value as Record<string, unknown>;

  return (
    "investmentHorizonYears" in raw &&
    "maxLossPercent" in raw &&
    "monthlyIncome" in raw &&
    "riskToleranceLevel" in raw &&
    "goal" in raw
  );
}

function parseAiTextToReport(aiText: string, fallback: PortfolioReport): PortfolioReport {
  const expectedTitles = ["推薦理由", "10年情境", "壓力測試", "風險提醒"] as const;
  const titlePattern = expectedTitles.map((title) => title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const normalized = aiText.replace(/\r/g, "").trim();

  const headingPattern = new RegExp(
    `^\\s*(?:#{1,4}\\s*|[-*]\\s*|\\d+[.)]\\s*)?(${titlePattern})[：:]?\\s*(.*)$`,
    "gm",
  );
  const matches = Array.from(normalized.matchAll(headingPattern));

  const contentByTitle = new Map<string, string>();
  for (let i = 0; i < matches.length; i += 1) {
    const match = matches[i];
    const nextMatch = matches[i + 1];
    const title = match[1];
    const lineTail = (match[2] ?? "").trim();
    const bodyStart = (match.index ?? 0) + match[0].length;
    const bodyEnd = nextMatch?.index ?? normalized.length;
    const bodyTail = normalized.slice(bodyStart, bodyEnd).trim();
    const body = [lineTail, bodyTail].filter(Boolean).join(" ").trim();

    if (title && body && !contentByTitle.has(title)) {
      contentByTitle.set(title, body);
    }
  }

  const recommendation = contentByTitle.get("推薦理由") ?? "";
  const scenario = contentByTitle.get("10年情境") ?? "";
  const stress = contentByTitle.get("壓力測試") ?? "";
  const warning = contentByTitle.get("風險提醒") ?? "";

  const summary = [recommendation, scenario].filter(Boolean).join(" ").trim() || fallback.summary;

  const riskWarnings = [stress, warning].filter(Boolean);

  return {
    summary,
    riskWarnings: riskWarnings.length > 0 ? riskWarnings : fallback.riskWarnings,
    actionItems: fallback.actionItems,
  };
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { answers?: unknown };

    if (!isOnboardingAnswers(body.answers)) {
      return NextResponse.json(
        {
          ok: false,
          source: "template",
          score: 0,
          report: {
            summary: "輸入資料格式不正確，請返回問卷重新填寫。",
            riskWarnings: ["缺少必要欄位，無法完成風險評估。"],
            actionItems: ["請重新提交問卷後再產生報告。"],
          },
          error: { code: "INVALID_INPUT", message: "answers payload is invalid" },
        } satisfies ReportApiResponse,
        { status: 400 },
      );
    }

    const answers = body.answers;
    const riskAnswer = buildRiskAnswerFromOnboarding(answers);
    const score = calculateRiskScore(riskAnswer);
    const templateReport = generateReport(riskAnswer, score);

    const aiResult = await generateInvestmentReport(buildInvestmentReportInput(answers, score));

    if (!aiResult.ok || !aiResult.report) {
      return NextResponse.json(
        {
          ok: true,
          source: "template",
          score,
          report: templateReport,
          error: aiResult.error,
        } satisfies ReportApiResponse,
        { status: 200 },
      );
    }

    return NextResponse.json(
      {
        ok: true,
        source: "ai",
        score,
        report: parseAiTextToReport(aiResult.report, templateReport),
      } satisfies ReportApiResponse,
      { status: 200 },
    );
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        source: "template",
        score: 0,
        report: {
          summary: "系統暫時無法生成 AI 報告，請稍後再試。",
          riskWarnings: ["報告服務發生異常，已切換至保底流程。"],
          actionItems: ["請稍後重新整理頁面，或先完成問卷後再試。"],
        },
        error: {
          code: "INTERNAL_ERROR",
          message: error instanceof Error ? error.message : "unknown error",
        },
      } satisfies ReportApiResponse,
      { status: 500 },
    );
  }
}
