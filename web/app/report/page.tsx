"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { EmptyStateActions } from "@/components/ui/EmptyStateActions";
import { PageShell } from "@/components/ui/PageShell";
import { SurfaceCard } from "@/components/ui/SurfaceCard";
import { buildRiskAnswerFromOnboarding } from "@/lib/mappers";
import { ReportApiResponse } from "@/lib/reportApi";
import { generateReport } from "@/lib/reportGenerator";
import { calculateRiskScore } from "@/lib/riskScoring";
import { OnboardingAnswers } from "@/lib/types";

const SESSION_STORAGE_KEY = "onboarding-answers";

export default function ReportPage() {
  const [answers, setAnswers] = useState<OnboardingAnswers | null>(null);
  const [apiResult, setApiResult] = useState<ReportApiResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return;

    try {
      setAnswers(JSON.parse(raw) as OnboardingAnswers);
    } catch {
      setAnswers(null);
    }
  }, []);

  const localRiskAnswer = useMemo(() => (answers ? buildRiskAnswerFromOnboarding(answers) : null), [answers]);
  const localScore = useMemo(() => (localRiskAnswer ? calculateRiskScore(localRiskAnswer) : null), [localRiskAnswer]);
  const localTemplateReport = useMemo(
    () => (localRiskAnswer && localScore !== null ? generateReport(localRiskAnswer, localScore) : null),
    [localRiskAnswer, localScore],
  );

  useEffect(() => {
    if (!answers || localScore === null || !localTemplateReport) {
      setApiResult(null);
      return;
    }

    const controller = new AbortController();

    const run = async () => {
      setLoading(true);

      try {
        const response = await fetch("/api/report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ answers }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP_${response.status}`);
        }

        const data = (await response.json()) as ReportApiResponse;
        if (!data.ok) throw new Error(data.error?.message || "REPORT_API_FAILED");

        setApiResult(data);
      } catch {
        if (controller.signal.aborted) return;

        setApiResult({
          ok: true,
          source: "template",
          score: localScore,
          report: localTemplateReport,
          error: { code: "FETCH_FAILED", message: "報告服務暫時無法連線，已回退標準版報告。" },
        });
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };

    run();

    return () => controller.abort();
  }, [answers, localScore, localTemplateReport]);

  const activeScore = apiResult?.score ?? localScore;
  const activeReport = apiResult?.report ?? localTemplateReport;
  const activeSource = apiResult?.source ?? (activeReport ? "template" : null);

  return (
    <PageShell
      title="投資報告"
      subtitle="有問卷答案時會自動生成 AI 報告；若 AI 服務不可用，系統會回退標準版內容。"
      rightSlot={
        <Link href="/index.html" className="btn-secondary">
          回首頁
        </Link>
      }
    >
      {!answers ? (
        <EmptyStateActions
          message="目前尚未取得你的問卷資料。可先開始問卷，系統會根據你的條件生成個人化報告。"
          actions={[
            { label: "開始問卷", href: "/onboarding" },
            { label: "回首頁", href: "/index.html", variant: "secondary" },
          ]}
        />
      ) : (
        <div className="grid gap-5 md:gap-6">
          {loading && !apiResult ? (
            <SurfaceCard>
              <p className="m-0 text-sm text-text-muted">AI 正在生成報告中...</p>
            </SurfaceCard>
          ) : null}

          {activeSource === "template" ? (
            <SurfaceCard className="border-amber-400/40 bg-amber-500/10">
              <p className="m-0 text-sm text-amber-100">
                已使用標準版報告（原因：{apiResult?.error?.message || "AI 服務不可用"}）。
              </p>
            </SurfaceCard>
          ) : null}

          {activeSource === "ai" ? (
            <SurfaceCard className="border-emerald-400/40 bg-emerald-500/10">
              <p className="m-0 text-sm text-emerald-100">AI 個人化報告已生成完成。</p>
            </SurfaceCard>
          ) : null}

          {activeReport && activeScore !== null ? (
            <>
              <SurfaceCard>
                <p className="m-0 text-sm text-text-muted">風險分數</p>
                <p className="mb-0 mt-2 text-3xl font-semibold tracking-tight text-text-primary">{activeScore}</p>
              </SurfaceCard>

              <SurfaceCard>
                <h2 className="m-0 text-xl font-semibold tracking-tight">摘要</h2>
                <p className="mt-3 text-text-secondary">{activeReport.summary}</p>
              </SurfaceCard>

              <div className="grid gap-5 lg:grid-cols-2 lg:gap-6">
                <SurfaceCard>
                  <h3 className="m-0 text-lg font-semibold tracking-tight">風險提醒</h3>
                  <ul className="mt-3 grid gap-2 pl-5 text-sm text-text-secondary">
                    {activeReport.riskWarnings.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </SurfaceCard>

                <SurfaceCard>
                  <h3 className="m-0 text-lg font-semibold tracking-tight">行動建議</h3>
                  <ul className="mt-3 grid gap-2 pl-5 text-sm text-text-secondary">
                    {activeReport.actionItems.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </SurfaceCard>
              </div>
            </>
          ) : null}
        </div>
      )}
    </PageShell>
  );
}
