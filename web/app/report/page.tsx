"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { EmptyStateActions } from "@/components/ui/EmptyStateActions";
import { PageShell } from "@/components/ui/PageShell";
import { SurfaceCard } from "@/components/ui/SurfaceCard";
import { generateReport } from "@/lib/reportGenerator";
import { calculateRiskScore, RiskAnswer } from "@/lib/riskScoring";
import { OnboardingAnswers } from "@/lib/types";

const SESSION_STORAGE_KEY = "onboarding-answers";

function buildRiskAnswer(answers: OnboardingAnswers): RiskAnswer {
  return {
    investmentHorizonYears: Number(answers.investmentHorizonYears) || 0,
    drawdownTolerancePercent: Number(answers.maxLossPercent) || 0,
    monthlyContribution: Number(answers.monthlyIncome) || 0,
  };
}

export default function ReportPage() {
  const [answers, setAnswers] = useState<OnboardingAnswers | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return;

    try {
      setAnswers(JSON.parse(raw) as OnboardingAnswers);
    } catch {
      setAnswers(null);
    }
  }, []);

  const answer = useMemo(() => (answers ? buildRiskAnswer(answers) : null), [answers]);
  const score = useMemo(() => (answer ? calculateRiskScore(answer) : null), [answer]);
  const report = useMemo(() => (answer && score !== null ? generateReport(answer, score) : null), [answer, score]);

  return (
    <PageShell
      title="投資報告"
      subtitle="你可以直接瀏覽報告頁；若先完成問卷，會得到更個人化的評估內容。"
      rightSlot={
        <Link href="/index.html" className="btn-secondary">
          回首頁
        </Link>
      }
    >
      {!answers || !report || score === null ? (
        <EmptyStateActions
          message="目前尚未取得你的問卷資料。可先開始問卷，系統會根據你的條件生成個人化報告。"
          actions={[
            { label: "開始問卷", href: "/onboarding" },
            { label: "回首頁", href: "/index.html", variant: "secondary" },
          ]}
        />
      ) : (
        <div className="grid gap-5 md:gap-6">
          <SurfaceCard>
            <p className="m-0 text-sm text-text-muted">風險分數</p>
            <p className="mb-0 mt-2 text-3xl font-semibold tracking-tight text-text-primary">{score}</p>
          </SurfaceCard>

          <SurfaceCard>
            <h2 className="m-0 text-xl font-semibold tracking-tight">摘要</h2>
            <p className="mt-3 text-text-secondary">{report.summary}</p>
          </SurfaceCard>

          <div className="grid gap-5 lg:grid-cols-2 lg:gap-6">
            <SurfaceCard>
              <h3 className="m-0 text-lg font-semibold tracking-tight">風險提醒</h3>
              <ul className="mt-3 grid gap-2 pl-5 text-sm text-text-secondary">
                {report.riskWarnings.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </SurfaceCard>

            <SurfaceCard>
              <h3 className="m-0 text-lg font-semibold tracking-tight">行動建議</h3>
              <ul className="mt-3 grid gap-2 pl-5 text-sm text-text-secondary">
                {report.actionItems.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </SurfaceCard>
          </div>
        </div>
      )}
    </PageShell>
  );
}
