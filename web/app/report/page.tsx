"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
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
    <main>
      <h1 style={{ marginTop: 0 }}>投資報告頁</h1>

      {!answers || !report || score === null ? (
        <section style={{ border: "1px solid #334155", borderRadius: 16, padding: 20, background: "#111827" }}>
          <p>尚無可顯示資料，請先完成問卷。</p>
          <Link href="/" style={{ color: "#93c5fd" }}>
            返回問卷首頁
          </Link>
        </section>
      ) : (
        <>
          <p style={{ color: "#cbd5e1" }}>風險分數：{score}</p>
          <section style={{ background: "#111827", borderRadius: 16, padding: 20, border: "1px solid #334155" }}>
            <h2 style={{ marginTop: 0 }}>摘要</h2>
            <p>{report.summary}</p>

            <h3>風險提醒</h3>
            <ul>
              {report.riskWarnings.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>

            <h3>行動建議</h3>
            <ul>
              {report.actionItems.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>

          <Link href="/" style={{ display: "inline-block", marginTop: 24, color: "#93c5fd" }}>
            回到問答首頁
          </Link>
        </>
      )}
    </main>
  );
}
