"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { EmptyStateActions } from "@/components/ui/EmptyStateActions";
import { PageShell } from "@/components/ui/PageShell";
import { SurfaceCard } from "@/components/ui/SurfaceCard";
import { buildRiskAnswerFromOnboarding } from "@/lib/mappers";
import { generateReport } from "@/lib/reportGenerator";
import { calculateRiskScore } from "@/lib/riskScoring";
import { OnboardingAnswers } from "@/lib/types";

const SESSION_STORAGE_KEY = "onboarding-answers";
const DEFAULT_GEM_URL = "";
const GOAL_LABEL: Record<Exclude<OnboardingAnswers["goal"], "">, string> = {
  capital_preservation: "資本保全",
  balanced_growth: "平衡成長",
  aggressive_growth: "積極成長",
};

declare global {
  interface Window {
    RebalanceLabConfig?: {
      gemTrialUrl?: string;
    };
  }
}

function getGoalLabel(goal: OnboardingAnswers["goal"]): string {
  return goal ? GOAL_LABEL[goal] : "平衡成長";
}

function getGemUrl(): string {
  return window.RebalanceLabConfig?.gemTrialUrl?.trim() || DEFAULT_GEM_URL;
}

function buildGemPrompts(answers: OnboardingAnswers, score: number): string[] {
  const goal = getGoalLabel(answers.goal);
  const horizonYears = Number(answers.investmentHorizonYears) || 10;
  const maxLossPercent = Number(answers.maxLossPercent) || 0;
  const investableAmount = Number(answers.investableAmount) || 0;
  const monthlyIncome = Number(answers.monthlyIncome) || 0;
  const riskToleranceLevel = Number(answers.riskToleranceLevel) || 3;

  return [
    `我的風險分數是 ${score}、投資目標是${goal}，請幫我整理未來 ${horizonYears} 年最需要注意的三個配置風險。`,
    `我目前可承受最大回撤約 ${maxLossPercent}% ，如果市場先跌一段，我應該怎麼安排資產配置與現金部位？`,
    `我有約 NT$${investableAmount.toLocaleString("zh-TW")} 可投資資金，每月可再投入 NT$${monthlyIncome.toLocaleString("zh-TW")}，請給我一個符合風險承受度 ${riskToleranceLevel}/5 的配置思路。`,
    `請用保守情境、基準情境、樂觀情境三種方式，分析我這種 ${goal} 型投資人在 ${horizonYears} 年內可能遇到的路徑。`,
  ];
}

export default function ReportPage() {
  const [answers, setAnswers] = useState<OnboardingAnswers | null>(null);
  const [gemUrl, setGemUrl] = useState(DEFAULT_GEM_URL);

  useEffect(() => {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return;

    try {
      setAnswers(JSON.parse(raw) as OnboardingAnswers);
    } catch {
      setAnswers(null);
    }
  }, []);

  useEffect(() => {
    setGemUrl(getGemUrl());
  }, []);

  const localRiskAnswer = useMemo(() => (answers ? buildRiskAnswerFromOnboarding(answers) : null), [answers]);
  const localScore = useMemo(() => (localRiskAnswer ? calculateRiskScore(localRiskAnswer) : null), [localRiskAnswer]);
  const report = useMemo(
    () => (localRiskAnswer && localScore !== null ? generateReport(localRiskAnswer, localScore) : null),
    [localRiskAnswer, localScore],
  );
  const gemPrompts = useMemo(
    () => (answers && localScore !== null ? buildGemPrompts(answers, localScore) : []),
    [answers, localScore],
  );

  return (
    <PageShell
      title="投資報告"
      subtitle="先用穩定的風險輪廓整理你的現況，再帶著關鍵問題去和 Gem 深聊。"
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
          {report && localScore !== null ? (
            <>
              <SurfaceCard className="gem-hero-card overflow-hidden">
                <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr] lg:items-end">
                  <div>
                    <p className="m-0 text-sm font-semibold uppercase tracking-[0.28em] text-cyan-100/70">
                      Gem Guided Report
                    </p>
                    <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white md:text-4xl">
                      帶著你的風險輪廓去和 Gem 對話
                    </h2>
                    <p className="mt-4 max-w-2xl text-base text-slate-200 md:text-lg">
                      你的報告先用本地模型穩定整理重點，接著把真正需要判斷的配置問題交給 Gem 深入討論，避免只看一份空泛的自動摘要。
                    </p>
                    <div className="mt-6 flex flex-wrap gap-3">
                      <a
                        href={gemUrl || "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-primary"
                        aria-disabled={!gemUrl}
                      >
                        開啟 Gem 深聊
                      </a>
                      <Link href="/onboarding" className="btn-secondary">
                        重新填寫問卷
                      </Link>
                    </div>
                  </div>

                  <div className="grid gap-3">
                    <div className="rounded-2xl border border-white/10 bg-white/8 p-4 backdrop-blur-sm">
                      <p className="m-0 text-sm text-slate-300">風險分數</p>
                      <p className="mb-0 mt-2 text-4xl font-semibold tracking-tight text-white">{localScore}</p>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-slate-950/30 p-4">
                      <p className="m-0 text-sm text-slate-300">本地摘要</p>
                      <p className="mb-0 mt-2 text-sm leading-7 text-slate-100">{report.summary}</p>
                    </div>
                  </div>
                </div>
              </SurfaceCard>

              <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr] lg:gap-6">
                <SurfaceCard>
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="m-0 text-sm uppercase tracking-[0.22em] text-text-muted">Prompt Starters</p>
                      <h3 className="mt-2 text-xl font-semibold tracking-tight">你可以直接問 Gem 什麼</h3>
                    </div>
                  </div>
                  <div className="mt-4 grid gap-3">
                    {gemPrompts.map((prompt) => (
                      <div key={prompt} className="gem-prompt-card">
                        <p className="m-0 text-sm leading-7 text-slate-100">{prompt}</p>
                      </div>
                    ))}
                  </div>
                </SurfaceCard>

                <div className="grid gap-5">
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

              <SurfaceCard>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="m-0 text-sm uppercase tracking-[0.22em] text-text-muted">Stable Report</p>
                    <h3 className="mt-2 text-xl font-semibold tracking-tight">穩定版投資摘要</h3>
                  </div>
                  <a href={gemUrl || "#"} target="_blank" rel="noopener noreferrer" className="btn-secondary" aria-disabled={!gemUrl}>
                    用 Gem 延伸分析
                  </a>
                </div>
                <p className="mt-4 text-text-secondary">{report.summary}</p>
                <div className="mt-5 grid gap-5 lg:grid-cols-2">
                  <div>
                    <h4 className="m-0 text-base font-semibold tracking-tight text-text-primary">風險提醒</h4>
                    <ul className="mt-3 grid gap-2 pl-5 text-sm text-text-secondary">
                      {report.riskWarnings.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="m-0 text-base font-semibold tracking-tight text-text-primary">下一步建議</h4>
                    <ul className="mt-3 grid gap-2 pl-5 text-sm text-text-secondary">
                      {report.actionItems.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </SurfaceCard>
            </>
          ) : null}
        </div>
      )}
    </PageShell>
  );
}
