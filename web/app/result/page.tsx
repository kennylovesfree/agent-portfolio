"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { AllocationPieChart } from "@/components/AllocationPieChart";
import { AssetCard } from "@/components/AssetCard";
import { EmptyStateActions } from "@/components/ui/EmptyStateActions";
import { PageShell } from "@/components/ui/PageShell";
import { SurfaceCard } from "@/components/ui/SurfaceCard";
import { buildRiskAnswerFromOnboarding } from "@/lib/mappers";
import { buildRecommendedAssets, recommendationCount } from "@/lib/recommendation";
import { calculateRiskScore } from "@/lib/riskScoring";
import { OnboardingAnswers } from "@/lib/types";

const SESSION_STORAGE_KEY = "onboarding-answers";

function buildAllocationByScore(score: number) {
  if (score >= 70) {
    return [
      { label: "股票", value: 60, color: "#60a5fa" },
      { label: "債券", value: 20, color: "#34d399" },
      { label: "商品", value: 10, color: "#fbbf24" },
      { label: "加密資產", value: 10, color: "#f87171" },
    ];
  }

  if (score >= 45) {
    return [
      { label: "股票", value: 45, color: "#60a5fa" },
      { label: "債券", value: 35, color: "#34d399" },
      { label: "商品", value: 12, color: "#fbbf24" },
      { label: "加密資產", value: 8, color: "#f87171" },
    ];
  }

  return [
    { label: "股票", value: 30, color: "#60a5fa" },
    { label: "債券", value: 50, color: "#34d399" },
    { label: "商品", value: 15, color: "#fbbf24" },
    { label: "加密資產", value: 5, color: "#f87171" },
  ];
}

export default function ResultPage() {
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

  const riskScore = useMemo(() => {
    if (!answers) return null;
    return calculateRiskScore(buildRiskAnswerFromOnboarding(answers));
  }, [answers]);

  const allocation = useMemo(() => {
    if (riskScore === null) return [];
    return buildAllocationByScore(riskScore);
  }, [riskScore]);

  const recommendedAssets = useMemo(() => {
    if (!answers || riskScore === null) return [];
    return buildRecommendedAssets(answers, riskScore);
  }, [answers, riskScore]);

  return (
    <PageShell
      title="配置結果"
      subtitle="依據問卷資料，提供你的風險分數、配置比例與候選標的。"
      rightSlot={
        <Link href="/index.html" className="btn-secondary">
          回首頁
        </Link>
      }
    >
      {!answers ? (
        <EmptyStateActions
          message="尚無可顯示資料。你可以先完成問卷，再回到此頁查看建議配置。"
          actions={[
            { label: "開始問卷", href: "/onboarding" },
            { label: "回首頁", href: "/index.html", variant: "secondary" },
          ]}
        />
      ) : (
        <div className="grid gap-5 md:gap-6">
          <SurfaceCard>
            <h2 className="m-0 text-xl font-semibold tracking-tight">風險摘要</h2>
            <div className="mt-4 grid gap-2 text-sm text-text-secondary md:grid-cols-2">
              <p className="m-0">
                風險分數：<span className="font-semibold text-text-primary">{riskScore}</span>
              </p>
              <p className="m-0">
                投資目標：<span className="font-semibold text-text-primary">{answers.goal || "未選擇"}</span>
              </p>
            </div>
          </SurfaceCard>

          <section className="grid gap-5 lg:grid-cols-[minmax(0,360px)_1fr] lg:gap-6">
            <AllocationPieChart data={allocation} />

            <div className="grid gap-3">
              <h3 className="m-0 text-lg font-semibold tracking-tight">推薦標的（{recommendationCount} 檔）</h3>
              <div className="grid gap-3 md:grid-cols-2">
                {recommendedAssets.map((asset) => (
                  <AssetCard key={asset.symbol} symbol={asset.symbol} name={asset.name} riskLevel={asset.volatility} />
                ))}
              </div>
            </div>
          </section>

          <div>
            <Link href="/report" className="btn-primary">
              下一步：查看完整報告
            </Link>
          </div>
        </div>
      )}
    </PageShell>
  );
}
