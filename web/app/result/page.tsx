"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { AllocationPieChart } from "@/components/AllocationPieChart";
import { AssetCard } from "@/components/AssetCard";
import { assetUniverse } from "@/lib/assetUniverse";
import { OnboardingAnswers } from "@/lib/types";

const SESSION_STORAGE_KEY = "onboarding-answers";

function buildRiskScore(answers: OnboardingAnswers): number {
  const horizon = Number(answers.investmentHorizonYears) || 0;
  const tolerance = Number(answers.riskToleranceLevel) || 0;
  const maxLoss = Number(answers.maxLossPercent) || 0;
  const experience = Number(answers.investmentExperienceYears) || 0;

  const horizonScore = Math.min(horizon * 2, 30);
  const toleranceScore = Math.min(tolerance * 10, 50);
  const lossScore = Math.min(maxLoss * 0.2, 10);
  const expScore = Math.min(experience, 10);
  return Math.round(horizonScore + toleranceScore + lossScore + expScore);
}

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

  const riskScore = useMemo(() => (answers ? buildRiskScore(answers) : null), [answers]);
  const allocation = useMemo(() => {
    if (riskScore === null) return [];
    return buildAllocationByScore(riskScore);
  }, [riskScore]);

  return (
    <main>
      <h1 style={{ marginTop: 0 }}>配置結果頁</h1>
      <p style={{ color: "#cbd5e1" }}>依據問卷結果輸出風險分數與建議配置。</p>

      {!answers ? (
        <section style={{ border: "1px solid #334155", borderRadius: 16, padding: 20, background: "#111827" }}>
          <p>尚無可顯示資料，請先完成問卷。</p>
          <Link href="/" style={{ color: "#93c5fd" }}>
            返回問卷首頁
          </Link>
        </section>
      ) : (
        <>
          <section style={{ border: "1px solid #334155", borderRadius: 16, padding: 20, background: "#111827" }}>
            <h2 style={{ marginTop: 0 }}>問卷摘要</h2>
            <p>風險分數：{riskScore}</p>
            <p>投資目標：{answers.goal || "未選擇"}</p>
          </section>

          <section className="grid two" style={{ alignItems: "start", marginTop: 20 }}>
            <AllocationPieChart data={allocation} />

            <div className="grid">
              {assetUniverse.map((asset) => (
                <AssetCard key={asset.symbol} symbol={asset.symbol} name={asset.name} riskLevel={asset.volatility} />
              ))}
            </div>
          </section>

          <Link href="/report" style={{ display: "inline-block", marginTop: 24, color: "#93c5fd" }}>
            下一步：查看完整報告 →
          </Link>
        </>
      )}
    </main>
  );
}
