"use client";

import { Dispatch, SetStateAction, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { FormField } from "@/components/ui/FormField";
import { PageShell } from "@/components/ui/PageShell";
import { PrimaryButton } from "@/components/ui/PrimaryButton";
import { ProgressHeader } from "@/components/ui/ProgressHeader";
import { SecondaryButton } from "@/components/ui/SecondaryButton";
import { SurfaceCard } from "@/components/ui/SurfaceCard";
import { DEFAULT_ONBOARDING_ANSWERS, OnboardingAnswers, TOTAL_ONBOARDING_STEPS } from "@/lib/types";
import { validateStep } from "@/lib/validation";

const SESSION_STORAGE_KEY = "onboarding-answers";

type GoalOption = OnboardingAnswers["goal"];

export default function OnboardingPage() {
  const router = useRouter();
  const [answers, setAnswers] = useState<OnboardingAnswers>(DEFAULT_ONBOARDING_ANSWERS);
  const [currentStep, setCurrentStep] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const progressValue = useMemo(
    () => Math.round(((currentStep + 1) / TOTAL_ONBOARDING_STEPS) * 100),
    [currentStep],
  );

  const updateNumberField = <K extends keyof OnboardingAnswers>(key: K, rawValue: string) => {
    setAnswers((prev) => ({
      ...prev,
      [key]: rawValue === "" ? "" : Number(rawValue),
    }));
  };

  const goNext = () => {
    const validation = validateStep(currentStep, answers);
    if (!validation.valid) {
      setErrorMessage(validation.error ?? "請檢查輸入內容");
      return;
    }

    setErrorMessage(null);

    if (currentStep === TOTAL_ONBOARDING_STEPS - 1) {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(answers));
      router.push("/result");
      return;
    }

    setCurrentStep((prev) => prev + 1);
  };

  const goPrevious = () => {
    setErrorMessage(null);
    setCurrentStep((prev) => Math.max(0, prev - 1));
  };

  return (
    <PageShell
      title="投資風險調查問卷"
      subtitle="快速完成 8 步評估，建立你的投資風險輪廓與配置方向。"
      rightSlot={
        <a href="/index.html" className="btn-secondary">
          回首頁
        </a>
      }
    >
      <SurfaceCard className="mx-auto w-full max-w-3xl">
        <ProgressHeader currentStep={currentStep} totalSteps={TOTAL_ONBOARDING_STEPS} progressValue={progressValue} />

        {errorMessage ? (
          <div className="mb-4 rounded-xl border border-danger-border bg-danger-soft px-4 py-3 text-sm text-red-100" role="alert">
            {errorMessage}
          </div>
        ) : null}

        <div key={currentStep} className="mb-6 min-h-40 animate-step-in">
          {renderStep(currentStep, answers, updateNumberField, setAnswers, errorMessage)}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <SecondaryButton type="button" onClick={goPrevious} disabled={currentStep === 0}>
            上一步
          </SecondaryButton>
          <PrimaryButton type="button" onClick={goNext}>
            {currentStep === TOTAL_ONBOARDING_STEPS - 1 ? "提交並查看建議" : "下一步"}
          </PrimaryButton>
        </div>
      </SurfaceCard>
    </PageShell>
  );
}

function renderStep(
  step: number,
  answers: OnboardingAnswers,
  updateNumberField: <K extends keyof OnboardingAnswers>(key: K, rawValue: string) => void,
  setAnswers: Dispatch<SetStateAction<OnboardingAnswers>>,
  errorMessage: string | null,
) {
  const stepError = errorMessage;

  switch (step) {
    case 0:
      return (
        <FormField label="1. 您的年齡？" hint="範圍 18 ~ 80 歲" error={stepError}>
          <input
            type="number"
            min={18}
            max={80}
            value={answers.age}
            onChange={(e) => updateNumberField("age", e.target.value)}
            className="input-base"
            placeholder="例如：32"
          />
        </FormField>
      );
    case 1:
      return (
        <FormField label="2. 每月收入（新台幣）" hint="用於評估可持續投入能力" error={stepError}>
          <input
            type="number"
            min={0}
            step={1000}
            value={answers.monthlyIncome}
            onChange={(e) => updateNumberField("monthlyIncome", e.target.value)}
            className="input-base"
            placeholder="例如：80000"
          />
        </FormField>
      );
    case 2:
      return (
        <FormField label="3. 目前可投資金額（新台幣）" hint="可立即配置的本金" error={stepError}>
          <input
            type="number"
            min={0}
            step={10000}
            value={answers.investableAmount}
            onChange={(e) => updateNumberField("investableAmount", e.target.value)}
            className="input-base"
            placeholder="例如：500000"
          />
        </FormField>
      );
    case 3:
      return (
        <FormField label="4. 投資經驗（年）" hint="你過去實際參與市場的時間" error={stepError}>
          <input
            type="number"
            min={0}
            max={60}
            value={answers.investmentExperienceYears}
            onChange={(e) => updateNumberField("investmentExperienceYears", e.target.value)}
            className="input-base"
            placeholder="例如：5"
          />
        </FormField>
      );
    case 4:
      return (
        <FormField label="5. 預計投資期限（年）" hint="建議以中長期視角評估" error={stepError}>
          <input
            type="number"
            min={1}
            max={50}
            value={answers.investmentHorizonYears}
            onChange={(e) => updateNumberField("investmentHorizonYears", e.target.value)}
            className="input-base"
            placeholder="例如：10"
          />
        </FormField>
      );
    case 5:
      return (
        <FormField label="6. 風險承受度" hint="1 最低，5 最高" error={stepError}>
          <select
            value={answers.riskToleranceLevel}
            onChange={(e) =>
              setAnswers((prev) => ({
                ...prev,
                riskToleranceLevel: e.target.value ? (Number(e.target.value) as 1 | 2 | 3 | 4 | 5) : "",
              }))
            }
            className="input-base"
          >
            <option value="">請選擇</option>
            <option value="1">1 - 保守</option>
            <option value="2">2 - 偏保守</option>
            <option value="3">3 - 平衡</option>
            <option value="4">4 - 成長</option>
            <option value="5">5 - 積極</option>
          </select>
        </FormField>
      );
    case 6:
      return (
        <FormField label="7. 可承受最大回撤（%）" hint="市場波動下可接受的短期虧損" error={stepError}>
          <input
            type="number"
            min={0}
            max={100}
            value={answers.maxLossPercent}
            onChange={(e) => updateNumberField("maxLossPercent", e.target.value)}
            className="input-base"
            placeholder="例如：20"
          />
        </FormField>
      );
    case 7:
      return (
        <FormField label="8. 投資目標" hint="請選擇最貼近你的主要目標" error={stepError}>
          <fieldset className="grid gap-3" aria-label="投資目標選擇">
            {[
              ["capital_preservation", "資本保全", "偏向穩定，重視本金保護"],
              ["balanced_growth", "平衡成長", "兼顧波動控制與資產增值"],
              ["aggressive_growth", "積極成長", "追求較高報酬並承擔較高波動"],
            ].map(([value, label, desc]) => {
              const selected = answers.goal === value;
              return (
                <label
                  key={value}
                  className={`cursor-pointer rounded-xl border px-4 py-3 transition ${
                    selected
                      ? "border-accent-blue bg-[rgba(79,143,247,0.16)]"
                      : "border-surface-border bg-surface-strong hover:border-accent-blue"
                  }`}
                >
                  <span className="flex items-center gap-2 text-sm font-medium text-text-secondary">
                    <input
                      type="radio"
                      name="goal"
                      value={value}
                      checked={selected}
                      onChange={(e) => setAnswers((prev) => ({ ...prev, goal: e.target.value as GoalOption }))}
                      className="h-4 w-4 accent-accent-blue"
                    />
                    {label}
                  </span>
                  <span className="mt-1 block text-xs text-text-muted">{desc}</span>
                </label>
              );
            })}
          </fieldset>
        </FormField>
      );
    default:
      return null;
  }
}
