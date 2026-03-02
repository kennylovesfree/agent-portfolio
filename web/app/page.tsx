"use client";

import { CSSProperties, Dispatch, SetStateAction, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { DEFAULT_ONBOARDING_ANSWERS, OnboardingAnswers, TOTAL_ONBOARDING_STEPS } from "@/lib/types";
import { validateStep } from "@/lib/validation";

const SESSION_STORAGE_KEY = "onboarding-answers";

type GoalOption = OnboardingAnswers["goal"];

export default function HomePage() {
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
    <main>
      <h1 style={{ marginTop: 0 }}>投資風險調查問卷</h1>
      <p style={{ color: "#cbd5e1" }}>完成 8 步問卷後，系統會提供對應的組合建議。</p>

      <section style={{ background: "#111827", borderRadius: 16, border: "1px solid #334155", padding: 20, marginTop: 16 }}>
        <p style={{ marginTop: 0, color: "#93c5fd" }}>
          第 {currentStep + 1} / {TOTAL_ONBOARDING_STEPS} 題（進度 {progressValue}%）
        </p>
        <div style={{ width: "100%", height: 8, borderRadius: 999, background: "#1e293b", overflow: "hidden", marginBottom: 20 }}>
          <div style={{ width: `${progressValue}%`, height: "100%", background: "#2563eb" }} />
        </div>

        {renderStep(currentStep, answers, updateNumberField, setAnswers)}

        {errorMessage ? <p style={{ color: "#fca5a5" }}>{errorMessage}</p> : null}

        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 20 }}>
          <button
            type="button"
            onClick={goPrevious}
            disabled={currentStep === 0}
            style={{
              border: "1px solid #334155",
              borderRadius: 10,
              padding: "10px 14px",
              background: "transparent",
              color: "#e2e8f0",
              cursor: currentStep === 0 ? "not-allowed" : "pointer",
              opacity: currentStep === 0 ? 0.5 : 1,
            }}
          >
            上一步
          </button>
          <button
            type="button"
            onClick={goNext}
            style={{
              border: 0,
              borderRadius: 10,
              padding: "10px 14px",
              background: "#2563eb",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            {currentStep === TOTAL_ONBOARDING_STEPS - 1 ? "提交並查看建議" : "下一步"}
          </button>
        </div>
      </section>
    </main>
  );
}

function renderStep(
  step: number,
  answers: OnboardingAnswers,
  updateNumberField: <K extends keyof OnboardingAnswers>(key: K, rawValue: string) => void,
  setAnswers: Dispatch<SetStateAction<OnboardingAnswers>>,
) {
  const inputStyle: CSSProperties = {
    width: "100%",
    marginTop: 8,
    borderRadius: 10,
    border: "1px solid #334155",
    background: "#0f172a",
    color: "#e2e8f0",
    padding: "10px 12px",
  };

  switch (step) {
    case 0:
      return (
        <label>
          1. 您的年齡？
          <input style={inputStyle} type="number" min={18} max={80} value={answers.age} onChange={(e) => updateNumberField("age", e.target.value)} />
        </label>
      );
    case 1:
      return (
        <label>
          2. 您的每月收入（TWD）？
          <input style={inputStyle} type="number" min={0} value={answers.monthlyIncome} onChange={(e) => updateNumberField("monthlyIncome", e.target.value)} />
        </label>
      );
    case 2:
      return (
        <label>
          3. 目前可投入的資金（TWD）？
          <input style={inputStyle} type="number" min={0} value={answers.investableAmount} onChange={(e) => updateNumberField("investableAmount", e.target.value)} />
        </label>
      );
    case 3:
      return (
        <label>
          4. 投資經驗年數？
          <input
            style={inputStyle}
            type="number"
            min={0}
            max={60}
            value={answers.investmentExperienceYears}
            onChange={(e) => updateNumberField("investmentExperienceYears", e.target.value)}
          />
        </label>
      );
    case 4:
      return (
        <label>
          5. 預計投資年限（年）？
          <input
            style={inputStyle}
            type="number"
            min={1}
            max={50}
            value={answers.investmentHorizonYears}
            onChange={(e) => updateNumberField("investmentHorizonYears", e.target.value)}
          />
        </label>
      );
    case 5:
      return (
        <label>
          6. 風險承受度（1 最低，5 最高）
          <input
            style={inputStyle}
            type="number"
            min={1}
            max={5}
            value={answers.riskToleranceLevel}
            onChange={(e) => updateNumberField("riskToleranceLevel", e.target.value)}
          />
        </label>
      );
    case 6:
      return (
        <label>
          7. 可接受最大虧損（%）
          <input
            style={inputStyle}
            type="number"
            min={0}
            max={100}
            value={answers.maxLossPercent}
            onChange={(e) => updateNumberField("maxLossPercent", e.target.value)}
          />
        </label>
      );
    case 7:
      return (
        <div>
          <p style={{ margin: 0 }}>8. 主要投資目標？</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 10, marginTop: 10 }}>
            {[
              { value: "capital_preservation", label: "保本" },
              { value: "balanced_growth", label: "穩健成長" },
              { value: "aggressive_growth", label: "積極成長" },
            ].map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setAnswers((prev) => ({ ...prev, goal: option.value as GoalOption }))}
                style={{
                  borderRadius: 10,
                  padding: "10px 12px",
                  border: answers.goal === option.value ? "1px solid #2563eb" : "1px solid #334155",
                  background: answers.goal === option.value ? "#1d4ed8" : "#0f172a",
                  color: "#e2e8f0",
                  cursor: "pointer",
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      );
    default:
      return null;
  }
}
