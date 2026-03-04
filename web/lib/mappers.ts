import { InvestmentReportInput } from "./reportGenerator";
import { RiskAnswer } from "./riskScoring";
import { OnboardingAnswers } from "./types";

const GOAL_LABEL: Record<Exclude<OnboardingAnswers["goal"], "">, string> = {
  capital_preservation: "資本保全",
  balanced_growth: "平衡成長",
  aggressive_growth: "積極成長",
};

export function buildRiskAnswerFromOnboarding(answers: OnboardingAnswers): RiskAnswer {
  return {
    investmentHorizonYears: Number(answers.investmentHorizonYears) || 0,
    drawdownTolerancePercent: Number(answers.maxLossPercent) || 0,
    monthlyContribution: Number(answers.monthlyIncome) || 0,
  };
}

export function buildInvestmentReportInput(answers: OnboardingAnswers, score: number): InvestmentReportInput {
  const goal = answers.goal ? GOAL_LABEL[answers.goal] : "未提供";

  const investorProfile = [
    `年齡 ${answers.age || "未知"} 歲`,
    `投資經驗 ${answers.investmentExperienceYears || 0} 年`,
    `投資目標 ${goal}`,
  ].join("；");

  const portfolioSummary = [
    `可投資資金 NT$${Number(answers.investableAmount) || 0}`,
    `每月可投入 NT$${Number(answers.monthlyIncome) || 0}`,
    `可承受回撤 ${Number(answers.maxLossPercent) || 0}%`,
  ].join("；");

  const riskPreference = [
    `風險承受度 ${Number(answers.riskToleranceLevel) || 0}/5`,
    `模型風險分數 ${score}`,
  ].join("；");

  return {
    investorProfile,
    portfolioSummary,
    riskPreference,
    horizonYears: Number(answers.investmentHorizonYears) || 10,
    extraContext: `目標以 ${goal} 為主。`,
  };
}
