import { OnboardingAnswers } from "./types";

export type ValidationResult = {
  valid: boolean;
  error?: string;
};

const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

export const validateAge = (age: OnboardingAnswers["age"]): ValidationResult => {
  if (!isNumber(age)) return { valid: false, error: "請輸入年齡" };
  if (age < 18 || age > 80) return { valid: false, error: "年齡需介於 18 到 80 歲" };
  return { valid: true };
};

export const validateMonthlyIncome = (
  monthlyIncome: OnboardingAnswers["monthlyIncome"],
): ValidationResult => {
  if (!isNumber(monthlyIncome)) return { valid: false, error: "請輸入每月收入" };
  if (monthlyIncome < 0) return { valid: false, error: "每月收入不得為負數" };
  return { valid: true };
};

export const validateInvestableAmount = (
  investableAmount: OnboardingAnswers["investableAmount"],
): ValidationResult => {
  if (!isNumber(investableAmount)) return { valid: false, error: "請輸入可投資金額" };
  if (investableAmount < 0) return { valid: false, error: "可投資金額不得為負數" };
  return { valid: true };
};

export const validateInvestmentExperienceYears = (
  years: OnboardingAnswers["investmentExperienceYears"],
): ValidationResult => {
  if (!isNumber(years)) return { valid: false, error: "請輸入投資經驗年數" };
  if (years < 0 || years > 60) return { valid: false, error: "投資經驗需介於 0 到 60 年" };
  return { valid: true };
};

export const validateInvestmentHorizonYears = (
  years: OnboardingAnswers["investmentHorizonYears"],
): ValidationResult => {
  if (!isNumber(years)) return { valid: false, error: "請輸入投資期限" };
  if (years < 1 || years > 50) return { valid: false, error: "投資期限需介於 1 到 50 年" };
  return { valid: true };
};

export const validateRiskToleranceLevel = (
  level: OnboardingAnswers["riskToleranceLevel"],
): ValidationResult => {
  if (!isNumber(level)) return { valid: false, error: "請選擇風險承受度" };
  if (level < 1 || level > 5) return { valid: false, error: "風險承受度需介於 1 到 5" };
  return { valid: true };
};

export const validateMaxLossPercent = (
  percent: OnboardingAnswers["maxLossPercent"],
): ValidationResult => {
  if (!isNumber(percent)) return { valid: false, error: "請輸入最大可承受虧損比例" };
  if (percent < 0 || percent > 100) {
    return { valid: false, error: "最大可承受虧損比例需介於 0% 到 100%" };
  }
  return { valid: true };
};

export const validateGoal = (goal: OnboardingAnswers["goal"]): ValidationResult => {
  if (!goal) return { valid: false, error: "請選擇投資目標" };
  return { valid: true };
};

export const stepValidators: Array<(answers: OnboardingAnswers) => ValidationResult> = [
  (answers) => validateAge(answers.age),
  (answers) => validateMonthlyIncome(answers.monthlyIncome),
  (answers) => validateInvestableAmount(answers.investableAmount),
  (answers) => validateInvestmentExperienceYears(answers.investmentExperienceYears),
  (answers) => validateInvestmentHorizonYears(answers.investmentHorizonYears),
  (answers) => validateRiskToleranceLevel(answers.riskToleranceLevel),
  (answers) => validateMaxLossPercent(answers.maxLossPercent),
  (answers) => validateGoal(answers.goal),
];

export const validateStep = (stepIndex: number, answers: OnboardingAnswers): ValidationResult => {
  const validator = stepValidators[stepIndex];
  if (!validator) return { valid: false, error: "無效步驟" };
  return validator(answers);
};
