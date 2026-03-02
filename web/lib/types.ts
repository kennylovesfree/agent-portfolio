export type OnboardingAnswers = {
  age: number | "";
  monthlyIncome: number | "";
  investableAmount: number | "";
  investmentExperienceYears: number | "";
  investmentHorizonYears: number | "";
  riskToleranceLevel: 1 | 2 | 3 | 4 | 5 | "";
  maxLossPercent: number | "";
  goal: "capital_preservation" | "balanced_growth" | "aggressive_growth" | "";
};

export const TOTAL_ONBOARDING_STEPS = 8;

export const DEFAULT_ONBOARDING_ANSWERS: OnboardingAnswers = {
  age: "",
  monthlyIncome: "",
  investableAmount: "",
  investmentExperienceYears: "",
  investmentHorizonYears: "",
  riskToleranceLevel: "",
  maxLossPercent: "",
  goal: "",
};
