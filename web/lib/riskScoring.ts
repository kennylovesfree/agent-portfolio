export type RiskAnswer = {
  investmentHorizonYears: number;
  drawdownTolerancePercent: number;
  monthlyContribution: number;
};

export function calculateRiskScore(answer: RiskAnswer): number {
  const horizonScore = Math.min(answer.investmentHorizonYears * 4, 40);
  const toleranceScore = Math.min(answer.drawdownTolerancePercent * 2, 40);
  const contributionScore = Math.min(answer.monthlyContribution / 500, 20);
  return Math.round(horizonScore + toleranceScore + contributionScore);
}
