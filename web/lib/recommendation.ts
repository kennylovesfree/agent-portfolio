import { AssetProfile, assetUniverse, GoalType } from "./assetUniverse";
import { OnboardingAnswers } from "./types";

const RECOMMENDATION_COUNT = 8;

function getGoal(answers: OnboardingAnswers): GoalType {
  return answers.goal || "balanced_growth";
}

function getCategoryCap(score: number): Record<AssetProfile["category"], number> {
  if (score >= 70) return { equity: 5, bond: 1, commodity: 2, crypto: 2 };
  if (score >= 45) return { equity: 4, bond: 2, commodity: 2, crypto: 1 };
  return { equity: 3, bond: 3, commodity: 2, crypto: 1 };
}

function getVolatilityPreference(score: number): AssetProfile["volatility"] {
  if (score >= 70) return "high";
  if (score >= 45) return "medium";
  return "low";
}

function rankAsset(asset: AssetProfile, score: number, goal: GoalType): number {
  const center = (asset.minRiskScore + asset.maxRiskScore) / 2;
  const distancePenalty = Math.abs(center - score);
  const goalBonus = asset.goalFit.includes(goal) ? -8 : 0;

  const pref = getVolatilityPreference(score);
  const volatilityBonus = asset.volatility === pref ? -6 : pref === "medium" && asset.volatility !== "high" ? -3 : 0;

  return distancePenalty + goalBonus + volatilityBonus;
}

function pickWithDiversification(candidates: AssetProfile[], score: number, goal: GoalType): AssetProfile[] {
  const caps = getCategoryCap(score);
  const selected: AssetProfile[] = [];
  const categoryCount: Record<AssetProfile["category"], number> = {
    equity: 0,
    bond: 0,
    commodity: 0,
    crypto: 0,
  };

  const ordered = [...candidates].sort((a, b) => rankAsset(a, score, goal) - rankAsset(b, score, goal));

  for (const asset of ordered) {
    if (selected.length >= RECOMMENDATION_COUNT) break;
    if (categoryCount[asset.category] >= caps[asset.category]) continue;
    selected.push(asset);
    categoryCount[asset.category] += 1;
  }

  if (selected.length < RECOMMENDATION_COUNT) {
    for (const asset of ordered) {
      if (selected.length >= RECOMMENDATION_COUNT) break;
      if (selected.some((item) => item.symbol === asset.symbol)) continue;
      selected.push(asset);
    }
  }

  return selected.slice(0, RECOMMENDATION_COUNT);
}

export function buildRecommendedAssets(
  answers: OnboardingAnswers,
  score: number,
  universe: AssetProfile[] = assetUniverse,
): AssetProfile[] {
  const goal = getGoal(answers);

  const strict = universe.filter(
    (asset) => score >= asset.minRiskScore && score <= asset.maxRiskScore && asset.goalFit.includes(goal),
  );

  if (strict.length >= RECOMMENDATION_COUNT) {
    return pickWithDiversification(strict, score, goal);
  }

  const relaxedByGoal = universe.filter((asset) => asset.goalFit.includes(goal));
  const relaxedByScore = universe.filter((asset) => score >= asset.minRiskScore - 10 && score <= asset.maxRiskScore + 10);

  const merged = [...strict];
  for (const asset of [...relaxedByGoal, ...relaxedByScore, ...universe]) {
    if (!merged.some((item) => item.symbol === asset.symbol)) {
      merged.push(asset);
    }
  }

  return pickWithDiversification(merged, score, goal);
}

export const recommendationCount = RECOMMENDATION_COUNT;
