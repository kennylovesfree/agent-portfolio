import type { OnboardingAnswers } from "./types";

export type GoalType = Exclude<OnboardingAnswers["goal"], "">;

export type AssetProfile = {
  symbol: string;
  name: string;
  category: "equity" | "bond" | "commodity" | "crypto";
  volatility: "low" | "medium" | "high";
  region: "US" | "TW" | "Global";
  goalFit: GoalType[];
  minRiskScore: number;
  maxRiskScore: number;
};

export const assetUniverse: AssetProfile[] = [
  { symbol: "SPY", name: "S&P 500 ETF", category: "equity", volatility: "medium", region: "US", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },
  { symbol: "VOO", name: "Vanguard S&P 500 ETF", category: "equity", volatility: "medium", region: "US", goalFit: ["capital_preservation", "balanced_growth", "aggressive_growth"], minRiskScore: 30, maxRiskScore: 100 },
  { symbol: "VTI", name: "Vanguard Total Stock Market ETF", category: "equity", volatility: "medium", region: "US", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },
  { symbol: "QQQ", name: "Invesco QQQ Trust", category: "equity", volatility: "high", region: "US", goalFit: ["aggressive_growth"], minRiskScore: 55, maxRiskScore: 100 },
  { symbol: "IVV", name: "iShares Core S&P 500 ETF", category: "equity", volatility: "medium", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 30, maxRiskScore: 100 },
  { symbol: "IWM", name: "iShares Russell 2000 ETF", category: "equity", volatility: "high", region: "US", goalFit: ["aggressive_growth"], minRiskScore: 60, maxRiskScore: 100 },
  { symbol: "SCHD", name: "Schwab U.S. Dividend Equity ETF", category: "equity", volatility: "medium", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 25, maxRiskScore: 85 },
  { symbol: "VIG", name: "Vanguard Dividend Appreciation ETF", category: "equity", volatility: "medium", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 25, maxRiskScore: 90 },
  { symbol: "VEA", name: "Vanguard FTSE Developed Markets ETF", category: "equity", volatility: "medium", region: "Global", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },
  { symbol: "VWO", name: "Vanguard FTSE Emerging Markets ETF", category: "equity", volatility: "high", region: "Global", goalFit: ["aggressive_growth"], minRiskScore: 60, maxRiskScore: 100 },
  { symbol: "ACWI", name: "iShares MSCI ACWI ETF", category: "equity", volatility: "medium", region: "Global", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },
  { symbol: "0050", name: "元大台灣50", category: "equity", volatility: "medium", region: "TW", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },
  { symbol: "006208", name: "富邦台50", category: "equity", volatility: "medium", region: "TW", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 30, maxRiskScore: 90 },
  { symbol: "00878", name: "國泰永續高股息", category: "equity", volatility: "medium", region: "TW", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 20, maxRiskScore: 80 },
  { symbol: "0056", name: "元大高股息", category: "equity", volatility: "medium", region: "TW", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 20, maxRiskScore: 85 },

  { symbol: "BND", name: "Vanguard Total Bond Market ETF", category: "bond", volatility: "low", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 70 },
  { symbol: "AGG", name: "iShares Core U.S. Aggregate Bond ETF", category: "bond", volatility: "low", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 70 },
  { symbol: "IEF", name: "iShares 7-10 Year Treasury ETF", category: "bond", volatility: "low", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 65 },
  { symbol: "SHY", name: "iShares 1-3 Year Treasury ETF", category: "bond", volatility: "low", region: "US", goalFit: ["capital_preservation"], minRiskScore: 0, maxRiskScore: 50 },
  { symbol: "LQD", name: "iShares iBoxx Investment Grade Corporate Bond ETF", category: "bond", volatility: "low", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 10, maxRiskScore: 75 },
  { symbol: "TLT", name: "20Y Treasury ETF", category: "bond", volatility: "low", region: "US", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 75 },
  { symbol: "BNDX", name: "Vanguard Total International Bond ETF", category: "bond", volatility: "low", region: "Global", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 65 },
  { symbol: "00720B", name: "元大投資級公司債", category: "bond", volatility: "low", region: "TW", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 70 },
  { symbol: "00679B", name: "元大美債20年", category: "bond", volatility: "low", region: "TW", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 70 },
  { symbol: "00687B", name: "國泰20年美債", category: "bond", volatility: "low", region: "TW", goalFit: ["capital_preservation", "balanced_growth"], minRiskScore: 0, maxRiskScore: 70 },

  { symbol: "GLD", name: "SPDR Gold Shares", category: "commodity", volatility: "medium", region: "Global", goalFit: ["capital_preservation", "balanced_growth", "aggressive_growth"], minRiskScore: 15, maxRiskScore: 100 },
  { symbol: "IAU", name: "iShares Gold Trust", category: "commodity", volatility: "medium", region: "Global", goalFit: ["capital_preservation", "balanced_growth", "aggressive_growth"], minRiskScore: 15, maxRiskScore: 100 },
  { symbol: "SLV", name: "iShares Silver Trust", category: "commodity", volatility: "high", region: "Global", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 50, maxRiskScore: 100 },
  { symbol: "DBC", name: "Invesco DB Commodity Index Tracking Fund", category: "commodity", volatility: "medium", region: "Global", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },
  { symbol: "USO", name: "United States Oil Fund", category: "commodity", volatility: "high", region: "Global", goalFit: ["aggressive_growth"], minRiskScore: 65, maxRiskScore: 100 },
  { symbol: "PDBC", name: "Invesco Optimum Yield Diversified Commodity Strategy", category: "commodity", volatility: "medium", region: "Global", goalFit: ["balanced_growth", "aggressive_growth"], minRiskScore: 35, maxRiskScore: 100 },

  { symbol: "BTC", name: "Bitcoin", category: "crypto", volatility: "high", region: "Global", goalFit: ["aggressive_growth"], minRiskScore: 70, maxRiskScore: 100 },
  { symbol: "ETH", name: "Ethereum", category: "crypto", volatility: "high", region: "Global", goalFit: ["aggressive_growth"], minRiskScore: 75, maxRiskScore: 100 },
  { symbol: "SOL", name: "Solana", category: "crypto", volatility: "high", region: "Global", goalFit: ["aggressive_growth"], minRiskScore: 80, maxRiskScore: 100 },
  { symbol: "BNB", name: "BNB", category: "crypto", volatility: "high", region: "Global", goalFit: ["aggressive_growth"], minRiskScore: 80, maxRiskScore: 100 },
];
