export type AssetProfile = {
  symbol: string;
  name: string;
  category: "equity" | "bond" | "commodity" | "crypto";
  volatility: "low" | "medium" | "high";
};

export const assetUniverse: AssetProfile[] = [
  { symbol: "SPY", name: "S&P 500 ETF", category: "equity", volatility: "medium" },
  { symbol: "TLT", name: "20Y Treasury ETF", category: "bond", volatility: "low" },
  { symbol: "GLD", name: "Gold ETF", category: "commodity", volatility: "medium" },
  { symbol: "BTC", name: "Bitcoin", category: "crypto", volatility: "high" },
];
