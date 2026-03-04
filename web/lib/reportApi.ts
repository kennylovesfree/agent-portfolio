import type { PortfolioReport } from "./reportGenerator";

export type ReportApiResponse = {
  ok: boolean;
  source: "ai" | "template";
  score: number;
  report: PortfolioReport;
  error?: {
    code: string;
    message: string;
  };
};
