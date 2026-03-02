import Link from "next/link";
import { calculateRiskScore } from "@/lib/riskScoring";
import { generateReport } from "@/lib/reportGenerator";

export default function ReportPage() {
  const answer = {
    investmentHorizonYears: 10,
    drawdownTolerancePercent: 20,
    monthlyContribution: 2000,
  };

  const score = calculateRiskScore(answer);
  const report = generateReport(answer, score);

  return (
    <main>
      <h1 style={{ marginTop: 0 }}>投資報告頁</h1>
      <p style={{ color: "#cbd5e1" }}>風險分數：{score}</p>

      <section style={{ background: "#111827", borderRadius: 16, padding: 20, border: "1px solid #334155" }}>
        <h2 style={{ marginTop: 0 }}>摘要</h2>
        <p>{report.summary}</p>

        <h3>風險提醒</h3>
        <ul>
          {report.riskWarnings.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>

        <h3>行動建議</h3>
        <ul>
          {report.actionItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <Link href="/" style={{ display: "inline-block", marginTop: 24, color: "#93c5fd" }}>
        回到問答首頁
      </Link>
    </main>
  );
}
