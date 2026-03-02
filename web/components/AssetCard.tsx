interface AssetCardProps {
  symbol: string;
  name: string;
  riskLevel: string;
}

export function AssetCard({ symbol, name, riskLevel }: AssetCardProps) {
  return (
    <article
      style={{
        border: "1px solid #334155",
        borderRadius: 14,
        padding: 16,
        background: "rgba(15, 23, 42, 0.75)",
      }}
    >
      <strong>{symbol}</strong>
      <p style={{ margin: "8px 0", color: "#e2e8f0" }}>{name}</p>
      <span
        style={{
          display: "inline-block",
          background: "#1d4ed8",
          borderRadius: 999,
          padding: "4px 10px",
          fontSize: 12,
        }}
      >
        風險等級：{riskLevel}
      </span>
    </article>
  );
}
