interface AssetCardProps {
  symbol: string;
  name: string;
  riskLevel: string;
}

const riskTone: Record<string, string> = {
  low: "bg-emerald-500/20 text-emerald-200",
  medium: "bg-amber-500/20 text-amber-200",
  high: "bg-rose-500/20 text-rose-200",
};

export function AssetCard({ symbol, name, riskLevel }: AssetCardProps) {
  const tone = riskTone[riskLevel] ?? "bg-slate-500/20 text-slate-200";

  return (
    <article className="surface-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="m-0 text-xs uppercase tracking-[0.2em] text-text-muted">{symbol}</p>
          <p className="mt-1 text-sm font-medium text-text-secondary">{name}</p>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${tone}`}>風險：{riskLevel}</span>
      </div>
    </article>
  );
}
