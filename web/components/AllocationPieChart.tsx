interface Allocation {
  label: string;
  value: number;
  color: string;
}

interface AllocationPieChartProps {
  data: Allocation[];
}

export function AllocationPieChart({ data }: AllocationPieChartProps) {
  const gradientStops = data
    .map((item, idx) => {
      const start = data.slice(0, idx).reduce((acc, cur) => acc + cur.value, 0);
      const end = start + item.value;
      return `${item.color} ${start}% ${end}%`;
    })
    .join(", ");

  return (
    <section className="surface-card p-5 md:p-6">
      <h3 className="mb-4 mt-0 text-lg font-semibold tracking-tight">資產配置比例</h3>
      <div
        className="mx-auto h-48 w-48 rounded-full"
        style={{ background: `conic-gradient(${gradientStops})` }}
        role="img"
        aria-label="建議資產配置圓餅圖"
      />
      <ul className="mt-5 grid gap-2 p-0 text-sm text-text-secondary" style={{ listStyle: "none" }}>
        {data.map((item) => (
          <li key={item.label} className="flex items-center gap-2">
            <span style={{ color: item.color }} aria-hidden="true">
              ●
            </span>
            <span className="min-w-16">{item.label}</span>
            <span className="text-text-muted">{item.value}%</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
