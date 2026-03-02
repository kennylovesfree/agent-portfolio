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
    <section
      style={{
        border: "1px solid #334155",
        borderRadius: 16,
        padding: 20,
        background: "#111827",
      }}
    >
      <div
        style={{
          width: 180,
          height: 180,
          borderRadius: "50%",
          margin: "0 auto",
          background: `conic-gradient(${gradientStops})`,
        }}
      />
      <ul style={{ listStyle: "none", padding: 0, margin: "16px 0 0" }}>
        {data.map((item) => (
          <li key={item.label} style={{ display: "flex", gap: 8, marginBottom: 6 }}>
            <span style={{ color: item.color }}>●</span>
            <span>
              {item.label}: {item.value}%
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
