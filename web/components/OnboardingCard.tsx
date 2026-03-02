interface OnboardingCardProps {
  title: string;
  description: string;
  step: number;
}

export function OnboardingCard({ title, description, step }: OnboardingCardProps) {
  return (
    <article
      style={{
        border: "1px solid #334155",
        borderRadius: 16,
        padding: 20,
        background: "#161d34",
      }}
    >
      <p style={{ color: "#93c5fd", margin: 0 }}>Step {step}</p>
      <h3 style={{ margin: "8px 0 10px" }}>{title}</h3>
      <p style={{ color: "#cbd5e1", margin: 0 }}>{description}</p>
    </article>
  );
}
