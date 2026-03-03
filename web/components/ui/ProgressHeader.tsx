type ProgressHeaderProps = {
  currentStep: number;
  totalSteps: number;
  progressValue: number;
};

export function ProgressHeader({ currentStep, totalSteps, progressValue }: ProgressHeaderProps) {
  return (
    <div className="mb-6">
      <p className="mb-3 text-sm font-medium text-text-secondary">
        第 {currentStep + 1} / {totalSteps} 題
        <span className="ml-2 text-text-muted">({progressValue}%)</span>
      </p>
      <div className="h-2 overflow-hidden rounded-full bg-slate-700/70">
        <div
          className="h-full rounded-full bg-accent-blue transition-all duration-200"
          style={{ width: `${progressValue}%` }}
          aria-hidden="true"
        />
      </div>
    </div>
  );
}
