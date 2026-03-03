import { ReactNode } from "react";

type FormFieldProps = {
  label: string;
  hint?: string;
  error?: string | null;
  children: ReactNode;
};

export function FormField({ label, hint, error, children }: FormFieldProps) {
  return (
    <div className="animate-step-in">
      <label className="field-label">{label}</label>
      {children}
      {hint ? <p className="field-hint">{hint}</p> : null}
      {error ? <p className="field-error" role="alert">{error}</p> : null}
    </div>
  );
}
