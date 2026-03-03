import Link from "next/link";

type ActionItem = {
  label: string;
  href: string;
  variant?: "primary" | "secondary";
};

type EmptyStateActionsProps = {
  message: string;
  actions: ActionItem[];
};

export function EmptyStateActions({ message, actions }: EmptyStateActionsProps) {
  return (
    <div className="surface-card p-6">
      <p className="mb-4 text-text-secondary">{message}</p>
      <div className="flex flex-wrap gap-3">
        {actions.map((action) => (
          <Link
            key={`${action.href}-${action.label}`}
            href={action.href}
            className={action.variant === "secondary" ? "btn-secondary" : "btn-primary"}
          >
            {action.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
