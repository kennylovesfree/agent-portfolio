import { ReactNode } from "react";

type PageShellProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
  rightSlot?: ReactNode;
};

export function PageShell({ title, subtitle, children, rightSlot }: PageShellProps) {
  return (
    <main className="page-shell animate-fade-in">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-3 md:mb-8">
        <div>
          <h1 className="m-0 text-3xl font-semibold tracking-tight text-text-primary md:text-4xl">{title}</h1>
          {subtitle ? <p className="mt-2 text-sm text-text-muted md:text-base">{subtitle}</p> : null}
        </div>
        {rightSlot}
      </header>
      {children}
    </main>
  );
}
