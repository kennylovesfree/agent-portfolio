import { ElementType, ReactNode } from "react";

type SurfaceCardProps = {
  children: ReactNode;
  className?: string;
  as?: ElementType;
};

export function SurfaceCard({ children, className = "", as: Component = "section" }: SurfaceCardProps) {
  return <Component className={`surface-card p-5 md:p-6 ${className}`.trim()}>{children}</Component>;
}
