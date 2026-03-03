import { ButtonHTMLAttributes } from "react";

type SecondaryButtonProps = ButtonHTMLAttributes<HTMLButtonElement>;

export function SecondaryButton({ className = "", ...props }: SecondaryButtonProps) {
  return <button className={`btn-secondary ${className}`.trim()} {...props} />;
}
