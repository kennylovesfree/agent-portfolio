import { ButtonHTMLAttributes } from "react";

type PrimaryButtonProps = ButtonHTMLAttributes<HTMLButtonElement>;

export function PrimaryButton({ className = "", ...props }: PrimaryButtonProps) {
  return <button className={`btn-primary ${className}`.trim()} {...props} />;
}
