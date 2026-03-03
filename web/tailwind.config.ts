import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "bg-base": "var(--bg-base)",
        "bg-elev": "var(--bg-elev)",
        surface: "var(--surface)",
        "surface-strong": "var(--surface-strong)",
        "surface-border": "var(--surface-border)",
        "text-primary": "var(--text-primary)",
        "text-secondary": "var(--text-secondary)",
        "text-muted": "var(--text-muted)",
        "accent-blue": "var(--accent-blue)",
        "accent-blue-press": "var(--accent-blue-press)",
        "danger-soft": "var(--danger-soft)",
        "danger-border": "var(--danger-border)",
      },
      boxShadow: {
        soft: "0 10px 30px rgba(8, 15, 30, 0.35)",
        card: "0 16px 40px rgba(8, 15, 30, 0.45)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
      keyframes: {
        "step-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      animation: {
        "step-in": "step-in 180ms ease-out",
        "fade-in": "fade-in 220ms ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
