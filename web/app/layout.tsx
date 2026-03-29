import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "Portfolio Risk Assistant",
  description: "Next.js 前端子專案（問答、結果、報告）",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-Hant">
      <body>
        <Script src="/gem-config.js" strategy="beforeInteractive" />
        {children}
      </body>
    </html>
  );
}
