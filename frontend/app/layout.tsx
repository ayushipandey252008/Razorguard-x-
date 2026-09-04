import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "@/components/layout/shell";
import { Toaster } from "sonner";

export const metadata: Metadata = {
  title: "RazorGuard X",
  description: "Agentic payment risk intelligence prototype — synthetic data only.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased">
        <Shell>{children}</Shell>
        <Toaster theme="dark" richColors />
      </body>
    </html>
  );
}
