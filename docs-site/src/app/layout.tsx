import type { Metadata } from "next";
import { Suspense } from "react";
import "highlight.js/styles/github-dark.min.css";
import "./globals.css";
import { PostHogProvider } from "@/components/providers/PostHogProvider";

export const metadata: Metadata = {
  title: "Everyrow Documentation",
  description: "Documentation for the Everyrow SDK - AI-powered data operations for pandas DataFrames",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Suspense fallback={null}>
          <PostHogProvider>{children}</PostHogProvider>
        </Suspense>
      </body>
    </html>
  );
}
