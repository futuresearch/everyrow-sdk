import type { Metadata } from "next";
import { Suspense } from "react";
import "highlight.js/styles/github-dark.min.css";
import "@/styles/notebook.css";
import "./globals.css";
import { PostHogProvider } from "@/components/providers/PostHogProvider";

export const metadata: Metadata = {
  metadataBase: new URL("https://everyrow.io"),
  title: "Everyrow Documentation",
  description: "Documentation for the Everyrow SDK - AI-powered data operations for pandas DataFrames",
  openGraph: {
    siteName: "Everyrow",
    type: "website",
    images: [{ url: "https://everyrow.io/everyrow-og.png" }],
  },
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
