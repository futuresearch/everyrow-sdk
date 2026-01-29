import type { Metadata } from "next";
import "./globals.css";

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
      <body>{children}</body>
    </html>
  );
}
