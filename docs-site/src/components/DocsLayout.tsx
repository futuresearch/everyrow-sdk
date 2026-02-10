"use client";

import { useState } from "react";
import { Sidebar } from "./Sidebar";
import { CodeBlockEnhancer } from "./CodeBlockEnhancer";
import type { NavSection } from "@/utils/docs";

interface DocsLayoutProps {
  navigation: NavSection[];
  children: React.ReactNode;
}

export function DocsLayout({ navigation, children }: DocsLayoutProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="docs-layout">
      <Sidebar
        navigation={navigation}
        isOpen={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
      />
      <main className="docs-content">
        <div className="docs-top-bar">
          <div className="docs-mobile-header">
            <div className="docs-mobile-brand">
              <a href="https://everyrow.io" className="docs-mobile-logo-text">everyrow</a>
              <a
                href="https://futuresearch.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="docs-mobile-futuresearch-link"
              >
                by futu<span className="highlight">re</span>search
              </a>
            </div>
            <button
              className="docs-mobile-menu-btn"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle navigation menu"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
          </div>
          <a
            href="https://futuresearch.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="docs-futuresearch-link"
          >
            by futu<span className="highlight">re</span>search
          </a>
        </div>
        <div className="docs-content-inner">
          {children}
          <CodeBlockEnhancer />
        </div>
      </main>
    </div>
  );
}
