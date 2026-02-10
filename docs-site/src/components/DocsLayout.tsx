import { Sidebar } from "./Sidebar";
import { CodeBlockEnhancer } from "./CodeBlockEnhancer";
import type { NavSection } from "@/utils/docs";

interface DocsLayoutProps {
  navigation: NavSection[];
  children: React.ReactNode;
}

export function DocsLayout({ navigation, children }: DocsLayoutProps) {
  return (
    <div className="docs-layout">
      <Sidebar navigation={navigation} />
      <main className="docs-content">
        <div className="docs-top-bar">
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
