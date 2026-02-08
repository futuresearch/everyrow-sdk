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
        <div className="docs-content-inner">
          {children}
          <CodeBlockEnhancer />
        </div>
      </main>
    </div>
  );
}
