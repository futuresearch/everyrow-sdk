import Link from "next/link";
import { DocsLayout } from "@/components/DocsLayout";
import { getNavigation, type NavSection } from "@/utils/docs";

const SECTION_ICONS: Record<string, string> = {
  "Getting Started": "rocket",
  Guides: "book",
  "API Reference": "code",
  "Case Studies": "lightbulb",
};

const SECTION_DESCRIPTIONS: Record<string, string> = {
  "Getting Started": "Install everyrow and start processing data with AI",
  Guides: "Step-by-step tutorials for common data processing tasks",
  "API Reference": "Detailed documentation for all everyrow functions",
  "Case Studies": "Real-world examples with Jupyter notebooks",
};

function SectionCard({ section }: { section: NavSection }) {
  const icon = SECTION_ICONS[section.title] || "file";
  const description = SECTION_DESCRIPTIONS[section.title] || "";
  const firstItem = section.items[0];

  if (!firstItem) return null;

  return (
    <Link href={`/${firstItem.slug}`} className="landing-card">
      <div className="landing-card-icon" data-icon={icon}>
        {icon === "rocket" && (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" />
            <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
            <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
            <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
          </svg>
        )}
        {icon === "book" && (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
          </svg>
        )}
        {icon === "code" && (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
        )}
        {icon === "lightbulb" && (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
            <path d="M9 18h6" />
            <path d="M10 22h4" />
          </svg>
        )}
      </div>
      <h2 className="landing-card-title">{section.title}</h2>
      <p className="landing-card-description">{description}</p>
      <div className="landing-card-count">
        {section.items.length} {section.items.length === 1 ? "page" : "pages"}
      </div>
    </Link>
  );
}

export default function DocsHome() {
  const navigation = getNavigation();

  return (
    <DocsLayout navigation={navigation}>
      <div className="landing-hero">
        <h1 className="landing-title">everyrow documentation</h1>
        <p className="landing-subtitle">
          Process every row of your data with AI-powered research, deduplication,
          merging, ranking, and screening.
        </p>
      </div>

      <div className="landing-grid">
        {navigation.map((section) => (
          <SectionCard key={section.title} section={section} />
        ))}
      </div>

      <div className="landing-quickstart">
        <h2>Quick Install</h2>
        <pre>
          <code>pip install everyrow</code>
        </pre>
        <p>
          Then head to <Link href="/installation">Installation</Link> to set up
          your API key and start processing data.
        </p>
      </div>
    </DocsLayout>
  );
}
