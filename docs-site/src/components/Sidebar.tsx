"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { NavSection } from "@/utils/docs";

interface SidebarProps {
  navigation: NavSection[];
}

export function Sidebar({ navigation }: SidebarProps) {
  const pathname = usePathname();

  // Remove leading/trailing slashes for comparison
  // Note: usePathname() returns path without basePath, so no need to strip /docs
  const currentSlug = pathname.replace(/^\//, "").replace(/\/$/, "");

  return (
    <aside className="docs-sidebar">
      <div className="docs-sidebar-logo">
        <a href="https://everyrow.io" className="docs-sidebar-logo-text">everyrow</a>
        <Link href="/" className="docs-sidebar-logo-chip">docs</Link>
      </div>

      {navigation.map((section) => (
        <div key={section.title} className="docs-sidebar-section">
          <div className="docs-sidebar-section-title">{section.title}</div>
          <ul className="docs-sidebar-nav">
            {section.items.map((item) => {
              const isActive = currentSlug === item.slug;
              return (
                <li key={item.slug}>
                  <Link
                    href={`/${item.slug}`}
                    className={isActive ? "active" : ""}
                  >
                    {item.title}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </aside>
  );
}
