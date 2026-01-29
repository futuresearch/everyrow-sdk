"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { NavSection } from "@/utils/docs";

interface SidebarProps {
  navigation: NavSection[];
}

export function Sidebar({ navigation }: SidebarProps) {
  const pathname = usePathname();

  // Remove /docs prefix and trailing slash for comparison
  const currentSlug = pathname.replace(/^\/docs\/?/, "").replace(/\/$/, "");

  return (
    <aside className="docs-sidebar">
      <Link href="/" className="docs-sidebar-logo">
        Everyrow
      </Link>

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
