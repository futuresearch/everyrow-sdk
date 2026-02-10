import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { execSync } from "child_process";
import { getAllNotebooks } from "./notebooks";

// Path to the docs content directory (relative to project root)
const DOCS_DIR = path.join(process.cwd(), "..", "docs");

function getGitLastModified(filePath: string): Date {
  const result = execSync(`git log -1 --format=%cI -- "${filePath}"`, {
    encoding: "utf-8",
  }).trim();
  if (!result) {
    throw new Error(`No git history found for ${filePath}`);
  }
  return new Date(result);
}

export interface DocMeta {
  slug: string;
  title: string;
  description?: string;
  category: string;
  format: "md" | "mdx";
  lastModified: Date;
}

export interface Doc extends DocMeta {
  content: string;
}

function slugToTitle(slug: string): string {
  return slug
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/Llm/g, "LLM")
    .replace(/Ml/g, "ML")
    .replace(/Api/g, "API");
}

function getCategory(filePath: string): string {
  if (filePath.includes("reference/")) return "Reference";
  if (filePath.includes("case_studies/")) return "Case Studies";
  return "Guides";
}

export function getAllDocs(): DocMeta[] {
  const docs: DocMeta[] = [];

  function scanDir(dir: string, prefix: string = "") {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        // Skip data directory
        if (entry.name === "data") continue;
        scanDir(fullPath, path.join(prefix, entry.name));
      } else if (entry.name.endsWith(".md") || entry.name.endsWith(".mdx")) {
        const isMdx = entry.name.endsWith(".mdx");
        const relativePath = path.join(prefix, entry.name);
        const slug = relativePath.replace(/\.mdx?$/, "");
        const content = fs.readFileSync(fullPath, "utf-8");
        const { data } = matter(content);

        docs.push({
          slug,
          title: data.title || slugToTitle(path.basename(slug)),
          description: data.description,
          category: getCategory(relativePath),
          format: isMdx ? "mdx" : "md",
          lastModified: getGitLastModified(fullPath),
        });
      }
    }
  }

  scanDir(DOCS_DIR);
  return docs;
}

export function getDocBySlug(slug: string): Doc | null {
  // Try .mdx first, then .md
  const baseSlug = slug.replace(/\.mdx?$/, "");

  for (const ext of [".mdx", ".md"] as const) {
    const fullPath = path.join(DOCS_DIR, `${baseSlug}${ext}`);

    if (fs.existsSync(fullPath)) {
      const fileContent = fs.readFileSync(fullPath, "utf-8");
      const { data, content } = matter(fileContent);

      return {
        slug: baseSlug,
        title: data.title || slugToTitle(path.basename(baseSlug)),
        description: data.description,
        category: getCategory(baseSlug),
        format: ext === ".mdx" ? "mdx" : "md",
        lastModified: getGitLastModified(fullPath),
        content,
      };
    }
  }

  return null;
}

export function getDocSlugs(): string[] {
  return getAllDocs().map((doc) => doc.slug);
}

// Navigation structure
export interface NavSection {
  title: string;
  items: { slug: string; title: string }[];
}

export function getNavigation(): NavSection[] {
  const docs = getAllDocs();
  const notebooks = getAllNotebooks();

  const guides = docs.filter((d) => d.category === "Guides");
  const reference = docs.filter((d) => d.category === "Reference");

  return [
    {
      title: "Overview",
      items: [
        { slug: "getting-started", title: "Getting Started" },
        { slug: "chaining-operations", title: "Chaining Operations" },
        { slug: "installation", title: "Installation" },
        { slug: "skills-vs-mcp", title: "Skills vs MCP" },
      ],
    },
    {
      title: "API Reference",
      items: reference.map((d) => ({
        slug: d.slug,
        title: d.title.replace(/^reference\//, ""),
      })),
    },
    {
      title: "Guides",
      items: guides
        .filter((d) => !["getting-started", "chaining-operations", "installation", "skills-vs-mcp"].includes(d.slug))
        .map((d) => ({ slug: d.slug, title: d.title })),
    },
    {
      title: "Case Studies",
      items: notebooks.map((n) => ({
        slug: `notebooks/${n.slug}`,
        title: n.title,
      })),
    },
  ];
}
