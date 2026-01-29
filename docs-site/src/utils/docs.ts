import fs from "fs";
import path from "path";
import matter from "gray-matter";

// Path to the docs content directory (relative to project root)
const DOCS_DIR = path.join(process.cwd(), "..", "docs");

export interface DocMeta {
  slug: string;
  title: string;
  description?: string;
  category: string;
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
      } else if (entry.name.endsWith(".md")) {
        const relativePath = path.join(prefix, entry.name);
        const slug = relativePath.replace(/\.md$/, "");
        const content = fs.readFileSync(fullPath, "utf-8");
        const { data } = matter(content);

        docs.push({
          slug,
          title: data.title || slugToTitle(path.basename(slug)),
          description: data.description,
          category: getCategory(relativePath),
        });
      }
    }
  }

  scanDir(DOCS_DIR);
  return docs;
}

export function getDocBySlug(slug: string): Doc | null {
  // Handle both with and without .md extension
  const slugPath = slug.endsWith(".md") ? slug : `${slug}.md`;
  const fullPath = path.join(DOCS_DIR, slugPath);

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContent = fs.readFileSync(fullPath, "utf-8");
  const { data, content } = matter(fileContent);

  return {
    slug: slug.replace(/\.md$/, ""),
    title: data.title || slugToTitle(path.basename(slug)),
    description: data.description,
    category: getCategory(slug),
    content,
  };
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

  const guides = docs.filter((d) => d.category === "Guides");
  const reference = docs.filter((d) => d.category === "Reference");

  return [
    {
      title: "Getting Started",
      items: [
        { slug: "installation", title: "Installation" },
      ],
    },
    {
      title: "Guides",
      items: guides
        .filter((d) => d.slug !== "installation")
        .map((d) => ({ slug: d.slug, title: d.title })),
    },
    {
      title: "API Reference",
      items: reference.map((d) => ({
        slug: d.slug,
        title: d.title.replace(/^reference\//, ""),
      })),
    },
  ];
}
