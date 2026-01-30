import fs from "fs";
import path from "path";

const NOTEBOOKS_DIR = path.join(process.cwd(), "src", "notebooks");

export interface NotebookMeta {
  slug: string;
  title: string;
}

export interface Notebook extends NotebookMeta {
  html: string;
}

function slugToTitle(slug: string): string {
  return slug
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/Crm/g, "CRM")
    .replace(/Llm/g, "LLM")
    .replace(/Ml/g, "ML")
    .replace(/Api/g, "API");
}

export function getAllNotebooks(): NotebookMeta[] {
  if (!fs.existsSync(NOTEBOOKS_DIR)) {
    return [];
  }

  const files = fs.readdirSync(NOTEBOOKS_DIR);
  return files
    .filter((f) => f.endsWith(".html"))
    .map((f) => {
      const slug = f.replace(/\.html$/, "");
      return {
        slug,
        title: slugToTitle(slug),
      };
    })
    .sort((a, b) => a.title.localeCompare(b.title));
}

export function getNotebookBySlug(slug: string): Notebook | null {
  const filePath = path.join(NOTEBOOKS_DIR, `${slug}.html`);

  if (!fs.existsSync(filePath)) {
    return null;
  }

  const html = fs.readFileSync(filePath, "utf-8");

  return {
    slug,
    title: slugToTitle(slug),
    html,
  };
}

export function getNotebookSlugs(): string[] {
  return getAllNotebooks().map((n) => n.slug);
}
