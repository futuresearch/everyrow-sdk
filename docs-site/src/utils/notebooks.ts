import fs from "fs";
import path from "path";
import { execSync } from "child_process";

const NOTEBOOKS_DIR = path.join(process.cwd(), "src", "notebooks");
const CASE_STUDIES_DIR = path.join(process.cwd(), "..", "docs", "case_studies");

function getGitLastModified(filePath: string): Date {
  try {
    const result = execSync(`git log -1 --format=%cI -- "${filePath}"`, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
    return result ? new Date(result) : new Date();
  } catch {
    return new Date();
  }
}

export interface NotebookMeta {
  slug: string;
  title: string;
  lastModified: Date;
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
      // Get last commit date from source notebook, not generated HTML
      const sourcePath = path.join(CASE_STUDIES_DIR, slug, "notebook.ipynb");
      return {
        slug,
        title: slugToTitle(slug),
        lastModified: getGitLastModified(sourcePath),
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
  // Get last commit date from source notebook, not generated HTML
  const sourcePath = path.join(CASE_STUDIES_DIR, slug, "notebook.ipynb");

  return {
    slug,
    title: slugToTitle(slug),
    lastModified: getGitLastModified(sourcePath),
    html,
  };
}

export function getNotebookSlugs(): string[] {
  return getAllNotebooks().map((n) => n.slug);
}
