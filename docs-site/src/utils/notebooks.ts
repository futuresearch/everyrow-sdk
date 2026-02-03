import fs from "fs";
import path from "path";

const NOTEBOOKS_DIR = path.join(process.cwd(), "src", "notebooks");
const SOURCE_NOTEBOOKS_DIR = path.join(process.cwd(), "..", "docs", "case_studies");

export interface NotebookMeta {
  slug: string;
  title: string;
  description: string;
}

export interface Notebook extends NotebookMeta {
  html: string;
}

interface NotebookCell {
  cell_type: string;
  source: string | string[];
}

interface NotebookJson {
  cells: NotebookCell[];
  metadata?: {
    everyrow?: {
      description?: string;
    };
  };
}

function extractMetadataFromSource(slug: string): { title: string; description: string } {
  const sourcePath = path.join(SOURCE_NOTEBOOKS_DIR, slug, "notebook.ipynb");

  if (!fs.existsSync(sourcePath)) {
    return { title: slugToTitle(slug), description: "" };
  }

  try {
    const content = fs.readFileSync(sourcePath, "utf-8");
    const notebook: NotebookJson = JSON.parse(content);
    const cells = notebook.cells || [];

    // Extract title from first markdown cell's H1
    let title = slugToTitle(slug);
    if (cells.length > 0 && cells[0].cell_type === "markdown") {
      const source = cells[0].source;
      const cellContent = Array.isArray(source) ? source.join("") : source;
      const firstLine = cellContent.trim().split("\n")[0];
      if (firstLine.startsWith("# ")) {
        title = firstLine.slice(2).trim();
      }
    }

    // Extract description from notebook metadata
    const description = notebook.metadata?.everyrow?.description || "";

    return { title, description };
  } catch {
    return { title: slugToTitle(slug), description: "" };
  }
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
      const { title, description } = extractMetadataFromSource(slug);
      return { slug, title, description };
    })
    .sort((a, b) => a.title.localeCompare(b.title));
}

export function getNotebookBySlug(slug: string): Notebook | null {
  const filePath = path.join(NOTEBOOKS_DIR, `${slug}.html`);

  if (!fs.existsSync(filePath)) {
    return null;
  }

  const html = fs.readFileSync(filePath, "utf-8");
  const { title, description } = extractMetadataFromSource(slug);

  return {
    slug,
    title,
    description,
    html,
  };
}

export function getNotebookSlugs(): string[] {
  return getAllNotebooks().map((n) => n.slug);
}
