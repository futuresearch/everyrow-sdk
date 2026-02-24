import fs from "fs";
import path from "path";
import matter from "gray-matter";

const BLOG_DIR = path.join(process.cwd(), "..", "docs", "blog");

export interface BlogPostMeta {
  slug: string;
  title: string;
  description: string;
  date: string;
  author: string;
}

export interface BlogPost extends BlogPostMeta {
  content: string;
}

export function getAllBlogPosts(): BlogPostMeta[] {
  if (!fs.existsSync(BLOG_DIR)) {
    return [];
  }

  const files = fs.readdirSync(BLOG_DIR);
  const posts: BlogPostMeta[] = [];

  for (const file of files) {
    if (!file.endsWith(".mdx") && !file.endsWith(".md")) continue;

    const fullPath = path.join(BLOG_DIR, file);
    const fileContent = fs.readFileSync(fullPath, "utf-8");
    const { data } = matter(fileContent);
    const slug = file.replace(/\.mdx?$/, "");

    posts.push({
      slug,
      title: data.title || slug,
      description: data.description || "",
      date: data.date || "",
      author: data.author || "",
    });
  }

  // Sort by date, newest first
  return posts.sort((a, b) => b.date.localeCompare(a.date));
}

export function getBlogPostBySlug(slug: string): BlogPost | null {
  for (const ext of [".mdx", ".md"]) {
    const fullPath = path.join(BLOG_DIR, `${slug}${ext}`);

    if (fs.existsSync(fullPath)) {
      const fileContent = fs.readFileSync(fullPath, "utf-8");
      const { data, content } = matter(fileContent);

      return {
        slug,
        title: data.title || slug,
        description: data.description || "",
        date: data.date || "",
        author: data.author || "",
        content,
      };
    }
  }

  return null;
}

export function getBlogPostSlugs(): string[] {
  return getAllBlogPosts().map((p) => p.slug);
}
