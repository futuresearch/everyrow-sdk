import type { Metadata } from "next";
import Link from "next/link";
import { DocsLayout } from "@/components/DocsLayout";
import { getNavigation } from "@/utils/docs";
import { getAllBlogPosts } from "@/utils/blog";

export const metadata: Metadata = {
  title: "Blog - Everyrow",
  description:
    "Technical blog posts from the everyrow team about LLMs, data processing, and building AI-powered infrastructure.",
  alternates: {
    canonical: "https://everyrow.io/docs/blog",
  },
  openGraph: {
    title: "Blog - Everyrow",
    description:
      "Technical blog posts from the everyrow team about LLMs, data processing, and building AI-powered infrastructure.",
    url: "https://everyrow.io/docs/blog",
    images: [{ url: "https://everyrow.io/everyrow-og.png" }],
  },
};

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export default async function BlogPage() {
  const navigation = getNavigation();
  const posts = getAllBlogPosts();

  return (
    <DocsLayout navigation={navigation}>
      <h1>Blog</h1>
      <p className="blog-listing-subtitle">
        Technical posts from the everyrow team.
      </p>
      <div className="blog-listing">
        {posts.map((post) => (
          <Link
            key={post.slug}
            href={`/blog/${post.slug}`}
            className="blog-listing-card"
          >
            <div className="blog-listing-card-meta">
              {post.date && <time dateTime={post.date}>{formatDate(post.date)}</time>}
              {post.author && <span>{post.author}</span>}
            </div>
            <h2 className="blog-listing-card-title">{post.title}</h2>
            {post.description && (
              <p className="blog-listing-card-description">{post.description}</p>
            )}
          </Link>
        ))}
      </div>
    </DocsLayout>
  );
}
