import { notFound } from "next/navigation";
import { DocsLayout } from "@/components/DocsLayout";
import { MDXContent } from "@/components/MDXContent";
import { getNavigation } from "@/utils/docs";
import { getBlogPostBySlug, getBlogPostSlugs } from "@/utils/blog";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const slugs = getBlogPostSlugs();
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const post = getBlogPostBySlug(slug);

  if (!post) {
    return { title: "Not Found" };
  }

  const canonicalUrl = `https://everyrow.io/docs/blog/${slug}`;

  return {
    title: post.title,
    description: post.description,
    alternates: { canonical: canonicalUrl },
    openGraph: {
      title: post.title,
      description: post.description,
      url: canonicalUrl,
      images: [{ url: "https://everyrow.io/everyrow-og.png" }],
    },
  };
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export default async function BlogPostPage({ params }: PageProps) {
  const { slug } = await params;
  const post = getBlogPostBySlug(slug);

  if (!post) {
    notFound();
  }

  const navigation = getNavigation();

  return (
    <DocsLayout navigation={navigation}>
      <article className="blog-post">
        <div className="blog-post-meta">
          {post.date && <time dateTime={post.date}>{formatDate(post.date)}</time>}
          {post.author && <span className="blog-post-author">{post.author}</span>}
        </div>
        <MDXContent source={post.content} />
      </article>
    </DocsLayout>
  );
}
