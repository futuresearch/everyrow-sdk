import { notFound } from "next/navigation";
import { DocsLayout } from "@/components/DocsLayout";
import { getNavigation } from "@/utils/docs";
import { getNotebookBySlug, getNotebookSlugs } from "@/utils/notebooks";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const slugs = getNotebookSlugs();
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const notebook = getNotebookBySlug(slug);

  if (!notebook) {
    return { title: "Not Found" };
  }

  return {
    title: `${notebook.title} | Everyrow Docs`,
    description: `Case study notebook: ${notebook.title}`,
  };
}

export default async function NotebookPage({ params }: PageProps) {
  const { slug } = await params;
  const notebook = getNotebookBySlug(slug);

  if (!notebook) {
    notFound();
  }

  const navigation = getNavigation();

  return (
    <DocsLayout navigation={navigation}>
      <article
        className="notebook-content"
        dangerouslySetInnerHTML={{ __html: notebook.html }}
      />
    </DocsLayout>
  );
}
