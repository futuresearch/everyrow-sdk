import { MetadataRoute } from "next";
import { getAllDocs } from "@/utils/docs";
import { getAllNotebooks } from "@/utils/notebooks";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://everyrow.io/docs";

  const docs = getAllDocs();
  const notebooks = getAllNotebooks();

  const docPages = docs.map((doc) => ({
    url: `${baseUrl}/${doc.slug}`,
    lastModified: doc.lastModified,
    changeFrequency: "weekly" as const,
    priority: 0.8,
  }));

  const notebookPages = notebooks.map((notebook) => ({
    url: `${baseUrl}/notebooks/${notebook.slug}`,
    lastModified: notebook.lastModified,
    changeFrequency: "monthly" as const,
    priority: 0.7,
  }));

  // Use the most recent doc modification for the index page
  const mostRecentDoc = docs.reduce(
    (latest, doc) => (doc.lastModified > latest ? doc.lastModified : latest),
    new Date(0)
  );

  return [
    {
      url: baseUrl,
      lastModified: mostRecentDoc,
      changeFrequency: "weekly" as const,
      priority: 1,
    },
    ...docPages,
    ...notebookPages,
  ];
}
