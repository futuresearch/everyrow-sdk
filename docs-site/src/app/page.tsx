import { redirect } from "next/navigation";

export default function DocsHome() {
  // Redirect to installation guide as the default page
  // basePath is already /docs, so we just redirect to /installation
  redirect("/installation");
}
