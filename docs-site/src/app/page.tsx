import { redirect } from "next/navigation";

export default function DocsHome() {
  // Redirect to installation guide as the default page.
  // Note: In production, this redirect is handled by Traefik.
  // This works in local dev but is a no-op in static export.
  redirect("/installation");
}
