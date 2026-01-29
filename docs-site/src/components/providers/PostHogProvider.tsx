"use client";

import posthog from "posthog-js";
import { PostHogProvider as PHProvider } from "posthog-js/react";
import { useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";

const posthogApiKey = process.env.NEXT_PUBLIC_POSTHOG_KEY;
const posthogApiHost = process.env.NEXT_PUBLIC_POSTHOG_HOST;
const environment = process.env.NEXT_PUBLIC_ENVIRONMENT ?? "development";

if (
  typeof window !== "undefined" &&
  posthogApiKey !== undefined &&
  posthogApiHost !== undefined
) {
  posthog.init(posthogApiKey, {
    api_host: posthogApiHost,
    ui_host: "https://us.posthog.com",
    person_profiles: "identified_only",
    capture_pageview: false,
    capture_pageleave: true,
    loaded: (posthog) => {
      posthog.register({ environment, app: "everyrow-docs" });
    },
  });
}

function PostHogPageView(): null {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (pathname && posthogApiKey) {
      let url = window.origin + pathname;
      if (searchParams && searchParams.toString()) {
        url = url + `?${searchParams.toString()}`;
      }
      posthog.capture("$pageview", {
        $current_url: url,
      });
    }
  }, [pathname, searchParams]);

  return null;
}

export function PostHogProvider({ children }: { children: React.ReactNode }) {
  if (!posthogApiKey) {
    return <>{children}</>;
  }
  return (
    <PHProvider client={posthog}>
      <PostHogPageView />
      {children}
    </PHProvider>
  );
}
