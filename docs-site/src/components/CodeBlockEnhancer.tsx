"use client";

import { useEffect } from "react";

export function CodeBlockEnhancer() {
  useEffect(() => {
    const enhanceCodeBlocks = () => {
      const preElements = document.querySelectorAll("pre");

      preElements.forEach((pre) => {
        // Skip if already wrapped
        if (pre.parentElement?.classList.contains("code-block-wrapper")) {
          return;
        }

        // Create wrapper
        const wrapper = document.createElement("div");
        wrapper.className = "code-block-wrapper";

        // Create copy button
        const button = document.createElement("button");
        button.className = "copy-button";
        button.setAttribute("aria-label", "Copy code");
        button.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>`;

        button.addEventListener("click", async () => {
          const code = pre.textContent || "";
          await navigator.clipboard.writeText(code);

          button.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>`;
          button.setAttribute("aria-label", "Copied");

          setTimeout(() => {
            button.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>`;
            button.setAttribute("aria-label", "Copy code");
          }, 2000);
        });

        // Wrap the pre element
        pre.parentNode?.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        wrapper.appendChild(button);
      });
    };

    enhanceCodeBlocks();
  }, []);

  return null;
}
