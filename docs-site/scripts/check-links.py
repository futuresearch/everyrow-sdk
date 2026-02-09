#!/usr/bin/env python3
"""Check for broken internal links in the static docs build output.

Parses all HTML files in the build output, extracts <a href> links,
and verifies that internal links resolve to existing pages.
"""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "out"
BASE_PATH = "/docs"


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def get_valid_paths(out_dir: Path) -> set[str]:
    """Build a set of valid URL paths from the static build output."""
    valid = set()
    for html_file in out_dir.rglob("*.html"):
        if html_file.name in ("404.html", "_not-found.html"):
            continue
        rel = html_file.relative_to(out_dir)
        # /index.html -> /
        # /foo.html -> /foo
        # /notebooks/bar.html -> /notebooks/bar
        path = "/" + str(rel.with_suffix(""))
        if path.endswith("/index"):
            path = path[: -len("/index")] or "/"
        valid.add(BASE_PATH + path if path != "/" else BASE_PATH)
        # Also allow with trailing slash
        valid.add(BASE_PATH + path + "/" if path != "/" else BASE_PATH + "/")
    # Root path
    valid.add(BASE_PATH)
    valid.add(BASE_PATH + "/")
    return valid


def check_file(html_file: Path, valid_paths: set[str]) -> list[str]:
    """Check all links in an HTML file. Returns list of error messages."""
    rel = html_file.relative_to(OUT_DIR)
    # Determine the URL path for this page (for resolving relative links)
    page_path = "/" + str(rel.with_suffix(""))
    if page_path.endswith("/index"):
        page_path = page_path[: -len("/index")] or "/"
    page_url = f"https://site{BASE_PATH}{page_path}"

    parser = LinkExtractor()
    parser.feed(html_file.read_text())

    errors = []
    for href in parser.links:
        # Skip anchors, mailto, tel, javascript
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        parsed = urlparse(href)

        # Skip external links
        if (
            parsed.scheme in ("http", "https")
            and parsed.netloc
            and parsed.netloc != "site"
        ):
            continue

        # Resolve relative links against the page URL
        if not parsed.scheme and not href.startswith("/"):
            resolved = urlparse(urljoin(page_url, href))
        elif href.startswith("/"):
            resolved = urlparse(f"https://site{href}")
        else:
            resolved = parsed

        path = resolved.path.rstrip("/") or BASE_PATH

        # Only check internal /docs links
        if not path.startswith(BASE_PATH):
            continue

        # Skip static assets
        if "/_next/" in path or path.endswith(
            (".css", ".js", ".png", ".jpg", ".svg", ".ico")
        ):
            continue

        if path not in valid_paths and path + "/" not in valid_paths:
            errors.append(f"  {BASE_PATH}{page_path}: broken link {href!r} -> {path}")

    return errors


def main() -> int:
    if not OUT_DIR.exists():
        print(f"Build output not found at {OUT_DIR}")
        print("Run 'pnpm build' first.")
        return 1

    valid_paths = get_valid_paths(OUT_DIR)
    html_files = [
        f
        for f in OUT_DIR.rglob("*.html")
        if f.name not in ("404.html", "_not-found.html")
    ]

    all_errors = []
    for html_file in sorted(html_files):
        errors = check_file(html_file, valid_paths)
        all_errors.extend(errors)

    if all_errors:
        print(f"Found {len(all_errors)} broken link(s):\n")
        for error in all_errors:
            print(error)
        return 1

    print(f"All links OK across {len(html_files)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
