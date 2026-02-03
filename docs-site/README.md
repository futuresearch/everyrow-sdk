# Everyrow Docs Site

Next.js static site for [everyrow.io/docs](https://everyrow.io/docs).

## Notebook Integration

Case study notebooks are converted to HTML and embedded in the docs site:

```
docs/case_studies/*/notebook.ipynb
        ↓
    nbconvert --template basic (body-only HTML)
        ↓
docs-site/src/notebooks/*.html
        ↓
    Next.js build (reads HTML, wraps in DocsLayout)
        ↓
docs-site/out/notebooks/*.html (full pages with sidebar)
```

The `src/notebooks/` directory is gitignored since files are generated at build time.

### Notebook Metadata

Page titles and descriptions for SEO are extracted from the **first markdown cell** of each notebook:

```markdown
# This Becomes the Page Title

This first paragraph becomes the meta description for search engines.

## Setup
...
```

The extraction logic (`src/utils/notebooks.ts`):
1. Parses the source `.ipynb` JSON
2. Takes the H1 (`# Title`) as the page title
3. Takes the first non-empty paragraph after the H1 as the description

**Requirements** (enforced by `scripts/validate-notebooks.py` in CI):
- First cell must be markdown
- Must start with an H1 title (`# Title`)
- Must have a description paragraph before any `##` heading or code

## Local Development

```bash
pnpm dev
```

This automatically runs `predev` which converts notebooks before starting the dev server (~2s).

## CI/Production

The GitHub Actions workflow (`deploy-docs.yaml`) runs the same conversion step before `pnpm build`, then deploys the `out/` directory to GCS.
