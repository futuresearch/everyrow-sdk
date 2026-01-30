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

## Local Development

```bash
pnpm dev
```

This automatically runs `predev` which converts notebooks before starting the dev server (~2s).

## CI/Production

The GitHub Actions workflow (`deploy-docs.yaml`) runs the same conversion step before `pnpm build`, then deploys the `out/` directory to GCS.
