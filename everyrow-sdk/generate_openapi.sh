#!/bin/bash
# Script to generate OpenAPI client files and clean up generated files

set -e  # Exit on error

echo "Generating OpenAPI client files..."
uv run openapi-python-client generate \
  --url "https://engine.futuresearch.ai/openapi.json" \
  --config openapi-python-client.yaml \
  --overwrite \
  --meta uv

echo "Removing generated files from everyrow..."
rm -f src/everyrow/README.md
rm -f src/everyrow/.gitignore
rm -f src/everyrow/pyproject.toml

echo "OpenAPI generation complete!"

