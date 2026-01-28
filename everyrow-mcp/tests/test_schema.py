import json

import httpx
import jsonschema
import pytest


def test_server_json_schema(pytestconfig: pytest.Config):
    """Validate server.json against its JSON schema."""
    root = pytestconfig.rootpath

    server_json_path = root / "server.json"
    with open(server_json_path) as f:
        server_json = json.load(f)

    schema_url = server_json.get("$schema")
    assert schema_url, "server.json must have a $schema field"

    response = httpx.get(schema_url)
    response.raise_for_status()
    schema = response.json()

    jsonschema.validate(instance=server_json, schema=schema)
