import json
import tomllib

import pytest


def test_version_consistency(pytestconfig: pytest.Config):
    """Check that version is consistent across pyproject.toml, plugin.json, gemini-extension.json, and marketplace.json."""
    root = pytestconfig.rootpath

    pyproject_path = root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    pyproject_version = pyproject["project"]["version"]

    plugin_json_path = root / ".claude-plugin" / "plugin.json"
    with open(plugin_json_path) as f:
        plugin_json = json.load(f)
    plugin_version = plugin_json["version"]

    gemini_json_path = root / "gemini-extension.json"
    with open(gemini_json_path) as f:
        gemini_json = json.load(f)
    gemini_version = gemini_json["version"]

    marketplace_json_path = root / ".claude-plugin" / "marketplace.json"
    with open(marketplace_json_path) as f:
        marketplace_json = json.load(f)
    marketplace_version = marketplace_json["plugins"][0]["version"]

    assert pyproject_version == plugin_version, (
        f"pyproject.toml version ({pyproject_version}) != plugin.json version ({plugin_version})"
    )
    assert pyproject_version == gemini_version, (
        f"pyproject.toml version ({pyproject_version}) != gemini-extension.json version ({gemini_version})"
    )
    assert pyproject_version == marketplace_version, (
        f"pyproject.toml version ({pyproject_version}) != marketplace.json version ({marketplace_version})"
    )
