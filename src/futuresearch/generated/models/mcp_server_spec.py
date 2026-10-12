from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="McpServerSpec")


@_attrs_define
class McpServerSpec:
    """A caller's MCP server whose tools the agents get alongside their own.

    Attributes:
        url (str): Streamable-HTTP MCP endpoint. Its host must be on the engine's allow-list. No headers: put a short-
            lived, task-scoped credential in the URL, since request parameters are stored with the task.
    """

    url: str

    def to_dict(self) -> dict[str, Any]:
        url = self.url

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "url": url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        url = d.pop("url")

        mcp_server_spec = cls(
            url=url,
        )

        return mcp_server_spec
