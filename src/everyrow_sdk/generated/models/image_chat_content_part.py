from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_chat_content_part_image_url import ImageChatContentPartImageUrl


T = TypeVar("T", bound="ImageChatContentPart")


@_attrs_define
class ImageChatContentPart:
    """
    Attributes:
        image_url (ImageChatContentPartImageUrl):
        type_ (Literal['image_url'] | Unset):  Default: 'image_url'.
    """

    image_url: ImageChatContentPartImageUrl
    type_: Literal["image_url"] | Unset = "image_url"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        image_url = self.image_url.to_dict()

        type_ = self.type_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "image_url": image_url,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.image_chat_content_part_image_url import ImageChatContentPartImageUrl

        d = dict(src_dict)
        image_url = ImageChatContentPartImageUrl.from_dict(d.pop("image_url"))

        type_ = cast(Literal["image_url"] | Unset, d.pop("type", UNSET))
        if type_ != "image_url" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'image_url', got '{type_}'")

        image_chat_content_part = cls(
            image_url=image_url,
            type_=type_,
        )

        image_chat_content_part.additional_properties = d
        return image_chat_content_part

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
