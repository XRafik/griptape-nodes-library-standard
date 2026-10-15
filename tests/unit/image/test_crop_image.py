"""Tests for CropImage's crop-box clamping."""

from __future__ import annotations

import logging

import pytest
from PIL import Image

from griptape_nodes_library.image.crop_image import CropImage

IMG_SIZE = (200, 100)


def _node() -> CropImage:
    return CropImage(name="CropImage")


def _box(node: CropImage, left: int, top: int, width: int, height: int) -> tuple[int, int, int, int]:
    area = node._calculate_crop_area({"left": left, "top": top, "width": width, "height": height}, IMG_SIZE)
    return area.left, area.top, area.right, area.bottom


@pytest.mark.parametrize(
    ("left", "top", "width", "height", "expected"),
    [
        # 0 width/height means full size
        (0, 0, 0, 0, (0, 0, 200, 100)),
        (10, 20, 0, 0, (10, 20, 200, 100)),
        # A box inside the image is unchanged
        (10, 20, 50, 40, (10, 20, 60, 60)),
        # A negative origin crops the part of the box that is on the image
        (-10, 0, 50, 40, (0, 0, 40, 40)),
        (0, -10, 50, 40, (0, 0, 50, 30)),
        (-10, -10, 0, 0, (0, 0, 200, 100)),
        # A box entirely left of or above the image leaves an empty crop
        (-60, 0, 50, 40, (0, 0, 0, 40)),
        (0, -60, 50, 40, (0, 0, 50, 0)),
        # A box past the right/bottom edge is trimmed to the image
        (180, 90, 50, 40, (180, 90, 200, 100)),
    ],
)
def test_calculate_crop_area(left: int, top: int, width: int, height: int, expected: tuple[int, int, int, int]) -> None:
    assert _box(_node(), left, top, width, height) == expected


def test_adjusted_crop_box_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="griptape_nodes"):
        _box(_node(), -10, 0, 50, 40)

    assert "adjusted to (0, 0, 40, 40)" in caplog.text


def test_full_size_sentinel_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="griptape_nodes"):
        _box(_node(), 0, 0, 0, 0)

    assert "adjusted" not in caplog.text


@pytest.mark.parametrize(("param", "value"), [("left", -10), ("top", -10)])
def test_negative_origin_passes_validation(param: str, value: int) -> None:
    node = _node()
    node.set_parameter_value(param, value)

    assert node._validate_parameters() == []


@pytest.mark.parametrize("param", ["width", "height"])
def test_negative_size_fails_validation(param: str) -> None:
    node = _node()
    node.set_parameter_value(param, -10)

    assert len(node._validate_parameters()) == 1


@pytest.fixture
def node() -> CropImage:
    return CropImage(name="test_crop_image")


def crop(node: CropImage, left: int, top: int, width: int, height: int) -> Image.Image:
    """Drive the node's crop path directly, skipping image loading and file writing."""
    params = {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "zoom": 100.0,
        "rotate": 0.0,
        "background_color": "#00000000",
    }
    img = Image.new("RGB", IMG_SIZE)
    crop_area = node._calculate_crop_area(params, img.size)
    return node._apply_crop_transformations(img, crop_area, params)


@pytest.mark.parametrize(
    ("left", "top", "width", "height", "expected_size"),
    [
        (210, 0, 50, 50, (1, 50)),
        (200, 0, 50, 50, (1, 50)),
        (0, 150, 50, 50, (50, 1)),
        (500, 500, 0, 0, (1, 1)),
    ],
)
def test_origin_past_edge_crops_to_edge_strip(
    node: CropImage, left: int, top: int, width: int, height: int, expected_size: tuple[int, int]
) -> None:
    assert crop(node, left, top, width, height).size == expected_size


def test_box_overhanging_edge_is_trimmed(node: CropImage) -> None:
    assert crop(node, 150, 50, 100, 100).size == (50, 50)


def test_zero_size_uses_full_image(node: CropImage) -> None:
    assert crop(node, 0, 0, 0, 0).size == IMG_SIZE
