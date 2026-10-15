"""Tests for CropImage's crop-box clamping."""

import pytest
from PIL import Image

from griptape_nodes_library.image.crop_image import CropImage

IMG_SIZE = (200, 100)


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


def test_negative_origin_clamps_to_zero(node: CropImage) -> None:
    assert crop(node, -10, -10, 50, 50).size == (50, 50)


def test_zero_size_uses_full_image(node: CropImage) -> None:
    assert crop(node, 0, 0, 0, 0).size == IMG_SIZE
