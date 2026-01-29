"""Tests for Trilumin Process image processing modules."""

import sys
import os

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from processing.outlines import OutlineExtractor, OutlineSettings, extract_outlines
from processing.shades import ShadeQuantizer, ShadeSettings, quantize_shades
from processing.palette import PaletteExtractor, PaletteSettings, ColorInfo, extract_palette


# Test fixtures
@pytest.fixture
def sample_color_image():
    """Create a sample color image for testing."""
    # Create a 100x100 BGR image with colored regions
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Red region (BGR)
    image[0:50, 0:50] = [0, 0, 255]
    # Green region
    image[0:50, 50:100] = [0, 255, 0]
    # Blue region
    image[50:100, 0:50] = [255, 0, 0]
    # Yellow region
    image[50:100, 50:100] = [0, 255, 255]
    return image


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    # Create a gradient image
    image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        image[i, :] = int(i * 255 / 99)
    return image


@pytest.fixture
def sample_edge_image():
    """Create an image with clear edges for testing."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # White background
    image[:] = 255
    # Black rectangle
    image[20:80, 20:80] = 0
    return image


class TestOutlineExtractor:
    """Tests for OutlineExtractor class."""

    def test_default_settings(self):
        """Test that default settings are applied."""
        extractor = OutlineExtractor()
        assert extractor.settings.low_threshold == 50
        assert extractor.settings.high_threshold == 150
        assert extractor.settings.invert is True

    def test_custom_settings(self):
        """Test custom settings initialization."""
        settings = OutlineSettings(low_threshold=30, high_threshold=100)
        extractor = OutlineExtractor(settings)
        assert extractor.settings.low_threshold == 30
        assert extractor.settings.high_threshold == 100

    def test_extract_returns_correct_shape(self, sample_color_image):
        """Test that extraction returns correct image shape."""
        extractor = OutlineExtractor()
        result = extractor.extract(sample_color_image)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_extract_from_grayscale(self, sample_grayscale_image):
        """Test extraction from grayscale input."""
        extractor = OutlineExtractor()
        result = extractor.extract(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_invert_option(self, sample_edge_image):
        """Test that invert option produces white background."""
        extractor = OutlineExtractor(OutlineSettings(invert=True))
        result = extractor.extract(sample_edge_image)
        # Most pixels should be white (255) with inverted output
        white_pixels = np.sum(result == 255)
        total_pixels = result.size
        assert white_pixels / total_pixels > 0.5

    def test_set_sensitivity(self):
        """Test sensitivity setting."""
        extractor = OutlineExtractor()
        extractor.set_sensitivity(0.0)
        assert extractor.settings.low_threshold == 20

        extractor.set_sensitivity(1.0)
        assert extractor.settings.low_threshold == 120

    def test_extract_outlines_function(self, sample_color_image):
        """Test convenience function."""
        result = extract_outlines(sample_color_image, sensitivity=0.5)
        assert result.shape == (100, 100)


class TestShadeQuantizer:
    """Tests for ShadeQuantizer class."""

    def test_default_settings(self):
        """Test that default settings are applied."""
        quantizer = ShadeQuantizer()
        assert quantizer.settings.num_values == 5

    def test_min_max_values(self):
        """Test that values are clamped to valid range."""
        settings = ShadeSettings(num_values=1)
        quantizer = ShadeQuantizer(settings)
        assert quantizer.settings.num_values == 3

        settings = ShadeSettings(num_values=20)
        quantizer = ShadeQuantizer(settings)
        assert quantizer.settings.num_values == 12

    def test_quantize_returns_correct_shape(self, sample_color_image):
        """Test that quantization returns correct shape."""
        quantizer = ShadeQuantizer()
        result = quantizer.quantize(sample_color_image)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_quantize_grayscale_input(self, sample_grayscale_image):
        """Test quantization of grayscale input."""
        quantizer = ShadeQuantizer()
        result = quantizer.quantize(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_correct_number_of_values(self, sample_grayscale_image):
        """Test that output has correct number of unique values."""
        quantizer = ShadeQuantizer(ShadeSettings(num_values=5))
        result = quantizer.quantize(sample_grayscale_image)
        unique_values = np.unique(result)
        assert len(unique_values) <= 5

    def test_get_value_levels(self):
        """Test that value levels are correctly computed."""
        quantizer = ShadeQuantizer(ShadeSettings(num_values=5))
        levels = quantizer.get_value_levels()
        assert len(levels) == 5
        assert levels[0] == 0
        assert levels[-1] == 255

    def test_create_value_strip(self):
        """Test value strip creation."""
        quantizer = ShadeQuantizer(ShadeSettings(num_values=5))
        strip = quantizer.create_value_strip(width=300, height=50)
        assert strip.shape == (50, 300)

    def test_quantize_shades_function(self, sample_color_image):
        """Test convenience function."""
        result = quantize_shades(sample_color_image, num_values=4)
        assert result.shape == (100, 100)


class TestPaletteExtractor:
    """Tests for PaletteExtractor class."""

    def test_default_settings(self):
        """Test that default settings are applied."""
        extractor = PaletteExtractor()
        assert extractor.settings.num_colors == 9

    def test_colors_rounded_to_multiple_of_3(self):
        """Test that color count is rounded to multiple of 3."""
        settings = PaletteSettings(num_colors=7)
        extractor = PaletteExtractor(settings)
        assert extractor.settings.num_colors == 6

        settings = PaletteSettings(num_colors=8)
        extractor = PaletteExtractor(settings)
        assert extractor.settings.num_colors == 9

    def test_min_max_colors(self):
        """Test that colors are clamped to valid range."""
        settings = PaletteSettings(num_colors=2)
        extractor = PaletteExtractor(settings)
        assert extractor.settings.num_colors == 6

        settings = PaletteSettings(num_colors=30)
        extractor = PaletteExtractor(settings)
        assert extractor.settings.num_colors == 24

    def test_extract_palette_returns_colors(self, sample_color_image):
        """Test that palette extraction returns correct number of colors."""
        extractor = PaletteExtractor(PaletteSettings(num_colors=6))
        colors = extractor.extract_palette(sample_color_image)
        assert len(colors) == 6
        assert all(isinstance(c, ColorInfo) for c in colors)

    def test_color_info_structure(self, sample_color_image):
        """Test ColorInfo structure."""
        extractor = PaletteExtractor(PaletteSettings(num_colors=6))
        colors = extractor.extract_palette(sample_color_image)

        for color in colors:
            assert len(color.rgb) == 3
            assert all(0 <= v <= 255 for v in color.rgb)
            assert color.hex_code.startswith("#")
            assert len(color.hex_code) == 7
            assert 0 <= color.percentage <= 100

    def test_create_posterized_image(self, sample_color_image):
        """Test posterized image creation."""
        extractor = PaletteExtractor(PaletteSettings(num_colors=6))
        extractor.extract_palette(sample_color_image)
        posterized = extractor.create_posterized_image(sample_color_image)

        assert posterized.shape == sample_color_image.shape
        assert posterized.dtype == np.uint8

    def test_create_palette_image(self, sample_color_image):
        """Test palette image creation."""
        extractor = PaletteExtractor(PaletteSettings(num_colors=6))
        colors = extractor.extract_palette(sample_color_image)
        palette_img = extractor.create_palette_image(colors)

        assert len(palette_img.shape) == 3
        assert palette_img.shape[2] == 3

    def test_extract_palette_function(self, sample_color_image):
        """Test convenience function."""
        colors, posterized, palette_img = extract_palette(sample_color_image, num_colors=6)

        assert len(colors) == 6
        assert posterized.shape == sample_color_image.shape
        assert len(palette_img.shape) == 3


class TestColorInfo:
    """Tests for ColorInfo dataclass."""

    def test_from_rgb(self):
        """Test ColorInfo creation from RGB values."""
        color = ColorInfo.from_rgb(255, 128, 0, 25.5)
        assert color.rgb == (255, 128, 0)
        assert color.hex_code == "#FF8000"
        assert color.percentage == 25.5

    def test_hex_code_format(self):
        """Test hex code formatting."""
        color = ColorInfo.from_rgb(0, 0, 0)
        assert color.hex_code == "#000000"

        color = ColorInfo.from_rgb(255, 255, 255)
        assert color.hex_code == "#FFFFFF"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
