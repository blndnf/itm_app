"""Image processing modules for Trilumin Process."""

from .outlines import OutlineExtractor
from .shades import ShadeQuantizer
from .palette import PaletteExtractor

__all__ = ["OutlineExtractor", "ShadeQuantizer", "PaletteExtractor"]
