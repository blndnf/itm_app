"""Image processing modules for Trilumin Process."""

from processing.outlines import OutlineExtractor
from processing.shades import ShadeQuantizer
from processing.palette import PaletteExtractor
from processing.abstraction import ImageAbstractor, AbstractionMethod, AbstractionSettings

__all__ = [
    "OutlineExtractor",
    "ShadeQuantizer",
    "PaletteExtractor",
    "ImageAbstractor",
    "AbstractionMethod",
    "AbstractionSettings",
]
