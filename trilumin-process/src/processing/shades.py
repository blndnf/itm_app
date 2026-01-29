"""Grayscale quantization for shade extraction."""

from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np


@dataclass
class ShadeSettings:
    """Settings for shade quantization."""

    num_values: int = 5  # Number of gray levels (3-12)


class ShadeQuantizer:
    """
    Quantizes images to a limited number of grayscale values.

    Creates posterized grayscale images suitable for
    planning oil painting value studies.
    """

    MIN_VALUES = 3
    MAX_VALUES = 12

    def __init__(self, settings: Optional[ShadeSettings] = None):
        """
        Initialize the shade quantizer.

        Args:
            settings: ShadeSettings instance, uses defaults if None.
        """
        self.settings = settings or ShadeSettings()
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Ensure settings are within valid range."""
        self.settings.num_values = max(
            self.MIN_VALUES, min(self.MAX_VALUES, self.settings.num_values)
        )

    def quantize(self, image: np.ndarray) -> np.ndarray:
        """
        Quantize image to limited grayscale values.

        Args:
            image: Input image in BGR, RGB, or grayscale format.

        Returns:
            Posterized grayscale image.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculate quantization levels
        num_values = self.settings.num_values
        step = 256 // num_values

        # Quantize using floor division and multiplication
        quantized = (gray // step) * step

        # Ensure we have distinct values by remapping
        # This creates evenly spaced gray values from 0 to 255
        levels = np.linspace(0, 255, num_values, dtype=np.uint8)
        indices = (gray / 256 * num_values).astype(np.uint8)
        indices = np.clip(indices, 0, num_values - 1)
        quantized = levels[indices]

        return quantized

    def get_value_levels(self) -> List[int]:
        """
        Get the gray levels used for quantization.

        Returns:
            List of gray values (0-255).
        """
        return list(
            np.linspace(0, 255, self.settings.num_values, dtype=int)
        )

    def set_num_values(self, num_values: int) -> None:
        """
        Set the number of grayscale values.

        Args:
            num_values: Number of gray levels (clamped to 3-12).
        """
        self.settings.num_values = max(
            self.MIN_VALUES, min(self.MAX_VALUES, num_values)
        )

    def create_value_strip(
        self, width: int = 300, height: int = 50
    ) -> np.ndarray:
        """
        Create a visual strip showing all quantization levels.

        Args:
            width: Width of the strip in pixels.
            height: Height of the strip in pixels.

        Returns:
            Grayscale image showing all value levels.
        """
        levels = self.get_value_levels()
        num_values = len(levels)
        strip_width = width // num_values

        strip = np.zeros((height, width), dtype=np.uint8)

        for i, level in enumerate(levels):
            x_start = i * strip_width
            x_end = (i + 1) * strip_width if i < num_values - 1 else width
            strip[:, x_start:x_end] = level

        return strip

    def analyze_image(self, image: np.ndarray) -> dict:
        """
        Analyze the value distribution in an image.

        Args:
            image: Input image.

        Returns:
            Dictionary with value distribution statistics.
        """
        quantized = self.quantize(image)
        levels = self.get_value_levels()

        # Calculate histogram
        hist = {}
        total_pixels = quantized.size

        for level in levels:
            count = np.sum(quantized == level)
            hist[int(level)] = {
                "count": int(count),
                "percentage": round(count / total_pixels * 100, 2),
            }

        return {
            "num_values": self.settings.num_values,
            "levels": levels,
            "histogram": hist,
        }


def quantize_shades(image: np.ndarray, num_values: int = 5) -> np.ndarray:
    """
    Convenience function for shade quantization.

    Args:
        image: Input image in BGR, RGB, or grayscale format.
        num_values: Number of grayscale levels (3-12).

    Returns:
        Posterized grayscale image.
    """
    settings = ShadeSettings(num_values=num_values)
    quantizer = ShadeQuantizer(settings)
    return quantizer.quantize(image)
