"""Grayscale quantization for shade extraction."""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np

from utils.color_naming import get_text_color_for_background, int_to_roman


@dataclass
class ShadeSettings:
    """Settings for shade quantization."""

    num_values: int = 5  # Number of gray levels (3-12)
    add_numbers: bool = False  # Add Roman numeral labels
    min_region_size: int = 500  # Minimum region size for labeling


class ShadeQuantizer:
    """
    Quantizes images to a limited number of grayscale values.

    Creates posterized grayscale images suitable for
    planning oil painting value studies.
    """

    MIN_VALUES = 2
    MAX_VALUES = 9  # Max 9 shades for better palette display

    def __init__(self, settings: Optional[ShadeSettings] = None):
        """
        Initialize the shade quantizer.

        Args:
            settings: ShadeSettings instance, uses defaults if None.
        """
        self.settings = settings or ShadeSettings()
        self._validate_settings()
        self._levels: Optional[List[int]] = None

    def _validate_settings(self) -> None:
        """Ensure settings are within valid range."""
        self.settings.num_values = max(
            self.MIN_VALUES, min(self.MAX_VALUES, self.settings.num_values)
        )

    def quantize(
        self,
        image: np.ndarray,
        add_numbers: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Quantize image to limited grayscale values.

        Args:
            image: Input image in BGR, RGB, or grayscale format.
            add_numbers: Override for adding Roman numeral labels.

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
        self._levels = list(np.linspace(0, 255, num_values, dtype=int))

        # Quantize by mapping to nearest level
        indices = (gray / 256 * num_values).astype(np.uint8)
        indices = np.clip(indices, 0, num_values - 1)
        levels_array = np.array(self._levels, dtype=np.uint8)
        quantized = levels_array[indices]

        # Add numbers if requested
        should_add_numbers = add_numbers if add_numbers is not None else self.settings.add_numbers
        if should_add_numbers:
            quantized = self._add_region_numbers(quantized)

        return quantized

    def _add_region_numbers(self, quantized: np.ndarray) -> np.ndarray:
        """Add Roman numeral labels to grayscale regions."""
        # Convert to BGR for colored text
        result = cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)
        height, width = result.shape[:2]

        if self._levels is None:
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # For each gray level, find and label only the LARGEST region
        for i, level in enumerate(self._levels):
            # Create mask for this gray level
            mask = (quantized == level).astype(np.uint8)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            # Find the LARGEST component for this shade (only show ONE number per shade)
            best_label_id = -1
            best_area = 0

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area > best_area:
                    best_area = area
                    best_label_id = label_id

            # Only draw number if largest region meets minimum size
            if best_label_id < 0 or best_area < self.settings.min_region_size:
                continue

            # Get centroid of largest region
            cx, cy = centroids[best_label_id]
            cx, cy = int(cx), int(cy)

            # Determine text color (white on dark, black on light)
            lv = int(level)
            tc = get_text_color_for_background((lv, lv, lv))
            text_color = (int(tc[0]), int(tc[1]), int(tc[2]))

            # Calculate font scale based on image size and region area
            image_diagonal = (width**2 + height**2) ** 0.5
            base_scale = image_diagonal / 1500.0

            image_area = width * height
            area_factor = min(1.3, max(0.7, (best_area / (image_area * 0.01)) ** 0.3))

            font_scale = min(2.5, max(0.4, base_scale * area_factor))
            thickness = max(1, int(font_scale * 2))

            # Draw Roman numeral
            roman = int_to_roman(i + 1)
            (text_width, text_height), baseline = cv2.getTextSize(
                roman, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Center text
            text_x = max(0, min(width - text_width, cx - text_width // 2))
            text_y = max(text_height, min(height - baseline, cy + text_height // 2))

            cv2.putText(
                result,
                roman,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        # Convert back to grayscale
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    def get_value_levels(self) -> List[int]:
        """
        Get the gray levels used for quantization.

        Returns:
            List of gray values (0-255).
        """
        if self._levels is None:
            self._levels = list(
                np.linspace(0, 255, self.settings.num_values, dtype=int)
            )
        return self._levels

    def set_num_values(self, num_values: int) -> None:
        """
        Set the number of grayscale values.

        Args:
            num_values: Number of gray levels (clamped to 3-12).
        """
        self.settings.num_values = max(
            self.MIN_VALUES, min(self.MAX_VALUES, num_values)
        )
        self._levels = None

    def create_value_strip(
        self,
        width: int = 300,
        height: int = 50,
        show_numbers: bool = True,
    ) -> np.ndarray:
        """
        Create a visual strip showing all quantization levels.

        Args:
            width: Width of the strip in pixels.
            height: Height of the strip in pixels.
            show_numbers: Whether to show Roman numerals.

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

        if show_numbers:
            # Convert to BGR for text
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

            for i, level in enumerate(levels):
                x_center = i * strip_width + strip_width // 2

                # Get text color
                text_color = get_text_color_for_background((level, level, level))

                # Draw Roman numeral
                roman = int_to_roman(i + 1)
                (tw, th), _ = cv2.getTextSize(
                    roman, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                text_x = x_center - tw // 2
                text_y = height // 2 + th // 2

                cv2.putText(
                    strip_bgr,
                    roman,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            strip = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)

        return strip

    def analyze_image(self, image: np.ndarray) -> dict:
        """
        Analyze the value distribution in an image.

        Args:
            image: Input image.

        Returns:
            Dictionary with value distribution statistics.
        """
        quantized = self.quantize(image, add_numbers=False)
        levels = self.get_value_levels()

        # Calculate histogram
        hist = {}
        total_pixels = quantized.size

        for i, level in enumerate(levels):
            count = np.sum(quantized == level)
            hist[int(level)] = {
                "index": i + 1,
                "roman": int_to_roman(i + 1),
                "count": int(count),
                "percentage": round(count / total_pixels * 100, 2),
            }

        return {
            "num_values": self.settings.num_values,
            "levels": levels,
            "histogram": hist,
        }


def quantize_shades(
    image: np.ndarray,
    num_values: int = 5,
    add_numbers: bool = False,
) -> np.ndarray:
    """
    Convenience function for shade quantization.

    Args:
        image: Input image in BGR, RGB, or grayscale format.
        num_values: Number of grayscale levels (3-12).
        add_numbers: Whether to add Roman numeral labels.

    Returns:
        Posterized grayscale image.
    """
    settings = ShadeSettings(num_values=num_values, add_numbers=add_numbers)
    quantizer = ShadeQuantizer(settings)
    return quantizer.quantize(image)
