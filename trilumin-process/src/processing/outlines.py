"""Outline extraction using Canny Edge Detection."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class OutlineSettings:
    """Settings for outline extraction."""

    low_threshold: int = 50
    high_threshold: int = 150
    blur_kernel_size: int = 5
    invert: bool = True  # Black lines on white background


class OutlineExtractor:
    """
    Extracts outlines/contours from images using Canny Edge Detection.

    This module creates black line drawings on white backgrounds,
    suitable for oil painting preparation.
    """

    def __init__(self, settings: Optional[OutlineSettings] = None):
        """
        Initialize the outline extractor.

        Args:
            settings: OutlineSettings instance, uses defaults if None.
        """
        self.settings = settings or OutlineSettings()

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract outlines from an image.

        Args:
            image: Input image in BGR or RGB format.

        Returns:
            Outline image (black lines on white background).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        kernel_size = self.settings.blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.settings.low_threshold,
            self.settings.high_threshold,
        )

        # Invert if needed (black lines on white background)
        if self.settings.invert:
            edges = cv2.bitwise_not(edges)

        return edges

    def extract_with_dilation(
        self, image: np.ndarray, dilation_iterations: int = 1
    ) -> np.ndarray:
        """
        Extract outlines with optional line thickening.

        Args:
            image: Input image in BGR or RGB format.
            dilation_iterations: Number of dilation iterations for thicker lines.

        Returns:
            Outline image with thickened lines.
        """
        edges = self.extract(image)

        if dilation_iterations > 0:
            # Need to invert for dilation to work on lines
            if self.settings.invert:
                edges = cv2.bitwise_not(edges)

            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)

            if self.settings.invert:
                edges = cv2.bitwise_not(edges)

        return edges

    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set edge detection sensitivity (0.0 to 1.0).

        Lower values = more edges detected (more sensitive).
        Higher values = fewer edges detected (less sensitive).

        Args:
            sensitivity: Value between 0.0 and 1.0.
        """
        sensitivity = max(0.0, min(1.0, sensitivity))

        # Map sensitivity to threshold values
        # Low sensitivity (1.0) = high thresholds = fewer edges
        # High sensitivity (0.0) = low thresholds = more edges
        self.settings.low_threshold = int(20 + sensitivity * 100)
        self.settings.high_threshold = int(60 + sensitivity * 200)

    def get_sensitivity(self) -> float:
        """
        Get current sensitivity as a normalized value.

        Returns:
            Sensitivity value between 0.0 and 1.0.
        """
        # Reverse the mapping from set_sensitivity
        return (self.settings.low_threshold - 20) / 100


def extract_outlines(
    image: np.ndarray,
    sensitivity: float = 0.5,
    line_thickness: int = 0,
) -> np.ndarray:
    """
    Convenience function for outline extraction.

    Args:
        image: Input image in BGR or RGB format.
        sensitivity: Edge detection sensitivity (0.0-1.0).
        line_thickness: Additional line thickening (0 = no thickening).

    Returns:
        Outline image (black lines on white background).
    """
    extractor = OutlineExtractor()
    extractor.set_sensitivity(sensitivity)

    if line_thickness > 0:
        return extractor.extract_with_dilation(image, line_thickness)
    return extractor.extract(image)
