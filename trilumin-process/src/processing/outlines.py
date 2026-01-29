"""Outline extraction using Canny Edge Detection."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class OutlineSource(Enum):
    """Source image type for outline extraction."""
    ORIGINAL = "original"
    POSTERIZED = "posterized"


@dataclass
class OutlineSettings:
    """Settings for outline extraction."""

    low_threshold: int = 50
    high_threshold: int = 150
    blur_kernel_size: int = 5
    invert: bool = True  # Black lines on white background
    use_clahe: bool = True  # Apply CLAHE for contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    line_thickness: int = 2  # Dilation iterations for thicker lines
    background_gray: int = 220  # Light gray background (0-255)


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

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        This enhances local contrast, making edges more visible in
        low-contrast regions.

        Args:
            gray: Grayscale input image.

        Returns:
            Contrast-enhanced grayscale image.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.settings.clahe_clip_limit,
            tileGridSize=(self.settings.clahe_grid_size, self.settings.clahe_grid_size),
        )
        return clahe.apply(gray)

    def extract(
        self,
        image: np.ndarray,
        posterized_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract outlines from an image.

        Args:
            image: Input image in BGR format (original).
            posterized_image: Optional posterized image to use as source.
                              If provided, edges will be extracted from this
                              image, which often has clearer color boundaries.

        Returns:
            Outline image (black lines on white background).
        """
        # Use posterized image if provided, otherwise use original
        source = posterized_image if posterized_image is not None else image

        # Convert to grayscale if needed
        if len(source.shape) == 3:
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            gray = source.copy()

        # Apply CLAHE for better contrast (especially in flat areas)
        if self.settings.use_clahe:
            gray = self._apply_clahe(gray)

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

        # Thicken lines if requested
        if self.settings.line_thickness > 1:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=self.settings.line_thickness - 1)

        # Create output with light gray background and black lines
        if self.settings.invert:
            # Create light gray background
            bg_value = self.settings.background_gray
            result = np.full_like(edges, bg_value)
            # Draw black lines where edges are detected
            result[edges > 0] = 0
            return result
        else:
            return edges

    def extract_with_dilation(
        self,
        image: np.ndarray,
        dilation_iterations: int = 1,
        posterized_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract outlines with optional line thickening.

        Args:
            image: Input image in BGR format.
            dilation_iterations: Number of dilation iterations for thicker lines.
            posterized_image: Optional posterized image to use as source.

        Returns:
            Outline image with thickened lines.
        """
        edges = self.extract(image, posterized_image)

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

    def set_use_clahe(self, use_clahe: bool) -> None:
        """Enable or disable CLAHE contrast enhancement."""
        self.settings.use_clahe = use_clahe


def extract_outlines(
    image: np.ndarray,
    sensitivity: float = 0.5,
    line_thickness: int = 0,
    posterized_image: Optional[np.ndarray] = None,
    use_clahe: bool = True,
) -> np.ndarray:
    """
    Convenience function for outline extraction.

    Args:
        image: Input image in BGR format.
        sensitivity: Edge detection sensitivity (0.0-1.0).
        line_thickness: Additional line thickening (0 = no thickening).
        posterized_image: Optional posterized image for clearer edges.
        use_clahe: Whether to apply CLAHE contrast enhancement.

    Returns:
        Outline image (black lines on white background).
    """
    settings = OutlineSettings(use_clahe=use_clahe)
    extractor = OutlineExtractor(settings)
    extractor.set_sensitivity(sensitivity)

    if line_thickness > 0:
        return extractor.extract_with_dilation(image, line_thickness, posterized_image)
    return extractor.extract(image, posterized_image)
