"""Image abstraction/simplification for pre-processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


class AbstractionMethod(Enum):
    """Available abstraction methods."""

    PIXELATE = "pixelate"
    BILATERAL = "bilateral"
    MEAN_SHIFT = "mean_shift"
    KMEANS = "kmeans"


@dataclass
class AbstractionSettings:
    """Settings for image abstraction."""

    enabled: bool = False
    method: AbstractionMethod = AbstractionMethod.BILATERAL
    detail_level: int = 5  # 1-10, where 1 = heavily abstracted, 10 = near original


class ImageAbstractor:
    """
    Applies abstraction/simplification to images before analysis.

    Reduces details and creates more defined color regions,
    useful for preparing images for oil painting studies.
    """

    def __init__(self, settings: Optional[AbstractionSettings] = None):
        """
        Initialize the abstractor.

        Args:
            settings: AbstractionSettings instance, uses defaults if None.
        """
        self.settings = settings or AbstractionSettings()

    def abstract(self, image: np.ndarray) -> np.ndarray:
        """
        Apply abstraction to an image.

        Args:
            image: Input image in BGR format.

        Returns:
            Abstracted image in BGR format.
        """
        if not self.settings.enabled:
            return image.copy()

        method = self.settings.method
        detail = self.settings.detail_level

        if method == AbstractionMethod.PIXELATE:
            return self._pixelate(image, detail)
        elif method == AbstractionMethod.BILATERAL:
            return self._bilateral_filter(image, detail)
        elif method == AbstractionMethod.MEAN_SHIFT:
            return self._mean_shift(image, detail)
        elif method == AbstractionMethod.KMEANS:
            return self._kmeans_quantize(image, detail)
        else:
            return image.copy()

    def _pixelate(self, image: np.ndarray, detail: int) -> np.ndarray:
        """
        Pixelate image by downscaling and upscaling.

        Args:
            image: Input image.
            detail: Detail level (1-10).

        Returns:
            Pixelated image.
        """
        height, width = image.shape[:2]

        # Calculate block size based on detail level
        # detail 1 = large blocks (more abstraction)
        # detail 10 = small blocks (less abstraction)
        max_block = 32
        min_block = 2
        block_size = int(max_block - (detail - 1) * (max_block - min_block) / 9)
        block_size = max(min_block, block_size)

        # Downscale
        small_width = max(1, width // block_size)
        small_height = max(1, height // block_size)
        small = cv2.resize(
            image, (small_width, small_height), interpolation=cv2.INTER_AREA
        )

        # Upscale back to original size
        pixelated = cv2.resize(
            small, (width, height), interpolation=cv2.INTER_NEAREST
        )

        return pixelated

    def _bilateral_filter(self, image: np.ndarray, detail: int) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.

        Args:
            image: Input image.
            detail: Detail level (1-10).

        Returns:
            Filtered image with painterly appearance.
        """
        # Calculate filter parameters based on detail level
        # detail 1 = strong smoothing
        # detail 10 = minimal smoothing
        d = int(15 - (detail - 1) * 1.2)  # diameter: 15 to ~4
        d = max(3, d)

        sigma_color = int(150 - (detail - 1) * 12)  # 150 to ~42
        sigma_color = max(20, sigma_color)

        sigma_space = int(150 - (detail - 1) * 12)  # 150 to ~42
        sigma_space = max(20, sigma_space)

        # Apply multiple iterations for stronger effect at low detail
        iterations = max(1, 4 - detail // 3)
        result = image.copy()

        for _ in range(iterations):
            result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)

        return result

    def _mean_shift(self, image: np.ndarray, detail: int) -> np.ndarray:
        """
        Apply mean shift segmentation for flat color regions.

        Args:
            image: Input image.
            detail: Detail level (1-10).

        Returns:
            Segmented image with distinct color regions.
        """
        # Calculate parameters based on detail level
        # detail 1 = large regions
        # detail 10 = small regions (more detail preserved)
        sp = int(30 - (detail - 1) * 2.5)  # spatial window: 30 to ~7
        sp = max(5, sp)

        sr = int(60 - (detail - 1) * 5)  # color window: 60 to ~15
        sr = max(10, sr)

        # pyrMeanShiftFiltering works best on smaller images for performance
        height, width = image.shape[:2]
        max_dim = 800

        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            small = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            filtered = cv2.pyrMeanShiftFiltering(small, sp, sr)
            result = cv2.resize(
                filtered, (width, height), interpolation=cv2.INTER_LINEAR
            )
        else:
            result = cv2.pyrMeanShiftFiltering(image, sp, sr)

        return result

    def _kmeans_quantize(self, image: np.ndarray, detail: int) -> np.ndarray:
        """
        Quantize colors using K-Means clustering.

        Args:
            image: Input image.
            detail: Detail level (1-10).

        Returns:
            Color-quantized image.
        """
        # Calculate number of colors based on detail level
        # detail 1 = few colors (more abstraction)
        # detail 10 = many colors (less abstraction)
        min_colors = 4
        max_colors = 32
        n_colors = int(min_colors + (detail - 1) * (max_colors - min_colors) / 9)
        n_colors = max(min_colors, n_colors)

        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Subsample for performance
        max_samples = 30000
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels

        # Perform K-Means
        kmeans = KMeans(n_clusters=n_colors, n_init=10, max_iter=100, random_state=42)
        kmeans.fit(sample)

        # Assign each pixel to nearest cluster center
        labels = kmeans.predict(pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)

        # Reconstruct image
        quantized = centers[labels].reshape(image.shape)

        return quantized

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable abstraction."""
        self.settings.enabled = enabled

    def set_method(self, method: AbstractionMethod) -> None:
        """Set the abstraction method."""
        self.settings.method = method

    def set_detail_level(self, level: int) -> None:
        """Set the detail level (1-10)."""
        self.settings.detail_level = max(1, min(10, level))

    def get_method_name(self, method: AbstractionMethod) -> str:
        """Get human-readable name for a method."""
        names = {
            AbstractionMethod.PIXELATE: "Pixelierung",
            AbstractionMethod.BILATERAL: "Bilateral Filter",
            AbstractionMethod.MEAN_SHIFT: "Mean Shift",
            AbstractionMethod.KMEANS: "K-Means Farben",
        }
        return names.get(method, str(method.value))


def abstract_image(
    image: np.ndarray,
    method: AbstractionMethod = AbstractionMethod.BILATERAL,
    detail_level: int = 5,
) -> np.ndarray:
    """
    Convenience function for image abstraction.

    Args:
        image: Input image in BGR format.
        method: Abstraction method to use.
        detail_level: Detail level (1-10).

    Returns:
        Abstracted image.
    """
    settings = AbstractionSettings(enabled=True, method=method, detail_level=detail_level)
    abstractor = ImageAbstractor(settings)
    return abstractor.abstract(image)
