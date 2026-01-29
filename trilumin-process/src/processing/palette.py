"""Color palette extraction using K-Means clustering."""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class PaletteSettings:
    """Settings for palette extraction."""

    num_colors: int = 9  # Must be multiple of 3 (6, 9, 12, 15)
    max_iterations: int = 200
    random_state: int = 42


@dataclass
class ColorInfo:
    """Information about a single color in the palette."""

    rgb: Tuple[int, int, int]
    hex_code: str
    percentage: float

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, percentage: float = 0.0) -> "ColorInfo":
        """Create ColorInfo from RGB values."""
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        return cls(rgb=(r, g, b), hex_code=hex_code, percentage=percentage)


class PaletteExtractor:
    """
    Extracts color palettes from images using K-Means clustering.

    Creates reduced color palettes and posterized images
    suitable for oil painting color planning.
    """

    VALID_STEPS = [6, 9, 12, 15, 18, 21, 24]  # Multiples of 3

    def __init__(self, settings: Optional[PaletteSettings] = None):
        """
        Initialize the palette extractor.

        Args:
            settings: PaletteSettings instance, uses defaults if None.
        """
        self.settings = settings or PaletteSettings()
        self._validate_settings()
        self._kmeans: Optional[KMeans] = None
        self._colors: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def _validate_settings(self) -> None:
        """Ensure settings are within valid range."""
        # Round to nearest valid step (multiple of 3)
        num_colors = self.settings.num_colors
        num_colors = max(6, min(24, num_colors))
        num_colors = round(num_colors / 3) * 3
        self.settings.num_colors = num_colors

    def extract_palette(self, image: np.ndarray) -> List[ColorInfo]:
        """
        Extract dominant colors from an image.

        Args:
            image: Input image in BGR format.

        Returns:
            List of ColorInfo objects sorted by percentage (descending).
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Reshape to list of pixels
        pixels = rgb_image.reshape(-1, 3).astype(np.float32)

        # Subsample for performance if image is large
        max_samples = 50000
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels

        # Perform K-Means clustering
        self._kmeans = KMeans(
            n_clusters=self.settings.num_colors,
            max_iter=self.settings.max_iterations,
            random_state=self.settings.random_state,
            n_init=10,
        )
        self._kmeans.fit(sample_pixels)

        # Get cluster centers (colors)
        self._colors = self._kmeans.cluster_centers_.astype(np.uint8)

        # Predict labels for all pixels to calculate percentages
        all_labels = self._kmeans.predict(pixels)
        self._labels = all_labels

        # Calculate percentage of each color
        unique, counts = np.unique(all_labels, return_counts=True)
        total_pixels = len(all_labels)
        percentages = {label: count / total_pixels * 100 for label, count in zip(unique, counts)}

        # Create ColorInfo list
        colors = []
        for i, color in enumerate(self._colors):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            pct = percentages.get(i, 0.0)
            colors.append(ColorInfo.from_rgb(r, g, b, round(pct, 2)))

        # Sort by percentage (descending)
        colors.sort(key=lambda c: c.percentage, reverse=True)
        return colors

    def create_posterized_image(self, image: np.ndarray) -> np.ndarray:
        """
        Create a posterized version of the image using extracted palette.

        Args:
            image: Input image in BGR format.

        Returns:
            Posterized image with reduced colors (BGR format).
        """
        # Extract palette if not already done
        if self._colors is None:
            self.extract_palette(image)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape to list of pixels
        pixels = rgb_image.reshape(-1, 3).astype(np.float32)

        # Predict cluster for each pixel
        labels = self._kmeans.predict(pixels)

        # Replace each pixel with its cluster center
        posterized = self._colors[labels]

        # Reshape back to image dimensions
        posterized = posterized.reshape(rgb_image.shape)

        # Convert back to BGR
        posterized = cv2.cvtColor(posterized, cv2.COLOR_RGB2BGR)
        return posterized

    def create_palette_image(
        self,
        colors: List[ColorInfo],
        swatch_size: int = 60,
        cols: int = 3,
        show_hex: bool = True,
    ) -> np.ndarray:
        """
        Create a visual palette image with color swatches.

        Args:
            colors: List of ColorInfo objects.
            swatch_size: Size of each color swatch in pixels.
            cols: Number of columns in the palette grid.
            show_hex: Whether to include hex codes (requires more space).

        Returns:
            RGB image of the color palette.
        """
        num_colors = len(colors)
        rows = (num_colors + cols - 1) // cols

        # Add space for hex codes if needed
        cell_height = swatch_size + (25 if show_hex else 0)
        cell_width = swatch_size

        # Create image
        width = cols * cell_width
        height = rows * cell_height
        palette_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        for i, color_info in enumerate(colors):
            row = i // cols
            col = i % cols

            x = col * cell_width
            y = row * cell_height

            # Draw color swatch
            r, g, b = color_info.rgb
            cv2.rectangle(
                palette_img,
                (x + 2, y + 2),
                (x + swatch_size - 2, y + swatch_size - 2),
                (r, g, b),
                -1,
            )
            # Draw border
            cv2.rectangle(
                palette_img,
                (x + 2, y + 2),
                (x + swatch_size - 2, y + swatch_size - 2),
                (0, 0, 0),
                1,
            )

            # Draw hex code if enabled
            if show_hex:
                text = color_info.hex_code
                text_x = x + 3
                text_y = y + swatch_size + 15
                cv2.putText(
                    palette_img,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 0),
                    1,
                )

        return palette_img

    def set_num_colors(self, num_colors: int) -> None:
        """
        Set the number of colors to extract.

        Args:
            num_colors: Number of colors (will be rounded to multiple of 3).
        """
        num_colors = max(6, min(24, num_colors))
        num_colors = round(num_colors / 3) * 3
        self.settings.num_colors = num_colors
        # Reset cached results
        self._kmeans = None
        self._colors = None
        self._labels = None


def extract_palette(
    image: np.ndarray, num_colors: int = 9
) -> Tuple[List[ColorInfo], np.ndarray, np.ndarray]:
    """
    Convenience function for palette extraction.

    Args:
        image: Input image in BGR format.
        num_colors: Number of colors (multiple of 3, 6-24).

    Returns:
        Tuple of (colors list, posterized image, palette image).
    """
    settings = PaletteSettings(num_colors=num_colors)
    extractor = PaletteExtractor(settings)

    colors = extractor.extract_palette(image)
    posterized = extractor.create_posterized_image(image)
    palette_img = extractor.create_palette_image(colors)

    return colors, posterized, palette_img
