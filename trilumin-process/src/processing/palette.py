"""Color palette extraction using K-Means clustering."""

import colorsys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils.color_naming import rgb_to_name, get_text_color_for_background, is_gray


class SortMethod(Enum):
    """Palette sorting methods."""
    PERCENTAGE = "percentage"  # By area percentage (default)
    HUE = "hue"  # By color hue
    LIGHTNESS = "lightness"  # By lightness
    SATURATION = "saturation"  # By saturation


@dataclass
class PaletteSettings:
    """Settings for palette extraction."""

    num_colors: int = 9  # Must be multiple of 3 (6, 9, 12, 15)
    max_iterations: int = 200
    random_state: int = 42
    sort_method: SortMethod = SortMethod.HUE
    grays_position: str = "end"  # "start", "end", or "mixed"
    gray_threshold: float = 12.0  # Saturation threshold for grays


@dataclass
class ColorInfo:
    """Information about a single color in the palette."""

    rgb: Tuple[int, int, int]
    hex_code: str
    percentage: float
    name: str = ""
    index: int = 0  # 1-based index for numbering
    is_gray: bool = False

    @classmethod
    def from_rgb(
        cls,
        r: int,
        g: int,
        b: int,
        percentage: float = 0.0,
        index: int = 0,
    ) -> "ColorInfo":
        """Create ColorInfo from RGB values."""
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        name = rgb_to_name(r, g, b)
        gray = is_gray(r, g, b)
        return cls(
            rgb=(r, g, b),
            hex_code=hex_code,
            percentage=percentage,
            name=name,
            index=index,
            is_gray=gray,
        )


def _get_hsl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to HSL (Hue 0-360, Saturation 0-100, Lightness 0-100)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s * 100, l * 100


def sort_palette(
    colors: List[ColorInfo],
    method: SortMethod = SortMethod.HUE,
    grays_position: str = "end",
    gray_threshold: float = 12.0,
) -> List[ColorInfo]:
    """
    Sort a color palette.

    Args:
        colors: List of ColorInfo objects.
        method: Sorting method.
        grays_position: Where to place grays ("start", "end", "mixed").
        gray_threshold: Saturation threshold for gray detection.

    Returns:
        Sorted list of ColorInfo objects with updated indices.
    """
    if method == SortMethod.PERCENTAGE:
        sorted_colors = sorted(colors, key=lambda c: c.percentage, reverse=True)
    else:
        # Separate grays from chromatic colors
        grays = []
        chromatic = []

        for color in colors:
            h, s, l = _get_hsl(color.rgb)
            if s < gray_threshold:
                grays.append((color, h, s, l))
            else:
                chromatic.append((color, h, s, l))

        # Sort chromatic colors
        if method == SortMethod.HUE:
            chromatic.sort(key=lambda x: x[1])  # Sort by hue
        elif method == SortMethod.LIGHTNESS:
            chromatic.sort(key=lambda x: x[3])  # Sort by lightness
        elif method == SortMethod.SATURATION:
            chromatic.sort(key=lambda x: x[2])  # Sort by saturation

        # Sort grays by lightness
        grays.sort(key=lambda x: x[3])

        # Combine based on grays_position
        if grays_position == "start":
            sorted_colors = [c[0] for c in grays] + [c[0] for c in chromatic]
        elif grays_position == "end":
            sorted_colors = [c[0] for c in chromatic] + [c[0] for c in grays]
        else:  # mixed
            sorted_colors = sorted(colors, key=lambda c: _get_hsl(c.rgb)[1])  # by hue

    # Update indices
    for i, color in enumerate(sorted_colors):
        color.index = i + 1

    return sorted_colors


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
        self._color_to_index: Dict[Tuple[int, int, int], int] = {}

    def _validate_settings(self) -> None:
        """Ensure settings are within valid range."""
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
            List of ColorInfo objects sorted according to settings.
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
            colors.append(ColorInfo.from_rgb(r, g, b, round(pct, 2), index=i + 1))

        # Sort according to settings
        colors = sort_palette(
            colors,
            self.settings.sort_method,
            self.settings.grays_position,
            self.settings.gray_threshold,
        )

        # Build color to index mapping for posterized numbering
        self._color_to_index = {c.rgb: c.index for c in colors}

        return colors

    def create_posterized_image(
        self,
        image: np.ndarray,
        add_numbers: bool = False,
        min_region_size: int = 500,
    ) -> np.ndarray:
        """
        Create a posterized version of the image using extracted palette.

        Args:
            image: Input image in BGR format.
            add_numbers: If True, add region numbers to the image.
            min_region_size: Minimum region size for numbering.

        Returns:
            Posterized image with reduced colors (BGR format).
        """
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

        # Convert back to BGR for display
        posterized_bgr = cv2.cvtColor(posterized, cv2.COLOR_RGB2BGR)

        if add_numbers and self._color_to_index:
            posterized_bgr = self._add_region_numbers(
                posterized_bgr, posterized, min_region_size
            )

        return posterized_bgr

    def _add_region_numbers(
        self,
        image_bgr: np.ndarray,
        image_rgb: np.ndarray,
        min_region_size: int,
    ) -> np.ndarray:
        """Add numbers to each color region in the posterized image."""
        result = image_bgr.copy()
        height, width = result.shape[:2]

        # For each unique color, find connected components
        unique_colors = np.unique(image_rgb.reshape(-1, 3), axis=0)

        for color in unique_colors:
            color_tuple = tuple(color)
            if color_tuple not in self._color_to_index:
                continue

            number = self._color_to_index[color_tuple]

            # Create mask for this color
            mask = np.all(image_rgb == color, axis=2).astype(np.uint8)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            # For each component (skip background label 0)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < min_region_size:
                    continue

                # Get centroid
                cx, cy = centroids[label_id]
                cx, cy = int(cx), int(cy)

                # Determine text color based on background luminance
                # Convert numpy types to Python int for OpenCV compatibility
                color_int = (int(color_tuple[0]), int(color_tuple[1]), int(color_tuple[2]))
                tc = get_text_color_for_background(color_int)
                text_color = (int(tc[0]), int(tc[1]), int(tc[2]))

                # Calculate font scale based on region size
                font_scale = min(1.0, max(0.4, area / 10000))

                # Draw number
                text = str(number)
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )

                # Center text
                text_x = max(0, min(width - text_width, cx - text_width // 2))
                text_y = max(text_height, min(height - baseline, cy + text_height // 2))

                cv2.putText(
                    result,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )

        return result

    def create_palette_image(
        self,
        colors: List[ColorInfo],
        swatch_size: int = 60,
        cols: int = 3,
        show_name: bool = True,
        show_number: bool = True,
        grayscales: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Create a visual palette image with color swatches.

        Args:
            colors: List of ColorInfo objects.
            swatch_size: Size of each color swatch in pixels.
            cols: Number of columns in the palette grid.
            show_name: Whether to show color names.
            show_number: Whether to show numbers on swatches.
            grayscales: Optional list of grayscale values to add below.

        Returns:
            RGB image of the color palette.
        """
        num_colors = len(colors)
        rows = (num_colors + cols - 1) // cols

        # Calculate dimensions - scale proportionally to swatch_size
        # Base reference: swatch_size=60 with text_height=20, margin=2, font_scale=0.6/0.3
        scale_factor = swatch_size / 60.0
        text_height = int(160 * scale_factor) if show_name else 0
        margin = max(2, int(2 * scale_factor))
        cell_height = swatch_size + text_height
        cell_width = swatch_size

        # Add space for grayscale section if provided
        gray_section_height = 0
        separator_gap = int(80 * scale_factor)  # Gap for separator
        if grayscales:
            gray_section_height = swatch_size + text_height + separator_gap

        # Create image with dark gray background
        width = cols * cell_width
        height = rows * cell_height + gray_section_height
        palette_img = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray bg

        # Pre-calculate border thickness for swatches
        border_thickness = max(1, int(scale_factor))

        # Draw color swatches
        for i, color_info in enumerate(colors):
            row = i // cols
            col = i % cols

            x = col * cell_width
            y = row * cell_height

            # Draw color swatch
            r, g, b = color_info.rgb
            # Convert to Python int for OpenCV compatibility
            color_tuple = (int(r), int(g), int(b))
            cv2.rectangle(
                palette_img,
                (x + margin, y + margin),
                (x + swatch_size - margin, y + swatch_size - margin),
                color_tuple,
                -1,
            )
            # Draw border
            cv2.rectangle(
                palette_img,
                (x + margin, y + margin),
                (x + swatch_size - margin, y + swatch_size - margin),
                (100, 100, 100),
                border_thickness,
            )

            # Draw number on swatch
            if show_number:
                tc = get_text_color_for_background((r, g, b))
                text_color = (int(tc[0]), int(tc[1]), int(tc[2]))
                number_text = str(color_info.index)
                num_font_scale = 0.6 * scale_factor
                num_thickness = max(2, int(2 * scale_factor))
                (tw, th), _ = cv2.getTextSize(
                    number_text, cv2.FONT_HERSHEY_SIMPLEX, num_font_scale, num_thickness
                )
                num_x = x + (swatch_size - tw) // 2
                num_y = y + (swatch_size + th) // 2
                cv2.putText(
                    palette_img,
                    number_text,
                    (num_x, num_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    num_font_scale,
                    text_color,
                    num_thickness,
                    cv2.LINE_AA,
                )

            # Draw name below swatch
            if show_name:
                name = color_info.name
                name_font_scale = 2.0 * scale_factor
                name_thickness = max(1, int(2 * scale_factor))
                text_x = x + int(10 * scale_factor)
                text_y = y + swatch_size + int(100 * scale_factor)
                cv2.putText(
                    palette_img,
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    name_font_scale,
                    (220, 220, 220),
                    name_thickness,
                    cv2.LINE_AA,
                )

        # Draw grayscale section if provided
        if grayscales:
            from utils.color_naming import int_to_roman

            separator_y = rows * cell_height + int(40 * scale_factor)
            line_thickness = max(1, int(scale_factor))
            line_margin = int(5 * scale_factor)
            # Draw separator line
            cv2.line(
                palette_img,
                (line_margin, separator_y),
                (width - line_margin, separator_y),
                (100, 100, 100),
                line_thickness,
            )

            gray_y = separator_y + int(40 * scale_factor)
            num_grays = len(grayscales)
            gray_swatch_width = min(swatch_size, (width - 2 * line_margin) // num_grays)

            for i, gray_val in enumerate(grayscales):
                gx = line_margin + i * gray_swatch_width
                gv = int(gray_val)  # Convert to Python int for OpenCV

                # Draw gray swatch
                cv2.rectangle(
                    palette_img,
                    (gx + margin, gray_y + margin),
                    (gx + gray_swatch_width - margin, gray_y + swatch_size - margin),
                    (gv, gv, gv),
                    -1,
                )
                cv2.rectangle(
                    palette_img,
                    (gx + margin, gray_y + margin),
                    (gx + gray_swatch_width - margin, gray_y + swatch_size - margin),
                    (100, 100, 100),
                    border_thickness,
                )

                # Draw Roman numeral
                tc = get_text_color_for_background((gv, gv, gv))
                text_color = (int(tc[0]), int(tc[1]), int(tc[2]))
                roman = int_to_roman(i + 1)
                gray_font_scale = 0.5 * scale_factor
                gray_thickness = max(1, int(scale_factor))
                (tw, th), _ = cv2.getTextSize(
                    roman, cv2.FONT_HERSHEY_SIMPLEX, gray_font_scale, gray_thickness
                )
                rx = gx + (gray_swatch_width - tw) // 2
                ry = gray_y + (swatch_size + th) // 2
                cv2.putText(
                    palette_img,
                    roman,
                    (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    gray_font_scale,
                    text_color,
                    gray_thickness,
                    cv2.LINE_AA,
                )

        return palette_img

    def set_num_colors(self, num_colors: int) -> None:
        """Set the number of colors to extract."""
        num_colors = max(6, min(24, num_colors))
        num_colors = round(num_colors / 3) * 3
        self.settings.num_colors = num_colors
        self._kmeans = None
        self._colors = None
        self._labels = None
        self._color_to_index = {}

    def set_sort_method(self, method: SortMethod) -> None:
        """Set the palette sorting method."""
        self.settings.sort_method = method

    def set_grays_position(self, position: str) -> None:
        """Set where grays appear in sorted palette."""
        if position in ("start", "end", "mixed"):
            self.settings.grays_position = position


def extract_palette(
    image: np.ndarray,
    num_colors: int = 9,
    sort_method: SortMethod = SortMethod.HUE,
    add_numbers: bool = False,
) -> Tuple[List[ColorInfo], np.ndarray, np.ndarray]:
    """
    Convenience function for palette extraction.

    Args:
        image: Input image in BGR format.
        num_colors: Number of colors (multiple of 3, 6-24).
        sort_method: How to sort the palette.
        add_numbers: Whether to add region numbers to posterized image.

    Returns:
        Tuple of (colors list, posterized image, palette image).
    """
    settings = PaletteSettings(num_colors=num_colors, sort_method=sort_method)
    extractor = PaletteExtractor(settings)

    colors = extractor.extract_palette(image)
    posterized = extractor.create_posterized_image(image, add_numbers=add_numbers)
    palette_img = extractor.create_palette_image(colors)

    return colors, posterized, palette_img
