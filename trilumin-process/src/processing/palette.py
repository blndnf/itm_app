"""Color palette extraction using K-Means clustering."""

import colorsys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont

from utils.color_naming import rgb_to_name, get_text_color_for_background, is_gray


class SortMethod(Enum):
    """Palette sorting methods."""
    PERCENTAGE = "percentage"  # By area percentage (default)
    HUE = "hue"  # By color hue
    LIGHTNESS = "lightness"  # By lightness
    SATURATION = "saturation"  # By saturation


class PaletteMethod(Enum):
    """Palette extraction methods."""
    STANDARD = "standard"  # K-Means by frequency
    DIVERSE = "diverse"  # Maximize color diversity/contrast
    SATURATED = "saturated"  # Prioritize most saturated colors
    INTENSIFY = "intensify"  # Maximize hue diversity, guarantee vibrant colors from each hue


@dataclass
class PaletteSettings:
    """Settings for palette extraction."""

    num_colors: int = 9  # Must be multiple of 3 (3, 6, 9, 12, 15)
    max_iterations: int = 200
    random_state: int = 42
    sort_method: SortMethod = SortMethod.HUE
    grays_position: str = "end"  # "start", "end", or "mixed"
    gray_threshold: float = 12.0  # Saturation threshold for grays
    palette_method: PaletteMethod = PaletteMethod.INTENSIFY  # Extraction method


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


def _is_pure_black_white_gray(r: int, g: int, b: int, tolerance: int = 10) -> bool:
    """
    Check if a color is pure black, pure white, or pure gray.
    These should be excluded as they are covered by the shades layer.

    Args:
        r, g, b: RGB values.
        tolerance: Maximum deviation allowed for "pure" colors.

    Returns:
        True if color is pure black, white, or gray.
    """
    # Check if all channels are very close (gray/black/white)
    max_diff = max(abs(r - g), abs(g - b), abs(r - b))
    if max_diff > tolerance:
        return False  # Has color, not a pure gray

    # It's a grayscale color - covered by shades
    return True


def _is_low_saturation(r: int, g: int, b: int, threshold: float = 20.0) -> bool:
    """
    Check if a color has low saturation (grayish).

    Args:
        r, g, b: RGB values.
        threshold: Minimum saturation percentage to be considered chromatic.

    Returns:
        True if saturation is below threshold.
    """
    _, s, _ = _get_hsl((r, g, b))
    return s < threshold


def _is_gray_color_name(name: str) -> bool:
    """Check if a color name indicates a gray/black/white color."""
    gray_terms = [
        "schwarz", "grau", "weiß", "weiss", "anthrazit",
        "black", "gray", "grey", "white", "charcoal",
        "silver", "silber", "silbergrau", "ash", "slate", "graphit",
        "dunkelgrau", "hellgrau", "mittelgrau", "warmgrau", "kaltgrau",
        "schiefergrau", "eisgrau", "stahlgrau", "mausgrau", "aschgrau",
    ]
    name_lower = name.lower()
    return any(term in name_lower for term in gray_terms)


def _get_hue_category(rgb: Tuple[int, int, int]) -> str:
    """Get the hue category name for a color."""
    h, s, l = _get_hsl(rgb)

    # Low saturation = gray
    if s < 15:
        return "gray"

    # Categorize by hue
    if h < 15 or h >= 345:
        return "rot"
    elif h < 45:
        return "orange"
    elif h < 75:
        return "gelb"
    elif h < 150:
        return "grün"
    elif h < 210:
        return "cyan"
    elif h < 270:
        return "blau"
    elif h < 310:
        return "violett"
    else:
        return "magenta"


def _check_hue_diversity(colors: List[ColorInfo]) -> bool:
    """
    Check if colors have good hue diversity.

    Returns False if >50% of colors share the same hue category.
    """
    if len(colors) < 3:
        return True  # Too few colors to judge

    hue_counts = {}
    for c in colors:
        cat = _get_hue_category(c.rgb)
        hue_counts[cat] = hue_counts.get(cat, 0) + 1

    max_count = max(hue_counts.values())
    return max_count <= len(colors) * 0.5  # No category should have >50%


def _find_most_vibrant_by_hue(
    colors: List[ColorInfo],
    num_hues: int = 6
) -> List[ColorInfo]:
    """
    Find the most vibrant color from each major hue category.

    This ensures diverse hue representation in the final palette.
    """
    hue_categories = {}

    for c in colors:
        cat = _get_hue_category(c.rgb)
        if cat == "gray":
            continue  # Skip grays

        sat = _get_color_saturation(*c.rgb)
        if cat not in hue_categories or sat > _get_color_saturation(*hue_categories[cat].rgb):
            hue_categories[cat] = c

    # Sort by saturation and return most vibrant from each hue
    result = list(hue_categories.values())
    result.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
    return result[:num_hues]


def _get_color_saturation(r: int, g: int, b: int) -> float:
    """Get saturation value (0-100) for an RGB color."""
    _, s, _ = _get_hsl((r, g, b))
    return s


def _color_distance_hsl(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptual color distance in HSL space.
    Weights hue differences more heavily for saturated colors.
    """
    h1, s1, l1 = _get_hsl(c1)
    h2, s2, l2 = _get_hsl(c2)

    # Hue distance (circular, 0-180 max)
    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
    hue_diff = hue_diff / 180.0  # Normalize to 0-1

    # Weight hue by average saturation (more important for saturated colors)
    avg_sat = (s1 + s2) / 200.0  # 0-1
    hue_weight = 2.0 * avg_sat

    # Saturation and lightness differences
    sat_diff = abs(s1 - s2) / 100.0
    light_diff = abs(l1 - l2) / 100.0

    # Combined distance
    return (hue_weight * hue_diff) + (0.5 * sat_diff) + (0.3 * light_diff)


def _select_diverse_colors(
    colors: List[ColorInfo],
    target_count: int,
    min_saturation: float = 25.0,
) -> List[ColorInfo]:
    """
    Select a diverse subset of colors maximizing contrast.

    Uses a greedy algorithm to pick colors that are maximally
    distant from already selected colors in HSL space.
    Strongly prefers saturated colors to preserve vibrancy.

    Args:
        colors: List of candidate colors.
        target_count: Number of colors to select.
        min_saturation: Minimum saturation for chromatic colors.

    Returns:
        List of selected diverse colors.
    """
    if len(colors) <= target_count:
        return colors

    # Separate into chromatic and achromatic, with stricter filtering
    chromatic = []
    low_saturation = []
    achromatic = []

    for c in colors:
        h, s, l = _get_hsl(c.rgb)
        # Filter by saturation AND by name - NO grays in the palette!
        if _is_gray_color_name(c.name):
            achromatic.append(c)
        elif s >= min_saturation:
            chromatic.append(c)
        elif s >= 15.0:  # Slightly desaturated but still colorful
            low_saturation.append(c)
        else:
            achromatic.append(c)

    if not chromatic:
        # No highly saturated colors, use low saturation ones
        chromatic = low_saturation
        low_saturation = []

    if not chromatic:
        # Fallback: use all colors sorted by saturation
        all_sorted = sorted(colors, key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
        return all_sorted[:target_count]

    # Greedy selection: start with most saturated color for maximum vibrancy
    chromatic.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
    selected = [chromatic[0]]
    remaining = chromatic[1:]

    while len(selected) < target_count and remaining:
        # Find color most distant from all selected colors
        best_color = None
        best_score = -1

        for candidate in remaining:
            # Minimum distance to any selected color
            min_dist = min(
                _color_distance_hsl(candidate.rgb, sel.rgb)
                for sel in selected
            )
            # Bonus for saturation to prefer vibrant colors
            sat_bonus = _get_color_saturation(*candidate.rgb) / 200.0
            score = min_dist + sat_bonus

            if score > best_score:
                best_score = score
                best_color = candidate

        if best_color:
            selected.append(best_color)
            remaining.remove(best_color)
        else:
            break

    # If still not enough, add low saturation colors
    while len(selected) < target_count and low_saturation:
        low_saturation.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
        selected.append(low_saturation.pop(0))

    # Last resort: add achromatic (but this should rarely happen)
    while len(selected) < target_count and achromatic:
        selected.append(achromatic.pop(0))

    return selected


class PaletteExtractor:
    """
    Extracts color palettes from images using K-Means clustering.

    Creates reduced color palettes and posterized images
    suitable for oil painting color planning.
    """

    VALID_STEPS = list(range(2, 25))  # 2 to 24, any increment

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
        self._palette_colors: Optional[np.ndarray] = None  # Selected palette RGB values

    def _validate_settings(self) -> None:
        """Ensure settings are within valid range."""
        num_colors = self.settings.num_colors
        num_colors = max(2, min(24, num_colors))  # Minimum 2 colors, max 24
        self.settings.num_colors = num_colors

    def extract_palette(self, image: np.ndarray) -> List[ColorInfo]:
        """
        Extract dominant colors from an image.

        Uses the configured palette_method to determine extraction strategy:
        - STANDARD: K-Means by frequency (original behavior)
        - DIVERSE: Maximize color diversity/contrast (recommended)
        - SATURATED: Prioritize most saturated colors

        Filters out pure black/white/gray and colors with gray names.

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

        # For diverse/saturated methods, extract many more clusters to find variety
        if self.settings.palette_method in (PaletteMethod.DIVERSE, PaletteMethod.SATURATED):
            # Extract 3-4x more colors to have good candidates for diversity selection
            total_clusters = min(48, max(24, self.settings.num_colors * 4))
        else:
            # Standard method: just a few extra
            extra_clusters = min(6, self.settings.num_colors)
            total_clusters = self.settings.num_colors + extra_clusters

        # Perform K-Means clustering
        self._kmeans = KMeans(
            n_clusters=total_clusters,
            max_iter=self.settings.max_iterations,
            random_state=self.settings.random_state,
            n_init=10,
        )
        self._kmeans.fit(sample_pixels)

        # Get cluster centers (colors)
        all_cluster_colors = self._kmeans.cluster_centers_.astype(np.uint8)

        # Predict labels for all pixels to calculate percentages
        all_labels = self._kmeans.predict(pixels)
        self._labels = all_labels

        # Calculate percentage of each color
        unique, counts = np.unique(all_labels, return_counts=True)
        total_pixels = len(all_labels)
        percentages = {label: count / total_pixels * 100 for label, count in zip(unique, counts)}

        # Create ColorInfo list, filtering out pure black/white/gray
        # These are covered by the shades layer and shouldn't be in color palette
        all_colors = []
        chromatic_colors = []  # Colors with good saturation

        for i, color in enumerate(all_cluster_colors):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            pct = percentages.get(i, 0.0)

            # Filter out pure black, white, and grays (covered by shades)
            if _is_pure_black_white_gray(r, g, b):
                continue

            color_info = ColorInfo.from_rgb(r, g, b, round(pct, 2), index=i + 1)
            all_colors.append(color_info)

            # Track chromatic colors (good saturation, not gray-named)
            if not _is_low_saturation(r, g, b, threshold=25.0) and not _is_gray_color_name(color_info.name):
                chromatic_colors.append(color_info)

        # Select colors based on method
        if self.settings.palette_method == PaletteMethod.INTENSIFY:
            # INTENSIFY: Guarantee vibrant colors from each major hue category
            # Step 1: Find most vibrant color from each hue
            hue_anchors = _find_most_vibrant_by_hue(chromatic_colors)

            # Step 2: Start with hue anchors (up to num_colors)
            colors = hue_anchors[: self.settings.num_colors]

            # Step 3: Fill remaining slots with diverse colors not too close to anchors
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in chromatic_colors if c not in colors]
                remaining.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)

                for candidate in remaining:
                    if len(colors) >= self.settings.num_colors:
                        break
                    # Check if this color's hue is already well-represented
                    candidate_hue = _get_hue_category(candidate.rgb)
                    hue_count = sum(1 for c in colors if _get_hue_category(c.rgb) == candidate_hue)
                    # Allow max 2 colors per hue category
                    if hue_count < 2:
                        colors.append(candidate)

            # Step 4: If still not enough, add any remaining chromatic
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in chromatic_colors if c not in colors]
                colors.extend(remaining[: self.settings.num_colors - len(colors)])

        elif self.settings.palette_method == PaletteMethod.DIVERSE:
            # Maximize color diversity - pick most distinct colors
            colors = _select_diverse_colors(
                all_colors,
                self.settings.num_colors,
                min_saturation=25.0,
            )

            # Check hue diversity - if >50% same hue, recalculate with INTENSIFY logic
            if not _check_hue_diversity(colors):
                # Recalculate: use hue anchors first
                hue_anchors = _find_most_vibrant_by_hue(chromatic_colors)
                colors = hue_anchors[: self.settings.num_colors]

                # Fill with diverse selection
                if len(colors) < self.settings.num_colors:
                    remaining = [c for c in chromatic_colors if c not in colors]
                    for candidate in remaining:
                        if len(colors) >= self.settings.num_colors:
                            break
                        # Add if different hue from existing
                        candidate_hue = _get_hue_category(candidate.rgb)
                        existing_hues = [_get_hue_category(c.rgb) for c in colors]
                        if candidate_hue not in existing_hues or existing_hues.count(candidate_hue) < 2:
                            colors.append(candidate)

        elif self.settings.palette_method == PaletteMethod.SATURATED:
            # Prioritize MOST saturated colors - preserve original vibrancy!
            saturated = [c for c in chromatic_colors
                        if _get_color_saturation(*c.rgb) >= 30.0]
            saturated.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
            colors = saturated[: self.settings.num_colors]

            # Fill with remaining chromatic if not enough
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in chromatic_colors if c not in colors]
                remaining.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
                colors.extend(remaining[: self.settings.num_colors - len(colors)])

            # Last resort: fill with any remaining colors
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in all_colors if c not in colors]
                remaining.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
                colors.extend(remaining[: self.settings.num_colors - len(colors)])
        else:
            # STANDARD: Original behavior - by percentage, but filter grays
            standard_colors = [c for c in all_colors
                             if not _is_gray_color_name(c.name)
                             and not _is_low_saturation(*c.rgb, threshold=20.0)]
            standard_colors.sort(key=lambda c: c.percentage, reverse=True)
            colors = standard_colors[: self.settings.num_colors]

            # Fill with remaining if not enough
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in all_colors if c not in colors]
                remaining.sort(key=lambda c: c.percentage, reverse=True)
                colors.extend(remaining[: self.settings.num_colors - len(colors)])

        # Store colors for posterization
        self._colors = all_cluster_colors

        # Sort according to settings
        colors = sort_palette(
            colors,
            self.settings.sort_method,
            self.settings.grays_position,
            self.settings.gray_threshold,
        )

        # Build color to index mapping for posterized numbering
        self._color_to_index = {c.rgb: c.index for c in colors}

        # Store selected palette colors as numpy array for posterization
        self._palette_colors = np.array([c.rgb for c in colors], dtype=np.uint8)

        return colors

    def create_posterized_image(
        self,
        image: np.ndarray,
        add_numbers: bool = False,
        min_region_size: int = 500,
    ) -> np.ndarray:
        """
        Create a posterized version of the image using extracted palette.

        Maps each pixel to the nearest color from the selected palette,
        ensuring only palette colors appear in the posterized image.

        Args:
            image: Input image in BGR format.
            add_numbers: If True, add region numbers to the image.
            min_region_size: Minimum region size for numbering.

        Returns:
            Posterized image with reduced colors (BGR format).
        """
        if self._palette_colors is None:
            self.extract_palette(image)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape to list of pixels
        pixels = rgb_image.reshape(-1, 3).astype(np.float32)

        # Map each pixel to nearest palette color (not all K-Means clusters)
        # Compute distance from each pixel to each palette color
        palette_float = self._palette_colors.astype(np.float32)

        # Process in chunks to avoid memory issues with large images
        chunk_size = 100000
        num_pixels = len(pixels)
        posterized_pixels = np.empty((num_pixels, 3), dtype=np.uint8)

        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            chunk = pixels[start:end]

            # Calculate squared Euclidean distance to each palette color
            # Shape: (chunk_size, num_palette_colors)
            distances = np.sum(
                (chunk[:, np.newaxis, :] - palette_float[np.newaxis, :, :]) ** 2,
                axis=2
            )

            # Find nearest palette color for each pixel
            nearest_indices = np.argmin(distances, axis=1)
            posterized_pixels[start:end] = self._palette_colors[nearest_indices]

        # Reshape back to image dimensions
        posterized = posterized_pixels.reshape(rgb_image.shape)

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

            # Find the LARGEST component for this color (only show ONE number per color)
            best_label_id = -1
            best_area = 0

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area > best_area:
                    best_area = area
                    best_label_id = label_id

            # Only draw number if largest region meets minimum size
            if best_label_id < 0 or best_area < min_region_size:
                continue

            # Get centroid of largest region
            cx, cy = centroids[best_label_id]
            cx, cy = int(cx), int(cy)

            # Determine text color based on background luminance
            color_int = (int(color_tuple[0]), int(color_tuple[1]), int(color_tuple[2]))
            tc = get_text_color_for_background(color_int)
            text_color = (int(tc[0]), int(tc[1]), int(tc[2]))

            # Calculate font scale based on image size and region area
            image_diagonal = (width**2 + height**2) ** 0.5
            base_scale = image_diagonal / 1500.0

            image_area = width * height
            area_factor = min(1.3, max(0.7, (best_area / (image_area * 0.01)) ** 0.3))

            font_scale = min(2.5, max(0.4, base_scale * area_factor))
            thickness = max(1, int(font_scale * 2))

            # Draw number
            text = str(number)
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
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
                thickness,
                cv2.LINE_AA,
            )

        return result

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a Unicode-supporting font at the given size."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def create_palette_image(
        self,
        colors: List[ColorInfo],
        swatch_size: int = 60,
        cols: int = 0,  # Ignored - now uses fixed A4 landscape grid
        show_name: bool = True,
        show_number: bool = True,
        grayscales: Optional[List[int]] = None,
        for_export: bool = False,
    ) -> np.ndarray:
        """
        Create a visual palette image with color swatches.

        Layout: A4 landscape format
        - Upper portion: 6 columns x 4 rows = 24 slots for colors
        - Lower portion: 9 square tiles for shades (limited to 9 max)
        - Numbers AND names are written INSIDE each tile
        - Unused slots are simply not drawn

        Uses PIL for text rendering to support Unicode (Umlauts).

        Args:
            colors: List of ColorInfo objects (max 24).
            swatch_size: Base size for calculations (scales the entire image).
            cols: Ignored - uses fixed 6 columns for colors.
            show_name: Whether to show color names.
            show_number: Whether to show numbers on swatches.
            grayscales: Optional list of grayscale values (max 9).
            for_export: If True, use white background for printing.

        Returns:
            RGB image of the color palette.
        """
        # Fixed A4 landscape grid layout
        COLOR_COLS = 6
        COLOR_ROWS = 4
        MAX_SHADES = 9  # Max 9 shades, square tiles

        # Calculate dimensions
        color_tile_size = swatch_size
        # Shade tiles are square, sized to fit 9 across the width
        shade_tile_size = (COLOR_COLS * color_tile_size) // MAX_SHADES

        # Image dimensions
        img_width = COLOR_COLS * color_tile_size
        img_height = COLOR_ROWS * color_tile_size + shade_tile_size

        # Background color: gray for display, white for export
        bg_color = (255, 255, 255) if for_export else (50, 50, 50)
        palette_img = np.ones((img_height, img_width, 3), dtype=np.uint8)
        palette_img[:, :] = bg_color

        margin = max(2, swatch_size // 30)
        border_thickness = max(1, swatch_size // 60)

        # Draw color tiles (6x4 grid, upper portion)
        for i, color_info in enumerate(colors[:24]):  # Max 24 colors
            row = i // COLOR_COLS
            col = i % COLOR_COLS
            x = col * color_tile_size
            y = row * color_tile_size

            r, g, b = color_info.rgb
            color_tuple = (int(r), int(g), int(b))

            # Fill tile
            cv2.rectangle(
                palette_img,
                (x + margin, y + margin),
                (x + color_tile_size - margin, y + color_tile_size - margin),
                color_tuple,
                -1,
            )
            # Border
            cv2.rectangle(
                palette_img,
                (x + margin, y + margin),
                (x + color_tile_size - margin, y + color_tile_size - margin),
                (100, 100, 100),
                border_thickness,
            )

        # Draw shade tiles (max 9, square tiles, bottom portion)
        if grayscales:
            shade_y = COLOR_ROWS * color_tile_size
            # Limit to 9 shades max
            shades_to_draw = grayscales[:MAX_SHADES]

            for i, gray_val in enumerate(shades_to_draw):
                gx = i * shade_tile_size
                gv = int(gray_val)

                # Fill square tile
                cv2.rectangle(
                    palette_img,
                    (gx + margin, shade_y + margin),
                    (gx + shade_tile_size - margin, shade_y + shade_tile_size - margin),
                    (gv, gv, gv),
                    -1,
                )
                # Border
                cv2.rectangle(
                    palette_img,
                    (gx + margin, shade_y + margin),
                    (gx + shade_tile_size - margin, shade_y + shade_tile_size - margin),
                    (100, 100, 100),
                    border_thickness,
                )

        # Convert to PIL for text rendering
        pil_image = Image.fromarray(palette_img)
        draw = ImageDraw.Draw(pil_image)

        # Font sizes relative to tile size
        number_font_size = max(16, color_tile_size // 3)
        name_font_size = max(10, color_tile_size // 6)
        shade_font_size = max(12, shade_tile_size // 3)

        number_font = self._get_font(number_font_size)
        name_font = self._get_font(name_font_size)
        shade_font = self._get_font(shade_font_size)

        # Draw text on color tiles (number AND name inside tile)
        for i, color_info in enumerate(colors[:24]):
            row = i // COLOR_COLS
            col = i % COLOR_COLS
            x = col * color_tile_size
            y = row * color_tile_size

            r, g, b = color_info.rgb
            tc = get_text_color_for_background((r, g, b))
            text_color = (int(tc[0]), int(tc[1]), int(tc[2]))

            tile_inner_width = color_tile_size - 2 * margin
            tile_inner_height = color_tile_size - 2 * margin

            # Draw number in upper portion of tile
            if show_number:
                number_text = str(color_info.index)
                bbox = draw.textbbox((0, 0), number_text, font=number_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                num_x = x + margin + (tile_inner_width - tw) // 2
                num_y = y + margin + tile_inner_height // 4 - th // 2

                draw.text((num_x, num_y), number_text, fill=text_color, font=number_font)

            # Draw name in lower portion of tile
            if show_name:
                name = color_info.name
                max_text_width = tile_inner_width - 10

                # Shrink font if name too long
                current_font = name_font
                current_size = name_font_size
                bbox = draw.textbbox((0, 0), name, font=current_font)
                tw = bbox[2] - bbox[0]

                while tw > max_text_width and current_size > 8:
                    current_size = int(current_size * 0.85)
                    current_font = self._get_font(current_size)
                    bbox = draw.textbbox((0, 0), name, font=current_font)
                    tw = bbox[2] - bbox[0]

                th = bbox[3] - bbox[1]
                name_x = x + margin + (tile_inner_width - tw) // 2
                name_y = y + margin + (tile_inner_height * 2) // 3

                draw.text((name_x, name_y), name, fill=text_color, font=current_font)

        # Draw Roman numerals on shade tiles (square tiles, max 9)
        if grayscales:
            from utils.color_naming import int_to_roman

            shade_y = COLOR_ROWS * color_tile_size
            shades_to_draw = grayscales[:MAX_SHADES]

            for i, gray_val in enumerate(shades_to_draw):
                gx = i * shade_tile_size
                gv = int(gray_val)

                tc = get_text_color_for_background((gv, gv, gv))
                text_color = (int(tc[0]), int(tc[1]), int(tc[2]))
                roman = int_to_roman(i + 1)

                bbox = draw.textbbox((0, 0), roman, font=shade_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                rx = gx + (shade_tile_size - tw) // 2
                ry = shade_y + (shade_tile_size - th) // 2 - bbox[1] // 2

                draw.text((rx, ry), roman, fill=text_color, font=shade_font)

        return np.array(pil_image)

    def set_num_colors(self, num_colors: int) -> None:
        """Set the number of colors to extract."""
        num_colors = max(2, min(24, num_colors))
        self.settings.num_colors = num_colors
        self._kmeans = None
        self._colors = None
        self._labels = None
        self._color_to_index = {}
        self._palette_colors = None

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
