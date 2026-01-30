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


@dataclass
class PaletteSettings:
    """Settings for palette extraction."""

    num_colors: int = 9  # Must be multiple of 3 (3, 6, 9, 12, 15)
    max_iterations: int = 200
    random_state: int = 42
    sort_method: SortMethod = SortMethod.HUE
    grays_position: str = "end"  # "start", "end", or "mixed"
    gray_threshold: float = 12.0  # Saturation threshold for grays
    palette_method: PaletteMethod = PaletteMethod.DIVERSE  # Extraction method


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


def _is_pure_black_white_gray(r: int, g: int, b: int, tolerance: int = 5) -> bool:
    """
    Check if a color is pure black, pure white, or pure gray.

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

    # It's a grayscale color - check if pure black or pure white
    avg = (r + g + b) // 3
    if avg <= tolerance:  # Pure black
        return True
    if avg >= 255 - tolerance:  # Pure white
        return True

    # It's a gray (covered by shades)
    return True


def _is_gray_color_name(name: str) -> bool:
    """Check if a color name indicates a gray/black/white color."""
    gray_terms = [
        "schwarz", "grau", "weiß", "weiss", "anthrazit",
        "black", "gray", "grey", "white", "charcoal",
        "silver", "silber", "ash", "slate", "graphit"
    ]
    name_lower = name.lower()
    return any(term in name_lower for term in gray_terms)


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
    min_saturation: float = 15.0,
) -> List[ColorInfo]:
    """
    Select a diverse subset of colors maximizing contrast.

    Uses a greedy algorithm to pick colors that are maximally
    distant from already selected colors in HSL space.

    Args:
        colors: List of candidate colors.
        target_count: Number of colors to select.
        min_saturation: Minimum saturation for chromatic colors.

    Returns:
        List of selected diverse colors.
    """
    if len(colors) <= target_count:
        return colors

    # Separate into chromatic and achromatic
    chromatic = []
    achromatic = []

    for c in colors:
        h, s, l = _get_hsl(c.rgb)
        # Filter by saturation AND by name
        if s >= min_saturation and not _is_gray_color_name(c.name):
            chromatic.append(c)
        else:
            achromatic.append(c)

    # If not enough chromatic colors, relax constraints
    if len(chromatic) < target_count:
        # Add less saturated but still colored ones
        for c in colors:
            if c not in chromatic and c not in achromatic:
                chromatic.append(c)
            if len(chromatic) >= target_count:
                break

    if not chromatic:
        # Fallback: use all colors sorted by saturation
        return sorted(colors, key=lambda c: _get_color_saturation(*c.rgb), reverse=True)[:target_count]

    # Greedy selection: start with most saturated color
    chromatic.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
    selected = [chromatic[0]]
    remaining = chromatic[1:]

    while len(selected) < target_count and remaining:
        # Find color most distant from all selected colors
        best_color = None
        best_min_dist = -1

        for candidate in remaining:
            # Minimum distance to any selected color
            min_dist = min(
                _color_distance_hsl(candidate.rgb, sel.rgb)
                for sel in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_color = candidate

        if best_color:
            selected.append(best_color)
            remaining.remove(best_color)
        else:
            break

    # If still not enough, add remaining chromatic or achromatic
    while len(selected) < target_count and remaining:
        selected.append(remaining.pop(0))

    while len(selected) < target_count and achromatic:
        selected.append(achromatic.pop(0))

    return selected


class PaletteExtractor:
    """
    Extracts color palettes from images using K-Means clustering.

    Creates reduced color palettes and posterized images
    suitable for oil painting color planning.
    """

    VALID_STEPS = [3, 6, 9, 12, 15, 18, 21, 24]  # Multiples of 3, min 3

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
        num_colors = max(3, min(24, num_colors))  # Minimum 3 colors
        num_colors = round(num_colors / 3) * 3
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
        all_colors = []
        for i, color in enumerate(all_cluster_colors):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            pct = percentages.get(i, 0.0)

            # Filter out pure black, white, and grays
            if _is_pure_black_white_gray(r, g, b):
                continue

            all_colors.append(ColorInfo.from_rgb(r, g, b, round(pct, 2), index=i + 1))

        # Select colors based on method
        if self.settings.palette_method == PaletteMethod.DIVERSE:
            # Maximize color diversity - pick most distinct colors
            colors = _select_diverse_colors(
                all_colors,
                self.settings.num_colors,
                min_saturation=15.0,
            )
        elif self.settings.palette_method == PaletteMethod.SATURATED:
            # Filter out gray-named colors, then sort by saturation
            chromatic = [c for c in all_colors if not _is_gray_color_name(c.name)]
            chromatic.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
            colors = chromatic[: self.settings.num_colors]
            # Fill with remaining if not enough
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in all_colors if c not in colors]
                colors.extend(remaining[: self.settings.num_colors - len(colors)])
        else:
            # STANDARD: Original behavior - by percentage
            colors = [c for c in all_colors if not _is_gray_color_name(c.name)]
            colors.sort(key=lambda c: c.percentage, reverse=True)
            colors = colors[: self.settings.num_colors]
            # Fill with remaining if not enough
            if len(colors) < self.settings.num_colors:
                remaining = [c for c in all_colors if c not in colors]
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

                # Calculate font scale based on image size and region area
                # Base scale proportional to image diagonal (reference: 1500px = 1.0)
                image_diagonal = (width**2 + height**2) ** 0.5
                base_scale = image_diagonal / 1500.0

                # Area factor: larger regions get slightly larger text
                image_area = width * height
                area_factor = min(1.3, max(0.7, (area / (image_area * 0.01)) ** 0.3))

                # Final scale, clamped to reasonable range
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
        cols: int = 0,  # 0 = auto-calculate for DIN A4 ratio
        show_name: bool = True,
        show_number: bool = True,
        grayscales: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Create a visual palette image with color swatches.

        Uses PIL for text rendering to support Unicode (Umlauts).

        Args:
            colors: List of ColorInfo objects.
            swatch_size: Size of each color swatch in pixels.
            cols: Number of columns (0 = auto for DIN A4 landscape ratio).
            show_name: Whether to show color names.
            show_number: Whether to show numbers on swatches.
            grayscales: Optional list of grayscale values to add below.

        Returns:
            RGB image of the color palette.
        """
        num_colors = len(colors)

        # Scale factor for fonts/margins (reference: swatch_size=60)
        scale_factor = swatch_size / 60.0

        # Text height for layout calculation
        text_height = int(35 * scale_factor) if show_name else 0
        margin = max(2, int(3 * scale_factor))
        padding = int(8 * scale_factor)

        cell_height = swatch_size + text_height + padding
        cell_width = swatch_size

        # Auto-calculate columns for DIN A4 landscape ratio (1.414:1)
        if cols <= 0:
            target_ratio = 1.414
            gray_row_height = (swatch_size // 2 + text_height + int(20 * scale_factor)) if grayscales else 0

            best_cols = 3
            best_diff = float('inf')
            for test_cols in range(3, min(num_colors + 1, 12)):
                test_rows = (num_colors + test_cols - 1) // test_cols
                test_width = test_cols * cell_width
                test_height = test_rows * cell_height + gray_row_height
                ratio = test_width / test_height if test_height > 0 else 0
                diff = abs(ratio - target_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_cols = test_cols
            cols = best_cols

        rows = (num_colors + cols - 1) // cols

        # Grayscale section
        gray_section_height = 0
        gray_swatch_height = swatch_size // 2
        if grayscales:
            gray_section_height = gray_swatch_height + text_height + int(20 * scale_factor)

        # Create image with dark gray background
        width = cols * cell_width
        height = rows * cell_height + gray_section_height
        palette_img = np.ones((height, width, 3), dtype=np.uint8) * 50

        border_thickness = max(1, int(scale_factor))

        # Draw color swatches using OpenCV
        for i, color_info in enumerate(colors):
            row = i // cols
            col = i % cols
            x = col * cell_width
            y = row * cell_height

            r, g, b = color_info.rgb
            color_tuple = (int(r), int(g), int(b))
            cv2.rectangle(
                palette_img,
                (x + margin, y + margin),
                (x + swatch_size - margin, y + swatch_size - margin),
                color_tuple,
                -1,
            )
            cv2.rectangle(
                palette_img,
                (x + margin, y + margin),
                (x + swatch_size - margin, y + swatch_size - margin),
                (100, 100, 100),
                border_thickness,
            )

        # Draw grayscale swatches if provided
        if grayscales:
            separator_y = rows * cell_height + int(8 * scale_factor)
            line_thickness = max(1, int(scale_factor))
            line_margin = int(5 * scale_factor)
            cv2.line(
                palette_img,
                (line_margin, separator_y),
                (width - line_margin, separator_y),
                (100, 100, 100),
                line_thickness,
            )

            gray_y = separator_y + int(8 * scale_factor)
            num_grays = len(grayscales)
            gray_swatch_width = (width - 2 * line_margin) // num_grays

            for i, gray_val in enumerate(grayscales):
                gx = line_margin + i * gray_swatch_width
                gv = int(gray_val)
                cv2.rectangle(
                    palette_img,
                    (gx + margin, gray_y + margin),
                    (gx + gray_swatch_width - margin, gray_y + gray_swatch_height - margin),
                    (gv, gv, gv),
                    -1,
                )
                cv2.rectangle(
                    palette_img,
                    (gx + margin, gray_y + margin),
                    (gx + gray_swatch_width - margin, gray_y + gray_swatch_height - margin),
                    (100, 100, 100),
                    border_thickness,
                )

        # Convert to PIL Image for text rendering (supports Unicode)
        pil_image = Image.fromarray(palette_img)
        draw = ImageDraw.Draw(pil_image)

        # Font sizes based on scale factor
        number_font_size = max(16, int(24 * scale_factor))
        name_font_size = max(12, int(18 * scale_factor))
        gray_font_size = max(12, int(16 * scale_factor))

        number_font = self._get_font(number_font_size)
        name_font = self._get_font(name_font_size)
        gray_font = self._get_font(gray_font_size)

        # Draw text on color swatches
        for i, color_info in enumerate(colors):
            row = i // cols
            col = i % cols
            x = col * cell_width
            y = row * cell_height

            r, g, b = color_info.rgb

            # Draw number on swatch
            if show_number:
                tc = get_text_color_for_background((r, g, b))
                text_color = (int(tc[0]), int(tc[1]), int(tc[2]))
                number_text = str(color_info.index)

                bbox = draw.textbbox((0, 0), number_text, font=number_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                num_x = x + (swatch_size - tw) // 2
                num_y = y + (swatch_size - th) // 2 - bbox[1]

                draw.text((num_x, num_y), number_text, fill=text_color, font=number_font)

            # Draw name below swatch
            if show_name:
                name = color_info.name
                max_text_width = swatch_size - int(10 * scale_factor)

                current_font = name_font
                current_size = name_font_size

                bbox = draw.textbbox((0, 0), name, font=current_font)
                tw = bbox[2] - bbox[0]

                while tw > max_text_width and current_size > 8:
                    current_size = int(current_size * 0.85)
                    current_font = self._get_font(current_size)
                    bbox = draw.textbbox((0, 0), name, font=current_font)
                    tw = bbox[2] - bbox[0]

                text_x = x + int(5 * scale_factor)
                text_y = y + swatch_size + int(8 * scale_factor)

                draw.text((text_x, text_y), name, fill=(220, 220, 220), font=current_font)

        # Draw Roman numerals on grayscale swatches
        if grayscales:
            from utils.color_naming import int_to_roman

            separator_y = rows * cell_height + int(8 * scale_factor)
            gray_y = separator_y + int(8 * scale_factor)
            line_margin = int(5 * scale_factor)
            num_grays = len(grayscales)
            gray_swatch_width = (width - 2 * line_margin) // num_grays

            for i, gray_val in enumerate(grayscales):
                gx = line_margin + i * gray_swatch_width
                gv = int(gray_val)

                tc = get_text_color_for_background((gv, gv, gv))
                text_color = (int(tc[0]), int(tc[1]), int(tc[2]))
                roman = int_to_roman(i + 1)

                bbox = draw.textbbox((0, 0), roman, font=gray_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                rx = gx + (gray_swatch_width - tw) // 2
                ry = gray_y + (gray_swatch_height - th) // 2 - bbox[1]

                draw.text((rx, ry), roman, fill=text_color, font=gray_font)

        # Convert back to numpy array
        return np.array(pil_image)

    def set_num_colors(self, num_colors: int) -> None:
        """Set the number of colors to extract."""
        num_colors = max(3, min(24, num_colors))
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
