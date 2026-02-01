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
    FAMILY = "family"  # By color family (warm to cold) - DEFAULT
    PERCENTAGE = "percentage"  # By area percentage
    HUE = "hue"  # By color hue
    LIGHTNESS = "lightness"  # By lightness
    SATURATION = "saturation"  # By saturation


class PaletteMethod(Enum):
    """Palette extraction methods."""
    STANDARD = "standard"  # K-Means by frequency
    DIVERSE = "diverse"  # Maximize color diversity/contrast
    SATURATED = "saturated"  # Prioritize most saturated colors
    INTENSIFY = "intensify"  # Maximize hue diversity, guarantee vibrant colors from each hue
    GLOW = "glow"  # Lightest colors from most frequent families
    LUMINOUS = "luminous"  # Diverse + Glow compromise


@dataclass
class PaletteSettings:
    """Settings for palette extraction."""

    num_colors: int = 9  # Must be multiple of 3 (3, 6, 9, 12, 15)
    max_iterations: int = 200
    random_state: int = 42
    sort_method: SortMethod = SortMethod.FAMILY  # Default: warm to cold
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


# Family order for warm-to-cold sorting
FAMILY_ORDER_WARM_TO_COLD = {
    "rot": 0,
    "orange": 1,
    "gelb": 2,
    "braun": 3,  # Warm but earthy
    "grün": 4,   # Neutral
    "blau": 5,   # Cold
    "violett": 6,  # Cold
    "gray": 7,   # Grays at end
    "extreme": 8,
}


def sort_palette(
    colors: List[ColorInfo],
    method: SortMethod = SortMethod.FAMILY,
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
    elif method == SortMethod.FAMILY:
        # Sort by color family (warm to cold), then by saturation within family
        def family_sort_key(color: ColorInfo):
            family = _get_color_family(color.rgb)
            family_order = FAMILY_ORDER_WARM_TO_COLD.get(family, 99)
            _, s, _ = _get_hsl(color.rgb)
            # Within family: most saturated first
            return (family_order, -s)

        sorted_colors = sorted(colors, key=family_sort_key)
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


def _is_extreme_lightness(r: int, g: int, b: int) -> bool:
    """
    Check if a color has extreme lightness (almost white or almost black).

    These should be left to the shades layer, not color palette.
    L > 95% = almost white, L < 5% = almost black

    Returns:
        True if lightness is extreme.
    """
    _, _, l = _get_hsl((r, g, b))
    return l > 95 or l < 5


def _is_chromatic_dynamic(r: int, g: int, b: int, for_glow: bool = False) -> bool:
    """
    Check if a color is chromatic using lightness-aware thresholds.

    For very light colors, we accept lower saturation because physically
    light colors have lower saturation but can still have clear hue.

    Args:
        r, g, b: RGB values.
        for_glow: If True, use more permissive thresholds for GLOW/LUMINOUS methods.

    Returns:
        True if the color is chromatic (has enough hue to be a "color").
    """
    _, s, l = _get_hsl((r, g, b))

    # Extreme lightness = not for color palette (shades layer)
    if l > 95 or l < 5:
        return False

    if for_glow:
        # GLOW/LUMINOUS: More permissive - we want to find light colors!
        # Very light (L > 80): S >= 5% is enough
        # Light (L > 65): S >= 8%
        # Medium: S >= 12%
        # Dark: S >= 15%
        if l > 80:
            return s >= 5
        elif l > 65:
            return s >= 8
        elif l > 40:
            return s >= 12
        else:
            return s >= 15
    else:
        # Standard methods: slightly more permissive for light colors
        # Very light (L > 80): S >= 8%
        # Light (L > 65): S >= 12%
        # Normal: S >= 18%
        if l > 80:
            return s >= 8
        elif l > 65:
            return s >= 12
        else:
            return s >= 18


# The 7 artist color families for palette diversity
COLOR_FAMILIES = ["rot", "orange", "gelb", "grün", "blau", "violett", "braun"]

# Complementary family pairs (for INTENSIFY method)
# Based on color wheel: opposite hues create strongest contrast
COMPLEMENTARY_PAIRS = [
    ("orange", "blau"),    # ~30° ↔ ~210°
    ("gelb", "violett"),   # ~60° ↔ ~270°
    ("rot", "grün"),       # ~0° ↔ ~120°
    ("braun", "blau"),     # Earthy warm ↔ cold
]

# Hue angles for each family (for DIVERSE method - color wheel spread)
FAMILY_HUE_ANGLES = {
    "rot": 0,
    "orange": 30,
    "gelb": 60,
    "grün": 120,
    "blau": 210,
    "violett": 270,
    "braun": 25,  # Between red and orange
}


def _get_family_stats(chromatic_colors: List["ColorInfo"]) -> Dict[str, Dict]:
    """
    Calculate statistics for each color family present in the image.

    Returns dict with family -> {
        'colors': List of ColorInfo,
        'area': Total percentage,
        'max_saturation': Highest saturation in family,
        'max_lightness': Highest lightness in family,
        'intensity': saturation * lightness (for dominance calc),
        'anchor': Most vibrant color (family representative)
    }
    """
    from collections import defaultdict

    family_data = defaultdict(lambda: {
        'colors': [],
        'area': 0.0,
        'max_saturation': 0.0,
        'max_lightness': 0.0,
        'intensity': 0.0,
        'anchor': None
    })

    for c in chromatic_colors:
        family = _get_color_family(c.rgb)
        if family in ("gray", "extreme"):
            continue

        h, s, l = _get_hsl(c.rgb)

        family_data[family]['colors'].append(c)
        family_data[family]['area'] += c.percentage

        if s > family_data[family]['max_saturation']:
            family_data[family]['max_saturation'] = s

        if l > family_data[family]['max_lightness']:
            family_data[family]['max_lightness'] = l

        # Intensity = saturation * lightness (normalized)
        intensity = (s / 100.0) * (l / 100.0)
        if intensity > family_data[family]['intensity']:
            family_data[family]['intensity'] = intensity

        # Anchor = most saturated color in family
        if family_data[family]['anchor'] is None or s > _get_color_saturation(*family_data[family]['anchor'].rgb):
            family_data[family]['anchor'] = c

    return dict(family_data)


def _rank_families_by_criterion(
    family_stats: Dict[str, Dict],
    criterion: str
) -> List[Tuple[str, "ColorInfo"]]:
    """
    Rank families by a criterion and return (family, anchor) pairs.

    Criteria:
    - 'area': By total percentage (largest first)
    - 'saturation': By max saturation (most saturated first)
    - 'lightness': By max lightness (lightest first)
    - 'intensity': By saturation × lightness (most intense first)
    """
    families = list(family_stats.keys())

    if criterion == 'area':
        families.sort(key=lambda f: family_stats[f]['area'], reverse=True)
    elif criterion == 'saturation':
        families.sort(key=lambda f: family_stats[f]['max_saturation'], reverse=True)
    elif criterion == 'lightness':
        families.sort(key=lambda f: family_stats[f]['max_lightness'], reverse=True)
    elif criterion == 'intensity':
        families.sort(key=lambda f: family_stats[f]['intensity'], reverse=True)

    return [(f, family_stats[f]['anchor']) for f in families if family_stats[f]['anchor']]


def _get_complementary_pair_scores(
    family_stats: Dict[str, Dict]
) -> List[Tuple[str, str, float, "ColorInfo", "ColorInfo"]]:
    """
    Score complementary pairs by their combined dominance.

    Dominance = rank_by_intensity × rank_by_area (lower = more dominant)

    Returns list of (family1, family2, score, anchor1, anchor2) sorted by score.
    The first family in each pair is the LIGHTER one (to be filled first).
    """
    # Get intensity and area ranks
    intensity_ranked = _rank_families_by_criterion(family_stats, 'intensity')
    area_ranked = _rank_families_by_criterion(family_stats, 'area')

    intensity_rank = {f: i + 1 for i, (f, _) in enumerate(intensity_ranked)}
    area_rank = {f: i + 1 for i, (f, _) in enumerate(area_ranked)}

    # Calculate dominance for each family
    dominance = {}
    for family in family_stats:
        if family in intensity_rank and family in area_rank:
            dominance[family] = intensity_rank[family] * area_rank[family]

    # Score each complementary pair
    pair_scores = []
    for f1, f2 in COMPLEMENTARY_PAIRS:
        if f1 in family_stats and f2 in family_stats:
            # Combined score = product of dominances (lower = more dominant pair)
            score = dominance.get(f1, 999) * dominance.get(f2, 999)

            # Determine which is lighter (to fill first)
            l1 = family_stats[f1]['max_lightness']
            l2 = family_stats[f2]['max_lightness']

            if l1 >= l2:
                pair_scores.append((f1, f2, score, family_stats[f1]['anchor'], family_stats[f2]['anchor']))
            else:
                pair_scores.append((f2, f1, score, family_stats[f2]['anchor'], family_stats[f1]['anchor']))

    # Sort by score (lower = more dominant pair first)
    pair_scores.sort(key=lambda x: x[2])
    return pair_scores


def _get_optimal_hue_spread(num_colors: int) -> float:
    """
    Get the optimal hue angle separation for N colors on color wheel.

    2 colors: 180° (opposite)
    3 colors: 120° (triadic)
    4 colors: 90° (tetradic)
    etc.
    """
    if num_colors <= 1:
        return 360.0
    return 360.0 / num_colors


def _find_closest_family_to_angle(
    target_angle: float,
    available_families: List[str]
) -> Optional[str]:
    """
    Find the family whose hue angle is closest to the target angle.
    """
    if not available_families:
        return None

    best_family = None
    best_diff = float('inf')

    for family in available_families:
        family_angle = FAMILY_HUE_ANGLES.get(family, 0)
        # Circular distance
        diff = min(abs(family_angle - target_angle), 360 - abs(family_angle - target_angle))
        if diff < best_diff:
            best_diff = diff
            best_family = family

    return best_family


def _blend_colors(c1: "ColorInfo", c2: "ColorInfo") -> Tuple[int, int, int]:
    """
    Blend two colors to their average RGB.
    """
    r = (c1.rgb[0] + c2.rgb[0]) // 2
    g = (c1.rgb[1] + c2.rgb[1]) // 2
    b = (c1.rgb[2] + c2.rgb[2]) // 2
    return (r, g, b)


def _balance_family_distribution(
    colors: List["ColorInfo"],
    family_stats: Dict[str, Dict],
    method: "PaletteMethod",
    chromatic_colors: List["ColorInfo"]
) -> List["ColorInfo"]:
    """
    Post-processing: Balance family distribution in palette.

    RULES:
    1. Every present family must have at least 1 color
    2. No family can have more than 2x the colors of any other family
       (If ratio > 2:1, merge two lowest-saturation of dominant family,
        fill freed slot with underrepresented family)
    3. Apply recursively (domino effect) until balanced
    """
    if len(colors) < 3:
        return colors

    result = list(colors)
    max_iterations = 15  # Increased for more thorough balancing
    tried_pairs = set()  # Track which dominant/underrep pairs we've tried

    for iteration in range(max_iterations):
        # Count colors per family in current result
        family_counts = {}
        family_colors = {}
        for i, c in enumerate(result):
            family = _get_color_family(c.rgb)
            if family not in ("gray", "extreme"):
                family_counts[family] = family_counts.get(family, 0) + 1
                if family not in family_colors:
                    family_colors[family] = []
                family_colors[family].append((i, c))

        if len(family_counts) < 2:
            break

        # Sort families by count (descending)
        sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)

        # PRIORITY 1: Check for families in IMAGE but missing from palette
        present_in_image = set(family_stats.keys())
        present_in_palette = set(family_counts.keys())
        missing_families = present_in_image - present_in_palette

        found_imbalance = False
        dominant_family = None
        underrepresented_family = None

        if missing_families:
            # CRITICAL: A family in the image has NO representation!
            # Find any family in palette with >= 2 colors to take from
            for fam, count in sorted_families:
                if count >= 2:
                    pair_key = (fam, tuple(sorted(missing_families)))
                    if pair_key not in tried_pairs:
                        dominant_family = fam
                        # Prefer missing families that had higher area in image
                        best_missing = max(
                            missing_families,
                            key=lambda f: family_stats[f]['area'] if f in family_stats else 0
                        )
                        underrepresented_family = best_missing
                        found_imbalance = True
                        tried_pairs.add(pair_key)
                        break

        # PRIORITY 2: Find ANY pair where ratio > 2:1
        if not found_imbalance:
            for i in range(len(sorted_families)):
                fam_i, count_i = sorted_families[i]
                for j in range(i + 1, len(sorted_families)):
                    fam_j, count_j = sorted_families[j]
                    # Check if ratio > 2:1 (i.e., count_i > count_j * 2)
                    if count_i > count_j * 2 and count_i >= 3:
                        pair_key = (fam_i, fam_j)
                        if pair_key not in tried_pairs:
                            dominant_family = fam_i
                            underrepresented_family = fam_j
                            found_imbalance = True
                            tried_pairs.add(pair_key)
                            break
                if found_imbalance:
                    break

        if not found_imbalance:
            break  # Palette is balanced

        # Check if we can actually rebalance
        if dominant_family not in family_colors or len(family_colors[dominant_family]) < 2:
            continue  # Can't rebalance this pair, try next iteration

        dominant_colors_list = family_colors[dominant_family]
        # Sort by saturation (lowest first - merge the least saturated ones)
        dominant_colors_list.sort(key=lambda x: _get_color_saturation(*x[1].rgb))

        # Get the two lowest saturation ones
        idx1, color1 = dominant_colors_list[0]
        idx2, color2 = dominant_colors_list[1]

        # Blend them into one color
        blended_rgb = _blend_colors(color1, color2)
        blended_color = ColorInfo.from_rgb(
            blended_rgb[0], blended_rgb[1], blended_rgb[2],
            (color1.percentage + color2.percentage) / 2,
            color1.index
        )

        # Replace first slot with blend
        result[idx1] = blended_color

        # Find replacement for second slot
        replacement_found = False

        # Strategy 1: Color from the specific underrepresented family
        if underrepresented_family and underrepresented_family in family_stats:
            candidates = sorted(
                family_stats[underrepresented_family]['colors'],
                key=lambda c: _get_color_saturation(*c.rgb),
                reverse=True
            )
            for candidate in candidates:
                if candidate not in result:
                    result[idx2] = candidate
                    replacement_found = True
                    break

        # Strategy 2: Any family with < 2 colors in palette
        if not replacement_found:
            low_count_families = [f for f, c in family_counts.items() if c < 2 and f != dominant_family]
            for target_fam in low_count_families:
                if target_fam in family_stats:
                    candidates = sorted(
                        family_stats[target_fam]['colors'],
                        key=lambda c: _get_color_saturation(*c.rgb),
                        reverse=True
                    )
                    for candidate in candidates:
                        if candidate not in result:
                            result[idx2] = candidate
                            replacement_found = True
                            break
                if replacement_found:
                    break

        # Strategy 3: Any missing family from image
        if not replacement_found and missing_families:
            for missing_fam in missing_families:
                if missing_fam in family_stats:
                    candidates = sorted(
                        family_stats[missing_fam]['colors'],
                        key=lambda c: _get_color_saturation(*c.rgb),
                        reverse=True
                    )
                    for candidate in candidates:
                        if candidate not in result:
                            result[idx2] = candidate
                            replacement_found = True
                            break
                if replacement_found:
                    break

        # Strategy 4: Fallback - any chromatic color from underrepresented family
        if not replacement_found:
            for c in chromatic_colors:
                c_family = _get_color_family(c.rgb)
                if c_family not in ("gray", "extreme") and c not in result:
                    c_family_count = family_counts.get(c_family, 0)
                    if c_family_count < 2:
                        result[idx2] = c
                        replacement_found = True
                        break

        # If still no replacement, revert the blend and continue trying other pairs
        if not replacement_found:
            result[idx1] = color1  # Undo the blend
            # Don't break - continue trying other imbalanced pairs

    return result


def _final_balance_palette(
    colors: List["ColorInfo"],
    family_stats: Dict[str, Dict],
    chromatic_colors: List["ColorInfo"],
    log_func=None
) -> List["ColorInfo"]:
    """
    Final balance enforcement on completed palette.

    Uses CASCADING approach:
    1. Find most imbalanced pair (e.g., B:O = 8:1)
    2. Cluster 2 from dominant family, add 1 from underrepresented
    3. Check if this creates new imbalance (cascade)
    4. Repeat until all pairs are ≤ 2:1

    Args:
        colors: List of colors in palette (all slots filled).
        family_stats: Statistics about families in image.
        chromatic_colors: All available chromatic colors.
        log_func: Optional logging function.

    Returns:
        Balanced list of colors.
    """
    if len(colors) < 3:
        return colors

    def log(msg):
        if log_func:
            log_func(msg)

    result = list(colors)
    max_iterations = 20

    for iteration in range(max_iterations):
        # Count colors per family
        family_counts = {}
        family_colors_in_result = {}
        for i, c in enumerate(result):
            fam = _get_color_family(c.rgb)
            if fam not in ("gray", "extreme"):
                family_counts[fam] = family_counts.get(fam, 0) + 1
                if fam not in family_colors_in_result:
                    family_colors_in_result[fam] = []
                family_colors_in_result[fam].append((i, c))

        if len(family_counts) < 2:
            break

        # Find most imbalanced pair
        sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)

        # Find pair with worst ratio
        worst_ratio = 0
        dominant_fam = None
        under_fam = None

        for i, (fam_i, count_i) in enumerate(sorted_families):
            for j in range(i + 1, len(sorted_families)):
                fam_j, count_j = sorted_families[j]
                if count_j > 0:
                    ratio = count_i / count_j
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        dominant_fam = fam_i
                        under_fam = fam_j

        # Check if ratio exceeds 2:1
        if worst_ratio <= 2.0:
            log(f"  Alle Verhältnisse ≤ 2:1, Balancierung abgeschlossen.")
            break

        log(f"  Iteration {iteration + 1}: {dominant_fam}:{under_fam} = "
            f"{family_counts[dominant_fam]}:{family_counts[under_fam]} "
            f"(Ratio: {worst_ratio:.1f}:1)")

        # Get colors from dominant family, sorted by saturation (lowest first)
        dom_colors = family_colors_in_result.get(dominant_fam, [])
        if len(dom_colors) < 2:
            log(f"    Kann nicht clustern - zu wenig Farben in {dominant_fam}")
            break

        dom_colors.sort(key=lambda x: _get_color_saturation(*x[1].rgb))

        # Cluster (merge) 2 lowest saturation colors from dominant family
        idx1, color1 = dom_colors[0]
        idx2, color2 = dom_colors[1]

        blended_rgb = _blend_colors(color1, color2)
        blended_color = ColorInfo.from_rgb(
            blended_rgb[0], blended_rgb[1], blended_rgb[2],
            (color1.percentage + color2.percentage) / 2,
            color1.index
        )

        # Find replacement from underrepresented family
        replacement = None

        # Try from family_stats first
        if under_fam in family_stats:
            candidates = sorted(
                family_stats[under_fam]['colors'],
                key=lambda c: _get_color_saturation(*c.rgb),
                reverse=True
            )
            for cand in candidates:
                if cand not in result:
                    replacement = cand
                    break

        # If not found, try generating a variation
        if not replacement:
            under_colors = [c for c in result if _get_color_family(c.rgb) == under_fam]
            if under_colors:
                base = under_colors[0]
                # Generate a variation
                var_rgb = _generate_color_variation(base.rgb, "lighter", 0.2)
                # Check it's different enough
                is_unique = all(
                    _color_distance_hsl(var_rgb, c.rgb) >= 0.08
                    for c in result
                )
                if is_unique:
                    replacement = ColorInfo.from_rgb(
                        var_rgb[0], var_rgb[1], var_rgb[2],
                        base.percentage * 0.5,
                        len(result)
                    )

        if replacement:
            # Apply changes
            result[idx1] = blended_color
            result[idx2] = replacement
            log(f"    Clustered: {color1.name} + {color2.name} → {blended_color.name}")
            log(f"    Added: {replacement.name} ({under_fam})")
        else:
            log(f"    Kein Ersatz für {under_fam} gefunden, überspringe...")
            # Mark this pair as tried to avoid infinite loop
            break

    return result


def _get_color_family(rgb: Tuple[int, int, int], for_glow: bool = False) -> str:
    """
    Get the color family for a color (7 artist families).

    Families: rot, orange, gelb, grün, blau, violett, braun
    Returns "gray" for achromatic colors or "extreme" for L>95%/L<5%.
    """
    r, g, b = rgb
    h, s, l = _get_hsl(rgb)

    # Extreme lightness = shades layer, not color palette
    if l > 95 or l < 5:
        return "extreme"

    # Gray threshold depends on lightness
    # Light colors need less saturation to show hue
    if for_glow:
        gray_threshold = 5 if l > 80 else (8 if l > 65 else 12)
    else:
        gray_threshold = 8 if l > 80 else (10 if l > 65 else 12)

    if s < gray_threshold:
        return "gray"

    # Detect brown: warm hues with LOW saturation and medium-low lightness
    # Browns are muted, earthy colors - NOT saturated dark reds!
    # Key distinction: S < 45% = brown territory, S >= 45% = dark red/orange
    # Browns are typically: H=15-50 (orange-yellow range), S=15-45, L=15-45
    # Dark reds (H=0-15 or 345-360) with S >= 45% should remain ROT
    if 15 <= h < 50 and s < 45 and 15 < l < 50:
        # Brown: orange-yellow hue range with low saturation
        if r > b:
            return "braun"

    # Very dark and very desaturated warm colors = brown
    if l < 30 and s < 35:
        # Only if NOT in pure red range (pure red stays red even when dark)
        if 20 <= h < 50:
            return "braun"

    # Categorize by hue angle (7 families, no cyan/magenta)
    if h < 15 or h >= 345:
        return "rot"
    elif h < 45:
        return "orange"
    elif h < 75:
        return "gelb"
    elif h < 165:
        return "grün"
    elif h < 270:
        return "blau"
    elif h < 330:
        return "violett"
    else:
        return "rot"  # Wrap around to red


def _is_invalid_color_by_name(name: str) -> bool:
    """
    Check if a color name indicates an invalid color (gray/white/black).

    These should be excluded from the color palette as they're covered by shades.
    """
    invalid_terms = [
        "schwarz", "grau", "weiß", "weiss", "anthrazit",
        "black", "gray", "grey", "white", "charcoal",
        "silver", "silber", "ash", "slate", "graphit",
    ]
    name_lower = name.lower()
    return any(term in name_lower for term in invalid_terms)


def _blend_toward_valid_color(
    invalid_color: Tuple[int, int, int],
    valid_neighbor: Tuple[int, int, int],
    ratio: float = 0.73
) -> Tuple[int, int, int]:
    """
    Blend an invalid color toward a valid neighbor.

    Args:
        invalid_color: RGB of the invalid (gray) color
        valid_neighbor: RGB of the valid chromatic neighbor
        ratio: How much of invalid to keep (0.73 = 73% invalid, 27% valid)

    Returns:
        Blended RGB tuple
    """
    r = int(invalid_color[0] * ratio + valid_neighbor[0] * (1 - ratio))
    g = int(invalid_color[1] * ratio + valid_neighbor[1] * (1 - ratio))
    b = int(invalid_color[2] * ratio + valid_neighbor[2] * (1 - ratio))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _find_valid_replacement(
    invalid_rgb: Tuple[int, int, int],
    all_colors: List[ColorInfo],
    max_iterations: int = 20
) -> Optional[Tuple[int, int, int]]:
    """
    Find a valid (non-gray) replacement for an invalid color by blending.

    Iteratively blends the invalid color toward a valid neighbor until
    the result is no longer named as gray/white/black.

    Args:
        invalid_rgb: The invalid color RGB
        all_colors: List of all available colors to find a neighbor
        max_iterations: Maximum blend iterations

    Returns:
        Valid RGB tuple or None if no valid replacement found
    """
    # Find the most saturated valid neighbor
    valid_neighbors = [
        c for c in all_colors
        if not _is_invalid_color_by_name(c.name)
        and _get_color_saturation(*c.rgb) > 25
    ]

    if not valid_neighbors:
        return None

    # Sort by saturation to get the most vibrant neighbor
    valid_neighbors.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
    neighbor = valid_neighbors[0]

    current_rgb = invalid_rgb
    for _ in range(max_iterations):
        # Blend toward valid neighbor
        current_rgb = _blend_toward_valid_color(current_rgb, neighbor.rgb, ratio=0.73)

        # Check if now valid
        name = rgb_to_name(*current_rgb)
        if not _is_invalid_color_by_name(name):
            return current_rgb

    return None


def _check_family_diversity(colors: List[ColorInfo]) -> bool:
    """
    Check if colors have good family diversity.

    Returns False if >50% of colors share the same color family.
    """
    if len(colors) < 3:
        return True  # Too few colors to judge

    family_counts = {}
    for c in colors:
        fam = _get_color_family(c.rgb)
        family_counts[fam] = family_counts.get(fam, 0) + 1

    max_count = max(family_counts.values())
    return max_count <= len(colors) * 0.5  # No family should have >50%


def _find_most_vibrant_by_family(
    colors: List[ColorInfo],
    for_glow: bool = False,
) -> Dict[str, ColorInfo]:
    """
    Find the most vibrant color from each of the 7 color families.

    Families: rot, orange, gelb, grün, blau, violett, braun
    This ensures diverse color representation in the final palette.

    Args:
        colors: List of ColorInfo objects.
        for_glow: If True, use permissive thresholds for family classification.

    Returns:
        Dict mapping family name to most vibrant ColorInfo from that family.
    """
    family_colors: Dict[str, ColorInfo] = {}

    for c in colors:
        family = _get_color_family(c.rgb, for_glow=for_glow)
        if family == "gray" or family == "extreme":
            continue  # Skip grays and extreme lightness

        sat = _get_color_saturation(*c.rgb)

        # Keep most vibrant per family
        if family not in family_colors:
            family_colors[family] = c
        elif sat > _get_color_saturation(*family_colors[family].rgb):
            family_colors[family] = c

    return family_colors


def _get_family_anchors(colors: List[ColorInfo], for_glow: bool = False) -> List[ColorInfo]:
    """
    Get one representative (most vibrant) color from each present color family.

    Args:
        colors: List of ColorInfo objects.
        for_glow: If True, use permissive thresholds for family classification.

    Returns list sorted by saturation (most vibrant first).
    """
    family_map = _find_most_vibrant_by_family(colors, for_glow=for_glow)
    anchors = list(family_map.values())
    anchors.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
    return anchors


def _get_color_lightness(r: int, g: int, b: int) -> float:
    """Get lightness value (0-100) for an RGB color."""
    _, _, l = _get_hsl((r, g, b))
    return l


def _get_family_by_area(colors: List[ColorInfo], for_glow: bool = False) -> List[str]:
    """
    Get list of color families sorted by total area (percentage).

    Args:
        colors: List of ColorInfo objects.
        for_glow: If True, use permissive thresholds for family classification.

    Returns families from most frequent to least frequent.
    """
    family_area: Dict[str, float] = {}

    for c in colors:
        family = _get_color_family(c.rgb, for_glow=for_glow)
        if family == "gray" or family == "extreme":
            continue
        family_area[family] = family_area.get(family, 0) + c.percentage

    # Sort by area descending
    sorted_families = sorted(family_area.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_families]


def _get_colors_by_family_and_lightness(
    colors: List[ColorInfo],
    for_glow: bool = False,
) -> Dict[str, List[ColorInfo]]:
    """
    Group colors by family, each list sorted by lightness (lightest first).

    Args:
        colors: List of ColorInfo objects.
        for_glow: If True, use permissive thresholds for family classification.

    Returns dict mapping family name to list of ColorInfo sorted by lightness.
    """
    family_colors: Dict[str, List[ColorInfo]] = {}

    for c in colors:
        family = _get_color_family(c.rgb, for_glow=for_glow)
        if family == "gray" or family == "extreme":
            continue
        if family not in family_colors:
            family_colors[family] = []
        family_colors[family].append(c)

    # Sort each family by lightness (lightest first)
    for family in family_colors:
        family_colors[family].sort(
            key=lambda c: _get_color_lightness(*c.rgb), reverse=True
        )

    return family_colors


def _select_glow_colors(
    colors: List[ColorInfo],
    target_count: int,
) -> List[ColorInfo]:
    """
    Select colors using the GLOW method:
    1. Start with family anchors (most vibrant from each family)
    2. Fill remaining slots by cycling through families by frequency,
       taking the LIGHTEST remaining color from each family.

    Uses permissive saturation thresholds (for_glow=True) to include
    light colors that still have discernible hue.

    Args:
        colors: List of chromatic colors (should be chromatic_colors_glow).
        target_count: Number of colors to select.

    Returns:
        List of selected colors.
    """
    # Step 1: Get family anchors (most vibrant per family)
    # Use for_glow=True for permissive family classification
    family_anchors = _get_family_anchors(colors, for_glow=True)
    selected = family_anchors[:target_count]

    if len(selected) >= target_count:
        return selected

    # Step 2: Get families sorted by area and colors by family/lightness
    # Use for_glow=True for permissive thresholds
    families_by_area = _get_family_by_area(colors, for_glow=True)
    colors_by_family = _get_colors_by_family_and_lightness(colors, for_glow=True)

    # Track which colors have been used
    used_colors = set(id(c) for c in selected)

    # Track current index in each family's lightness list
    family_index: Dict[str, int] = {f: 0 for f in families_by_area}

    # Cycle through families, taking lightest available from each
    while len(selected) < target_count:
        added_any = False

        for family in families_by_area:
            if len(selected) >= target_count:
                break

            if family not in colors_by_family:
                continue

            family_list = colors_by_family[family]

            # Find next unused color in this family
            while family_index[family] < len(family_list):
                candidate = family_list[family_index[family]]
                family_index[family] += 1

                if id(candidate) not in used_colors:
                    selected.append(candidate)
                    used_colors.add(id(candidate))
                    added_any = True
                    break

        # If no colors were added in this cycle, break to avoid infinite loop
        if not added_any:
            break

    return selected


def _select_luminous_colors(
    colors: List[ColorInfo],
    target_count: int,
) -> List[ColorInfo]:
    """
    Select colors using the LUMINOUS method:
    Compromise between DIVERSE (maximum contrast) and GLOW (lightest by frequency).

    Combines both approaches:
    1. Start with family anchors
    2. For remaining slots, score candidates by both:
       - Distance to selected colors (DIVERSE component)
       - Lightness weighted by family frequency (GLOW component)

    Uses permissive saturation thresholds (for_glow=True) to include
    light colors that still have discernible hue.

    Args:
        colors: List of chromatic colors (should be chromatic_colors_glow).
        target_count: Number of colors to select.

    Returns:
        List of selected colors.
    """
    # Step 1: Get family anchors (use for_glow=True for permissive thresholds)
    family_anchors = _get_family_anchors(colors, for_glow=True)
    selected = family_anchors[:target_count]

    if len(selected) >= target_count:
        return selected

    # Get family areas for weighting (use for_glow=True)
    families_by_area = _get_family_by_area(colors, for_glow=True)
    family_rank = {f: i for i, f in enumerate(families_by_area)}
    max_rank = len(families_by_area)

    # Track used colors
    used_colors = set(id(c) for c in selected)
    remaining = [c for c in colors if id(c) not in used_colors]

    while len(selected) < target_count and remaining:
        best_color = None
        best_score = -1

        for candidate in remaining:
            # DIVERSE component: minimum distance to selected colors
            min_dist = min(
                _color_distance_hsl(candidate.rgb, sel.rgb)
                for sel in selected
            )

            # GLOW component: lightness weighted by family rank
            lightness = _get_color_lightness(*candidate.rgb) / 100.0
            family = _get_color_family(candidate.rgb, for_glow=True)
            rank = family_rank.get(family, max_rank)
            # Higher rank = less frequent = lower weight
            frequency_weight = 1.0 - (rank / (max_rank + 1))

            # Combined score: 50% diverse, 50% glow
            diverse_score = min_dist
            glow_score = lightness * frequency_weight

            score = 0.5 * diverse_score + 0.5 * glow_score

            if score > best_score:
                best_score = score
                best_color = candidate

        if best_color:
            selected.append(best_color)
            used_colors.add(id(best_color))
            remaining.remove(best_color)
        else:
            break

    return selected


def _get_color_saturation(r: int, g: int, b: int) -> float:
    """Get saturation value (0-100) for an RGB color."""
    _, s, _ = _get_hsl((r, g, b))
    return s


def _generate_color_variation(
    rgb: Tuple[int, int, int],
    variation_type: str,
    strength: float = 0.15,
) -> Tuple[int, int, int]:
    """
    Generate a color variation for filling missing palette slots.

    Args:
        rgb: Original RGB color.
        variation_type: Type of variation:
            - "lighter": Increase lightness
            - "darker": Decrease lightness
            - "saturated": Increase saturation
            - "desaturated": Decrease saturation
        strength: How much to vary (0.0-1.0).

    Returns:
        New RGB tuple.
    """
    r, g, b = rgb
    # Convert to HLS (note: colorsys uses HLS, not HSL)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

    if variation_type == "lighter":
        l = min(0.95, l + (1.0 - l) * strength)
    elif variation_type == "darker":
        l = max(0.05, l - l * strength)
    elif variation_type == "saturated":
        s = min(1.0, s + (1.0 - s) * strength)
    elif variation_type == "desaturated":
        s = max(0.0, s - s * strength)

    # Convert back to RGB
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
    return (
        max(0, min(255, int(r_new * 255))),
        max(0, min(255, int(g_new * 255))),
        max(0, min(255, int(b_new * 255))),
    )


def _calculate_balanced_slot_distribution(
    family_counts: Dict[str, int],
    target_count: int,
) -> Dict[str, int]:
    """
    Calculate how many slots each family should get, respecting 2:1 max ratio.

    Args:
        family_counts: Current count of colors per family.
        target_count: Total slots to fill.

    Returns:
        Dict mapping family -> target slot count.
    """
    families = list(family_counts.keys())
    num_families = len(families)

    if num_families == 0:
        return {}

    if num_families == 1:
        # Only one family - gets all slots
        return {families[0]: target_count}

    # Calculate fair distribution with max 2:1 ratio
    # With N families and max 2:1 ratio:
    # If we have 2 families and 12 slots: max is 8:4 (2:1)
    # If we have 3 families and 12 slots: could be 6:3:3 or 5:4:3, etc.

    # Start with equal distribution
    base_per_family = target_count // num_families
    remainder = target_count % num_families

    # Distribute slots, giving extras to families with more current colors
    sorted_families = sorted(families, key=lambda f: family_counts[f], reverse=True)

    target_slots = {}
    for i, fam in enumerate(sorted_families):
        target_slots[fam] = base_per_family + (1 if i < remainder else 0)

    # Enforce max 2:1 ratio
    # Find min and max, adjust if ratio > 2:1
    for _ in range(10):  # Iterate to converge
        min_count = min(target_slots.values())
        max_count = max(target_slots.values())

        if max_count <= min_count * 2:
            break  # Ratio is OK

        # Find families with max count and reduce them
        # Give to families with min count
        max_families = [f for f, c in target_slots.items() if c == max_count]
        min_families = [f for f, c in target_slots.items() if c == min_count]

        if max_families and min_families:
            target_slots[max_families[0]] -= 1
            target_slots[min_families[0]] += 1

    return target_slots


def _fill_missing_colors(
    colors: List[ColorInfo],
    target_count: int,
    method: "PaletteMethod",
) -> List[ColorInfo]:
    """
    Fill missing color slots with method-appropriate variations.

    CRITICAL: Respects 2:1 max ratio between family slot counts!
    Distributes slots fairly among families, then fills each family
    with variations up to its target.

    Args:
        colors: Current list of colors.
        target_count: Desired number of colors.
        method: Palette method (determines variation type).

    Returns:
        Extended list with generated color variations.
    """
    if len(colors) >= target_count:
        return colors[:target_count]

    result = list(colors)

    # Count current families
    family_colors: Dict[str, List[ColorInfo]] = {}
    for c in colors:
        fam = _get_color_family(c.rgb)
        if fam not in ("gray", "extreme"):
            if fam not in family_colors:
                family_colors[fam] = []
            family_colors[fam].append(c)

    if not family_colors:
        return result  # No chromatic colors to work with

    # Calculate target slots per family (respecting 2:1 ratio)
    current_counts = {f: len(cols) for f, cols in family_colors.items()}
    target_slots = _calculate_balanced_slot_distribution(current_counts, target_count)

    # Determine variation strategy based on method
    if method == PaletteMethod.GLOW or method == PaletteMethod.LUMINOUS:
        variation_types = ["lighter", "lighter", "saturated"]
    elif method == PaletteMethod.SATURATED or method == PaletteMethod.INTENSIFY:
        variation_types = ["saturated", "darker", "saturated"]
    elif method == PaletteMethod.DIVERSE:
        variation_types = ["lighter", "darker", "saturated"]
    else:  # STANDARD
        variation_types = ["lighter", "darker", "desaturated"]

    # Fill each family up to its target slot count
    for family, target in target_slots.items():
        family_result_count = sum(1 for c in result if _get_color_family(c.rgb) == family)

        if family_result_count >= target:
            continue  # Already have enough for this family

        base_colors = family_colors.get(family, [])
        if not base_colors:
            continue

        slots_to_fill = target - family_result_count
        variation_index = 0
        color_cycle_index = 0
        strength = 0.15
        attempts = 0
        max_attempts = slots_to_fill * 20  # Safety limit

        while slots_to_fill > 0 and attempts < max_attempts:
            attempts += 1
            base_color = base_colors[color_cycle_index % len(base_colors)]
            var_type = variation_types[variation_index % len(variation_types)]

            # Generate variation
            new_rgb = _generate_color_variation(base_color.rgb, var_type, strength)

            # Check if this color is too similar to existing ones
            is_unique = True
            for existing in result:
                dist = _color_distance_hsl(new_rgb, existing.rgb)
                if dist < 0.08:  # Too similar
                    is_unique = False
                    break

            if is_unique:
                new_color = ColorInfo.from_rgb(
                    new_rgb[0], new_rgb[1], new_rgb[2],
                    base_color.percentage * 0.5,
                    index=len(result) + 1,
                )
                result.append(new_color)
                slots_to_fill -= 1

            color_cycle_index += 1
            if color_cycle_index >= len(base_colors):
                color_cycle_index = 0
                variation_index += 1
                strength += 0.08

    return result


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
    Select a diverse subset of colors maximizing contrast across families.

    ALWAYS starts with family anchors (one per family present) to guarantee
    broad hue diversity. Then fills remaining slots with the most distant colors.

    Args:
        colors: List of candidate colors.
        target_count: Number of colors to select.
        min_saturation: Minimum saturation for chromatic colors.

    Returns:
        List of selected diverse colors.
    """
    if len(colors) <= target_count:
        return colors

    # Separate valid chromatic colors from invalid ones
    chromatic = []
    low_saturation = []
    invalid = []

    for c in colors:
        h, s, l = _get_hsl(c.rgb)
        # Filter by name - NO grays in the palette!
        if _is_invalid_color_by_name(c.name):
            invalid.append(c)
        elif s >= min_saturation:
            chromatic.append(c)
        elif s >= 15.0:  # Slightly desaturated but still colorful
            low_saturation.append(c)
        else:
            invalid.append(c)

    if not chromatic:
        chromatic = low_saturation
        low_saturation = []

    if not chromatic:
        # Fallback: use all colors sorted by saturation
        all_sorted = sorted(colors, key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
        return all_sorted[:target_count]

    # STEP 1: Start with family anchors (one most vibrant per family)
    # This GUARANTEES broad hue diversity
    family_anchors = _get_family_anchors(chromatic)
    selected = family_anchors[:target_count]

    if len(selected) >= target_count:
        return selected

    # STEP 2: Fill remaining with most distant colors (greedy)
    remaining = [c for c in chromatic if c not in selected]

    while len(selected) < target_count and remaining:
        best_color = None
        best_score = -1

        for candidate in remaining:
            # Minimum distance to any selected color
            min_dist = min(
                _color_distance_hsl(candidate.rgb, sel.rgb)
                for sel in selected
            )
            # Bonus for saturation
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

    # STEP 3: If still not enough, add low saturation colors
    while len(selected) < target_count and low_saturation:
        low_saturation.sort(key=lambda c: _get_color_saturation(*c.rgb), reverse=True)
        selected.append(low_saturation.pop(0))

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
        self._debug_log: List[str] = []  # Debug logging for analysis

    def _log(self, message: str) -> None:
        """Add a message to the debug log."""
        self._debug_log.append(message)

    def get_debug_log(self) -> str:
        """Get the complete debug log as a string."""
        return "\n".join(self._debug_log)

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
        # Use fixed random state for reproducible sampling
        # This ensures palette doesn't change when only sorting/contour settings change
        rng = np.random.RandomState(self.settings.random_state)

        max_samples = 50000
        if len(pixels) > max_samples:
            indices = rng.choice(len(pixels), max_samples, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels

        # ALWAYS use same number of clusters for consistent analysis
        # Methods only SELECT from analyzed colors - analysis must be identical!
        total_clusters = 48  # Fine-grained analysis for all methods

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
        chromatic_colors = []  # Colors with good saturation and valid names
        chromatic_colors_glow = []  # Extended pool for GLOW/LUMINOUS (more permissive)

        for i, color in enumerate(all_cluster_colors):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            pct = percentages.get(i, 0.0)

            # Filter out pure black, white, and grays (covered by shades)
            if _is_pure_black_white_gray(r, g, b):
                continue

            color_info = ColorInfo.from_rgb(r, g, b, round(pct, 2), index=i + 1)
            all_colors.append(color_info)

            # Skip extreme lightness for chromatic pools (shades layer handles these)
            # BUT keep them in all_colors for fallback/posterization
            if _is_extreme_lightness(r, g, b):
                continue

            # Track chromatic colors (good saturation, valid name - NOT gray/white/black)
            # Standard threshold for most methods
            if not _is_low_saturation(r, g, b, threshold=20.0) and not _is_invalid_color_by_name(color_info.name):
                chromatic_colors.append(color_info)

            # Extended pool for GLOW/LUMINOUS: use dynamic thresholds that allow
            # lighter colors with lower saturation (they're still chromatic!)
            if _is_chromatic_dynamic(r, g, b, for_glow=True) and not _is_invalid_color_by_name(color_info.name):
                chromatic_colors_glow.append(color_info)

        # =====================================================================
        # ANALYZE IMAGE: Get statistics for each color family present
        # =====================================================================
        family_stats = _get_family_stats(chromatic_colors)
        present_families = list(family_stats.keys())
        num_families = len(present_families)

        # Clear and start debug log
        self._debug_log = []
        self._log("=" * 60)
        self._log("FARBANALYSE DEBUG LOG")
        self._log("=" * 60)
        self._log(f"\nMethode: {self.settings.palette_method.value}")
        self._log(f"Zielanzahl Farben: {self.settings.num_colors}")
        self._log(f"\n--- K-MEANS CLUSTERING ---")
        self._log(f"Cluster extrahiert: {len(all_cluster_colors)}")
        self._log(f"Alle Farben (nach Filter): {len(all_colors)}")
        self._log(f"Chromatische Farben: {len(chromatic_colors)}")

        self._log(f"\n--- FARBFAMILIEN IM BILD ---")
        self._log(f"Anzahl Familien gefunden: {num_families}")
        for fam in present_families:
            stats = family_stats[fam]
            self._log(f"  {fam.upper()}: {len(stats['colors'])} Farben, "
                     f"Fläche: {stats['area']:.1f}%, "
                     f"Max Sättigung: {stats['max_saturation']:.0f}%, "
                     f"Anchor: {stats['anchor'].name if stats['anchor'] else 'None'}")

        # Log each chromatic color with its family classification
        self._log(f"\n--- ALLE CHROMATISCHEN FARBEN ({len(chromatic_colors)}) ---")
        for c in chromatic_colors:
            h, s, l = _get_hsl(c.rgb)
            fam = _get_color_family(c.rgb)
            self._log(f"  {c.name}: RGB{c.rgb}, HSL({h:.0f}°, {s:.0f}%, {l:.0f}%), "
                     f"Familie={fam}, Fläche={c.percentage:.1f}%")

        # =====================================================================
        # METHOD-SPECIFIC COLOR SELECTION
        # All methods: First fill one color per family, then add more by method
        # =====================================================================

        colors = []
        num_colors = self.settings.num_colors
        self._log(f"\n--- FARBAUSWAHL STARTET ---")

        if self.settings.palette_method == PaletteMethod.STANDARD:
            # -----------------------------------------------------------------
            # STANDARD: Fill by AREA (most dominant family first)
            # -----------------------------------------------------------------
            ranked = _rank_families_by_criterion(family_stats, 'area')

            # First pass: one color per family (most dominant families first)
            for family, anchor in ranked[:num_colors]:
                colors.append(anchor)

            # Second pass: fill remaining slots cycling through families
            if len(colors) < num_colors:
                for family, _ in ranked:
                    if len(colors) >= num_colors:
                        break
                    # Get second-most dominant color in this family
                    family_colors_sorted = sorted(
                        family_stats[family]['colors'],
                        key=lambda c: c.percentage,
                        reverse=True
                    )
                    for c in family_colors_sorted:
                        if c not in colors:
                            colors.append(c)
                            break
                    if len(colors) >= num_colors:
                        break

        elif self.settings.palette_method == PaletteMethod.SATURATED:
            # -----------------------------------------------------------------
            # SATURATED: Fill by SATURATION (most saturated family first)
            # -----------------------------------------------------------------
            ranked = _rank_families_by_criterion(family_stats, 'saturation')

            # First pass: one color per family
            for family, anchor in ranked[:num_colors]:
                colors.append(anchor)

            # Second pass: fill with second-most saturated per family
            if len(colors) < num_colors:
                for family, _ in ranked:
                    if len(colors) >= num_colors:
                        break
                    family_colors_sorted = sorted(
                        family_stats[family]['colors'],
                        key=lambda c: _get_color_saturation(*c.rgb),
                        reverse=True
                    )
                    for c in family_colors_sorted:
                        if c not in colors:
                            colors.append(c)
                            break

        elif self.settings.palette_method == PaletteMethod.GLOW:
            # -----------------------------------------------------------------
            # GLOW: Fill by LIGHTNESS (lightest family first)
            # -----------------------------------------------------------------
            # Use glow pool for lighter colors
            glow_stats = _get_family_stats(chromatic_colors_glow if chromatic_colors_glow else chromatic_colors)
            ranked = _rank_families_by_criterion(glow_stats, 'lightness')

            # First pass: one color per family (lightest families first)
            for family, anchor in ranked[:num_colors]:
                colors.append(anchor)

            # Second pass: fill with second-lightest per family
            if len(colors) < num_colors:
                for family, _ in ranked:
                    if len(colors) >= num_colors:
                        break
                    if family in glow_stats:
                        family_colors_sorted = sorted(
                            glow_stats[family]['colors'],
                            key=lambda c: _get_color_lightness(*c.rgb),
                            reverse=True
                        )
                        for c in family_colors_sorted:
                            if c not in colors:
                                colors.append(c)
                                break

        elif self.settings.palette_method == PaletteMethod.INTENSIFY:
            # -----------------------------------------------------------------
            # INTENSIFY: Fill by COMPLEMENTARY PAIRS for maximum contrast
            # Dominance = intensity_rank × area_rank (lower = more dominant)
            # Complementary pairs scored by combined dominance
            # CRITICAL: Track family counts and enforce 2:1 max ratio DURING fill
            # -----------------------------------------------------------------
            self._log(f"\n--- INTENSIFY METHODE ---")
            pair_scores = _get_complementary_pair_scores(family_stats)
            used_families = set()
            family_slot_counts = {}  # Track how many slots each family has

            self._log(f"Komplementärpaare (nach Score sortiert):")
            for f1, f2, score, a1, a2 in pair_scores:
                self._log(f"  {f1} ↔ {f2}: Score={score:.1f}, "
                         f"Anchors: {a1.name if a1 else 'None'} / {a2.name if a2 else 'None'}")

            def can_add_family(fam: str) -> bool:
                """Check if we can add another color from this family (2:1 rule)."""
                if fam not in family_slot_counts or family_slot_counts[fam] == 0:
                    return True  # First color from family always OK
                current_count = family_slot_counts.get(fam, 0)
                # Find minimum count among other families in palette
                other_counts = [c for f, c in family_slot_counts.items() if f != fam and c > 0]
                if not other_counts:
                    return True  # Only one family so far
                min_other = min(other_counts)
                # Can add if it won't exceed 2:1 ratio
                return (current_count + 1) <= min_other * 2

            def add_color(c: "ColorInfo", fam: str, reason: str = "") -> bool:
                """Add color if family balance allows."""
                can_add = can_add_family(fam)
                if not can_add:
                    self._log(f"  BLOCKIERT: {c.name} ({fam}) - 2:1 Regel verletzt! "
                             f"Counts: {dict(family_slot_counts)}")
                    return False
                colors.append(c)
                family_slot_counts[fam] = family_slot_counts.get(fam, 0) + 1
                self._log(f"  + Slot {len(colors)}: {c.name} ({fam}) {reason}")
                return True

            self._log(f"\nPhase 1: Fülle aus Komplementärpaaren...")
            # Fill slots from complementary pairs (lighter color first in each pair)
            for f1, f2, score, anchor1, anchor2 in pair_scores:
                if len(colors) >= num_colors:
                    break

                # Add lighter color of pair first
                if f1 not in used_families and len(colors) < num_colors:
                    if add_color(anchor1, f1, f"[Paar {f1}↔{f2}, heller]"):
                        used_families.add(f1)

                # Then add the complementary color
                if f2 not in used_families and len(colors) < num_colors:
                    if add_color(anchor2, f2, f"[Paar {f1}↔{f2}, dunkler]"):
                        used_families.add(f2)

            self._log(f"\nPhase 2: Fülle verbleibende Familien...")
            # Fill remaining families not in pairs
            for family in present_families:
                if len(colors) >= num_colors:
                    break
                if family not in used_families:
                    if add_color(family_stats[family]['anchor'], family, "[nicht in Paar]"):
                        used_families.add(family)

            self._log(f"\nPhase 3: Fülle restliche Slots fair verteilt...")
            self._log(f"  Bisherige Counts: {dict(family_slot_counts)}")
            # Fill remaining slots - cycle through ALL families fairly
            # When 2:1 rule blocks, generate VARIATIONS of underrepresented families
            if len(colors) < num_colors:
                max_passes = 15

                for pass_num in range(max_passes):
                    if len(colors) >= num_colors:
                        break
                    self._log(f"  Pass {pass_num + 1}:")
                    # Get families sorted by slot count (ascending - least filled first)
                    sorted_by_count = sorted(
                        present_families,
                        key=lambda f: family_slot_counts.get(f, 0)
                    )
                    added_any = False

                    for family in sorted_by_count:
                        if len(colors) >= num_colors:
                            break
                        if not can_add_family(family):
                            continue

                        # Try to find an unused color from this family
                        family_colors_sorted = sorted(
                            family_stats[family]['colors'],
                            key=lambda c: _get_color_saturation(*c.rgb) * _get_color_lightness(*c.rgb),
                            reverse=True
                        )
                        found_color = False
                        for c in family_colors_sorted:
                            if c not in colors:
                                colors.append(c)
                                family_slot_counts[family] = family_slot_counts.get(family, 0) + 1
                                self._log(f"    + Slot {len(colors)}: {c.name} ({family})")
                                added_any = True
                                found_color = True
                                break

                        # If no unused color available, generate a variation
                        if not found_color:
                            base_colors = [c for c in colors if _get_color_family(c.rgb) == family]
                            if base_colors:
                                base = base_colors[-1]
                                for var_type in ["lighter", "darker", "saturated"]:
                                    var_rgb = _generate_color_variation(base.rgb, var_type, 0.2 + pass_num * 0.05)
                                    is_unique = all(
                                        _color_distance_hsl(var_rgb, c.rgb) >= 0.07
                                        for c in colors
                                    )
                                    if is_unique:
                                        new_color = ColorInfo.from_rgb(
                                            var_rgb[0], var_rgb[1], var_rgb[2],
                                            base.percentage * 0.5,
                                            len(colors) + 1
                                        )
                                        colors.append(new_color)
                                        family_slot_counts[family] = family_slot_counts.get(family, 0) + 1
                                        self._log(f"    + Slot {len(colors)}: {new_color.name} ({family}) [Var. von {base.name}]")
                                        added_any = True
                                        break

                    # If 2:1 blocks all AND no variations were added, generate for lowest-count families
                    if not added_any:
                        self._log(f"    Alle blockiert - generiere Variationen für min-count Familien...")
                        min_count = min(family_slot_counts.values()) if family_slot_counts else 0
                        for family in sorted_by_count:
                            if len(colors) >= num_colors:
                                break
                            current = family_slot_counts.get(family, 0)
                            if current > min_count:
                                continue  # Only add to families with minimum count
                            base_colors = [c for c in colors if _get_color_family(c.rgb) == family]
                            if base_colors:
                                base = base_colors[-1]
                                for var_type, strength in [("lighter", 0.25), ("darker", 0.25), ("saturated", 0.3)]:
                                    var_rgb = _generate_color_variation(base.rgb, var_type, strength)
                                    is_unique = all(
                                        _color_distance_hsl(var_rgb, c.rgb) >= 0.05
                                        for c in colors
                                    )
                                    if is_unique:
                                        new_color = ColorInfo.from_rgb(
                                            var_rgb[0], var_rgb[1], var_rgb[2],
                                            base.percentage * 0.5,
                                            len(colors) + 1
                                        )
                                        colors.append(new_color)
                                        family_slot_counts[family] = family_slot_counts.get(family, 0) + 1
                                        self._log(f"    + Slot {len(colors)}: {new_color.name} ({family}) [Var. {var_type}]")
                                        added_any = True
                                        break
                            if added_any:
                                break
                        if not added_any:
                            self._log(f"    Keine weiteren Variationen möglich")
                            break

        elif self.settings.palette_method == PaletteMethod.DIVERSE:
            # -----------------------------------------------------------------
            # DIVERSE: Fill by COLOR WHEEL SPREAD (360°/N separation)
            # Start with most dominant color, then find colors at optimal angles
            # -----------------------------------------------------------------
            # Score colors by area × luminance × saturation
            def diverse_score(c):
                return c.percentage * _get_color_lightness(*c.rgb) * _get_color_saturation(*c.rgb) / 10000.0

            all_chromatic_scored = sorted(chromatic_colors, key=diverse_score, reverse=True)

            if all_chromatic_scored:
                # First color: highest score
                first_color = all_chromatic_scored[0]
                colors.append(first_color)
                first_hue = _get_hsl(first_color.rgb)[0]

                # Calculate optimal spread angle
                optimal_spread = _get_optimal_hue_spread(num_colors)

                # Find colors at optimal angles
                for i in range(1, num_colors):
                    if len(colors) >= num_colors:
                        break

                    target_angle = (first_hue + i * optimal_spread) % 360

                    # Find best color near target angle (not already selected)
                    best_color = None
                    best_score = -1

                    for c in chromatic_colors:
                        if c in colors:
                            continue
                        c_hue = _get_hsl(c.rgb)[0]
                        # Circular distance to target
                        angle_diff = min(abs(c_hue - target_angle), 360 - abs(c_hue - target_angle))
                        # Score: closer to target angle + higher diverse_score
                        angle_score = 1.0 - (angle_diff / 180.0)  # 1.0 = exact, 0.0 = opposite
                        combined_score = angle_score * 0.7 + diverse_score(c) * 0.3

                        if combined_score > best_score:
                            best_score = combined_score
                            best_color = c

                    if best_color:
                        colors.append(best_color)

        elif self.settings.palette_method == PaletteMethod.LUMINOUS:
            # -----------------------------------------------------------------
            # LUMINOUS: Average of INTENSIFY and GLOW approaches
            # Calculate both, then blend/merge the results
            # -----------------------------------------------------------------
            # Get INTENSIFY colors
            intensify_colors = []
            pair_scores = _get_complementary_pair_scores(family_stats)
            used_families = set()
            for f1, f2, score, anchor1, anchor2 in pair_scores:
                if len(intensify_colors) >= num_colors:
                    break
                if f1 not in used_families:
                    intensify_colors.append(anchor1)
                    used_families.add(f1)
                if f2 not in used_families and len(intensify_colors) < num_colors:
                    intensify_colors.append(anchor2)
                    used_families.add(f2)
            for family in present_families:
                if len(intensify_colors) >= num_colors:
                    break
                if family not in used_families:
                    intensify_colors.append(family_stats[family]['anchor'])

            # Get GLOW colors
            glow_stats = _get_family_stats(chromatic_colors_glow if chromatic_colors_glow else chromatic_colors)
            glow_ranked = _rank_families_by_criterion(glow_stats, 'lightness')
            glow_colors = [anchor for _, anchor in glow_ranked[:num_colors]]

            # Merge: for each family, average the colors from both methods
            family_merged = {}
            for c in intensify_colors + glow_colors:
                family = _get_color_family(c.rgb)
                if family not in family_merged:
                    family_merged[family] = []
                family_merged[family].append(c)

            # Create merged palette
            for family, family_colors_list in family_merged.items():
                if len(colors) >= num_colors:
                    break
                if len(family_colors_list) == 1:
                    colors.append(family_colors_list[0])
                else:
                    # Blend colors from same family
                    avg_rgb = _blend_colors(family_colors_list[0], family_colors_list[1])
                    # Find closest actual color to this blend
                    closest = min(
                        chromatic_colors,
                        key=lambda c: _color_distance_hsl(c.rgb, avg_rgb)
                    )
                    if closest not in colors:
                        colors.append(closest)
                    elif family_colors_list[0] not in colors:
                        colors.append(family_colors_list[0])

        else:
            # Fallback: use family anchors
            family_anchors = _get_family_anchors(chromatic_colors)
            colors = family_anchors[:num_colors]

        # =====================================================================
        # POST-PROCESSING: Balance family distribution
        # If one family has 3x more colors than next, cluster and redistribute
        # =====================================================================
        self._log(f"\n--- VOR BALANCIERUNG ---")
        pre_balance_counts = {}
        for c in colors:
            fam = _get_color_family(c.rgb)
            pre_balance_counts[fam] = pre_balance_counts.get(fam, 0) + 1
        self._log(f"Familien-Verteilung: {pre_balance_counts}")
        for i, c in enumerate(colors):
            fam = _get_color_family(c.rgb)
            self._log(f"  Slot {i+1}: {c.name} ({fam})")

        colors = _balance_family_distribution(
            colors, family_stats, self.settings.palette_method, chromatic_colors
        )

        self._log(f"\n--- NACH BALANCIERUNG ---")
        post_balance_counts = {}
        for c in colors:
            fam = _get_color_family(c.rgb)
            post_balance_counts[fam] = post_balance_counts.get(fam, 0) + 1
        self._log(f"Familien-Verteilung: {post_balance_counts}")
        for i, c in enumerate(colors):
            fam = _get_color_family(c.rgb)
            self._log(f"  Slot {i+1}: {c.name} ({fam})")

        # =====================================================================
        # FILL MISSING SLOTS: Ensure all requested colors are provided
        # CRITICAL: Must respect 2:1 max ratio between families!
        # =====================================================================
        self._log(f"\n--- SLOT-BEFÜLLUNG ---")
        self._log(f"Aktuelle Anzahl: {len(colors)}, Ziel: {self.settings.num_colors}")

        # Calculate current family counts
        current_family_counts = {}
        for c in colors:
            fam = _get_color_family(c.rgb)
            if fam not in ("gray", "extreme"):
                current_family_counts[fam] = current_family_counts.get(fam, 0) + 1

        # Calculate target distribution respecting 2:1 ratio
        target_slots = _calculate_balanced_slot_distribution(
            current_family_counts, self.settings.num_colors
        )
        self._log(f"Ziel-Verteilung (max 2:1 Ratio): {target_slots}")

        # Group remaining colors by family
        remaining_by_family: Dict[str, List[ColorInfo]] = {}
        for c in all_colors:
            if c not in colors:
                fam = _get_color_family(c.rgb)
                if fam not in ("gray", "extreme"):
                    if fam not in remaining_by_family:
                        remaining_by_family[fam] = []
                    remaining_by_family[fam].append(c)

        # Sort each family's remaining colors by saturation (most saturated first)
        for fam in remaining_by_family:
            remaining_by_family[fam].sort(
                key=lambda c: _get_color_saturation(*c.rgb), reverse=True
            )

        # Fill slots respecting target distribution
        self._log(f"\nFülle Slots nach Ziel-Verteilung:")
        while len(colors) < self.settings.num_colors:
            added_any = False

            # Find family that needs most additional slots
            for fam, target in sorted(target_slots.items(),
                                      key=lambda x: x[1] - current_family_counts.get(x[0], 0),
                                      reverse=True):
                current = current_family_counts.get(fam, 0)
                if current < target and fam in remaining_by_family and remaining_by_family[fam]:
                    # Add next color from this family
                    next_color = remaining_by_family[fam].pop(0)
                    colors.append(next_color)
                    current_family_counts[fam] = current + 1
                    self._log(f"  + {next_color.name} ({fam}) [Slot {len(colors)}/{self.settings.num_colors}]")
                    added_any = True

                    if len(colors) >= self.settings.num_colors:
                        break

            if not added_any:
                break  # No more colors to add from remaining

        # If still missing, generate variations (respecting ratios)
        if len(colors) < self.settings.num_colors:
            self._log(f"\nGeneriere Variationen ({len(colors)} → {self.settings.num_colors})...")
            colors = _fill_missing_colors(
                colors,
                self.settings.num_colors,
                self.settings.palette_method,
            )

        # =====================================================================
        # FINAL BALANCE CHECK: Enforce 2:1 ratio on completed palette
        # Uses cascading approach: balance most imbalanced pair first
        # =====================================================================
        self._log(f"\n--- FINALE BALANCIERUNG ---")
        colors = _final_balance_palette(colors, family_stats, chromatic_colors, self._log)

        # Log final distribution
        self._log(f"\n--- FINALE PALETTE ({len(colors)} Farben) ---")
        final_family_counts = {}
        for c in colors:
            fam = _get_color_family(c.rgb)
            final_family_counts[fam] = final_family_counts.get(fam, 0) + 1
        self._log(f"Finale Familien-Verteilung: {final_family_counts}")
        for i, c in enumerate(colors):
            fam = _get_color_family(c.rgb)
            self._log(f"  Slot {i+1}: {c.name} ({fam})")

        # =====================================================================
        # GRAY AVOIDANCE: Replace any remaining gray-named colors
        # Blend invalid colors toward valid neighbors until valid
        # =====================================================================
        final_colors = []
        for color_info in colors:
            if _is_invalid_color_by_name(color_info.name):
                # Try to find a valid replacement by blending
                valid_rgb = _find_valid_replacement(color_info.rgb, chromatic_colors)
                if valid_rgb:
                    # Create new ColorInfo with blended color
                    new_info = ColorInfo.from_rgb(
                        valid_rgb[0], valid_rgb[1], valid_rgb[2],
                        color_info.percentage, color_info.index
                    )
                    final_colors.append(new_info)
                else:
                    # No valid replacement found, keep original (should be rare)
                    final_colors.append(color_info)
            else:
                final_colors.append(color_info)

        colors = final_colors

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

            # Analyze pixel properties for special handling
            max_vals = np.max(chunk, axis=1)
            min_vals = np.min(chunk, axis=1)
            lightness_sum = max_vals + min_vals
            color_range = max_vals - min_vals

            # 1. Extreme lightness (L>95% or L<5%) → pure white/black (shades layer)
            # L > 95% means (max + min) / 510 > 0.95, i.e. max + min > 484
            # L < 5% means (max + min) / 510 < 0.05, i.e. max + min < 26
            is_almost_white = lightness_sum > 484
            is_almost_black = lightness_sum < 26

            # 2. Gray pixels (low saturation) → keep as gray (shades layer)
            # Saturation is ~0 when max - min is small
            # Threshold ~25-30 corresponds to roughly S < 10-12%
            is_gray = color_range < 28

            # Calculate gray values for gray pixels (lightness as grayscale)
            gray_values = (lightness_sum / 2).astype(np.uint8)

            # Calculate squared Euclidean distance to each palette color
            # Shape: (chunk_size, num_palette_colors)
            distances = np.sum(
                (chunk[:, np.newaxis, :] - palette_float[np.newaxis, :, :]) ** 2,
                axis=2
            )

            # Find nearest palette color for each pixel
            nearest_indices = np.argmin(distances, axis=1)
            chunk_posterized = self._palette_colors[nearest_indices]

            # Override special pixels:
            # Gray pixels → their grayscale value (NOT a palette color!)
            # This is key: gray areas belong to shades layer, not color palette
            gray_mask = is_gray & ~is_almost_white & ~is_almost_black
            chunk_posterized[gray_mask, 0] = gray_values[gray_mask]
            chunk_posterized[gray_mask, 1] = gray_values[gray_mask]
            chunk_posterized[gray_mask, 2] = gray_values[gray_mask]

            # Extreme lightness → pure white/black
            chunk_posterized[is_almost_white] = [255, 255, 255]
            chunk_posterized[is_almost_black] = [0, 0, 0]

            posterized_pixels[start:end] = chunk_posterized

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
