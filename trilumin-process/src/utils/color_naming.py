"""Color naming utilities for converting RGB values to human-readable names."""

from typing import Tuple, Dict, List
import colorsys


# Extended color database with German names
COLOR_DATABASE: Dict[str, Tuple[int, int, int]] = {
    # Whites and Blacks
    "Weiß": (255, 255, 255),
    "Elfenbein": (255, 255, 240),
    "Schneeweiß": (255, 250, 250),
    "Cremeweiß": (255, 253, 208),
    "Schwarz": (0, 0, 0),
    "Anthrazit": (43, 43, 43),

    # Grays
    "Hellgrau": (211, 211, 211),
    "Silbergrau": (192, 192, 192),
    "Mittelgrau": (128, 128, 128),
    "Dunkelgrau": (64, 64, 64),
    "Schiefergrau": (112, 128, 144),
    "Warmgrau": (128, 118, 105),
    "Kaltgrau": (108, 118, 128),

    # Reds
    "Rot": (255, 0, 0),
    "Dunkelrot": (139, 0, 0),
    "Karminrot": (150, 0, 24),
    "Zinnoberrot": (227, 66, 52),
    "Korallenrot": (255, 127, 80),
    "Lachsrot": (250, 128, 114),
    "Terrakotta": (204, 78, 92),
    "Bordeaux": (109, 7, 26),
    "Weinrot": (114, 47, 55),
    "Rostrot": (183, 65, 14),
    "Indischrot": (205, 92, 92),

    # Oranges
    "Orange": (255, 165, 0),
    "Dunkelorange": (255, 140, 0),
    "Hellorange": (255, 200, 100),
    "Apricot": (251, 206, 177),
    "Pfirsich": (255, 218, 185),
    "Kupfer": (184, 115, 51),
    "Bernstein": (255, 191, 0),

    # Yellows
    "Gelb": (255, 255, 0),
    "Zitronengelb": (255, 247, 0),
    "Kadmiumgelb": (255, 246, 0),
    "Goldgelb": (255, 223, 0),
    "Safrangelb": (244, 196, 48),
    "Ocker": (204, 170, 102),
    "Dunkelocker": (150, 120, 60),
    "Hellgelb": (255, 255, 180),
    "Senfgelb": (205, 175, 0),
    "Neapelgelb": (250, 218, 94),

    # Greens
    "Grün": (0, 128, 0),
    "Hellgrün": (144, 238, 144),
    "Dunkelgrün": (0, 100, 0),
    "Olivgrün": (107, 142, 35),
    "Olivbraun": (128, 128, 0),
    "Moosgrün": (138, 154, 91),
    "Tannengrün": (32, 96, 61),
    "Smaragdgrün": (0, 158, 96),
    "Jadegrün": (0, 168, 107),
    "Mintgrün": (152, 255, 152),
    "Grasgrün": (124, 185, 75),
    "Chromoxidgrün": (31, 78, 47),
    "Seegrün": (46, 139, 87),
    "Türkisgrün": (0, 150, 136),

    # Blues
    "Blau": (0, 0, 255),
    "Hellblau": (173, 216, 230),
    "Dunkelblau": (0, 0, 139),
    "Ultramarinblau": (65, 102, 245),
    "Kobaltblau": (0, 71, 171),
    "Preußischblau": (0, 49, 83),
    "Cyanblau": (0, 255, 255),
    "Himmelblau": (135, 206, 235),
    "Stahlblau": (70, 130, 180),
    "Marineblau": (0, 0, 128),
    "Nachtblau": (25, 25, 112),
    "Eisblau": (175, 238, 238),
    "Taubenblau": (119, 158, 203),
    "Indigoblau": (75, 0, 130),
    "Coelinblau": (0, 123, 167),

    # Purples/Violets (extended with muted variants)
    "Violett": (138, 43, 226),
    "Lila": (128, 0, 128),
    "Magenta": (255, 0, 255),
    "Purpur": (128, 0, 128),
    "Lavendel": (230, 230, 250),
    "Flieder": (200, 162, 200),
    "Mauve": (224, 176, 255),
    "Pflaume": (142, 69, 133),
    "Aubergine": (97, 64, 81),
    "Dunkellila": (75, 0, 75),
    "Gedecktes Violett": (100, 80, 120),
    "Gedämpftes Lila": (120, 100, 130),
    "Blassviolett": (150, 130, 160),
    "Grauviolett": (130, 120, 145),
    "Dunkles Mauve": (90, 70, 95),
    "Rauchviolett": (115, 100, 125),

    # Browns
    "Braun": (139, 69, 19),
    "Hellbraun": (181, 137, 100),
    "Dunkelbraun": (92, 64, 51),
    "Siena gebrannt": (160, 82, 45),
    "Siena natur": (136, 120, 75),
    "Umbra gebrannt": (138, 89, 51),
    "Umbra natur": (99, 81, 71),
    "Kastanienbraun": (128, 0, 32),
    "Schokoladenbraun": (123, 63, 0),
    "Nussbraun": (127, 100, 80),
    "Caramel": (175, 120, 60),
    "Sepiabraun": (112, 66, 20),
    "Van-Dyck-Braun": (99, 59, 35),
    "Kaffeebraun": (75, 54, 33),

    # Pinks
    "Rosa": (255, 192, 203),
    "Pink": (255, 105, 180),
    "Altrosa": (188, 143, 143),
    "Hellrosa": (255, 182, 193),
    "Fuchsia": (255, 0, 255),
    "Malve": (200, 100, 150),

    # Beiges/Tans
    "Beige": (245, 245, 220),
    "Sand": (194, 178, 128),
    "Champagner": (247, 231, 206),
    "Khaki": (195, 176, 145),
    "Taupe": (72, 60, 50),
    "Leinen": (250, 240, 230),
}


def get_text_color_for_background(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Returns white for dark backgrounds, black for light backgrounds.

    Args:
        bg_color: RGB tuple (0-255).

    Returns:
        (255, 255, 255) for dark backgrounds, (0, 0, 0) for light.
    """
    r, g, b = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def int_to_roman(num: int) -> str:
    """
    Convert integer to Roman numeral.

    Args:
        num: Integer from 1 to 3999.

    Returns:
        Roman numeral string.
    """
    if num < 1 or num > 3999:
        return str(num)

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']

    result = ''
    for i, v in enumerate(val):
        while num >= v:
            result += syms[i]
            num -= v
    return result


def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculate weighted Euclidean distance between two RGB colors."""
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    # Weighted distance - human eye is more sensitive to green
    return ((r1 - r2) ** 2 * 0.3 +
            (g1 - g2) ** 2 * 0.59 +
            (b1 - b2) ** 2 * 0.11) ** 0.5


def _get_hsl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to HSL."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s * 100, l * 100


def _generate_descriptive_name(rgb: Tuple[int, int, int]) -> str:
    """
    Generate a descriptive color name based on HSL properties.

    Uses a very conservative gray threshold (s < 5) to ensure
    colors with even slight hue are named properly.

    Args:
        rgb: RGB tuple (0-255).

    Returns:
        Descriptive name like "gedecktes Violett" or "gedämpftes Blau".
    """
    h, s, l = _get_hsl(rgb)

    # Only truly achromatic colors (s < 5) are gray
    # Any color with s >= 5 has enough hue to be named as a color
    if s < 5:
        if l < 15:
            return "Schwarz"
        elif l < 30:
            return "Dunkelgrau"
        elif l < 50:
            return "Mittelgrau"
        elif l < 70:
            return "Hellgrau"
        elif l < 90:
            return "Sehr hellgrau"
        else:
            return "Weiß"

    # Determine base hue name (7 families + cyan for blue-green)
    if h < 15 or h >= 345:
        base = "Rot"
    elif h < 45:
        base = "Orange"
    elif h < 70:
        base = "Gelb"
    elif h < 150:
        base = "Grün"
    elif h < 195:
        base = "Türkis"
    elif h < 270:
        base = "Blau"
    elif h < 330:
        base = "Violett"
    else:
        base = "Magenta"

    # Add modifiers
    modifiers = []

    # Lightness modifier
    if l < 25:
        modifiers.append("dunkles")
    elif l < 40:
        modifiers.append("tiefes")
    elif l > 75:
        modifiers.append("helles")
    elif l > 60:
        modifiers.append("lichtes")

    # Saturation modifier
    if s < 30:
        modifiers.append("gedecktes")
    elif s < 50:
        modifiers.append("gedämpftes")
    elif s > 80:
        modifiers.append("leuchtendes")

    # Warm/Cool modifier based on hue
    if 20 < h < 70:  # Warm colors
        if "gedecktes" not in modifiers and "gedämpftes" not in modifiers:
            if len(modifiers) == 0:
                modifiers.append("warmes")
    elif 180 < h < 280:  # Cool colors
        if "gedecktes" not in modifiers and "gedämpftes" not in modifiers:
            if len(modifiers) == 0:
                modifiers.append("kühles")

    if modifiers:
        # Capitalize first modifier
        modifiers[0] = modifiers[0].capitalize()
        return " ".join(modifiers) + " " + base
    return base


def _is_gray_name(name: str) -> bool:
    """Check if a color name is a gray/white/black name."""
    gray_terms = [
        "grau", "weiß", "weiss", "schwarz", "anthrazit",
        "gray", "grey", "white", "black", "charcoal", "silver", "silber"
    ]
    name_lower = name.lower()
    return any(term in name_lower for term in gray_terms)


def rgb_to_name(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to the closest color name.

    Uses hue-aware matching: colors with significant saturation
    will NOT match to gray names, even if RGB distance is close.

    Args:
        r, g, b: Red, green, blue values (0-255).

    Returns:
        Human-readable color name in German.
    """
    rgb = (r, g, b)
    h, s, l = _get_hsl(rgb)

    # Determine if this color has significant hue (is chromatic)
    # Even low saturation (>8%) with distinct hue should NOT be gray
    is_chromatic = s > 8

    # Find closest match in database
    min_distance = float('inf')
    closest_name = None
    closest_chromatic_name = None
    min_chromatic_distance = float('inf')

    for name, color_rgb in COLOR_DATABASE.items():
        distance = _color_distance(rgb, color_rgb)

        if distance < min_distance:
            min_distance = distance
            closest_name = name

        # Also track closest chromatic (non-gray) match
        if not _is_gray_name(name) and distance < min_chromatic_distance:
            min_chromatic_distance = distance
            closest_chromatic_name = name

    # If the input color has hue (is chromatic), prefer chromatic names
    if is_chromatic:
        # If closest match is gray but color has hue, use chromatic match instead
        if _is_gray_name(closest_name) and closest_chromatic_name:
            # Only use chromatic if not too far off
            if min_chromatic_distance < 60:
                return closest_chromatic_name
            # Otherwise generate descriptive name (which uses hue)
            return _generate_descriptive_name(rgb)

    # If very close match (distance < 30), use database name
    if min_distance < 30:
        return closest_name

    # Otherwise generate descriptive name
    return _generate_descriptive_name(rgb)


def rgb_to_name_with_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB to name with hex code in parentheses.

    Args:
        r, g, b: Red, green, blue values (0-255).

    Returns:
        String like "Ultramarinblau (#4166F5)".
    """
    name = rgb_to_name(r, g, b)
    hex_code = f"#{r:02X}{g:02X}{b:02X}"
    return f"{name} ({hex_code})"


def is_gray(r: int, g: int, b: int, threshold: float = 12.0) -> bool:
    """
    Check if a color is a gray (low saturation).

    Args:
        r, g, b: RGB values (0-255).
        threshold: Saturation threshold percentage (default 12%).

    Returns:
        True if the color is considered gray.
    """
    _, s, _ = _get_hsl((r, g, b))
    return s < threshold


def get_color_category(r: int, g: int, b: int) -> str:
    """
    Get the general color category.

    Args:
        r, g, b: RGB values (0-255).

    Returns:
        Category name: "Grau", "Rot", "Orange", "Gelb", "Grün", "Blau", "Violett".
    """
    h, s, l = _get_hsl((r, g, b))

    if s < 12:
        return "Grau"

    if h < 15 or h >= 345:
        return "Rot"
    elif h < 45:
        return "Orange"
    elif h < 70:
        return "Gelb"
    elif h < 150:
        return "Grün"
    elif h < 270:
        return "Blau"
    else:
        return "Violett"
