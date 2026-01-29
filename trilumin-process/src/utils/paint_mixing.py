"""
Paint mixing utilities for oil painting color recipes.

This module provides structures and functions for converting RGB colors
to oil paint mixing recipes using a defined pigment palette.

Note: Full implementation of mixing algorithms is planned for future versions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Opacity(Enum):
    """Paint opacity/transparency levels."""
    TRANSPARENT = "transparent"
    SEMI_TRANSPARENT = "semi-transparent"
    SEMI_OPAQUE = "semi-opaque"
    OPAQUE = "opaque"


class TintingStrength(Enum):
    """Tinting strength of pigments."""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very-strong"


@dataclass
class OilPigment:
    """
    Represents an oil paint pigment.

    Attributes:
        name: German name of the pigment.
        name_en: English name of the pigment.
        rgb: Approximate RGB value of the pure pigment.
        opacity: Opacity level.
        tinting_strength: How strongly it affects mixtures.
        pigment_code: Standard pigment code (e.g., PW6, PB29).
        warm: True if warm, False if cool.
    """
    name: str
    name_en: str
    rgb: Tuple[int, int, int]
    opacity: Opacity
    tinting_strength: TintingStrength
    pigment_code: str
    warm: bool = True


# Standard oil painting palette
STANDARD_PALETTE: Dict[str, OilPigment] = {
    # Whites
    "titanweiss": OilPigment(
        name="Titanweiß",
        name_en="Titanium White",
        rgb=(255, 255, 255),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.VERY_STRONG,
        pigment_code="PW6",
        warm=False,
    ),
    "zinkweiss": OilPigment(
        name="Zinkweiß",
        name_en="Zinc White",
        rgb=(252, 252, 252),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="PW4",
        warm=False,
    ),

    # Blacks
    "elfenbeinschwarz": OilPigment(
        name="Elfenbeinschwarz",
        name_en="Ivory Black",
        rgb=(35, 31, 32),
        opacity=Opacity.SEMI_OPAQUE,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PBk9",
        warm=True,
    ),
    "lampenschwarz": OilPigment(
        name="Lampenschwarz",
        name_en="Lamp Black",
        rgb=(28, 28, 28),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PBk6",
        warm=False,
    ),

    # Yellows
    "kadmiumgelb_hell": OilPigment(
        name="Kadmiumgelb hell",
        name_en="Cadmium Yellow Light",
        rgb=(255, 237, 0),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PY37",
        warm=True,
    ),
    "kadmiumgelb_mittel": OilPigment(
        name="Kadmiumgelb mittel",
        name_en="Cadmium Yellow Medium",
        rgb=(255, 211, 0),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PY37",
        warm=True,
    ),
    "neapelgelb": OilPigment(
        name="Neapelgelb",
        name_en="Naples Yellow",
        rgb=(250, 218, 94),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PY41",
        warm=True,
    ),
    "ocker_gelb": OilPigment(
        name="Gelber Ocker",
        name_en="Yellow Ochre",
        rgb=(204, 170, 102),
        opacity=Opacity.SEMI_OPAQUE,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PY43",
        warm=True,
    ),

    # Oranges
    "kadmiumorange": OilPigment(
        name="Kadmiumorange",
        name_en="Cadmium Orange",
        rgb=(255, 153, 0),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PO20",
        warm=True,
    ),

    # Reds
    "kadmiumrot_hell": OilPigment(
        name="Kadmiumrot hell",
        name_en="Cadmium Red Light",
        rgb=(227, 66, 52),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PR108",
        warm=True,
    ),
    "kadmiumrot_dunkel": OilPigment(
        name="Kadmiumrot dunkel",
        name_en="Cadmium Red Deep",
        rgb=(180, 32, 37),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PR108",
        warm=True,
    ),
    "alizarinkarmesin": OilPigment(
        name="Alizarin-Karmesin",
        name_en="Alizarin Crimson",
        rgb=(227, 38, 54),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.VERY_STRONG,
        pigment_code="PR83",
        warm=False,
    ),
    "venezianischrot": OilPigment(
        name="Venezianischrot",
        name_en="Venetian Red",
        rgb=(200, 90, 83),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PR101",
        warm=True,
    ),

    # Browns
    "umbra_gebrannt": OilPigment(
        name="Umbra gebrannt",
        name_en="Burnt Umber",
        rgb=(138, 89, 51),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PBr7",
        warm=True,
    ),
    "umbra_natur": OilPigment(
        name="Umbra natur",
        name_en="Raw Umber",
        rgb=(99, 81, 71),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="PBr7",
        warm=False,
    ),
    "siena_gebrannt": OilPigment(
        name="Siena gebrannt",
        name_en="Burnt Sienna",
        rgb=(160, 82, 45),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PBr7",
        warm=True,
    ),
    "siena_natur": OilPigment(
        name="Siena natur",
        name_en="Raw Sienna",
        rgb=(136, 120, 75),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="PY43",
        warm=True,
    ),
    "van_dyck_braun": OilPigment(
        name="Van-Dyck-Braun",
        name_en="Van Dyke Brown",
        rgb=(99, 59, 35),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="NBr8",
        warm=True,
    ),

    # Greens
    "chromoxidgruen": OilPigment(
        name="Chromoxidgrün",
        name_en="Chromium Oxide Green",
        rgb=(31, 78, 47),
        opacity=Opacity.OPAQUE,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PG17",
        warm=False,
    ),
    "viridian": OilPigment(
        name="Viridian",
        name_en="Viridian",
        rgb=(64, 130, 109),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="PG18",
        warm=False,
    ),
    "saftgruen": OilPigment(
        name="Saftgrün",
        name_en="Sap Green",
        rgb=(80, 125, 42),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PG36",
        warm=True,
    ),

    # Blues
    "ultramarinblau": OilPigment(
        name="Ultramarinblau",
        name_en="Ultramarine Blue",
        rgb=(65, 102, 245),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.STRONG,
        pigment_code="PB29",
        warm=True,
    ),
    "kobaltblau": OilPigment(
        name="Kobaltblau",
        name_en="Cobalt Blue",
        rgb=(0, 71, 171),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.MEDIUM,
        pigment_code="PB28",
        warm=False,
    ),
    "preussischblau": OilPigment(
        name="Preußischblau",
        name_en="Prussian Blue",
        rgb=(0, 49, 83),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.VERY_STRONG,
        pigment_code="PB27",
        warm=False,
    ),
    "coelinblau": OilPigment(
        name="Cölinblau",
        name_en="Cerulean Blue",
        rgb=(0, 123, 167),
        opacity=Opacity.SEMI_OPAQUE,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="PB35",
        warm=False,
    ),
    "phthaloblau": OilPigment(
        name="Phthaloblau",
        name_en="Phthalo Blue",
        rgb=(0, 15, 137),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.VERY_STRONG,
        pigment_code="PB15",
        warm=False,
    ),

    # Violets
    "kobaltviolett": OilPigment(
        name="Kobaltviolett",
        name_en="Cobalt Violet",
        rgb=(145, 92, 131),
        opacity=Opacity.SEMI_TRANSPARENT,
        tinting_strength=TintingStrength.WEAK,
        pigment_code="PV14",
        warm=True,
    ),
    "dioxazinviolett": OilPigment(
        name="Dioxazinviolett",
        name_en="Dioxazine Violet",
        rgb=(92, 0, 128),
        opacity=Opacity.TRANSPARENT,
        tinting_strength=TintingStrength.VERY_STRONG,
        pigment_code="PV23",
        warm=False,
    ),
}


@dataclass
class MixingRecipe:
    """
    A recipe for mixing a target color from pigments.

    Attributes:
        target_rgb: The target RGB color.
        pigments: Dict mapping pigment key to percentage (0-100).
        accuracy: How close this recipe matches the target (0-1).
        notes: Optional mixing notes or warnings.
    """
    target_rgb: Tuple[int, int, int]
    pigments: Dict[str, float]
    accuracy: float
    notes: Optional[str] = None


def get_available_pigments() -> List[str]:
    """Get list of all available pigment names."""
    return [p.name for p in STANDARD_PALETTE.values()]


def get_pigment_info(key: str) -> Optional[OilPigment]:
    """Get information about a specific pigment."""
    return STANDARD_PALETTE.get(key.lower().replace(" ", "_").replace("-", "_"))


def rgb_to_mixing_recipe(
    r: int,
    g: int,
    b: int,
    palette: Optional[Dict[str, OilPigment]] = None,
) -> MixingRecipe:
    """
    Convert an RGB color to a paint mixing recipe.

    NOTE: This is a placeholder implementation. Full color mixing simulation
    using Kubelka-Munk theory is planned for future versions.

    Args:
        r, g, b: Target RGB values (0-255).
        palette: Optional custom palette, uses STANDARD_PALETTE if None.

    Returns:
        MixingRecipe with approximate pigment percentages.
    """
    if palette is None:
        palette = STANDARD_PALETTE

    target = (r, g, b)

    # Placeholder: Simple nearest-neighbor matching
    # Real implementation would use subtractive color mixing simulation
    min_dist = float('inf')
    closest_pigment = None

    for key, pigment in palette.items():
        dist = sum((a - b) ** 2 for a, b in zip(target, pigment.rgb)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_pigment = key

    # Estimate if we need white to lighten
    avg_target = sum(target) / 3
    needs_white = avg_target > 180

    # Estimate if we need black to darken
    needs_black = avg_target < 60

    # Build simple recipe
    pigments = {}

    if needs_white:
        white_amount = min(80, (avg_target - 128) / 127 * 60)
        pigments["titanweiss"] = round(white_amount, 1)
        pigments[closest_pigment] = round(100 - white_amount, 1)
    elif needs_black:
        black_amount = min(50, (128 - avg_target) / 128 * 40)
        pigments["elfenbeinschwarz"] = round(black_amount, 1)
        pigments[closest_pigment] = round(100 - black_amount, 1)
    else:
        pigments[closest_pigment] = 100.0

    return MixingRecipe(
        target_rgb=target,
        pigments=pigments,
        accuracy=max(0.0, 1.0 - min_dist / 255),
        notes="Vereinfachte Schätzung - experimentelles Feature",
    )


def format_recipe(recipe: MixingRecipe) -> str:
    """
    Format a mixing recipe as human-readable string.

    Args:
        recipe: The mixing recipe.

    Returns:
        Formatted string describing the recipe.
    """
    lines = [f"Mischrezept für RGB({recipe.target_rgb[0]}, {recipe.target_rgb[1]}, {recipe.target_rgb[2]}):"]
    lines.append("-" * 40)

    for pigment_key, percentage in sorted(recipe.pigments.items(), key=lambda x: -x[1]):
        pigment = STANDARD_PALETTE.get(pigment_key)
        if pigment:
            lines.append(f"  {percentage:5.1f}%  {pigment.name}")
        else:
            lines.append(f"  {percentage:5.1f}%  {pigment_key}")

    lines.append("-" * 40)
    lines.append(f"Genauigkeit: {recipe.accuracy * 100:.0f}%")

    if recipe.notes:
        lines.append(f"Hinweis: {recipe.notes}")

    return "\n".join(lines)
