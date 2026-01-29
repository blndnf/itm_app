"""Image I/O utilities for loading and saving images."""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class ImageIO:
    """Handles loading and saving of images with various formats."""

    SUPPORTED_FORMATS = {
        "load": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
        "save": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
    }

    @classmethod
    def load_image(cls, file_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.

        Args:
            file_path: Path to the image file.

        Returns:
            Image as numpy array in BGR format, or None if loading failed.
        """
        if not os.path.exists(file_path):
            return None

        ext = Path(file_path).suffix.lower()
        if ext not in cls.SUPPORTED_FORMATS["load"]:
            return None

        # Read image with OpenCV (returns BGR format)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        return image

    @classmethod
    def load_image_rgb(cls, file_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file and convert to RGB.

        Args:
            file_path: Path to the image file.

        Returns:
            Image as numpy array in RGB format, or None if loading failed.
        """
        image = cls.load_image(file_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @classmethod
    def save_image(
        cls,
        image: np.ndarray,
        file_path: str,
        quality: int = 95,
        is_rgb: bool = False,
    ) -> bool:
        """
        Save an image to file.

        Args:
            image: Image as numpy array.
            file_path: Destination path for the image.
            quality: JPEG quality (1-100), default 95.
            is_rgb: If True, convert from RGB to BGR before saving.

        Returns:
            True if saving was successful, False otherwise.
        """
        ext = Path(file_path).suffix.lower()
        if ext not in cls.SUPPORTED_FORMATS["save"]:
            return False

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        # Convert RGB to BGR if needed
        if is_rgb and len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Set compression parameters based on format
        params = []
        if ext in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".png":
            # PNG compression level (0-9, higher = more compression)
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

        return cv2.imwrite(file_path, image, params)

    @classmethod
    def get_image_info(cls, file_path: str) -> Optional[dict]:
        """
        Get information about an image file.

        Args:
            file_path: Path to the image file.

        Returns:
            Dictionary with image info, or None if file cannot be read.
        """
        image = cls.load_image(file_path)
        if image is None:
            return None

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        return {
            "path": file_path,
            "width": width,
            "height": height,
            "channels": channels,
            "size_bytes": os.path.getsize(file_path),
            "format": Path(file_path).suffix.lower(),
        }

    @classmethod
    def resize_image(
        cls,
        image: np.ndarray,
        max_size: Tuple[int, int],
        maintain_aspect: bool = True,
    ) -> np.ndarray:
        """
        Resize an image to fit within max dimensions.

        Args:
            image: Input image as numpy array.
            max_size: Maximum (width, height) tuple.
            maintain_aspect: If True, maintain aspect ratio.

        Returns:
            Resized image.
        """
        height, width = image.shape[:2]
        max_width, max_height = max_size

        if maintain_aspect:
            # Calculate scaling factor
            scale = min(max_width / width, max_height / height)
            if scale >= 1.0:
                return image  # No upscaling needed

            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width = min(width, max_width)
            new_height = min(height, max_height)

        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

    @classmethod
    def get_file_filter(cls, mode: str = "load") -> str:
        """
        Get file filter string for file dialogs.

        Args:
            mode: Either "load" or "save".

        Returns:
            File filter string for Qt file dialogs.
        """
        formats = cls.SUPPORTED_FORMATS.get(mode, cls.SUPPORTED_FORMATS["load"])
        extensions = " ".join(f"*{ext}" for ext in formats)
        return f"Images ({extensions});;All Files (*.*)"
