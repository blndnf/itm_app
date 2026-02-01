"""Color analysis engine with multiple clustering methods.

This module provides a unified interface for different color extraction
algorithms, implementing a Strategy pattern for easy method switching.
"""

from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# Scikit-learn imports
from sklearn.cluster import MiniBatchKMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture


class AnalysisMethod(Enum):
    """Available color analysis methods."""
    KMEANS = "kmeans"
    MEDIAN_CUT = "median_cut"
    MEAN_SHIFT = "mean_shift"
    DBSCAN = "dbscan"
    OCTREE = "octree"
    HYBRID = "hybrid"
    GMM = "gmm"


# Method descriptions for tooltips
METHOD_DESCRIPTIONS = {
    AnalysisMethod.KMEANS: "Schneller Clustering-Algorithmus, guter Allrounder",
    AnalysisMethod.MEDIAN_CUT: "Deterministisch, gut für gleichmäßige Farbverteilung",
    AnalysisMethod.MEAN_SHIFT: "Findet Clusteranzahl automatisch, langsamer",
    AnalysisMethod.DBSCAN: "Dichtebasiert, erkennt Ausreißer als Rauschen",
    AnalysisMethod.OCTREE: "Extrem schnell, für Echtzeit-Vorschau",
    AnalysisMethod.HYBRID: "Vorfilterung + Feinanalyse, beste Qualität",
    AnalysisMethod.GMM: "Weiche Cluster-Zuordnung, gut für Farbverläufe",
}


@dataclass
class AnalysisParameters:
    """Parameters for color analysis methods."""

    # Common
    n_colors: int = 12
    random_state: Optional[int] = 42
    use_fixed_seed: bool = True

    # K-Means specific
    kmeans_batch_size: int = 1024
    kmeans_max_iter: int = 100

    # Mean Shift specific
    mean_shift_bandwidth: Optional[float] = None
    mean_shift_auto_bandwidth: bool = True
    mean_shift_bin_seeding: bool = True

    # DBSCAN specific
    dbscan_eps: float = 10.0
    dbscan_min_samples: int = 50
    dbscan_colorspace: str = "Lab"

    # Octree specific (uses n_colors)

    # Hybrid specific
    hybrid_octree_prefilter: int = 128
    hybrid_final_clusters: int = 12
    hybrid_colorspace: str = "Lab"

    # GMM specific
    gmm_covariance_type: str = "full"
    gmm_max_iter: int = 100


def _rgb_to_lab(rgb_pixels: np.ndarray) -> np.ndarray:
    """Convert RGB pixels to Lab color space."""
    try:
        from skimage import color as skcolor
        # Reshape if needed
        if len(rgb_pixels.shape) == 2:
            # (N, 3) -> (1, N, 3)
            reshaped = rgb_pixels.reshape(1, -1, 3).astype(np.float64) / 255.0
            lab = skcolor.rgb2lab(reshaped)
            return lab.reshape(-1, 3)
        return skcolor.rgb2lab(rgb_pixels.astype(np.float64) / 255.0)
    except ImportError:
        # Fallback: return RGB if skimage not available
        return rgb_pixels.astype(np.float64)


def _lab_to_rgb(lab_pixels: np.ndarray) -> np.ndarray:
    """Convert Lab pixels back to RGB."""
    try:
        from skimage import color as skcolor
        if len(lab_pixels.shape) == 2:
            reshaped = lab_pixels.reshape(1, -1, 3)
            rgb = skcolor.lab2rgb(reshaped)
            return (rgb.reshape(-1, 3) * 255).astype(np.uint8)
        rgb = skcolor.lab2rgb(lab_pixels)
        return (rgb * 255).astype(np.uint8)
    except ImportError:
        return lab_pixels.astype(np.uint8)


class ColorAnalysisEngine:
    """
    Unified color analysis engine supporting multiple clustering methods.

    Implements Strategy pattern for easy method switching.
    Thread-safe for use with QThread.

    Example:
        engine = ColorAnalysisEngine(method="kmeans", n_colors=12)
        palette = engine.extract_palette(image)
    """

    def __init__(
        self,
        method: str = "kmeans",
        params: Optional[AnalysisParameters] = None,
        **kwargs
    ):
        """
        Initialize the color analysis engine.

        Args:
            method: Analysis method name (kmeans, median_cut, etc.)
            params: AnalysisParameters instance
            **kwargs: Override specific parameters
        """
        self.method = AnalysisMethod(method) if isinstance(method, str) else method
        self.params = params or AnalysisParameters()

        # Apply any kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

        self._cluster_labels: Optional[np.ndarray] = None
        self._cluster_centers: Optional[np.ndarray] = None
        self._n_clusters_found: int = 0

    def extract_palette(
        self,
        image: np.ndarray,
        n_colors: Optional[int] = None
    ) -> List[Tuple[int, int, int]]:
        """
        Extract color palette from image.

        Args:
            image: Input image in RGB or BGR format (H, W, 3).
            n_colors: Number of colors to extract (overrides params).

        Returns:
            List of RGB tuples representing the palette.
        """
        if n_colors is not None:
            self.params.n_colors = n_colors

        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR if coming from OpenCV, convert to RGB
            import cv2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Reshape to pixel array
        pixels = rgb_image.reshape(-1, 3).astype(np.float32)

        # Subsample for performance
        max_samples = 50000
        if len(pixels) > max_samples:
            rng = np.random.RandomState(
                self.params.random_state if self.params.use_fixed_seed else None
            )
            indices = rng.choice(len(pixels), max_samples, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels

        # Dispatch to appropriate method
        method_map = {
            AnalysisMethod.KMEANS: self._extract_kmeans,
            AnalysisMethod.MEDIAN_CUT: self._extract_median_cut,
            AnalysisMethod.MEAN_SHIFT: self._extract_mean_shift,
            AnalysisMethod.DBSCAN: self._extract_dbscan,
            AnalysisMethod.OCTREE: self._extract_octree,
            AnalysisMethod.HYBRID: self._extract_hybrid,
            AnalysisMethod.GMM: self._extract_gmm,
        }

        extract_func = method_map.get(self.method, self._extract_kmeans)
        palette = extract_func(sample_pixels)

        return palette

    def get_cluster_labels(self, image: np.ndarray) -> np.ndarray:
        """
        Get cluster assignment for each pixel.

        Must be called after extract_palette().

        Args:
            image: Input image (same as used for extract_palette).

        Returns:
            Array of cluster indices for each pixel.
        """
        if self._cluster_labels is None:
            raise ValueError("Must call extract_palette() first")
        return self._cluster_labels

    def get_n_clusters_found(self) -> int:
        """Get number of clusters found (for auto-detecting methods)."""
        return self._n_clusters_found

    def _extract_kmeans(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using Mini-Batch K-Means."""
        random_state = self.params.random_state if self.params.use_fixed_seed else None

        kmeans = MiniBatchKMeans(
            n_clusters=self.params.n_colors,
            batch_size=self.params.kmeans_batch_size,
            max_iter=self.params.kmeans_max_iter,
            random_state=random_state,
            n_init=3,
        )
        kmeans.fit(pixels)

        self._cluster_labels = kmeans.labels_
        self._cluster_centers = kmeans.cluster_centers_
        self._n_clusters_found = self.params.n_colors

        return [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in kmeans.cluster_centers_.astype(np.uint8)
        ]

    def _extract_median_cut(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using Median Cut via PIL."""
        # Convert to PIL Image
        # Need to reshape back to image-like for PIL
        side = int(np.sqrt(len(pixels)))
        if side * side < len(pixels):
            side += 1

        # Pad to square
        padded = np.zeros((side * side, 3), dtype=np.uint8)
        padded[:len(pixels)] = pixels[:len(pixels)].astype(np.uint8)
        img_array = padded.reshape(side, side, 3)

        pil_img = Image.fromarray(img_array, mode='RGB')
        quantized = pil_img.quantize(colors=self.params.n_colors, method=Image.Quantize.MEDIANCUT)

        # Get palette
        palette = quantized.getpalette()
        colors = []
        for i in range(self.params.n_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            colors.append((r, g, b))

        self._n_clusters_found = len(colors)
        return colors

    def _extract_mean_shift(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using Mean Shift clustering."""
        # Mean Shift can be slow, subsample aggressively
        if len(pixels) > 5000:
            rng = np.random.RandomState(
                self.params.random_state if self.params.use_fixed_seed else None
            )
            indices = rng.choice(len(pixels), 5000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels

        bandwidth = self.params.mean_shift_bandwidth
        if self.params.mean_shift_auto_bandwidth or bandwidth is None:
            # Estimate bandwidth
            from sklearn.cluster import estimate_bandwidth
            bandwidth = estimate_bandwidth(sample, quantile=0.1, n_samples=500)
            if bandwidth <= 0:
                bandwidth = 30.0  # Fallback

        ms = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=self.params.mean_shift_bin_seeding,
        )
        ms.fit(sample)

        self._cluster_labels = ms.labels_
        self._cluster_centers = ms.cluster_centers_
        self._n_clusters_found = len(ms.cluster_centers_)

        return [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in ms.cluster_centers_.astype(np.uint8)
        ]

    def _extract_dbscan(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using DBSCAN clustering."""
        # Convert to Lab if requested
        if self.params.dbscan_colorspace == "Lab":
            sample = _rgb_to_lab(pixels.astype(np.uint8))
        else:
            sample = pixels

        # Subsample for DBSCAN (very slow on large datasets)
        if len(sample) > 5000:
            rng = np.random.RandomState(
                self.params.random_state if self.params.use_fixed_seed else None
            )
            indices = rng.choice(len(sample), 5000, replace=False)
            sample = sample[indices]
            pixels_subset = pixels[indices]
        else:
            pixels_subset = pixels

        db = DBSCAN(
            eps=self.params.dbscan_eps,
            min_samples=self.params.dbscan_min_samples,
        )
        labels = db.fit_predict(sample)

        self._cluster_labels = labels

        # Get cluster centers (mean of each cluster)
        unique_labels = set(labels) - {-1}  # Exclude noise
        centers = []
        for label in sorted(unique_labels):
            mask = labels == label
            center = pixels_subset[mask].mean(axis=0)
            centers.append(center)

        self._n_clusters_found = len(centers)

        # If too few clusters, fall back to K-Means
        if len(centers) < 2:
            return self._extract_kmeans(pixels)

        return [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in np.array(centers).astype(np.uint8)
        ]

    def _extract_octree(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using Octree quantization via PIL."""
        # Similar to median cut but uses FASTOCTREE
        side = int(np.sqrt(len(pixels)))
        if side * side < len(pixels):
            side += 1

        padded = np.zeros((side * side, 3), dtype=np.uint8)
        padded[:len(pixels)] = pixels[:len(pixels)].astype(np.uint8)
        img_array = padded.reshape(side, side, 3)

        pil_img = Image.fromarray(img_array, mode='RGB')
        quantized = pil_img.quantize(colors=self.params.n_colors, method=Image.Quantize.FASTOCTREE)

        palette = quantized.getpalette()
        colors = []
        for i in range(self.params.n_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            colors.append((r, g, b))

        self._n_clusters_found = len(colors)
        return colors

    def _extract_hybrid(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using Hybrid (Octree prefilter + K-Means)."""
        # Step 1: Octree prefilter
        side = int(np.sqrt(len(pixels)))
        if side * side < len(pixels):
            side += 1

        padded = np.zeros((side * side, 3), dtype=np.uint8)
        padded[:len(pixels)] = pixels[:len(pixels)].astype(np.uint8)
        img_array = padded.reshape(side, side, 3)

        pil_img = Image.fromarray(img_array, mode='RGB')
        quantized = pil_img.quantize(
            colors=self.params.hybrid_octree_prefilter,
            method=Image.Quantize.FASTOCTREE
        )

        # Get prefiltered palette
        palette = quantized.getpalette()
        prefiltered = []
        for i in range(self.params.hybrid_octree_prefilter):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            prefiltered.append([r, g, b])

        prefiltered = np.array(prefiltered, dtype=np.float32)

        # Step 2: K-Means on prefiltered colors
        # Convert to Lab if requested
        if self.params.hybrid_colorspace == "Lab":
            prefiltered_lab = _rgb_to_lab(prefiltered.astype(np.uint8))
            kmeans_input = prefiltered_lab
        else:
            kmeans_input = prefiltered

        random_state = self.params.random_state if self.params.use_fixed_seed else None
        n_final = min(self.params.hybrid_final_clusters, len(prefiltered))

        kmeans = MiniBatchKMeans(
            n_clusters=n_final,
            random_state=random_state,
            n_init=3,
        )
        kmeans.fit(kmeans_input)

        if self.params.hybrid_colorspace == "Lab":
            centers = _lab_to_rgb(kmeans.cluster_centers_)
        else:
            centers = kmeans.cluster_centers_.astype(np.uint8)

        self._n_clusters_found = n_final

        return [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in centers
        ]

    def _extract_gmm(self, pixels: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract palette using Gaussian Mixture Model."""
        random_state = self.params.random_state if self.params.use_fixed_seed else None

        # Subsample for GMM
        if len(pixels) > 10000:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(pixels), 10000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels

        gmm = GaussianMixture(
            n_components=self.params.n_colors,
            covariance_type=self.params.gmm_covariance_type,
            max_iter=self.params.gmm_max_iter,
            random_state=random_state,
        )
        gmm.fit(sample)

        self._cluster_labels = gmm.predict(sample)
        self._cluster_centers = gmm.means_
        self._n_clusters_found = self.params.n_colors

        return [
            (int(c[0]), int(c[1]), int(c[2]))
            for c in gmm.means_.astype(np.uint8)
        ]


def estimate_processing_time(
    method: AnalysisMethod,
    image_size: Tuple[int, int],
    n_colors: int
) -> float:
    """
    Estimate processing time in seconds.

    Args:
        method: Analysis method.
        image_size: (width, height) of image.
        n_colors: Number of colors to extract.

    Returns:
        Estimated time in seconds.
    """
    total_pixels = image_size[0] * image_size[1]

    # Base times per 1M pixels
    base_times = {
        AnalysisMethod.KMEANS: 0.5,
        AnalysisMethod.MEDIAN_CUT: 0.3,
        AnalysisMethod.MEAN_SHIFT: 5.0,
        AnalysisMethod.DBSCAN: 3.0,
        AnalysisMethod.OCTREE: 0.2,
        AnalysisMethod.HYBRID: 0.8,
        AnalysisMethod.GMM: 2.0,
    }

    base = base_times.get(method, 1.0)
    pixel_factor = total_pixels / 1_000_000

    # Adjust for number of colors
    color_factor = n_colors / 12

    return max(0.1, base * pixel_factor * color_factor)
