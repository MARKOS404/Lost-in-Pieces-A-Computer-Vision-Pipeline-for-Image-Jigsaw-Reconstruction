"""
Feature Extraction Module
==========================
Implements multiple descriptor families for puzzle pieces:
1. Color histogram descriptors
2. Texture descriptors (filter bank responses)
3. Local interest point descriptors (Harris + gradient histograms)
4. Deep-like CNN descriptors (multi-scale Gabor filter bank)

Each descriptor can be computed on full tiles or on border strips (sides).
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Border Strip Extraction
# =============================================================================

def extract_border_strip(tile: np.ndarray, side: str, wb: int = 10) -> np.ndarray:
    """
    Extract a border strip from a tile.
    
    Parameters
    ----------
    tile : np.ndarray
        Tile image of shape (h, w, C).
    side : str
        One of 'top', 'bottom', 'left', 'right' (or 'N', 'E', 'S', 'W').
    wb : int
        Width of the border strip in pixels.
    
    Returns
    -------
    strip : np.ndarray
        The border strip image.
    """
    side_map = {'N': 'top', 'E': 'right', 'S': 'bottom', 'W': 'left'}
    side = side_map.get(side, side)

    h, w = tile.shape[:2]
    wb = min(wb, h // 2, w // 2)

    if side == 'top':
        return tile[:wb, :].copy()
    elif side == 'bottom':
        return tile[-wb:, :].copy()
    elif side == 'left':
        return tile[:, :wb].copy()
    elif side == 'right':
        return tile[:, -wb:].copy()
    else:
        raise ValueError(f"Unknown side: {side}")


def get_side_pairs():
    """
    Returns the matching side pairs for adjacency.
    If piece A is to the LEFT of piece B, then A's RIGHT side faces B's LEFT side.
    """
    return {
        'horizontal': ('right', 'left'),   # A.right <-> B.left  (A left of B)
        'vertical': ('bottom', 'top'),      # A.bottom <-> B.top  (A above B)
    }


# =============================================================================
# Color Histogram Descriptors
# =============================================================================

class ColorHistogramDescriptor:
    """
    Computes normalized color histograms over image regions.
    Supports RGB, HSV, and Lab color spaces.
    """

    def __init__(self, color_space: str = 'hsv', bins_per_channel: int = 16):
        self.color_space = color_space
        self.bins = bins_per_channel

    def _convert_color(self, region: np.ndarray) -> np.ndarray:
        if self.color_space == 'hsv':
            return cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'lab':
            return cv2.cvtColor(region, cv2.COLOR_BGR2Lab)
        elif self.color_space == 'rgb':
            return region.copy()
        else:
            return region.copy()

    def compute(self, region: np.ndarray) -> np.ndarray:
        """
        Compute normalized color histogram for a region.
        
        Returns
        -------
        descriptor : np.ndarray
            Concatenated per-channel histograms, normalized.
        """
        if region.size == 0:
            return np.zeros(self.bins * 3)

        converted = self._convert_color(region)
        histograms = []

        for c in range(converted.shape[2] if converted.ndim == 3 else 1):
            channel = converted[:, :, c] if converted.ndim == 3 else converted
            hist, _ = np.histogram(channel.ravel(), bins=self.bins,
                                   range=(0, 256), density=True)
            histograms.append(hist)

        descriptor = np.concatenate(histograms)
        norm = np.linalg.norm(descriptor) + 1e-10
        return descriptor / norm


# =============================================================================
# Texture Descriptors (Filter Bank)
# =============================================================================

class TextureDescriptor:
    """
    Computes texture descriptors using a bank of filters:
    - Gaussian derivatives at multiple scales
    - Sobel edge detectors
    - Laplacian of Gaussian (LoG) at multiple scales
    
    Statistics (mean, std, energy) of filter responses form the descriptor.
    """

    def __init__(self, scales: List[float] = None):
        if scales is None:
            self.scales = [1.0, 2.0, 4.0]
        else:
            self.scales = scales

    def _build_filter_bank(self):
        """Build the set of filter functions."""
        filters = []
        for sigma in self.scales:
            # Gaussian first derivatives (x and y)
            filters.append(('dx', sigma))
            filters.append(('dy', sigma))
            # Laplacian of Gaussian
            filters.append(('log', sigma))
        # Sobel
        filters.append(('sobel_x', None))
        filters.append(('sobel_y', None))
        return filters

    def _apply_filter(self, gray: np.ndarray, ftype: str, param) -> np.ndarray:
        """Apply a single filter and return response map."""
        if ftype == 'dx':
            return ndimage.gaussian_filter(gray, sigma=param, order=[0, 1])
        elif ftype == 'dy':
            return ndimage.gaussian_filter(gray, sigma=param, order=[1, 0])
        elif ftype == 'log':
            return ndimage.gaussian_laplace(gray, sigma=param)
        elif ftype == 'sobel_x':
            return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        elif ftype == 'sobel_y':
            return cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.zeros_like(gray, dtype=np.float64)

    def compute(self, region: np.ndarray) -> np.ndarray:
        """
        Compute texture descriptor for a region.
        
        Returns
        -------
        descriptor : np.ndarray
            Concatenated [mean, std, energy] for each filter response.
        """
        if region.size == 0:
            n_filters = len(self.scales) * 3 + 2
            return np.zeros(n_filters * 3)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0 \
            if region.ndim == 3 else region.astype(np.float64) / 255.0

        filters = self._build_filter_bank()
        stats = []

        for ftype, param in filters:
            response = self._apply_filter(gray, ftype, param)
            mean_val = np.mean(response)
            std_val = np.std(response)
            energy = np.mean(response ** 2)
            stats.extend([mean_val, std_val, energy])

        descriptor = np.array(stats, dtype=np.float64)
        norm = np.linalg.norm(descriptor) + 1e-10
        return descriptor / norm


# =============================================================================
# Gabor Texture Descriptor
# =============================================================================

class GaborDescriptor:
    """
    Texture descriptor based on a bank of Gabor filters with multiple
    orientations and wavelengths.
    """

    def __init__(self, orientations: int = 6, wavelengths: List[float] = None):
        self.orientations = orientations
        self.wavelengths = wavelengths or [4.0, 8.0, 16.0]
        self.kernels = self._build_gabor_bank()

    def _build_gabor_bank(self):
        """Create a bank of Gabor filter kernels."""
        kernels = []
        for lam in self.wavelengths:
            for i in range(self.orientations):
                theta = i * np.pi / self.orientations
                sigma = lam * 0.56
                kernel = cv2.getGaborKernel(
                    ksize=(int(4 * sigma + 1) | 1, int(4 * sigma + 1) | 1),
                    sigma=sigma, theta=theta, lambd=lam,
                    gamma=0.5, psi=0
                )
                kernels.append(kernel)
        return kernels

    def compute(self, region: np.ndarray) -> np.ndarray:
        """Compute Gabor-based texture descriptor."""
        if region.size == 0:
            return np.zeros(len(self.kernels) * 2)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float64) \
            if region.ndim == 3 else region.astype(np.float64)

        stats = []
        for kernel in self.kernels:
            response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            stats.append(np.mean(response))
            stats.append(np.mean(response ** 2))  # energy

        descriptor = np.array(stats, dtype=np.float64)
        norm = np.linalg.norm(descriptor) + 1e-10
        return descriptor / norm


# =============================================================================
# Local Interest Point Descriptors (Harris + Gradient Histograms)
# =============================================================================

class LocalDescriptor:
    """
    Detects Harris corners in border strips and computes gradient histogram
    descriptors (simplified SIFT-like) at each keypoint, then aggregates.
    """

    def __init__(self, n_bins: int = 8, patch_size: int = 16,
                 harris_k: float = 0.04, max_keypoints: int = 50):
        self.n_bins = n_bins
        self.patch_size = patch_size
        self.harris_k = harris_k
        self.max_keypoints = max_keypoints

    def _detect_harris_keypoints(self, gray: np.ndarray) -> List[Tuple[int, int]]:
        """Detect Harris corners and return keypoint locations."""
        gray_f = np.float32(gray)
        harris_response = cv2.cornerHarris(gray_f, blockSize=3, ksize=3, k=self.harris_k)

        # Threshold at percentile
        threshold = max(np.percentile(harris_response, 95), 1e-6)
        candidates = np.argwhere(harris_response > threshold)

        if len(candidates) == 0:
            # Fallback: sample uniformly
            h, w = gray.shape
            ys = np.linspace(self.patch_size // 2, h - self.patch_size // 2 - 1, 5).astype(int)
            xs = np.linspace(self.patch_size // 2, w - self.patch_size // 2 - 1, 5).astype(int)
            keypoints = [(y, x) for y in ys for x in xs if
                         0 <= y < h and 0 <= x < w]
            return keypoints[:self.max_keypoints]

        # Non-maximum suppression
        nms_mask = np.zeros_like(harris_response, dtype=bool)
        for y, x in candidates:
            y_lo = max(0, y - 2)
            y_hi = min(harris_response.shape[0], y + 3)
            x_lo = max(0, x - 2)
            x_hi = min(harris_response.shape[1], x + 3)
            if harris_response[y, x] == harris_response[y_lo:y_hi, x_lo:x_hi].max():
                nms_mask[y, x] = True

        keypoints_arr = np.argwhere(nms_mask)

        # Filter keypoints that are too close to edges for patch extraction
        half = self.patch_size // 2
        h, w = gray.shape
        valid = []
        for y, x in keypoints_arr:
            if half <= y < h - half and half <= x < w - half:
                valid.append((y, x))

        # Sort by response strength and limit
        valid.sort(key=lambda p: harris_response[p[0], p[1]], reverse=True)
        return valid[:self.max_keypoints]

    def _compute_patch_descriptor(self, gray: np.ndarray, y: int, x: int) -> np.ndarray:
        """Compute gradient histogram descriptor for a patch centered at (y, x)."""
        half = self.patch_size // 2
        patch = gray[y - half:y + half, x - half:x + half].astype(np.float64)

        if patch.shape[0] < 3 or patch.shape[1] < 3:
            return np.zeros(self.n_bins)

        # Gradient computation
        gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx) % (2 * np.pi)  # [0, 2*pi)

        # Gaussian weighting
        sigma_w = half * 0.5
        yy, xx = np.mgrid[-half:half, -half:half]
        weights = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_w ** 2))
        weighted_mag = magnitude * weights[:magnitude.shape[0], :magnitude.shape[1]]

        # Orientation histogram
        bin_edges = np.linspace(0, 2 * np.pi, self.n_bins + 1)
        hist = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            mask = (orientation >= bin_edges[b]) & (orientation < bin_edges[b + 1])
            hist[b] = np.sum(weighted_mag[mask])

        # Normalize
        norm = np.linalg.norm(hist) + 1e-10
        return hist / norm

    def compute(self, region: np.ndarray) -> np.ndarray:
        """
        Compute aggregated local descriptor for a region.
        Detects keypoints, computes descriptors, and averages.
        """
        if region.size == 0 or min(region.shape[:2]) < self.patch_size:
            return np.zeros(self.n_bins)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) \
            if region.ndim == 3 else region

        keypoints = self._detect_harris_keypoints(gray)

        if not keypoints:
            return np.zeros(self.n_bins)

        descriptors = []
        for y, x in keypoints:
            desc = self._compute_patch_descriptor(gray, y, x)
            descriptors.append(desc)

        # Aggregate by averaging
        aggregated = np.mean(descriptors, axis=0)
        return aggregated


# =============================================================================
# Deep-Like CNN Descriptor (Multi-Scale Gabor + Pooling)
# =============================================================================

class DeepLikeDescriptor:
    """
    Simulates deep CNN feature extraction using a multi-layer pipeline:
    Layer 1: Low-level edge/gradient filters
    Layer 2: Mid-level Gabor texture filters
    Layer 3: High-level statistical aggregation
    
    Each layer's output is spatially pooled (global average) to produce
    a fixed-length descriptor, similar to extracting intermediate CNN activations.
    """

    def __init__(self, target_size: int = 64):
        """
        Parameters
        ----------
        target_size : int
            Resize input regions to target_size x target_size before processing.
        """
        self.target_size = target_size
        self.layer1_filters = self._build_layer1()
        self.layer2_kernels = self._build_layer2()

    def _build_layer1(self):
        """Low-level edge/gradient filters (like Conv1 in a CNN)."""
        filters = []
        # Sobel-like
        filters.append(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64))
        filters.append(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64))
        # Diagonal edges
        filters.append(np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float64))
        filters.append(np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float64))
        # Laplacian
        filters.append(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64))
        # Smoothing
        filters.append(np.ones((3, 3), dtype=np.float64) / 9.0)
        return filters

    def _build_layer2(self):
        """Mid-level Gabor filters (like Conv2-3 in a CNN)."""
        kernels = []
        for lam in [4.0, 8.0]:
            for i in range(4):
                theta = i * np.pi / 4
                sigma = lam * 0.56
                ksize = int(4 * sigma + 1) | 1
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, 0.5, 0
                )
                kernels.append(kernel)
        return kernels

    def compute(self, region: np.ndarray) -> np.ndarray:
        """
        Compute deep-like descriptor with hierarchical feature extraction.
        """
        if region.size == 0:
            return np.zeros(self._get_dim())

        # Preprocess: resize to fixed size
        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        resized = cv2.resize(gray, (self.target_size, self.target_size)).astype(np.float64)
        resized = resized / 255.0

        features = []

        # Layer 1: Low-level features + ReLU + pooling
        layer1_maps = []
        for f in self.layer1_filters:
            response = cv2.filter2D(resized, cv2.CV_64F, f)
            activated = np.maximum(response, 0)  # ReLU
            layer1_maps.append(activated)
            # Global average pooling
            features.append(np.mean(activated))
            # Global max pooling
            features.append(np.max(activated))
            # Standard deviation
            features.append(np.std(activated))

        # Layer 2: Mid-level features on pooled layer1 output
        # Combine layer1 maps
        if layer1_maps:
            combined = np.mean(layer1_maps, axis=0)
        else:
            combined = resized

        for k in self.layer2_kernels:
            response = cv2.filter2D(combined, cv2.CV_64F, k)
            activated = np.maximum(response, 0)
            features.append(np.mean(activated))
            features.append(np.max(activated))
            features.append(np.std(activated))

        # Layer 3: Higher-order statistics on full image
        # Spatial pyramid: split into 2x2 quadrants
        h, w = resized.shape
        for qi in range(2):
            for qj in range(2):
                quadrant = resized[qi * h // 2:(qi + 1) * h // 2,
                                   qj * w // 2:(qj + 1) * w // 2]
                features.append(np.mean(quadrant))
                features.append(np.std(quadrant))
                # Gradient energy in quadrant
                gx = cv2.Sobel(quadrant, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(quadrant, cv2.CV_64F, 0, 1, ksize=3)
                features.append(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

        descriptor = np.array(features, dtype=np.float64)
        norm = np.linalg.norm(descriptor) + 1e-10
        return descriptor / norm

    def _get_dim(self) -> int:
        """Return the descriptor dimensionality."""
        return len(self.layer1_filters) * 3 + len(self.layer2_kernels) * 3 + 4 * 3


# =============================================================================
# Pixel Row/Column Descriptor (Simple but effective for jigsaw)
# =============================================================================

class PixelBorderDescriptor:
    """
    Directly uses the pixel values along the 1-pixel-wide edge of each side.
    This is highly discriminative for jigsaw puzzles since adjacent tiles
    share nearly identical pixel rows/columns at their interface.
    """

    def __init__(self, n_samples: int = 32):
        """
        Parameters
        ----------
        n_samples : int
            Number of uniformly sampled pixel values along the border.
        """
        self.n_samples = n_samples

    def compute_side(self, tile: np.ndarray, side: str) -> np.ndarray:
        """Extract pixel values from the 1-pixel edge of a side."""
        if tile.ndim == 2:
            tile = tile[:, :, np.newaxis]

        side_map = {'N': 'top', 'E': 'right', 'S': 'bottom', 'W': 'left'}
        side = side_map.get(side, side)

        if side == 'top':
            border = tile[0, :, :]
        elif side == 'bottom':
            border = tile[-1, :, :]
        elif side == 'left':
            border = tile[:, 0, :]
        elif side == 'right':
            border = tile[:, -1, :]
        else:
            raise ValueError(f"Unknown side: {side}")

        # Resample to fixed length
        if len(border) != self.n_samples:
            indices = np.linspace(0, len(border) - 1, self.n_samples).astype(int)
            border = border[indices]

        return border.flatten().astype(np.float64) / 255.0


# =============================================================================
# Feature Extraction Pipeline
# =============================================================================

class FeatureExtractor:
    """
    Master feature extraction class that computes all descriptor families
    for each tile and each side.
    """

    SIDES = ['top', 'right', 'bottom', 'left']

    def __init__(self, border_width: int = 10, use_color: bool = True,
                 use_texture: bool = True, use_local: bool = True,
                 use_deep: bool = True, use_gabor: bool = True,
                 use_pixel_border: bool = True):
        self.border_width = border_width
        self.use_color = use_color
        self.use_texture = use_texture
        self.use_local = use_local
        self.use_deep = use_deep
        self.use_gabor = use_gabor
        self.use_pixel_border = use_pixel_border

        # Initialize descriptors
        self.color_desc = ColorHistogramDescriptor(color_space='hsv', bins_per_channel=16)
        self.color_lab_desc = ColorHistogramDescriptor(color_space='lab', bins_per_channel=16)
        self.texture_desc = TextureDescriptor(scales=[1.0, 2.0, 4.0])
        self.local_desc = LocalDescriptor(n_bins=8, patch_size=12, max_keypoints=30)
        self.deep_desc = DeepLikeDescriptor(target_size=64)
        self.gabor_desc = GaborDescriptor(orientations=6, wavelengths=[4.0, 8.0, 16.0])
        self.pixel_desc = PixelBorderDescriptor(n_samples=32)

    def extract_tile_features(self, tile: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract tile-level (piece-level) descriptors."""
        features = {}
        if self.use_color:
            features['color_hsv'] = self.color_desc.compute(tile)
            features['color_lab'] = self.color_lab_desc.compute(tile)
        if self.use_texture:
            features['texture'] = self.texture_desc.compute(tile)
        if self.use_deep:
            features['deep'] = self.deep_desc.compute(tile)
        if self.use_gabor:
            features['gabor'] = self.gabor_desc.compute(tile)
        return features

    def extract_side_features(self, tile: np.ndarray, side: str) -> Dict[str, np.ndarray]:
        """Extract side-level descriptors from a border strip."""
        strip = extract_border_strip(tile, side, self.border_width)
        features = {}

        if self.use_color:
            features['color_hsv'] = self.color_desc.compute(strip)
            features['color_lab'] = self.color_lab_desc.compute(strip)
        if self.use_texture:
            features['texture'] = self.texture_desc.compute(strip)
        if self.use_local:
            features['local'] = self.local_desc.compute(strip)
        if self.use_deep:
            features['deep'] = self.deep_desc.compute(strip)
        if self.use_gabor:
            features['gabor'] = self.gabor_desc.compute(strip)
        if self.use_pixel_border:
            features['pixel_border'] = self.pixel_desc.compute_side(tile, side)

        return features

    def extract_all(self, tiles: List[np.ndarray]) -> Dict:
        """
        Extract all features for all tiles and all sides.
        
        Returns
        -------
        features : dict with keys:
            'tile_features': list of dicts (one per tile)
            'side_features': list of dicts (one per tile), each mapping side -> descriptor dict
        """
        tile_features = []
        side_features = []

        for i, tile in enumerate(tiles):
            # Tile-level
            tf = self.extract_tile_features(tile)
            tile_features.append(tf)

            # Side-level
            sf = {}
            for side in self.SIDES:
                sf[side] = self.extract_side_features(tile, side)
            side_features.append(sf)

        return {
            'tile_features': tile_features,
            'side_features': side_features,
        }


def rotate_side(side: str, angle: int) -> str:
    """
    Given a side label and a rotation angle (0, 90, 180, 270),
    return the original side label before rotation.
    
    If a tile was rotated by `angle` degrees CCW, the pixel data that
    is now on `side` originally came from `rotate_side(side, angle)`.
    """
    sides = ['top', 'right', 'bottom', 'left']
    idx = sides.index(side)
    steps = angle // 90
    return sides[(idx + steps) % 4]


def opposite_side(side: str) -> str:
    """Return the opposite side."""
    opp = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
    return opp[side]
