"""
Puzzle Generator Module
========================
Handles image segmentation into tiles, shuffling, rotation, and ground truth storage.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional


class PuzzleGenerator:
    """
    Generates a jigsaw puzzle from an input image by:
    1. Partitioning the image into a P x Q grid of tiles.
    2. Applying a random permutation (shuffle) to the tiles.
    3. Optionally rotating each tile by a random multiple of 90 degrees.
    """

    def __init__(self, image: np.ndarray, P: int, Q: int, allow_rotation: bool = True,
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        image : np.ndarray
            Input image of shape (H, W, C) or (H, W).
        P : int
            Number of rows in the tile grid.
        Q : int
            Number of columns in the tile grid.
        allow_rotation : bool
            Whether to apply random rotations to tiles.
        seed : int or None
            Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

        # Ensure image is 3-channel
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.original_image = image.copy()

        H, W, C = image.shape
        # Crop image so dimensions are divisible by P and Q
        self.H = (H // P) * P
        self.W = (W // Q) * Q
        self.image = image[:self.H, :self.W].copy()
        self.P = P
        self.Q = Q
        self.C = C
        self.N = P * Q
        self.h = self.H // P  # tile height
        self.w = self.W // Q  # tile width
        self.allow_rotation = allow_rotation
        self.ROTATIONS = [0, 90, 180, 270]

        # Ground truth storage
        self.ground_truth_positions = {}   # k -> (r, c)
        self.ground_truth_rotations = {}   # k -> angle in degrees
        self.permutation = None

        # Generated puzzle data
        self.tiles = []           # List of tile images (after shuffle + rotation)
        self.original_tiles = []  # List of tile images (before shuffle + rotation)

    def generate(self) -> List[np.ndarray]:
        """
        Generate the puzzle: extract tiles, shuffle, rotate.
        
        Returns
        -------
        tiles : list of np.ndarray
            The shuffled (and possibly rotated) tile images.
        """
        # Step 1: Extract tiles from the grid
        self.original_tiles = []
        for r in range(self.P):
            for c in range(self.Q):
                y_start = r * self.h
                y_end = (r + 1) * self.h
                x_start = c * self.w
                x_end = (c + 1) * self.w
                tile = self.image[y_start:y_end, x_start:x_end].copy()
                self.original_tiles.append(tile)

        # Step 2: Create ground truth mapping (before shuffle)
        for k in range(self.N):
            r = k // self.Q
            c = k % self.Q
            self.ground_truth_positions[k] = (r, c)

        # Step 3: Generate random permutation
        self.permutation = self.rng.permutation(self.N)

        # Step 4: Apply shuffle and rotation
        self.tiles = []
        for idx, k in enumerate(self.permutation):
            tile = self.original_tiles[k].copy()

            if self.allow_rotation:
                angle = self.ROTATIONS[self.rng.randint(0, 4)]
            else:
                angle = 0

            self.ground_truth_rotations[idx] = angle

            # Rotate tile
            if angle == 90:
                tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                tile = cv2.rotate(tile, cv2.ROTATE_180)
            elif angle == 270:
                tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)

            self.tiles.append(tile)

        # Update ground truth: for shuffled piece idx, what is its true position?
        self.gt_placement = {}      # shuffled_idx -> (r, c)
        self.gt_orientation = {}    # shuffled_idx -> angle to undo
        for idx, k in enumerate(self.permutation):
            self.gt_placement[idx] = self.ground_truth_positions[k]
            self.gt_orientation[idx] = self.ground_truth_rotations[idx]

        return self.tiles

    def get_ground_truth(self) -> Tuple[Dict, Dict]:
        """
        Returns the ground truth placement and orientation.
        
        Returns
        -------
        gt_placement : dict
            Maps shuffled piece index -> (row, col) in original grid.
        gt_orientation : dict
            Maps shuffled piece index -> rotation angle applied.
        """
        return self.gt_placement, self.gt_orientation

    def get_grid_dims(self) -> Tuple[int, int]:
        """Return (P, Q) grid dimensions."""
        return self.P, self.Q

    def get_tile_dims(self) -> Tuple[int, int]:
        """Return (h, w) tile dimensions in pixels."""
        return self.h, self.w

    def reconstruct_from_solution(self, placement: Dict[int, Tuple[int, int]],
                                   orientations: Dict[int, int]) -> np.ndarray:
        """
        Reconstruct an image from a solution (placement + orientations).
        
        Parameters
        ----------
        placement : dict
            Maps piece index -> (row, col) grid position.
        orientations : dict
            Maps piece index -> estimated rotation angle to undo.
        
        Returns
        -------
        reconstructed : np.ndarray
            The reconstructed image.
        """
        reconstructed = np.zeros((self.H, self.W, self.C), dtype=np.uint8)

        for idx in range(self.N):
            tile = self.tiles[idx].copy()

            # Undo estimated rotation
            angle = orientations.get(idx, 0)
            if angle == 90:
                tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                tile = cv2.rotate(tile, cv2.ROTATE_180)
            elif angle == 270:
                tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)

            r, c = placement[idx]
            y_start = r * self.h
            x_start = c * self.w

            # Handle potential size mismatches from rotation of non-square tiles
            th, tw = tile.shape[:2]
            if th == self.h and tw == self.w:
                reconstructed[y_start:y_start + self.h, x_start:x_start + self.w] = tile
            else:
                # Resize if needed
                tile = cv2.resize(tile, (self.w, self.h))
                reconstructed[y_start:y_start + self.h, x_start:x_start + self.w] = tile

        return reconstructed


def create_sample_image(height: int = 400, width: int = 400) -> np.ndarray:
    """
    Create a sample test image with distinct regions, gradients, and patterns.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(50 + 150 * x / width),
                int(50 + 150 * y / height),
                int(200 - 100 * (x + y) / (width + height))
            ]

    # Draw colored shapes for distinctive regions
    cv2.circle(img, (width // 4, height // 4), min(width, height) // 6,
               (255, 100, 50), -1)
    cv2.rectangle(img, (width // 2, height // 2),
                  (3 * width // 4, 3 * height // 4), (50, 255, 100), -1)
    cv2.ellipse(img, (3 * width // 4, height // 4),
                (width // 8, height // 6), 30, 0, 360, (100, 50, 255), -1)

    # Add lines for edge features
    cv2.line(img, (0, height // 2), (width, height // 2), (255, 255, 0), 3)
    cv2.line(img, (width // 2, 0), (width // 2, height), (0, 255, 255), 3)

    # Add text-like patterns
    cv2.putText(img, "A", (width // 8, 7 * height // 8),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, "B", (5 * width // 8, 7 * height // 8),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)

    # Add checkerboard pattern region
    block = 20
    for y in range(0, height // 4, block):
        for x in range(width // 2, 3 * width // 4, block):
            if (y // block + x // block) % 2 == 0:
                cv2.rectangle(img, (x, y), (min(x + block, 3 * width // 4),
                              min(y + block, height // 4)), (180, 180, 180), -1)

    return img


def create_natural_test_image(height: int = 400, width: int = 400) -> np.ndarray:
    """Create a more natural-looking test image with smooth gradients and textures."""
    img = np.zeros((height, width, 3), dtype=np.float64)

    # Create smooth multi-frequency patterns
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(float)

    # Channel 0: smooth waves
    img[:, :, 0] = (
        128 + 60 * np.sin(2 * np.pi * x_coords / width * 3) *
        np.cos(2 * np.pi * y_coords / height * 2) +
        40 * np.sin(2 * np.pi * (x_coords + y_coords) / (width + height) * 5)
    )

    # Channel 1: radial gradient with perturbation
    cx, cy = width / 2, height / 2
    r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    img[:, :, 1] = 128 + 100 * np.cos(r / max(width, height) * 4 * np.pi)

    # Channel 2: diagonal stripes
    img[:, :, 2] = (
        128 + 80 * np.sin(2 * np.pi * (2 * x_coords - y_coords) / width * 2) +
        30 * np.cos(2 * np.pi * y_coords / height * 7)
    )

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Add Gaussian noise for texture
    noise = np.random.RandomState(42).normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img
