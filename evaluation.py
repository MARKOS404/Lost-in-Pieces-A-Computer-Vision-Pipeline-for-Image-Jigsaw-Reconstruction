"""
Evaluation and Visualization Module
=====================================
Computes quantitative metrics for reconstruction quality and
generates visual comparisons.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, List, Optional


# =============================================================================
# Quantitative Evaluation Metrics
# =============================================================================

class ReconstructionEvaluator:
    """
    Evaluates reconstruction quality against ground truth.
    
    Metrics:
    - Piece placement accuracy
    - Neighbor accuracy
    - Rotation accuracy
    - Direct comparison accuracy (placement + rotation correct)
    """

    def __init__(self, P: int, Q: int):
        self.P = P
        self.Q = Q
        self.N = P * Q

    def piece_placement_accuracy(self, predicted: Dict[int, Tuple[int, int]],
                                   gt: Dict[int, Tuple[int, int]]) -> float:
        """
        Fraction of pieces placed at their correct absolute position.
        """
        correct = 0
        for piece in gt:
            if piece in predicted and predicted[piece] == gt[piece]:
                correct += 1
        return correct / max(len(gt), 1)

    def rotation_accuracy(self, predicted_rot: Dict[int, int],
                           gt_rot: Dict[int, int]) -> float:
        """
        Fraction of pieces with correctly estimated orientation.
        """
        correct = 0
        for piece in gt_rot:
            if piece in predicted_rot and predicted_rot[piece] % 360 == gt_rot[piece] % 360:
                correct += 1
        return correct / max(len(gt_rot), 1)

    def neighbor_accuracy(self, predicted: Dict[int, Tuple[int, int]],
                           gt: Dict[int, Tuple[int, int]]) -> float:
        """
        Fraction of true neighboring pairs recovered in reconstruction.
        """
        P, Q = self.P, self.Q

        # Build grids
        gt_grid = {}
        pred_grid = {}
        for piece, pos in gt.items():
            gt_grid[pos] = piece
        for piece, pos in predicted.items():
            pred_grid[pos] = piece

        # Build predicted piece-to-position
        pred_pos = {piece: pos for piece, pos in predicted.items()}

        # Count true neighbor pairs recovered
        true_neighbors = 0
        recovered = 0

        for piece_a, (r, c) in gt.items():
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < P and 0 <= nc < Q:
                    # Find piece at (nr, nc) in ground truth
                    piece_b = None
                    for p, pos in gt.items():
                        if pos == (nr, nc):
                            piece_b = p
                            break

                    if piece_b is not None:
                        true_neighbors += 1

                        # Check if they are still neighbors in prediction
                        if piece_a in pred_pos and piece_b in pred_pos:
                            pr_a = pred_pos[piece_a]
                            pr_b = pred_pos[piece_b]
                            # Are they adjacent in any direction?
                            diff = (pr_b[0] - pr_a[0], pr_b[1] - pr_a[1])
                            if diff in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                recovered += 1

        return recovered / max(true_neighbors, 1)

    def direct_comparison_accuracy(self, predicted: Dict[int, Tuple[int, int]],
                                     gt: Dict[int, Tuple[int, int]],
                                     predicted_rot: Dict[int, int],
                                     gt_rot: Dict[int, int]) -> float:
        """
        Fraction of pieces with both correct placement AND correct rotation.
        """
        correct = 0
        for piece in gt:
            pos_ok = (piece in predicted and predicted[piece] == gt[piece])
            rot_ok = (piece in predicted_rot and
                      predicted_rot[piece] % 360 == gt_rot[piece] % 360)
            if pos_ok and rot_ok:
                correct += 1
        return correct / max(len(gt), 1)

    def evaluate_all(self, predicted_placement: Dict, gt_placement: Dict,
                      predicted_orientation: Dict, gt_orientation: Dict) -> Dict[str, float]:
        """Compute all metrics and return as a dictionary."""
        results = {
            'placement_accuracy': self.piece_placement_accuracy(
                predicted_placement, gt_placement),
            'neighbor_accuracy': self.neighbor_accuracy(
                predicted_placement, gt_placement),
            'rotation_accuracy': self.rotation_accuracy(
                predicted_orientation, gt_orientation),
            'direct_comparison_accuracy': self.direct_comparison_accuracy(
                predicted_placement, gt_placement,
                predicted_orientation, gt_orientation),
        }
        return results


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_puzzle_tiles(tiles: List[np.ndarray], P: int, Q: int,
                            title: str = "Puzzle Tiles",
                            save_path: Optional[str] = None):
    """Visualize shuffled puzzle tiles in a grid."""
    fig, axes = plt.subplots(P, Q, figsize=(Q * 2, P * 2))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if P == 1 and Q == 1:
        axes = np.array([[axes]])
    elif P == 1:
        axes = axes.reshape(1, -1)
    elif Q == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(P * Q):
        r = idx // Q
        c = idx % Q
        if idx < len(tiles):
            tile_rgb = cv2.cvtColor(tiles[idx], cv2.COLOR_BGR2RGB)
            axes[r, c].imshow(tile_rgb)
        axes[r, c].axis('off')
        axes[r, c].set_title(f'#{idx}', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_reconstruction_comparison(original: np.ndarray,
                                          reconstructed: np.ndarray,
                                          shuffled_tiles: List[np.ndarray],
                                          P: int, Q: int,
                                          metrics: Dict[str, float],
                                          title: str = "Reconstruction Results",
                                          save_path: Optional[str] = None):
    """
    Create a comprehensive comparison visualization showing:
    1. Original image
    2. Shuffled tiles
    3. Reconstructed image
    4. Difference map
    5. Metrics table
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')

    # 2. Shuffled tiles
    ax2 = fig.add_subplot(gs[0, 2])
    # Create a montage of shuffled tiles
    montage = create_montage(shuffled_tiles, P, Q)
    ax2.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
    ax2.set_title('Shuffled Pieces', fontsize=12)
    ax2.axis('off')

    # 3. Reconstructed image
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    ax3.set_title('Reconstructed', fontsize=12)
    ax3.axis('off')

    # 4. Difference map
    ax4 = fig.add_subplot(gs[1, 0:2])
    if original.shape == reconstructed.shape:
        diff = np.abs(original.astype(float) - reconstructed.astype(float))
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
        ax4.imshow(cv2.cvtColor(diff_normalized, cv2.COLOR_BGR2RGB))
    else:
        ax4.text(0.5, 0.5, 'Size mismatch', ha='center', va='center', fontsize=14)
    ax4.set_title('Absolute Difference', fontsize=12)
    ax4.axis('off')

    # 5. Grid overlay on reconstruction (show tile boundaries)
    ax5 = fig.add_subplot(gs[1, 2])
    recon_rgb = cv2.cvtColor(reconstructed.copy(), cv2.COLOR_BGR2RGB)
    h = original.shape[0] // P
    w = original.shape[1] // Q
    for r in range(1, P):
        cv2.line(recon_rgb, (0, r * h), (recon_rgb.shape[1], r * h), (255, 0, 0), 2)
    for c in range(1, Q):
        cv2.line(recon_rgb, (c * w, 0), (c * w, recon_rgb.shape[0]), (255, 0, 0), 2)
    ax5.imshow(recon_rgb)
    ax5.set_title('Reconstruction (Grid)', fontsize=12)
    ax5.axis('off')

    # 6. Metrics display
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    metrics_text = "Evaluation Metrics\n" + "=" * 30 + "\n\n"
    for name, value in metrics.items():
        display_name = name.replace('_', ' ').title()
        metrics_text += f"{display_name}:\n  {value:.1%}\n\n"
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_montage(tiles: List[np.ndarray], P: int, Q: int,
                     gap: int = 2) -> np.ndarray:
    """Create a montage image of tiles arranged in P x Q grid with gaps."""
    if not tiles:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = tiles[0].shape[:2]
    C = tiles[0].shape[2] if tiles[0].ndim == 3 else 1

    montage_h = P * h + (P - 1) * gap
    montage_w = Q * w + (Q - 1) * gap
    montage = np.ones((montage_h, montage_w, 3), dtype=np.uint8) * 200

    for idx in range(min(len(tiles), P * Q)):
        r = idx // Q
        c = idx % Q
        y = r * (h + gap)
        x = c * (w + gap)

        tile = tiles[idx]
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        # Handle size mismatch
        th, tw = tile.shape[:2]
        if th != h or tw != w:
            tile = cv2.resize(tile, (w, h))

        montage[y:y + h, x:x + w] = tile

    return montage


def visualize_compatibility_matrix(h_compat: np.ndarray, v_compat: np.ndarray,
                                     save_path: Optional[str] = None):
    """Visualize the compatibility matrices as heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    data_h = h_compat if h_compat.ndim == 2 else np.max(h_compat, axis=(1, 3))
    data_v = v_compat if v_compat.ndim == 2 else np.max(v_compat, axis=(1, 3))

    im1 = ax1.imshow(data_h, cmap='hot', aspect='auto')
    ax1.set_title('Horizontal Compatibility\n(i_right ↔ j_left)', fontsize=12)
    ax1.set_xlabel('Piece j')
    ax1.set_ylabel('Piece i')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(data_v, cmap='hot', aspect='auto')
    ax2.set_title('Vertical Compatibility\n(i_bottom ↔ j_top)', fontsize=12)
    ax2.set_xlabel('Piece j')
    ax2.set_ylabel('Piece i')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_placement_accuracy(predicted: Dict, gt: Dict, P: int, Q: int,
                                   save_path: Optional[str] = None):
    """
    Create a grid visualization showing which pieces are correctly placed.
    Green = correct position, Red = wrong position.
    """
    fig, ax = plt.subplots(figsize=(Q * 1.5, P * 1.5))

    grid_colors = np.zeros((P, Q, 3))

    for piece, gt_pos in gt.items():
        pred_pos = predicted.get(piece)
        if pred_pos == gt_pos:
            grid_colors[gt_pos[0], gt_pos[1]] = [0, 0.7, 0]  # Green
        else:
            grid_colors[gt_pos[0], gt_pos[1]] = [0.7, 0, 0]  # Red

    ax.imshow(grid_colors, aspect='auto')

    for r in range(P):
        for c in range(Q):
            # Find which piece should be at (r,c) and where it is predicted
            for piece, pos in gt.items():
                if pos == (r, c):
                    pred = predicted.get(piece, (-1, -1))
                    color = 'white'
                    ax.text(c, r, f'#{piece}\n→{pred}', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')
                    break

    ax.set_title('Placement Accuracy (Green=Correct, Red=Wrong)', fontsize=12)
    ax.set_xticks(range(Q))
    ax.set_yticks(range(P))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_feature_comparison(save_path: Optional[str] = None,
                                   results: Optional[Dict] = None):
    """
    Create a bar chart comparing different feature configurations.
    """
    if results is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = list(results.keys())
    metrics_names = ['placement_accuracy', 'neighbor_accuracy', 'rotation_accuracy']
    titles = ['Placement Accuracy', 'Neighbor Accuracy', 'Rotation Accuracy']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
        values = [results[config].get(metric, 0) for config in configs]
        bars = axes[idx].bar(range(len(configs)), values, color=colors[idx], alpha=0.8)
        axes[idx].set_xticks(range(len(configs)))
        axes[idx].set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
        axes[idx].set_ylabel('Accuracy')
        axes[idx].set_title(title, fontsize=12)
        axes[idx].set_ylim(0, 1.05)

        # Add value labels
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                           f'{val:.1%}', ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
