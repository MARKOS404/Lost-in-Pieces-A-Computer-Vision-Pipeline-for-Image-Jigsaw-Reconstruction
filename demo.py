"""
Interactive Demo: Lost in Pieces
==================================
A self-contained demo that:
1. Creates a test image with visual elements
2. Cuts it into pieces, shuffles, and optionally rotates them
3. Reconstructs step-by-step with visualization of the process
4. Shows before/after comparison with metrics

Usage:
    python demo.py [--image PATH] [--grid PxQ] [--rotate] [--seed N]
"""

import sys
import os
import argparse
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_generator import PuzzleGenerator, create_sample_image, create_natural_test_image
from feature_extraction import FeatureExtractor
from adjacency import AdjacencyModel, RotationAwareAdjacencyModel
from reconstruction import GreedyReconstructor, SimulatedAnnealingReconstructor, try_all_offsets_alignment
from evaluation import ReconstructionEvaluator, create_montage


def create_step_by_step_visualization(puzzle, tiles, placement_history,
                                        P, Q, output_path):
    """
    Create a multi-panel figure showing the reconstruction step by step.
    """
    n_steps = min(len(placement_history), 8)
    step_indices = np.linspace(0, len(placement_history) - 1, n_steps).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Greedy Reconstruction: Step-by-Step', fontsize=16, fontweight='bold')

    for panel_idx, step_idx in enumerate(step_indices):
        r_ax = panel_idx // 4
        c_ax = panel_idx % 4
        ax = axes[r_ax, c_ax]

        # Build partial reconstruction at this step
        placement, orientations = placement_history[step_idx]
        partial = np.ones((puzzle.H, puzzle.W, 3), dtype=np.uint8) * 180

        for piece_idx, (r, c) in placement.items():
            tile = tiles[piece_idx].copy()
            angle = orientations.get(piece_idx, 0)
            if angle == 90:
                tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                tile = cv2.rotate(tile, cv2.ROTATE_180)
            elif angle == 270:
                tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)

            th, tw = tile.shape[:2]
            y = r * puzzle.h
            x = c * puzzle.w
            if th == puzzle.h and tw == puzzle.w:
                partial[y:y + puzzle.h, x:x + puzzle.w] = tile

        ax.imshow(cv2.cvtColor(partial, cv2.COLOR_BGR2RGB))
        n_placed = len(placement)
        ax.set_title(f'Step {step_idx + 1}: {n_placed}/{P * Q} pieces', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_demo(image_path=None, P=4, Q=4, allow_rotation=True, seed=42,
             output_dir='/home/claude/jigsaw_project/demo_output'):
    """Run the interactive demo."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("   LOST IN PIECES - Interactive Demo")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Load or create image
    # -------------------------------------------------------------------------
    if image_path and os.path.exists(image_path):
        print(f"\nLoading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print("  Failed to load image, using default.")
            image = create_sample_image(400, 400)
    else:
        print("\nCreating demo image with distinctive patterns...")
        image = create_sample_image(400, 400)

    cv2.imwrite(os.path.join(output_dir, 'original.png'), image)
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # -------------------------------------------------------------------------
    # Generate Puzzle
    # -------------------------------------------------------------------------
    print(f"\nGenerating {P}x{Q} puzzle (rotation={allow_rotation})...")
    puzzle = PuzzleGenerator(image, P, Q, allow_rotation=allow_rotation, seed=seed)
    tiles = puzzle.generate()
    gt_placement, gt_orientation = puzzle.get_ground_truth()

    # Save shuffled state
    montage = create_montage(tiles, P, Q, gap=4)
    cv2.imwrite(os.path.join(output_dir, 'shuffled_montage.png'), montage)

    # Visualize individual tiles
    fig, axes = plt.subplots(P, Q, figsize=(Q * 2.5, P * 2.5))
    fig.suptitle(f'Shuffled Puzzle Pieces ({P}×{Q})', fontsize=14, fontweight='bold')
    if P == 1 and Q == 1:
        axes = np.array([[axes]])
    elif P == 1:
        axes = axes.reshape(1, -1)
    elif Q == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(P * Q):
        r, c = idx // Q, idx % Q
        tile_rgb = cv2.cvtColor(tiles[idx], cv2.COLOR_BGR2RGB)
        axes[r, c].imshow(tile_rgb)
        axes[r, c].set_title(f'Piece #{idx}', fontsize=9)
        axes[r, c].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shuffled_tiles.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created {len(tiles)} pieces of size {puzzle.h}x{puzzle.w}")

    # -------------------------------------------------------------------------
    # Extract Features
    # -------------------------------------------------------------------------
    print("\nExtracting features (all descriptors)...")
    t0 = time.time()
    border_width = max(3, min(puzzle.h, puzzle.w) // 6)
    extractor = FeatureExtractor(border_width=border_width)
    features = extractor.extract_all(tiles)
    t_feat = time.time() - t0
    print(f"  Done in {t_feat:.2f}s")

    # Show feature dimensions
    side_feats = features['side_features'][0]['top']
    total_dim = sum(len(v) for v in side_feats.values())
    print(f"  Total feature dimension per side: {total_dim}")
    for name, vec in side_feats.items():
        print(f"    {name}: {len(vec)} dims")

    # -------------------------------------------------------------------------
    # Compute Compatibility
    # -------------------------------------------------------------------------
    print("\nComputing compatibility scores...")
    t0 = time.time()
    if allow_rotation:
        rot_model = RotationAwareAdjacencyModel(features, sigma=0.5)
        h_compat, v_compat = rot_model.compute_rotation_compatibility_matrices()
    else:
        adj_model = AdjacencyModel(features, sigma=0.5)
        h_compat, v_compat = adj_model.compute_compatibility_matrix()
    t_compat = time.time() - t0
    print(f"  Done in {t_compat:.2f}s")

    # -------------------------------------------------------------------------
    # Reconstruct (Greedy)
    # -------------------------------------------------------------------------
    print("\nPhase 1: Greedy reconstruction...")
    t0 = time.time()
    greedy = GreedyReconstructor(P, Q, allow_rotation)
    if allow_rotation:
        placement, orientations = greedy.reconstruct_with_rotation(h_compat, v_compat)
    else:
        placement, orientations = greedy.reconstruct(h_compat, v_compat)

    t_greedy = time.time() - t0
    placement = try_all_offsets_alignment(placement, gt_placement, P, Q)
    evaluator = ReconstructionEvaluator(P, Q)
    greedy_metrics = evaluator.evaluate_all(placement, gt_placement,
                                              orientations, gt_orientation)
    print(f"  Greedy: placement={greedy_metrics['placement_accuracy']:.1%}, "
          f"neighbor={greedy_metrics['neighbor_accuracy']:.1%} ({t_greedy:.2f}s)")

    # Save greedy reconstruction
    greedy_img = puzzle.reconstruct_from_solution(placement, orientations)
    cv2.imwrite(os.path.join(output_dir, 'reconstruction_greedy.png'), greedy_img)

    # -------------------------------------------------------------------------
    # Refine (Simulated Annealing)
    # -------------------------------------------------------------------------
    print("\nPhase 2: Simulated annealing refinement...")
    t0 = time.time()
    sa = SimulatedAnnealingReconstructor(
        P, Q, max_iterations=8000, T_init=1.0, T_min=0.001,
        cooling_rate=0.997, allow_rotation=allow_rotation
    )
    placement, orientations = sa.reconstruct(h_compat, v_compat,
                                                placement, orientations)
    t_sa = time.time() - t0
    placement = try_all_offsets_alignment(placement, gt_placement, P, Q)

    final_metrics = evaluator.evaluate_all(placement, gt_placement,
                                              orientations, gt_orientation)
    print(f"  SA: placement={final_metrics['placement_accuracy']:.1%}, "
          f"neighbor={final_metrics['neighbor_accuracy']:.1%} ({t_sa:.2f}s)")

    # Save final reconstruction
    final_img = puzzle.reconstruct_from_solution(placement, orientations)
    cv2.imwrite(os.path.join(output_dir, 'reconstruction_final.png'), final_img)

    # -------------------------------------------------------------------------
    # Create comprehensive output figure
    # -------------------------------------------------------------------------
    print("\nGenerating output visualizations...")

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Lost in Pieces: Jigsaw Reconstruction Demo',
                 fontsize=18, fontweight='bold', y=0.98)

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(puzzle.image, cv2.COLOR_BGR2RGB))
    ax.set_title('1. Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Shuffled montage
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
    ax.set_title(f'2. Shuffled ({P}×{Q}, rot={allow_rotation})', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Greedy result
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(cv2.cvtColor(greedy_img, cv2.COLOR_BGR2RGB))
    ax.set_title(f'3. Greedy ({greedy_metrics["placement_accuracy"]:.0%})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Final result
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    ax.set_title(f'4. Final ({final_metrics["placement_accuracy"]:.0%})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Difference map
    ax = fig.add_subplot(gs[1, 0:2])
    if puzzle.image.shape == final_img.shape:
        diff = np.abs(puzzle.image.astype(float) - final_img.astype(float))
        diff_vis = (diff / max(diff.max(), 1) * 255).astype(np.uint8)
        ax.imshow(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB))
    ax.set_title('5. Absolute Difference (Original vs Final)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Placement grid
    ax = fig.add_subplot(gs[1, 2])
    grid_vis = np.zeros((P, Q, 3))
    for piece, gt_pos in gt_placement.items():
        pred_pos = placement.get(piece)
        if pred_pos == gt_pos:
            grid_vis[gt_pos[0], gt_pos[1]] = [0, 0.8, 0]
        else:
            grid_vis[gt_pos[0], gt_pos[1]] = [0.8, 0, 0]
    ax.imshow(grid_vis, aspect='auto')
    for r in range(P):
        for c in range(Q):
            ax.text(c, r, '✓' if grid_vis[r, c, 1] > 0 else '✗',
                    ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    ax.set_title('6. Placement Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(range(Q))
    ax.set_yticks(range(P))

    # Metrics panel
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    metrics_text = (
        f"METRICS\n{'─' * 25}\n\n"
        f"Placement:  {final_metrics['placement_accuracy']:.1%}\n"
        f"Neighbor:   {final_metrics['neighbor_accuracy']:.1%}\n"
        f"Rotation:   {final_metrics['rotation_accuracy']:.1%}\n"
        f"Combined:   {final_metrics['direct_comparison_accuracy']:.1%}\n\n"
        f"{'─' * 25}\n"
        f"TIMING\n{'─' * 25}\n\n"
        f"Features:  {t_feat:.1f}s\n"
        f"Compat:    {t_compat:.1f}s\n"
        f"Greedy:    {t_greedy:.1f}s\n"
        f"SA:        {t_sa:.1f}s\n"
        f"Total:     {t_feat+t_compat+t_greedy+t_sa:.1f}s"
    )
    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='orange', alpha=0.8))

    # Method comparison bar chart
    ax = fig.add_subplot(gs[2, 0:2])
    methods = ['Greedy', 'Greedy + SA']
    placement_acc = [greedy_metrics['placement_accuracy'],
                     final_metrics['placement_accuracy']]
    neighbor_acc = [greedy_metrics['neighbor_accuracy'],
                    final_metrics['neighbor_accuracy']]

    x = np.arange(len(methods))
    width = 0.3
    bars1 = ax.bar(x - width / 2, placement_acc, width, label='Placement', color='#2196F3')
    bars2 = ax.bar(x + width / 2, neighbor_acc, width, label='Neighbor', color='#4CAF50')
    ax.set_ylabel('Accuracy')
    ax.set_title('7. Method Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.15)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{bar.get_height():.0%}', ha='center', fontsize=9)

    # Feature contribution (side-by-side of reconstruction with grid)
    ax = fig.add_subplot(gs[2, 2:4])
    recon_rgb = cv2.cvtColor(final_img.copy(), cv2.COLOR_BGR2RGB)
    h_tile = puzzle.h
    w_tile = puzzle.w
    for r in range(1, P):
        cv2.line(recon_rgb, (0, r * h_tile), (recon_rgb.shape[1], r * h_tile), (255, 50, 50), 2)
    for c in range(1, Q):
        cv2.line(recon_rgb, (c * w_tile, 0), (c * w_tile, recon_rgb.shape[0]), (255, 50, 50), 2)
    ax.imshow(recon_rgb)
    ax.set_title('8. Final Reconstruction with Grid Overlay', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.savefig(os.path.join(output_dir, 'demo_complete.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  All outputs saved to: {output_dir}/")
    print(f"  Main figure: demo_complete.png")
    print("\nDemo complete!")

    return final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jigsaw Puzzle Reconstruction Demo')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--grid', type=str, default='4x4', help='Grid size PxQ')
    parser.add_argument('--rotate', action='store_true', help='Enable rotation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    P, Q = map(int, args.grid.split('x'))
    run_demo(image_path=args.image, P=P, Q=Q, allow_rotation=args.rotate, seed=args.seed)
