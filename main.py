"""
Main Pipeline: Lost in Pieces - Image Jigsaw Reconstruction
==============================================================
Complete pipeline that:
1. Generates puzzles from test images
2. Extracts features (color, texture, local, deep-like)
3. Computes adjacency/compatibility scores
4. Reconstructs images using greedy + simulated annealing
5. Evaluates results with quantitative metrics
6. Generates comprehensive visualizations
7. Compares feature configurations

Usage:
    python main.py
"""

import sys
import os
import time
import numpy as np
import cv2

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puzzle_generator import PuzzleGenerator, create_sample_image, create_natural_test_image
from feature_extraction import FeatureExtractor
from adjacency import AdjacencyModel, RotationAwareAdjacencyModel
from reconstruction import (
    GreedyReconstructor, LocalSearchReconstructor,
    SimulatedAnnealingReconstructor, HungarianReconstructor,
    try_all_offsets_alignment,
)
from evaluation import (
    ReconstructionEvaluator,
    visualize_puzzle_tiles,
    visualize_reconstruction_comparison,
    visualize_compatibility_matrix,
    visualize_placement_accuracy,
    visualize_feature_comparison,
)


def run_experiment(image: np.ndarray, image_name: str, P: int, Q: int,
                    allow_rotation: bool, feature_config: dict,
                    output_dir: str, method: str = 'greedy+sa') -> dict:
    """
    Run a single experiment: generate puzzle, extract features, reconstruct, evaluate.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    image_name : str
        Name identifier for this image.
    P, Q : int
        Grid dimensions.
    allow_rotation : bool
        Whether tiles are rotated.
    feature_config : dict
        Which features to enable (e.g., {'use_color': True, 'use_deep': False}).
    output_dir : str
        Directory to save outputs.
    method : str
        Reconstruction method.
    
    Returns
    -------
    results : dict with metrics and timing info.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_name = feature_config.get('name', 'default')
    print(f"\n{'='*60}")
    print(f"  Experiment: {image_name} | Grid: {P}x{Q} | "
          f"Rotation: {allow_rotation} | Config: {config_name}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Step 1: Generate Puzzle
    # -------------------------------------------------------------------------
    print("\n[1/5] Generating puzzle...")
    t0 = time.time()
    puzzle = PuzzleGenerator(image, P, Q, allow_rotation=allow_rotation, seed=42)
    tiles = puzzle.generate()
    gt_placement, gt_orientation = puzzle.get_ground_truth()
    t_puzzle = time.time() - t0
    print(f"  Generated {len(tiles)} tiles of size {puzzle.h}x{puzzle.w} in {t_puzzle:.2f}s")

    # Save shuffled tiles visualization
    visualize_puzzle_tiles(tiles, P, Q,
                            title=f"Shuffled Tiles - {image_name}",
                            save_path=os.path.join(output_dir, f'{image_name}_shuffled.png'))

    # -------------------------------------------------------------------------
    # Step 2: Extract Features
    # -------------------------------------------------------------------------
    print("\n[2/5] Extracting features...")
    t0 = time.time()

    border_width = max(3, min(puzzle.h, puzzle.w) // 6)

    extractor = FeatureExtractor(
        border_width=border_width,
        use_color=feature_config.get('use_color', True),
        use_texture=feature_config.get('use_texture', True),
        use_local=feature_config.get('use_local', True),
        use_deep=feature_config.get('use_deep', True),
        use_gabor=feature_config.get('use_gabor', True),
        use_pixel_border=feature_config.get('use_pixel_border', True),
    )
    features = extractor.extract_all(tiles)
    t_feat = time.time() - t0

    # Report feature dimensions
    sample_side = features['side_features'][0]['top']
    dims = {k: len(v) for k, v in sample_side.items()}
    print(f"  Feature dimensions per side: {dims}")
    print(f"  Feature extraction: {t_feat:.2f}s")

    # -------------------------------------------------------------------------
    # Step 3: Compute Compatibility Scores
    # -------------------------------------------------------------------------
    print("\n[3/5] Computing compatibility scores...")
    t0 = time.time()

    if allow_rotation:
        rot_model = RotationAwareAdjacencyModel(features, sigma=0.5)
        h_compat, v_compat = rot_model.compute_rotation_compatibility_matrices()
    else:
        adj_model = AdjacencyModel(features, sigma=0.5)
        h_compat, v_compat = adj_model.compute_compatibility_matrix()

    t_adj = time.time() - t0
    print(f"  Compatibility computed in {t_adj:.2f}s")

    # Visualize compatibility
    visualize_compatibility_matrix(h_compat, v_compat,
                                     save_path=os.path.join(output_dir,
                                                            f'{image_name}_{config_name}_compat.png'))

    # -------------------------------------------------------------------------
    # Step 4: Reconstruct
    # -------------------------------------------------------------------------
    print("\n[4/5] Reconstructing image...")
    t0 = time.time()

    # Greedy phase
    greedy = GreedyReconstructor(P, Q, allow_rotation)
    if allow_rotation:
        placement, orientations = greedy.reconstruct_with_rotation(h_compat, v_compat)
    else:
        placement, orientations = greedy.reconstruct(h_compat, v_compat)

    greedy_time = time.time() - t0
    print(f"  Greedy reconstruction: {greedy_time:.2f}s")

    # Evaluate greedy result
    evaluator = ReconstructionEvaluator(P, Q)
    placement = try_all_offsets_alignment(placement, gt_placement, P, Q)
    greedy_metrics = evaluator.evaluate_all(placement, gt_placement,
                                              orientations, gt_orientation)
    print(f"  Greedy metrics: placement={greedy_metrics['placement_accuracy']:.1%}, "
          f"neighbor={greedy_metrics['neighbor_accuracy']:.1%}, "
          f"rotation={greedy_metrics['rotation_accuracy']:.1%}")

    # Simulated Annealing refinement
    if 'sa' in method:
        print("  Running simulated annealing refinement...")
        t_sa = time.time()
        sa = SimulatedAnnealingReconstructor(
            P, Q, max_iterations=8000, T_init=1.0, T_min=0.001,
            cooling_rate=0.997, allow_rotation=allow_rotation
        )
        placement, orientations = sa.reconstruct(h_compat, v_compat,
                                                    placement, orientations)
        sa_time = time.time() - t_sa
        print(f"  SA refinement: {sa_time:.2f}s")

    # Local search refinement
    if 'local' in method:
        print("  Running local search refinement...")
        t_ls = time.time()
        local = LocalSearchReconstructor(P, Q, max_iterations=3000,
                                           allow_rotation=allow_rotation)
        placement, orientations = local.improve(placement, orientations,
                                                  h_compat, v_compat)
        ls_time = time.time() - t_ls
        print(f"  Local search: {ls_time:.2f}s")

    t_recon = time.time() - t0

    # Align to maximize placement accuracy
    placement = try_all_offsets_alignment(placement, gt_placement, P, Q)

    # -------------------------------------------------------------------------
    # Step 5: Evaluate
    # -------------------------------------------------------------------------
    print("\n[5/5] Evaluating reconstruction...")

    metrics = evaluator.evaluate_all(placement, gt_placement,
                                       orientations, gt_orientation)

    print(f"\n  Final Metrics:")
    print(f"    Placement Accuracy:  {metrics['placement_accuracy']:.1%}")
    print(f"    Neighbor Accuracy:   {metrics['neighbor_accuracy']:.1%}")
    print(f"    Rotation Accuracy:   {metrics['rotation_accuracy']:.1%}")
    print(f"    Direct Comparison:   {metrics['direct_comparison_accuracy']:.1%}")

    # Reconstruct image
    reconstructed = puzzle.reconstruct_from_solution(placement, orientations)

    # Save visualizations
    visualize_reconstruction_comparison(
        puzzle.image, reconstructed, tiles, P, Q, metrics,
        title=f"Reconstruction: {image_name} ({config_name})",
        save_path=os.path.join(output_dir, f'{image_name}_{config_name}_results.png')
    )

    visualize_placement_accuracy(
        placement, gt_placement, P, Q,
        save_path=os.path.join(output_dir, f'{image_name}_{config_name}_placement.png')
    )

    # Save reconstructed image
    cv2.imwrite(os.path.join(output_dir, f'{image_name}_{config_name}_reconstructed.png'),
                reconstructed)

    results = {
        **metrics,
        'time_puzzle': t_puzzle,
        'time_features': t_feat,
        'time_adjacency': t_adj,
        'time_reconstruction': t_recon,
        'greedy_metrics': greedy_metrics,
    }

    return results


def main():
    """Run the full experimental evaluation."""
    output_dir = '/home/claude/jigsaw_project/results'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  LOST IN PIECES: Image Jigsaw Reconstruction Challenge")
    print("=" * 70)

    # =========================================================================
    # Create test images
    # =========================================================================
    print("\nCreating test images...")
    images = {
        'geometric': create_sample_image(400, 400),
        'natural': create_natural_test_image(400, 400),
    }

    # Save original images
    for name, img in images.items():
        cv2.imwrite(os.path.join(output_dir, f'{name}_original.png'), img)

    # =========================================================================
    # Feature configurations to compare
    # =========================================================================
    feature_configs = [
        {
            'name': 'color_only',
            'use_color': True, 'use_texture': False, 'use_local': False,
            'use_deep': False, 'use_gabor': False, 'use_pixel_border': False,
        },
        {
            'name': 'deep_only',
            'use_color': False, 'use_texture': False, 'use_local': False,
            'use_deep': True, 'use_gabor': False, 'use_pixel_border': False,
        },
        {
            'name': 'pixel_border',
            'use_color': False, 'use_texture': False, 'use_local': False,
            'use_deep': False, 'use_gabor': False, 'use_pixel_border': True,
        },
        {
            'name': 'classical',
            'use_color': True, 'use_texture': True, 'use_local': True,
            'use_deep': False, 'use_gabor': True, 'use_pixel_border': True,
        },
        {
            'name': 'combined',
            'use_color': True, 'use_texture': True, 'use_local': True,
            'use_deep': True, 'use_gabor': True, 'use_pixel_border': True,
        },
    ]

    # =========================================================================
    # Experiment Set 1: Without rotation (4x4 grid)
    # =========================================================================
    all_results = {}

    print("\n" + "#" * 70)
    print("  EXPERIMENT SET 1: No Rotation (4x4 grid)")
    print("#" * 70)

    for config in feature_configs:
        for img_name in ['geometric', 'natural']:
            exp_key = f"{img_name}_norot_{config['name']}"
            results = run_experiment(
                images[img_name], img_name, P=4, Q=4,
                allow_rotation=False, feature_config=config,
                output_dir=os.path.join(output_dir, 'norot'),
                method='greedy+sa+local',
            )
            all_results[exp_key] = results

    # =========================================================================
    # Experiment Set 2: With rotation (4x4 grid)
    # =========================================================================
    print("\n" + "#" * 70)
    print("  EXPERIMENT SET 2: With Rotation (4x4 grid)")
    print("#" * 70)

    for config in feature_configs:
        for img_name in ['geometric', 'natural']:
            exp_key = f"{img_name}_rot_{config['name']}"
            results = run_experiment(
                images[img_name], img_name, P=4, Q=4,
                allow_rotation=True, feature_config=config,
                output_dir=os.path.join(output_dir, 'rot'),
                method='greedy+sa+local',
            )
            all_results[exp_key] = results

    # =========================================================================
    # Experiment Set 3: Different grid sizes (combined features, no rotation)
    # =========================================================================
    print("\n" + "#" * 70)
    print("  EXPERIMENT SET 3: Different Grid Sizes (Combined, No Rotation)")
    print("#" * 70)

    combined_config = feature_configs[-1]  # 'combined'
    for P, Q in [(2, 2), (3, 3), (4, 4), (5, 5)]:
        for img_name in ['geometric']:
            exp_key = f"{img_name}_{P}x{Q}_combined"
            results = run_experiment(
                images[img_name], img_name, P=P, Q=Q,
                allow_rotation=False, feature_config=combined_config,
                output_dir=os.path.join(output_dir, 'grid_sizes'),
                method='greedy+sa+local',
            )
            all_results[exp_key] = results

    # =========================================================================
    # Generate comparative visualizations
    # =========================================================================
    print("\n\nGenerating comparative analysis...")

    # Compare feature configs for no-rotation case
    norot_results = {}
    for config in feature_configs:
        config_name = config['name']
        key = f"geometric_norot_{config_name}"
        if key in all_results:
            norot_results[config_name] = all_results[key]

    visualize_feature_comparison(
        save_path=os.path.join(output_dir, 'feature_comparison_norot.png'),
        results=norot_results,
    )

    # Compare feature configs for rotation case
    rot_results = {}
    for config in feature_configs:
        config_name = config['name']
        key = f"geometric_rot_{config_name}"
        if key in all_results:
            rot_results[config_name] = all_results[key]

    visualize_feature_comparison(
        save_path=os.path.join(output_dir, 'feature_comparison_rot.png'),
        results=rot_results,
    )

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)

    print(f"\n{'Experiment':<45} {'Placement':>10} {'Neighbor':>10} {'Rotation':>10}")
    print("-" * 75)
    for key, res in sorted(all_results.items()):
        print(f"{key:<45} {res['placement_accuracy']:>10.1%} "
              f"{res['neighbor_accuracy']:>10.1%} {res['rotation_accuracy']:>10.1%}")

    # Save summary to text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("LOST IN PIECES - Experiment Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Experiment':<45} {'Placement':>10} {'Neighbor':>10} {'Rotation':>10}\n")
        f.write("-" * 75 + "\n")
        for key, res in sorted(all_results.items()):
            f.write(f"{key:<45} {res['placement_accuracy']:>10.1%} "
                    f"{res['neighbor_accuracy']:>10.1%} {res['rotation_accuracy']:>10.1%}\n")

    print(f"\n  All results saved to: {output_dir}/")
    print("  Done!")

    return all_results


if __name__ == '__main__':
    results = main()
