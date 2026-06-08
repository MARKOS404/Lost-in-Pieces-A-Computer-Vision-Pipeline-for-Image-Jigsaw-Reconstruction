# Lost in Pieces: Image Jigsaw Reconstruction

A complete pipeline for reconstructing images from shuffled and rotated rectangular tiles.

## Project Structure

| File | Description |
|------|-------------|
| `puzzle_generator.py` | Image segmentation, tile shuffling/rotation, ground truth |
| `feature_extraction.py` | Color, texture, local, deep-like, pixel border descriptors |
| `adjacency.py` | Compatibility scoring between tile sides |
| `reconstruction.py` | Greedy, local search, simulated annealing, Hungarian solvers |
| `evaluation.py` | Metrics computation and visualization |
| `main.py` | Full experimental suite (5 feature configs × 2 images × 2 rotation modes) |
| `demo.py` | Interactive demo with step-by-step visualization |
| `generate_report.py` | Technical report PDF generation |

## Quick Start

```bash
# Run demo (no rotation, 4x4 grid)
python demo.py --grid 4x4

# Run demo with rotation
python demo.py --grid 4x4 --rotate

# Run demo with custom image
python demo.py --image path/to/image.jpg --grid 3x3

# Run full experiments
python main.py
```

## Feature Descriptors (323 dimensions total)

- **Color HSV/Lab** (96 dims): Normalized histograms in two color spaces
- **Texture** (33 dims): Multi-scale filter bank (Gaussian derivatives, LoG, Sobel)
- **Gabor** (36 dims): Orientation-frequency sensitive texture
- **Local** (8 dims): Harris corners + gradient histogram (SIFT-style)
- **Deep-like CNN** (54 dims): Hierarchical multi-layer filter pipeline
- **Pixel Border** (96 dims): Direct 1-pixel edge values (strongest for jigsaw)

## Key Results

| Setting | Best Config | Placement | Neighbor |
|---------|------------|-----------|----------|
| 4×4 no rotation (geometric) | combined | **100%** | **100%** |
| 5×5 no rotation (geometric) | combined | **100%** | **100%** |
| 4×4 with rotation (geometric) | color_only | **100%** | **100%** |
| 4×4 no rotation (natural) | classical/combined | 0% | **83%** |

## Dependencies

- NumPy, SciPy, OpenCV, scikit-image, scikit-learn, matplotlib
- (Optional) PyTorch/TensorFlow for real CNN features
- reportlab (for PDF report generation)
