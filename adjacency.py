"""
Adjacency Modeling Module
==========================
Computes compatibility scores between puzzle piece sides using
multiple descriptor families. Supports both distance-based and
learned compatibility scoring.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two feature vectors."""
    return np.linalg.norm(a - b)


def chi_squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-squared distance for histograms."""
    denom = a + b + 1e-10
    return 0.5 * np.sum((a - b) ** 2 / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def gaussian_similarity(dist: float, sigma: float = 1.0) -> float:
    """Convert distance to similarity via Gaussian kernel."""
    return np.exp(-dist ** 2 / (2 * sigma ** 2))


class AdjacencyModel:
    """
    Computes pairwise compatibility scores between tile sides.
    
    Combines multiple descriptor families with configurable weights
    at both the side-level and tile-level.
    """

    # Descriptor families and their distance functions
    DISTANCE_FUNCS = {
        'color_hsv': chi_squared_distance,
        'color_lab': chi_squared_distance,
        'texture': euclidean_distance,
        'local': euclidean_distance,
        'deep': euclidean_distance,
        'gabor': euclidean_distance,
        'pixel_border': euclidean_distance,
    }

    # Default weights for each descriptor family
    DEFAULT_WEIGHTS = {
        'color_hsv': {'side': 1.0, 'tile': 0.3},
        'color_lab': {'side': 1.0, 'tile': 0.3},
        'texture': {'side': 0.8, 'tile': 0.2},
        'local': {'side': 0.5, 'tile': 0.0},
        'deep': {'side': 1.0, 'tile': 0.5},
        'gabor': {'side': 0.7, 'tile': 0.2},
        'pixel_border': {'side': 3.0, 'tile': 0.0},  # High weight: very discriminative
    }

    def __init__(self, features: Dict, weights: Optional[Dict] = None,
                 sigma: float = 0.5):
        """
        Parameters
        ----------
        features : dict
            Output of FeatureExtractor.extract_all().
        weights : dict or None
            Custom weights per descriptor family. 
        sigma : float
            Bandwidth for Gaussian similarity conversion.
        """
        self.tile_features = features['tile_features']
        self.side_features = features['side_features']
        self.N = len(self.tile_features)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.sigma = sigma

        # Precompute normalized distances for adaptive sigma
        self._precompute_stats()

    def _precompute_stats(self):
        """Precompute distance statistics for normalization."""
        self.dist_stats = {}
        sides = ['top', 'right', 'bottom', 'left']

        for desc_name in self.side_features[0]['top'].keys():
            distances = []
            # Sample some distances to estimate scale
            n_sample = min(self.N, 30)
            indices = np.random.choice(self.N, n_sample, replace=False) \
                if self.N > n_sample else np.arange(self.N)

            dist_func = self.DISTANCE_FUNCS.get(desc_name, euclidean_distance)

            for i in indices:
                for j in indices:
                    if i != j:
                        for s1 in sides:
                            for s2 in sides:
                                f1 = self.side_features[i][s1].get(desc_name)
                                f2 = self.side_features[j][s2].get(desc_name)
                                if f1 is not None and f2 is not None:
                                    distances.append(dist_func(f1, f2))

            if distances:
                self.dist_stats[desc_name] = {
                    'mean': np.mean(distances),
                    'std': np.std(distances) + 1e-10,
                    'median': np.median(distances),
                }
            else:
                self.dist_stats[desc_name] = {'mean': 1.0, 'std': 1.0, 'median': 1.0}

    def side_compatibility(self, i: int, s_i: str, j: int, s_j: str) -> float:
        """
        Compute compatibility score between side s_i of piece i and
        side s_j of piece j.
        
        Returns
        -------
        score : float
            Higher = more compatible (more likely neighbors).
        """
        total = 0.0

        for desc_name in self.side_features[i][s_i].keys():
            w = self.weights.get(desc_name, {'side': 0.5, 'tile': 0.1})
            w_side = w.get('side', 0.5)
            w_tile = w.get('tile', 0.1)
            dist_func = self.DISTANCE_FUNCS.get(desc_name, euclidean_distance)
            stats = self.dist_stats.get(desc_name, {'std': 1.0})

            # Side-level compatibility
            f_side_i = self.side_features[i][s_i].get(desc_name)
            f_side_j = self.side_features[j][s_j].get(desc_name)

            if f_side_i is not None and f_side_j is not None:
                d_side = dist_func(f_side_i, f_side_j)
                # Normalize by scale
                d_normalized = d_side / stats['std']
                c_side = np.exp(-d_normalized ** 2 / (2 * self.sigma ** 2))
                total += w_side * c_side

            # Tile-level compatibility
            if w_tile > 0:
                f_tile_i = self.tile_features[i].get(desc_name)
                f_tile_j = self.tile_features[j].get(desc_name)
                if f_tile_i is not None and f_tile_j is not None:
                    d_tile = dist_func(f_tile_i, f_tile_j)
                    d_normalized = d_tile / stats['std']
                    c_tile = np.exp(-d_normalized ** 2 / (2 * self.sigma ** 2))
                    total += w_tile * c_tile

        return total

    def compute_all_compatibilities(self, orientations: Optional[Dict[int, int]] = None) \
            -> Dict[Tuple, float]:
        """
        Compute compatibility scores for all relevant side pairings.
        
        For efficiency, we only compute scores between sides that could
        be adjacent (right<->left, bottom<->top).
        
        Parameters
        ----------
        orientations : dict or None
            If provided, maps piece index -> rotation angle. The side
            features are assumed to already correspond to the rotated state.
        
        Returns
        -------
        scores : dict
            Maps (i, s_i, j, s_j) -> compatibility score.
        """
        scores = {}

        # For horizontal adjacency: right side of i matches left side of j
        # For vertical adjacency: bottom side of i matches top side of j
        adj_pairs = [
            ('right', 'left'),
            ('bottom', 'top'),
        ]

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                for s_i, s_j in adj_pairs:
                    key = (i, s_i, j, s_j)
                    scores[key] = self.side_compatibility(i, s_i, j, s_j)

        return scores

    def compute_compatibility_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute compatibility matrices for horizontal and vertical adjacency.
        
        Returns
        -------
        h_compat : np.ndarray of shape (N, N)
            h_compat[i,j] = compatibility of piece i's right with piece j's left
        v_compat : np.ndarray of shape (N, N)
            v_compat[i,j] = compatibility of piece i's bottom with piece j's top
        """
        h_compat = np.zeros((self.N, self.N))
        v_compat = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                h_compat[i, j] = self.side_compatibility(i, 'right', j, 'left')
                v_compat[i, j] = self.side_compatibility(i, 'bottom', j, 'top')

        return h_compat, v_compat


class RotationAwareAdjacencyModel:
    """
    Extends the adjacency model to handle rotation estimation.
    For each candidate piece placement and orientation, computes
    the compatibility accounting for the rotated side assignments.
    """

    SIDES = ['top', 'right', 'bottom', 'left']
    ROTATIONS = [0, 90, 180, 270]

    def __init__(self, features: Dict, weights: Optional[Dict] = None,
                 sigma: float = 0.5):
        self.base_model = AdjacencyModel(features, weights, sigma)
        self.N = self.base_model.N
        self.features = features

    def get_rotated_side(self, side: str, angle: int) -> str:
        """
        If a tile is rotated by `angle` CCW, what was originally on `side`
        is now on a different side. This returns which side of the ROTATED
        tile corresponds to the given side.
        
        Equivalently: after rotating by `angle`, what appears on `side`?
        That content originally came from rotate_back(side, angle).
        """
        sides = self.SIDES
        idx = sides.index(side)
        steps = angle // 90
        # After CCW rotation by `angle`, content at original position
        # (idx + steps) % 4 is now at position idx.
        return sides[(idx + steps) % 4]

    def compatibility_with_rotation(self, i: int, angle_i: int,
                                      j: int, angle_j: int,
                                      direction: str) -> float:
        """
        Compute compatibility between piece i (rotated by angle_i) and
        piece j (rotated by angle_j), where piece i is `direction` of piece j.
        
        Parameters
        ----------
        direction : str
            'left' means i is to the left of j, 'above' means i is above j.
        """
        if direction == 'left':
            # i's right side faces j's left side
            # After rotation, what appears on i's right originally came from...
            s_i = self.get_rotated_side('right', angle_i)
            s_j = self.get_rotated_side('left', angle_j)
            return self.base_model.side_compatibility(i, s_i, j, s_j)
        elif direction == 'above':
            s_i = self.get_rotated_side('bottom', angle_i)
            s_j = self.get_rotated_side('top', angle_j)
            return self.base_model.side_compatibility(i, s_i, j, s_j)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def compute_rotation_compatibility_matrices(self):
        """
        Precompute compatibility for all piece pairs, all orientation pairs,
        for both horizontal and vertical adjacency.
        
        Returns
        -------
        h_compat : np.ndarray of shape (N, 4, N, 4)
            h_compat[i, ai, j, aj] = score for i(rot=ai*90) right-of j(rot=aj*90)
        v_compat : np.ndarray of shape (N, 4, N, 4)
        """
        N = self.N
        h_compat = np.zeros((N, 4, N, 4))
        v_compat = np.zeros((N, 4, N, 4))

        for i in range(N):
            for ai in range(4):
                angle_i = ai * 90
                for j in range(N):
                    if i == j:
                        continue
                    for aj in range(4):
                        angle_j = aj * 90
                        h_compat[i, ai, j, aj] = self.compatibility_with_rotation(
                            i, angle_i, j, angle_j, 'left')
                        v_compat[i, ai, j, aj] = self.compatibility_with_rotation(
                            i, angle_i, j, angle_j, 'above')

        return h_compat, v_compat


class LearnedAdjacencyModel:
    """
    Uses a simple logistic regression classifier trained on true/false
    neighbor pairs to learn optimal weights for combining descriptor families.
    """

    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_training_data(self, features: Dict, gt_placement: Dict,
                                gt_orientation: Dict, P: int, Q: int):
        """
        Build training data from ground truth adjacencies.
        
        Returns positive (true neighbor) and negative (non-neighbor) feature vectors.
        """
        side_features = features['side_features']
        tile_features = features['tile_features']
        N = len(side_features)

        # Build ground truth grid
        grid = np.full((P, Q), -1, dtype=int)
        for idx, (r, c) in gt_placement.items():
            grid[r, c] = idx

        positive_pairs = []
        negative_pairs = []

        # Positive: true horizontal neighbors
        for r in range(P):
            for c in range(Q - 1):
                i = grid[r, c]
                j = grid[r, c + 1]
                if i >= 0 and j >= 0:
                    positive_pairs.append((i, 'right', j, 'left'))

        # Positive: true vertical neighbors
        for r in range(P - 1):
            for c in range(Q):
                i = grid[r, c]
                j = grid[r + 1, c]
                if i >= 0 and j >= 0:
                    positive_pairs.append((i, 'bottom', j, 'top'))

        # Negative: random non-neighbor pairs
        rng = np.random.RandomState(42)
        n_neg = len(positive_pairs) * 3
        for _ in range(n_neg):
            i = rng.randint(N)
            j = rng.randint(N)
            if i != j:
                s1 = ['right', 'bottom'][rng.randint(2)]
                s2 = 'left' if s1 == 'right' else 'top'
                negative_pairs.append((i, s1, j, s2))

        return positive_pairs, negative_pairs

    def _compute_feature_vector(self, features: Dict, i: int, s_i: str,
                                  j: int, s_j: str) -> np.ndarray:
        """Compute a feature vector for a candidate pair."""
        side_feats = features['side_features']
        tile_feats = features['tile_features']
        vec = []

        for desc_name in side_feats[i][s_i].keys():
            f_si = side_feats[i][s_i][desc_name]
            f_sj = side_feats[j][s_j][desc_name]

            # Side-level distances
            vec.append(euclidean_distance(f_si, f_sj))
            vec.append(cosine_similarity(f_si, f_sj))

            # Tile-level distance (if available)
            if desc_name in tile_feats[i] and desc_name in tile_feats[j]:
                vec.append(euclidean_distance(tile_feats[i][desc_name],
                                               tile_feats[j][desc_name]))
            else:
                vec.append(0.0)

        return np.array(vec)

    def fit(self, features: Dict, gt_placement: Dict, gt_orientation: Dict,
            P: int, Q: int):
        """Train the classifier on ground truth data."""
        pos_pairs, neg_pairs = self.prepare_training_data(
            features, gt_placement, gt_orientation, P, Q)

        X = []
        y = []

        for i, s_i, j, s_j in pos_pairs:
            vec = self._compute_feature_vector(features, i, s_i, j, s_j)
            X.append(vec)
            y.append(1)

        for i, s_i, j, s_j in neg_pairs:
            vec = self._compute_feature_vector(features, i, s_i, j, s_j)
            X.append(vec)
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True

        # Report accuracy
        acc = self.classifier.score(X_scaled, y)
        print(f"  Learned adjacency model training accuracy: {acc:.3f}")

    def predict_score(self, features: Dict, i: int, s_i: str,
                       j: int, s_j: str) -> float:
        """Predict compatibility score for a candidate pair."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        vec = self._compute_feature_vector(features, i, s_i, j, s_j)
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))
        return self.classifier.predict_proba(vec_scaled)[0, 1]
