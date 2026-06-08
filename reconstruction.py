"""
Global Reconstruction Module
==============================
Implements several algorithms for solving the jigsaw puzzle:
1. Greedy placement (best-first construction)
2. Local search with swaps
3. Simulated annealing
4. Hungarian algorithm for simplified (no-rotation) case

The reconstruction handles both placement (π) and orientation (θ) estimation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import time


class GreedyReconstructor:
    """
    Greedy best-first puzzle reconstruction.
    
    Strategy:
    1. Place the piece with highest average compatibility in the center.
    2. Iteratively place the piece+orientation that best matches already-placed neighbors.
    3. Use BFS expansion from the initial piece.
    """

    def __init__(self, P: int, Q: int, allow_rotation: bool = True):
        self.P = P
        self.Q = Q
        self.N = P * Q
        self.allow_rotation = allow_rotation
        self.ROTATIONS = [0, 90, 180, 270] if allow_rotation else [0]

    def reconstruct(self, h_compat: np.ndarray, v_compat: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Greedy reconstruction without rotation (uses precomputed compat matrices).
        Tries multiple starting pieces and picks the best overall solution.
        """
        N = self.N
        P, Q = self.P, self.Q

        # Try top starting pieces
        avg_compat = np.zeros(N)
        for i in range(N):
            avg_compat[i] = np.sum(np.maximum(h_compat[i, :], h_compat[:, i])) + \
                            np.sum(np.maximum(v_compat[i, :], v_compat[:, i]))
        
        n_tries = min(N, 5)
        top_pieces = np.argsort(avg_compat)[-n_tries:]

        best_placement = None
        best_orientations = None
        best_total = -np.inf

        for start_piece in top_pieces:
            placement, orientations, grid = self._greedy_from_start(
                start_piece, 0, 0, h_compat, v_compat)
            
            # Compute total score
            total = 0.0
            for r in range(P):
                for c in range(Q - 1):
                    if grid[r, c] >= 0 and grid[r, c + 1] >= 0:
                        total += h_compat[grid[r, c], grid[r, c + 1]]
            for r in range(P - 1):
                for c in range(Q):
                    if grid[r, c] >= 0 and grid[r + 1, c] >= 0:
                        total += v_compat[grid[r, c], grid[r + 1, c]]
            
            if total > best_total:
                best_total = total
                best_placement = placement
                best_orientations = orientations

        return best_placement, best_orientations

    def _greedy_from_start(self, start_piece, start_r, start_c,
                            h_compat, v_compat):
        """Run greedy placement from a specific starting piece and position."""
        N = self.N
        P, Q = self.P, self.Q

        placement = {start_piece: (start_r, start_c)}
        orientations = {start_piece: 0}
        grid = np.full((P, Q), -1, dtype=int)
        grid[start_r, start_c] = start_piece
        placed = {start_piece}

        # BFS frontier: positions adjacent to placed pieces
        frontier = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < P and 0 <= nc < Q:
                frontier.add((nr, nc))

        while frontier and len(placed) < N:
            best_score = -np.inf
            best_piece = -1
            best_pos = None

            for r, c in frontier:
                # Score each unplaced piece at this position
                for piece in range(N):
                    if piece in placed:
                        continue

                    score = 0.0
                    n_neighbors = 0

                    # Check left neighbor
                    if c > 0 and grid[r, c - 1] >= 0:
                        left_piece = grid[r, c - 1]
                        score += h_compat[left_piece, piece]
                        n_neighbors += 1

                    # Check right neighbor
                    if c < Q - 1 and grid[r, c + 1] >= 0:
                        right_piece = grid[r, c + 1]
                        score += h_compat[piece, right_piece]
                        n_neighbors += 1

                    # Check top neighbor
                    if r > 0 and grid[r - 1, c] >= 0:
                        top_piece = grid[r - 1, c]
                        score += v_compat[top_piece, piece]
                        n_neighbors += 1

                    # Check bottom neighbor
                    if r < P - 1 and grid[r + 1, c] >= 0:
                        bottom_piece = grid[r + 1, c]
                        score += v_compat[piece, bottom_piece]
                        n_neighbors += 1

                    if n_neighbors > 0:
                        score /= n_neighbors  # Normalize by neighbors

                    if score > best_score:
                        best_score = score
                        best_piece = piece
                        best_pos = (r, c)

            if best_piece >= 0 and best_pos is not None:
                r, c = best_pos
                placement[best_piece] = (r, c)
                orientations[best_piece] = 0
                grid[r, c] = best_piece
                placed.add(best_piece)
                frontier.discard((r, c))

                # Add new frontier positions
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < P and 0 <= nc < Q and grid[nr, nc] < 0:
                        frontier.add((nr, nc))
            else:
                # No valid placement found, break and fill randomly
                break

        # Fill any remaining pieces randomly
        remaining = [p for p in range(N) if p not in placed]
        empty_positions = [(r, c) for r in range(P) for c in range(Q) if grid[r, c] < 0]
        np.random.shuffle(remaining)
        for piece, (r, c) in zip(remaining, empty_positions):
            placement[piece] = (r, c)
            orientations[piece] = 0
            grid[r, c] = piece

        return placement, orientations, grid

    def reconstruct_with_rotation(self, h_compat_rot: np.ndarray,
                                    v_compat_rot: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Greedy reconstruction WITH rotation estimation.
        Tries multiple starting pieces/orientations and picks the best.
        """
        N = self.N
        P, Q = self.P, self.Q

        # Score each piece-orientation as starting candidate
        avg_score = np.zeros((N, 4))
        for i in range(N):
            for ai in range(4):
                s = 0.0
                cnt = 0
                for j in range(N):
                    if j != i:
                        s += np.max(h_compat_rot[i, ai, j, :])
                        s += np.max(v_compat_rot[i, ai, j, :])
                        cnt += 2
                avg_score[i, ai] = s / max(cnt, 1)

        # Try top candidates
        n_tries = min(N, 3)
        flat_indices = np.argsort(avg_score.ravel())[-n_tries:]

        best_placement = None
        best_orientations = None
        best_total = -np.inf

        for flat_idx in flat_indices:
            start_piece = flat_idx // 4
            start_rot_idx = flat_idx % 4
            placement, orientations, grid, grid_rot = self._greedy_rot_from_start(
                start_piece, start_rot_idx, 0, 0, h_compat_rot, v_compat_rot)

            # Score
            total = 0.0
            for r in range(P):
                for c in range(Q - 1):
                    if grid[r, c] >= 0 and grid[r, c + 1] >= 0:
                        total += h_compat_rot[grid[r, c], grid_rot[r, c],
                                               grid[r, c + 1], grid_rot[r, c + 1]]
            for r in range(P - 1):
                for c in range(Q):
                    if grid[r, c] >= 0 and grid[r + 1, c] >= 0:
                        total += v_compat_rot[grid[r, c], grid_rot[r, c],
                                               grid[r + 1, c], grid_rot[r + 1, c]]

            if total > best_total:
                best_total = total
                best_placement = placement
                best_orientations = orientations

        return best_placement, best_orientations

    def _greedy_rot_from_start(self, start_piece, start_rot_idx, start_r, start_c,
                                 h_compat_rot, v_compat_rot):
        """Greedy with rotation from a specific starting config."""
        N = self.N
        P, Q = self.P, self.Q

        placement = {start_piece: (start_r, start_c)}
        orientations = {start_piece: start_rot_idx * 90}
        grid = np.full((P, Q), -1, dtype=int)
        grid_rot = np.full((P, Q), -1, dtype=int)
        grid[start_r, start_c] = start_piece
        grid_rot[start_r, start_c] = start_rot_idx
        placed = {start_piece}

        frontier = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < P and 0 <= nc < Q:
                frontier.add((nr, nc))

        while frontier and len(placed) < N:
            best_score = -np.inf
            best_piece = -1
            best_pos = None
            best_rot = 0

            for r, c in frontier:
                for piece in range(N):
                    if piece in placed:
                        continue

                    for rot_idx in range(4):
                        score = 0.0
                        n_neighbors = 0

                        if c > 0 and grid[r, c - 1] >= 0:
                            left_p = grid[r, c - 1]
                            left_r = grid_rot[r, c - 1]
                            score += h_compat_rot[left_p, left_r, piece, rot_idx]
                            n_neighbors += 1

                        if c < Q - 1 and grid[r, c + 1] >= 0:
                            right_p = grid[r, c + 1]
                            right_r = grid_rot[r, c + 1]
                            score += h_compat_rot[piece, rot_idx, right_p, right_r]
                            n_neighbors += 1

                        if r > 0 and grid[r - 1, c] >= 0:
                            top_p = grid[r - 1, c]
                            top_r = grid_rot[r - 1, c]
                            score += v_compat_rot[top_p, top_r, piece, rot_idx]
                            n_neighbors += 1

                        if r < P - 1 and grid[r + 1, c] >= 0:
                            bot_p = grid[r + 1, c]
                            bot_r = grid_rot[r + 1, c]
                            score += v_compat_rot[piece, rot_idx, bot_p, bot_r]
                            n_neighbors += 1

                        if n_neighbors > 0:
                            score /= n_neighbors

                        if score > best_score:
                            best_score = score
                            best_piece = piece
                            best_pos = (r, c)
                            best_rot = rot_idx

            if best_piece >= 0 and best_pos is not None:
                r, c = best_pos
                placement[best_piece] = (r, c)
                orientations[best_piece] = best_rot * 90
                grid[r, c] = best_piece
                grid_rot[r, c] = best_rot
                placed.add(best_piece)
                frontier.discard((r, c))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < P and 0 <= nc < Q and grid[nr, nc] < 0:
                        frontier.add((nr, nc))
            else:
                break

        # Fill remaining randomly
        remaining = [p for p in range(N) if p not in placed]
        empty_positions = [(r, c) for r in range(P) for c in range(Q) if grid[r, c] < 0]
        np.random.shuffle(remaining)
        for piece, (r, c) in zip(remaining, empty_positions):
            placement[piece] = (r, c)
            orientations[piece] = 0
            grid[r, c] = piece
            grid_rot[r, c] = 0

        return placement, orientations, grid, grid_rot


class LocalSearchReconstructor:
    """
    Improves a solution via local search moves:
    - Swap two pieces
    - Rotate a single piece
    - Swap + rotate
    """

    def __init__(self, P: int, Q: int, max_iterations: int = 5000,
                 allow_rotation: bool = True):
        self.P = P
        self.Q = Q
        self.N = P * Q
        self.max_iterations = max_iterations
        self.allow_rotation = allow_rotation

    def _compute_objective(self, grid: np.ndarray, rot_grid: np.ndarray,
                            h_compat: np.ndarray, v_compat: np.ndarray) -> float:
        """Compute total adjacency score for current configuration."""
        score = 0.0
        P, Q = self.P, self.Q

        if h_compat.ndim == 4:
            for r in range(P):
                for c in range(Q - 1):
                    i = grid[r, c]
                    j = grid[r, c + 1]
                    ai = rot_grid[r, c]
                    aj = rot_grid[r, c + 1]
                    score += h_compat[i, ai, j, aj]

            for r in range(P - 1):
                for c in range(Q):
                    i = grid[r, c]
                    j = grid[r + 1, c]
                    ai = rot_grid[r, c]
                    aj = rot_grid[r + 1, c]
                    score += v_compat[i, ai, j, aj]
        else:
            for r in range(P):
                for c in range(Q - 1):
                    score += h_compat[grid[r, c], grid[r, c + 1]]
            for r in range(P - 1):
                for c in range(Q):
                    score += v_compat[grid[r, c], grid[r + 1, c]]

        return score

    def _local_score_change(self, grid, rot_grid, h_compat, v_compat,
                             r1, c1, r2, c2, new_rot1=None, new_rot2=None):
        """
        Compute the change in objective from swapping pieces at (r1,c1) and (r2,c2).
        Returns delta = new_score - old_score (for the affected edges only).
        """
        P, Q = self.P, self.Q
        has_rot = h_compat.ndim == 4

        def edge_score(ri, ci, rj, cj, direction):
            """Score for a single edge."""
            pi = grid[ri, ci]
            pj = grid[rj, cj]
            if has_rot:
                ai = rot_grid[ri, ci]
                aj = rot_grid[rj, cj]
                if direction == 'h':
                    return h_compat[pi, ai, pj, aj]
                else:
                    return v_compat[pi, ai, pj, aj]
            else:
                if direction == 'h':
                    return h_compat[pi, pj]
                else:
                    return v_compat[pi, pj]

        # Collect all edges affected by positions (r1,c1) and (r2,c2)
        affected = set()
        for r, c in [(r1, c1), (r2, c2)]:
            if c > 0:
                affected.add((r, c - 1, r, c, 'h'))
            if c < Q - 1:
                affected.add((r, c, r, c + 1, 'h'))
            if r > 0:
                affected.add((r - 1, c, r, c, 'v'))
            if r < P - 1:
                affected.add((r, c, r + 1, c, 'v'))

        old_score = sum(edge_score(ri, ci, rj, cj, d) for ri, ci, rj, cj, d in affected)

        # Perform swap
        grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
        if has_rot:
            old_rot1 = rot_grid[r1, c1]
            old_rot2 = rot_grid[r2, c2]
            rot_grid[r1, c1] = new_rot1 if new_rot1 is not None else old_rot2
            rot_grid[r2, c2] = new_rot2 if new_rot2 is not None else old_rot1

        new_score = sum(edge_score(ri, ci, rj, cj, d) for ri, ci, rj, cj, d in affected)

        # Undo swap
        grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
        if has_rot:
            rot_grid[r1, c1] = old_rot1
            rot_grid[r2, c2] = old_rot2

        return new_score - old_score

    def improve(self, placement: Dict, orientations: Dict,
                h_compat: np.ndarray, v_compat: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Improve a solution using local search.
        """
        P, Q = self.P, self.Q
        rng = np.random.RandomState(42)

        # Build grid representation
        grid = np.full((P, Q), -1, dtype=int)
        rot_grid = np.zeros((P, Q), dtype=int)
        pos_of = {}

        for piece, (r, c) in placement.items():
            grid[r, c] = piece
            rot_grid[r, c] = orientations.get(piece, 0) // 90
            pos_of[piece] = (r, c)

        current_score = self._compute_objective(grid, rot_grid, h_compat, v_compat)
        best_score = current_score
        no_improve = 0

        for iteration in range(self.max_iterations):
            # Random move type
            move_type = rng.randint(3) if self.allow_rotation else 0

            if move_type == 0:
                # Swap two random pieces
                r1, c1 = rng.randint(P), rng.randint(Q)
                r2, c2 = rng.randint(P), rng.randint(Q)
                if (r1, c1) == (r2, c2):
                    continue

                delta = self._local_score_change(grid, rot_grid, h_compat, v_compat,
                                                  r1, c1, r2, c2)

                if delta > 0:
                    # Accept
                    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
                    if h_compat.ndim == 4:
                        rot_grid[r1, c1], rot_grid[r2, c2] = \
                            rot_grid[r2, c2], rot_grid[r1, c1]
                    current_score += delta
                    no_improve = 0
                else:
                    no_improve += 1

            elif move_type == 1:
                # Rotate a single piece
                r, c = rng.randint(P), rng.randint(Q)
                old_rot = rot_grid[r, c]
                new_rot = rng.randint(4)
                if new_rot == old_rot:
                    continue

                rot_grid[r, c] = new_rot
                new_score = self._compute_objective(grid, rot_grid, h_compat, v_compat)
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                else:
                    rot_grid[r, c] = old_rot
                    no_improve += 1

            elif move_type == 2:
                # Swap + rotate both
                r1, c1 = rng.randint(P), rng.randint(Q)
                r2, c2 = rng.randint(P), rng.randint(Q)
                if (r1, c1) == (r2, c2):
                    continue

                new_rot1 = rng.randint(4)
                new_rot2 = rng.randint(4)

                # Save old state
                old_p1, old_p2 = grid[r1, c1], grid[r2, c2]
                old_r1, old_r2 = rot_grid[r1, c1], rot_grid[r2, c2]

                grid[r1, c1], grid[r2, c2] = old_p2, old_p1
                rot_grid[r1, c1], rot_grid[r2, c2] = new_rot1, new_rot2

                new_score = self._compute_objective(grid, rot_grid, h_compat, v_compat)
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                else:
                    grid[r1, c1], grid[r2, c2] = old_p1, old_p2
                    rot_grid[r1, c1], rot_grid[r2, c2] = old_r1, old_r2
                    no_improve += 1

            if no_improve > 500:
                break

        # Convert back to dictionaries
        placement_out = {}
        orientations_out = {}
        for r in range(P):
            for c in range(Q):
                piece = grid[r, c]
                placement_out[piece] = (r, c)
                orientations_out[piece] = int(rot_grid[r, c]) * 90

        return placement_out, orientations_out


class SimulatedAnnealingReconstructor:
    """
    Simulated annealing for puzzle reconstruction.
    Allows uphill moves with decreasing probability.
    """

    def __init__(self, P: int, Q: int, max_iterations: int = 10000,
                 T_init: float = 1.0, T_min: float = 0.001,
                 cooling_rate: float = 0.995, allow_rotation: bool = True):
        self.P = P
        self.Q = Q
        self.N = P * Q
        self.max_iterations = max_iterations
        self.T_init = T_init
        self.T_min = T_min
        self.cooling_rate = cooling_rate
        self.allow_rotation = allow_rotation

    def reconstruct(self, h_compat: np.ndarray, v_compat: np.ndarray,
                     init_placement: Optional[Dict] = None,
                     init_orientations: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Run simulated annealing starting from an initial solution."""
        P, Q = self.P, self.Q
        rng = np.random.RandomState(123)
        has_rot = h_compat.ndim == 4

        # Initialize grid
        grid = np.full((P, Q), -1, dtype=int)
        rot_grid = np.zeros((P, Q), dtype=int)

        if init_placement:
            for piece, (r, c) in init_placement.items():
                grid[r, c] = piece
                rot_grid[r, c] = (init_orientations or {}).get(piece, 0) // 90
        else:
            pieces = list(range(self.N))
            rng.shuffle(pieces)
            idx = 0
            for r in range(P):
                for c in range(Q):
                    grid[r, c] = pieces[idx]
                    rot_grid[r, c] = rng.randint(4) if self.allow_rotation else 0
                    idx += 1

        def compute_score():
            score = 0.0
            if has_rot:
                for r in range(P):
                    for c in range(Q - 1):
                        score += h_compat[grid[r, c], rot_grid[r, c],
                                          grid[r, c + 1], rot_grid[r, c + 1]]
                for r in range(P - 1):
                    for c in range(Q):
                        score += v_compat[grid[r, c], rot_grid[r, c],
                                          grid[r + 1, c], rot_grid[r + 1, c]]
            else:
                for r in range(P):
                    for c in range(Q - 1):
                        score += h_compat[grid[r, c], grid[r, c + 1]]
                for r in range(P - 1):
                    for c in range(Q):
                        score += v_compat[grid[r, c], grid[r + 1, c]]
            return score

        current_score = compute_score()
        best_score = current_score
        best_grid = grid.copy()
        best_rot = rot_grid.copy()

        T = self.T_init

        for iteration in range(self.max_iterations):
            # Generate neighbor
            move = rng.randint(3) if self.allow_rotation and has_rot else 0

            if move == 0:
                # Swap two pieces
                r1, c1 = rng.randint(P), rng.randint(Q)
                r2, c2 = rng.randint(P), rng.randint(Q)
                while (r1, c1) == (r2, c2):
                    r2, c2 = rng.randint(P), rng.randint(Q)

                grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
                if has_rot:
                    rot_grid[r1, c1], rot_grid[r2, c2] = \
                        rot_grid[r2, c2], rot_grid[r1, c1]

                new_score = compute_score()
                delta = new_score - current_score

                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_score = new_score
                else:
                    grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
                    if has_rot:
                        rot_grid[r1, c1], rot_grid[r2, c2] = \
                            rot_grid[r2, c2], rot_grid[r1, c1]

            elif move == 1:
                r, c = rng.randint(P), rng.randint(Q)
                old_rot = rot_grid[r, c]
                rot_grid[r, c] = rng.randint(4)

                new_score = compute_score()
                delta = new_score - current_score

                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_score = new_score
                else:
                    rot_grid[r, c] = old_rot

            elif move == 2:
                r1, c1 = rng.randint(P), rng.randint(Q)
                r2, c2 = rng.randint(P), rng.randint(Q)
                while (r1, c1) == (r2, c2):
                    r2, c2 = rng.randint(P), rng.randint(Q)

                old_p1, old_p2 = grid[r1, c1], grid[r2, c2]
                old_r1, old_r2 = rot_grid[r1, c1], rot_grid[r2, c2]

                grid[r1, c1], grid[r2, c2] = old_p2, old_p1
                rot_grid[r1, c1] = rng.randint(4)
                rot_grid[r2, c2] = rng.randint(4)

                new_score = compute_score()
                delta = new_score - current_score

                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_score = new_score
                else:
                    grid[r1, c1], grid[r2, c2] = old_p1, old_p2
                    rot_grid[r1, c1] = old_r1
                    rot_grid[r2, c2] = old_r2

            if current_score > best_score:
                best_score = current_score
                best_grid = grid.copy()
                best_rot = rot_grid.copy()

            T = max(T * self.cooling_rate, self.T_min)

        # Convert best to dictionaries
        placement = {}
        orientations = {}
        for r in range(P):
            for c in range(Q):
                piece = best_grid[r, c]
                placement[piece] = (r, c)
                orientations[piece] = int(best_rot[r, c]) * 90

        return placement, orientations


class HungarianReconstructor:
    """
    Uses the Hungarian algorithm for optimal assignment of pieces
    to grid positions. Works best for small puzzles without rotation.
    
    Formulates as a linear assignment problem where the cost is
    based on how well a piece fits at each position given its neighbors.
    """

    def __init__(self, P: int, Q: int):
        self.P = P
        self.Q = Q
        self.N = P * Q

    def _position_score(self, piece: int, r: int, c: int,
                         h_compat: np.ndarray, v_compat: np.ndarray,
                         partial_grid: np.ndarray) -> float:
        """Score how well `piece` fits at position (r, c) given partial grid."""
        P, Q = self.P, self.Q
        score = 0.0

        if c > 0 and partial_grid[r, c - 1] >= 0:
            score += h_compat[partial_grid[r, c - 1], piece]
        if c < Q - 1 and partial_grid[r, c + 1] >= 0:
            score += h_compat[piece, partial_grid[r, c + 1]]
        if r > 0 and partial_grid[r - 1, c] >= 0:
            score += v_compat[partial_grid[r - 1, c], piece]
        if r < P - 1 and partial_grid[r + 1, c] >= 0:
            score += v_compat[piece, partial_grid[r + 1, c]]

        return score

    def reconstruct(self, h_compat: np.ndarray, v_compat: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Row-by-row Hungarian assignment.
        For each row, find the optimal assignment of remaining pieces to columns.
        """
        P, Q = self.P, self.Q
        N = self.N

        grid = np.full((P, Q), -1, dtype=int)
        placed = set()

        # Place row by row
        for r in range(P):
            remaining = [p for p in range(N) if p not in placed]

            if len(remaining) <= Q:
                # Simple assignment for last row
                cost_matrix = np.zeros((len(remaining), Q))
                for pi, piece in enumerate(remaining):
                    for c in range(Q):
                        cost_matrix[pi, c] = -self._position_score(
                            piece, r, c, h_compat, v_compat, grid)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for pi, c in zip(row_ind, col_ind):
                    grid[r, c] = remaining[pi]
                    placed.add(remaining[pi])
            else:
                # Build cost matrix: remaining pieces x Q columns
                cost_matrix = np.zeros((len(remaining), Q))
                for pi, piece in enumerate(remaining):
                    for c in range(Q):
                        cost_matrix[pi, c] = -self._position_score(
                            piece, r, c, h_compat, v_compat, grid)

                        # Also consider horizontal compatibility within row
                        if c > 0 and grid[r, c - 1] >= 0:
                            cost_matrix[pi, c] -= h_compat[grid[r, c - 1], piece]

                # Use Hungarian to find best Q pieces for this row
                row_ind, col_ind = linear_sum_assignment(cost_matrix[:, :Q])
                for pi, c in zip(row_ind, col_ind):
                    if c < Q:
                        grid[r, c] = remaining[pi]
                        placed.add(remaining[pi])

        placement = {}
        orientations = {}
        for r in range(P):
            for c in range(Q):
                piece = grid[r, c]
                if piece >= 0:
                    placement[piece] = (r, c)
                    orientations[piece] = 0

        return placement, orientations


def align_placement_to_grid(placement: Dict[int, Tuple[int, int]],
                            P: int, Q: int) -> Dict[int, Tuple[int, int]]:
    """
    Shift the entire placement so that it aligns with the valid grid [0..P-1] x [0..Q-1].
    The greedy algorithm may start from the center, causing all positions to be offset.
    """
    if not placement:
        return placement

    # Find current bounding box
    rows = [pos[0] for pos in placement.values()]
    cols = [pos[1] for pos in placement.values()]
    min_r, min_c = min(rows), min(cols)

    # Shift so top-left is at (0, 0)
    aligned = {}
    for piece, (r, c) in placement.items():
        new_r = r - min_r
        new_c = c - min_c
        if 0 <= new_r < P and 0 <= new_c < Q:
            aligned[piece] = (new_r, new_c)
        else:
            aligned[piece] = (r, c)

    # Verify all positions are valid
    used_positions = set(aligned.values())
    if len(used_positions) == len(aligned):
        return aligned

    return placement


def try_all_offsets_alignment(placement: Dict[int, Tuple[int, int]],
                               gt_placement: Dict[int, Tuple[int, int]],
                               P: int, Q: int) -> Dict[int, Tuple[int, int]]:
    """
    Try all valid row/col offsets and pick the one maximizing placement accuracy.
    This handles the case where the reconstruction is correct but shifted.
    """
    best_placement = placement
    best_correct = 0

    # Try all offsets
    rows = [pos[0] for pos in placement.values()]
    cols = [pos[1] for pos in placement.values()]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    for dr in range(-max_r, P - min_r + 1):
        for dc in range(-max_c, Q - min_c + 1):
            shifted = {}
            valid = True
            for piece, (r, c) in placement.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < P and 0 <= nc < Q:
                    shifted[piece] = (nr, nc)
                else:
                    valid = False
                    break

            if valid and len(set(shifted.values())) == len(shifted):
                correct = sum(1 for p in gt_placement
                              if p in shifted and shifted[p] == gt_placement[p])
                if correct > best_correct:
                    best_correct = correct
                    best_placement = shifted

    return best_placement


def full_reconstruction_pipeline(features: Dict, P: int, Q: int,
                                   allow_rotation: bool = True,
                                   method: str = 'greedy+sa') -> Tuple[Dict, Dict, float]:
    """
    Full reconstruction pipeline that combines multiple strategies.
    
    Parameters
    ----------
    features : dict from FeatureExtractor
    P, Q : grid dimensions
    allow_rotation : whether tiles may be rotated
    method : 'greedy', 'greedy+local', 'greedy+sa', 'sa'
    
    Returns
    -------
    placement, orientations, final_score
    """
    from adjacency import AdjacencyModel, RotationAwareAdjacencyModel

    print(f"\n  Computing compatibility matrices...")
    start = time.time()

    if allow_rotation:
        rot_model = RotationAwareAdjacencyModel(features, sigma=0.5)
        h_compat, v_compat = rot_model.compute_rotation_compatibility_matrices()
    else:
        model = AdjacencyModel(features, sigma=0.5)
        h_compat, v_compat = model.compute_compatibility_matrix()

    print(f"  Compatibility computed in {time.time() - start:.1f}s")

    N = P * Q

    if 'greedy' in method:
        print(f"  Running greedy reconstruction...")
        start = time.time()
        greedy = GreedyReconstructor(P, Q, allow_rotation)

        if allow_rotation:
            placement, orientations = greedy.reconstruct_with_rotation(h_compat, v_compat)
        else:
            placement, orientations = greedy.reconstruct(h_compat, v_compat)

        print(f"  Greedy done in {time.time() - start:.1f}s")

    if 'local' in method:
        print(f"  Running local search improvement...")
        start = time.time()
        local = LocalSearchReconstructor(P, Q, max_iterations=5000,
                                          allow_rotation=allow_rotation)
        placement, orientations = local.improve(placement, orientations,
                                                  h_compat, v_compat)
        print(f"  Local search done in {time.time() - start:.1f}s")

    if 'sa' in method:
        print(f"  Running simulated annealing...")
        start = time.time()
        sa = SimulatedAnnealingReconstructor(
            P, Q, max_iterations=8000, T_init=1.0, T_min=0.001,
            cooling_rate=0.997, allow_rotation=allow_rotation
        )
        init_p = placement if 'greedy' in method else None
        init_o = orientations if 'greedy' in method else None
        placement, orientations = sa.reconstruct(h_compat, v_compat,
                                                    init_p, init_o)
        print(f"  SA done in {time.time() - start:.1f}s")

    # Compute final objective
    final_score = 0.0
    grid = np.full((P, Q), -1, dtype=int)
    rot_grid = np.zeros((P, Q), dtype=int)
    for piece, (r, c) in placement.items():
        grid[r, c] = piece
        rot_grid[r, c] = orientations.get(piece, 0) // 90

    if allow_rotation:
        for r in range(P):
            for c in range(Q - 1):
                final_score += h_compat[grid[r, c], rot_grid[r, c],
                                         grid[r, c + 1], rot_grid[r, c + 1]]
        for r in range(P - 1):
            for c in range(Q):
                final_score += v_compat[grid[r, c], rot_grid[r, c],
                                         grid[r + 1, c], rot_grid[r + 1, c]]
    else:
        for r in range(P):
            for c in range(Q - 1):
                final_score += h_compat[grid[r, c], grid[r, c + 1]]
        for r in range(P - 1):
            for c in range(Q):
                final_score += v_compat[grid[r, c], grid[r + 1, c]]

    return placement, orientations, final_score
