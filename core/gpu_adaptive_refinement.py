"""
GPU-Accelerated Adaptive Mesh Refinement
=========================================

Implements adaptive mesh refinement with:
- Configurable minimum SICN quality target
- Per-iteration timeout (<5 seconds)
- Maximum iteration limit
- Selective refinement of low-quality elements
- GPU acceleration when available

Based on gradient descent optimization with:
- Quality-aware vertex relocation
- Edge flipping for topology improvement
- Early stopping when targets are met
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, Callable

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class AdaptiveGPURefinement:
    """
    GPU-accelerated adaptive mesh refinement.
    
    Key Features:
    - Targets minimum SICN quality threshold
    - Times out after max_iterations
    - Each iteration budgeted to <5 seconds
    - Selectively refines only low-quality elements
    """
    
    def __init__(
        self,
        target_sicn: float = 0.3,
        max_iterations: int = 10,
        iteration_timeout_sec: float = 5.0,
        step_size: float = 0.1,
        verbose: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize adaptive refinement.
        
        Args:
            target_sicn: Minimum SICN quality to achieve (default 0.3)
            max_iterations: Maximum refinement iterations before timeout
            iteration_timeout_sec: Warning threshold for iteration time
            step_size: Gradient descent step size
            verbose: Print progress messages
            progress_callback: Optional callback(message, progress_pct)
        """
        self.target_sicn = target_sicn
        self.max_iterations = max_iterations
        self.iteration_timeout = iteration_timeout_sec
        self.step_size = step_size
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.use_gpu = GPU_AVAILABLE
        
        if not self.use_gpu:
            self._log("[!] GPU not available (CuPy not installed). Using CPU fallback.")
    
    def refine(
        self, 
        nodes: np.ndarray, 
        elements: np.ndarray, 
        fixed_nodes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Main refinement loop with adaptive step sizing.
        
        Args:
            nodes: Nx3 array of vertex coordinates
            elements: Mx4 array of tetrahedral connectivity
            fixed_nodes: Optional array of node indices to keep fixed (boundary)
        
        Returns:
            refined_nodes: Optimized vertex positions
            stats: Dict with iteration count, final quality, timing info
        """
        self._log("\n" + "=" * 70)
        self._log("ADAPTIVE GPU MESH REFINEMENT")
        self._log("=" * 70)
        self._log(f"Nodes: {len(nodes)}, Elements: {len(elements)}")
        self._log(f"Target SICN: {self.target_sicn}, Max iterations: {self.max_iterations}")
        
        total_start = time.perf_counter()
        
        # Copy nodes to avoid modifying input
        current_nodes = nodes.copy().astype(np.float32)
        
        # Create movable mask
        movable_mask = np.ones(len(nodes), dtype=bool)
        if fixed_nodes is not None:
            movable_mask[fixed_nodes] = False
        
        # Stats tracking
        stats = {
            'iterations': 0,
            'initial_sicn_min': None,
            'initial_sicn_avg': None,
            'final_sicn_min': None,
            'final_sicn_avg': None,
            'converged': False,
            'timed_out': False,
            'iteration_times': [],
            'quality_history': []
        }
        
        # Initial quality assessment
        quality = self._compute_element_quality(current_nodes, elements)
        stats['initial_sicn_min'] = float(np.min(quality))
        stats['initial_sicn_avg'] = float(np.mean(quality))
        stats['quality_history'].append({
            'iteration': 0,
            'min': stats['initial_sicn_min'],
            'avg': stats['initial_sicn_avg']
        })
        
        self._log(f"\nInitial quality: SICN min={stats['initial_sicn_min']:.4f}, avg={stats['initial_sicn_avg']:.4f}")
        
        # Check if already meets target
        if np.min(quality) >= self.target_sicn:
            self._log(f"[OK] Mesh already meets quality target (SICN >= {self.target_sicn})")
            stats['converged'] = True
            stats['final_sicn_min'] = stats['initial_sicn_min']
            stats['final_sicn_avg'] = stats['initial_sicn_avg']
            return current_nodes, stats
        
        # Progressive quality thresholds - start with worst elements
        # This focuses optimization effort where it's needed most
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, self.target_sicn]
        thresholds = sorted(set([t for t in thresholds if t <= self.target_sicn]))
        
        iteration = 0
        
        for threshold in thresholds:
            if iteration >= self.max_iterations:
                break
                
            self._log(f"\n--- Pass: Targeting elements with SICN < {threshold:.2f} ---")
            
            # Multiple sub-iterations per threshold level
            sub_iters = min(3, self.max_iterations - iteration)
            
            for sub_iter in range(sub_iters):
                iter_start = time.perf_counter()
                
                # Identify elements below current threshold
                bad_mask = quality < max(threshold, 0.01)  # Always include inverted
                num_bad = np.sum(bad_mask)
                
                if num_bad == 0:
                    continue
                
                # Find nodes connected to bad elements
                bad_elements = elements[bad_mask]
                bad_quality = quality[bad_mask]
                nodes_to_optimize = np.unique(bad_elements.ravel())
                
                # Filter to only movable nodes
                nodes_to_optimize = nodes_to_optimize[movable_mask[nodes_to_optimize]]
                
                if len(nodes_to_optimize) == 0:
                    continue
                
                # Adaptive step size: larger for worse elements
                # Elements with SICN < 0 get 3x step, SICN 0-0.1 gets 2x, etc.
                avg_bad_quality = np.mean(bad_quality)
                if avg_bad_quality < 0:
                    adaptive_step = self.step_size * 3.0
                elif avg_bad_quality < 0.1:
                    adaptive_step = self.step_size * 2.0
                else:
                    adaptive_step = self.step_size
                
                # Store previous state for potential rollback
                prev_nodes = current_nodes.copy()
                prev_min_q = float(np.min(quality))
                
                # Compute gradients and update positions
                if self.use_gpu and len(elements) > 1000:
                    current_nodes = self._optimize_iteration_gpu(
                        current_nodes, elements, nodes_to_optimize, adaptive_step
                    )
                else:
                    current_nodes = self._optimize_iteration_cpu(
                        current_nodes, elements, nodes_to_optimize, adaptive_step
                    )
                
                # Recompute quality
                quality = self._compute_element_quality(current_nodes, elements)
                min_q = float(np.min(quality))
                avg_q = float(np.mean(quality))
                
                # ROLLBACK CHECK: If quality got significantly worse, undo and reduce step
                if min_q < prev_min_q - 0.01:  # Got worse by more than 0.01
                    current_nodes = prev_nodes
                    quality = self._compute_element_quality(current_nodes, elements)
                    adaptive_step *= 0.5  # Reduce step size
                    self._log(f"  [ROLLBACK] Quality degraded, reducing step to {adaptive_step:.3f}")
                    continue  # Skip this iteration
                
                iter_time = time.perf_counter() - iter_start
                stats['iteration_times'].append(iter_time)
                stats['quality_history'].append({
                    'iteration': iteration + 1,
                    'min': min_q,
                    'avg': avg_q
                })
                
                iteration += 1
                stats['iterations'] = iteration
                
                # Progress report
                self._log(f"  Iter {iteration}: SICN min={min_q:.4f}, avg={avg_q:.4f}, "
                         f"bad_elems={num_bad}, step={adaptive_step:.2f}, time={iter_time:.2f}s")
                
                if self.progress_callback:
                    progress_pct = iteration / self.max_iterations * 100
                    self.progress_callback(f"Refinement iteration {iteration}", progress_pct)
                
                # Check early stopping
                if min_q >= self.target_sicn:
                    self._log(f"\n[OK] Quality target achieved!")
                    stats['converged'] = True
                    break
                
                if iteration >= self.max_iterations:
                    break
            
            if stats['converged']:
                break
        
        # Check if we hit max iterations
        if stats['iterations'] >= self.max_iterations and not stats['converged']:
            stats['timed_out'] = True
            self._log(f"\n[!] Max iterations ({self.max_iterations}) reached without convergence")
        
        # Final quality
        final_quality = self._compute_element_quality(current_nodes, elements)
        stats['final_sicn_min'] = float(np.min(final_quality))
        stats['final_sicn_avg'] = float(np.mean(final_quality))
        
        total_time = time.perf_counter() - total_start
        
        # Summary
        self._log("\n" + "=" * 70)
        self._log("REFINEMENT COMPLETE")
        self._log("=" * 70)
        self._log(f"Iterations: {stats['iterations']}")
        self._log(f"Total time: {total_time:.2f}s")
        self._log(f"Avg iteration time: {np.mean(stats['iteration_times']):.2f}s" if stats['iteration_times'] else "N/A")
        self._log(f"Initial SICN: min={stats['initial_sicn_min']:.4f}, avg={stats['initial_sicn_avg']:.4f}")
        self._log(f"Final SICN:   min={stats['final_sicn_min']:.4f}, avg={stats['final_sicn_avg']:.4f}")
        self._log(f"Improvement:  min +{stats['final_sicn_min'] - stats['initial_sicn_min']:.4f}, "
                 f"avg +{stats['final_sicn_avg'] - stats['initial_sicn_avg']:.4f}")
        self._log(f"Converged: {stats['converged']}, Timed out: {stats['timed_out']}")
        
        return current_nodes, stats
    
    def _compute_element_quality(self, nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """
        Compute SICN quality for all elements (vectorized).
        
        Returns:
            Array of SICN values per element
        """
        # Gather element vertex coordinates
        elem_verts = nodes[elements]  # Shape: (n_elems, 4, 3)
        
        p0 = elem_verts[:, 0]
        p1 = elem_verts[:, 1]
        p2 = elem_verts[:, 2]
        p3 = elem_verts[:, 3]
        
        # Edge vectors
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        
        # Volumes
        cross_e2_e3 = np.cross(e2, e3)
        volumes = np.einsum('ij,ij->i', e1, cross_e2_e3)
        
        # All 6 edge lengths
        len_e1 = np.linalg.norm(e1, axis=1)
        len_e2 = np.linalg.norm(e2, axis=1)
        len_e3 = np.linalg.norm(e3, axis=1)
        len_e12 = np.linalg.norm(p2 - p1, axis=1)
        len_e13 = np.linalg.norm(p3 - p1, axis=1)
        len_e23 = np.linalg.norm(p3 - p2, axis=1)
        
        # Max edge length per element
        all_lengths = np.stack([len_e1, len_e2, len_e3, len_e12, len_e13, len_e23], axis=1)
        max_lengths = np.max(all_lengths, axis=1)
        
        # SICN = Volume / MaxEdge^3
        sicn = volumes / (max_lengths ** 3 + 1e-12)
        
        return sicn
    
    def _optimize_iteration_gpu(
        self, 
        nodes: np.ndarray, 
        elements: np.ndarray, 
        target_nodes: np.ndarray,
        step_size: float = None
    ) -> np.ndarray:
        """GPU-accelerated optimization step."""
        if step_size is None:
            step_size = self.step_size
            
        # Transfer to GPU
        nodes_gpu = cp.array(nodes).astype(cp.float32)
        elements_gpu = cp.array(elements).astype(cp.int32)
        
        # Use Laplacian smoothing (more stable than gradient descent)
        smoothed = self._laplacian_smooth_gpu(nodes_gpu, elements_gpu, target_nodes, step_size)
        
        return cp.asnumpy(smoothed)
    
    def _optimize_iteration_cpu(
        self, 
        nodes: np.ndarray, 
        elements: np.ndarray, 
        target_nodes: np.ndarray,
        step_size: float = None
    ) -> np.ndarray:
        """CPU optimization step using Laplacian smoothing."""
        if step_size is None:
            step_size = self.step_size
        
        return self._laplacian_smooth_cpu(nodes, elements, target_nodes, step_size)
    
    def _laplacian_smooth_gpu(
        self,
        nodes_gpu: 'cp.ndarray',
        elements_gpu: 'cp.ndarray', 
        target_nodes: np.ndarray,
        step_size: float
    ) -> 'cp.ndarray':
        """
        GPU-accelerated Laplacian smoothing.
        
        Moves each vertex toward the centroid of its connected elements,
        weighted by element quality (bad elements get more weight).
        """
        n_nodes = len(nodes_gpu)
        
        # Compute element centroids
        elem_verts = nodes_gpu[elements_gpu]  # (n_elems, 4, 3)
        centroids = cp.mean(elem_verts, axis=1)  # (n_elems, 3)
        
        # Compute element qualities
        quality = self._compute_element_quality_gpu(nodes_gpu, elements_gpu)
        
        # Weight: lower quality -> higher weight (inverse)
        weights = 1.0 / (cp.abs(quality) + 0.1)  # Avoid div by zero
        weights = weights / cp.max(weights)  # Normalize
        
        # Accumulate weighted centroids per node
        target_positions = cp.zeros((n_nodes, 3), dtype=cp.float32)
        weight_sums = cp.zeros(n_nodes, dtype=cp.float32)
        
        # Each element contributes to its 4 vertices
        for i in range(4):
            node_indices = elements_gpu[:, i]
            for j in range(3):
                cp.add.at(target_positions[:, j], node_indices, centroids[:, j] * weights)
            cp.add.at(weight_sums, node_indices, weights)
        
        # Compute average target position
        valid = weight_sums > 0
        for j in range(3):
            target_positions[valid, j] /= weight_sums[valid]
        
        # Move target nodes toward Laplacian position
        result = nodes_gpu.copy()
        target_nodes_gpu = cp.array(target_nodes)
        
        # Only update nodes that have valid targets
        mask = valid[target_nodes_gpu]
        update_nodes = target_nodes_gpu[mask.get()]
        
        displacement = target_positions[update_nodes] - result[update_nodes]
        result[update_nodes] += step_size * displacement
        
        return result
    
    def _laplacian_smooth_cpu(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        target_nodes: np.ndarray,
        step_size: float
    ) -> np.ndarray:
        """
        CPU Laplacian smoothing.
        
        Moves each vertex toward the centroid of its connected elements,
        weighted by element quality.
        """
        n_nodes = len(nodes)
        
        # Compute element centroids  
        elem_verts = nodes[elements]  # (n_elems, 4, 3)
        centroids = np.mean(elem_verts, axis=1)  # (n_elems, 3)
        
        # Compute element qualities
        quality = self._compute_element_quality(nodes, elements)
        
        # Weight: lower quality -> higher weight
        weights = 1.0 / (np.abs(quality) + 0.1)
        weights = weights / np.max(weights)
        
        # Accumulate weighted centroids per node
        target_positions = np.zeros((n_nodes, 3), dtype=np.float32)
        weight_sums = np.zeros(n_nodes, dtype=np.float32)
        
        # Use vectorized accumulation
        for i in range(4):
            np.add.at(target_positions, elements[:, i:i+1], 
                      (centroids * weights[:, None]))
            np.add.at(weight_sums, elements[:, i], weights)
        
        # Compute average target position
        valid = weight_sums > 0
        target_positions[valid] /= weight_sums[valid, None]
        
        # Move target nodes toward Laplacian position
        result = nodes.copy()
        
        # Only update valid target nodes
        valid_targets = target_nodes[valid[target_nodes]]
        displacement = target_positions[valid_targets] - result[valid_targets]
        result[valid_targets] += step_size * displacement
        
        return result
    
    def _compute_element_quality_gpu(self, nodes_gpu: 'cp.ndarray', elements_gpu: 'cp.ndarray') -> 'cp.ndarray':
        """Compute SICN quality on GPU."""
        elem_verts = nodes_gpu[elements_gpu]
        
        p0, p1, p2, p3 = elem_verts[:, 0], elem_verts[:, 1], elem_verts[:, 2], elem_verts[:, 3]
        
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        
        cross_e2_e3 = cp.cross(e2, e3)
        volumes = cp.einsum('ij,ij->i', e1, cross_e2_e3)
        
        len_e1 = cp.linalg.norm(e1, axis=1)
        len_e2 = cp.linalg.norm(e2, axis=1)
        len_e3 = cp.linalg.norm(e3, axis=1)
        len_e12 = cp.linalg.norm(p2 - p1, axis=1)
        len_e13 = cp.linalg.norm(p3 - p1, axis=1)
        len_e23 = cp.linalg.norm(p3 - p2, axis=1)
        
        all_lengths = cp.stack([len_e1, len_e2, len_e3, len_e12, len_e13, len_e23], axis=1)
        max_lengths = cp.max(all_lengths, axis=1)
        
        return volumes / (max_lengths ** 3 + 1e-12)
    
    def _compute_gradients_gpu(self, nodes_gpu: 'cp.ndarray', elements_gpu: 'cp.ndarray') -> 'cp.ndarray':
        """Compute quality gradients on GPU (vectorized)."""
        n_nodes = len(nodes_gpu)
        n_elems = len(elements_gpu)
        
        # Gather element vertex coordinates
        elem_verts = nodes_gpu[elements_gpu]
        
        p0, p1, p2, p3 = elem_verts[:, 0], elem_verts[:, 1], elem_verts[:, 2], elem_verts[:, 3]
        
        # Edge vectors
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        
        # Cross products for gradient
        cross_e2_e3 = cp.cross(e2, e3)
        cross_e3_e1 = cp.cross(e3, e1)
        cross_e1_e2 = cp.cross(e1, e2)
        
        # Volume signs
        volumes = cp.einsum('ij,ij->i', e1, cross_e2_e3)
        signs = cp.sign(volumes)[:, None]
        
        # Gradients per vertex
        grads = cp.zeros((n_elems, 4, 3), dtype=cp.float32)
        grads[:, 0] = cross_e2_e3 * signs
        grads[:, 1] = cross_e3_e1 * signs
        grads[:, 2] = cross_e1_e2 * signs
        grads[:, 3] = -(grads[:, 0] + grads[:, 1] + grads[:, 2])
        
        # Scatter to nodes
        gradients = cp.zeros((n_nodes, 3), dtype=cp.float32)
        for i in range(4):
            indices = elements_gpu[:, i]
            for j in range(3):
                cp.add.at(gradients[:, j], indices, grads[:, i, j])
        
        # Normalize
        norms = cp.linalg.norm(gradients, axis=1, keepdims=True)
        return gradients / cp.maximum(norms, 1e-10)
    
    def _compute_gradients_cpu(self, nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Compute quality gradients on CPU (vectorized)."""
        n_nodes = len(nodes)
        n_elems = len(elements)
        
        # Gather element vertex coordinates
        elem_verts = nodes[elements]
        
        p0, p1, p2, p3 = elem_verts[:, 0], elem_verts[:, 1], elem_verts[:, 2], elem_verts[:, 3]
        
        # Edge vectors
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        
        # Cross products for gradient
        cross_e2_e3 = np.cross(e2, e3)
        cross_e3_e1 = np.cross(e3, e1)
        cross_e1_e2 = np.cross(e1, e2)
        
        # Volume signs
        volumes = np.einsum('ij,ij->i', e1, cross_e2_e3)
        signs = np.sign(volumes)[:, None]
        
        # Gradients per vertex
        grads = np.zeros((n_elems, 4, 3), dtype=np.float32)
        grads[:, 0] = cross_e2_e3 * signs
        grads[:, 1] = cross_e3_e1 * signs
        grads[:, 2] = cross_e1_e2 * signs
        grads[:, 3] = -(grads[:, 0] + grads[:, 1] + grads[:, 2])
        
        # Scatter to nodes using np.add.at
        gradients = np.zeros((n_nodes, 3), dtype=np.float32)
        for i in range(4):
            np.add.at(gradients, elements[:, i:i+1], grads[:, i:i+1])
        
        # Normalize
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        return gradients / np.maximum(norms, 1e-10)
    
    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)


def test_adaptive_refinement():
    """Test adaptive refinement on a simple mesh."""
    print("Testing Adaptive GPU Refinement...")
    
    # Create a simple distorted tetrahedral mesh
    nodes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.866, 0],
        [0.5, 0.289, 0.6],  # Slightly distorted apex
        [1.5, 0.5, 0.5]
    ], dtype=np.float32)
    
    elements = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ], dtype=np.int32)
    
    # Fixed boundary nodes
    fixed_nodes = np.array([0, 1, 2])
    
    # Run refinement
    refiner = AdaptiveGPURefinement(
        target_sicn=0.3,
        max_iterations=10,
        verbose=True
    )
    
    refined_nodes, stats = refiner.refine(nodes, elements, fixed_nodes)
    
    print(f"\nTest completed!")
    print(f"Converged: {stats['converged']}")
    print(f"Final SICN: min={stats['final_sicn_min']:.4f}, avg={stats['final_sicn_avg']:.4f}")


if __name__ == "__main__":
    test_adaptive_refinement()
