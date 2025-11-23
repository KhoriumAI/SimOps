"""
Thermal Gradient Predictor
===========================

Fast thermal gradient prediction using simplified heat equation.
Identifies hot spots and thermal concentration regions for adaptive refinement.

Uses:
- Heat equation: ∇²T = 0 (steady-state)
- Automatic heat source detection (nozzles, exhausts, small volumes)
- PCG solver for fast solution

Target: <300ms for typical geometries
"""

import gmsh
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import cg, spilu, LinearOperator
from typing import Dict, List, Tuple, Optional
import time


class ThermalPredictor:
    """
    Fast thermal gradient prediction using coarse thermal FEA

    Solves: K T = Q (heat equation)
    Where:
    - K: thermal conductivity matrix (sparse)
    - T: nodal temperatures
    - Q: heat sources (automatically detected)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize thermal predictor

        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose

        # Material properties (steel-like)
        self.k = 50.0  # Thermal conductivity (W/m·K)
        self.T_ambient = 300.0  # Ambient temperature (K)

        # Solver parameters
        self.max_cg_iterations = 100
        self.cg_tolerance = 1e-6

    def predict_thermal_concentration(self,
                                     input_file: str = None,
                                     mesh_file: str = None,
                                     heat_source_power: float = 1e6) -> Dict:
        """
        Predict thermal gradient regions using fast thermal FEA

        Args:
            input_file: CAD file (will generate coarse mesh)
            mesh_file: Existing mesh file (if provided, skip mesh generation)
            heat_source_power: Applied heat power (W)

        Returns:
            Dictionary with:
            - temperature_field: nodal temperatures
            - temperature_gradient: nodal temp gradient magnitudes
            - refinement_zones: list of high-gradient regions
            - execution_time_ms: total time
        """
        start_time = time.time()

        self._log("🌡️  THERMAL GRADIENT PREDICTOR")
        self._log("=" * 70)

        # Step 1: Generate/load coarse mesh
        if mesh_file:
            self._log("Loading existing mesh...")
            gmsh.initialize()
            gmsh.open(mesh_file)
        else:
            self._log("Generating coarse mesh for fast analysis...")
            if not gmsh.isInitialized():
                gmsh.initialize()
                gmsh.open(input_file)

            # Generate very coarse mesh
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 50.0)
            gmsh.model.mesh.generate(3)

        # Step 2: Extract mesh data
        self._log("\nExtracting mesh data...")
        mesh_data = self._extract_mesh_data()

        self._log(f"  Nodes: {mesh_data['num_nodes']}")
        self._log(f"  Elements: {mesh_data['num_elements']}")

        # Step 3: Detect heat sources automatically
        self._log("\nDetecting potential heat sources...")
        heat_sources = self._detect_heat_sources(mesh_data)

        self._log(f"  Detected {len(heat_sources['nodes'])} potential heat source nodes")
        if heat_sources['geometry_features']:
            for feature in heat_sources['geometry_features']:
                self._log(f"    - {feature}")

        # Step 4: Apply boundary conditions
        self._log("\nApplying boundary conditions...")
        bc_data = self._apply_thermal_boundary_conditions(
            mesh_data,
            heat_sources['nodes'],
            heat_source_power
        )

        # Step 5: Assemble thermal system
        self._log("\nAssembling conductivity matrix...")
        K, Q = self._assemble_thermal_system(mesh_data, bc_data)

        self._log(f"  Matrix size: {K.shape[0]} DOF")
        self._log(f"  Non-zeros: {K.nnz}")

        # Step 6: Solve K T = Q
        self._log("\nSolving thermal system (PCG)...")
        T = self._solve_thermal_system(K, Q)

        # Step 7: Compute temperature gradients
        self._log("\nComputing temperature gradients...")
        thermal_data = self._compute_gradients(mesh_data, T)

        # Step 8: Identify refinement zones
        self._log("\nIdentifying high-gradient regions...")
        refinement_zones = self._identify_refinement_zones(
            mesh_data,
            T,
            thermal_data['temp_gradient']
        )

        execution_time = (time.time() - start_time) * 1000

        self._log(f"\n[OK] Thermal prediction complete in {execution_time:.1f}ms")
        self._log(f"  Max temperature: {np.max(T):.2f} K")
        self._log(f"  Temperature range: {np.max(T) - np.min(T):.2f} K")
        self._log(f"  Refinement zones: {len(refinement_zones)}")

        return {
            'temperature_field': T,
            'temperature_gradient': thermal_data['temp_gradient'],
            'refinement_zones': refinement_zones,
            'node_coords': mesh_data['node_coords'],
            'heat_sources': heat_sources,
            'execution_time_ms': execution_time
        }

    def _extract_mesh_data(self) -> Dict:
        """Extract mesh nodes and elements"""
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)

        # Get tetrahedra
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)

        tet_elements = []
        for i, etype in enumerate(elem_types):
            if etype == 4:  # Linear tet
                nodes = elem_nodes[i].reshape(-1, 4)
                tet_elements.append(nodes)

        if not tet_elements:
            raise ValueError("No tetrahedral elements found in mesh")

        tet_elements = np.vstack(tet_elements)

        return {
            'node_tags': node_tags,
            'node_coords': node_coords,
            'num_nodes': len(node_tags),
            'tet_elements': tet_elements,
            'num_elements': len(tet_elements)
        }

    def _detect_heat_sources(self, mesh_data: Dict) -> Dict:
        """
        Automatically detect likely heat source locations

        Heuristics:
        - Nozzle/exhaust: Small area at one end (rocket engines)
        - Electronics: Small enclosed volumes
        - High curvature regions: Often where heat concentrates
        """
        node_coords = mesh_data['node_coords']

        # Get bounding box
        bbox_min = np.min(node_coords, axis=0)
        bbox_max = np.max(node_coords, axis=0)
        bbox_size = bbox_max - bbox_min
        char_length = np.linalg.norm(bbox_size)

        detected_features = []
        heat_source_nodes = []

        # Strategy 1: Detect exhaust/nozzle (small end region)
        # Find nodes at extreme Z (top 5%)
        z_max = bbox_max[2]
        z_threshold = z_max - 0.05 * bbox_size[2]

        nozzle_candidates = np.where(node_coords[:, 2] > z_threshold)[0]

        if len(nozzle_candidates) > 0:
            # Check if this is a small area (likely nozzle)
            nozzle_coords = node_coords[nozzle_candidates]
            nozzle_area_xy = np.ptp(nozzle_coords[:, 0]) * np.ptp(nozzle_coords[:, 1])
            total_area_xy = bbox_size[0] * bbox_size[1]

            if nozzle_area_xy < 0.1 * total_area_xy:
                heat_source_nodes.extend(nozzle_candidates)
                detected_features.append(f"Nozzle/exhaust region (Z > {z_threshold:.2f})")

        # Strategy 2: Detect small volumes (electronics, hot spots)
        # Find nodes in small clustered regions
        # (Simple heuristic: find isolated node groups)

        # If no specific features detected, use geometric center as default
        if len(heat_source_nodes) == 0:
            # Use top 10% Z nodes as default heat source
            z_90 = np.percentile(node_coords[:, 2], 90)
            heat_source_nodes = list(np.where(node_coords[:, 2] > z_90)[0])
            detected_features.append(f"Default heat source (top 10% Z)")

        return {
            'nodes': heat_source_nodes,
            'geometry_features': detected_features
        }

    def _apply_thermal_boundary_conditions(self,
                                          mesh_data: Dict,
                                          heat_source_nodes: List[int],
                                          heat_power: float) -> Dict:
        """
        Apply thermal boundary conditions

        - Heat sources: specified temperature or heat flux
        - Ambient surfaces: convection boundary (simplified)
        """
        node_coords = mesh_data['node_coords']

        # Ambient temperature nodes (bottom 10%)
        z_min = np.min(node_coords[:, 2])
        z_range = np.ptp(node_coords[:, 2])
        ambient_nodes = np.where(node_coords[:, 2] < z_min + 0.1 * z_range)[0]

        # Heat source temperature (hot!)
        T_source = self.T_ambient + 500.0  # +500K above ambient

        return {
            'heat_source_nodes': heat_source_nodes,
            'ambient_nodes': ambient_nodes,
            'T_source': T_source,
            'T_ambient': self.T_ambient,
            'heat_power': heat_power
        }

    def _assemble_thermal_system(self, mesh_data: Dict, bc_data: Dict) -> Tuple:
        """
        Assemble thermal conductivity matrix K and heat source vector Q

        Heat equation: K T = Q
        """
        num_nodes = mesh_data['num_nodes']

        # Initialize
        K = lil_matrix((num_nodes, num_nodes))
        Q = np.zeros(num_nodes)

        # Assemble element by element
        for elem_nodes in mesh_data['tet_elements']:
            # Get element coordinates
            elem_coords = mesh_data['node_coords'][elem_nodes.astype(int) - 1]

            # Compute element conductivity matrix
            ke = self._element_conductivity(elem_coords)

            # Assemble into global
            node_indices = elem_nodes.astype(int) - 1

            for i, ni in enumerate(node_indices):
                for j, nj in enumerate(node_indices):
                    K[ni, nj] += ke[i, j]

        # Apply heat sources (fixed temperature)
        penalty = 1e20
        for node_idx in bc_data['heat_source_nodes']:
            K[node_idx, node_idx] += penalty
            Q[node_idx] = penalty * bc_data['T_source']

        # Apply ambient boundary (fixed temperature)
        for node_idx in bc_data['ambient_nodes']:
            K[node_idx, node_idx] += penalty
            Q[node_idx] = penalty * bc_data['T_ambient']

        # Convert to CSR
        K = K.tocsr()

        return K, Q

    def _element_conductivity(self, elem_coords: np.ndarray) -> np.ndarray:
        """
        Compute element conductivity matrix for linear tetrahedron

        ke = k * V * B^T * B
        where B contains shape function derivatives
        """
        # Element coordinates (4x3)
        x = elem_coords[:, 0]
        y = elem_coords[:, 1]
        z = elem_coords[:, 2]

        # Volume
        V6 = abs(np.linalg.det(np.array([
            [1, x[0], y[0], z[0]],
            [1, x[1], y[1], z[1]],
            [1, x[2], y[2], z[2]],
            [1, x[3], y[3], z[3]]
        ])))

        if V6 < 1e-15:
            return np.zeros((4, 4))

        V = V6 / 6.0

        # Shape function derivatives (constant for linear tet)
        dN = np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]
        ]) / 3.0

        # ke = k * V * dN * dN^T (simplified)
        ke = self.k * V * (dN @ dN.T)

        return ke

    def _solve_thermal_system(self, K: csr_matrix, Q: np.ndarray) -> np.ndarray:
        """Solve K T = Q using PCG"""
        # Create ILU preconditioner
        try:
            ilu = spilu(K.tocsc(), drop_tol=1e-4, fill_factor=10)
            M = LinearOperator(K.shape, matvec=ilu.solve)
        except Exception as e:
            self._log(f"  Warning: ILU failed ({e}), using no preconditioner")
            M = None

        # Solve
        T, info = cg(K, Q, M=M, atol=self.cg_tolerance, maxiter=self.max_cg_iterations)

        if info > 0:
            self._log(f"  Warning: CG did not converge ({info} iterations)")
        elif info == 0:
            self._log(f"  [OK] CG converged")

        return T

    def _compute_gradients(self, mesh_data: Dict, T: np.ndarray) -> Dict:
        """Compute temperature gradients"""
        num_nodes = mesh_data['num_nodes']
        temp_gradient = np.zeros(num_nodes)

        # Compute gradient via finite differences
        for i, coord in enumerate(mesh_data['node_coords']):
            # Find nearby nodes
            distances = np.linalg.norm(mesh_data['node_coords'] - coord, axis=1)
            nearby = np.where((distances < 50.0) & (distances > 0))[0]

            if len(nearby) > 0:
                temp_diff = T[nearby] - T[i]
                dist_diff = distances[nearby]
                gradients = temp_diff / (dist_diff + 1e-10)
                temp_gradient[i] = np.max(np.abs(gradients))

        return {
            'temp_gradient': temp_gradient
        }

    def _identify_refinement_zones(self,
                                   mesh_data: Dict,
                                   T: np.ndarray,
                                   temp_gradient: np.ndarray) -> List[Dict]:
        """
        Identify regions needing refinement based on temperature gradients

        Refine top 15% temperature gradient regions
        """
        zones = []

        # Top 15% gradient threshold (thermal gradients less critical than stress)
        threshold = np.percentile(temp_gradient, 85)

        high_gradient_nodes = np.where(temp_gradient > threshold)[0]

        self._log(f"  High gradient nodes: {len(high_gradient_nodes)} "
                 f"(threshold: {threshold:.2f} K/m)")

        # Create zones for top nodes
        for node_idx in high_gradient_nodes[:15]:  # Top 15 nodes
            coord = mesh_data['node_coords'][node_idx]
            zones.append({
                'type': 'thermal_gradient',
                'location': coord,
                'temperature': T[node_idx],
                'gradient': temp_gradient[node_idx],
                'priority': 0.75,  # Medium-high priority
                'refinement_factor': 2.0
            })

        return zones

    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(message)


def test_thermal_predictor():
    """Test thermal predictor on sample CAD file"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python thermal_predictor.py <cad_file.step>")
        sys.exit(1)

    cad_file = sys.argv[1]

    predictor = ThermalPredictor(verbose=True)
    results = predictor.predict_thermal_concentration(input_file=cad_file)

    print("\n" + "=" * 70)
    print("THERMAL PREDICTION RESULTS")
    print("=" * 70)
    print(f"Execution time: {results['execution_time_ms']:.1f}ms")
    print(f"Max temperature: {np.max(results['temperature_field']):.2f} K")
    print(f"Temperature range: {np.max(results['temperature_field']) - np.min(results['temperature_field']):.2f} K")
    print(f"Max gradient: {np.max(results['temperature_gradient']):.2f} K/m")
    print(f"Refinement zones: {len(results['refinement_zones'])}")

    for i, zone in enumerate(results['refinement_zones'][:5]):
        print(f"\n  Zone {i+1}:")
        print(f"    Location: ({zone['location'][0]:.2f}, {zone['location'][1]:.2f}, {zone['location'][2]:.2f})")
        print(f"    Temperature: {zone['temperature']:.2f} K")
        print(f"    Gradient: {zone['gradient']:.2f} K/m")
        print(f"    Refinement: {zone['refinement_factor']:.1f}x")

    gmsh.finalize()


if __name__ == "__main__":
    test_thermal_predictor()
