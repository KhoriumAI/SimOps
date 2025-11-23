"""
Stress Concentration Predictor
===============================

Fast stress prediction using simplified linear elasticity.
Identifies stress concentration regions for adaptive mesh refinement.

Uses:
- Linear elasticity with uniform loading
- Preconditioned Conjugate Gradient (PCG) solver
- Stress gradient analysis

Target: <500ms for typical geometries (1,000-10,000 elements)
"""

import gmsh
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import cg, spilu, LinearOperator
from typing import Dict, List, Tuple, Optional
import time


class StressPredictor:
    """
    Fast stress concentration prediction using coarse FEA

    Solves: K u = F (linear elasticity)
    Where:
    - K: stiffness matrix (sparse)
    - u: nodal displacements
    - F: applied forces (uniform pressure)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize stress predictor

        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose

        # Material properties (steel-like)
        self.E = 200e9  # Young's modulus (Pa)
        self.nu = 0.3   # Poisson's ratio

        # Solver parameters
        self.max_cg_iterations = 100
        self.cg_tolerance = 1e-6

    def predict_stress_concentration(self,
                                     input_file: str = None,
                                     mesh_file: str = None,
                                     load_magnitude: float = 1e6) -> Dict:
        """
        Predict stress concentration regions using fast FEA

        Args:
            input_file: CAD file (will generate coarse mesh)
            mesh_file: Existing mesh file (if provided, skip mesh generation)
            load_magnitude: Applied pressure (Pa)

        Returns:
            Dictionary with:
            - stress_field: nodal von Mises stress values
            - stress_gradient: nodal stress gradient magnitudes
            - refinement_zones: list of high-stress regions
            - execution_time_ms: total time
        """
        start_time = time.time()

        self._log("🔧 STRESS CONCENTRATION PREDICTOR")
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

            # Generate very coarse mesh (500-2000 elements)
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 50.0)  # Large size
            gmsh.model.mesh.generate(3)

        # Step 2: Extract mesh data
        self._log("\nExtracting mesh data...")
        mesh_data = self._extract_mesh_data()

        self._log(f"  Nodes: {mesh_data['num_nodes']}")
        self._log(f"  Elements: {mesh_data['num_elements']}")

        # Step 3: Apply boundary conditions
        self._log("\nApplying boundary conditions...")
        bc_data = self._apply_boundary_conditions(mesh_data, load_magnitude)

        # Step 4: Assemble stiffness matrix
        self._log("\nAssembling stiffness matrix...")
        K, F = self._assemble_system(mesh_data, bc_data)

        self._log(f"  Matrix size: {K.shape[0]} DOF")
        self._log(f"  Non-zeros: {K.nnz}")

        # Step 5: Solve K u = F
        self._log("\nSolving linear system (PCG)...")
        u = self._solve_system(K, F)

        # Step 6: Compute stresses
        self._log("\nComputing stress field...")
        stress_data = self._compute_stresses(mesh_data, u)

        # Step 7: Identify refinement zones
        self._log("\nIdentifying high-stress regions...")
        refinement_zones = self._identify_refinement_zones(
            mesh_data,
            stress_data['von_mises'],
            stress_data['stress_gradient']
        )

        execution_time = (time.time() - start_time) * 1000

        self._log(f"\n[OK] Stress prediction complete in {execution_time:.1f}ms")
        self._log(f"  Max von Mises: {np.max(stress_data['von_mises']):.2e} Pa")
        self._log(f"  Refinement zones: {len(refinement_zones)}")

        return {
            'stress_field': stress_data['von_mises'],
            'stress_gradient': stress_data['stress_gradient'],
            'refinement_zones': refinement_zones,
            'node_coords': mesh_data['node_coords'],
            'execution_time_ms': execution_time
        }

    def _extract_mesh_data(self) -> Dict:
        """Extract mesh nodes and elements"""
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)

        # Get tetrahedra (type 4 = linear tet)
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)

        # Extract only tets
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

    def _apply_boundary_conditions(self, mesh_data: Dict, load_magnitude: float) -> Dict:
        """
        Apply boundary conditions: fixed base, uniform pressure on top

        Strategy: Fix bottom 10% nodes, apply pressure on top 10%
        """
        node_coords = mesh_data['node_coords']

        # Find min/max Z coordinates
        z_min = np.min(node_coords[:, 2])
        z_max = np.max(node_coords[:, 2])
        z_range = z_max - z_min

        # Fixed nodes (bottom 10%)
        fixed_nodes = np.where(node_coords[:, 2] < z_min + 0.1 * z_range)[0]

        # Loaded nodes (top 10%)
        loaded_nodes = np.where(node_coords[:, 2] > z_max - 0.1 * z_range)[0]

        # Uniform load in -Z direction
        load_vector = np.array([0, 0, -load_magnitude])

        return {
            'fixed_nodes': fixed_nodes,
            'loaded_nodes': loaded_nodes,
            'load_vector': load_vector
        }

    def _assemble_system(self, mesh_data: Dict, bc_data: Dict) -> Tuple:
        """
        Assemble global stiffness matrix K and load vector F

        Uses linear tetrahedral elements (4 nodes, 12 DOF)
        """
        num_nodes = mesh_data['num_nodes']
        num_dof = num_nodes * 3

        # Initialize sparse matrix (LIL format for assembly)
        K = lil_matrix((num_dof, num_dof))
        F = np.zeros(num_dof)

        # Material matrix D (plane strain assumption)
        D = self._get_material_matrix()

        # Assemble element by element
        for elem_nodes in mesh_data['tet_elements']:
            # Get element node coordinates
            elem_coords = mesh_data['node_coords'][elem_nodes.astype(int) - 1]

            # Compute element stiffness
            ke = self._element_stiffness(elem_coords, D)

            # Assemble into global matrix
            dof_indices = []
            for node in elem_nodes:
                node_idx = int(node) - 1
                dof_indices.extend([node_idx*3, node_idx*3+1, node_idx*3+2])

            dof_indices = np.array(dof_indices, dtype=int)

            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    K[gi, gj] += ke[i, j]

        # Apply loads
        for node_idx in bc_data['loaded_nodes']:
            F[node_idx*3:node_idx*3+3] += bc_data['load_vector']

        # Apply fixed boundary conditions (penalty method)
        penalty = 1e20
        for node_idx in bc_data['fixed_nodes']:
            for d in range(3):
                dof = node_idx * 3 + d
                K[dof, dof] += penalty
                F[dof] = 0

        # Convert to CSR for solving
        K = K.tocsr()

        return K, F

    def _get_material_matrix(self) -> np.ndarray:
        """Compute material matrix D for 3D elasticity"""
        E = self.E
        nu = self.nu

        factor = E / ((1 + nu) * (1 - 2*nu))

        D = factor * np.array([
            [1-nu,   nu,     nu,     0,           0,           0],
            [nu,     1-nu,   nu,     0,           0,           0],
            [nu,     nu,     1-nu,   0,           0,           0],
            [0,      0,      0,      (1-2*nu)/2,  0,           0],
            [0,      0,      0,      0,           (1-2*nu)/2,  0],
            [0,      0,      0,      0,           0,           (1-2*nu)/2]
        ])

        return D

    def _element_stiffness(self, elem_coords: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Compute element stiffness matrix for linear tetrahedron

        Uses constant strain formulation (simplest, fastest)
        """
        # Element coordinates (4x3)
        x = elem_coords[:, 0]
        y = elem_coords[:, 1]
        z = elem_coords[:, 2]

        # Volume (6V)
        V6 = abs(np.linalg.det(np.array([
            [1, x[0], y[0], z[0]],
            [1, x[1], y[1], z[1]],
            [1, x[2], y[2], z[2]],
            [1, x[3], y[3], z[3]]
        ])))

        if V6 < 1e-15:
            return np.zeros((12, 12))  # Degenerate element

        # Strain-displacement matrix B (6x12)
        # Simplified formulation (constant strain tet)
        B = np.zeros((6, 12))

        # Shape function derivatives (constant for linear tet)
        # This is a simplified placeholder - full implementation would compute
        # from geometry
        dN = np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]
        ]) / 3.0

        # Build B matrix
        for i in range(4):
            B[0, i*3]     = dN[i, 0]
            B[1, i*3+1]   = dN[i, 1]
            B[2, i*3+2]   = dN[i, 2]
            B[3, i*3]     = dN[i, 1]
            B[3, i*3+1]   = dN[i, 0]
            B[4, i*3+1]   = dN[i, 2]
            B[4, i*3+2]   = dN[i, 1]
            B[5, i*3]     = dN[i, 2]
            B[5, i*3+2]   = dN[i, 0]

        # ke = V * B^T * D * B
        ke = (V6 / 6.0) * B.T @ D @ B

        return ke

    def _solve_system(self, K: csr_matrix, F: np.ndarray) -> np.ndarray:
        """
        Solve K u = F using Preconditioned Conjugate Gradient

        Much faster than direct solve for large systems
        """
        # Create ILU preconditioner
        try:
            ilu = spilu(K.tocsc(), drop_tol=1e-4, fill_factor=10)
            M = LinearOperator(K.shape, matvec=ilu.solve)
        except Exception as e:
            self._log(f"  Warning: ILU failed ({e}), using no preconditioner")
            M = None

        # Solve with PCG
        u, info = cg(K, F, M=M, atol=self.cg_tolerance, maxiter=self.max_cg_iterations)

        if info > 0:
            self._log(f"  Warning: CG did not converge ({info} iterations)")
        elif info == 0:
            self._log(f"  [OK] CG converged")

        return u

    def _compute_stresses(self, mesh_data: Dict, u: np.ndarray) -> Dict:
        """Compute von Mises stress and stress gradients"""
        num_nodes = mesh_data['num_nodes']
        von_mises = np.zeros(num_nodes)
        stress_gradient = np.zeros(num_nodes)

        # Compute element-wise stresses (constant strain tet)
        D = self._get_material_matrix()

        element_stresses = []

        for elem_nodes in mesh_data['tet_elements']:
            # Get element displacements
            elem_u = []
            for node in elem_nodes:
                node_idx = int(node) - 1
                elem_u.extend(u[node_idx*3:node_idx*3+3])

            elem_u = np.array(elem_u)

            # Get element coordinates
            elem_coords = mesh_data['node_coords'][elem_nodes.astype(int) - 1]

            # Compute strain (simplified)
            # For constant strain tet, this is uniform over element
            dN = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 3.0
            B = np.zeros((6, 12))
            for i in range(4):
                B[0, i*3]     = dN[i, 0]
                B[1, i*3+1]   = dN[i, 1]
                B[2, i*3+2]   = dN[i, 2]
                B[3, i*3]     = dN[i, 1]
                B[3, i*3+1]   = dN[i, 0]
                B[4, i*3+1]   = dN[i, 2]
                B[4, i*3+2]   = dN[i, 1]
                B[5, i*3]     = dN[i, 2]
                B[5, i*3+2]   = dN[i, 0]

            strain = B @ elem_u
            stress = D @ strain

            # Von Mises stress
            sx, sy, sz = stress[0], stress[1], stress[2]
            txy, tyz, txz = stress[3], stress[4], stress[5]

            vm = np.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2 + 6*(txy**2 + tyz**2 + txz**2)))

            element_stresses.append(vm)

            # Assign to nodes (average)
            for node in elem_nodes:
                node_idx = int(node) - 1
                von_mises[node_idx] += vm / 4.0  # Average contribution

        # Compute stress gradient (finite difference)
        for i, coord in enumerate(mesh_data['node_coords']):
            # Find nearby nodes (simple radius search)
            distances = np.linalg.norm(mesh_data['node_coords'] - coord, axis=1)
            nearby = np.where((distances < 50.0) & (distances > 0))[0]

            if len(nearby) > 0:
                stress_diff = von_mises[nearby] - von_mises[i]
                dist_diff = distances[nearby]
                gradients = stress_diff / (dist_diff + 1e-10)
                stress_gradient[i] = np.max(np.abs(gradients))

        return {
            'von_mises': von_mises,
            'stress_gradient': stress_gradient,
            'element_stresses': np.array(element_stresses)
        }

    def _identify_refinement_zones(self,
                                   mesh_data: Dict,
                                   von_mises: np.ndarray,
                                   stress_gradient: np.ndarray) -> List[Dict]:
        """
        Identify regions needing refinement based on stress gradients

        Refine top 20% stress gradient regions
        """
        zones = []

        # Top 20% stress gradient threshold
        threshold = np.percentile(stress_gradient, 80)

        # Find high-gradient nodes
        high_gradient_nodes = np.where(stress_gradient > threshold)[0]

        self._log(f"  High gradient nodes: {len(high_gradient_nodes)} "
                 f"(threshold: {threshold:.2e} Pa/m)")

        # Group nearby nodes into zones (simple spatial clustering)
        # For now, just return individual node zones
        for node_idx in high_gradient_nodes[:20]:  # Limit to top 20 nodes
            coord = mesh_data['node_coords'][node_idx]
            zones.append({
                'type': 'stress_concentration',
                'location': coord,
                'stress': von_mises[node_idx],
                'gradient': stress_gradient[node_idx],
                'priority': 0.85,
                'refinement_factor': 2.5
            })

        return zones

    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(message)


def test_stress_predictor():
    """Test stress predictor on sample CAD file"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stress_predictor.py <cad_file.step>")
        sys.exit(1)

    cad_file = sys.argv[1]

    predictor = StressPredictor(verbose=True)
    results = predictor.predict_stress_concentration(input_file=cad_file)

    print("\n" + "=" * 70)
    print("STRESS PREDICTION RESULTS")
    print("=" * 70)
    print(f"Execution time: {results['execution_time_ms']:.1f}ms")
    print(f"Max stress: {np.max(results['stress_field']):.2e} Pa")
    print(f"Max gradient: {np.max(results['stress_gradient']):.2e} Pa/m")
    print(f"Refinement zones: {len(results['refinement_zones'])}")

    for i, zone in enumerate(results['refinement_zones'][:5]):
        print(f"\n  Zone {i+1}:")
        print(f"    Location: ({zone['location'][0]:.2f}, {zone['location'][1]:.2f}, {zone['location'][2]:.2f})")
        print(f"    Stress: {zone['stress']:.2e} Pa")
        print(f"    Gradient: {zone['gradient']:.2e} Pa/m")
        print(f"    Refinement: {zone['refinement_factor']:.1f}x")

    gmsh.finalize()


if __name__ == "__main__":
    test_stress_predictor()
