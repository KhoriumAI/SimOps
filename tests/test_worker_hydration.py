
import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from simops_worker import run_thermal_solver
from core.schemas.config_schema import SimulationConfig, PhysicsConfig

class TestWorkerHydration(unittest.TestCase):
    @patch('simops_worker.CalculiXAdapter')
    def test_hydration_copper(self, MockAdapter):
        # Setup
        mock_instance = MockAdapter.return_value
        mock_instance.run.return_value = {
            'min_temp': 300.0, 
            'max_temp': 400.0, 
            'elements': [],
            'node_coords': [],
            'temperature': []
        }
        
        # Config with Copper (Table in Library)
        sim_config = SimulationConfig(
            physics=PhysicsConfig(material="Copper_C110")
        )
        
        # Act
        run_thermal_solver(Path("dummy.msh"), Path("output"), "TestStrat", sim_config)
        
        # Assert
        # args: mesh, output, config
        call_args = mock_instance.run.call_args
        config_passed = call_args[0][2] 
        
        print(f"Hydrated Config: {config_passed}")
        
        # Verify Copper Properties
        # Conductivity: [[400.0, 250.0], [390.0, 300.0], ...]
        k = config_passed.get('thermal_conductivity')
        self.assertIsNotNone(k)
        self.assertTrue(isinstance(k, list), "Conductivity should be a list (Table)")
        self.assertEqual(len(k), 3)
        self.assertEqual(k[0], [400.0, 250.0]) # Value, Temp
        
        # Density
        rho = config_passed.get('density')
        self.assertEqual(rho, 8960.0)
        
        # Specific Heat
        cp = config_passed.get('specific_heat')
        self.assertEqual(cp, 385.0)

    @patch('simops_worker.CalculiXAdapter')
    def test_local_override(self, MockAdapter):
        mock_instance = MockAdapter.return_value
        mock_instance.run.return_value = {
            'min_temp': 300.0, 'max_temp': 400.0, 'elements': []
        }
        
        # Config with Aluminum BUT override K
        # Aluminum default K is ~150 (Table)
        # Override with Scalar 200.0
        phy = PhysicsConfig(material="Aluminum_6061")
        phy.thermal_conductivity = 200.0 # Override
        
        sim_config = SimulationConfig(physics=phy)
        
        run_thermal_solver(Path("dummy.msh"), Path("output"), "TestStrat", sim_config)
        
        config_passed = mock_instance.run.call_args[0][2]
        
        # Assert Override
        self.assertEqual(config_passed['thermal_conductivity'], 200.0)
        # Assert loaded others
        self.assertEqual(config_passed['density'], 2700.0)

if __name__ == '__main__':
    unittest.main()
