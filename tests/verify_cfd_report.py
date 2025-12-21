
import unittest
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.reporting.cfd_report import CFDPDFReportGenerator

class TestCFDReport(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("output/test_report")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy image
        self.img_path = self.output_dir / "dummy_flow.png"
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([0,1],[0,1])
        plt.title("Dummy Flow")
        plt.savefig(self.img_path)
        plt.close()

    def test_generate_report(self):
        generator = CFDPDFReportGenerator()
        
        data = {
            'strategy_name': 'HighFi_CFD',
            'reynolds_number': 1234.5,
            'drag_coefficient': 0.45,
            'lift_coefficient': 0.01,
            'u_inlet': 5.0,
            'mesh_cells': 50000,
            'solve_time': 12.3,
            'converged': True
        }
        
        pdf_path = generator.generate(
            job_name="UnitTest_CFD",
            output_dir=self.output_dir,
            data=data,
            image_paths=[str(self.img_path)]
        )
        
        print(f"Generated PDF: {pdf_path}")
        self.assertTrue(Path(pdf_path).exists())
        self.assertTrue(Path(pdf_path).stat().st_size > 1000)

if __name__ == '__main__':
    unittest.main()
