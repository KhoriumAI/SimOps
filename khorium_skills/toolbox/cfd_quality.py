"""
CFD Quality Analyzer CLI (Skill)
================================
Standalone CLI for CFD mesh quality analysis matching OpenFOAM checkMesh output.
Refactored from core/cfd_quality.py for standalone usage.

Usage:
    python khorium_skills/toolbox/cfd_quality.py mesh.msh
    python khorium_skills/toolbox/cfd_quality.py mesh.msh --json
    python khorium_skills/toolbox/cfd_quality.py mesh.msh --quiet --json
"""

import sys
import os
import argparse

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.cfd_quality import CFDQualityAnalyzer, CFDQualityReport


def main():
    parser = argparse.ArgumentParser(
        description="CFD Mesh Quality Analyzer (OpenFOAM checkMesh equivalent)"
    )
    parser.add_argument("mesh_file", help="Path to mesh file (.msh)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mesh_file):
        print(f"Error: File not found: {args.mesh_file}")
        sys.exit(1)
    
    analyzer = CFDQualityAnalyzer(verbose=not args.quiet)
    report = analyzer.analyze_mesh_file(args.mesh_file)
    
    if args.json:
        print(report.to_json())
    else:
        report.print_report()
    
    sys.exit(0 if report.cfd_ready else 1)


if __name__ == "__main__":
    main()
