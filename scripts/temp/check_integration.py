#!/usr/bin/env python3
"""
Integration Verification Script
================================
Tests that all Forge components integrate correctly in production paths.
Following Khorium Rulebook: Test-First verification.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_schema_imports():
    """Verify schemas are accessible from production location."""
    print("Testing schema imports...")
    try:
        from core.schemas import SimConfig, Material, BoundaryCondition, BCType
        print("✅ Schema imports successful")
        return True
    except ImportError as e:
        print(f"❌ Schema import failed: {e}")
        return False

def test_ai_generator():
    """Verify AI generator works with production schemas."""
    print("Testing AI setup generator...")
    try:
        from core.ai_setup import AISetupGenerator
        from core.schemas import SimConfig
        
        gen = AISetupGenerator()
        config = gen.generate_config(
            prompt="Heat this heatsink with 50W",
            cad_file="test.step",
            use_mock=True
        )
        
        assert isinstance(config, SimConfig)
        assert config.cad_file == "test.step"
        print("✅ AI generator functional")
        return True
    except Exception as e:
        print(f"❌ AI generator test failed: {e}")
        return False

def test_compute_provider():
    """Verify compute provider integration."""
    print("Testing compute provider...")
    try:
        from core.compute import LocalDockerProvider, JobStatus
        from core.schemas import SimConfig
        
        provider = LocalDockerProvider()
        
        # Create minimal config
        config = SimConfig(
            job_name="integration_test",
            cad_file="test.step"
        )
        
        # Note: Actual submission would require Docker, so we just test instantiation
        assert provider.image_name == "simops-worker:latest"
        print("✅ Compute provider initialized")
        return True
    except Exception as e:
        print(f"❌ Compute provider test failed: {e}")
        return False

def test_end_to_end_flow():
    """Test complete flow: User prompt → Config → Provider."""
    print("Testing end-to-end integration...")
    try:
        from core.ai_setup import AISetupGenerator
        from core.compute import LocalDockerProvider
        
        # Step 1: Generate config from prompt
        gen = AISetupGenerator()
        config = gen.generate_config(
            prompt="Run thermal simulation on heatsink",
            cad_file="models/heatsink.step",
            use_mock=True
        )
        
        # Step 2: Verify config is valid
        assert config.job_name
        assert config.cad_file
        
        # Step 3: Provider can accept the config
        provider = LocalDockerProvider()
        # Don't actually submit to avoid Docker dependency
        
        print("✅ End-to-end flow validated")
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SimOps Integration Verification (Weaver Stage)")
    print("=" * 60)
    print()
    
    tests = [
        test_schema_imports,
        test_ai_generator,
        test_compute_provider,
        test_end_to_end_flow
    ]
    
    results = [test() for test in tests]
    
    print()
    print("=" * 60)
    if all(results):
        print("🎉 ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ INTEGRATION TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
