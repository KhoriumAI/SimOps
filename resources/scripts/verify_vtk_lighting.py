import vtk
import sys

def verify_light_kit():
    print("Verifying vtkLightKit API...")
    try:
        light_kit = vtk.vtkLightKit()
        
        # Test Key Light Intensity
        print("Testing SetKeyLightIntensity(0.75)...")
        light_kit.SetKeyLightIntensity(0.75)
        
        # Test Fill Ratio (The fix)
        print("Testing SetKeyToFillRatio(2.0)...")
        if hasattr(light_kit, 'SetKeyToFillRatio'):
            light_kit.SetKeyToFillRatio(2.0)
            print("  [OK] SetKeyToFillRatio exists")
        else:
            print("  [FAIL] SetKeyToFillRatio NOT found")
            return False
            
        # Test Head Ratio
        print("Testing SetKeyToHeadRatio(3.0)...")
        if hasattr(light_kit, 'SetKeyToHeadRatio'):
            light_kit.SetKeyToHeadRatio(3.0)
            print("  [OK] SetKeyToHeadRatio exists")
        else:
            print("  [FAIL] SetKeyToHeadRatio NOT found")
            return False
            
        print("\nAll API calls successful.")
        return True
        
    except Exception as e:
        print(f"\n[CRITICAL] Exception during verification: {e}")
        return False

if __name__ == "__main__":
    success = verify_light_kit()
    sys.exit(0 if success else 1)
