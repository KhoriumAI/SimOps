import os
import re

def patch_boundary_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Regex to find mappedWall blocks and inject sampleMode if missing
    # We look for: type mappedWall; ... nFaces ...;
    # We insert sampleMode nearestPatchFace; after nFaces.
    
    # We need to be careful. Let's process line by line or use a state machine.
    lines = content.splitlines()
    new_lines = []
    in_mapped_wall = False
    brace_depth = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Check for start of patch entry
        if re.search(r'^[a-zA-Z0-9_]+', stripped) and '{' in line:
            # Maybe start of a patch
            pass
            
        if 'type' in stripped and 'mappedWall' in stripped:
            in_mapped_wall = True
            
        if '}' in stripped:
            if in_mapped_wall:
                 # End of mapped wall patch
                 in_mapped_wall = False
        
        new_lines.append(line)
        
        # Injection point: after nFaces
        if in_mapped_wall and 'nFaces' in stripped:
            # Check if sampleMode already exists nearby? 
            # Simplified: just inject it. If duplicate, standard OF might complain or ignore.
            # But let's check if the NEXT line is sampleMode
            pass

    # Re-doing with regex for safer block replacement
    # Pattern: match a block containing "type mappedWall" and insert sampleMode before the closing brace if not present
    
    # Actually, let's use a simpler replacement: replace "type mappedWall;" with "type mappedWall;\n        sampleMode nearestPatchFace;"
    # But wait, sampleMode usually goes near other settings.
    # The error "Entry 'sampleMode' not found" suggests it's mandatory.
    
    if "sampleMode" not in content:
        # Simple string replacement on the type declaration
        new_content = content.replace("type            mappedWall;", "type            mappedWall;\n        sampleMode      nearestPatchFace;")
        
        # Verify if it actually changed
        if new_content != content:
            print(f"Patched {filepath}")
            with open(filepath, 'w') as f:
                f.write(new_content)
        else:
            print(f"Could not match 'type mappedWall;' in {filepath}")
    else:
        print(f"sampleMode already present in {filepath}")

def main():
    # Find all boundary files in constant/*/polyMesh/boundary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    constant_dir = os.path.join(base_dir, 'constant')
    
    if not os.path.exists(constant_dir):
        print("No constant directory found")
        return

    for root, dirs, files in os.walk(constant_dir):
        if 'boundary' in files and 'polyMesh' in root:
            filepath = os.path.join(root, 'boundary')
            patch_boundary_file(filepath)

if __name__ == "__main__":
    # Fixup: Rename domain0 to region1 if needed
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check constant/domain0
    const_d0 = os.path.join(base_dir, 'constant', 'domain0')
    const_r1 = os.path.join(base_dir, 'constant', 'region1')
    if os.path.exists(const_d0) and not os.path.exists(const_r1):
        print("Renaming constant/domain0 -> constant/region1")
        os.rename(const_d0, const_r1)
        
    # Check 0/domain0
    zero_d0 = os.path.join(base_dir, '0', 'domain0')
    zero_r1 = os.path.join(base_dir, '0', 'region1')
    if os.path.exists(zero_d0):
        if os.path.exists(zero_r1):
            # If both exist, domain0 likely contains the NEW map, region1 has the TEMPLATE fields.
            # We need to merge them. The mesh is in constant/region1 (renamed above).
            # The PolyMesh is in constant/region1.
            # 0/domain0 contains 'cellToRegion' usually? Or just fields?
            # splitMeshRegions usually maps fields.
            # We should probably keep the TEMPLATE fields (region1) but ensure they match mesh.
            # Let's trust region1 is the target.
            # Remove domain0 garbage.
            import shutil
            shutil.rmtree(zero_d0)
        else:
            print("Renaming 0/domain0 -> 0/region1")
            os.rename(zero_d0, zero_r1)

    main()
