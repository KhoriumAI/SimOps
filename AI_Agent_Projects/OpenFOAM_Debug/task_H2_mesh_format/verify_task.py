import sys
import os

def verify():
    filename = "fixed.msh"
    if not os.path.exists(filename):
        print(f"FAIL: {filename} does not exist.")
        sys.exit(1)

    print(f"Verifying {filename}...")
    
    has_error = False
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # Check Version 2.2
    # First line: $MeshFormat
    # Second line: 2.2 0 8
    if len(lines) > 2:
        version_line = lines[1].strip()
        if not version_line.startswith("2.2"):
            print(f"FAIL: Header version is not 2.2. Found: '{version_line}'")
            has_error = True
        else:
            print("PASS: Header version is 2.2")
    else:
        print("FAIL: File too short")
        has_error = True

    # Check for Element Type 15 (Point) in $Elements section
    # Format 2.2 Elements:
    # $Elements
    # numElements
    # serialNum type tagCount tag1 ... node1 ...
    
    in_elements = False
    found_type_15 = False
    
    for line in lines:
        line = line.strip()
        if line == "$Elements":
            in_elements = True
            continue
        if line == "$EndElements":
            in_elements = False
            continue
            
        if in_elements:
            parts = line.split()
            if len(parts) > 1:
                # First line is count, skip constraint
                if len(parts) == 1: continue 
                
                # Element line parts:
                # [0] serial
                # [1] type
                # Type 15 = 1-node point
                elem_type = parts[1]
                if elem_type == '15':
                    found_type_15 = True
                    # Let's verify tags to see if it's truly a physical point? 
                    # Yes, any element 15 is bad for gmshToFoam usually.
                    print(f"FAIL: Found Element Type 15: {line}")
                    break
    
    if found_type_15:
        has_error = True
    else:
        print("PASS: No Element Type 15 found")

    if has_error:
        print("VERIFICATION FAILED")
        sys.exit(1)
    else:
        print("VERIFICATION SUCCESS")
        sys.exit(0)

if __name__ == "__main__":
    verify()
