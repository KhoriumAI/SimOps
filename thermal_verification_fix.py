import os

def fix_calculix_adapter():
    path = "core/solvers/calculix_adapter.py"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # 1. Fix NSET writing to avoid trailing comma
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "*NSET, NSET=N_All" in line:
            new_lines.append("                # Corrected NSET writing (no trailing comma)\n")
            new_lines.append("                f.write(\"*NSET, NSET=N_All\\n\")\n")
            new_lines.append("                tag_strs = [str(t) for t in node_tags]\n")
            new_lines.append("                for k in range(0, len(tag_strs), 8):\n")
            new_lines.append("                    f.write(\", \".join(tag_strs[k:k+8]) + \"\\n\")\n")
            # Skip old loop
            while i < len(lines) and ("f.write" in lines[i] or "for " in lines[i] or "if " in lines[i] or "*NSET" in lines[i]):
                i += 1
            # Adjust i to skip the next line if it's empty
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        new_lines.append(line)
        i += 1
        
    # 2. Fix FRD parsing to use robust split
    final_lines = []
    i = 0
    while i < len(new_lines):
        line = new_lines[i]
        if "T_array.append(float(line" in line or "val_raw = line[" in line:
            # Reconstruct the try block
            final_lines.append("                            try:\n")
            final_lines.append("                                # Robust parsing: -1 NNN VVVV\n")
            final_lines.append("                                parts = line.split()\n")
            final_lines.append("                                if len(parts) >= 3 and parts[0] == \"-1\":\n")
            final_lines.append("                                    T_array.append(float(parts[2]))\n")
            final_lines.append("                            except (ValueError, IndexError):\n")
            final_lines.append("                                pass\n")
            # Scan forward to find end of previous messy try block
            j = i
            while j < len(new_lines) and "pass" not in new_lines[j]:
                j += 1
            i = j + 1
            continue
        final_lines.append(line)
        i += 1
        
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(final_lines)
    print("Fixed calculix_adapter.py")

if __name__ == "__main__":
    fix_calculix_adapter()
