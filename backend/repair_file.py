import os
import sys

path = r'c:\Users\markm\Downloads\MeshPackageLean\backend\api_server.py'

# Read entire file
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Look for corruption pattern
corrected_lines = []
for line in lines:
    if "commit()L_COMPUTE" in line:
        print(f"Found corruption in line: {line.strip()}")
        # Based on my earlier observation, it looks like a \r or overlap
        # Let's just manually fix known corrupted patterns if they appear
        line = line.replace("commit()L_COMPUTE', False):", "commit()\n            if app.config.get('USE_MODAL_COMPUTE', False):")
    corrected_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(corrected_lines)

print("Repair attempt finished.")
