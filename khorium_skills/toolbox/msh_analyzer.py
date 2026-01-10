"""
MSH Analyzer (Skill)
====================
Analyzes the structure of a .msh file, specifically checking for MSH 4.x element blocks.
Refactored from analyze_msh.py.

Usage:
    python khorium_skills/toolbox/msh_analyzer.py file.msh
"""

import sys
import os
import argparse

def analyze_msh_file(path: str):
    """
    Reads a .msh file and prints its header and element block structure.
    """
    print(f"Analyzing {path}")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        with open(path, 'r') as f:
            content = f.read()

        if '$MeshFormat' in content:
            header = content.split('$MeshFormat')[1].strip().split('\n')[0]
            print(f"Mesh Format: {header}")
        
        if '$Elements' in content:
            el_section = content.split('$Elements')[1].split('$EndElements')[0].strip().split('\n')
            header = el_section[0].split()
            print(f"Elements Header: {header}")
            
            # MSH 4 check
            if header[0].startswith('4') or (len(header) == 4 and '.' not in header[0]):
                 # numEntityBlocks numElements minTag maxTag
                 num_blocks = int(header[0])
                 print(f"Num Blocks: {num_blocks}")
                 
                 curr = 1
                 for i in range(num_blocks):
                     if curr >= len(el_section): break
                     block_header = el_section[curr].split()
                     if not block_header: continue
                     # tag(0) dim(1) type(2) num(3)
                     print(f"Block {i}: {block_header}")
                     try:
                         num = int(block_header[3])
                         curr += num + 1
                     except:
                         print("Error parsing block")
                         break
            else:
                 print("Not MSH 4.x element structure?")
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze MSH file")
    parser.add_argument("path", help="Path to .msh file")
    args = parser.parse_args()
    
    analyze_msh_file(args.path)

if __name__ == "__main__":
    main()
