
import sys

def inspect(file_path):
    print(f"Inspecting {file_path}")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        print("Block Headers:")
        for i, line in enumerate(lines):
            if line.startswith(" -4"):
                print(f"Line {i}: {line.strip()}")
            if i < 10: # Print first 10 header lines
                print(f"Header {i}: {line.strip()}")
    except UnicodeDecodeError:
        print("File appears to be binary (UnicodeDecodeError).")
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_frd.py <file>")
