import os

# usage: python pack_code.py > context.txt

# Extensions to include
EXTENSIONS = {
    '.py', '.js', '.ts', '.html', '.css', '.json', 
    '.sh', '.bat', '.md', '.mmd', '.vue', '.jsx', '.tsx'
}

# Folders to explicitly ignore for MeshPackageLean
IGNORE_DIRS = {
    # Standard noise
    'node_modules', '.git', '.github', '__pycache__', 'venv', 'env', 
    'dist', 'build', '.idea', '.vscode',
    
    # Project specific data/assets
    'cad_files',    # likely binaries/large stls
    'output',       # generated simulation results
    'resources',    # images/icons
    'docs',         # usually not needed for coding tasks
    'distutils'
}

def is_text_file(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def print_files(startpath):
    for root, dirs, files in os.walk(startpath):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for f in files:
            if is_text_file(f):
                path = os.path.join(root, f)
                # Make path relative for cleaner context
                rel_path = os.path.relpath(path, startpath)
                
                print(f"<file path=\"{rel_path}\">")
                try:
                    with open(path, 'r', encoding='utf-8') as content:
                        print(content.read())
                except Exception as e:
                    print(f"# Error reading file: {e}")
                print("</file>\n")

if __name__ == '__main__':
    print_files('.')
