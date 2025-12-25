import os

# usage: python pack_code.py > context.txt

# Extensions to include (Source code and config)
EXTENSIONS = {
    # Core Code
    '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.java', '.cpp', '.c', '.h',
    # Config & Scripts
    '.json', '.yaml', '.yml', '.xml', '.bat', '.sh', '.dockerfile',
    # Documentation
    '.md', '.txt'
}

# Specific filenames to include (if they don't match extensions)
INCLUDE_FILES = {
    'Dockerfile', 'Dockerfile.worker', 'Dockerfile.watcher', 
    'Dockerfile.worker-base', 'Dockerfile.worker-fast',
    'requirements.txt', 'requirements-worker.txt', 'requirements-watcher.txt',
    'Appfile', 'Procfile'
}

# Directories to IGNORE (Data, Builds, Env, Git)
IGNORE_DIRS = {
    # Version Control
    '.git', '.github', '.svn', '.hg',
    # Python/Env
    '__pycache__', 'venv', '.venv', 'env', 'dist', 'build', 'distutils', 'egg-info',
    # Node/JS
    'node_modules', 'bower_components',
    # Data / Artifacts / Binary Assets
    'logs', 
    'output', 
    'input', 
    'cad_files', 
    'generated_meshes',
    'validation_results',
    'strategy_test_env',
    'structural_test_env',
    # IDE Settings
    '.idea', '.vscode', '.vs'
}

def is_text_file(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS) or filename in INCLUDE_FILES

def print_files(startpath):
    print(f"Packing code from: {os.path.abspath(startpath)}")
    print(f"Ignoring directories: {', '.join(sorted(IGNORE_DIRS))}")
    print("-" * 50)
    
    for root, dirs, files in os.walk(startpath):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for f in files:
            if is_text_file(f):
                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, startpath)
                
                # XML-style tags help Gemini distinguish files clearly
                print(f"<file path=\"{rel_path}\">")
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as content:
                        print(content.read())
                except Exception as e:
                    print(f"# Error reading file: {e}")
                print("</file>\n")

if __name__ == '__main__':
    print_files('.')
