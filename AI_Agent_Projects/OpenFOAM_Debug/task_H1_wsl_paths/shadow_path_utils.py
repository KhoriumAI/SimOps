from pathlib import Path

def to_wsl(path_str):
    """
    Robustly convert a Windows path to a WSL path without external commands.
    Assumes standard /mnt/[drive]/ structure.
    """
    p = Path(path_str).absolute()
    # 1. Get drive letter
    drive = p.drive.lower().replace(':', '')
    # 2. Convert raw path to posix (handles backslashes)
    posix_path = p.as_posix() # e.g. C:/Users/...
    # 3. Construct WSL path
    # posix_path[2:] strips the "C:" part from the start
    return f"/mnt/{drive}{posix_path[2:]}"
