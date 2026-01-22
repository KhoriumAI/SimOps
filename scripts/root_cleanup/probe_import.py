
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

def probe():
    print(f"CWD: {os.getcwd()}")
    try:
        import core.reporting.cfd_report as cr
        print(f"Found module: {cr}")
        print(f"File: {cr.__file__}")
        
        from core.reporting.cfd_report import CFDPDFReportGenerator
        print(f"Found Class: {CFDPDFReportGenerator}")
    except ImportError as e:
        print(f"Import Failed: {e}")
        # Search for file manually
        search_path = Path("core/reporting")
        if search_path.exists():
            print(f"Listing {search_path}:")
            for f in search_path.glob("*"):
                print(f" - {f.name}")
        else:
            print(f"{search_path} does not exist")

if __name__ == "__main__":
    probe()
