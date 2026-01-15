# CalculiX Native

This directory contains the native binaries for CalculiX, an Open Source Finite Element Program. It is used for thermal and structural simulations within the MeshGen pipeline.

## Contents

- **CalculiX-2.23.0-win-x64/**: Pre-built Windows 64-bit binaries (ccx.exe).

## Integration

The Python backend interacts with these binaries by:
1. Generating input files (`.inp`).
2. Executing `ccx.exe` as a subprocess.
3. Parsing the output files (`.frd`, `.dat`) for visualization and reporting.

## External Links
- [CalculiX Official Site](http://www.calculix.de/)
- [CalculiX Documentation](http://www.dhondt.de/ccx_2.21.pdf)
