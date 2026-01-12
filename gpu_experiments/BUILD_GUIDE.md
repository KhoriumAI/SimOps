# GPU Mesher Build Guide

## Prerequisites

1. **NVIDIA CUDA Toolkit 12.x** (you have 12.8 installed ✓)
2. **Visual Studio 2019/2022** with C++ development tools
3. **CMake 3.10+**
4. **Python 3.10** environment (for binary compatibility)

## Quick Build (Windows)

```powershell
# 1. Navigate to gpu_experiments
cd C:\Users\markm\Downloads\MeshPackageLean\gpu_experiments

# 2. Create build directory
mkdir build
cd build

# 3. Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64

# 4. Build the Python module
cmake --build . --config Release --target _gpumesher

# 5. Copy the built _gpumesher.pyd to core/
copy Release\_gpumesher.pyd ..\..core\_gpumesher.pyd
```

## Troubleshooting

### "CUDA not found"
- Ensure `nvcc --version` works in your terminal
- Add CUDA bin to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`

### "Python.h not found"
- Activate your Python 3.10 environment before running CMake: `conda activate mesh310`

### Compiler Errors
- The build uses `-arch=sm_75` (Turing/Ampere GPUs). If you have a different GPU architecture, update line 60/84/125/140 in `CMakeLists.txt`:
  - RTX 30-series: `-arch=sm_86`
  - RTX 40-series: `-arch=sm_89`
  - Check your GPU: `nvidia-smi`

## Docker Build (Future)

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
# Install Python 3.10, CMake, etc.
# Build _gpumesher.so using the same CMake commands
```

## AWS/Cloud Build

For AWS deployment, use the same Docker image and ensure the instance has:
- NVIDIA drivers
- CUDA runtime 12.x
- Matching GPU architecture (`sm_*`)

---

*Last updated: 2026-01-11*
