#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "gDel3D/GpuDelaunay.h"

namespace py = pybind11;

// Wrapper function to expose to Python
py::array_t<int> compute_delaunay(py::array_t<double> input_points) {
    // 1. Initialize CUDA (Safety)
    cudaSetDevice(0);

    // 2. Parse Input (Numpy -> Host Vector)
    py::buffer_info buf = input_points.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input must be an N x 3 numpy array of float64");
    }

    int num_points = buf.shape[0];
    double* ptr = static_cast<double*>(buf.ptr);

    // Create Thrust Host Vector
    // Note: We copy data here because Point3 struct layout matches 3 doubles,
    // but we need to load it into the Thrust container gDel3D expects.
    Point3HVec points(num_points);
    for (int i = 0; i < num_points; i++) {
        points[i]._p[0] = ptr[i*3 + 0];
        points[i]._p[1] = ptr[i*3 + 1];
        points[i]._p[2] = ptr[i*3 + 2];
    }

    // 3. Run gDel3D
    GDelParams params;
    params.verbose = false; // Silence output for Python
    params.noSorting = false;
    params.noSplaying = true;  // CRITICAL: Disable splaying to preserve full mesh
    
    GpuDel triangulator(params);
    GDelOutput output;
    
    try {
        triangulator.compute(points, &output);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("GPU Meshing Failed: ") + e.what());
    }

    // 4. Output (Tetrahedra -> Numpy)
    // gDel3D outputs 'Tet' structs with int _v[4]
    int num_tets = output.tetVec.size();
    
    // Allocate Numpy array (NumTets x 4)
    py::array_t<int> result({num_tets, 4});
    auto r = result.mutable_unchecked<2>();

    for (int i = 0; i < num_tets; i++) {
        const Tet& t = output.tetVec[i];
        r(i, 0) = t._v[0];
        r(i, 1) = t._v[1];
        r(i, 2) = t._v[2];
        r(i, 3) = t._v[3];
    }

    return result;
}

PYBIND11_MODULE(_gpumesher, m) {
    m.doc() = "Khorium AI GPU Mesher Backend - gDel3D Wrapper";
    m.def("compute_delaunay", &compute_delaunay, 
          py::arg("points"),
          "Computes 3D Delaunay triangulation on GPU.\n\n"
          "Parameters:\n"
          "    points : numpy.ndarray (N, 3) of float64\n"
          "        Input point cloud\n\n"
          "Returns:\n"
          "    numpy.ndarray (M, 4) of int32\n"
          "        Tetrahedra as vertex indices");
}
