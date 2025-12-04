#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

#include "gDel3D/GpuDelaunay.h"

// Helper to inspect the raw hex bytes of a float/double
void printHex(RealType v) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
    for (size_t i = 0; i < sizeof(RealType); ++i) 
        printf("%02x", p[i]);
}

int main() {
    // 1. Hardware Init
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) return 1;

    std::cout << "=== DIAGNOSTIC MODE ===" << std::endl;
    std::cout << "Size of RealType: " << sizeof(RealType) << " bytes (Expected: 8 for double)" << std::endl;
    std::cout << "Size of Point3:   " << sizeof(Point3)   << " bytes" << std::endl;

    // 2. Data Gen
    const int numPoints = 100000;
    Point3HVec points(numPoints);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<RealType> dist(0.1, 0.9); // Safe internal range

    std::cout << "Generating points..." << std::endl;
    for(int i = 0; i < numPoints; ++i) {
        points[i]._p[0] = dist(rng);
        points[i]._p[1] = dist(rng);
        points[i]._p[2] = dist(rng);
    }

    // 3. DATA INSPECTION (Crucial Step)
    std::cout << "\n--- Input Data Sample (Host) ---" << std::endl;
    for(int i = 0; i < 3; i++) {
        std::cout << "Pt " << i << ": " 
                  << points[i]._p[0] << ", " 
                  << points[i]._p[1] << ", " 
                  << points[i]._p[2] << " | Hex: ";
        printHex(points[i]._p[0]);
        std::cout << std::endl;
    }

    // 4. Setup with VERBOSE
    GDelParams params;
    params.verbose = true;    // <--- ENABLE THIS to see kernel stages
    params.noSorting = false;
    params.noSplaying = true;  // DISABLE splaying - it's destroying the mesh!
    
    GpuDel triangulator(params);
    GDelOutput output;

    std::cout << "\n--- Starting Compute ---" << std::endl;
    try {
        triangulator.compute(points, &output);
    } catch (const std::exception& e) {
        std::cerr << "CRASH: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Final Tetrahedra: " << output.stats.finalStarNum << std::endl;
    
    // Check internal counters if available in your fork's stats
    // Note: Standard gDel3D doesn't expose 'pointNum' in stats, 
    // but looking at verbose output will tell us.

    return 0;
}