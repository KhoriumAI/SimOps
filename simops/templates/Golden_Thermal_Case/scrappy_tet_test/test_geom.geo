
// Gmsh script for a simple thermal test case (Tetrahedral mesh)
// SCRAPPY: Just a simple block (chip) on a plate (heatsink)

// Dimensions (meters)
hs_L = 0.1;
hs_W = 0.1;
hs_H = 0.01;

chip_L = 0.02;
chip_W = 0.02;
chip_H = 0.005;

// Heatsink Volume
Point(1) = {-hs_L/2, -hs_W/2, 0, 0.01};
Point(2) = { hs_L/2, -hs_W/2, 0, 0.01};
Point(3) = { hs_L/2,  hs_W/2, 0, 0.01};
Point(4) = {-hs_L/2,  hs_W/2, 0, 0.01};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Extrude {0, 0, -hs_H} {
  Surface{1};
}

// Chip Volume (Sitting on top of heatsink at z=0)
Point(101) = {-chip_L/2, -chip_W/2, 0, 0.002};
Point(102) = { chip_L/2, -chip_W/2, 0, 0.002};
Point(103) = { chip_L/2,  chip_W/2, 0, 0.002};
Point(104) = {-chip_L/2,  chip_W/2, 0, 0.002};

Line(101) = {101, 102};
Line(102) = {102, 103};
Line(103) = {103, 104};
Line(104) = {104, 101};

Curve Loop(101) = {101, 102, 103, 104};
Plane Surface(101) = {101};

Extrude {0, 0, chip_H} {
  Surface{101};
}

// Physical Groups (IMPORTANT for OpenFOAM/CalculiX)
// Volume 1 is Heatsink, Volume 2 is Chip
Physical Volume("solid_heatsink") = {1};
Physical Volume("solid_chip") = {2};

// Surfaces
Physical Surface("heatsink_bottom") = {26}; // Bottom of heatsink
Physical Surface("chip_top") = {50};       // Top of chip

// Coherence to merge vertices at interface?
Coherence;
