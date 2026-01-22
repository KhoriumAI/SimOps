
import os

TEMPLATE_DIR = r"c:\Users\markm\Downloads\Simops\simops\templates\Golden_Thermal_Case"

def write_file(path, content):
    full_path = os.path.join(TEMPLATE_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    print(f"Created {path}")

# 1. regionProperties
write_file("constant/regionProperties", """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      regionProperties;
}

regions
(
    fluid       ()
    solid       (solid_heatsink solid_chip)
);
""")

# 2. Thermophysical Properties (Aluminum for both for simplicity)
thermo_props = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties;
}

thermoType
{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie
    {
        molWeight   26.98;
    }
    thermodynamics
    {
        Cp          900;
        Hf          0;
    }
    transport
    {
        kappa       200;
    }
    equationOfState
    {
        rho         2700;
    }
}
"""
write_file("constant/solid_heatsink/thermophysicalProperties", thermo_props)
write_file("constant/solid_chip/thermophysicalProperties", thermo_props)

# 3. fvSchemes (Minimal)
fv_schemes = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none; 
    div(phi,h)      Gauss upwind;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""
write_file("system/solid_heatsink/fvSchemes", fv_schemes)
write_file("system/solid_chip/fvSchemes", fv_schemes)

# 4. fvSolution (Minimal)
fv_solution = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    h
    {
        solver          PBiCGStab;
        preconditioner  DIC;
        tolerance       1e-6;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    equations
    {
        h               1;
    }
}
"""
write_file("system/solid_heatsink/fvSolution", fv_solution)
write_file("system/solid_chip/fvSolution", fv_solution)

# 5. T Fields
# Heatsink: Bottom fixed 300K, Interface Coupled, Rest ZeroGradient
t_heatsink = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0/solid_heatsink";
    object      T;
}

dimensions      [0 0 0 1 0 0 0];
internalField   uniform 300;

boundaryField
{
    heatsink_bottom
    {
        type            fixedValue;
        value           uniform 300;
    }
    solid_heatsink_to_solid_chip
    {
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        Tnbr            T;
        kappaMethod     solidThermo;
        value           uniform 300;
        sampleMode      nearestPatchFace;
    }
    defaultFaces
    {
        type            zeroGradient;
    }
}
"""
write_file("0/solid_heatsink/T", t_heatsink)

# Chip: Top fixed 350K (Source), Interface Coupled, Rest ZeroGradient
t_chip = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0/solid_chip";
    object      T;
}

dimensions      [0 0 0 1 0 0 0];
internalField   uniform 300;

boundaryField
{
    chip_top
    {
        type            fixedValue;
        value           uniform 350;
    }
    solid_chip_to_solid_heatsink
    {
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        Tnbr            T;
        kappaMethod     solidThermo;
        value           uniform 300;
        sampleMode      nearestPatchFace;
    }
    defaultFaces
    {
        type            zeroGradient;
    }
}
"""
write_file("0/solid_chip/T", t_chip)

# 5b. Pressure Fields (Required by basicThermo even for solids sometimes)
p_solid = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0/solid_heatsink"; // Will be overwritten by file write
    object      p;
}

dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 100000;

boundaryField
{
    ".*"
    {
        type            calculated;
        value           uniform 100000;
    }
}
"""
write_file("0/solid_heatsink/p", p_solid)
write_file("0/solid_chip/p", p_solid)

# 6. Gravity (Required by chtMultiRegionFoam)
g_file = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 -9.81);
"""
write_file("constant/g", g_file)

print("Case setup complete.")
