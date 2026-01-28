import json
from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

class OpenFOAMFile(BaseModel):
    """
    Base class for all OpenFOAM file models.
    """
    model_config = ConfigDict(extra='ignore', populate_by_name=True)
    
    foam_class: str = "dictionary"
    foam_location: Optional[str] = None
    foam_object: str = "unknown"
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def capture_extras(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = set(cls.model_fields.keys())
            
            extras = {}
            new_data = {}
            existing_extra = data.get('extra_data', {})
            if not isinstance(existing_extra, dict): existing_extra = {}

            for k, v in data.items():
                if k in known:
                    new_data[k] = v
                else:
                    extras[k] = v
            
            if existing_extra:
                extras.update(existing_extra)
            
            new_data['extra_data'] = extras
            return new_data
        return data

    def to_foam_string(self) -> str:
        """
        Convert the model back to OpenFOAM dictionary format string.
        """
        cls_name = getattr(self, "foam_class", "dictionary")
        obj_name = getattr(self, "foam_object", "unknown")
        
        loc = getattr(self, "foam_location", None)
        if not loc:
            loc = "system"
            if obj_name in ["blockMeshDict", "controlDict", "fvSchemes", "fvSolution", "snappyHexMeshDict"]: loc = "system"
            elif obj_name in ["thermophysicalProperties", "turbulenceProperties", "g", "regionProperties"]: loc = "constant"
            elif obj_name in ["faces", "points", "owner", "neighbour", "boundary", "cellZones", "faceZones", "pointZones", "level0Edge", "surfaceIndex", "pointLevel", "cellLevel"]: loc = "constant/polyMesh"
        
        lines = [
            "FoamFile", "{", 
            "    version     2.0;", 
            "    format      ascii;", 
            f"    class       {cls_name};", 
            f"    location    \"{loc}\";", 
            f"    object      {obj_name};", 
            "}", ""
        ]
        
        all_data = self.model_dump(exclude_none=True)
        for k in ["foam_header", "file_name", "foam_class", "foam_object", "foam_location", "extra_data"]:
            all_data.pop(k, None)
            
        all_data.update(self.extra_data)
        
        for field_name, value in all_data.items():
            lines.append(self._format_value(field_name, value))
            
        return "\n".join(lines)

    def _format_value(self, key: str, value: Any, indent: int = 0) -> str:
        prefix = "    " * indent
        if isinstance(value, dict):
            lines = [f"{prefix}{key}", f"{prefix}{{"]
            for k, v in value.items():
                lines.append(self._format_value(k, v, indent + 1))
            lines.append(f"{prefix}}}")
            return "\n".join(lines)
        elif isinstance(value, list):
            val_str = self._format_list_recursive(value)
            return f"{prefix}{key}    {val_str};"
        else:
            if isinstance(value, bool):
                val_str = "true" if value else "false"
            else:
                val_str = str(value)
            return f"{prefix}{key:<16} {val_str};"

    def _format_list_recursive(self, value: List[Any]) -> str:
        items = []
        for v in value:
            if isinstance(v, list):
                items.append(self._format_list_recursive(v))
            else:
                items.append(str(v))
        return "(" + " ".join(items) + ")"

class GenericFOAMFile(OpenFOAMFile):
    model_config = ConfigDict(extra='ignore')
    pass

class FoamListFile(OpenFOAMFile):
    """
    Represents a raw OpenFOAM list file (e.g. faces, owner, neighbour).
    Structure:
    size
    (
      item1
      item2
      ...
    )
    """
    foam_object: str = "unknown"
    size: int
    content: List[Any]

    def to_foam_string(self) -> str:
        cls_name = getattr(self, "foam_class", "faceList" if self.foam_object == "faces" else "labelList") # heuristic
        
        loc = getattr(self, "foam_location", None)
        if not loc:
            loc = "constant/polyMesh" if self.foam_object in ["faces", "points", "owner", "neighbour", "boundary", "cellZones", "faceZones", "pointZones", "level0Edge", "surfaceIndex", "pointLevel", "cellLevel"] else "constant"
        
        lines = [
            "FoamFile", "{", 
            "    version     2.0;", 
            "    format      ascii;", 
            f"    class       {cls_name};", 
            f"    location    \"{loc}\";", 
            f"    object      {self.foam_object};", 
            "}", ""
        ]
        
        is_boundary = cls_name == "polyBoundaryMesh"
        
        if is_boundary:
            lines.append(f"{len(self.content)}")
        else:
            lines.append(f"{self.size}")
            
        lines.append("(")
        
        is_faces = self.foam_object == "faces"
        
        if is_faces:
            i = 0
            while i < len(self.content):
                item = self.content[i]
                
                if isinstance(item, int) and i+1 < len(self.content) and isinstance(self.content[i+1], list):
                    face_size = item
                    face_labels = self.content[i+1]
                    lbls = " ".join(str(x) for x in face_labels)
                    lines.append(f"{face_size}({lbls})")
                    i += 2
                
                elif isinstance(item, list):
                     face_size = len(item)
                     face_labels = item
                     lbls = " ".join(str(x) for x in face_labels)
                     lines.append(f"{face_size}({lbls})")
                     i += 1
                     
                else:
                    lines.append(str(item))
                    i += 1
        elif is_boundary:
            for item in self.content:
                 if isinstance(item, dict):
                     for k, v in item.items():
                         lines.append(self._format_value(k, v, indent=0))
                 else:
                     lines.append(str(item))
        else:
            for item in self.content:
                if isinstance(item, list):
                    formatted_item = "(" + " ".join(str(x) for x in item) + ")"
                    lines.append(formatted_item)
                elif isinstance(item, dict):
                    lines.append("{")
                    for k, v in item.items():
                        lines.append(self._format_value(k, v, indent=1))
                    lines.append("}")
                else:
                    lines.append(str(item))
                
        lines.append(")")
        return "\n".join(lines)

class ControlDict(OpenFOAMFile):
    """
    Defines time control and write settings.
    """
    foam_object: str = "controlDict"
    
    application: str = "foamMultiRun"
    startFrom: str = "startTime"
    startTime: float = 0
    stopAt: str = "endTime"
    endTime: float
    deltaT: float
    writeControl: str = "timeStep"
    writeInterval: float
    purgeWrite: int = 0
    writeFormat: str = "ascii"
    writePrecision: int = 8
    writeCompression: Union[str, bool] = "off"
    timeFormat: str = "general"
    timePrecision: int = 6
    runTimeModifiable: Union[str, bool] = "true"
    
    adjustTimeStep: Optional[Union[str, bool]] = None
    maxCo: Optional[float] = None
    maxDeltaT: Optional[float] = None
    functions: Optional[Dict[str, Any]] = None

    @field_validator('endTime')
    @classmethod
    def check_end_time(cls, v):
        if v <= 0:
            raise ValueError("endTime must be positive")
        return v

class FvSchemes(OpenFOAMFile):
    """
    Defines numerical schemes for discretization.
    """
    foam_object: str = "fvSchemes"

    ddtSchemes: Dict[str, str] = Field(default_factory=lambda: {"default": "steadyState"})
    gradSchemes: Dict[str, str] = Field(default_factory=lambda: {"default": "Gauss linear"})
    divSchemes: Dict[str, str] = Field(default_factory=lambda: {"default": "none"})
    laplacianSchemes: Dict[str, str] = Field(default_factory=lambda: {"default": "Gauss linear limited 0.5"})
    interpolationSchemes: Dict[str, str] = Field(default_factory=lambda: {"default": "linear"})
    snGradSchemes: Dict[str, str] = Field(default_factory=lambda: {"default": "limited 0.5"})
    
    fluxRequired: Optional[Dict[str, Any]] = None
    wallDist: Optional[Dict[str, Any]] = None

class SolverSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    solver: str
    tolerance: float = 1e-7
    relTol: float = 0.1
    smoother: Optional[str] = None
    preconditioner: Optional[str] = None
    minIter: Optional[int] = None
    maxIter: Optional[int] = None
    
    # GAMG specific
    nPreSweeps: Optional[int] = None
    nPostSweeps: Optional[int] = None
    cacheAgglomeration: Optional[Union[bool, str]] = None
    agglomerator: Optional[str] = None
    nCellsInCoarsestLevel: Optional[int] = None
    mergeLevels: Optional[int] = None

class FvSolution(OpenFOAMFile):
    foam_object: str = "fvSolution"

    solvers: Dict[str, Union[SolverSettings, Dict[str, Any]]] = Field(default_factory=dict)
    PIMPLE: Optional[Dict[str, Any]] = None
    SIMPLE: Optional[Dict[str, Any]] = None
    relaxationFactors: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def check_critical_solvers(self):
        if not self.solvers:
            return self
        
        solver_keys = self.solvers.keys()
        
        # Check for Pressure (p, p_rgh, or regex)
        has_pressure = any(k in solver_keys for k in ["p", "p_rgh", "p_rghFinal"]) or \
                       any("p" in k and "rgh" in k for k in solver_keys)
        
        # Check for Temperature/Energy (T, h, e, or regex)
        has_energy = any(k in solver_keys for k in ["T", "h", "e", "TFinal", "hFinal"]) or \
                     any(x in str(list(solver_keys)) for x in ["T", "h", "e"])

        return self

class Specie(BaseModel):
    molWeight: float

class Thermodynamics(BaseModel):
    Cp: Optional[float] = None
    Cv: Optional[float] = None
    Hf: float = 0

class Transport(BaseModel):
    mu: Optional[float] = None
    Pr: Optional[float] = None
    kappa: Optional[float] = None

class EquationOfState(BaseModel):
    rho: Optional[float] = None

class Mixture(BaseModel):
    specie: Specie
    thermodynamics: Thermodynamics
    transport: Transport
    equationOfState: Optional[EquationOfState] = None

class ThermoType(BaseModel):
    type: str
    mixture: str
    transport: str
    thermo: str
    equationOfState: str
    specie: str
    energy: str

class ThermophysicalProperties(OpenFOAMFile):
    foam_object: str = "thermophysicalProperties"
    
    thermoType: ThermoType
    mixture: Mixture

class VolField(BaseModel):
    """
    Represents a field file (scalar or vector) in the 0/ directory.
    """
    model_config = ConfigDict(extra='ignore')
    
    foam_class: str = "volScalarField"
    foam_object: str = "T" # Default
    
    dimensions: List[int]
    internalField: Union[float, List[float], List[List[float]], str] 
    
    boundaryField: Optional[Dict[str, Any]] = None

    def to_foam_string(self) -> str:
        lines = [
            "FoamFile", "{", 
            "    version     2.0;", 
            "    format      ascii;", 
            f"    class       {self.foam_class};", 
            f"    object      {self.foam_object};", 
            "}", ""
        ]
        
        dims = " ".join(str(x) for x in self.dimensions)
        lines.append(f"dimensions      [{dims}];")
        lines.append("")
        
        val = self.internalField
        if isinstance(val, (float, int)):
            lines.append(f"internalField   uniform {val};")
        elif isinstance(val, list) and len(val) == 3 and isinstance(val[0], (int, float)):
             vec = " ".join(str(x) for x in val)
             lines.append(f"internalField   uniform ({vec});")
        elif isinstance(val, list):
            lines.append(f"internalField   nonuniform List<{ 'scalar' if 'Scalar' in self.foam_class else 'vector' }> ")
            lines.append(f"{len(val)}")
            lines.append("(")
            for v in val:
                if isinstance(v, list):
                    lines.append(f"({ ' '.join(str(x) for x in v) })")
                else:
                    lines.append(str(v))
            lines.append(");")
        else:
             lines.append(f"internalField   {val};")
             
        lines.append("")

        lines.append("boundaryField")
        lines.append("{")
        if self.boundaryField:
            for patch, data in self.boundaryField.items():
                lines.append(f"    {patch}")
                lines.append("    {")
                for k, v in data.items():
                    lines.append(f"        {k:<16} {v};")
                lines.append("    }")
        lines.append("}")
        
        return "\n".join(lines)

class CHTSimulation:
    """
    Root object representing the entire case folder structure.
    """
    
    TYPE_MAP = {
        "ControlDict": ControlDict,
        "FvSchemes": FvSchemes,
        "FvSolution": FvSolution,
        "ThermophysicalProperties": ThermophysicalProperties,
        "VolField": VolField,
        "GenericFOAMFile": GenericFOAMFile,
        "FoamListFile": FoamListFile
    }
    
    CLASS_TO_TYPE = {
        ControlDict: "ControlDict",
        FvSchemes: "FvSchemes",
        FvSolution: "FvSolution",
        ThermophysicalProperties: "ThermophysicalProperties",
        VolField: "VolField",
        GenericFOAMFile: "GenericFOAMFile",
        FoamListFile: "FoamListFile"
    }

    def __init__(self, case_structure: Dict[str, Any] = None):
        self.case_structure = case_structure or {}

    @classmethod
    def load_from_json(cls, data: Dict[str, Any]) -> 'CHTSimulation':
        structure = cls._parse_recursive(data)
        return cls(structure)

    @classmethod
    def _parse_recursive(cls, item: Any) -> Any:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
            type_name, kwargs = item
            model_class = cls.TYPE_MAP.get(type_name)
            if model_class:
                try:
                    return model_class(**kwargs)
                except Exception as e:
                    print(f"Warning: strict validation failed for {type_name}, falling back to Generic. Error: {e}")
                    return GenericFOAMFile(**kwargs)
            else:
                return item
        
        elif isinstance(item, dict):
            return {k: cls._parse_recursive(v) for k, v in item.items()}
        
        elif isinstance(item, list):
             return [cls._parse_recursive(v) for v in item]

        else:
            return item

    def to_json(self) -> Dict[str, Any]:
        return self._serialize_recursive(self.case_structure)

    def _serialize_recursive(self, item: Any) -> Any:
        if isinstance(item, OpenFOAMFile) or isinstance(item, VolField):
            type_name = self.CLASS_TO_TYPE.get(type(item), "GenericFOAMFile")
            return [type_name, item.model_dump(exclude_none=True)]
        
        elif isinstance(item, dict):
            return {k: self._serialize_recursive(v) for k, v in item.items()}
            
        else:
            return item

    def get_file(self, path: str) -> Optional[Any]:
        parts = path.strip('/').split('/')
        current = self.case_structure
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validates the simulation case parameters against physical and OpenFOAM constraints.
        Returns (True, []) if all checks pass.
        Returns (False, [errors]) otherwise.
        Prints all errors found.
        """
        errors = []

        control_dicts: List[Tuple[str, ControlDict]] = []
        vol_fields: List[Tuple[str, VolField]] = []
        thermo_props: List[Tuple[str, ThermophysicalProperties]] = []

        def traverse(item: Any, path: str = ""):
            if isinstance(item, ControlDict):
                control_dicts.append((path, item))
            elif isinstance(item, VolField):
                vol_fields.append((path, item))
            elif isinstance(item, ThermophysicalProperties):
                thermo_props.append((path, item))
            elif isinstance(item, dict):
                for k, v in item.items():
                    traverse(v, f"{path}/{k}" if path else k)
            elif isinstance(item, list):
                pass

        traverse(self.case_structure)

        constant_dir = self.case_structure.get("constant")
        if not constant_dir or not isinstance(constant_dir, dict):
            errors.append("Missing 'constant' directory.")
        else:
            if "polyMesh" not in constant_dir:
                errors.append("Missing 'constant/polyMesh' directory. Mesh definition is required.")

        for path, cd in control_dicts:
            if cd.deltaT <= 0:
                errors.append(f"[{path}] deltaT must be > 0 (found {cd.deltaT})")
            if cd.endTime <= cd.startTime:
                errors.append(f"[{path}] endTime ({cd.endTime}) must be > startTime ({cd.startTime})")
            if cd.writeInterval <= 0:
                errors.append(f"[{path}] writeInterval must be > 0 (found {cd.writeInterval})")
            if cd.maxCo is not None and cd.maxCo <= 0:
                errors.append(f"[{path}] maxCo must be > 0 (found {cd.maxCo})")

        for path, vf in vol_fields:
            name = vf.foam_object
            val = vf.internalField

            if name == "T":
                if isinstance(val, (int, float)):
                    if val < 0:
                        errors.append(f"[{path}] Temperature (T) internalField must be >= 0 K (found {val})")
                elif isinstance(val, list):
                    try:
                        flat_vals = []
                        for x in val:
                            if isinstance(x, list): flat_vals.extend(x)
                            else: flat_vals.append(x)
                        
                        min_t = min(flat_vals)
                        if min_t < 0:
                            errors.append(f"[{path}] Temperature (T) field contains values < 0 K (min: {min_t})")
                    except Exception:
                        pass

            if name == "p":
                if isinstance(val, (int, float)):
                    if val < 0:
                        errors.append(f"[{path}] Pressure (p) internalField must be >= 0 Pa (found {val})")

        for path, tp in thermo_props:
            mix = tp.mixture
            
            if mix.specie.molWeight <= 0:
                errors.append(f"[{path}] molWeight must be > 0 (found {mix.specie.molWeight})")
            
            if mix.thermodynamics.Cp is not None and mix.thermodynamics.Cp <= 0:
                errors.append(f"[{path}] Specific heat (Cp) must be > 0 (found {mix.thermodynamics.Cp})")
            if mix.thermodynamics.Cv is not None and mix.thermodynamics.Cv <= 0:
                errors.append(f"[{path}] Specific heat (Cv) must be > 0 (found {mix.thermodynamics.Cv})")
                
            if mix.transport.mu is not None and mix.transport.mu <= 0:
                errors.append(f"[{path}] Dynamic viscosity (mu) must be > 0 (found {mix.transport.mu})")
            if mix.transport.Pr is not None and mix.transport.Pr <= 0:
                errors.append(f"[{path}] Prandtl number (Pr) must be > 0 (found {mix.transport.Pr})")
            if mix.transport.kappa is not None and mix.transport.kappa <= 0:
                errors.append(f"[{path}] Thermal conductivity (kappa) must be > 0 (found {mix.transport.kappa})")
                
            if mix.equationOfState and mix.equationOfState.rho is not None:
                if mix.equationOfState.rho <= 0:
                     errors.append(f"[{path}] Density (rho) must be > 0 (found {mix.equationOfState.rho})")

        if errors:
            print("\n=== Validation Errors Detected ===")
            for e in errors:
                print(f"  - {e}")
            print("==================================")
            return False, errors
            
        print("Validation Passed: Physical parameters are within reasonable bounds.")
        return True, []
