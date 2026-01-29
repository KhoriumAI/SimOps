import os
import json
import typing
from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from google import genai
from cht_sim import (
    CHTSimulation,
    ControlDict,
    FvSchemes,
    FvSolution,
    ThermophysicalProperties,
    VolField,
    GenericFOAMFile,
    FoamListFile,
    SolverSettings
)
from verify_integrity import reconstruct_case
from pathlib import Path
import shutil
import sys

sys.path.append(os.path.join(os.getcwd(), "openfoam_validator"))
from openfoam_validator.validator import OpenFOAMValidator

if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not set. Please set it before running generation.")


class KeyValueStr(BaseModel):
    key: str
    value: str

class KeyValueAny(BaseModel):
    key: str
    value: Union[str, float, int, bool, List[Union[str, float, int, bool]]]

# --- 1. ControlDict ---
class GeminiControlDict(BaseModel):
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
    
    functions: Optional[List[KeyValueAny]] = None

# --- 2. FvSchemes ---
class GeminiFvSchemes(BaseModel):
    foam_object: str = "fvSchemes"
    ddtSchemes: List[KeyValueStr] = Field(default_factory=list)
    gradSchemes: List[KeyValueStr] = Field(default_factory=list)
    divSchemes: List[KeyValueStr] = Field(default_factory=list)
    laplacianSchemes: List[KeyValueStr] = Field(default_factory=list)
    interpolationSchemes: List[KeyValueStr] = Field(default_factory=list)
    snGradSchemes: List[KeyValueStr] = Field(default_factory=list)
    
    fluxRequired: Optional[List[KeyValueAny]] = None
    wallDist: Optional[List[KeyValueAny]] = None

# --- 3. FvSolution ---
class GeminiSolverEntry(BaseModel):
    name: str # The key (e.g. "p", "U")
    settings: Union[SolverSettings, List[KeyValueAny]]

class GeminiFvSolution(BaseModel):
    foam_object: str = "fvSolution"
    solvers: List[GeminiSolverEntry] = Field(default_factory=list)
    PIMPLE: Optional[List[KeyValueAny]] = None
    SIMPLE: Optional[List[KeyValueAny]] = None
    relaxationFactors: Optional[List[KeyValueAny]] = None

# --- 4. ThermophysicalProperties ---
class GeminiSpecie(BaseModel):
    molWeight: float

class GeminiThermodynamics(BaseModel):
    Cp: Optional[float] = None
    Cv: Optional[float] = None
    Hf: float = 0

class GeminiTransport(BaseModel):
    mu: Optional[float] = None
    Pr: Optional[float] = None
    kappa: Optional[float] = None

class GeminiEquationOfState(BaseModel):
    rho: Optional[float] = None

class GeminiMixture(BaseModel):
    specie: GeminiSpecie
    thermodynamics: GeminiThermodynamics
    transport: GeminiTransport
    equationOfState: Optional[GeminiEquationOfState] = None

class GeminiThermoType(BaseModel):
    type: str
    mixture: str
    transport: str
    thermo: str
    equationOfState: str
    specie: str
    energy: str

class GeminiThermophysicalProperties(BaseModel):
    foam_object: str = "thermophysicalProperties"
    thermoType: GeminiThermoType
    mixture: GeminiMixture

# --- 5. VolField ---
class GeminiBoundaryEntry(BaseModel):
    patchName: str
    properties: List[KeyValueStr]

class GeminiVolField(BaseModel):
    foam_class: str = "volScalarField"
    foam_object: str = "T"
    dimensions: List[int]
    internalField: Union[float, List[float], List[List[float]], str]
    
    boundaryField: Optional[List[GeminiBoundaryEntry]] = None

# --- 6. Generic/List ---
class GeminiGenericFOAMFile(BaseModel):
    foam_class: str = "dictionary"
    foam_location: Optional[str] = None
    foam_object: str = "unknown"
    data: List[KeyValueAny] = Field(default_factory=list)

class GeminiBlockEntry(BaseModel):
    name: str
    properties: List[KeyValueStr]

class GeminiFoamListFile(BaseModel):
    foam_object: str = "unknown"
    foam_class: str = "labelList"
    foam_location: Optional[str] = None
    size: int
    content: List[Union[str, int, float, List[int], List[float], GeminiBlockEntry]] 

# --- FileEntry Wrapper ---
class GeminiFileEntry(BaseModel):
    path: str = Field(..., description="The relative path to the file (e.g., 'system/controlDict', '0/T')")
    content: Union[
        GeminiControlDict,
        GeminiFvSchemes,
        GeminiFvSolution,
        GeminiThermophysicalProperties,
        GeminiVolField,
        GeminiFoamListFile,
        GeminiGenericFOAMFile
    ]

class GeminiCaseResponse(BaseModel):
    files: List[GeminiFileEntry]

def _list_to_dict(kv_list: List[Any]) -> Dict[str, Any]:
    if not kv_list: return {}
    res = {}
    for item in kv_list:
        if hasattr(item, 'key') and hasattr(item, 'value'):
            res[item.key] = item.value
        elif hasattr(item, 'name') and hasattr(item, 'settings'): # SolverEntry
            val = item.settings
            if hasattr(val, 'model_dump'):
                val = val.model_dump(exclude_none=True)
            res[item.name] = val
        elif hasattr(item, 'patchName') and hasattr(item, 'properties'): # BoundaryEntry
            res[item.patchName] = _list_to_dict(item.properties)
        elif hasattr(item, 'name') and hasattr(item, 'properties'): # BlockEntry
            res[item.name] = _list_to_dict(item.properties)
    return res

def convert_gemini_to_cht(response: GeminiCaseResponse) -> Dict[str, Any]:
    case_structure = {}

    for entry in response.files:
        parts = entry.path.strip("/").split("/")
        current_level = case_structure
        
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
            if not isinstance(current_level, dict):
                current_level = {} 
        
        filename = parts[-1]
        
        gem_obj = entry.content
        cht_obj = None
        type_name = "GenericFOAMFile"
        
        if isinstance(gem_obj, GeminiControlDict):
            type_name = "ControlDict"
            d = gem_obj.model_dump(exclude={'functions'})
            if gem_obj.functions:
                d['functions'] = _list_to_dict(gem_obj.functions)
            cht_obj = d
            
        elif isinstance(gem_obj, GeminiFvSchemes):
            type_name = "FvSchemes"
            d = gem_obj.model_dump(exclude={'ddtSchemes', 'gradSchemes', 'divSchemes', 'laplacianSchemes', 'interpolationSchemes', 'snGradSchemes', 'fluxRequired', 'wallDist'})
            d['ddtSchemes'] = _list_to_dict(gem_obj.ddtSchemes)
            d['gradSchemes'] = _list_to_dict(gem_obj.gradSchemes)
            d['divSchemes'] = _list_to_dict(gem_obj.divSchemes)
            d['laplacianSchemes'] = _list_to_dict(gem_obj.laplacianSchemes)
            d['interpolationSchemes'] = _list_to_dict(gem_obj.interpolationSchemes)
            d['snGradSchemes'] = _list_to_dict(gem_obj.snGradSchemes)
            if gem_obj.fluxRequired: d['fluxRequired'] = _list_to_dict(gem_obj.fluxRequired)
            if gem_obj.wallDist: d['wallDist'] = _list_to_dict(gem_obj.wallDist)
            cht_obj = d
            
        elif isinstance(gem_obj, GeminiFvSolution):
            type_name = "FvSolution"
            d = gem_obj.model_dump(exclude={'solvers', 'PIMPLE', 'SIMPLE', 'relaxationFactors'})
            d['solvers'] = _list_to_dict(gem_obj.solvers)
            if gem_obj.PIMPLE: d['PIMPLE'] = _list_to_dict(gem_obj.PIMPLE)
            if gem_obj.SIMPLE: d['SIMPLE'] = _list_to_dict(gem_obj.SIMPLE)
            if gem_obj.relaxationFactors: d['relaxationFactors'] = _list_to_dict(gem_obj.relaxationFactors)
            cht_obj = d
            
        elif isinstance(gem_obj, GeminiThermophysicalProperties):
            type_name = "ThermophysicalProperties"
            cht_obj = gem_obj.model_dump(exclude_none=True)
            
        elif isinstance(gem_obj, GeminiVolField):
            type_name = "VolField"
            d = gem_obj.model_dump(exclude={'boundaryField'})
            if gem_obj.boundaryField:
                d['boundaryField'] = _list_to_dict(gem_obj.boundaryField)
            cht_obj = d
            
        elif isinstance(gem_obj, GeminiFoamListFile):
            type_name = "FoamListFile"
            d = gem_obj.model_dump(exclude={'content'}, exclude_none=True)
            content = []
            for item in gem_obj.content:
                if isinstance(item, GeminiBlockEntry):
                    content.append(_list_to_dict([item]))
                else:
                    content.append(item)
            d['content'] = content
            cht_obj = d
            
        elif isinstance(gem_obj, GeminiGenericFOAMFile):
            type_name = "GenericFOAMFile"
            d = gem_obj.model_dump(exclude={'data'})
            if gem_obj.data:
                d.update(_list_to_dict(gem_obj.data))
            cht_obj = d

        current_level[filename] = [type_name, cht_obj]

    return case_structure

def clean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively removes 'additionalProperties' key from JSON schema dictionary.
    """
    if not isinstance(schema, dict):
        return schema
    new_schema = schema.copy()
    if "additionalProperties" in new_schema:
        del new_schema["additionalProperties"]
    if "properties" in new_schema:
        for k, v in new_schema["properties"].items():
            new_schema["properties"][k] = clean_schema(v)
    if "$defs" in new_schema:
        for k, v in new_schema["$defs"].items():
            new_schema["$defs"][k] = clean_schema(v)
    if "definitions" in new_schema:
        for k, v in new_schema["definitions"].items():
            new_schema["definitions"][k] = clean_schema(v)
    if "items" in new_schema:
        new_schema["items"] = clean_schema(new_schema["items"])
    for comb in ["anyOf", "allOf", "oneOf"]:
        if comb in new_schema:
            new_schema[comb] = [clean_schema(item) for item in new_schema[comb]]
    return new_schema

class ConfigGenerator:
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview", max_retries: int = 3):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.system_prompt = """
        You are an expert OpenFOAM computational fluid dynamics (CFD) engineer.
        Your task is to generate complete, valid OpenFOAM case structures based on user requests.
        
        Requirements for output case:
        1. Structure should have 'system', 'constant', and '0' directories.
        2. 'system' should contain 'controlDict', 'fvSchemes', 'fvSolution'.
        3. 'constant' should contain 'thermophysicalProperties' for air and 'polyMesh' folder.
           - 'polyMesh' MUST contain valid 'points', 'faces', 'owner', 'neighbour', and 'boundary' files to be a valid mesh.
        4. '0' should contain 'T' (Temperature) initialized to 300K and 'fluid' folder with 'p_rgh' and 'U' as vector fields.
        
        Domain Knowledge:
        - CHT (Conjugate Heat Transfer) cases involve multiple regions (fluid/solid).
        - 'system' directory contains control dictionaries (controlDict, fvSchemes, fvSolution).
        - 'constant' directory contains mesh (polyMesh) and physical properties (thermophysicalProperties, turbulenceProperties).
        - '0' directory contains initial conditions for fields (T, p_rgh, U, etc.).
        - Ensure physical consistency: endTime > startTime, deltaT > 0.
        - Mesh files (points, faces, owner, neighbour, boundary) are mandatory for checkMesh.
        - 'boundary' file MUST have class 'polyBoundaryMesh' and content as a list of patch definitions (using Key-Value blocks).
        - Use 'foamMultiRun' or 'chtMultiRegionFoam' as application in controlDict for CHT.
        """

    def generate(self, user_request: str) -> Optional[CHTSimulation]:
        current_prompt = user_request
        
        raw_schema = GeminiCaseResponse.model_json_schema()
        cleaned_schema = clean_schema(raw_schema)
        
        config = {
            "response_mime_type": "application/json",
            "response_schema": cleaned_schema,
            "system_instruction": self.system_prompt
        }

        for attempt in range(self.max_retries):
            print(f"Attempt {attempt + 1}/{self.max_retries}...")
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=current_prompt,
                    config=config
                )
                
                gemini_case = None
                try:
                    if hasattr(response, 'parsed') and response.parsed:
                        if isinstance(response.parsed, dict):
                            gemini_case = GeminiCaseResponse(**response.parsed)
                        else:
                            gemini_case = response.parsed
                    elif response.text:
                        json_data = json.loads(response.text)
                        gemini_case = GeminiCaseResponse(**json_data)
                    else:
                        raise ValueError("Empty response")
                except Exception as e:
                    print(f"Failed to parse JSON: {e}")
                    continue

                structure = convert_gemini_to_cht(gemini_case)
                sim = CHTSimulation.load_from_json(structure)
                
                is_valid, errors = sim.validate()
                
                if is_valid:
                    print("Validation successful!")
                    return sim
                else:
                    print(f"Validation failed with {len(errors)} errors:")
                    for err in errors:
                        print(f"  - {err}")
                    
                    feedback = "\nThe generated configuration failed validation with the following errors:\n"
                    for err in errors:
                        feedback += f"- {err}\n"
                    feedback += "\nPlease correct these errors and output the valid JSON again."
                    current_prompt += feedback
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Generation error: {e}")
                
        print("Max retries reached. Generation failed.")
        return None

if __name__ == "__main__":
    user_request = """
    Generate a simple OpenFOAM CHT (Conjugate Heat Transfer) simulation case.
    
    Specifics:
    - controlDict: run from 0 to 10s, deltaT 0.01.
    """
    
    print("--- Starting ConfigGenerator ---")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set.")
        exit(1)

    generator = ConfigGenerator(api_key=os.environ["GOOGLE_API_KEY"])
    sim = generator.generate(user_request)
    
    if sim:
        print(f"\nGenerated Valid Simulation.")
        
        output_path = "gemini_generated.json"
        print(f"Saving to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(sim.to_json(), f, indent=2)
            
        print("\nDone. JSON saved.")

        reconstruct_dir = Path("_generated") / "gemini_case"
        if reconstruct_dir.exists():
            shutil.rmtree(reconstruct_dir)
        
        print(f"\n--- Reconstructing OpenFOAM Case in {reconstruct_dir} ---")
        reconstruct_case(sim.to_json(), reconstruct_dir)
        
        print(f"\n--- Running OpenFOAM Validator on Generated Case ---")
        validator = OpenFOAMValidator(str(reconstruct_dir), openfoam_path="/Volumes/OpenFOAM-v2512")
        results = validator.validate()
        
        print("\n=== Validation Results ===")
        print(f"Structure Check:   {results['structure']}")
        print(f"Environment Check: {results['environment']}")
        print(f"CheckMesh Status:  {results.get('checkMesh', 'N/A')}")
        
        if results['errors']:
            print("\n[ERRORS]:")
            for e in results['errors']:
                print(f"  - {e}")
        
        if results['structure'] == 'PASS' and results['checkMesh'] == 'PASS':
             print("\n>>> SUCCESS: Generated case is valid and runnable! <<<")
        else:
             print("\n>>> FAILURE: Generated case has issues. <<<")
    else:
        print("Failed to generate valid case.")
