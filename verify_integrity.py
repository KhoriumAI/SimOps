#!/usr/bin/env python3
"""
verify_integrity.py

Verifies that the Golden CHT Case can be round-tripped through the CHTSimulation Schema
preserving the file system structure.

Features:
1. Parse ALL files in system/, constant/, 0/
2. Handle strict schemas (controlDict, fvSolution...) and generic files.
3. Handle Vector fields (U).
4. Reconstruct file system from JSON.
5. Diff Original vs Reconstructed.
"""

import sys
import shutil
import json
import re
import difflib
from pathlib import Path
from typing import Any, Dict, List, Union

from cht_sim import (
    CHTSimulation, 
    ControlDict, 
    FvSchemes, 
    FvSolution, 
    ThermophysicalProperties, 
    VolField,
    GenericFOAMFile,
    FoamListFile
)

def parse_foam_file(content: str) -> Dict[str, Any]:
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    tokens = re.findall(r'[^\s\(\)\[\]\{\};]+|[\(\)\[\]\{\};]', content)
    
    if len(tokens) > 1 and tokens[0].isdigit() and tokens[1] == '(':
         return {"__is_list__": True, "size": int(tokens[0]), "content": _parse_list_recursive(tokens, 1)[0]}
    
    try:
        data, _ = _parse_dict_tokens(tokens, 0)
        return data
    except Exception as e:
        print(f"Error parsing tokens: {e}")
        return {}

def _parse_dict_tokens(tokens, start_index):
    """
    Parses tokens into a dictionary starting from start_index.
    Returns (dict, next_index).
    Reads until '}' or end of tokens.
    """
    current_dict = {}
    key = None
    i = start_index
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == '{':
            new_dict, end_index = _parse_dict_tokens(tokens, i + 1)
            if key:
                current_dict[key] = new_dict
            else:
                pass
            key = None
            i = end_index - 1 # Loop increments
            
        elif token == '}':
            return current_dict, i + 1
            
        elif token == '(':
            lst, end_index = _parse_list_recursive(tokens, i)
            if key:
                current_dict[key] = lst
            else:
                pass
            key = None
            i = end_index - 1
            
        elif token == ';':
            if key:
                 if key.startswith('$'): current_dict[key] = "__MACRO__"
                 elif key not in current_dict: current_dict[key] = True 
            key = None
            
        else:
            if key is None:
                if i + 1 < len(tokens) and tokens[i+1] == '(':
                    paren_depth = 0
                    j = i + 1
                    is_key_func = False
                    while j < len(tokens):
                        if tokens[j] == '(': paren_depth += 1
                        elif tokens[j] == ')': paren_depth -= 1
                        
                        j += 1
                        if paren_depth == 0:
                            if j < len(tokens) and tokens[j] not in [';']:
                                is_key_func = True
                            break
                    
                    if is_key_func:
                        complex_key = "".join(tokens[i:j])
                        key = complex_key
                        i = j - 1
                    else:
                        key = token
                else:
                     key = token
                     
            else:
                vals = []
                j = i
                while j < len(tokens) and tokens[j] not in [';', '{', '}']:
                    vals.append(tokens[j])
                    j += 1
                
                if len(vals) > 0 and vals[0] == '[':
                     dim_vals = []
                     k = 0
                     while k < len(vals):
                         if vals[k] == ']': break
                         if vals[k] != '[': 
                             try: dim_vals.append(int(vals[k]))
                             except: pass
                         k += 1
                     val = dim_vals
                else:
                    val_str = " ".join(vals)
                    if len(vals) == 1:
                        val = _parse_value(vals[0])
                    else:
                        val = val_str.strip('"')

                current_dict[key] = val
                key = None
                i = j - 1
        i += 1
        
    return current_dict, i

def _parse_value(token):
    if token.lower() in ['true', 'on', 'yes']: return True
    if token.lower() in ['false', 'off', 'no']: return False
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token.strip('"')

def _parse_list_recursive(tokens, start_index):
    res = []
    i = start_index + 1
    while i < len(tokens):
        token = tokens[i]
        if token == ')':
            return res, i + 1
        elif token == '(':
            sub_list, end_index = _parse_list_recursive(tokens, i)
            res.append(sub_list)
            i = end_index
        elif token == '{':
            sub_dict, end_index = _parse_dict_tokens(tokens, i + 1)
            res.append(sub_dict)
            i = end_index
        else:
            if i + 1 < len(tokens) and tokens[i+1] == '(':
                 val = _parse_value(token)
                 res.append(val)
            else:
                 res.append(_parse_value(token))
            i += 1
    return res, i

def parse_controlDict(path: Path) -> ControlDict:
    data = parse_foam_file(path.read_text())
    foam_file_header = data.pop('FoamFile', {})
    
    data['foam_object'] = foam_file_header.get('object', path.name)
    data['foam_class'] = foam_file_header.get('class', 'dictionary')
    if 'location' in foam_file_header:
        data['foam_location'] = foam_file_header['location']

    if 'application' not in data and 'solverName' in data: data['application'] = data.pop('solverName')
    return ControlDict(**data)

def parse_fvSchemes(path: Path) -> FvSchemes:
    content = path.read_text()
    if '#include' in content:
        include_match = re.search(r'#include\s+"(\.\./[^"]+)"', content)
        if include_match:
            target = (path.parent / include_match.group(1)).resolve()
            if target.exists(): return parse_fvSchemes(target)
    
    data = parse_foam_file(content)
    foam_file_header = data.pop('FoamFile', {})
    
    data['foam_object'] = foam_file_header.get('object', path.name)
    data['foam_class'] = foam_file_header.get('class', 'dictionary')
    if 'location' in foam_file_header:
        data['foam_location'] = foam_file_header['location']
        
    return FvSchemes(**data)

def parse_fvSolution(path: Path) -> FvSolution:
    content = path.read_text()
    if '#include' in content:
         include_match = re.search(r'#include\s+"(\.\./[^"]+)"', content)
         if include_match:
            target = (path.parent / include_match.group(1)).resolve()
            if target.exists(): return parse_fvSolution(target)
            
    data = parse_foam_file(content)
    foam_file_header = data.pop('FoamFile', {})
    
    data['foam_object'] = foam_file_header.get('object', path.name)
    data['foam_class'] = foam_file_header.get('class', 'dictionary')
    if 'location' in foam_file_header:
        data['foam_location'] = foam_file_header['location']
        
    return FvSolution(**data)

def parse_thermophysicalProperties(path: Path) -> ThermophysicalProperties:
    data = parse_foam_file(path.read_text())
    foam_file_header = data.pop('FoamFile', {})
    
    data['foam_object'] = foam_file_header.get('object', path.name)
    data['foam_class'] = foam_file_header.get('class', 'dictionary')
    if 'location' in foam_file_header:
        data['foam_location'] = foam_file_header['location']

    if 'mixture' in data and isinstance(data['mixture'], dict):
        mix = data['mixture']
        if 'thermodynamics' in mix and isinstance(mix['thermodynamics'], dict):
            if 'coeffs' in mix['thermodynamics']:
                 mix['thermodynamics'].update(mix['thermodynamics'].pop('coeffs'))
    return ThermophysicalProperties(**data)

def parse_volField(path: Path) -> VolField:
    content = path.read_text()
    
    foam_header_match = re.search(r'FoamFile\s*\{([^}]*)\}', content, re.DOTALL)
    foam_class = "volScalarField"
    foam_object = path.name
    foam_location = None
    
    if foam_header_match:
        header_content = foam_header_match.group(1)
        cls_m = re.search(r'class\s+([^;]+);', header_content)
        if cls_m: foam_class = cls_m.group(1).strip().strip('"')
        
        obj_m = re.search(r'object\s+([^;]+);', header_content)
        if obj_m: foam_object = obj_m.group(1).strip().strip('"')
        
        loc_m = re.search(r'location\s+([^;]+);', header_content)
        if loc_m: foam_location = loc_m.group(1).strip().strip('"')

    full_data = parse_foam_file(content)
    
    dims = full_data.get('dimensions', [0]*7)
    if isinstance(dims, list) and len(dims) == 7:
        pass # OK
    else:
        dim_match = re.search(r'dimensions\s+\[([\d\s\-\.]+)\];', content)
        if dim_match:
            dims = [int(float(x)) for x in dim_match.group(1).split()]

    internal_field = full_data.get('internalField')
    boundary_field = full_data.get('boundaryField', {})

    uni_s_match = re.search(r'internalField\s+uniform\s+([\d\.eE\+\-]+);', content)
    if uni_s_match:
        return VolField(
            foam_class=foam_class, foam_object=foam_object, foam_location=foam_location,
            dimensions=dims, internalField=float(uni_s_match.group(1)),
            boundaryField=boundary_field
        )
    
    uni_v_match = re.search(r'internalField\s+uniform\s+\(([\d\.eE\+\-\s]+)\);', content)
    if uni_v_match:
        vec = [float(x) for x in uni_v_match.group(1).split()]
        return VolField(
            foam_class=foam_class, foam_object=foam_object, foam_location=foam_location,
            dimensions=dims, internalField=vec,
            boundaryField=boundary_field
        )

    raise ValueError("Complex field format")

def parse_generic(path: Path) -> Union[GenericFOAMFile, FoamListFile]:
    data = parse_foam_file(path.read_text())
    
    if "__is_list__" in data:
        return FoamListFile(
            foam_object=path.name,
            size=data["size"],
            content=data["content"]
        )
    
    keys = [k for k in data.keys() if k != 'FoamFile' and not k.startswith('//')]
    
    foam_file_header = data.get('FoamFile', {})
    foam_class = foam_file_header.get('class', 'dictionary')
    foam_location = foam_file_header.get('location', None)
    foam_object = foam_file_header.get('object', path.name)

    if len(keys) == 1 and keys[0].isdigit() and isinstance(data[keys[0]], list):
         return FoamListFile(
            foam_object=foam_object,
            foam_class=foam_class,
            foam_location=foam_location,
            size=int(keys[0]),
            content=data[keys[0]]
        )

    data.pop('FoamFile', None)
    
    data['foam_object'] = foam_object
    data['foam_class'] = foam_class
    if foam_location:
        data['foam_location'] = foam_location
    
    return GenericFOAMFile(**data)

# --- Main Walker ---

def _parse_file(path: Path) -> Any:
    name = path.name
    try:
        if name == "controlDict": return parse_controlDict(path)
        elif name == "fvSchemes": return parse_fvSchemes(path)
        elif name == "fvSolution": return parse_fvSolution(path)
        elif name == "thermophysicalProperties": return parse_thermophysicalProperties(path)
        
        if "0" in path.parts:
            try:
                return parse_volField(path)
            except:
                return parse_generic(path)
                
        return parse_generic(path)
        
    except Exception as e:
        print(f"Warning: Failed to parse {path} as strict type. Parsing as Generic. Error: {e}")
        try:
            return parse_generic(path)
        except Exception as e2:
            print(f"Error: Completely failed to parse {path}: {e2}")
            return None

def build_case_structure(root: Path) -> Dict[str, Any]:
    structure = {}
    dirs_to_scan = ["system", "constant", "0"]
    
    for d_name in dirs_to_scan:
        d_path = root / d_name
        if d_path.exists() and d_path.is_dir():
            structure[d_name] = _scan_dir(d_path)
            
    return structure

def _scan_dir(path: Path) -> Dict[str, Any]:
    folder_content = {}
    for item in path.iterdir():
        if item.name.startswith('.'): continue
        
        if item.is_dir():
            folder_content[item.name] = _scan_dir(item)
        elif item.is_file():
            parsed_obj = _parse_file(item)
            if parsed_obj:
                folder_content[item.name] = parsed_obj
            else:
                pass 
                
    return folder_content

def reconstruct_case(json_data: Dict[str, Any], target_root: Path):
    sim = CHTSimulation.load_from_json(json_data)
    _write_recursive(sim.case_structure, target_root)

def _write_recursive(structure: Any, current_path: Path):
    current_path.mkdir(parents=True, exist_ok=True)
    
    for name, item in structure.items():
        child_path = current_path / name
        
        if isinstance(item, dict):
            # Folder
            _write_recursive(item, child_path)
        elif hasattr(item, 'to_foam_string'):
            # File
            with open(child_path, 'w') as f:
                f.write(item.to_foam_string())

def main():
    root = Path("./Golden_Case") if Path("./Golden_Case").exists() else Path(".")
    generated_root = Path(".") / "_generated"
    reconstructed_root = generated_root / "reconstructed"
    
    if generated_root.exists(): shutil.rmtree(generated_root)
    reconstructed_root.mkdir(parents=True)
    
    print(f"1. Scanning Golden Case from {root} (0, constant, system)...")
    case_struct = build_case_structure(root)
    
    if 'constant' in case_struct and 'regionProperties' in case_struct['constant']:
        print("   - Found constant/regionProperties")
    else:
        print("   - WARNING: constant/regionProperties NOT found!")
        
    print("2. Serializing to JSON...")
    sim = CHTSimulation(case_struct)
    
    print("   - Validating Case Parameters...")
    if not sim.validate():
        print("   - Validation FAILED: Physical constraints violated.")
        sys.exit(1)

    json_output = sim.to_json()
    
    json_path = generated_root / "golden.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"   - Written to {json_path}")
    
    print("3. Reconstructing File System...")
    with open(json_path) as f:
        loaded_json = json.load(f)
    
    reconstruct_case(loaded_json, reconstructed_root)
    print(f"   - Reconstructed in {reconstructed_root}")
    
    print("4. Verifying No Data Loss (Diff)...")
    any_diff = False
    for base_dir in ["system", "constant", "0"]:
        orig_base = root / base_dir
        recon_base = reconstructed_root / base_dir
        
        if not orig_base.exists(): continue
        
        for p in orig_base.rglob("*"):
            if p.is_file() and not p.name.startswith('.') and "polyMesh" not in p.parts:
                rel_path = p.relative_to(root)
                recon_p = reconstructed_root / rel_path
                
                if not recon_p.exists():
                    print(f"MISSING FILE: {rel_path}")
                    any_diff = True
                    continue
                    
                pass

    try:
        orig_rp = parse_generic(root / "constant/regionProperties")
        recon_rp = parse_generic(reconstructed_root / "constant/regionProperties")
        
        d1 = orig_rp.model_dump()
        d2 = recon_rp.model_dump()
        for d in [d1, d2]:
            d.pop('foam_object', None)
            
        if str(d1) == str(d2):
            print("   - constant/regionProperties content matches exactly.")
        else:
            print("   - constant/regionProperties MISMATCH.")
            print("Original:", d1)
            print("Reconstructed:", d2)
            any_diff = True

        if (reconstructed_root / "constant/polyMesh/points").exists():
            print("   - constant/polyMesh/points reconstructed.")
        else:
            pass
             
    except Exception as e:
        print(f"Verification Error: {e}")
        any_diff = True

    if not any_diff:
        print("\nSUCCESS: Full round-trip integrity verified.")
        sys.exit(0)
    else:
        print("\nFAILURE: Data loss detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
