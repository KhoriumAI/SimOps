# Paintbrush Debugging Guide

## Test Results

**Backend Pipeline: [OK] ALL TESTS PASSED**
- Config file passing: Working
- Backend injection: Working  
- Gmsh field creation: Working
- Mesh refinement: Working (5x finer elements in painted regions)

## The Problem

The backend works perfectly, but the GUI isn't collecting or passing painted regions to the worker.

## Debug Steps for GUI

### Step 1: Check if painting is being captured

Add this debug print to `gui_final.py` in the `on_paint_at_cursor` method (search for "def on_paint_at_cursor"):

```python
def on_paint_at_cursor(self, x, y):
    """Handle painting at cursor position"""
    print(f"[DEBUG-PAINT] on_paint_at_cursor called at ({x}, {y})")
    
    if not hasattr(self, 'viewer') or not self.viewer:
        print("[DEBUG-PAINT] No viewer!")
        return
    
    # ... rest of method
```

### Step 2: Check if painted regions are being stored

Search for where `painted_regions` list is created/updated. It should be in `ModernMeshGenGUI` class.
Look for something like:
```python
self.painted_regions = []
```

Add debug print when regions are added:
```python
self.painted_regions.append(region_data)
print(f"[DEBUG-PAINT] Added region, total count: {len(self.painted_regions)}")
print(f"[DEBUG-PAINT] Region data: {region_data}")
```

### Step 3: Check if regions are passed to worker

In `start_mesh_generation` method, add:
```python
def start_mesh_generation(self):
    # ... existing code ...
    
    quality_params = {
        'quality_preset': self.quality_preset_combo.currentText(),
        # ... other params ...
    }
    
    # ADD THIS:
    if hasattr(self, 'painted_regions') and self.painted_regions:
        quality_params['painted_regions'] = self.painted_regions
        print(f"[DEBUG-PAINT] Passing {len(self.painted_regions)} painted regions to worker")
    else:
        print(f"[DEBUG-PAINT] NO painted regions to pass!")
    
    print(f"[DEBUG-PAINT] quality_params keys: {list(quality_params.keys())}")
```

### Step 4: Verify in worker log

When mesh generation runs, check the log for:
```
[DEBUG] Injected X painted regions into config
Applying paintbrush refinement to X regions...
[OK] Applied refinement fields for X regions
```

## Quick Fix Locations

If painted regions aren't being collected, check these files:
1. `apps/desktop/gui_final.py` - `ModernMeshGenGUI` class
2. `apps/desktop/paintbrush_widget.py` - If it exists
3. `core/paintbrush_geometry.py` - If it exists

The GUI needs to:
1. Capture mouse clicks when in paint mode
2. Store painted region data (center + radius)
3. Pass that data in `quality_params` when starting mesh generation
