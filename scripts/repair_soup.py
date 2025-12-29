import pyvista as pv
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUP_FILE = os.path.join(PROJECT_ROOT, "robust_soup.stl")
REPAIRED_FILE = os.path.join(PROJECT_ROOT, "repaired_soup.stl")

def log(msg):
    print(msg)

def main():
    log("--- REPAIRING SOUP ---")
    mesh = pv.read(SOUP_FILE)
    log(f"Original Open Edges: {mesh.n_open_edges}")
    
    if mesh.n_open_edges > 0:
        log("Attempting to fill holes...")
        # Fill holes with size limit to avoid closing valid large gaps (though soup shouldn't have any large gaps?)
        # Actually, separate bodies shouldn't have gaps.
        # But if we blindly fill, we might connect Body A to Body B?
        # NO. We should process each connected component separately.
        
        # Split bodies
        bodies_ug = mesh.split_bodies()
        bodies = [b.extract_surface() for b in bodies_ug]
        log(f"Split into {len(bodies)} bodies for repair.")
        
        repaired_blocks = pv.MultiBlock()
        total_vol = 0.0
        
        for i, body in enumerate(bodies):
            # Repair body
            if body.n_open_edges > 0:
                # Try fill
                fixed = body.fill_holes(1000) # Arbitrary hole size limit
                # If still open, try triangulate?
                if fixed.n_open_edges > 0:
                    print(f"  [!] Body {i} still has {fixed.n_open_edges} open edges after fill.")
                    # Keep anyway? Or discard? 
                    # Discarding loses volume. Keeping crashes TetGen.
                    # Let's keep and hope TetGen 'convex_hull' doesn't ruin everything?
                    # Or use 'triangulate' filter?
                    pass
                body = fixed
            
            repaired_blocks.append(body)
            # Calculate volume if closed
            if body.n_open_edges == 0:
                total_vol += body.volume
            
        log(f"Merging repaired bodies...")
        merged = repaired_blocks.combine().extract_surface()
        merged.save(REPAIRED_FILE)
        
        log(f"Repaired Open Edges: {merged.n_open_edges}")
        log(f"Total Volume of Closed Shells: {total_vol:.2f}")
        
    else:
        log("Mesh is already watertight.")
        mesh.save(REPAIRED_FILE)

if __name__ == "__main__":
    main()
