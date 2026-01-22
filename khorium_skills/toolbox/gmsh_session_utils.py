"""
Gmsh Session Utilities - Optimized session management and analysis
===============================================================

This skill provides patterns for managing Gmsh sessions robustly, 
specifically avoiding common crashes associated with re-initialization 
and dual-finalization in multi-module applications.

Key Pattern: In-Memory Analysis
-----------------------------
Always prefer analyzing the current active Gmsh model ('analyze_current_mesh') 
over opening a file from disk ('analyze_mesh_file'). This avoids:
1. Redundant library initialization overhead.
2. Signal/threading conflicts of re-init.
3. Disk I/O bottlenecks.
4. Finalization race conditions.

Sample Usage:
    import gmsh
    # ... model is already open and meshed ...
    
    from core.cfd_quality import CFDQualityAnalyzer
    analyzer = CFDQualityAnalyzer(verbose=False)
    
    # WRONG: causes re-init and potential double-finalize crash
    # report = analyzer.analyze_mesh_file("mesh.msh") 
    
    # CORRECT: uses the active session
    report = analyzer.analyze_current_mesh() 
    
    # Safe Finalization
    try:
        gmsh.finalize()
    except:
        pass
"""

def safe_gmsh_finalize():
    """Finalize gmsh only if initialized, preventing crashes on double-finalize."""
    try:
        import gmsh
        gmsh.finalize()
        return True
    except Exception:
        return False
