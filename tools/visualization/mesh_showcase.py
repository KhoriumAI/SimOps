#!/usr/bin/env python3
"""
Professional Mesh Showcase Tool
================================
Create stunning rotating visualizations of your meshes for demo videos.

Features:
- Smooth rotation at configurable speed
- Professional 3-point lighting setup
- Gradient backgrounds
- Platform/pedestal display
- High-quality rendering
- Image sequence or video export
- Batch processing for multiple meshes
"""

import vtk
import argparse
import sys
from pathlib import Path
import math
import gmsh
import subprocess
import shutil


class MeshShowcase:
    """Professional mesh visualization and animation tool"""

    def __init__(self, mesh_file, output_dir="showcase_output",
                 seconds_per_revolution=30, fps=30, resolution=(1920, 1080)):
        """
        Initialize showcase

        Args:
            mesh_file: Path to .msh file
            output_dir: Directory for output images/video
            seconds_per_revolution: Rotation speed (seconds for 360°)
            fps: Frames per second for animation
            resolution: Output resolution (width, height)
        """
        self.mesh_file = Path(mesh_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.seconds_per_revolution = seconds_per_revolution
        self.fps = fps
        self.resolution = resolution

        # Calculate animation parameters
        self.total_frames = int(self.fps * self.seconds_per_revolution)
        self.degrees_per_frame = 360.0 / self.total_frames

        # VTK components
        self.renderer = None
        self.render_window = None
        self.mesh_actor = None
        self.platform_actor = None
        self.mesh_center = None  # Store mesh center for consistent rotation

    def load_mesh(self):
        """Load mesh from .msh file using gmsh API"""
        print(f"Loading mesh: {self.mesh_file}")

        gmsh.initialize()
        gmsh.open(str(self.mesh_file))

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Create VTK structures
        points = vtk.vtkPoints()
        node_map = {}

        for i, node_id in enumerate(node_tags):
            idx = i * 3
            x, y, z = node_coords[idx], node_coords[idx+1], node_coords[idx+2]
            points.InsertNextPoint(x, y, z)
            node_map[int(node_id)] = i

        # Get surface triangles (dim=2)
        cells = vtk.vtkCellArray()
        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)

        triangle_count = 0
        for elem_type, elem_tags, elem_node_tags in zip(elem_types, elem_tags_list, elem_node_tags_list):
            elem_type = int(elem_type)

            # Triangle types: 2 (3-node), 9 (6-node)
            if elem_type in [2, 9]:
                nodes_per_elem = 3 if elem_type == 2 else 6

                for i in range(len(elem_tags)):
                    start_idx = i * nodes_per_elem
                    # Use first 3 nodes for visualization
                    node_ids = [int(n) for n in elem_node_tags[start_idx:start_idx+3]]

                    tri = vtk.vtkTriangle()
                    for j, node_id in enumerate(node_ids):
                        tri.GetPointIds().SetId(j, node_map[node_id])
                    cells.InsertNextCell(tri)
                    triangle_count += 1

        gmsh.finalize()

        # Create polydata
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Compute normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly_data)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.Update()

        print(f"  Loaded: {points.GetNumberOfPoints()} points, {triangle_count} triangles")

        return normals.GetOutput()

    def create_platform(self, mesh_bounds):
        """Create a large cube table that extends beyond screen edges for grounded look"""
        # VTK coordinate system: X-right, Y-up, Z-towards viewer
        x_range = mesh_bounds[1] - mesh_bounds[0]
        y_range = mesh_bounds[3] - mesh_bounds[2]
        z_range = mesh_bounds[5] - mesh_bounds[4]
        y_min = mesh_bounds[2]  # Bottom of mesh in Y

        # Create LARGE table - much bigger than the object to extend off-screen
        max_dim = max(x_range, y_range, z_range)
        table_size = max_dim * 5.0  # 5x the object size so it goes off-screen
        table_height = max_dim * 0.15  # Thicker for more substantial look

        # Create cube for table
        platform = vtk.vtkCubeSource()
        platform.SetXLength(table_size)
        platform.SetYLength(table_height)
        platform.SetZLength(table_size)
        platform.Update()

        # Position table below mesh, centered
        transform = vtk.vtkTransform()
        transform.Translate(0, y_min - table_height/2, 0)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(platform.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        return transform_filter.GetOutput()

    def setup_lighting(self):
        """Create CINEMATIC 4-point lighting setup with dramatic backlighting"""
        # Remove default lights
        self.renderer.RemoveAllLights()

        # Key Light (main light) - bright, from front-top-right
        key_light = vtk.vtkLight()
        key_light.SetPosition(3, 4, 3)
        key_light.SetFocalPoint(0, 0, 0)
        key_light.SetColor(1.0, 1.0, 1.0)  # Pure white
        key_light.SetIntensity(1.0)
        self.renderer.AddLight(key_light)

        # Fill Light - softer, from front-left to fill shadows
        fill_light = vtk.vtkLight()
        fill_light.SetPosition(-2, 2, 2.5)
        fill_light.SetFocalPoint(0, 0, 0)
        fill_light.SetColor(0.85, 0.9, 1.0)  # Cool blue tint
        fill_light.SetIntensity(0.35)
        self.renderer.AddLight(fill_light)

        # Back Light 1 - DRAMATIC rim light from behind-above
        back_light_1 = vtk.vtkLight()
        back_light_1.SetPosition(-1, 4, -3)
        back_light_1.SetFocalPoint(0, 0, 0)
        back_light_1.SetColor(1.0, 0.95, 0.85)  # Warm golden edge
        back_light_1.SetIntensity(0.9)  # Strong for rim lighting effect
        self.renderer.AddLight(back_light_1)

        # Back Light 2 - Secondary rim light from opposite side
        back_light_2 = vtk.vtkLight()
        back_light_2.SetPosition(2, 3, -3)
        back_light_2.SetFocalPoint(0, 0, 0)
        back_light_2.SetColor(0.9, 0.95, 1.0)  # Cool blue edge
        back_light_2.SetIntensity(0.7)
        self.renderer.AddLight(back_light_2)

        # Ambient light - very subtle for depth
        self.renderer.SetAmbient(0.15, 0.15, 0.18)

    def setup_gradient_background(self, color1=(0.1, 0.1, 0.15), color2=(0.3, 0.35, 0.4)):
        """Setup gradient background"""
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(color1[0], color1[1], color1[2])  # Bottom color
        self.renderer.SetBackground2(color2[0], color2[1], color2[2])  # Top color

    def setup_scene(self):
        """Setup the complete scene"""
        print("Setting up scene...")

        # Load mesh
        mesh_data = self.load_mesh()
        bounds = mesh_data.GetBounds()

        # Create mesh actor
        mesh_mapper = vtk.vtkPolyDataMapper()
        mesh_mapper.SetInputData(mesh_data)

        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(mesh_mapper)

        # Mesh appearance - CINEMATIC white/gray material
        mesh_property = self.mesh_actor.GetProperty()
        mesh_property.SetColor(0.9, 0.9, 0.92)  # Light gray/white with subtle blue tint
        mesh_property.SetSpecular(0.8)  # High specular for metallic look
        mesh_property.SetSpecularPower(60)  # Sharp highlights
        mesh_property.SetDiffuse(0.7)
        mesh_property.SetAmbient(0.15)
        mesh_property.SetInterpolationToPhong()  # Smooth shading

        # Show edges for mesh visualization
        mesh_property.EdgeVisibilityOn()
        mesh_property.SetEdgeColor(0.2, 0.2, 0.25)  # Darker edges for contrast
        mesh_property.SetLineWidth(2.0)  # Thicker lines to reduce flickering

        # Create platform
        platform_data = self.create_platform(bounds)

        platform_mapper = vtk.vtkPolyDataMapper()
        platform_mapper.SetInputData(platform_data)

        # Push platform slightly back to prevent edge jitter/z-fighting
        platform_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        platform_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)

        self.platform_actor = vtk.vtkActor()
        self.platform_actor.SetMapper(platform_mapper)

        # Platform appearance - darker gray glossy surface
        platform_property = self.platform_actor.GetProperty()
        platform_property.SetColor(0.25, 0.25, 0.25)  # Darker gray
        platform_property.SetSpecular(0.9)
        platform_property.SetSpecularPower(50)
        platform_property.SetDiffuse(0.6)
        platform_property.SetAmbient(0.3)

        # Store mesh center for consistent rotation
        self.mesh_center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        # Setup renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.mesh_actor)
        self.renderer.AddActor(self.platform_actor)

        # Setup gradient background - VERY SUBTLE like professional interview backgrounds
        self.setup_gradient_background(
            color1=(0.78, 0.78, 0.78),   # Bottom - light grey
            color2=(0.88, 0.88, 0.88)    # Top - slightly lighter grey (subtle gradient)
        )

        # Setup professional lighting
        self.setup_lighting()

        # Setup camera
        camera = self.renderer.GetActiveCamera()

        # Center mesh in view
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        # Calculate good camera distance
        max_dim = max(bounds[1] - bounds[0],
                     bounds[3] - bounds[2],
                     bounds[5] - bounds[4])
        distance = max_dim * 2.5

        # Position camera at 30-degree angle for professional look
        angle_rad = math.radians(30)
        camera.SetPosition(
            center[0] + distance * math.cos(angle_rad),
            center[1] + distance * 0.8,  # Elevated
            center[2] + distance * math.sin(angle_rad)
        )
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 1, 0)

        # Setup render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(self.resolution[0], self.resolution[1])
        self.render_window.SetOffScreenRendering(1)  # Render without window

        # Enable anti-aliasing and quality settings for smooth rendering
        self.render_window.SetMultiSamples(8)  # MSAA for smooth edges
        self.render_window.LineSmoothingOn()  # Smooth line rendering
        self.render_window.PolygonSmoothingOn()  # Smooth polygons

        print("[OK] Scene setup complete")

    def apply_clipping(self, axis='x', offset_percentage=0.0):
        """
        Apply a capped cross-section cut to the mesh.
        
        Args:
            axis: 'x', 'y', or 'z'
            offset_percentage: Offset from center as percentage of bounding box dimension (-50 to 50)
        """
        print(f"Applying cross-section cut: Axis={axis.upper()}, Offset={offset_percentage}%")
        
        # Get current mesh data
        poly_data = self.mesh_actor.GetMapper().GetInput()
        bounds = poly_data.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]
        
        # Determine plane normal and origin
        plane = vtk.vtkPlane()
        
        if axis.lower() == 'x':
            normal = (-1, 0, 0) # Cut to reveal inside from positive X
            dim_size = bounds[1] - bounds[0]
            origin = [center[0] + (dim_size * offset_percentage / 100.0), center[1], center[2]]
        elif axis.lower() == 'y':
            normal = (0, -1, 0) # Cut top to look down
            dim_size = bounds[3] - bounds[2]
            origin = [center[0], center[1] + (dim_size * offset_percentage / 100.0), center[2]]
        else: # z
            normal = (0, 0, -1) # Cut front to look in
            dim_size = bounds[5] - bounds[4]
            origin = [center[0], center[1], center[2] + (dim_size * offset_percentage / 100.0)]
            
        plane.SetNormal(normal)
        plane.SetOrigin(origin)
        
        # Create plane collection for clipper
        planes = vtk.vtkPlaneCollection()
        planes.AddItem(plane)
        
        # Use vtkClipClosedSurface to create capped cut
        clipper = vtk.vtkClipClosedSurface()
        clipper.SetInputData(poly_data)
        clipper.SetClippingPlanes(planes)
        clipper.SetActivePlaneId(0)
        clipper.SetScalarModeToColors()
        clipper.SetClipColor(0.8, 0.3, 0.3) # Reddish cap color for visibility
        clipper.SetBaseColor(0.9, 0.9, 0.92) # Match mesh color
        clipper.SetActivePlaneColor(0.8, 0.3, 0.3) # Cap color
        
        # Update clipper
        try:
            clipper.Update()
            
            # Update mesh mapper with clipped output
            self.mesh_actor.GetMapper().SetInputData(clipper.GetOutput())
            
            # Ensure backface culling is ON (default) so we see the cap, not inside of back faces
            # But vtkClipClosedSurface creates a solid cap, so standard rendering works
            self.mesh_actor.GetProperty().BackfaceCullingOn()
            
            print("[OK] Cross-section applied successfully")
            
        except Exception as e:
            print(f"[!] Failed to apply clipping: {e}")


    def render_frame(self, frame_number):
        """Render a single frame"""
        # Rotate mesh around Y-axis (vertical) through its stored center
        angle = self.degrees_per_frame * frame_number

        # Create transform to rotate around mesh center (use stored center for consistency)
        transform = vtk.vtkTransform()
        transform.Translate(self.mesh_center[0], self.mesh_center[1], self.mesh_center[2])  # Move to mesh center
        transform.RotateY(angle)  # Rotate around Y (vertical)
        transform.Translate(-self.mesh_center[0], -self.mesh_center[1], -self.mesh_center[2])  # Move back

        # Apply transform
        self.mesh_actor.SetUserTransform(transform)

        # Render
        self.render_window.Render()

        # Capture frame
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.Update()

        return window_to_image.GetOutput()

    def create_video(self, frames_path, video_quality='high', delete_frames=False):
        """
        Automatically create video from frames using ffmpeg

        Args:
            frames_path: Path to directory containing frames
            video_quality: 'high', 'medium', or 'web'
            delete_frames: Delete frames after video creation to save storage
        """
        # Check if ffmpeg is installed
        if not shutil.which('ffmpeg'):
            print("\n[!] ffmpeg not found - skipping video creation")
            print("  Install with: brew install ffmpeg")
            return None

        mesh_name = self.mesh_file.stem
        video_file = frames_path / f"{mesh_name}_showcase.mp4"

        # Quality presets
        quality_settings = {
            'high': {'crf': 15, 'preset': 'slow'},      # Highest quality
            'medium': {'crf': 20, 'preset': 'medium'},  # Balanced
            'web': {'crf': 23, 'preset': 'fast'}        # Smaller file
        }

        settings = quality_settings.get(video_quality, quality_settings['medium'])

        print(f"\n🎬 Creating video with ffmpeg ({video_quality} quality)...")

        # FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(self.fps),
            '-i', str(frames_path / 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-preset', settings['preset'],
            '-crf', str(settings['crf']),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',  # Optimize for web streaming
            str(video_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Get file size
                file_size = video_file.stat().st_size / (1024 * 1024)  # MB
                print(f"[OK] Video created: {video_file.name} ({file_size:.1f} MB)")

                # Delete frames if requested to save storage
                if delete_frames:
                    frame_count = 0
                    for frame_file in frames_path.glob('frame_*.png'):
                        frame_file.unlink()
                        frame_count += 1
                    print(f"[OK] Deleted {frame_count} frames to save storage")

                return video_file
            else:
                print(f"[X] FFmpeg error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("[X] Video creation timed out")
            return None
        except Exception as e:
            print(f"[X] Error creating video: {e}")
            return None

    def export_image_sequence(self, create_video=True, video_quality='high', delete_frames=True):
        """
        Export animation as image sequence and optionally create video

        Args:
            create_video: Automatically create video after rendering frames
            video_quality: 'high', 'medium', or 'web'
            delete_frames: Delete frames after video creation to save storage
        """
        print(f"\nExporting image sequence ({self.total_frames} frames @ {self.fps} fps)...")
        print(f"Rotation: 360° in {self.seconds_per_revolution} seconds")
        print(f"Output: {self.output_dir}/")

        # Create output subdirectory for this mesh
        mesh_name = self.mesh_file.stem
        output_path = self.output_dir / mesh_name
        output_path.mkdir(exist_ok=True)

        for frame in range(self.total_frames):
            image_data = self.render_frame(frame)

            # Write PNG
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(output_path / f"frame_{frame:04d}.png"))
            writer.SetInputData(image_data)
            writer.Write()

            # Progress indicator
            percent = (frame + 1) / self.total_frames * 100
            print(f"\r  Progress: {percent:.1f}% ({frame + 1}/{self.total_frames} frames)", end="", flush=True)

        print(f"\n[OK] Image sequence saved to: {output_path}/")

        # Automatically create video if requested
        if create_video:
            video_file = self.create_video(output_path, video_quality, delete_frames=delete_frames)
            if video_file:
                print(f"\n✨ Ready! Open: {video_file}")
        else:
            print(f"\nTo create video manually:")
            print(f"  cd {output_path}")
            print(f"  ffmpeg -framerate {self.fps} -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {mesh_name}_showcase.mp4")

        return output_path

    def preview(self):
        """Interactive preview (opens window)"""
        print("\nOpening interactive preview...")
        print("Controls:")
        print("  - Drag to rotate")
        print("  - Scroll to zoom")
        print("  - 'q' to quit")

        # Re-enable on-screen rendering
        self.render_window.SetOffScreenRendering(0)

        # Add interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(self.render_window)

        # Use trackball camera style
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        self.render_window.Render()
        interactor.Start()


def create_global_platform_at_origin(max_dimension, platform_y):
    """Create platform at fixed world position (no per-mesh variation)"""
    table_size = max_dimension * 5.0
    table_height = max_dimension * 0.15

    platform = vtk.vtkCubeSource()
    platform.SetXLength(table_size)
    platform.SetYLength(table_height)
    platform.SetZLength(table_size)
    platform.Update()

    # Position at fixed Y in world space
    transform = vtk.vtkTransform()
    transform.Translate(0, platform_y, 0)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(platform.GetOutput())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


def create_mesh_shadow_projection(mesh_polydata, shadow_y, mesh_bounds):
    """Project mesh silhouette onto platform as shadow"""
    # Project all mesh points down to shadow_y plane
    shadow_points = vtk.vtkPoints()
    original_points = mesh_polydata.GetPoints()

    # Get XZ bounds for shadow scaling
    x_size = mesh_bounds[1] - mesh_bounds[0]
    z_size = mesh_bounds[5] - mesh_bounds[4]

    # Project each point to shadow plane (Y = shadow_y)
    for i in range(original_points.GetNumberOfPoints()):
        point = original_points.GetPoint(i)
        # Keep X and Z, set Y to shadow plane
        shadow_points.InsertNextPoint(point[0], shadow_y, point[2])

    # Create shadow polydata
    shadow_polydata = vtk.vtkPolyData()
    shadow_polydata.SetPoints(shadow_points)
    shadow_polydata.SetPolys(mesh_polydata.GetPolys())

    # Create actor with shadow appearance
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(shadow_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.15, 0.15, 0.15)  # Dark gray
    actor.GetProperty().SetOpacity(0.4)  # Semi-transparent
    actor.GetProperty().LightingOff()  # No lighting on shadow

    # Use polygon offset to prevent z-fighting with platform
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -2)

    return actor


def create_montage_video(mesh_files, output_file="montage_showcase.mp4",
                         rotation_degrees=60, fps=30, video_quality='high',
                         seconds_per_revolution=30, resolution=(1920, 1080),
                         mesh_orientations=None, mesh_x_rotations=None, mesh_z_rotations=None,
                         camera_start_pos=None, camera_end_pos=None,
                         background_mode="Cinematic (Gradient)",
                         clipping_enabled=False, clipping_axis='x', clipping_offset=0.0):
    """
    Create a montage video where multiple objects rotate and match-cut between each other

    Args:
        mesh_files: List of mesh file paths in the order they should appear
        output_file: Output video filename
        rotation_degrees: How many degrees each object rotates before cutting (default: 180)
        fps: Frames per second
        video_quality: 'high', 'medium', or 'web'
        seconds_per_revolution: Rotation speed (seconds for 360°)
        resolution: Video resolution (width, height)
        mesh_orientations: List of initial Y-axis rotations for each mesh (degrees)
        mesh_x_rotations: List of initial X-axis rotations for each mesh (degrees) - useful for correcting orientation
        mesh_z_rotations: List of initial Z-axis rotations for each mesh (degrees)
        camera_start_pos: Starting camera position (elevation, azimuth) in degrees
        camera_end_pos: Ending camera position (elevation, azimuth) in degrees
        background_mode: Background style - 'Cinematic (Gradient)', 'Green Screen', 'Blue Screen', or 'Black'
    """
    if not shutil.which('ffmpeg'):
        print("[!] ffmpeg not found - cannot create montage video")
        print("  Install with: brew install ffmpeg")
        return None

    # Set defaults for orientations and camera
    if mesh_orientations is None:
        mesh_orientations = [0] * len(mesh_files)
    if mesh_x_rotations is None:
        mesh_x_rotations = [0] * len(mesh_files)
    if mesh_z_rotations is None:
        mesh_z_rotations = [0] * len(mesh_files)
    if camera_start_pos is None:
        camera_start_pos = (-5, 5)  # Subtle angle variation: (-5° to +5°)
    if camera_end_pos is None:
        camera_end_pos = (5, -5)   # Keeps view centered on object

    # Support per-mesh rotation degrees (for speed ramping!)
    if isinstance(rotation_degrees, (int, float)):
        rotation_degrees = [rotation_degrees] * len(mesh_files)
    elif len(rotation_degrees) != len(mesh_files):
        raise ValueError(f"rotation_degrees list length ({len(rotation_degrees)}) must match mesh_files length ({len(mesh_files)})")

    print("=" * 70)
    print(f"Creating Cinematic Montage Video: {len(mesh_files)} objects")
    print("=" * 70)

    # Calculate frames per object (can be different for each mesh now!)
    degrees_per_frame = 360.0 / (fps * seconds_per_revolution)
    frames_per_mesh = [int(deg / degrees_per_frame) for deg in rotation_degrees]
    total_frames = sum(frames_per_mesh)

    print(f"\nSettings:")
    print(f"  Rotation degrees: {rotation_degrees}")
    print(f"  Frames per mesh: {frames_per_mesh}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total duration: {total_frames / fps:.1f} seconds")
    print(f"  Camera path: ({camera_start_pos[0]}°, {camera_start_pos[1]}°) -> ({camera_end_pos[0]}°, {camera_end_pos[1]}°)")

    # Create temporary directory for montage frames
    temp_dir = Path("showcase_output") / "montage_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    global_frame_number = 0
    cumulative_rotation = 0  # Track continuous rotation across meshes

    # First pass: find maximum bounding box size for consistent scaling
    print("\nAnalyzing mesh sizes for consistent framing...")
    max_dimension = 0
    mesh_heights = []  # Store heights to calculate focal Y after we know max_dimension

    for mesh_file in mesh_files:
        gmsh.initialize()
        gmsh.open(str(mesh_file))
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        xs = [node_coords[i*3] for i in range(len(node_tags))]
        ys = [node_coords[i*3+1] for i in range(len(node_tags))]
        zs = [node_coords[i*3+2] for i in range(len(node_tags))]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)

        dim = max(x_range, y_range, z_range)
        max_dimension = max(max_dimension, dim)
        mesh_heights.append(y_range)

        gmsh.finalize()

    # NOW calculate global focal Y using the FINAL max_dimension for all meshes
    FLOAT_HEIGHT = max_dimension * 0.02
    max_focal_y = 0
    for mesh_height in mesh_heights:
        # When positioned, bottom is at FLOAT_HEIGHT, so center is at:
        focal_y = FLOAT_HEIGHT + mesh_height / 2
        max_focal_y = max(max_focal_y, focal_y)

    print(f"  Maximum mesh dimension: {max_dimension:.3f}")
    print(f"  Float height (FLOAT_HEIGHT): {FLOAT_HEIGHT:.3f}")
    print(f"  Global focal point Y: {max_focal_y:.3f}")
    print(f"  All meshes will be normalized to world origin")
    print(f"  Camera will perform ONE continuous pan across all meshes")

    # GLOBAL CONSTANTS - same for all meshes (no teleporting!)
    # Note: FLOAT_HEIGHT already calculated above after finding max_dimension
    WORLD_ORIGIN = [0, 0, 0]  # All meshes centered here
    PLATFORM_Y = -max_dimension * 0.15  # Platform CENTER much closer (was -0.5)
    PLATFORM_HEIGHT = max_dimension * 0.15  # Platform thickness
    PLATFORM_TOP_Y = PLATFORM_Y + PLATFORM_HEIGHT / 2  # Top surface of platform
    GLOBAL_FOCAL_Y = max_focal_y  # Camera looks at this Y for ALL meshes (no refocusing!)
    base_camera_distance = max_dimension * 2.5

    # GLOBAL CAMERA TIMING - ONE continuous movement across entire video
    TOTAL_FRAMES = total_frames

    print(f"  Platform center at Y={PLATFORM_Y:.2f}")
    print(f"  Platform top at Y={PLATFORM_TOP_Y:.2f}")
    print(f"  Objects floating at Y={FLOAT_HEIGHT:.2f}")
    print(f"  Camera focal Y: {GLOBAL_FOCAL_Y:.2f} (constant for all meshes)")
    print(f"  Gap (float to platform top): {FLOAT_HEIGHT - PLATFORM_TOP_Y:.2f}")

    # Create global platform at fixed world position
    global_platform_data = create_global_platform_at_origin(max_dimension, PLATFORM_Y)

    # Render each mesh
    for mesh_idx, mesh_file in enumerate(mesh_files):
        print(f"\n[{mesh_idx + 1}/{len(mesh_files)}] Processing: {Path(mesh_file).name}")
        print(f"  Initial orientation X: {mesh_x_rotations[mesh_idx]}°, Y: {mesh_orientations[mesh_idx]}°, Z: {mesh_z_rotations[mesh_idx]}°")
        print(f"  Starting rotation: {cumulative_rotation:.1f}°")

        try:
            # Create showcase for this mesh
            showcase = MeshShowcase(
                mesh_file,
                output_dir=temp_dir,
                seconds_per_revolution=seconds_per_revolution,
                fps=fps,
                resolution=resolution
            )

            showcase.setup_scene()

            # Override background based on mode
            if background_mode == "Green Screen":
                # Bright green for chroma keying (RGB: 0, 255, 0)
                showcase.renderer.GradientBackgroundOff()
                showcase.renderer.SetBackground(0.0, 1.0, 0.0)
            elif background_mode == "Blue Screen":
                # Bright blue for chroma keying (RGB: 0, 0, 255)
                showcase.renderer.GradientBackgroundOff()
                showcase.renderer.SetBackground(0.0, 0.0, 1.0)
            elif background_mode == "Black":
                # Pure black background
                showcase.renderer.GradientBackgroundOff()
                showcase.renderer.SetBackground(0.0, 0.0, 0.0)
            # else: keep the cinematic gradient background from setup_scene()

            # Apply clipping if enabled
            if clipping_enabled:
                showcase.apply_clipping(axis=clipping_axis, offset_percentage=clipping_offset)

            # NORMALIZE MESH TO WORLD ORIGIN
            # Position mesh so BOTTOM is at FLOAT_HEIGHT above platform
            mesh_bounds = showcase.mesh_actor.GetBounds()

            # Calculate X/Z center but use BOTTOM Y
            current_center_x = (mesh_bounds[0] + mesh_bounds[1]) / 2
            current_center_z = (mesh_bounds[4] + mesh_bounds[5]) / 2
            current_bottom_y = mesh_bounds[2]  # Minimum Y = bottom

            # Offset to move mesh: X/Z centered, bottom at float height
            offset_to_origin = [
                WORLD_ORIGIN[0] - current_center_x,
                WORLD_ORIGIN[1] + FLOAT_HEIGHT - current_bottom_y,  # Bottom to float height
                WORLD_ORIGIN[2] - current_center_z
            ]

            # Calculate where mesh center will be after positioning (for rotation pivot)
            mesh_height = mesh_bounds[3] - mesh_bounds[2]
            mesh_center_y = FLOAT_HEIGHT + mesh_height / 2

            # Apply translation to mesh
            mesh_translate = vtk.vtkTransform()
            mesh_translate.Translate(offset_to_origin[0], offset_to_origin[1], offset_to_origin[2])

            # Update mesh position
            mesh_filter = vtk.vtkTransformPolyDataFilter()
            mesh_filter.SetInputData(showcase.mesh_actor.GetMapper().GetInput())
            mesh_filter.SetTransform(mesh_translate)
            mesh_filter.Update()

            # Apply X/Z-axis orientation correction if needed (e.g., for tree standing upright)
            # This is a STATIC correction applied to the geometry, not part of animation
            if mesh_x_rotations[mesh_idx] != 0 or mesh_z_rotations[mesh_idx] != 0:
                orient_transform = vtk.vtkTransform()
                # Rotate around world origin (where mesh center is now)
                orient_transform.Translate(WORLD_ORIGIN[0], mesh_center_y, WORLD_ORIGIN[2])

                # Apply rotations in order: X then Z
                if mesh_x_rotations[mesh_idx] != 0:
                    orient_transform.RotateX(mesh_x_rotations[mesh_idx])
                if mesh_z_rotations[mesh_idx] != 0:
                    orient_transform.RotateZ(mesh_z_rotations[mesh_idx])

                orient_transform.Translate(-WORLD_ORIGIN[0], -mesh_center_y, -WORLD_ORIGIN[2])

                orient_filter = vtk.vtkTransformPolyDataFilter()
                orient_filter.SetInputData(mesh_filter.GetOutput())
                orient_filter.SetTransform(orient_transform)
                orient_filter.Update()
                mesh_filter = orient_filter  # Replace with oriented version

            showcase.mesh_actor.GetMapper().SetInputData(mesh_filter.GetOutput())

            # Update showcase mesh_center to actual center (used for rotation pivot, not camera focal)
            showcase.mesh_center = [WORLD_ORIGIN[0], mesh_center_y, WORLD_ORIGIN[2]]

            # Replace platform with global platform at fixed position
            platform_mapper = vtk.vtkPolyDataMapper()
            platform_mapper.SetInputData(global_platform_data)
            showcase.platform_actor.SetMapper(platform_mapper)

            # Create projected shadow on platform surface
            # Project mesh silhouette onto platform TOP surface (just barely above to prevent z-fighting)
            shadow_actor = create_mesh_shadow_projection(
                mesh_filter.GetOutput(),
                PLATFORM_TOP_Y + 0.1,  # Just 0.1 units above platform top surface
                mesh_bounds
            )
            showcase.renderer.AddActor(shadow_actor)

            # Store shadow actor so we can rotate it with the mesh
            mesh_shadow_actor = shadow_actor

            # Render frames for this object with camera animation
            for local_frame in range(frames_per_mesh[mesh_idx]):
                # GLOBAL PROGRESS: Camera moves continuously across ALL meshes!
                # Not per-mesh - ONE smooth pan from start to finish
                global_progress = global_frame_number / (TOTAL_FRAMES - 1) if TOTAL_FRAMES > 1 else 0

                # Calculate mesh rotation (continuous + initial orientation)
                mesh_rotation = cumulative_rotation + mesh_orientations[mesh_idx] + (degrees_per_frame * local_frame)

                # Apply Y-axis rotation only (continuous animation + initial Y orientation)
                # X-axis orientation was already applied to geometry, not part of animation
                transform = vtk.vtkTransform()
                transform.Translate(showcase.mesh_center[0], showcase.mesh_center[1], showcase.mesh_center[2])
                transform.RotateY(mesh_rotation)
                transform.Translate(-showcase.mesh_center[0], -showcase.mesh_center[1], -showcase.mesh_center[2])
                showcase.mesh_actor.SetUserTransform(transform)

                # Apply SAME transform to shadow so it rotates with mesh
                mesh_shadow_actor.SetUserTransform(transform)

                # CAMERA MOVEMENT: ONE continuous pan across entire video
                # Uses global_progress (0->1 across ALL meshes), not per-mesh progress
                # PURELY LINEAR interpolation to eliminate all jitter!

                # Calculate camera position with linear interpolation
                # Start: top-right, End: bottom-left
                start_x = base_camera_distance * 0.5
                end_x = -base_camera_distance * 0.5
                start_y = base_camera_distance * 0.4
                end_y = base_camera_distance * 0.15

                # Linear interpolation: pos = start + t * (end - start)
                camera_pos = [
                    WORLD_ORIGIN[0] + start_x + global_progress * (end_x - start_x),
                    GLOBAL_FOCAL_Y + start_y + global_progress * (end_y - start_y),
                    WORLD_ORIGIN[2] + base_camera_distance
                ]

                # Camera focal point - CONSTANT for ALL meshes (no refocusing!)
                camera = showcase.renderer.GetActiveCamera()
                camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])
                camera.SetFocalPoint(WORLD_ORIGIN[0], GLOBAL_FOCAL_Y, WORLD_ORIGIN[2])
                camera.SetViewUp(0, 1, 0)

                # Set clipping range based on GLOBAL max_dimension (only needs to be set once per mesh)
                if local_frame == 0:
                    # Near: 10% of max dimension, Far: 10x max dimension
                    camera.SetClippingRange(max_dimension * 0.1, max_dimension * 10)

                # Render
                showcase.render_window.Render()

                # Capture frame
                window_to_image = vtk.vtkWindowToImageFilter()
                window_to_image.SetInput(showcase.render_window)
                window_to_image.Update()

                # Write frame
                writer = vtk.vtkPNGWriter()
                writer.SetFileName(str(temp_dir / f"montage_{global_frame_number:04d}.png"))
                writer.SetInputData(window_to_image.GetOutput())
                writer.Write()

                global_frame_number += 1

                # Progress indicator - show OVERALL progress, not per-mesh
                overall_percent = (global_frame_number / TOTAL_FRAMES) * 100
                print(f"\r  Progress: {overall_percent:.1f}% ({global_frame_number}/{TOTAL_FRAMES} frames) - Mesh {mesh_idx + 1}/{len(mesh_files)}", end="", flush=True)

            print()  # New line after progress

            # Update cumulative rotation for next mesh
            cumulative_rotation += rotation_degrees[mesh_idx]

        except Exception as e:
            print(f"\n[X] Error processing {mesh_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n[OK] All frames rendered ({global_frame_number} total)")

    # Create video with ffmpeg
    print(f"\n🎬 Creating montage video with ffmpeg ({video_quality} quality)...")

    quality_settings = {
        'high': {'crf': 15, 'preset': 'slow'},
        'medium': {'crf': 20, 'preset': 'medium'},
        'web': {'crf': 23, 'preset': 'fast'}
    }
    settings = quality_settings.get(video_quality, quality_settings['medium'])

    output_path = Path("showcase_output") / output_file

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', str(temp_dir / 'montage_%04d.png'),
        '-c:v', 'libx264',
        '-preset', settings['preset'],
        '-crf', str(settings['crf']),
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Montage video created: {output_path.name} ({file_size:.1f} MB)")

            # Clean up temporary frames
            print(f"[OK] Cleaning up temporary frames...")
            for frame_file in temp_dir.glob('montage_*.png'):
                frame_file.unlink()
            temp_dir.rmdir()

            print(f"\n✨ Ready! Open: {output_path}")
            return output_path
        else:
            print(f"[X] FFmpeg error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("[X] Video creation timed out")
        return None
    except Exception as e:
        print(f"[X] Error creating video: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Create professional rotating mesh showcases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Preview a mesh interactively
  python mesh_showcase.py mesh.msh --preview

  # Generate rotating animation (image sequence)
  python mesh_showcase.py mesh.msh --export

  # Custom rotation speed (10 seconds per revolution)
  python mesh_showcase.py mesh.msh --export --speed 10

  # High resolution 4K output
  python mesh_showcase.py mesh.msh --export --resolution 3840 2160

  # Batch process multiple meshes
  python mesh_showcase.py generated_meshes/*.msh --export

Output:
  Image sequences are saved to showcase_output/<mesh_name>/frame_*.png
  Use ffmpeg to combine into video (command is printed after export)
        '''
    )

    parser.add_argument('mesh_files', nargs='+', help='Mesh file(s) to visualize (.msh format)')
    parser.add_argument('--preview', action='store_true', help='Interactive preview (opens window)')
    parser.add_argument('--export', action='store_true', help='Export image sequence and create video')
    parser.add_argument('--speed', type=float, default=30, help='Seconds per revolution (default: 30)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'), help='Resolution (default: 1920 1080)')
    parser.add_argument('--output-dir', default='showcase_output', help='Output directory')
    parser.add_argument('--video-quality', choices=['high', 'medium', 'web'], default='high',
                       help='Video quality: high (CRF 15), medium (CRF 20), web (CRF 23) (default: high)')
    parser.add_argument('--frames-only', action='store_true',
                       help='Export frames only without creating video')
    parser.add_argument('--keep-frames', action='store_true',
                       help='Keep frame files after video creation (default: delete to save space)')
    parser.add_argument('--montage', action='store_true',
                       help='Create montage video with match-cuts between multiple meshes')
    parser.add_argument('--rotation-degrees', type=float, default=180,
                       help='Degrees of rotation per object in montage (default: 180)')
    parser.add_argument('--montage-output', default='montage_showcase.mp4',
                       help='Output filename for montage video (default: montage_showcase.mp4)')
    parser.add_argument('--clip-axis', choices=['x', 'y', 'z'], default='x',
                       help='Axis for cross-section cut (default: x)')
    parser.add_argument('--clip-offset', type=float, default=0.0,
                       help='Offset for cross-section cut as percentage (-50 to 50) (default: 0)')
    parser.add_argument('--clip', action='store_true',
                       help='Enable cross-section clipping')

    args = parser.parse_args()

    # Handle montage mode
    if args.montage:
        if len(args.mesh_files) < 2:
            print("Error: Montage mode requires at least 2 mesh files")
            sys.exit(1)

        create_montage_video(
            mesh_files=args.mesh_files,
            output_file=args.montage_output,
            rotation_degrees=args.rotation_degrees,
            fps=args.fps,
            video_quality=args.video_quality,
            seconds_per_revolution=args.speed,
            resolution=tuple(args.resolution),
            clipping_enabled=args.clip,
            clipping_axis=args.clip_axis,
            clipping_offset=args.clip_offset
        )
        return

    if not args.preview and not args.export:
        print("Error: Specify --preview or --export (or both)")
        sys.exit(1)

    # Process each mesh file
    for mesh_file in args.mesh_files:
        print("=" * 70)
        print(f"Processing: {mesh_file}")
        print("=" * 70)

        try:
            showcase = MeshShowcase(
                mesh_file,
                output_dir=args.output_dir,
                seconds_per_revolution=args.speed,
                fps=args.fps,
                resolution=tuple(args.resolution)
            )

            showcase.setup_scene()

            if args.clip:
                showcase.apply_clipping(axis=args.clip_axis, offset_percentage=args.clip_offset)

            if args.export:
                # Export frames and optionally create video
                create_video = not args.frames_only
                delete_frames = not args.keep_frames  # Delete frames by default unless --keep-frames is specified
                showcase.export_image_sequence(
                    create_video=create_video,
                    video_quality=args.video_quality,
                    delete_frames=delete_frames
                )

            if args.preview:
                showcase.preview()

            print()

        except Exception as e:
            print(f"Error processing {mesh_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
