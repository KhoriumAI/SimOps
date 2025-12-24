"""
Custom VTK Interactor Style
=============================

Enhanced mouse/keyboard interaction for VTK viewer with paintbrush support.
"""

import vtk


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom interactor style with improved pan controls and paintbrush support:
    - Left mouse: Rotate (or paint if paintbrush mode enabled)
    - Right mouse: Rotate (in paint mode) or Pan (in normal mode)
    - Middle mouse or Shift+Left: Pan (alternative)
    - Scroll wheel: Zoom
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.painting_mode = False
        self.is_painting = False
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)
        self.AddObserver("MouseMoveEvent", self.mouse_move)

    def right_button_press(self, obj, event):
        if self.painting_mode:
            # In paint mode, right mouse rotates
            self.OnLeftButtonDown()
        else:
            # In normal mode, right mouse pans
            self.OnMiddleButtonDown()
        return

    def right_button_release(self, obj, event):
        if self.painting_mode:
            # In paint mode, right mouse rotates
            self.OnLeftButtonUp()
        else:
            # In normal mode, right mouse pans
            self.OnMiddleButtonUp()
        return

    def left_button_press(self, obj, event):
        # Store start position for click detection
        self.start_x, self.start_y = self.GetInteractor().GetEventPosition()
        self.is_click_candidate = True
        
        # print(f"[DEBUG] Left press at {self.start_x}, {self.start_y}")

        if self.painting_mode and self.parent:
            # Start painting - consume event to prevent rotation
            # print("[DEBUG] Starting paint operation")
            self.is_painting = True
            self.paint_at_cursor()
            # Abort event to prevent further processing
            self.GetInteractor().SetAbortFlag(1)
        else:
            # Normal rotate
            # print("[DEBUG] Normal rotation mode")
            self.OnLeftButtonDown()

    def left_button_release(self, obj, event):
        if self.painting_mode:
            self.is_painting = False
            # Abort event
            self.GetInteractor().SetAbortFlag(1)
        else:
            # Check for click (vs drag)
            end_x, end_y = self.GetInteractor().GetEventPosition()
            dx = abs(end_x - self.start_x)
            dy = abs(end_y - self.start_y)
            
            # If moved less than 3 pixels, consider it a click
            if self.is_click_candidate and dx < 3 and dy < 3:
                # print(f"[DEBUG] Click detected at {end_x}, {end_y}")
                self.on_click(end_x, end_y)
            
            self.OnLeftButtonUp()
            
    def on_click(self, x, y):
        """Handle click event (selection)"""
        if self.parent:
            # Check for modifiers
            shift = self.GetInteractor().GetShiftKey()
            ctrl = self.GetInteractor().GetControlKey()
            
            # Double click is handled separately by VTK usually, 
            # but we can implement basic selection here
            
            if hasattr(self.parent, 'on_scene_click'):
                self.parent.on_scene_click(x, y, shift=shift, ctrl=ctrl)

    def mouse_move(self, obj, event):
        if self.painting_mode and self.is_painting and self.parent:
            # Continue painting while dragging - DON'T call OnMouseMove to prevent rotation
            self.paint_at_cursor()
            # Abort event to prevent camera rotation
            self.GetInteractor().SetAbortFlag(1)
        elif self.painting_mode and self.parent:
            # In paint mode but not actively painting - update cursor position only
            x, y = self.GetInteractor().GetEventPosition()
            if hasattr(self.parent, 'viewer') and self.parent.viewer:
                self.parent.viewer.update_brush_cursor_position(x, y)
            # Abort event to prevent rotation while in paint mode
            self.GetInteractor().SetAbortFlag(1)
        else:
            # Normal mode - support hover highlighting
            # self.is_click_candidate = False # If moved significantly, it's not a click
            # Actually, standard OnMouseMove handles rotation, so if we are here, we might be dragging
            
            # Check if we are dragging (button down)
            # VTK Interactor doesn't easily tell us button state in MouseMove without tracking
            # But OnLeftButtonDown sets interaction state.
            
            if self.parent and hasattr(self.parent, 'on_scene_hover'):
                # Pass hover event (only if not rotating?)
                # We can check interactors state, but simpler:
                # just pass it. The viewer can decide to ignore if rotating.
                x, y = self.GetInteractor().GetEventPosition()
                # print(f"[INT] Mouse Move: {x}, {y}")
                self.parent.on_scene_hover(x, y)
                
            self.OnMouseMove()

            # Check dynamic edge visibility based on zoom level
            if self.parent and hasattr(self.parent, 'check_edge_visibility'):
                self.parent.check_edge_visibility()

    def paint_at_cursor(self):
        """Paint surfaces at cursor location"""
        if not self.parent or not hasattr(self.parent, 'on_paint_at_cursor'):
            return

        # Get mouse position
        x, y = self.GetInteractor().GetEventPosition()

        # Pass to parent for handling
        self.parent.on_paint_at_cursor(x, y)

    def OnMouseWheelForward(self):
        super().OnMouseWheelForward()
        if self.parent and hasattr(self.parent, 'check_edge_visibility'):
            self.parent.check_edge_visibility()

    def OnMouseWheelBackward(self):
        super().OnMouseWheelBackward()
        if self.parent and hasattr(self.parent, 'check_edge_visibility'):
            self.parent.check_edge_visibility()
