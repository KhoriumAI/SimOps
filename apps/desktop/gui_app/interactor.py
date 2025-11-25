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
        print(f"[DEBUG] Left button press - Paint mode: {self.painting_mode}, Parent: {self.parent is not None}")
        if self.painting_mode and self.parent:
            # Start painting - consume event to prevent rotation
            print("[DEBUG] Starting paint operation")
            self.is_painting = True
            self.paint_at_cursor()
            # Abort event to prevent further processing
            self.GetInteractor().SetAbortFlag(1)
        else:
            # Normal rotate
            print("[DEBUG] Normal rotation mode")
            self.OnLeftButtonDown()

    def left_button_release(self, obj, event):
        if self.painting_mode:
            self.is_painting = False
            # Abort event
            self.GetInteractor().SetAbortFlag(1)
        else:
            self.OnLeftButtonUp()

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
            # Normal rotation mode
            self.OnMouseMove()

    def paint_at_cursor(self):
        """Paint surfaces at cursor location"""
        if not self.parent or not hasattr(self.parent, 'on_paint_at_cursor'):
            return

        # Get mouse position
        x, y = self.GetInteractor().GetEventPosition()

        # Pass to parent for handling
        self.parent.on_paint_at_cursor(x, y)
