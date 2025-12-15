"""
Splash Screen Module
====================

Cross-platform splash screen with loading progress for Windows and macOS.
"""

import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QFont, QPainter, QLinearGradient, QColor, QPalette, QBrush
)


class SplashScreen(QWidget):
    """Modern splash screen with gradient background and progress bar."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.SplashScreen
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        # Fixed size for splash
        self.setFixedSize(480, 320)
        
        # Center on screen
        self.center_on_screen()
        
        self.init_ui()

    def center_on_screen(self):
        """Center the splash screen on the primary display."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)

    def force_focus(self):
        """Force the window to the front and gain focus (macOS fix)."""
        self.show()
        self.raise_()
        self.activateWindow()


    def init_ui(self):
        """Initialize the splash screen UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 50, 40, 40)
        layout.setSpacing(15)

        # Spacer at top
        layout.addStretch(1)

        # App title
        self.title_label = QLabel("Khorium MeshGen")
        self.title_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Subtitle
        self.subtitle_label = QLabel("Parallel Mesh Generation Engine")
        self.subtitle_label.setFont(QFont("Arial", 14))
        self.subtitle_label.setStyleSheet("color: rgba(255, 255, 255, 0.8);")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.subtitle_label)

        # Spacer
        layout.addStretch(1)

        # Status text
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: rgba(255, 255, 255, 0.2);
            }
            QProgressBar::chunk {
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4facfe, stop:1 #00f2fe);
            }
        """)
        layout.addWidget(self.progress_bar)

        # Percentage text
        self.percent_label = QLabel("0%")
        self.percent_label.setFont(QFont("Arial", 10))
        self.percent_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        self.percent_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.percent_label)

        # Version/copyright
        self.version_label = QLabel("v1.0 - Parallel Edition")
        self.version_label.setFont(QFont("Arial", 9))
        self.version_label.setStyleSheet("color: rgba(255, 255, 255, 0.4);")
        self.version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.version_label)

    def paintEvent(self, event):
        """Draw gradient background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create gradient from dark blue to purple
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(25, 55, 109))   # Dark navy blue
        gradient.setColorAt(0.5, QColor(54, 51, 107))   # Deep purple
        gradient.setColorAt(1.0, QColor(78, 42, 92))    # Purple/magenta

        # Draw rounded rectangle
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 12, 12)

    def set_status(self, text: str):
        """Update the status text."""
        self.status_label.setText(text)
        QApplication.processEvents()

    def set_progress(self, value: int):
        """Update the progress bar value (0-100)."""
        self.progress_bar.setValue(value)
        self.percent_label.setText(f"{value}%")
        QApplication.processEvents()

    def finish(self, main_window):
        """Close the splash screen and show the main window."""
        self.set_progress(100)
        self.set_status("Ready!")
        QApplication.processEvents()
        
        # Brief pause to show 100%
        QTimer.singleShot(200, lambda: self._do_finish(main_window))

    def _do_finish(self, main_window):
        """Actually close and show main window."""
        main_window.show()
        self.close()


def create_light_palette() -> QPalette:
    """
    Create a light mode palette for consistent appearance on macOS and Windows.
    This prevents the dark mode issue on macOS.
    """
    palette = QPalette()
    
    # Window background
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    # Base (text input backgrounds)
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    # AlternateBase (alternating rows)
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    # ToolTipBase
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    # ToolTipText
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    # Text (general text)
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    # WindowText
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
    # Button
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    # ButtonText
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    # BrightText
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    # Highlight (selections)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    # HighlightedText
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    # Link
    palette.setColor(QPalette.Link, QColor(0, 120, 215))
    
    return palette


# For standalone testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(create_light_palette())
    
    splash = SplashScreen()
    splash.show()
    
    # Simulate loading
    def simulate_loading():
        splash.set_status("Loading PyQt5...")
        splash.set_progress(20)
        QTimer.singleShot(500, lambda: (
            splash.set_status("Loading VTK..."),
            splash.set_progress(50)
        ))
        QTimer.singleShot(1000, lambda: (
            splash.set_status("Initializing UI..."),
            splash.set_progress(80)
        ))
        QTimer.singleShot(1500, lambda: (
            splash.set_status("Ready!"),
            splash.set_progress(100)
        ))
        QTimer.singleShot(2000, app.quit)
    
    QTimer.singleShot(100, simulate_loading)
    sys.exit(app.exec_())
