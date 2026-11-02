from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

if not getattr(sys, "frozen", False) and find_spec("PySide6") is None:
    # Conda environments can reuse a Qt binding installed in their base prefix.
    # Project dependencies remain declared in requirements-app.txt for portability.
    conda_base_site = Path(sys.prefix).parent.parent / "Lib" / "site-packages"
    if (conda_base_site / "PySide6").is_dir():
        sys.path.append(str(conda_base_site))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Point2Change")
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
