#!/usr/bin/env python3
"""
D-Bruitage Images
=================
Logiciel de débruitage d'image haute performance avec interface graphique.

Lancement :
    python main.py [image]

Arguments optionnels :
    image   Chemin vers une image à ouvrir au démarrage.

Algorithmes disponibles :
    - Gaussien
    - Médian
    - Bilatéral
    - Non-Local Means (NLM)
    - Ondelettes (Wavelet)
    - Total Variation (TV)
    - BM3D-like (double passe NLM + Wiener)
    - DnCNN (réseau de neurones convolutionnel)
"""

import sys
import os

# Désactive les avertissements Qt non critiques
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.xcb=false")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from gui.main_window import MainWindow


def main():
    # Support HiDPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("D-Bruitage Images")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("D-Bruitage")

    window = MainWindow()
    window.show()

    # Ouvrir une image si passée en argument
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            window._load_image(path)
        else:
            print(f"Avertissement : fichier introuvable : {path}", file=sys.stderr)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
