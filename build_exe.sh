#!/bin/bash
# ============================================================
#  Script de build Linux — D-Bruitage Images
#  Génère dist/D-Bruitage-Images
# ============================================================

set -e

echo "=== Installation des dépendances ==="
pip3 install PyQt5 opencv-python-headless numpy scipy scikit-image \
             Pillow PyWavelets pyinstaller

echo ""
echo "=== Construction de l'exécutable ==="
pyinstaller d_bruitage.spec --clean

echo ""
if [ -f "dist/D-Bruitage-Images" ]; then
    chmod +x dist/D-Bruitage-Images
    SIZE=$(du -sh dist/D-Bruitage-Images | cut -f1)
    echo "SUCCESS : dist/D-Bruitage-Images créé (${SIZE})"
else
    echo "ERREUR : la construction a échoué."
    exit 1
fi
