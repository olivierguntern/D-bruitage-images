@echo off
REM ============================================================
REM  Script de build Windows — D-Bruitage Images
REM  Génère dist\D-Bruitage-Images.exe
REM ============================================================

echo === Installation des dependances ===
pip install PyQt5 opencv-python numpy scipy scikit-image Pillow PyWavelets pyinstaller

echo.
echo === Construction de l'executable ===
pyinstaller d_bruitage.spec --clean

echo.
if exist "dist\D-Bruitage-Images.exe" (
    echo SUCCESS : dist\D-Bruitage-Images.exe cree avec succes !
) else (
    echo ERREUR : la construction a echoue.
    exit /b 1
)
pause
