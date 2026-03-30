# -*- mode: python ; coding: utf-8 -*-
"""
Fichier de configuration PyInstaller pour D-Bruitage Images.
Génère un exécutable monofichier autonome.

Linux  : pyinstaller d_bruitage.spec
Windows: pyinstaller d_bruitage.spec  (depuis une machine Windows)
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Inclure les données scikit-image (modèles internes)
        ('denoiser', 'denoiser'),
        ('gui',      'gui'),
    ],
    hiddenimports=[
        # PyQt5
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        # NumPy / SciPy / scikit-image
        'numpy',
        'numpy.core._multiarray_umath',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        'scipy.signal._signaltools',
        'scipy.signal.windows._windows',
        'scipy.sparse.csgraph._validation',
        'skimage',
        'skimage.restoration',
        'skimage.metrics',
        'skimage._shared',
        'skimage.util',
        # PyWavelets
        'pywt',
        # OpenCV
        'cv2',
        # Pillow
        'PIL',
        'PIL.Image',
        # Autres
        'networkx',
        'tifffile',
        'imageio',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclure les modules inutiles pour réduire la taille
        'tkinter',
        'matplotlib',
        'jupyter',
        'IPython',
        'sphinx',
        'pytest',
        'setuptools',
        'distutils',
        'xml',
        'xmlrpc',
        'email',
        'html',
        'http',
        'ftplib',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='D-Bruitage-Images',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,        # compresse si UPX est disponible
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # pas de fenêtre console noire
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',   # décommenter si un icône .ico est disponible
)
