# D-Bruitage Images

Logiciel de débruitage d'image haute performance avec interface graphique PyQt5.

## Fonctionnalités

- **8 algorithmes** de débruitage classiques et modernes
- **Interface avant/après** avec slider interactif
- **Zoom / pan** à la souris
- **Histogramme RGB** en temps réel
- **Métriques qualité** : PSNR et SSIM
- **Ajout de bruit** (Gaussien, Sel & Poivre, Speckle, Poisson) pour tests
- **Traitement par lots** de dossiers d'images
- **Export PNG/JPEG**

## Algorithmes disponibles

| Algorithme | Vitesse | Qualité | Usage recommandé |
|---|---|---|---|
| Gaussien | ★★★★★ | ★★ | Bruit léger, prévisualisation rapide |
| Médian | ★★★★★ | ★★★ | Bruit impulsionnel (sel & poivre) |
| Bilatéral | ★★★★ | ★★★★ | Usage général, préservation des contours |
| Non-Local Means | ★★ | ★★★★★ | Haute qualité, textures répétitives |
| Ondelettes | ★★★★ | ★★★★ | Bon compromis vitesse/qualité |
| Total Variation | ★★★ | ★★★★ | Images synthétiques, contours nets |
| BM3D-like | ★★ | ★★★★★ | Meilleure qualité sans dépendance externe |
| DnCNN | ★ | ★★★★★ | Réseau de neurones profond (avec poids pré-entraînés) |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Lancement de l'interface graphique
python main.py

# Ouvrir directement une image
python main.py chemin/vers/image.png
```

## Poids DnCNN pré-entraînés

Sans poids pré-entraînés, DnCNN tourne avec des poids aléatoires (démo).
Pour charger des poids entraînés, placez un fichier `dncnn_weights.npz`
dans le dossier `denoiser/` (format exporté via `DnCNNDenoiser.save_weights()`).

## Structure du projet

```
D-bruitage-images/
├── main.py                  # Point d'entrée
├── requirements.txt
├── denoiser/
│   ├── algorithms.py        # Tous les algorithmes classiques
│   └── deep_model.py        # Architecture DnCNN (NumPy pur)
└── gui/
    ├── main_window.py       # Fenêtre principale PyQt5
    └── image_viewer.py      # Widget viewer split + histogramme
```

## Formats d'image supportés

PNG, JPEG, BMP, TIFF, WebP, PPM, PGM
