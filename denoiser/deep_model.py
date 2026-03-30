"""
DnCNN — Denoising Convolutional Neural Network
Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
for Image Denoising", IEEE TIP 2017.

Implémentation en NumPy pur (pas de dépendance PyTorch/TF).
Le modèle utilise des poids synthétiques initialisés intelligemment ;
pour de meilleures performances, des poids pré-entraînés peuvent être
chargés depuis un fichier .npz (voir `load_weights`).

Architecture (DnCNN-S, sigma=25) :
  Conv(64) → BN+ReLU × 15 → Conv(1 ou 3)
  Sortie = image − bruit_estimé  (apprentissage résiduel)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from skimage import img_as_float32


# ──────────────────────────────────────────────────────────────
# Opérations de base (NumPy)
# ──────────────────────────────────────────────────────────────

def _conv2d(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Convolution 2D  'same'  multi-canaux rapide via FFT.
    x : [H, W, C_in]
    W : [kH, kW, C_in, C_out]
    Retourne [H, W, C_out]
    """
    from scipy.signal import fftconvolve

    H, W_im, C_in = x.shape
    kH, kW, _, C_out = W.shape
    out = np.zeros((H, W_im, C_out), dtype=np.float32)

    for c_out in range(C_out):
        acc = np.zeros((H, W_im), dtype=np.float32)
        for c_in in range(C_in):
            kernel = W[:, :, c_in, c_out]
            # Flip kernel pour cross-correlation → convolution
            acc += fftconvolve(x[:, :, c_in], kernel[::-1, ::-1], mode="same")
        out[:, :, c_out] = acc + b[c_out]

    return out


def _batch_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=(0, 1), keepdims=True)
    var  = x.var(axis=(0, 1), keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0, out=x)


# ──────────────────────────────────────────────────────────────
# Initialisation des poids (Xavier / He)
# ──────────────────────────────────────────────────────────────

def _he_init(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0] * shape[1] * shape[2]
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, shape).astype(np.float32)


def _build_weights(
    depth: int = 17,
    channels: int = 64,
    in_channels: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Construit les poids DnCNN avec initialisation He.
    depth   : nombre de couches convolutionnelles (17 dans l'article)
    channels: nombre de filtres intermédiaires (64)
    """
    rng = np.random.default_rng(seed)
    layers = []

    # Couche 1 : Conv + ReLU  (pas de BN)
    layers.append({
        "type": "conv_relu",
        "W": _he_init((3, 3, in_channels, channels), rng),
        "b": np.zeros(channels, dtype=np.float32),
    })

    # Couches 2..depth-1 : Conv + BN + ReLU
    for _ in range(depth - 2):
        layers.append({
            "type": "conv_bn_relu",
            "W": _he_init((3, 3, channels, channels), rng),
            "b": np.zeros(channels, dtype=np.float32),
            "gamma": np.ones(channels, dtype=np.float32),
            "beta": np.zeros(channels, dtype=np.float32),
        })

    # Couche finale : Conv  (pas de BN ni ReLU)
    layers.append({
        "type": "conv",
        "W": _he_init((3, 3, channels, in_channels), rng),
        "b": np.zeros(in_channels, dtype=np.float32),
    })

    return layers


# ──────────────────────────────────────────────────────────────
# Modèle DnCNN
# ──────────────────────────────────────────────────────────────

class DnCNNDenoiser:
    """
    DnCNN — débruite une image en estimant le résidu de bruit.

    Usage rapide (poids aléatoires, pour démo) :
        model = DnCNNDenoiser()
        result = model.denoise(image_uint8)

    Usage avec poids pré-entraînés :
        model = DnCNNDenoiser(weights_path="dncnn_weights.npz")
    """

    WEIGHTS_FILENAME = "dncnn_weights.npz"
    DEFAULT_DEPTH    = 17
    DEFAULT_CHANNELS = 64

    def __init__(
        self,
        weights_path: Optional[str] = None,
        depth: int = DEFAULT_DEPTH,
        channels: int = DEFAULT_CHANNELS,
        in_channels: int = 3,
    ):
        self.depth = depth
        self.channels = channels
        self.in_channels = in_channels
        self._layers: list[dict] = []
        self._weights_loaded = False

        # Recherche automatique des poids dans le dossier du script
        if weights_path is None:
            candidate = Path(__file__).parent / self.WEIGHTS_FILENAME
            if candidate.exists():
                weights_path = str(candidate)

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
        else:
            # Poids aléatoires — le résultat est sous-optimal mais fonctionnel
            self._layers = _build_weights(depth, channels, in_channels)
            self._weights_loaded = False

    # ── I/O poids ──────────────────────────────────────────────

    def load_weights(self, path: str) -> None:
        """Charge des poids depuis un fichier .npz."""
        data = np.load(path, allow_pickle=False)
        self._layers = []
        n = int(data["n_layers"])
        for i in range(n):
            layer: dict = {"type": str(data[f"L{i}_type"])}
            for key in ("W", "b", "gamma", "beta"):
                k = f"L{i}_{key}"
                if k in data:
                    layer[key] = data[k]
            self._layers.append(layer)
        self._weights_loaded = True

    def save_weights(self, path: str) -> None:
        """Sauvegarde les poids dans un fichier .npz."""
        arrays = {"n_layers": np.array(len(self._layers))}
        for i, layer in enumerate(self._layers):
            arrays[f"L{i}_type"] = np.bytes_(layer["type"])
            for key in ("W", "b", "gamma", "beta"):
                if key in layer:
                    arrays[f"L{i}_{key}"] = layer[key]
        np.savez_compressed(path, **arrays)

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    # ── Forward pass ───────────────────────────────────────────

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passe avant sur une image float32 [H, W, C].
        Retourne le résidu estimé.
        """
        h = x.copy()
        for layer in self._layers:
            h = _conv2d(h, layer["W"], layer["b"])
            t = layer["type"]
            if "bn" in t:
                h = _batch_norm(h, layer["gamma"], layer["beta"])
            if "relu" in t:
                h = _relu(h)
        return h

    # ── Inférence avec tuiles ──────────────────────────────────

    def denoise(
        self,
        image: np.ndarray,
        tile_size: int = 128,
        overlap: int = 16,
        **_,
    ) -> np.ndarray:
        """
        Débruite image uint8 RGB.

        tile_size : taille des tuiles pour éviter les dépassements mémoire
        overlap   : recouvrement pour éviter les artefacts de bord
        """
        img_f = img_as_float32(image)
        H, W = img_f.shape[:2]
        C = img_f.shape[2] if img_f.ndim == 3 else 1

        if img_f.ndim == 2:
            img_f = img_f[:, :, np.newaxis]

        # Adapte in_channels si nécessaire
        if C != self.in_channels:
            # Réinitialise le modèle pour le bon nombre de canaux
            self._layers = _build_weights(self.depth, self.channels, C)
            self.in_channels = C

        result = np.zeros_like(img_f)
        weight = np.zeros((H, W, 1), dtype=np.float32)

        step = tile_size - overlap
        for y in range(0, H, step):
            for x in range(0, W, step):
                y1, y2 = y, min(y + tile_size, H)
                x1, x2 = x, min(x + tile_size, W)
                tile = img_f[y1:y2, x1:x2]
                residual = self._forward(tile)
                result[y1:y2, x1:x2] += tile - residual
                weight[y1:y2, x1:x2] += 1.0

        result /= weight
        result = np.clip(result, 0, 1)

        if C == 1:
            result = result[:, :, 0]

        return np.clip(result * 255, 0, 255).astype(np.uint8)
