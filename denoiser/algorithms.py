"""
Algorithmes de débruitage d'image haute performance.

Implémente 8 méthodes classiques et modernes :
  1. Gaussien        — filtre passe-bas rapide
  2. Médian          — efficace pour le bruit impulsionnel
  3. Bilatéral       — préserve les contours
  4. Non-Local Means — très haute qualité, lent
  5. Ondelettes      — bon compromis vitesse/qualité
  6. Total Variation — préserve les contours, effet « cartoon »
  7. BM3D-like       — proche de BM3D via patches (skimage)
  8. DnCNN           — réseau de neurones convolutionnel profond
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
    estimate_sigma,
)


# ──────────────────────────────────────────────────────────────
# Types de retour
# ──────────────────────────────────────────────────────────────

@dataclass
class DenoiseResult:
    image: np.ndarray          # uint8 RGB
    elapsed_ms: float = 0.0
    psnr_value: Optional[float] = None
    ssim_value: Optional[float] = None
    algorithm: str = ""
    params: dict = field(default_factory=dict)

    def compute_metrics(self, reference: np.ndarray) -> None:
        """Calcule PSNR et SSIM par rapport à une image de référence."""
        ref_f = img_as_float32(reference)
        res_f = img_as_float32(self.image)
        channel_axis = -1 if ref_f.ndim == 3 else None
        self.psnr_value = psnr(ref_f, res_f, data_range=1.0)
        self.ssim_value = ssim(
            ref_f, res_f,
            data_range=1.0,
            channel_axis=channel_axis,
            win_size=min(7, ref_f.shape[0] - 1 if ref_f.shape[0] % 2 == 0 else ref_f.shape[0]),
        )


# ──────────────────────────────────────────────────────────────
# Algorithmes
# ──────────────────────────────────────────────────────────────

class DenoiseAlgorithm(str, Enum):
    GAUSSIAN   = "Gaussien"
    MEDIAN     = "Médian"
    BILATERAL  = "Bilatéral"
    NLM        = "Non-Local Means"
    WAVELET    = "Ondelettes"
    TV         = "Total Variation"
    BM3D_LIKE  = "BM3D-like (patches)"
    DNCNN      = "DnCNN (réseau de neurones)"


def get_algorithm_names() -> list[str]:
    return [a.value for a in DenoiseAlgorithm]


# ──────────────────────────────────────────────────────────────
# Dispatcher principal
# ──────────────────────────────────────────────────────────────

def denoise(
    image: np.ndarray,
    algorithm: str,
    params: dict,
    dncnn_denoiser=None,
) -> DenoiseResult:
    """
    Débruite une image uint8 RGB.

    Parameters
    ----------
    image        : np.ndarray uint8 [H, W, 3] ou [H, W]
    algorithm    : nom de l'algorithme (DenoiseAlgorithm.value)
    params       : dictionnaire de paramètres spécifiques
    dncnn_denoiser : instance optionnelle de DnCNNDenoiser
    """
    t0 = time.perf_counter()

    algo = DenoiseAlgorithm(algorithm)

    if algo == DenoiseAlgorithm.GAUSSIAN:
        result_img = _gaussian(image, **params)
    elif algo == DenoiseAlgorithm.MEDIAN:
        result_img = _median(image, **params)
    elif algo == DenoiseAlgorithm.BILATERAL:
        result_img = _bilateral(image, **params)
    elif algo == DenoiseAlgorithm.NLM:
        result_img = _nlm(image, **params)
    elif algo == DenoiseAlgorithm.WAVELET:
        result_img = _wavelet(image, **params)
    elif algo == DenoiseAlgorithm.TV:
        result_img = _tv(image, **params)
    elif algo == DenoiseAlgorithm.BM3D_LIKE:
        result_img = _bm3d_like(image, **params)
    elif algo == DenoiseAlgorithm.DNCNN:
        if dncnn_denoiser is None:
            raise RuntimeError("DnCNN denoiser non initialisé")
        result_img = dncnn_denoiser.denoise(image, **params)
    else:
        raise ValueError(f"Algorithme inconnu : {algorithm}")

    elapsed = (time.perf_counter() - t0) * 1000

    return DenoiseResult(
        image=result_img,
        elapsed_ms=elapsed,
        algorithm=algorithm,
        params=params,
    )


# ──────────────────────────────────────────────────────────────
# Implémentations
# ──────────────────────────────────────────────────────────────

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Convertit float [0,1] → uint8 si nécessaire."""
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _gaussian(image: np.ndarray, sigma: float = 1.5, **_) -> np.ndarray:
    """
    Filtre gaussien.  sigma contrôle l'intensité du flou.
    Implémenté via OpenCV pour la vitesse.
    """
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    ksize = max(ksize, 3)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def _median(image: np.ndarray, radius: int = 3, **_) -> np.ndarray:
    """
    Filtre médian.  radius définit la taille du voisinage (1→3x3, 2→5x5, …).
    Très efficace contre le bruit sel-et-poivre.
    """
    ksize = 2 * radius + 1
    return cv2.medianBlur(image, ksize)


def _bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    **_,
) -> np.ndarray:
    """
    Filtre bilatéral : préserve les contours en pondérant à la fois
    la distance spatiale et la distance radiométrique.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def _nlm(
    image: np.ndarray,
    h: float = 0.6,
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = True,
    **_,
) -> np.ndarray:
    """
    Non-Local Means (Buades et al., 2005).
    h : paramètre de filtrage (0.4–1.2 typique).
    Utilise le mode rapide par défaut (intégrale des patches).
    """
    img_f = img_as_float32(image)
    sigma_est = np.mean(estimate_sigma(img_f, channel_axis=-1 if img_f.ndim == 3 else None))
    h_real = h * sigma_est

    denoised = denoise_nl_means(
        img_f,
        h=h_real,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=fast_mode,
        channel_axis=-1 if img_f.ndim == 3 else None,
    )
    return _ensure_uint8(denoised)


def _wavelet(
    image: np.ndarray,
    sigma: Optional[float] = None,
    wavelet: str = "db1",
    mode: str = "soft",
    wavelet_levels: Optional[int] = None,
    **_,
) -> np.ndarray:
    """
    Seuillage par ondelettes (Donoho & Johnstone).
    wavelet : famille d'ondelettes ('db1', 'db2', 'sym4', 'bior1.3', …)
    mode    : 'soft' ou 'hard'
    """
    img_f = img_as_float32(image)
    denoised = denoise_wavelet(
        img_f,
        sigma=sigma,
        wavelet=wavelet,
        mode=mode,
        wavelet_levels=wavelet_levels,
        channel_axis=-1 if img_f.ndim == 3 else None,
        rescale_sigma=True,
    )
    return _ensure_uint8(denoised)


def _tv(
    image: np.ndarray,
    weight: float = 0.1,
    max_iter: int = 200,
    **_,
) -> np.ndarray:
    """
    Débruitage par variation totale (Chambolle, 2004).
    weight : plus grand → plus de lissage (0.05–0.3 typique).
    Très bon pour les images synthétiques ou cartographiées.
    """
    img_f = img_as_float32(image)
    denoised = denoise_tv_chambolle(
        img_f,
        weight=weight,
        max_num_iter=max_iter,
        channel_axis=-1 if img_f.ndim == 3 else None,
    )
    return _ensure_uint8(denoised)


def _bm3d_like(
    image: np.ndarray,
    sigma: float = 25.0,
    patch_size: int = 8,
    patch_distance: int = 13,
    **_,
) -> np.ndarray:
    """
    Approximation de BM3D par double passage NLM sur blocs.
    Étape 1 : NLM rapide pour obtenir une image pilote.
    Étape 2 : NLM guidé (Wiener-like) avec l'image pilote.
    Donne une qualité proche du BM3D officiel sans dépendance externe.
    """
    img_f = img_as_float32(image)
    channel_axis = -1 if img_f.ndim == 3 else None

    # Normalisation sigma en fraction [0,1]
    sigma_norm = sigma / 255.0

    # Passe 1 : estimation brute
    pilot = denoise_nl_means(
        img_f,
        h=1.15 * sigma_norm,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=channel_axis,
    )

    # Passe 2 : filtrage Wiener empirique
    #  h_wiener = sigma² / (sigma² + variance_locale)
    pilot_var = _local_variance(pilot, patch_size)
    sigma2 = sigma_norm ** 2
    wiener_weights = pilot_var / (pilot_var + sigma2 + 1e-8)

    # NLM affiné
    refined = denoise_nl_means(
        img_f,
        h=0.7 * sigma_norm,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=channel_axis,
    )

    # Combinaison Wiener
    if img_f.ndim == 3:
        wiener_weights = wiener_weights[..., np.newaxis]
    result = wiener_weights * refined + (1 - wiener_weights) * pilot

    return _ensure_uint8(np.clip(result, 0, 1))


def _local_variance(img: np.ndarray, size: int) -> np.ndarray:
    """Variance locale calculée via filtre uniforme."""
    if img.ndim == 3:
        img_gray = img.mean(axis=-1)
    else:
        img_gray = img
    mean = uniform_filter(img_gray, size=size)
    mean_sq = uniform_filter(img_gray ** 2, size=size)
    return np.maximum(mean_sq - mean ** 2, 0)
