"""
ImageViewer — widget d'affichage d'image avec zoom/pan et split avant/après.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QPoint, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor,
)
from PyQt5.QtWidgets import QWidget, QSizePolicy


def ndarray_to_qimage(arr) -> QImage:
    """Convertit un ndarray uint8 RGB en QImage."""
    import numpy as np
    if arr is None:
        return QImage()
    if arr.ndim == 2:
        # Niveaux de gris → RGB
        arr = np.stack([arr] * 3, axis=-1)
    h, w, c = arr.shape
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, w, h, w * c, QImage.Format_RGB888)


class SplitImageViewer(QWidget):
    """
    Affiche deux images (originale / débruitée) côte à côte avec un slider
    vertical que l'utilisateur peut déplacer.

    Supporte :
    - Zoom molette souris
    - Pan (glisser avec clic gauche)
    - Slider avant/après (déplacer la ligne de séparation)
    """

    split_moved = pyqtSignal(float)   # position relative [0, 1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

        self._orig_pixmap: QPixmap | None = None
        self._proc_pixmap: QPixmap | None = None

        self._split = 0.5        # position du slider [0, 1]
        self._zoom = 1.0
        self._offset = QPoint(0, 0)

        self._dragging_split = False
        self._panning = False
        self._last_mouse = QPoint()

        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.ArrowCursor))

    # ── API publique ───────────────────────────────────────────

    def set_original(self, arr) -> None:
        if arr is None:
            self._orig_pixmap = None
        else:
            self._orig_pixmap = QPixmap.fromImage(ndarray_to_qimage(arr))
        self.update()

    def set_processed(self, arr) -> None:
        if arr is None:
            self._proc_pixmap = None
        else:
            self._proc_pixmap = QPixmap.fromImage(ndarray_to_qimage(arr))
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._split = 0.5
        self.update()

    def fit_to_window(self) -> None:
        if self._orig_pixmap is None:
            return
        pw, ph = self._orig_pixmap.width(), self._orig_pixmap.height()
        ww, wh = self.width(), self.height()
        self._zoom = min(ww / pw, wh / ph) * 0.95
        self._offset = QPoint(0, 0)
        self.update()

    # ── Calcul de la géométrie ─────────────────────────────────

    def _image_rect(self) -> QRectF:
        """Retourne le rectangle de l'image dans les coordonnées du widget."""
        if self._orig_pixmap is None:
            return QRectF()
        pw = self._orig_pixmap.width() * self._zoom
        ph = self._orig_pixmap.height() * self._zoom
        x = (self.width() - pw) / 2 + self._offset.x()
        y = (self.height() - ph) / 2 + self._offset.y()
        return QRectF(x, y, pw, ph)

    def _split_x(self) -> int:
        r = self._image_rect()
        return int(r.left() + r.width() * self._split)

    # ── Peinture ───────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 35))

        r = self._image_rect()
        if r.isEmpty():
            # Message si pas d'image
            painter.setPen(QColor(120, 120, 130))
            painter.setFont(QFont("Sans", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Ouvrez une image pour commencer\n(Fichier → Ouvrir)")
            return

        split_x = self._split_x()

        # ── Côté gauche : image originale ──────────────────────
        if self._orig_pixmap:
            painter.save()
            painter.setClipRect(int(r.left()), 0, split_x - int(r.left()), self.height())
            painter.drawPixmap(r.toRect(), self._orig_pixmap)
            painter.restore()

        # ── Côté droit : image traitée (ou originale si pas encore traitée)
        right_px = self._proc_pixmap if self._proc_pixmap else self._orig_pixmap
        if right_px:
            painter.save()
            painter.setClipRect(split_x, 0, self.width() - split_x, self.height())
            painter.drawPixmap(r.toRect(), right_px)
            painter.restore()

        # ── Ligne de séparation ────────────────────────────────
        pen = QPen(QColor(255, 255, 255, 220), 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, self.height())

        # Poignée centrale
        handle_r = 14
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        cx, cy = split_x, self.height() // 2
        painter.drawEllipse(cx - handle_r, cy - handle_r, handle_r * 2, handle_r * 2)
        # Chevrons
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawLine(cx - 6, cy - 5, cx - 2, cy)
        painter.drawLine(cx - 2, cy, cx - 6, cy + 5)
        painter.drawLine(cx + 6, cy - 5, cx + 2, cy)
        painter.drawLine(cx + 2, cy, cx + 6, cy + 5)

        # Labels
        font = QFont("Sans", 10, QFont.Bold)
        painter.setFont(font)
        # Original
        if split_x > 80:
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawText(int(r.left()) + 8, int(r.top()) + 22, "ORIGINAL")
        # Traité
        if self.width() - split_x > 80:
            label = "DÉBRUITÉ" if self._proc_pixmap else "ORIGINAL"
            painter.setPen(QColor(100, 220, 100, 200))
            painter.drawText(split_x + 8, int(r.top()) + 22, label)

        # Niveau de zoom
        painter.setPen(QColor(180, 180, 180, 180))
        painter.setFont(QFont("Mono", 9))
        painter.drawText(8, self.height() - 8, f"zoom : {self._zoom:.2f}×")

    # ── Souris ────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            sx = self._split_x()
            if abs(event.x() - sx) < 18:
                self._dragging_split = True
            else:
                self._panning = True
                self._last_mouse = event.pos()
                self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseReleaseEvent(self, event):
        self._dragging_split = False
        self._panning = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseMoveEvent(self, event):
        sx = self._split_x()

        if self._dragging_split:
            r = self._image_rect()
            if r.width() > 0:
                rel = (event.x() - r.left()) / r.width()
                self._split = max(0.02, min(0.98, rel))
                self.update()
                self.split_moved.emit(self._split)
        elif self._panning:
            delta = event.pos() - self._last_mouse
            self._offset += delta
            self._last_mouse = event.pos()
            self.update()
        else:
            if abs(event.x() - sx) < 18:
                self.setCursor(QCursor(Qt.SizeHorCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.05, min(20.0, self._zoom * factor))
        self.update()


# ──────────────────────────────────────────────────────────────
# Histogramme
# ──────────────────────────────────────────────────────────────

class HistogramWidget(QWidget):
    """Affiche l'histogramme RGB superposé d'une image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(130)
        self._hists: list | None = None  # liste de 3 arrays (R, G, B)

    def set_image(self, arr) -> None:
        import numpy as np
        if arr is None:
            self._hists = None
            self.update()
            return
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        hists = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c], bins=256, range=(0, 255))
            hists.append(h.astype(np.float32))
        # Normalise chaque canal
        mx = max(h.max() for h in hists) or 1
        self._hists = [h / mx for h in hists]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 25))

        if self._hists is None:
            painter.setPen(QColor(80, 80, 90))
            painter.drawText(self.rect(), Qt.AlignCenter, "histogramme")
            return

        w, h = self.width(), self.height()
        colors = [QColor(220, 60, 60, 160), QColor(60, 200, 60, 160), QColor(60, 100, 220, 160)]

        for hist, color in zip(self._hists, colors):
            painter.setPen(color)
            n = len(hist)
            for i, v in enumerate(hist):
                x = int(i * w / n)
                bar_h = int(v * (h - 4))
                painter.drawLine(x, h - 2, x, h - 2 - bar_h)
