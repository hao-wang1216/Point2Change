from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class PairCanvas(QGraphicsView):
    point_added = Signal(int, int, int)
    cursor_moved = Signal(int, int)
    view_changed = Signal(float, int, int)
    polygon_finished = Signal(object)
    brush_stroke = Signal(object, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#ffffff"))
        self._image_item: QGraphicsPixmapItem | None = None
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._image_size: tuple[int, int] | None = None
        self._point_label = 1
        self._tool = "positive"
        self._brush_value = 1
        self._points: list[tuple[int, int, int]] = []
        self._crosshair: tuple[int, int] | None = None
        self._polygon_points: list[tuple[int, int]] = []
        self._stroke: list[tuple[int, int]] = []
        self._syncing_view = False

    @property
    def image_size(self) -> tuple[int, int] | None:
        return self._image_size

    @property
    def has_unfinished_edit(self) -> bool:
        return bool(self._polygon_points or self._stroke)

    def cancel_drawing(self) -> None:
        self._polygon_points.clear()
        self._stroke.clear()
        self._redraw_marks()

    def set_image(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("canvas image must be HWC RGB")
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        self.scene().clear()
        self._image_item = self.scene().addPixmap(QPixmap.fromImage(image))
        self._image_item.setZValue(0)
        self._overlay_item = None
        self._image_size = (width, height)
        self.scene().setSceneRect(0, 0, width, height)
        self._points.clear()
        self._crosshair = None
        self._polygon_points.clear()
        self._stroke.clear()
        self.fit_image()

    def clear_image(self) -> None:
        self.scene().clear()
        self._image_item = None
        self._overlay_item = None
        self._image_size = None
        self._points.clear()
        self._crosshair = None
        self._polygon_points.clear()
        self._stroke.clear()

    def fit_image(self) -> None:
        if self._image_item is not None:
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_tool(self, tool: str, *, point_label: int = 1, brush_value: int = 1) -> None:
        self._tool = tool
        self._point_label = point_label
        self._brush_value = brush_value
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if tool == "pan" else Qt.CursorShape.CrossCursor
        )

    def set_points(self, points) -> None:
        self._points = [(int(x), int(y), int(label)) for x, y, label in points]
        self._redraw_marks()

    def set_overlay(self, mask: np.ndarray | None) -> None:
        if self._image_size is None:
            return
        if mask is None:
            if self._overlay_item is not None:
                self.scene().removeItem(self._overlay_item)
                self._overlay_item = None
            return
        array = np.asarray(mask, dtype=bool)
        width, height = self._image_size
        if array.shape != (height, width):
            raise ValueError("overlay mask dimensions must match the image")
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[array] = (53, 205, 117, 105)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        if self._overlay_item is None:
            self._overlay_item = self.scene().addPixmap(pixmap)
            self._overlay_item.setZValue(1)
        else:
            self._overlay_item.setPixmap(pixmap)
        self._redraw_marks()

    def set_crosshair(self, point: tuple[int, int] | None) -> None:
        self._crosshair = point
        self._redraw_marks()

    def set_view_state(self, scale: float, horizontal: int, vertical: int) -> None:
        self._syncing_view = True
        try:
            current = self.transform().m11()
            if current > 0 and scale > 0:
                self.scale(scale / current, scale / current)
            self.horizontalScrollBar().setValue(horizontal)
            self.verticalScrollBar().setValue(vertical)
        finally:
            self._syncing_view = False

    def _redraw_marks(self) -> None:
        if self._image_size is None:
            return
        for item in list(self.scene().items()):
            if item.data(0) == "mark":
                self.scene().removeItem(item)
        width, height = self._image_size
        for index, (x, y, label) in enumerate(self._points):
            color = QColor("#f44b63") if label == 1 else QColor("#46a6ff")
            pen = QPen(color, 2)
            radius = 4.0
            circle = self.scene().addEllipse(x - radius, y - radius, 2 * radius, 2 * radius, pen)
            circle.setZValue(3)
            circle.setData(0, "mark")
            text = self.scene().addText(str(index + 1))
            text.setDefaultTextColor(color)
            text.setPos(x + radius, y - radius)
            text.setZValue(3)
            text.setData(0, "mark")
        if self._crosshair is not None:
            x, y = self._crosshair
            pen = QPen(QColor(255, 232, 90, 210), 1, Qt.PenStyle.DashLine)
            for line in (
                self.scene().addLine(0, y, width, y, pen),
                self.scene().addLine(x, 0, x, height, pen),
            ):
                line.setZValue(4)
                line.setData(0, "mark")

    def _image_position(self, viewport_pos: QPoint) -> tuple[int, int] | None:
        if self._image_size is None:
            return None
        scene_pos = self.mapToScene(viewport_pos)
        x, y = int(round(scene_pos.x())), int(round(scene_pos.y()))
        width, height = self._image_size
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    def mouseMoveEvent(self, event) -> None:
        image_pos = self._image_position(event.position().toPoint())
        if image_pos:
            self.cursor_moved.emit(*image_pos)
            if self._tool in ("brush", "erase") and self._stroke:
                if image_pos != self._stroke[-1]:
                    self._stroke.append(image_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._stroke and event.button() == Qt.MouseButton.LeftButton:
            image_pos = self._image_position(event.position().toPoint())
            if image_pos and image_pos != self._stroke[-1]:
                self._stroke.append(image_pos)
            self.brush_stroke.emit(list(self._stroke), self._brush_value)
            self._stroke.clear()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if not self._syncing_view:
            self.view_changed.emit(
                self.transform().m11(),
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )

    def mouseDoubleClickEvent(self, event) -> None:
        if self._tool == "polygon" and len(self._polygon_points) >= 3:
            self.polygon_finished.emit(list(self._polygon_points))
            self._polygon_points.clear()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: F811 - handles all canvas tools
        image_pos = self._image_position(event.position().toPoint())
        if image_pos and event.button() == Qt.MouseButton.LeftButton:
            if self._tool in ("positive", "negative"):
                label = 1 if self._tool == "positive" else 0
                self.point_added.emit(image_pos[0], image_pos[1], label)
                event.accept()
                return
            if self._tool == "polygon":
                self._polygon_points.append(image_pos)
                event.accept()
                return
        if self._tool in ("brush", "erase") and event.button() == Qt.MouseButton.LeftButton and image_pos:
            self._stroke = [image_pos]
            event.accept()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0 or self._image_item is None:
            event.ignore()
            return
        current_scale = self.transform().m11()
        factor = 1.15 if delta > 0 else 1 / 1.15
        next_scale = min(64.0, max(0.05, current_scale * factor))
        factor = next_scale / current_scale
        self.scale(factor, factor)
        event.accept()
        if not self._syncing_view:
            self.view_changed.emit(
                self.transform().m11(),
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
