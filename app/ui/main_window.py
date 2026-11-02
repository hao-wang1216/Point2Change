from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import json

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from app.domain.annotation import PairAnnotation, PointPrompt
from app.paths import model_checkpoint
from app.inference.sam2_pair_predictor import PairPrediction, PairSam2Predictor
from app.io.exporter import export_annotation, load_annotation
from app.io.pair_loader import load_pair
from app.ui.pair_canvas import PairCanvas
from app.ui.dataset_session import DatasetSession


class _InferenceWorker(QObject):
    prediction_ready = Signal(int, str, object)
    prediction_failed = Signal(int, str, str)
    status_changed = Signal(int, str, str)

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor: PairSam2Predictor | None = None
        self.pair_identity: str | None = None

    @Slot(int, str, object, object, object)
    def predict(self, generation: int, pair_id: str, t1: np.ndarray, t2: np.ndarray, points: list[PointPrompt]) -> None:
        try:
            if self.predictor is None:
                self.status_changed.emit(generation, pair_id, "Loading SAM2 model…")
                self.predictor = PairSam2Predictor(self.checkpoint_path, self.device)
            if pair_id != self.pair_identity:
                self.predictor.set_pair(t1, t2)
                self.pair_identity = pair_id
            prediction = self.predictor.predict_change(points)
            self.prediction_ready.emit(generation, pair_id, prediction)
        except Exception as error:  # keep the GUI alive and preserve current edits
            self.prediction_failed.emit(generation, pair_id, f"{type(error).__name__}: {error}")


class MainWindow(QMainWindow):
    request_prediction = Signal(int, str, object, object, object)

    def __init__(self, *, enable_inference: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Point2Change")
        self.resize(1500, 850)
        self.annotation: PairAnnotation | None = None
        self._t1_rgb: np.ndarray | None = None
        self._t2_rgb: np.ndarray | None = None
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._generation = 0
        self._pair_id = ""
        self._active_request: tuple[int, str] | None = None
        self._queued_request: tuple | None = None
        self._enable_inference = enable_inference
        self._active_tool = "positive"
        self._dirty = False
        self._standalone_output: Path | None = None
        self._save_message = "Open a dataset to enable autosave"
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(900)
        self._autosave_timer.timeout.connect(self._autosave)

        self.left_canvas = PairCanvas(self)
        self.right_canvas = PairCanvas(self)
        self.left_label = QLabel("T1 · Time 1")
        self.right_label = QLabel("T2 · Time 2")
        for label in (self.left_label, self.right_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-weight: 600; padding: 4px")

        split = QSplitter(Qt.Orientation.Horizontal, self)
        left_panel, right_panel = QWidget(), QWidget()
        left_layout, right_layout = QVBoxLayout(left_panel), QVBoxLayout(right_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.addWidget(self.left_label)
        left_layout.addWidget(self.left_canvas)
        right_layout.addWidget(self.right_label)
        right_layout.addWidget(self.right_canvas)
        split.addWidget(left_panel)
        split.addWidget(right_panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        center = QWidget()
        layout = QHBoxLayout(center)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(split)
        self.setCentralWidget(center)
        self.dataset_session = DatasetSession(self)
        self._make_toolbar()
        self.statusBar().showMessage("Open a pair of registered RGB images of the same size to start annotating")

        self.left_canvas.point_added.connect(self._on_point_added)
        self.right_canvas.point_added.connect(self._on_point_added)
        self.left_canvas.cursor_moved.connect(self._sync_crosshair)
        self.right_canvas.cursor_moved.connect(self._sync_crosshair)
        self.left_canvas.view_changed.connect(self._sync_view_from_left)
        self.right_canvas.view_changed.connect(self._sync_view_from_right)
        self.left_canvas.polygon_finished.connect(self._apply_polygon)
        self.right_canvas.polygon_finished.connect(self._apply_polygon)
        self.left_canvas.brush_stroke.connect(self.apply_manual_stroke)
        self.right_canvas.brush_stroke.connect(self.apply_manual_stroke)

        self._thread: QThread | None = None
        self._worker: _InferenceWorker | None = None
        if enable_inference:
            checkpoint = model_checkpoint()
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            self._thread = QThread(self)
            self._worker = _InferenceWorker(checkpoint, device)
            self._worker.moveToThread(self._thread)
            self.request_prediction.connect(self._worker.predict, Qt.ConnectionType.QueuedConnection)
            self._worker.prediction_ready.connect(self._on_prediction_ready)
            self._worker.prediction_failed.connect(self._on_prediction_failed)
            self._worker.status_changed.connect(self._on_prediction_status)
            self._thread.start()

    def _make_toolbar(self) -> None:
        toolbar = QToolBar("Annotation Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._add_action(toolbar, "Open Pair", self.open_pair_dialog)
        self._add_action(toolbar, "Open Annotation", self.open_annotation_dialog)
        toolbar.addSeparator()
        group = QActionGroup(self)
        group.setExclusive(True)
        for text, tool, label, value in (
            ("Positive Point", "positive", 1, 1),
            ("Negative Point", "negative", 0, 0),
            ("Polygon", "polygon", 1, 1),
            ("Brush", "brush", 1, 1),
            ("Erase", "erase", 1, 0),
            ("Pan", "pan", 1, 1),
        ):
            action = QAction(text, self, checkable=True)
            action.triggered.connect(
                lambda checked=False, selected=tool, point_label=label, brush_value=value:
                self.set_active_tool(selected, point_label, brush_value)
            )
            group.addAction(action)
            toolbar.addAction(action)
            if tool == "positive":
                action.setChecked(True)
        toolbar.addSeparator()
        self._add_action(toolbar, "Delete Last Point", self.delete_last_point)
        self._add_action(toolbar, "Undo", self.undo, "Ctrl+Z")
        self._add_action(toolbar, "Redo", self.redo, "Ctrl+Y")
        self._add_action(toolbar, "Cancel Outline", self.cancel_drawing, "Esc")
        toolbar.addSeparator()
        self._add_action(toolbar, "Export", self.export_dialog)
        toolbar.addWidget(QLabel("  Threshold "))
        self.threshold_control = QDoubleSpinBox()
        self.threshold_control.setRange(0.0, 1.0)
        self.threshold_control.setSingleStep(0.05)
        self.threshold_control.setValue(0.5)
        self.threshold_control.valueChanged.connect(self._on_threshold_changed)
        toolbar.addWidget(self.threshold_control)

    def _add_action(self, toolbar: QToolBar, label: str, callback, shortcut: str | None = None) -> QAction:
        action = QAction(label, self)
        action.triggered.connect(lambda checked=False: callback())
        if shortcut:
            action.setShortcut(shortcut)
            action.setToolTip(f"{label} ({shortcut})")
        toolbar.addAction(action)
        return action

    def set_active_tool(self, tool: str, point_label: int = 1, brush_value: int = 1) -> None:
        self._active_tool = tool
        self.left_canvas.set_tool(tool, point_label=point_label, brush_value=brush_value)
        self.right_canvas.set_tool(tool, point_label=point_label, brush_value=brush_value)

    def set_pair_arrays(
        self,
        t1_rgb: np.ndarray,
        t2_rgb: np.ndarray,
        t1_path: str | Path,
        t2_path: str | Path,
        *,
        annotation: PairAnnotation | None = None,
    ) -> None:
        t1 = np.ascontiguousarray(t1_rgb, dtype=np.uint8)
        t2 = np.ascontiguousarray(t2_rgb, dtype=np.uint8)
        self.validate_pair_arrays(t1, t2, annotation)
        self._autosave_timer.stop()
        self._invalidate_prediction()
        self._pair_id = uuid4().hex
        self._t1_rgb, self._t2_rgb = t1, t2
        self.annotation = annotation if annotation is not None else PairAnnotation(
            Path(t1_path),
            Path(t2_path),
            manual_override_mask=np.full(t1.shape[:2], -1, dtype=np.int8),
        )
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._save_message = "Annotation restored" if annotation is not None else "Not annotated"
        with QSignalBlocker(self.threshold_control):
            self.threshold_control.setValue(self.annotation.threshold)
        self.left_canvas.set_image(t1)
        self.right_canvas.set_image(t2)
        self._render_state()
        self.left_label.setText(f"T1 · {Path(t1_path).name}")
        self.right_label.setText(f"T2 · {Path(t2_path).name}")
        self.statusBar().showMessage(f"Loaded image pair: {t1.shape[1]}×{t1.shape[0]}")
        self.dataset_session.refresh()
        if self.annotation.prediction_pending:
            self._request_prediction()

    @staticmethod
    def validate_pair_arrays(t1: np.ndarray, t2: np.ndarray, annotation: PairAnnotation | None = None) -> None:
        if t1.shape != t2.shape or t1.ndim != 3 or t1.shape[2] != 3:
            raise ValueError("T1 and T2 must be same-size HWC RGB images")
        if annotation is not None:
            for mask in (annotation.soft_mask, annotation.manual_override_mask):
                if mask is not None and mask.shape != t1.shape[:2]:
                    raise ValueError("saved annotation dimensions do not match the image pair")
            height, width = t1.shape[:2]
            if any(not (0 <= p.x < width and 0 <= p.y < height) for p in annotation.points):
                raise ValueError("saved annotation point is outside the image")

    def open_pair_dialog(self) -> None:
        t1_path, _ = QFileDialog.getOpenFileName(
            self, "Select T1 Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if not t1_path:
            return
        t2_path, _ = QFileDialog.getOpenFileName(
            self, "Select T2 Image", str(Path(t1_path).parent), "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if not t2_path:
            return
        try:
            t1, t2 = load_pair(t1_path, t2_path)
            if not self.save_before_leaving():
                return
            self.dataset_session.clear()
            self._standalone_output = None
            self.set_pair_arrays(t1, t2, t1_path, t2_path)
        except Exception as error:
            QMessageBox.critical(self, "Load Failed", str(error))

    def open_annotation_dialog(self) -> None:
        metadata_path, _ = QFileDialog.getOpenFileName(self, "Open Annotation JSON", "", "JSON (*.json)")
        if not metadata_path:
            return
        try:
            metadata_path = Path(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "dataset_annotation_version" in metadata:
                self.dataset_session.open_annotation_file(metadata_path, metadata)
                return
            annotation = load_annotation(metadata_path)
            t1, t2 = load_pair(annotation.t1_path, annotation.t2_path)
            self.validate_pair_arrays(t1, t2, annotation)
            if not self.save_before_leaving():
                return
            self.dataset_session.clear()
            # Open-dataset manages its own commit metadata. Standalone export uses
            # the existing flat PNG/JSON convention, never the dataset pointer.
            self._standalone_output = None
            self.set_pair_arrays(t1, t2, annotation.t1_path, annotation.t2_path, annotation=annotation)
        except Exception as error:
            self.report_error("Failed to Open Annotation", str(error))

    def export_dialog(self) -> None:
        if self.annotation is None:
            QMessageBox.information(self, "No Images Loaded", "Please open a T1/T2 image pair first.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not output_dir:
            return
        try:
            result = export_annotation(self.annotation, output_dir, model_id="sam2_hiera_large")
            if self.dataset_session.index is None:
                self._standalone_output = Path(output_dir)
                self._dirty = False
                self._save_message = "Saved"
                self.dataset_session.refresh()
            self.statusBar().showMessage(f"Masks and annotation exported: {result.metadata}", 8000)
        except Exception as error:
            QMessageBox.critical(self, "Export Failed", str(error))

    def _snapshot(self) -> dict:
        return {
            "points": list(self.annotation.points),
            "manual": None
            if self.annotation.manual_override_mask is None
            else self.annotation.manual_override_mask.copy(),
            "soft": None if self.annotation.soft_mask is None else self.annotation.soft_mask.copy(),
            "pending": self.annotation.prediction_pending,
            "threshold": self.annotation.threshold,
            "review_status": self.annotation.review_status,
        }

    def _save_history(self) -> None:
        if self.annotation is None:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_snapshot(self, snapshot: dict) -> None:
        self.annotation.points = list(snapshot["points"])
        self.annotation.manual_override_mask = (
            None if snapshot["manual"] is None else snapshot["manual"].copy()
        )
        self._invalidate_prediction()
        self.annotation.soft_mask = None if snapshot["soft"] is None else snapshot["soft"].copy()
        self.annotation.prediction_pending = snapshot["pending"]
        self.annotation.threshold = snapshot["threshold"]
        self.annotation.review_status = snapshot["review_status"]
        with QSignalBlocker(self.threshold_control):
            self.threshold_control.setValue(self.annotation.threshold)
        self._render_state()
        self._mark_dirty(demote_review=False)
        if self.annotation.prediction_pending:
            self._request_prediction()

    def undo(self) -> bool:
        if not self._undo_stack or self.annotation is None:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore_snapshot(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack or self.annotation is None:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore_snapshot(self._redo_stack.pop())
        return True

    def _on_point_added(self, x: int, y: int, label: int) -> None:
        if self.annotation is None:
            return
        self._save_history()
        self.annotation.points.append(PointPrompt(x, y, label))
        self._mark_dirty()
        self._render_state()
        self._request_prediction()

    def delete_last_point(self) -> None:
        if self.annotation is None or not self.annotation.points:
            return
        self._save_history()
        self.annotation.points.pop()
        self._mark_dirty()
        self._render_state()
        self._request_prediction()

    def apply_manual_stroke(self, points, value: int) -> None:
        if self.annotation is None or not points:
            return
        if value not in (0, 1):
            raise ValueError("manual edit value must be 0 or 1")
        self._save_history()
        override = self.annotation.manual_override_mask
        if override is None:
            override = np.full(self._t1_rgb.shape[:2], -1, dtype=np.int8)
            self.annotation.manual_override_mask = override
        height, width = override.shape
        for x, y in points:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4 and 0 <= x + dx < width and 0 <= y + dy < height:
                        override[y + dy, x + dx] = value
        self._mark_dirty()
        self._render_state()

    def _apply_polygon(self, points) -> None:
        if self.annotation is None or len(points) < 3:
            return
        self._save_history()
        override = self.annotation.manual_override_mask
        height, width = self._t1_rgb.shape[:2]
        if override is None:
            override = np.full((height, width), -1, dtype=np.int8)
            self.annotation.manual_override_mask = override
        from PIL import Image, ImageDraw

        fill_value = 0 if self._active_tool == "erase" else 1
        layer = Image.new("1", (width, height), 0)
        ImageDraw.Draw(layer).polygon(points, fill=1)
        selected = np.asarray(layer, dtype=bool)
        override[selected] = fill_value
        self._mark_dirty()
        self._render_state()

    def _effective_mask(self) -> np.ndarray | None:
        if self.annotation is None or self._t1_rgb is None:
            return None
        if self.annotation.soft_mask is None:
            result = np.zeros(self._t1_rgb.shape[:2], dtype=np.uint8)
        else:
            if self.annotation.soft_mask.shape != self._t1_rgb.shape[:2]:
                return None
            result = (self.annotation.soft_mask >= self.annotation.threshold).astype(np.uint8)
        override = self.annotation.manual_override_mask
        if override is not None:
            result[override == 0] = 0
            result[override == 1] = 1
        return result

    def _render_state(self) -> None:
        if self.annotation is None:
            return
        points = [(point.x, point.y, point.label) for point in self.annotation.points]
        self.left_canvas.set_points(points)
        self.right_canvas.set_points(points)
        overlay = self._effective_mask()
        self.left_canvas.set_overlay(overlay)
        self.right_canvas.set_overlay(overlay)

    def _sync_crosshair(self, x: int, y: int) -> None:
        self.left_canvas.set_crosshair((x, y))
        self.right_canvas.set_crosshair((x, y))

    def _sync_view_from_left(self, scale: float, horizontal: int, vertical: int) -> None:
        self.right_canvas.set_view_state(scale, horizontal, vertical)

    def _sync_view_from_right(self, scale: float, horizontal: int, vertical: int) -> None:
        self.left_canvas.set_view_state(scale, horizontal, vertical)

    def _on_threshold_changed(self, value: float) -> None:
        if self.annotation is not None:
            self._save_history()
            self.annotation.threshold = float(value)
            self._mark_dirty()
            self._render_state()

    def _request_prediction(self) -> None:
        if self.annotation is None or self._t1_rgb is None:
            return
        self._invalidate_prediction()
        if not any(point.label == 1 for point in self.annotation.points):
            self.annotation.soft_mask = None
            self.annotation.prediction_pending = False
            self._render_state()
            self.dataset_session.refresh()
            self.statusBar().showMessage("Add a positive point to generate a candidate change mask")
            return
        self.annotation.prediction_pending = True
        self._mark_dirty()
        if not self._enable_inference:
            return
        self.statusBar().showMessage("Updating change mask from point prompts…")
        self._queued_request = (
            self._generation,
            self._pair_id,
            self._t1_rgb,
            self._t2_rgb,
            list(self.annotation.points),
        )
        self._start_queued_prediction()

    def _invalidate_prediction(self) -> None:
        self._generation += 1
        self._queued_request = None

    def _start_queued_prediction(self) -> None:
        if self._active_request is None and self._queued_request is not None:
            request = self._queued_request
            self._queued_request = None
            self._active_request = request[:2]
            self.request_prediction.emit(*request)

    def _finish_request(self, generation: int, pair_id: str) -> None:
        if self._active_request == (generation, pair_id):
            self._active_request = None
        self._start_queued_prediction()

    @Slot(int, str, object)
    def _on_prediction_ready(self, generation: int, pair_id: str, prediction: PairPrediction) -> None:
        try:
            if (generation, pair_id) != (self._generation, self._pair_id) or self.annotation is None:
                return
            if prediction.soft_mask.shape != self._t1_rgb.shape[:2]:
                self._on_prediction_failed(generation, pair_id, "Predicted mask dimensions do not match the current images")
                return
            self.annotation.soft_mask = prediction.soft_mask
            self.annotation.prediction_pending = False
            self._mark_dirty(demote_review=False)
            self._render_state()
            self.statusBar().showMessage(
                f"Mask updated · Source {prediction.source_time} · Score {prediction.score:.3f}", 5000,
            )
        finally:
            self._finish_request(generation, pair_id)

    @Slot(int, str, str)
    def _on_prediction_failed(self, generation: int, pair_id: str, message: str) -> None:
        if (generation, pair_id) == (self._generation, self._pair_id):
            # Keep pending=True: saved prompts must be recomputed on reopen, and
            # an old mask must not be certified as the result of these prompts.
            self.statusBar().showMessage(f"SAM2 inference failed; existing annotations preserved: {message}", 12000)
        self._finish_request(generation, pair_id)

    @Slot(int, str, str)
    def _on_prediction_status(self, generation: int, pair_id: str, message: str) -> None:
        if (generation, pair_id) == (self._generation, self._pair_id):
            self.statusBar().showMessage(message)

    def _mark_dirty(self, *, demote_review: bool = True) -> None:
        if self.annotation is None:
            return
        if demote_review and self.annotation.review_status != "needs_review":
            self.annotation.review_status = "draft"
        self._dirty = True
        self._save_message = "Unsaved changes"
        self.dataset_session.refresh()
        if self.dataset_session.index is not None:
            self._autosave_timer.start()

    def _autosave(self) -> None:
        if self._dirty and self.dataset_session.index is not None:
            self.save_current(show_error=False)

    def report_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def save_current(self, *, show_error: bool = True) -> bool:
        if self.annotation is None:
            return True
        self._autosave_timer.stop()
        try:
            if self.dataset_session.index is not None:
                self.dataset_session.save()
            else:
                if self._standalone_output is None:
                    folder = QFileDialog.getExistingDirectory(self, "Select Annotation Folder")
                    if not folder:
                        return False
                    self._standalone_output = Path(folder)
                export_annotation(self.annotation, self._standalone_output, model_id="sam2_hiera_large")
        except Exception as error:
            self._save_message = "Save failed; edits remain in the current view"
            self.statusBar().showMessage(f"Save failed: {error}")
            self.dataset_session.refresh()
            if show_error:
                self.report_error("Save Failed", str(error))
            return False
        self._dirty = False
        self._save_message = "Saved · Inference pending" if self.annotation.prediction_pending else "Saved"
        self.dataset_session.refresh()
        return True

    def save_before_leaving(self) -> bool:
        if self.left_canvas.has_unfinished_edit or self.right_canvas.has_unfinished_edit:
            self.report_error("Unfinished Outline", "Double-click to finish the polygon or press Esc to cancel before switching or closing.")
            return False
        if not self._dirty:
            return True
        return self.save_current()

    def cancel_drawing(self) -> None:
        self.left_canvas.cancel_drawing()
        self.right_canvas.cancel_drawing()
        self.statusBar().showMessage("Unfinished outline canceled", 3000)

    def set_review_status(self, status: str) -> bool:
        if self.annotation is None:
            return False
        if self.left_canvas.has_unfinished_edit or self.right_canvas.has_unfinished_edit:
            self.report_error("Unfinished Outline", "Finish the outline or press Esc to cancel.")
            return False
        if status not in ("reviewed", "no_change", "needs_review", "draft"):
            raise ValueError(f"unknown review status: {status}")
        if status == "reviewed":
            if self.annotation.prediction_pending:
                self.report_error("Review Not Available", "Please wait for inference to finish. If it fails, edit the prompt points to retry.")
                return False
            mask = self._effective_mask()
            if mask is None or not mask.any():
                self.report_error("Empty Mask", "If you have checked both images and found no changes, use Confirm No Change.")
                return False
        self._save_history()
        if status == "no_change":
            # An explicit no-change decision overrides even an in-flight prediction.
            # Keep an undo snapshot so an accidental click can be reversed.
            self._invalidate_prediction()
            self.annotation.points.clear()
            self.annotation.soft_mask = np.zeros(self._t1_rgb.shape[:2], dtype=np.float32)
            # Force background also at a threshold of zero.
            self.annotation.manual_override_mask = np.zeros(self._t1_rgb.shape[:2], dtype=np.int8)
            self.annotation.prediction_pending = False
            self._render_state()
        self.annotation.review_status = status
        self._mark_dirty(demote_review=False)
        return self.save_current()

    def closeEvent(self, event) -> None:
        if not self.save_before_leaving():
            event.ignore()
            return
        self._autosave_timer.stop()
        self._invalidate_prediction()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)
