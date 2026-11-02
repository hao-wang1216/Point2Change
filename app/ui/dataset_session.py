from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QToolBar, QVBoxLayout, QWidget,
)

from app.domain.annotation import REVIEW_LABELS
from app.io.dataset import DatasetIndex, build_dataset_index
from app.io.dataset_store import DatasetStore
from app.io.pair_loader import load_pair

if TYPE_CHECKING:
    from app.ui.main_window import MainWindow


class DatasetSession:
    """Dataset navigation and persistence; every entry represents a complete T1/T2 pair."""

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.index: DatasetIndex | None = None
        self.store: DatasetStore | None = None
        self.current_index = -1
        self.statuses: dict[str, str] = {}
        self._items: list[QListWidgetItem] = []
        self._make_ui()

    def _make_ui(self) -> None:
        w = self.window
        toolbar = QToolBar("Dataset Navigation", w)
        toolbar.setMovable(False)
        w.addToolBar(toolbar)
        w._add_action(toolbar, "Open Dataset", self.open_dialog, "Ctrl+U")
        self.previous_action = w._add_action(toolbar, "Previous", self.previous, "A")
        self.next_action = w._add_action(toolbar, "Next", self.next, "D")
        self.save_action = w._add_action(toolbar, "Save", w.save_current, "Ctrl+S")
        self.complete_action = w._add_action(toolbar, "Complete & Next", self.complete_and_next, "Ctrl+Return")
        toolbar.addSeparator()
        self.position_label = QLabel("  No dataset open  ")
        toolbar.addWidget(self.position_label)
        w.addToolBarBreak()

        self.dock = QDockWidget("Samples & Review", w)
        self.dock.setObjectName("dataset_samples")
        self.dock.setMinimumWidth(245)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.root_label = QLabel("Open the parent folder containing T1 / T2")
        self.root_label.setWordWrap(True)
        layout.addWidget(self.root_label)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select_row)
        layout.addWidget(self.list, 1)
        self.progress_label = QLabel("No samples")
        layout.addWidget(self.progress_label)
        self.review_label = QLabel("Status: Not started")
        layout.addWidget(self.review_label)
        self.save_label = QLabel("Autosave: enabled in dataset mode")
        layout.addWidget(self.save_label)
        self.review_buttons = []
        for label, status in (
            ("Mark Reviewed", "reviewed"),
            ("Confirm No Change", "no_change"),
            ("Needs Review", "needs_review"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, s=status: w.set_review_status(s))
            layout.addWidget(button)
            self.review_buttons.append(button)
        self.issues_button = QPushButton("View Pairing Report")
        self.issues_button.clicked.connect(self.show_issues)
        layout.addWidget(self.issues_button)
        self.dock.setWidget(container)
        w.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.refresh()

    def open_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self.window, "Select Dataset Folder Containing T1 and T2",
            str(self.index.root) if self.index else "",
        )
        if path:
            self.open_dataset(path, show_issues=True)

    def open_dataset(self, root: str | Path, *, show_issues: bool = False, sample_key: str | None = None) -> bool:
        w = self.window
        try:
            index = build_dataset_index(root)
            if not index.pairs:
                details = "\n".join(f"{i.key}: {i.message}" for i in index.issues)
                raise ValueError("No valid T1/T2 image pairs found.\n" + details)
            store = DatasetStore(index.root)
            statuses = {}
            for pair in index.pairs:
                try:
                    statuses[pair.key] = store.read_status(pair)
                except Exception:
                    statuses[pair.key] = "invalid"
            last = sample_key if sample_key is not None else store.read_last_key()
            if sample_key is not None and not any(p.key == sample_key for p in index.pairs):
                raise ValueError(f"The annotated image pair failed pairing checks: {sample_key}")
            start = next((i for i, p in enumerate(index.pairs) if p.key == last), 0)
            pair = index.pairs[start]
            # Validate the destination before saving or replacing the current view.
            t1, t2 = load_pair(pair.t1_path, pair.t2_path)
            annotation = store.load(pair)
            w.validate_pair_arrays(t1, t2, annotation)
            if not w.save_before_leaving():
                return False
            # Reopening the current dataset must use the just-saved snapshot.
            annotation = store.load(pair)
            w.validate_pair_arrays(t1, t2, annotation)
            store.write_last_key(pair.key)
        except Exception as error:
            w.report_error("Failed to Open Dataset", str(error))
            return False

        self.index, self.store = index, store
        self.statuses = statuses
        self.current_index = start
        w._standalone_output = None
        self._populate()
        self.root_label.setText(index.root.name)
        self.root_label.setToolTip(str(index.root))
        w.set_pair_arrays(t1, t2, pair.t1_path, pair.t2_path, annotation=annotation)
        self.refresh()
        w.statusBar().showMessage(f"Opened {len(index.pairs)} image pairs · {len(index.issues)} pairing issues", 8000)
        if show_issues and index.issues:
            self.show_issues()
        return True

    def open_annotation_file(self, path: Path, metadata: dict) -> bool:
        key = metadata.get("pair_key")
        if not isinstance(key, str) or not key or "\\" in key or ":" in key:
            raise ValueError("Dataset annotation has no valid sample name")
        parts = PurePosixPath(key).parts
        if PurePosixPath(key).is_absolute() or any(part in (".", "..") for part in parts):
            raise ValueError("Invalid sample path in dataset annotation")
        path = path.resolve()
        for root in path.parents:
            expected = root / "annotations" / key / "annotation.json"
            if expected == path and (root / "T1").is_dir() and (root / "T2").is_dir():
                return self.open_dataset(root, sample_key=key)
        raise ValueError("Open annotations/<sample>/annotation.json in the dataset, or open the parent folder containing T1/T2.")

    def clear(self) -> None:
        self.index = None
        self.store = None
        self.current_index = -1
        self.statuses.clear()
        self._populate()
        self.root_label.setText("Single-pair mode; open a dataset to annotate multiple pairs")
        self.refresh()

    def _populate(self) -> None:
        with QSignalBlocker(self.list):
            self.list.clear()
            self._items.clear()
            if self.index:
                for pair in self.index.pairs:
                    item = QListWidgetItem()
                    item.setToolTip(f"T1: {pair.t1_path}\nT2: {pair.t2_path}")
                    self.list.addItem(item)
                    self._items.append(item)
                self.list.setCurrentRow(self.current_index)

    def _select_row(self, row: int) -> None:
        if row >= 0 and not self.switch_pair(row):
            with QSignalBlocker(self.list):
                self.list.setCurrentRow(self.current_index)

    def switch_pair(self, index: int) -> bool:
        if self.index is None or not 0 <= index < len(self.index.pairs):
            return False
        if index == self.current_index:
            return True
        pair = self.index.pairs[index]
        w = self.window
        try:
            t1, t2 = load_pair(pair.t1_path, pair.t2_path)
            if (t1.shape[1], t1.shape[0]) != pair.image_size:
                raise ValueError("Image dimensions have changed. Reopen the dataset to check pairing.")
            annotation = self.store.load(pair)
            w.validate_pair_arrays(t1, t2, annotation)
            if not w.save_before_leaving():
                return False
            self.store.write_last_key(pair.key)
        except Exception as error:
            w.report_error("Failed to Switch Image Pair", str(error))
            return False
        self.current_index = index
        w.set_pair_arrays(t1, t2, pair.t1_path, pair.t2_path, annotation=annotation)
        self.refresh()
        return True

    def previous(self) -> bool:
        return self.switch_pair(self.current_index - 1)

    def next(self) -> bool:
        return self.switch_pair(self.current_index + 1)

    def complete_and_next(self) -> None:
        annotation = self.window.annotation
        status = "no_change" if annotation and annotation.review_status == "no_change" else "reviewed"
        saved = self.window.save_current() if status == "no_change" else self.window.set_review_status(status)
        if saved and self.index:
            if self.current_index + 1 < len(self.index.pairs):
                self.next()

    def save(self) -> Path:
        if self.index is None or self.store is None or self.current_index < 0:
            raise RuntimeError("No dataset open")
        pair = self.index.pairs[self.current_index]
        path = self.store.save(pair, self.window.annotation)
        self.statuses[pair.key] = self.window.annotation.review_status
        return path

    def refresh(self) -> None:
        annotation = self.window.annotation
        count = len(self.index.pairs) if self.index else 0
        self.previous_action.setEnabled(self.current_index > 0)
        self.next_action.setEnabled(0 <= self.current_index < count - 1)
        self.save_action.setEnabled(annotation is not None)
        self.complete_action.setEnabled(annotation is not None and count > 0)
        for button in self.review_buttons:
            button.setEnabled(annotation is not None)
        self.issues_button.setEnabled(self.index is not None)
        self.position_label.setText(
            f"  Pair {self.current_index + 1} / {count}  " if count else "  Single-pair mode  "
        )
        if self.index:
            if annotation is not None and 0 <= self.current_index < count:
                self.statuses[self.index.pairs[self.current_index].key] = annotation.review_status
            for i, (pair, item) in enumerate(zip(self.index.pairs, self._items)):
                status = self.statuses.get(pair.key, "unstarted")
                label = REVIEW_LABELS.get(status, "Annotation error")
                dirty = " *" if i == self.current_index and self.window._dirty else ""
                item.setText(f"{pair.key}  ·  {label}{dirty}")
            with QSignalBlocker(self.list):
                self.list.setCurrentRow(self.current_index)
            reviewed = sum(s in ("reviewed", "no_change") for s in self.statuses.values())
            self.progress_label.setText(f"Reviewed {reviewed} / {count} · Pairing issues {len(self.index.issues)}")
        else:
            self.progress_label.setText("No dataset")
        status = annotation.review_status if annotation else "unstarted"
        pending = " · Inference pending" if annotation and annotation.prediction_pending else ""
        self.review_label.setText(f"Status: {REVIEW_LABELS[status]}{pending}")
        self.save_label.setText(self.window._save_message)

    def show_issues(self) -> None:
        if self.index is None:
            return
        dialog = QMessageBox(self.window)
        dialog.setWindowTitle("Pairing Report")
        dialog.setText(f"Valid pairs: {len(self.index.pairs)}; Issues: {len(self.index.issues)}")
        if self.index.issues:
            dialog.setInformativeText("Invalid samples are excluded from the list. Expand details to see the reasons.")
            dialog.setDetailedText("\n".join(f"{i.key}: {i.message}" for i in self.index.issues))
        dialog.exec()
