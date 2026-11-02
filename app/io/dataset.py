"""Discover image pairs without guessing correspondence between two dates."""

from __future__ import annotations

import os
import re
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


@dataclass(frozen=True)
class DatasetPair:
    key: str
    t1_path: Path
    t2_path: Path
    image_size: tuple[int, int]


@dataclass(frozen=True)
class PairIssue:
    key: str
    message: str


@dataclass
class DatasetIndex:
    root: Path
    pairs: list[DatasetPair]
    issues: list[PairIssue]


def _natural_key(value: str) -> tuple:
    chunks = tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    )
    return chunks + ((2, value.casefold()), (3, value))


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _scan_images(directory: Path, label: str) -> tuple[dict[str, list[tuple[str, Path]]], set[str], list[PairIssue]]:
    by_name: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    by_stem: dict[str, list[str]] = defaultdict(list)
    issues: list[PairIssue] = []

    def walk_error(error: OSError) -> None:
        issues.append(PairIssue(str(error.filename or label), f"Cannot read {label} directory: {error}"))

    for current, directories, files in os.walk(directory, followlinks=False, onerror=walk_error):
        current_path = Path(current)
        retained = []
        for name in sorted(directories, key=_natural_key):
            child = current_path / name
            key = child.relative_to(directory).as_posix()
            try:
                directory_alias = child.is_symlink() or bool(
                    getattr(child.lstat(), "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
            except OSError as error:
                issues.append(PairIssue(key, f"Cannot read {label} subdirectory: {error}"))
                continue
            if not _inside(child, directory):
                issues.append(PairIssue(key, f"Skipped {label} subdirectory outside the image directory"))
            elif directory_alias:
                # Do not create duplicate pairs through directory aliases or loops.
                issues.append(PairIssue(key, f"Skipped symbolic-link directory in {label}"))
            else:
                retained.append(name)
        directories[:] = retained
        for name in sorted(files, key=_natural_key):
            path = current_path / name
            if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            key = path.relative_to(directory).as_posix()
            if not _inside(path, directory):
                issues.append(PairIssue(key, f"Skipped {label} file outside the image directory"))
                continue
            normalized = key.casefold()
            by_name[normalized].append((key, path))
            by_stem[str(Path(key).with_suffix("")).replace("\\", "/").casefold()].append(normalized)

    ambiguous: set[str] = set()
    for entries in by_stem.values():
        if len(entries) > 1:
            ambiguous.update(entries)
    return dict(by_name), ambiguous, issues


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()  # Checking headers alone misses truncated or otherwise unreadable images.
        return image.size


def build_dataset_index(root: str | Path) -> DatasetIndex:
    """Match complete relative names, case-insensitively, below ``T1`` and ``T2``.

    Ambiguous names, unreadable files, missing partners and mismatched sizes are
    reported individually; they never shift the correspondence of valid pairs.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    folders = [root / "T1", root / "T2"]
    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Dataset must contain T1 and T2 subdirectories; missing: {folder.name}")
        if not _inside(folder, root):
            raise ValueError(f"{folder.name} must not point outside the dataset directory")
    t1, t1_ambiguous, t1_issues = _scan_images(folders[0].resolve(), "T1")
    t2, t2_ambiguous, t2_issues = _scan_images(folders[1].resolve(), "T2")
    pairs: list[DatasetPair] = []
    issues = t1_issues + t2_issues
    for normalized in sorted(t1.keys() | t2.keys(), key=_natural_key):
        left, right = t1.get(normalized, []), t2.get(normalized, [])
        key = (left or right)[0][0]
        ambiguous_labels = [
            label for label, duplicates in (("T1", t1_ambiguous), ("T2", t2_ambiguous))
            if normalized in duplicates
        ]
        if ambiguous_labels:
            issues.append(PairIssue(key, f"{', '.join(ambiguous_labels)} contains multiple images with the same name or stem in the same directory; pairing is ambiguous"))
            continue
        if not left or not right:
            issues.append(PairIssue(key, f"Missing image with the same relative filename in {'T1' if not left else 'T2'}"))
            continue
        sizes = []
        invalid = False
        for label, entry in (("T1", left[0]), ("T2", right[0])):
            try:
                sizes.append(_image_size(entry[1]))
            except Exception as error:
                issues.append(PairIssue(key, f"Cannot read {label} image: {error}"))
                invalid = True
        if invalid:
            continue
        if sizes[0] != sizes[1]:
            issues.append(PairIssue(key, f"Image dimensions differ: T1={sizes[0][0]}×{sizes[0][1]}, T2={sizes[1][0]}×{sizes[1][1]}"))
            continue
        pairs.append(DatasetPair(key, left[0][1], right[0][1], sizes[0]))
    issues.sort(key=lambda issue: (_natural_key(issue.key), issue.message))
    return DatasetIndex(root, pairs, issues)
