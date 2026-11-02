# Point2Change

**Turn point prompts into change masks for bi-temporal remote sensing imagery with SAM 2, reducing manual annotation effort.**

Point2Change is a local Python desktop application for labeling changes between
two registered RGB images. Click a changed object in either image to generate
candidate masks with SAM 2. The application ranks candidates using a change
response map and combines them into a change mask that you can refine manually.

## Features

- Positive point prompts generate change-mask candidates; negative points correct unwanted regions.
- Synchronized T1/T2 views support paired-image inspection.
- Brush, polygon, and eraser tools refine masks, with undo/redo and threshold adjustment.
- Dataset browsing pairs images by relative filename and supports nested folders.
- Automatic saving preserves prompts, soft masks, manual edits, and review status.
- Explicit review states distinguish drafts, reviewed samples, confirmed no-change samples, and samples needing review.
- Export binary and soft masks for downstream use.

Predictions are annotation suggestions and should be reviewed. This project
does not claim a measured reduction in annotation time or a benchmark accuracy.

## Requirements

- Python 3.10–3.12; Python 3.11 is a good starting point for a new environment.
- A desktop environment capable of running PySide6/Qt.
- PyTorch and TorchVision versions within the ranges in `requirements.txt`.
- The original **SAM 2 Hiera Large** checkpoint, downloaded separately.

CUDA is selected automatically when available; otherwise inference uses the CPU.
CPU inference can be slow with the large model. The project was developed on
Windows; other desktop platforms have not been verified for this release.

## Quick start

### 1. Clone the repository

```shell
git clone https://github.com/hao-wang1216/Point2Change.git
cd Point2Change
```

### 2. Create an environment and install dependencies

Using Conda:

```shell
conda create -n point2change python=3.11 -y
conda activate point2change
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, use a Python virtual environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA-specific PyTorch installation, use the
[official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
to install a matching `torch`/`torchvision` pair compatible with your driver and
the version ranges in `requirements.txt`, then install the remaining requirements.

The SAM 2 runtime is included in this repository. Run from the repository root;
a separate `pip install sam2` is not required. The image annotation workflow
uses the default predictor settings without the optional compiled CUDA extension.
Precompiled extensions are not distributed here.

### 3. Download the model checkpoint

Download [the original SAM 2 Hiera Large checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)
and place it at:

```text
Point2Change/
└── models/
    └── sam2_hiera_large.pt
```

For example, from the repository root in Windows PowerShell:

```powershell
curl.exe -L --fail --output models/sam2_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
```

Use the original **SAM 2** checkpoint, not a SAM 2.1 checkpoint. The bundled
configuration is `sam2_configs/sam2_hiera_l.yaml`.

To keep the checkpoint elsewhere, set its path before starting the application:

```powershell
$env:CHANGE_LABELER_MODEL = "D:\Models\sam2_hiera_large.pt"
```

On Linux/macOS shells, the equivalent is:

```shell
export CHANGE_LABELER_MODEL="/path/to/sam2_hiera_large.pt"
```

`CHANGE_LABELER_MODEL` retains its original name for compatibility.
Weights are excluded from Git and must be downloaded separately.

### 4. Start the application

```shell
python run.py
```

The model loads after the first valid positive point. The window can open
before the checkpoint is available, but automatic mask generation requires it.

## Annotate a dataset

Prepare two folders containing registered, equally sized RGB image pairs:

```text
my_dataset/
├── T1/
│   ├── sample_001.png
│   └── sample_002.png
└── T2/
    ├── sample_001.png
    └── sample_002.png
```

1. Click **Open Dataset** and select `my_dataset`, the parent of `T1` and `T2`.
2. Choose **Positive Point** and click a changed object in either view.
3. Add more positive points for other changed objects and negative points to correct false positives.
4. Refine the mask with the brush, polygon, or eraser and adjust the threshold as needed. Double-click to finish a polygon.
5. Review the result and mark it reviewed, confirm no change, or flag it for another review.
6. Move to the next pair using the dataset list or keyboard shortcuts.

Images are matched by full relative filename, including extension, with natural
sorting. Nested directories are supported. Missing, damaged, or differently sized
pairs are reported; valid pairs can still be annotated. Filenames are matched
case-insensitively on Windows.

Edits are automatically saved after approximately 0.9 seconds of inactivity and
before changing samples or closing the application. Reopening the dataset
restores the last selected sample and its annotation state. Editing a reviewed
annotation returns it to draft status.

For a single pair, click **Open Pair** and select the two images. Save or export
the annotation explicitly; dataset autosave applies when a dataset is open.

## Saved annotations and exports

Dataset annotations are stored alongside the images:

```text
my_dataset/
├── T1/
├── T2/
└── annotations/
    ├── project.json
    └── sample_001.png/
        ├── annotation.json
        └── rev-<revision-id>/
```

Each saved annotation includes point prompts, the threshold, review status,
manual corrections, and a full-precision soft mask. Source image paths are
relative to the dataset root so that the dataset can be moved as a unit.

Use **Export** to write the current sample into a selected directory:

| File | Contents |
| --- | --- |
| `<name>_binary.png` | Binary change mask with values 0 and 255, including manual edits |
| `<name>_soft.png` | 8-bit probability mask |
| `<name>_manual.png` | Optional manual overrides: 0 unchanged, 127 forced background, 255 forced change |
| `<name>.json` | Image paths, prompts, threshold, review state, and mask filenames |

## Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| Ctrl+U | Open a dataset |
| A / D | Previous / next pair |
| Ctrl+S | Save the current sample |
| Ctrl+Enter | Complete and move to the next pair |
| Ctrl+Z / Ctrl+Y | Undo / redo |
| Esc | Cancel an unfinished polygon |

## Troubleshooting

- **Checkpoint not found:** check `models/sam2_hiera_large.pt` or the path in `CHANGE_LABELER_MODEL`.
- **Checkpoint loading error:** use the original SAM 2 Hiera Large checkpoint matching the bundled configuration.
- **CUDA is unavailable:** check the installed PyTorch build and NVIDIA driver. CPU fallback is automatic.
- **CUDA runs out of memory:** close other GPU applications or start with CUDA hidden (`$env:CUDA_VISIBLE_DEVICES = "-1"` in PowerShell) to use the CPU.
- **No valid pairs:** select the parent of `T1` and `T2`, then check relative filenames, extensions, and image dimensions.
- **Qt fails to start:** use a desktop session and install dependencies in the active Python environment.

## Scope

Inputs must already be spatially registered and have identical pixel dimensions.
PNG, JPEG, and TIFF inputs are read as RGB images. The application does not
perform image registration, tiled processing of large GeoTIFFs, or georeferenced
vector export. No standalone executable, datasets, or pretrained weights are
included in this repository.

## Acknowledgments and third-party licenses

Mask proposals use [SAM 2](https://github.com/facebookresearch/sam2) by Meta.
The desktop interface uses PySide6. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[LICENSE](LICENSE), and [LICENSE_cctorch](LICENSE_cctorch) for the bundled
SAM 2 runtime's license notices. Dependencies retain their own licenses.
