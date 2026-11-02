# Third-party notices

## SAM 2

The `sam2/` runtime and `sam2_configs/` configurations are vendored from
[Meta's SAM 2 project](https://github.com/facebookresearch/sam2). They retain
their upstream copyright headers and are distributed under Apache License 2.0;
see `LICENSE`. This is a standalone runtime snapshot, not an installation of
the latest upstream SAM 2 package.

The CUDA connected-components source `sam2/csrc/connected_components.cu`
contains code adapted from Connected_components_PyTorch. Its additional
BSD 3-Clause notice is provided in `LICENSE_cctorch`.

Model checkpoints are downloaded separately from Meta and are not included.

## Runtime dependencies

PySide6 (Qt), PyTorch, TorchVision, NumPy, Pillow, OpenCV, Hydra, OmegaConf,
and tqdm are installed separately and retain their respective licenses.
See each upstream project's license and distribution terms.

## Application code

The application was extracted from a research workspace. The change-response
and mask-fusion helpers are retained as standalone application source; no
training datasets, experimental results, or trained research models are included.
