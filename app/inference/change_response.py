import cv2
import numpy as np
from typing import Iterable, Tuple


def load_rgb_image(path: str) -> np.ndarray:
    """Read an RGB image for SAM2."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x_min) / (x_max - x_min)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    return clahe.apply(gray)


def get_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def fft_highpass(gray: np.ndarray, radius: int = 20) -> np.ndarray:
    gray_f = gray.astype(np.float32)

    f = np.fft.fft2(gray_f)
    fshift = np.fft.fftshift(f)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist2 = (x - cx) ** 2 + (y - cy) ** 2

    mask = np.ones((h, w), dtype=np.float32)
    mask[dist2 <= radius * radius] = 0.0

    fshift_filtered = fshift * mask
    ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(ishift)
    return np.abs(img_back)


def build_change_rgb(img_t1: np.ndarray, img_t2: np.ndarray, hp_radius: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a change response map.

    Preserve the original change-map logic without feature differences.
    Use the change map only for candidate scoring, not as a SAM2 input image.
    """
    if img_t1.shape != img_t2.shape:
        raise ValueError(f"img_t1.shape={img_t1.shape} != img_t2.shape={img_t2.shape}")

    g1 = cv2.cvtColor(img_t1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img_t2, cv2.COLOR_RGB2GRAY)

    n1 = apply_clahe(g1)
    n2 = apply_clahe(g2)

    grad1 = get_gradient_magnitude(n1)
    grad2 = get_gradient_magnitude(n2)
    grad_diff = np.abs(grad1 - grad2)

    hp1 = fft_highpass(n1, radius=hp_radius)
    hp2 = fft_highpass(n2, radius=hp_radius)
    hp_diff = np.abs(hp1 - hp2)

    gray_diff = cv2.absdiff(n1, n2).astype(np.float32)

    diff_fused = 0.4 * gray_diff + 0.3 * grad_diff + 0.3 * hp_diff
    diff_fused_u8 = normalize_to_uint8(diff_fused)

    change_rgb = np.stack(
        [normalize_to_uint8(n1), normalize_to_uint8(n2), diff_fused_u8],
        axis=-1,
    ).astype(np.uint8)
    return change_rgb, diff_fused_u8


def resize_rgb(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale augmentation used only to construct the change map."""
    if abs(scale - 1.0) < 1e-6:
        return image

    h, w = image.shape[:2]
    new_w = max(32, int(round(w * scale)))
    new_h = max(32, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def rotate_rgb(image: np.ndarray, k: int) -> np.ndarray:
    """Rotate by multiples of 90 degrees, with k in 0/1/2/3."""
    k = int(k) % 4
    if k == 0:
        return image
    return np.rot90(image, k=k).copy()


def inverse_rotate_map(mask: np.ndarray, k: int) -> np.ndarray:
    """Restore a rotated change map to the original image orientation."""
    k = int(k) % 4
    if k == 0:
        return mask
    return np.rot90(mask, k=(4 - k) % 4).copy()


def build_augmented_change_map(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    hp_radius: int = 20,
    scales: Iterable[float] = (1.0, 0.75, 1.25),
    rotations: Iterable[int] = (0, 1),
) -> np.ndarray:
    """
    Build a more stable change map using lightweight scale and rotation augmentation.

    These augmentations preserve the output dimensions and improve diff_map consistency
    to reduce false detections and improve precision.
    """
    base_h, base_w = img_t1.shape[:2]
    diff_maps = []

    for scale in scales:
        t1_scaled = resize_rgb(img_t1, scale)
        t2_scaled = resize_rgb(img_t2, scale)

        for rot_k in rotations:
            t1_aug = rotate_rgb(t1_scaled, rot_k)
            t2_aug = rotate_rgb(t2_scaled, rot_k)
            _, diff_aug = build_change_rgb(t1_aug, t2_aug, hp_radius=hp_radius)

            diff_back = inverse_rotate_map(diff_aug, rot_k)
            if diff_back.shape[:2] != (base_h, base_w):
                diff_back = cv2.resize(diff_back, (base_w, base_h), interpolation=cv2.INTER_LINEAR)

            diff_maps.append(diff_back.astype(np.float32))

    if not diff_maps:
        _, diff_map = build_change_rgb(img_t1, img_t2, hp_radius=hp_radius)
        return diff_map

    return normalize_to_uint8(np.mean(diff_maps, axis=0))
