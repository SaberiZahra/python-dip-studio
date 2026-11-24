"""
Utility Functions for DIP Gonzalez Project
Common helper functions used across all modules.
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')          # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
from typing import Callable

__all__ = [
    'load_image',
    'save_image',
    'normalize_image',
    'to_uint8',
    'to_float',
    'ensure_grayscale',
    'compute_mse',
    'compute_psnr',
    'compute_ssim',
    'add_text_overlay',
    'create_comparison_grid',
    'get_sample_images',
    'apply_pipeline',
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image(path: str | Path, mode: str = 'gray') -> np.ndarray:
    """
    Load an image from disk and return it as a NumPy ``uint8`` array.

    Parameters
    ----------
    path : str or pathlib.Path
        Absolute or relative path to the image file.
    mode : str
        One of:

        * ``'gray'``  — single-channel grayscale (shape ``(H, W)``).
        * ``'rgb'``   — 3-channel RGB (shape ``(H, W, 3)``), BGR→RGB conversion.
        * ``'rgba'``  — 4-channel RGBA (shape ``(H, W, 4)``).

    Returns
    -------
    numpy.ndarray of dtype uint8
        Loaded image array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If *mode* is not recognised or the image cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mode = mode.lower()
    if mode == 'gray':
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
    elif mode == 'rgb':
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'rgba':
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"mode must be 'gray', 'rgb', or 'rgba'; got '{mode}'.")

    return img.astype(np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> None:
    """
    Save a NumPy image array to disk.

    Handles both ``float64`` images (expected range ``[0.0, 1.0]`` or
    ``[0, 255]``) and ``uint8`` images.  For float images, values are scaled
    to ``[0, 255]`` before saving.

    Parameters
    ----------
    image : numpy.ndarray
        Image to save.  Grayscale ``(H, W)`` or colour ``(H, W, 3)``/``(H, W, 4)``.
    path : str or pathlib.Path
        Destination file path.  The format is inferred from the extension
        (e.g., ``.png``, ``.jpg``).

    Notes
    -----
    For colour images the input is assumed to be RGB; it is converted to BGR
    before writing via OpenCV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = image.copy()

    # Convert float → uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0 + 1e-9 and img.min() >= -1e-9:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # Convert RGB → BGR for cv2 if colour
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Dtype conversion helpers
# ---------------------------------------------------------------------------

def normalize_image(
    image: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 255.0,
) -> np.ndarray:
    """
    Linear (min-max) normalisation to the range ``[out_min, out_max]``.

    ::

        I_out = (I - I_min) / (I_max - I_min) * (out_max - out_min) + out_min

    Parameters
    ----------
    image : numpy.ndarray
        Input image (any numeric dtype).
    out_min : float
        Minimum value of the output range.
    out_max : float
        Maximum value of the output range.

    Returns
    -------
    numpy.ndarray of dtype float64

    Notes
    -----
    If the image is constant (``I_min == I_max``), all pixels are set to
    ``out_min`` to avoid division by zero.
    """
    img = image.astype(np.float64)
    i_min, i_max = img.min(), img.max()
    if i_max == i_min:
        return np.full_like(img, out_min, dtype=np.float64)
    return (img - i_min) / (i_max - i_min) * (out_max - out_min) + out_min


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Clip and convert a floating-point image to ``uint8``.

    Values are clipped to ``[0, 255]`` before casting, so out-of-range floats
    are safely clamped.

    Parameters
    ----------
    image : numpy.ndarray
        Float image with values expected in ``[0, 255]``.

    Returns
    -------
    numpy.ndarray of dtype uint8
    """
    return np.clip(image, 0, 255).astype(np.uint8)


def to_float(image: np.ndarray) -> np.ndarray:
    """
    Convert a ``uint8`` image to ``float64`` in the range ``[0.0, 1.0]``.

    ::

        I_float = I_uint8 / 255.0

    Parameters
    ----------
    image : numpy.ndarray
        Input image (uint8 expected; other dtypes are accepted but divided
        by 255 regardless).

    Returns
    -------
    numpy.ndarray of dtype float64 with values in ``[0.0, 1.0]``.
    """
    return image.astype(np.float64) / 255.0


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is a 2-D grayscale array.

    * If *image* is already 2-D, it is returned unchanged.
    * If *image* is 3-channel (RGB or BGR), it is converted to grayscale using
      the standard luminance formula: ``0.299·R + 0.587·G + 0.114·B``.
    * If *image* is 4-channel (RGBA), the alpha channel is discarded first.

    Parameters
    ----------
    image : numpy.ndarray
        Input image of shape ``(H, W)``, ``(H, W, 3)``, or ``(H, W, 4)``.

    Returns
    -------
    numpy.ndarray
        2-D grayscale array with the same dtype as the input.

    Raises
    ------
    ValueError
        If the number of channels is not 1, 3, or 4.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if channels == 4:
            rgb = image[:, :, :3]
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    raise ValueError(
        f"Cannot convert image with shape {image.shape} to grayscale."
    )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Mean Squared Error between two images.

    ::

        MSE = (1/MN) · Σ Σ [f(x,y) − g(x,y)]²

    Parameters
    ----------
    img1, img2 : numpy.ndarray
        Images of the same shape.  Internally cast to ``float64``.

    Returns
    -------
    float
        MSE value.  ``0.0`` for identical images.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §2.4.3, Eq. (2-28).
    """
    f1 = img1.astype(np.float64)
    f2 = img2.astype(np.float64)
    return float(np.mean((f1 - f2) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in decibels.

    ::

        PSNR = 10 · log₁₀ (MAX² / MSE)

    where ``MAX = 255`` for uint8 images.

    Parameters
    ----------
    original : numpy.ndarray
        Reference image (uint8 or float64 in ``[0, 255]``).
    processed : numpy.ndarray
        Processed / reconstructed image, same shape as *original*.

    Returns
    -------
    float
        PSNR in dB.  ``+inf`` if MSE = 0.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §2.4.3, Eq. (2-31).
    """
    mse = compute_mse(original, processed)
    if mse == 0.0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified Structural Similarity Index (SSIM).

    Uses an 11×11 Gaussian window (σ = 1.5) to compute local statistics.

    ::

        SSIM(x, y) = (2·μ_x·μ_y + C1) · (2·σ_xy + C2)
                     ─────────────────────────────────────
                     (μ_x² + μ_y² + C1) · (σ_x² + σ_y² + C2)

    Constants:  ``C1 = (K1·L)²``,  ``C2 = (K2·L)²``,
    with  ``K1 = 0.01``,  ``K2 = 0.03``,  ``L = 255``.

    Parameters
    ----------
    img1, img2 : numpy.ndarray
        Grayscale images of the same shape (uint8 or float64).

    Returns
    -------
    float
        Mean SSIM over all windows; in ``[-1, 1]``.  ``1.0`` for identical images.

    References
    ----------
    Wang, Z. et al. (2004). "Image quality assessment: from error visibility
    to structural similarity." *IEEE Trans. Image Processing*, 13(4), 600–612.
    Gonzalez & Woods, 3rd ed., §2.4.3.
    """
    from scipy.ndimage import gaussian_filter

    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    f1 = img1.astype(np.float64)
    f2 = img2.astype(np.float64)
    sigma = 1.5

    mu1 = gaussian_filter(f1, sigma=sigma, truncate=3.5)
    mu2 = gaussian_filter(f2, sigma=sigma, truncate=3.5)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(f1 ** 2, sigma=sigma, truncate=3.5) - mu1_sq
    sigma2_sq = gaussian_filter(f2 ** 2, sigma=sigma, truncate=3.5) - mu2_sq
    sigma12   = gaussian_filter(f1 * f2, sigma=sigma, truncate=3.5) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """
    Overlay text on an image using OpenCV.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or colour, uint8).
    text : str
        String to render.
    position : tuple of int
        ``(x, y)`` pixel coordinate of the bottom-left corner of the text.
    font_scale : float
        Font size scale factor passed to ``cv2.putText``.
    color : tuple of int
        BGR colour of the text (for grayscale images, use a scalar wrapped
        in a tuple, e.g. ``(200,)``).

    Returns
    -------
    numpy.ndarray
        Copy of *image* with text rendered on top.
    """
    out = image.copy()
    if out.dtype != np.uint8:
        out = to_uint8(out)
    cv2.putText(
        out,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return out


def create_comparison_grid(
    images: list[np.ndarray],
    titles: list[str],
    n_cols: int = 3,
) -> plt.Figure:
    """
    Create a matplotlib Figure displaying images in a grid with titles.

    Parameters
    ----------
    images : list of numpy.ndarray
        Images to display.  Grayscale (2-D) or colour (3-D) arrays.
    titles : list of str
        Title for each image.  Must have the same length as *images*.
    n_cols : int
        Number of columns in the grid.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.  Call ``fig.savefig(...)`` or ``plt.show()`` as needed.

    Notes
    -----
    Figure size is scaled automatically: ``(4 * n_cols) × (4 * n_rows)``
    inches.
    """
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            # Assume RGB
            ax.imshow(img.astype(np.uint8))
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sample test images
# ---------------------------------------------------------------------------

def get_sample_images() -> dict[str, np.ndarray]:
    """
    Generate or load a dictionary of standard DIP test images.

    Returns
    -------
    dict mapping str → numpy.ndarray (uint8)
        * 'cameraman'    — Classic 512×512 cameraman benchmark
        * 'moon'         — Low contrast moon (classic for Chapter 3 Laplacian/HE)
        * 'coins'        — Coins image (classic for Chapter 10 segmentation/Otsu)
        * 'astronaut'    — 3-channel RGB image (classic for Chapter 6 color processing)
        * 'checkerboard' — 8×8 squares alternating black and white (256×256)
        * 'gradient'     — Horizontal linear gradient from 0 to 255 (256×256)
        * 'circles'      — White filled circles on black background (256×256)
        * 'text_pattern' — Text-like pattern for morphology/thinning (256×256)
    """
    images: dict[str, np.ndarray] = {}

    try:
        import skimage.data as skdata
        images['cameraman'] = skdata.camera()
        images['moon'] = skdata.moon()
        images['coins'] = skdata.coins()
        images['astronaut'] = skdata.astronaut()
    except Exception:
        pass

    H, W = 256, 256

    # --- checkerboard ---
    sq = 32  # each square is 32×32 px (8 squares per row/column)
    cb = np.zeros((H, W), dtype=np.uint8)
    for r in range(H // sq):
        for c in range(W // sq):
            if (r + c) % 2 == 0:
                cb[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    images['checkerboard'] = cb

    # --- horizontal gradient ---
    gradient = np.tile(np.linspace(0, 255, W, dtype=np.float32), (H, 1))
    images['gradient'] = gradient.astype(np.uint8)

    # --- circles ---
    circles = np.zeros((H, W), dtype=np.uint8)
    centres = [(64, 64, 40), (64, 192, 25), (192, 64, 55), (192, 192, 30), (128, 128, 50)]
    yy, xx = np.ogrid[:H, :W]
    for cy, cx, r in centres:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        circles[mask] = 255
    images['circles'] = circles

    # --- text_pattern (simple horizontal bars simulating strokes) ---
    text_pat = np.zeros((H, W), dtype=np.uint8)
    for row_start in [30, 60, 90, 120, 150, 180, 210]:
        text_pat[row_start:row_start + 8, 20:236] = 255
    for col in [20, 60, 100, 140, 180, 220, 236]:
        text_pat[30:220, col:col + 8] = 255
    images['text_pattern'] = text_pat

    # Fallback for cameraman/moon if skimage unavailable
    if 'cameraman' not in images:
        images['cameraman'] = cb
    if 'moon' not in images:
        images['moon'] = gradient.astype(np.uint8)

    return images


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def apply_pipeline(
    image: np.ndarray,
    operations: list[tuple[Callable, dict]],
) -> list[np.ndarray]:
    """
    Apply a sequence of image-processing operations and collect intermediates.

    Each operation is a ``(function, kwargs)`` tuple.  The function receives
    the **current image** as its first positional argument plus any additional
    keyword arguments from *kwargs*.

    Parameters
    ----------
    image : numpy.ndarray
        Initial input image.
    operations : list of (callable, dict)
        Pipeline steps.  Each element is a 2-tuple:

        * ``function`` — a callable that accepts an image as its first argument
          and returns a processed image.
        * ``kwargs``   — keyword arguments forwarded to *function*.

    Returns
    -------
    list of numpy.ndarray
        List of **all** intermediate results, **including the original** as the
        first element.  The list therefore has ``len(operations) + 1`` entries.

    Example
    -------
    >>> from modules.ch9_morphology import binary_erosion_custom, get_structuring_element
    >>> from modules.utils import get_sample_images, apply_pipeline
    >>> imgs = get_sample_images()
    >>> B = get_structuring_element('disk', 5)
    >>> steps = [(binary_erosion_custom, {'B': B})]
    >>> results = apply_pipeline(imgs['circles'] > 128, steps)
    >>> len(results)   # original + 1 step
    2

    Notes
    -----
    Use :func:`create_comparison_grid` on the returned list to visualise every
    step of the pipeline in a single figure.
    """
    results: list[np.ndarray] = [image.copy()]
    current = image.copy()

    for func, kwargs in operations:
        current = func(current, **kwargs)
        results.append(current.copy())

    return results
