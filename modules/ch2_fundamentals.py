# -*- coding: utf-8 -*-
"""
Chapter 2: Digital Image Fundamentals
Based on: Gonzalez & Woods, "Digital Image Processing", 3rd Edition

Topics covered:
- Image sampling and quantization
- Spatial and intensity resolution
- Image interpolation (Nearest Neighbor, Bilinear, Bicubic)
- Arithmetic and logical operations
- Geometric transformations

Author: Generated for DIP course companion
"""

import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# 2.4.4  Image Interpolation
# ---------------------------------------------------------------------------

def nearest_neighbor_interpolation(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Nearest Neighbor Interpolation (Section 2.4.4)

    Theory: For each output pixel (x', y'), find the nearest input pixel by rounding:
        x = round(x' / scale_x),  y = round(y' / scale_y)
    This is the simplest interpolation method but produces blocky artifacts (pixelation).
    Fast but low quality for upscaling.

    Mathematical Detail
    -------------------
    Given an input image f of size (H, W) and scaling factors (scale_y, scale_x),
    the output image g has size (H_out, W_out) = (round(H*scale_y), round(W*scale_x)).

    For every output coordinate (r', c'):
        r = clip( round(r' / scale_y), 0, H-1 )
        c = clip( round(c' / scale_x), 0, W-1 )
        g(r', c') = f(r, c)

    No new pixel values are created; the output is always a subset/superset
    of the original discrete levels, making this method fast but prone to
    "blocky" or "pixelated" artifacts on upscaling.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale (H, W) or colour (H, W, C) image, dtype uint8.
    scale_x : float
        Horizontal scaling factor  (> 1 → enlarge, < 1 → shrink).
    scale_y : float
        Vertical scaling factor    (> 1 → enlarge, < 1 → shrink).

    Returns
    -------
    np.ndarray
        Resampled image, same dtype as input.
    """
    image = np.asarray(image)
    h_in, w_in = image.shape[:2]
    h_out = max(1, round(h_in * scale_y))
    w_out = max(1, round(w_in * scale_x))

    # Build output row/column index arrays using the inverse mapping
    row_indices = np.round(np.arange(h_out) / scale_y).astype(int)
    col_indices = np.round(np.arange(w_out) / scale_x).astype(int)

    # Clip to valid range (handles boundary pixels)
    row_indices = np.clip(row_indices, 0, h_in - 1)
    col_indices = np.clip(col_indices, 0, w_in - 1)

    # Apply mapping via advanced indexing (works for both grayscale and colour)
    output = image[np.ix_(row_indices, col_indices)]
    return output.astype(image.dtype)


def bilinear_interpolation(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Bilinear Interpolation (Section 2.4.4)

    Theory: Uses weighted average of 4 nearest neighbors.
    For a point (x, y) surrounded by Q11(x1,y1), Q21(x2,y1), Q12(x1,y2), Q22(x2,y2):

    Step 1 (interpolate in x):
        f(x, y1) = ((x2-x)/(x2-x1))*f(Q11) + ((x-x1)/(x2-x1))*f(Q21)
        f(x, y2) = ((x2-x)/(x2-x1))*f(Q12) + ((x-x1)/(x2-x1))*f(Q22)

    Step 2 (interpolate in y):
        f(x, y) = ((y2-y)/(y2-y1))*f(x,y1) + ((y-y1)/(y2-y1))*f(x,y2)

    Produces smoother results than nearest neighbor. No blocking artifacts.

    Mathematical Detail
    -------------------
    The continuous output coordinates map back to fractional input coordinates:
        r_src = r' / scale_y
        c_src = c' / scale_x

    Let r0 = floor(r_src), r1 = r0+1  (clamped to [0, H-1])
        c0 = floor(c_src), c1 = c0+1  (clamped to [0, W-1])

    Fractional parts:
        dr = r_src - r0
        dc = c_src - c0

    The four neighbours and weights:
        g = (1-dr)*(1-dc)*f[r0,c0]
          + (1-dr)*  dc  *f[r0,c1]
          +   dr  *(1-dc)*f[r1,c0]
          +   dr  *  dc  *f[r1,c1]

    Parameters
    ----------
    image : np.ndarray
        Input grayscale (H, W) or colour (H, W, C) image, dtype uint8.
    scale_x : float
        Horizontal scaling factor.
    scale_y : float
        Vertical scaling factor.

    Returns
    -------
    np.ndarray
        Resampled image as uint8.
    """
    image = np.asarray(image, dtype=np.float64)
    h_in, w_in = image.shape[:2]
    h_out = max(1, round(h_in * scale_y))
    w_out = max(1, round(w_in * scale_x))

    # Continuous source coordinates for every output pixel
    r_src = np.arange(h_out, dtype=np.float64) / scale_y   # shape (h_out,)
    c_src = np.arange(w_out, dtype=np.float64) / scale_x   # shape (w_out,)

    # Integer floor neighbours
    r0 = np.floor(r_src).astype(int)
    c0 = np.floor(c_src).astype(int)
    r1 = np.clip(r0 + 1, 0, h_in - 1)
    c1 = np.clip(c0 + 1, 0, w_in - 1)
    r0 = np.clip(r0, 0, h_in - 1)
    c0 = np.clip(c0, 0, w_in - 1)

    # Fractional distances
    dr = (r_src - np.floor(r_src))[:, np.newaxis]   # (h_out, 1)
    dc = (c_src - np.floor(c_src))[np.newaxis, :]   # (1,  w_out)

    # Four corner lookups — works for grayscale (H,W) and colour (H,W,C)
    if image.ndim == 2:
        Q11 = image[np.ix_(r0, c0)]
        Q12 = image[np.ix_(r0, c1)]
        Q21 = image[np.ix_(r1, c0)]
        Q22 = image[np.ix_(r1, c1)]
    else:
        Q11 = image[np.ix_(r0, c0)].reshape(len(r0), len(c0), image.shape[2])
        Q12 = image[np.ix_(r0, c1)].reshape(len(r0), len(c1), image.shape[2])
        Q21 = image[np.ix_(r1, c0)].reshape(len(r1), len(c0), image.shape[2])
        Q22 = image[np.ix_(r1, c1)].reshape(len(r1), len(c1), image.shape[2])
        dr = dr[:, :, np.newaxis]
        dc = dc[:, :, np.newaxis]

    # Bilinear combination (Eq. 2.4-2 in G&W)
    output = (
        (1 - dr) * (1 - dc) * Q11
        + (1 - dr) *      dc  * Q12
        +      dr  * (1 - dc) * Q21
        +      dr  *      dc  * Q22
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def bicubic_interpolation(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Bicubic Interpolation (Section 2.4.4)

    Theory: Uses 16 nearest neighbors (4x4 grid) and cubic polynomial fitting.
    f(x,y) = sum_{i=0}^{3} sum_{j=0}^{3} a_ij * x^i * y^j

    Provides the highest quality among the three interpolation methods.
    Preserves edges and fine details better. Used in professional image editing.
    Uses scipy.ndimage.map_coordinates with order=3.

    Mathematical Detail
    -------------------
    The cubic convolution kernel (Keys 1981, cited in G&W) for a distance |t| is:

        w(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1,         |t| <= 1
               a|t|^3 - 5a|t|^2 + 8a|t| - 4a,        1 < |t| < 2
               0,                                       otherwise
    with a = -0.5 (bicubic spline approximation).

    scipy.ndimage.map_coordinates implements this with order=3 (cubic spline),
    which is equivalent in quality to bicubic convolution and is numerically
    robust with prefiltering.

    For each output pixel (r', c'):
        r_src = r' / scale_y
        c_src = c' / scale_x
    Then map_coordinates evaluates the cubic spline at that point.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale (H, W) or colour (H, W, C) image, dtype uint8.
    scale_x : float
        Horizontal scaling factor.
    scale_y : float
        Vertical scaling factor.

    Returns
    -------
    np.ndarray
        Resampled image as uint8.
    """
    image = np.asarray(image, dtype=np.float64)
    h_in, w_in = image.shape[:2]
    h_out = max(1, round(h_in * scale_y))
    w_out = max(1, round(w_in * scale_x))

    # Build output grid coordinates in input-image space
    r_src = np.linspace(0, h_in - 1, h_out)
    c_src = np.linspace(0, w_in - 1, w_out)
    grid_r, grid_c = np.meshgrid(r_src, c_src, indexing='ij')  # (h_out, w_out)
    coords = np.array([grid_r.ravel(), grid_c.ravel()])         # (2, h_out*w_out)

    if image.ndim == 2:
        output = map_coordinates(image, coords, order=3, mode='reflect')
        output = output.reshape(h_out, w_out)
    else:
        channels = []
        for ch in range(image.shape[2]):
            ch_out = map_coordinates(image[:, :, ch], coords, order=3, mode='reflect')
            channels.append(ch_out.reshape(h_out, w_out))
        output = np.stack(channels, axis=2)

    return np.clip(output, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2.4.3  Intensity Level (Quantization) Reduction
# ---------------------------------------------------------------------------

def reduce_gray_levels(image: np.ndarray, k: int) -> np.ndarray:
    """
    Intensity Level (Bit-Depth) Reduction (Section 2.4.3)

    Theory: Reduces an 8-bit image (256 levels) to 2^k gray levels.
    Formula: f_k(x,y) = floor(f(x,y) / 2^(8-k)) * (255 / (2^k - 1))

    Used to demonstrate the effect of quantization and the minimum number of
    gray levels needed to avoid false contouring artifacts.
    k=8: original (256 levels)
    k=1: binary (2 levels)

    Mathematical Detail
    -------------------
    An 8-bit image stores pixel values in [0, 255].  When we reduce to k bits
    we keep only the k most-significant bits (MSBs) of each pixel:

        step 1: bin_index = floor( f(x,y) / 2^(8-k) )   ∈ {0, 1, …, 2^k − 1}
        step 2: f_k(x,y)  = bin_index * 255 / (2^k − 1)  (re-scale to [0,255])

    When k < 4, false contours appear — abrupt jumps at quantisation
    boundaries that the human visual system mistakenly perceives as real edges.
    Dithering (not covered here) can mitigate this artefact.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) image, dtype uint8, values in [0, 255].
    k : int
        Target bit depth (1 ≤ k ≤ 8).

    Returns
    -------
    np.ndarray
        Quantised image, dtype uint8, rescaled to full [0, 255] range.

    Raises
    ------
    ValueError
        If k is not in [1, 8].
    """
    if not (1 <= k <= 8):
        raise ValueError(f"k must be in [1, 8], got {k}.")

    image = np.asarray(image, dtype=np.float64)
    levels = 2 ** k          # number of distinct grey levels
    divisor = 2 ** (8 - k)   # bin width in original [0,255] space

    # Quantise: map each pixel to its bin index
    bin_index = np.floor(image / divisor)
    # Re-scale bin indices back to the full [0, 255] display range
    if levels > 1:
        f_k = bin_index * (255.0 / (levels - 1))
    else:
        f_k = np.zeros_like(image)   # degenerate k=0 case (not normally reached)

    return np.clip(f_k, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2.4.2  Spatial Subsampling
# ---------------------------------------------------------------------------

def subsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Spatial Subsampling (Section 2.4.2)

    Theory: Reduces spatial resolution by taking every `factor`-th pixel.
    Demonstrates aliasing (Moire patterns, staircase effects) when sampling
    rate falls below the Nyquist rate (2 * max_frequency).
    The image is then upsampled back to original size using nearest neighbor
    to make the aliasing effect visible.

    Mathematical Detail
    -------------------
    The Nyquist–Shannon sampling theorem states that a signal must be sampled
    at a rate fs ≥ 2 * f_max (the Nyquist rate) to be perfectly reconstructed.

    When we keep only every `factor`-th pixel, the effective sampling rate
    drops to fs / factor.  For high-frequency content (fine textures, thin
    lines) this causes *aliasing* — high-frequency components fold back into
    the low-frequency band, manifesting as:
        • Moiré patterns on regular textures
        • Staircase (jagging) on diagonal edges

    Anti-aliasing is achieved by low-pass filtering *before* subsampling
    (e.g. a Gaussian blur), which is not applied here deliberately so the
    aliasing artefacts are clearly visible.

    After subsampling the small image is magnified back to the original
    dimensions with nearest-neighbour interpolation so the two images can
    be compared at the same display size.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale (H, W) or colour (H, W, C) image, dtype uint8.
    factor : int
        Subsampling factor ≥ 1.  factor=2 keeps 1 in every 2 pixels.

    Returns
    -------
    np.ndarray
        Image subsampled then magnified back to original size (uint8).

    Raises
    ------
    ValueError
        If factor < 1.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}.")

    h_orig, w_orig = image.shape[:2]
    # Subsample: keep every `factor`-th row and column
    small = image[::factor, ::factor]
    # Upsample back to original size using nearest-neighbour so artefacts show
    scale_y = h_orig / small.shape[0]
    scale_x = w_orig / small.shape[1]
    return nearest_neighbor_interpolation(small, scale_x=scale_x, scale_y=scale_y)


# ---------------------------------------------------------------------------
# 2.6  Arithmetic and Logical Operations
# ---------------------------------------------------------------------------

def image_average(images: list) -> np.ndarray:
    """
    Noise Reduction by Image Averaging (Section 2.6.1)

    Theory: If g_i(x,y) = f(x,y) + eta_i(x,y) where eta_i are independent
    Gaussian noise with zero mean and variance sigma^2, then:

    g_bar(x,y) = (1/K) * sum_{i=1}^{K} g_i(x,y)

    The noise variance of the averaged image = sigma^2 / K
    SNR improves by factor sqrt(K).

    Used in astronomy, microscopy, and medical imaging.

    Mathematical Detail
    -------------------
    Each noisy observation is modelled as:
        g_i(x,y) = f(x,y) + η_i(x,y)

    where f is the true (noise-free) image and η_i ~ N(0, σ²) are i.i.d.

    The ensemble average:
        ĝ(x,y) = (1/K) Σ_{i=1}^{K} g_i(x,y)

    has expected value:
        E[ĝ(x,y)] = f(x,y)      (unbiased estimate)

    and variance:
        Var[ĝ(x,y)] = σ² / K   (variance decreases linearly with K)

    Signal-to-noise ratio therefore improves as √K — doubling the SNR
    requires 4× as many frames.

    Parameters
    ----------
    images : list of np.ndarray
        List of K images, all the same shape (H, W) or (H, W, C) and dtype.

    Returns
    -------
    np.ndarray
        Pixel-wise average image, dtype uint8.

    Raises
    ------
    ValueError
        If the list is empty or images have inconsistent shapes.
    """
    if len(images) == 0:
        raise ValueError("images list must not be empty.")

    stack = np.stack([np.asarray(img, dtype=np.float64) for img in images], axis=0)
    avg = np.mean(stack, axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)


def image_subtract(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Image Subtraction (Section 2.6.2)

    Theory: d(x,y) = |f1(x,y) - f2(x,y)|

    Applications:
    - Background subtraction (detect moving objects)
    - Change detection between frames
    - Medical imaging: detect lesion growth
    Result is clipped to [0, 255] and returned as uint8.

    Mathematical Detail
    -------------------
    The difference image d(x,y) = |f1(x,y) − f2(x,y)| has the following
    properties:

    • Zero where the two images are identical.
    • Large values wherever the images differ (changed regions, moving objects).
    • The result is always non-negative (absolute value), enabling direct
      display as a greyscale image.

    In medical imaging, subtracting a pre-contrast image from a post-contrast
    image (digital subtraction angiography, DSA) isolates blood vessels by
    cancelling static background tissue.

    In motion detection, subtracting the previous frame from the current frame
    highlights pixels that belong to moving objects.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images of the same shape (H, W) or (H, W, C), dtype uint8.

    Returns
    -------
    np.ndarray
        Absolute difference image, dtype uint8, values in [0, 255].
    """
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    diff = np.abs(a - b)
    return np.clip(diff, 0, 255).astype(np.uint8)


def image_multiply(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Image Multiplication / Masking (Section 2.6.3)

    Theory: g(x,y) = f(x,y) * h(x,y)

    Applications:
    - Shading correction (multiply by inverse of illumination pattern)
    - Region of interest masking
    - Windowing functions for DFT preprocessing
    Result normalized to [0, 255].

    Mathematical Detail
    -------------------
    When both images are uint8 in [0, 255], the raw product can reach 255² = 65 025.
    We therefore normalise the result:

        g(x,y) = ( f(x,y) * h(x,y) ) / 255

    so that the output stays in [0, 255].

    Key use cases:
    • **ROI masking**: h is a binary mask (0 or 255); the product retains f
      only inside the region of interest.
    • **Shading correction**: h is the reciprocal of a smooth illumination
      field; multiplication flattens the brightness gradient.
    • **DFT windowing**: multiplying by a Hanning/Hamming window (re-scaled to
      [0,255]) reduces spectral leakage before computing the Fourier transform.

    Parameters
    ----------
    img1 : np.ndarray
        Image to be masked / corrected, shape (H, W) or (H, W, C), uint8.
    img2 : np.ndarray
        Second image (mask, shading field, window), same shape, uint8.

    Returns
    -------
    np.ndarray
        Element-wise product normalised to [0, 255], dtype uint8.
    """
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    product = (a * b) / 255.0          # keep in [0, 255] range
    return np.clip(product, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2.7  Geometric Transformations
# ---------------------------------------------------------------------------

def geometric_transform(image: np.ndarray, transform_type: str, **kwargs) -> np.ndarray:
    """
    Geometric Transformations (Section 2.7)

    Types supported:
    - 'rotation'   : rotate by angle (degrees), uses bilinear interpolation
    - 'translation': shift by (tx, ty) pixels
    - 'flip_h'     : horizontal flip
    - 'flip_v'     : vertical flip
    - 'scale'      : scale by (sx, sy) factors

    Affine transform matrix for rotation by theta:
    [cos(theta)  -sin(theta)  tx]
    [sin(theta)   cos(theta)  ty]
    [0            0            1]

    Mathematical Detail
    -------------------
    All supported transforms are *affine* (preserve parallel lines), expressed
    as a 3×3 homogeneous matrix T:

        [x']   [T] [x]
        [y'] =      [y]
        [1 ]        [1]

    **Inverse mapping** (used for all types) avoids holes in the output:
    for every output pixel p_out we compute p_in = T⁻¹ · p_out and
    sample the input image at p_in using bilinear interpolation.

    Supported keyword arguments
    ---------------------------
    rotation:
        angle (float)  – counter-clockwise rotation in degrees.
                         The image is rotated about its centre; output size
                         is chosen to contain the entire rotated image.
    translation:
        tx (int/float) – horizontal shift in pixels (positive → right).
        ty (int/float) – vertical shift in pixels   (positive → down).
    flip_h, flip_v:
        No extra kwargs.
    scale:
        sx (float) – horizontal scale factor.
        sy (float) – vertical scale factor.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale (H, W) or colour (H, W, C) image, dtype uint8.
    transform_type : str
        One of 'rotation', 'translation', 'flip_h', 'flip_v', 'scale'.
    **kwargs
        Transform-specific parameters (see above).

    Returns
    -------
    np.ndarray
        Transformed image, dtype uint8.

    Raises
    ------
    ValueError
        If transform_type is not recognised.
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    is_color = image.ndim == 3

    transform_type = transform_type.lower()

    # ------------------------------------------------------------------
    # Simple cases that do not require inverse-mapping via affine matrix
    # ------------------------------------------------------------------
    if transform_type == 'flip_h':
        return image[:, ::-1].copy()

    if transform_type == 'flip_v':
        return image[::-1, :].copy()

    # ------------------------------------------------------------------
    # Affine transforms via inverse mapping + bilinear interpolation
    # ------------------------------------------------------------------

    def _apply_affine(img_in, M_inv, h_out, w_out):
        """
        Apply an inverse affine mapping M_inv to img_in.
        For each (r_out, c_out) in the output, compute the source coordinate
        via M_inv and sample img_in with bilinear interpolation.
        """
        # Build output grid in homogeneous coordinates
        c_out, r_out = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(r_out)
        pts_out = np.stack([c_out.ravel(), r_out.ravel(), ones.ravel()], axis=0)  # (3, N)
        pts_in = M_inv @ pts_out                                                   # (3, N)
        c_src = pts_in[0].reshape(h_out, w_out)
        r_src = pts_in[1].reshape(h_out, w_out)

        # Clip and get four neighbours for bilinear interpolation
        h_in, w_in = img_in.shape[:2]
        r0 = np.clip(np.floor(r_src).astype(int), 0, h_in - 1)
        r1 = np.clip(r0 + 1,                       0, h_in - 1)
        c0 = np.clip(np.floor(c_src).astype(int), 0, w_in - 1)
        c1 = np.clip(c0 + 1,                       0, w_in - 1)

        # Out-of-bounds mask (pixels that map outside the input → black)
        valid = (r_src >= 0) & (r_src <= h_in - 1) & (c_src >= 0) & (c_src <= w_in - 1)

        dr = (r_src - np.floor(r_src))[..., np.newaxis] if img_in.ndim == 3 else (r_src - np.floor(r_src))
        dc = (c_src - np.floor(c_src))[..., np.newaxis] if img_in.ndim == 3 else (c_src - np.floor(c_src))

        img_f = img_in.astype(np.float64)
        if img_in.ndim == 3:
            Q11 = img_f[r0, c0]
            Q12 = img_f[r0, c1]
            Q21 = img_f[r1, c0]
            Q22 = img_f[r1, c1]
            out = (1 - dr) * (1 - dc) * Q11 + (1 - dr) * dc * Q12 \
                +      dr  * (1 - dc) * Q21 +      dr  * dc * Q22
            out[~valid] = 0
        else:
            Q11 = img_f[r0, c0]
            Q12 = img_f[r0, c1]
            Q21 = img_f[r1, c0]
            Q22 = img_f[r1, c1]
            out = (1 - dr) * (1 - dc) * Q11 + (1 - dr) * dc * Q12 \
                +      dr  * (1 - dc) * Q21 +      dr  * dc * Q22
            out[~valid] = 0

        return np.clip(out, 0, 255).astype(np.uint8)

    # ---- Rotation --------------------------------------------------------
    if transform_type == 'rotation':
        angle = kwargs.get('angle', 0.0)
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Compute output canvas size to contain the full rotated image
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
        cx_in, cy_in = w / 2.0, h / 2.0
        # Rotate corners around the input centre
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rot_corners = (R @ (corners - [cx_in, cy_in]).T).T + [cx_in, cy_in]
        x_min, y_min = rot_corners.min(axis=0)
        x_max, y_max = rot_corners.max(axis=0)
        w_out = int(np.ceil(x_max - x_min))
        h_out = int(np.ceil(y_max - y_min))
        cx_out, cy_out = w_out / 2.0, h_out / 2.0

        # Forward transform T maps input → output (rotation about input centre,
        # then shift to output canvas origin).
        # Inverse T⁻¹ is used for backward mapping.
        # T: p_out = R * (p_in - c_in) + c_out
        # T⁻¹: p_in = R^T * (p_out - c_out) + c_in
        # In homogeneous form:
        def _make_inv_rotation(cos_t, sin_t, cx_in, cy_in, cx_out, cy_out):
            """Build the 3×3 inverse rotation matrix (output → input coords)."""
            # Translate output origin to output centre, then rotate by -theta,
            # then translate to input centre.
            T_shift_out = np.array([[1, 0, -cx_out],
                                    [0, 1, -cy_out],
                                    [0, 0,       1]], dtype=float)
            R_inv = np.array([[cos_t,  sin_t, 0],
                              [-sin_t, cos_t, 0],
                              [0,      0,     1]], dtype=float)
            T_shift_in = np.array([[1, 0, cx_in],
                                   [0, 1, cy_in],
                                   [0, 0,     1]], dtype=float)
            return T_shift_in @ R_inv @ T_shift_out

        M_inv = _make_inv_rotation(cos_t, sin_t, cx_in, cy_in, cx_out, cy_out)
        return _apply_affine(image, M_inv, h_out, w_out)

    # ---- Translation --------------------------------------------------------
    if transform_type == 'translation':
        tx = float(kwargs.get('tx', 0))
        ty = float(kwargs.get('ty', 0))
        # Inverse mapping: input coord = output coord - translation
        M_inv = np.array([[1, 0, -tx],
                          [0, 1, -ty],
                          [0, 0,   1]], dtype=float)
        return _apply_affine(image, M_inv, h, w)

    # ---- Scale -------------------------------------------------------------
    if transform_type == 'scale':
        sx = float(kwargs.get('sx', 1.0))
        sy = float(kwargs.get('sy', 1.0))
        # Delegate to bilinear_interpolation for clean implementation
        return bilinear_interpolation(image, scale_x=sx, scale_y=sy)

    raise ValueError(
        f"Unknown transform_type '{transform_type}'. "
        "Choose from: 'rotation', 'translation', 'flip_h', 'flip_v', 'scale'."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "nearest_neighbor_interpolation",
    "bilinear_interpolation",
    "bicubic_interpolation",
    "reduce_gray_levels",
    "subsample_image",
    "image_average",
    "image_subtract",
    "image_multiply",
    "geometric_transform",
]
