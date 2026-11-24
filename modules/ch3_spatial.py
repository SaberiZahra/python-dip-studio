# -*- coding: utf-8 -*-
"""
Chapter 3: Intensity Transformations & Spatial Filtering
=========================================================
Based on: Gonzalez & Woods, "Digital Image Processing", 3rd Edition

Topics covered:
- Intensity transformation functions (point operations)
- Histogram processing
- Spatial smoothing filters
- Spatial sharpening filters

All functions follow the exact mathematical formulations given in the textbook.
Input images are assumed to be 8-bit (uint8, L=256) unless otherwise stated.
"""

import numpy as np
from scipy.ndimage import (
    convolve,
    median_filter,
    maximum_filter,
    minimum_filter,
)
from scipy.signal import convolve2d  # noqa: F401 (available for external use)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
L = 256        # Number of gray levels (8-bit images)
L_MAX = L - 1  # Maximum gray value = 255


# ===========================================================================
# SECTION 3.2 -- INTENSITY TRANSFORMATION FUNCTIONS (Point Operations)
# ===========================================================================

def image_negative(image):
    """
    Compute the photographic negative of an image.

    Mathematical formula (Eq. 3.2-1):
        s = (L - 1) - r

    where r is the input intensity and s is the output intensity,
    both in the range [0, L-1].

    Background & Use Cases
    ----------------------
    The image negative transformation reverses the intensity scale so that
    dark areas become bright and bright areas become dark -- analogous to a
    photographic negative.  Applications include:

    * Medical X-ray imaging: radiologists sometimes find it easier to examine
      white detail on a black background by viewing the negative of the
      original X-ray (which is itself already a negative of the scene).
    * Enhancing white detail in dark regions: when the region of interest
      contains bright features embedded in a predominantly dark image,
      computing the negative brings those features into the bright end of the
      display range where the human visual system is more sensitive.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].

    Returns
    -------
    np.ndarray
        Negated image, uint8, same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    >>> image_negative(img)
    array([[255, 127], [ 55,   0]], dtype=uint8)
    """
    r = np.asarray(image, dtype=np.int32)
    result = L_MAX - r
    return np.clip(result, 0, L_MAX).astype(np.uint8)


def log_transform(image, c=None):
    """
    Apply the logarithmic intensity transformation.

    Mathematical formula (Eq. 3.2-2):
        s = c * log(1 + r)

    where r is the input intensity, s is the output intensity, and c is a
    scaling constant chosen so that the output spans the full range [0, L-1]:
        c = (L - 1) / log(1 + max(r))

    Background & Use Cases
    ----------------------
    The log transform maps a narrow range of low-intensity values to a wider
    range of output levels and compresses the high-intensity end.  This is
    called dynamic-range compression.

    Key applications:

    * Fourier spectrum display: the magnitude spectrum of the DFT spans an
      enormous dynamic range (sometimes 10^6 : 1).  Displaying
      c * log(1 + |F(u,v)|) makes both low- and high-energy components
      visible simultaneously on an 8-bit screen.
    * Sensor data with high dynamic range: satellite or medical images where
      bright regions (bone, sky) would otherwise swamp dim regions.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255], or float.
    c : float, optional
        Scaling constant.  Defaults to (L-1) / log(1 + max(r)).

    Returns
    -------
    np.ndarray
        Log-transformed image, uint8 in [0, 255].
    """
    r = np.asarray(image, dtype=np.float64)
    r_max = r.max()
    if r_max == 0:
        # Degenerate case: all-black image
        return np.zeros_like(image, dtype=np.uint8)
    if c is None:
        c = L_MAX / np.log(1.0 + r_max)
    s = c * np.log(1.0 + r)
    return np.clip(np.round(s), 0, L_MAX).astype(np.uint8)


def power_law_transform(image, gamma, c=1.0):
    """
    Apply the power-law (gamma) intensity transformation.

    Mathematical formula (Eq. 3.2-4):
        s = c * r^gamma

    where r is the input intensity normalized to [0, 1] and s is the
    normalized output intensity.  The result is rescaled back to [0, L-1].

    Background & Use Cases
    ----------------------
    Power-law transformations arise naturally in many imaging devices:

    * CRT monitor calibration (gamma correction): older CRT monitors respond
      to input voltage V with brightness proportional to V^2.5.  Applying a
      pre-correction with gamma = 1/2.5 ~= 0.4 linearises the display.
    * gamma > 1  ->  darkening: compresses bright regions, expands dark.
      Useful for images that appear too bright or washed out.
    * gamma < 1  ->  brightening: expands bright regions, compresses dark.
      Useful for images that appear too dark (e.g., underexposed photos).
    * gamma = 1  ->  identity (linear; no change when c=1).

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    gamma : float
        Exponent of the power-law transform.  Positive real number.
    c : float, optional
        Multiplicative scaling constant.  Default is 1.0.

    Returns
    -------
    np.ndarray
        Gamma-corrected image, uint8 in [0, 255].

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
    >>> power_law_transform(img, gamma=0.5)   # brightens image
    >>> power_law_transform(img, gamma=2.0)   # darkens image
    """
    r = np.asarray(image, dtype=np.float64) / L_MAX   # normalize to [0, 1]
    s = c * np.power(r, gamma)
    s_scaled = s * L_MAX
    return np.clip(np.round(s_scaled), 0, L_MAX).astype(np.uint8)


def piecewise_linear_contrast_stretch(image, r1, s1, r2, s2):
    """
    Apply piecewise-linear contrast stretching (3-segment linear transform).

    The transform passes through four control points:
        (0, 0)  ->  (r1, s1)  ->  (r2, s2)  ->  (L-1, L-1)

    For each input intensity r:
        Segment 1  (0 <= r <= r1):   s = (s1/r1) * r
        Segment 2  (r1 < r <= r2):   s = s1 + ((s2-s1)/(r2-r1)) * (r-r1)
        Segment 3  (r2 < r <= L-1):  s = s2 + ((L-1-s2)/(L-1-r2)) * (r-r2)

    Background & Use Cases
    ----------------------
    Many real-world images are captured under non-ideal conditions:

    * Poorly illuminated scenes: pixel values cluster in a narrow band
      (e.g., all between 50 and 150), leaving most of the display range
      unused.  Contrast stretching expands this range to [0, 255].
    * Scientific imaging: microscopy, medical, or satellite images often
      require custom tone curves to bring specific intensity bands into focus.

    Setting (r1, s1) = (r_min, 0) and (r2, s2) = (r_max, 255) is equivalent
    to full linear stretch (also called normalisation).

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    r1, s1 : float
        First control point in (input, output) space.
    r2, s2 : float
        Second control point.  Must have r1 <= r2 and s1 <= s2 for a
        monotonically increasing (non-decreasing) transform.

    Returns
    -------
    np.ndarray
        Contrast-stretched image, uint8 in [0, 255].

    Examples
    --------
    >>> img = np.array([[50, 100, 128, 150, 200]], dtype=np.uint8)
    >>> piecewise_linear_contrast_stretch(img, r1=70, s1=0, r2=150, s2=255)
    """
    r = np.asarray(image, dtype=np.float64)
    s = np.zeros_like(r)

    # Segment slopes -- guard against division by zero
    slope1 = (s1 / r1) if r1 != 0 else 0.0
    slope2 = ((s2 - s1) / (r2 - r1)) if (r2 - r1) != 0 else 0.0
    slope3 = ((L_MAX - s2) / (L_MAX - r2)) if (L_MAX - r2) != 0 else 0.0

    mask1 = r <= r1
    s[mask1] = slope1 * r[mask1]

    mask2 = (r > r1) & (r <= r2)
    s[mask2] = s1 + slope2 * (r[mask2] - r1)

    mask3 = r > r2
    s[mask3] = s2 + slope3 * (r[mask3] - r2)

    return np.clip(np.round(s), 0, L_MAX).astype(np.uint8)


def gray_level_slicing(image, A, B, highlight=True):
    """
    Gray-level (intensity) slicing -- highlight a specific band of intensities.

    Two modes (Gonzalez & Woods, Fig. 3.17):

    highlight=True  (binary plane):
        s = L-1  if  A <= r <= B
        s = 0    otherwise

    highlight=False (preserving background):
        s = L-1  if  A <= r <= B
        s = r    otherwise   (original value retained outside the band)

    Background & Use Cases
    ----------------------
    * Medical lesion detection: isolate lesions or tumours that fall in a
      specific intensity range (e.g., contrast-enhanced tissue).
    * Remote sensing: highlight water bodies, vegetation, or cloud-covered
      regions based on their characteristic brightness.
    * Non-destructive testing: highlight cracks or voids that appear at a
      particular intensity in X-ray or ultrasound images.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    A : int
        Lower bound of the intensity range to highlight (inclusive).
    B : int
        Upper bound of the intensity range to highlight (inclusive).
    highlight : bool, optional
        If True, pixels outside [A, B] are set to 0 (binary plane mode).
        If False, pixels outside [A, B] retain their original values.

    Returns
    -------
    np.ndarray
        Sliced image, uint8 in [0, 255].

    Examples
    --------
    >>> img = np.array([[50, 100, 150, 200]], dtype=np.uint8)
    >>> gray_level_slicing(img, A=90, B=160, highlight=True)
    array([[  0, 255, 255,   0]], dtype=uint8)
    >>> gray_level_slicing(img, A=90, B=160, highlight=False)
    array([[ 50, 255, 255, 200]], dtype=uint8)
    """
    r = np.asarray(image, dtype=np.uint8)
    s = np.zeros_like(r) if highlight else r.copy()
    mask = (r >= A) & (r <= B)
    s[mask] = L_MAX
    return s


def bit_plane_slicing(image, bit_n):
    """
    Extract a single bit plane from an 8-bit grayscale image.

    Mathematical formula (Gonzalez & Woods, Eq. 3.4-1):
        b_n(x, y) = floor(f(x,y) / 2^n)  mod  2

    where bit_n = 0 is the LSB (least significant bit) and
          bit_n = 7 is the MSB (most significant bit).

    The result is a binary image with values {0, 255} where 255 represents
    a '1' bit at position bit_n.

    Background & Use Cases
    ----------------------
    * Image compression: the MSB planes (bits 7, 6, 5) contain most of
      the visually significant information.  Transmitting only the top 4-5
      planes can approximate the original image with roughly half the data.
    * Digital watermarking (steganography): hidden information can be encoded
      in the LSB (bit 0) plane without perceptible visual change, since the
      LSB contributes only 1/255 ~= 0.4% to the total range.
    * Image analysis: bit planes can reveal structural patterns, texture,
      and quantisation artefacts not obvious in the full image.

    Parameters
    ----------
    image : np.ndarray
        Input 8-bit grayscale image, uint8 in [0, 255].
    bit_n : int
        Bit plane index to extract.  Must be in [0, 7].
        0 = LSB (least significant), 7 = MSB (most significant).

    Returns
    -------
    np.ndarray
        Binary image (uint8), same shape as input, with values 0 or 255.

    Raises
    ------
    ValueError
        If bit_n is not in the range [0, 7].

    Examples
    --------
    >>> img = np.array([[200]], dtype=np.uint8)   # 200 = 1100_1000
    >>> bit_plane_slicing(img, bit_n=7)
    array([[255]], dtype=uint8)   # MSB is 1
    >>> bit_plane_slicing(img, bit_n=0)
    array([[0]], dtype=uint8)     # LSB is 0
    """
    if not (0 <= bit_n <= 7):
        raise ValueError("bit_n must be in [0, 7], got {}".format(bit_n))
    r = np.asarray(image, dtype=np.uint8)
    # Extract the nth bit: shift right by n, then AND with 1
    bit_plane = (r >> bit_n) & 1
    # Scale to [0, 255] so the result is a visible binary image
    return (bit_plane * L_MAX).astype(np.uint8)


# ===========================================================================
# SECTION 3.3 -- HISTOGRAM PROCESSING
# ===========================================================================

def compute_histogram(image):
    """
    Compute the normalised histogram (probability mass function) of an image.

    For a discrete image f with M rows and N columns:
        p_r(r_k) = n_k / (M * N),   k = 0, 1, ..., L-1

    where:
        r_k    = k-th gray level (k in [0, 255])
        n_k    = number of pixels with gray level k
        M * N  = total number of pixels
        p_r    = estimated probability (normalised histogram)

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (any dtype; cast to uint8 for binning).

    Returns
    -------
    hist : np.ndarray, shape (256,)
        Normalised histogram: hist[k] = n_k / (M*N).
        Values are in [0, 1] and sum to 1.0.
    bins : np.ndarray, shape (256,)
        Gray-level indices 0, 1, ..., 255.

    Examples
    --------
    >>> img = np.array([[0, 0, 1, 2]], dtype=np.uint8)
    >>> hist, bins = compute_histogram(img)
    >>> hist[0], hist[1], hist[2]
    (0.5, 0.25, 0.25)
    """
    r = np.asarray(image, dtype=np.uint8).ravel()
    MN = r.size
    counts = np.bincount(r, minlength=L)   # n_k for k in [0, 255]
    hist = counts / MN                      # p_r(r_k)
    bins = np.arange(L, dtype=np.float64)
    return hist, bins


def histogram_equalization(image):
    """
    Global Histogram Equalisation (HE).

    Mathematical derivation (Gonzalez & Woods, Section 3.3.1):

    For a discrete image the equalisation transform is the CDF of the
    input histogram scaled to [0, L-1]:

        s_k = T(r_k) = (L-1) * CDF(r_k)
                     = (L-1) * sum_{j=0}^{k}  p_r(r_j)
                     = (L-1) * sum_{j=0}^{k}  n_j / (M*N)

    HE is proven to produce a uniform output histogram when the input is
    continuous (Gonzalez & Woods, Theorem 3.3-1).  For discrete images the
    output histogram is approximately uniform.

    Why it works
    ------------
    The CDF is chosen because it is the probability integral transform.
    If r has PDF p_r(r) and s = T(r) is a monotone function of r, then the
    PDF of s is p_s(s) = p_r(r) |dr/ds|.  Setting
        T(r) = (L-1) * integral_0^r p_r(w) dw
    yields |dr/ds| = 1 / ((L-1) * p_r(r)), so p_s(s) = 1/(L-1) -- a uniform
    distribution, maximising entropy.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].

    Returns
    -------
    eq_image : np.ndarray
        Histogram-equalised image, uint8, same shape as input.
    transform : np.ndarray, shape (256,)
        Look-up table T(r_k) mapping input gray levels to output gray levels
        (rounded to nearest integer, in [0, 255]).

    Examples
    --------
    >>> img = np.random.randint(0, 100, (512, 512), dtype=np.uint8)
    >>> eq_img, lut = histogram_equalization(img)
    """
    r = np.asarray(image, dtype=np.uint8)
    hist, _ = compute_histogram(r)
    cdf = np.cumsum(hist)                              # sum_{j=0}^{k} p_r(r_j)
    transform = np.round(L_MAX * cdf).astype(np.uint8)  # s_k = (L-1) * CDF
    eq_image = transform[r]                            # apply LUT
    return eq_image, transform


def histogram_specification(image, target_hist):
    """
    Histogram Matching (Specification) -- transform an image so its histogram
    approximates a desired target histogram.

    Algorithm (Gonzalez & Woods, Section 3.3.2):

    Step 1 -- Equalise the source image:
        T(r_k) = (L-1) * sum_{j=0}^{k} p_r(r_j)       [source CDF]

    Step 2 -- Compute the CDF of the target histogram:
        G(z_q) = (L-1) * sum_{i=0}^{q} p_z(z_i)        [target CDF]

    Step 3 -- Invert G using nearest-match lookup:
        For each gray level k in the source image:
            Find z* = argmin_q |G(z_q) - T(r_k)|
        The mapping is z* = G^{-1}(T(r_k)).

    Background & Use Cases
    ----------------------
    * Remote sensing / satellite imagery: images acquired by the same sensor
      on different days or under different atmospheric conditions have
      different histograms.  Histogram matching normalises them so that
      mosaics appear seamless and change-detection algorithms are not
      confused by illumination drift.
    * Medical imaging: aligning the contrast of images from different
      scanners or acquisition protocols for consistent CAD performance.
    * Photo editing: match the tonal character of one image to another
      (e.g., apply a "film look").

    Parameters
    ----------
    image : np.ndarray
        Input source image, uint8 in [0, 255].
    target_hist : np.ndarray, shape (256,)
        Desired output histogram.  Can be normalised (sums to 1) or raw
        counts; it will be normalised internally.

    Returns
    -------
    np.ndarray
        Histogram-matched image, uint8, same shape as input.

    Examples
    --------
    >>> src = np.random.randint(0, 128, (256, 256), dtype=np.uint8)
    >>> target = np.ones(256) / 256   # uniform target
    >>> matched = histogram_specification(src, target)
    """
    r = np.asarray(image, dtype=np.uint8)

    # Step 1: equalize source -> T(r_k)
    src_hist, _ = compute_histogram(r)
    src_cdf = np.cumsum(src_hist)
    T = np.round(L_MAX * src_cdf)   # shape (256,)

    # Step 2: CDF of target histogram -> G(z_q)
    tgt = np.asarray(target_hist, dtype=np.float64)
    tgt = tgt / tgt.sum()           # normalise
    tgt_cdf = np.cumsum(tgt)
    G = np.round(L_MAX * tgt_cdf)  # shape (256,)

    # Step 3: invert G -- for each source gray level k find z* = G^{-1}(s)
    lut = np.zeros(L, dtype=np.uint8)
    for k in range(L):
        s = T[k]
        # Find z that minimises |G(z) - s|
        diff = np.abs(G - s)
        lut[k] = int(np.argmin(diff))

    return lut[r]


def local_histogram_equalization(image, kernel_size=8, clip_limit=2.0):
    """
    Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    CLAHE is a tiled variant of Histogram Equalisation that:

    1. Divides the image into non-overlapping tiles of size
       kernel_size x kernel_size.
    2. Computes a local histogram for each tile.
    3. Clips the histogram at clip_limit * (tile_pixels / L) to prevent
       over-amplification of noise.  Redistributes clipped counts uniformly
       to other bins.
    4. Applies the equalisation CDF for each tile, then uses bilinear
       interpolation between adjacent tile transforms to avoid blocking
       artefacts.

    Why CLAHE instead of global HE?
    --------------------------------
    Global HE uses a single CDF computed from the entire image, which works
    well for globally under/over-exposed images but fails when contrast
    varies spatially (e.g., the dark background of a chest X-ray gets
    equalised together with the bright lung tissue, amplifying noise in
    dark regions).  CLAHE adapts the transform locally while the clip limit
    prevents unchecked noise amplification.

    Implementation note
    -------------------
    When skimage is available, we delegate to
    skimage.exposure.equalize_adapthist (full CLAHE with bilinear
    interpolation).  Otherwise we fall back to a simpler tile-based local HE
    without bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    kernel_size : int, optional
        Side length (in pixels) of each local tile.  Default 8.
    clip_limit : float, optional
        Clip limit expressed as a multiple of the uniform histogram height.
        A value of 1.0 disables clipping.  Typical range: 1-40.  Default 2.0.

    Returns
    -------
    np.ndarray
        CLAHE-processed image, uint8 in [0, 255].

    References
    ----------
    Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
    In P. Heckbert (Ed.), Graphics Gems IV (pp. 474-485).
    """
    try:
        from skimage.exposure import equalize_adapthist  # type: ignore
        img_float = image.astype(np.float64) / L_MAX
        result_float = equalize_adapthist(
            img_float,
            kernel_size=kernel_size,
            clip_limit=clip_limit / L_MAX
        )
        return np.clip(np.round(result_float * L_MAX), 0, L_MAX).astype(np.uint8)

    except ImportError:
        # --------------------------------------------------------------------
        # Fallback: simple tile-based local HE (no bilinear interpolation)
        # --------------------------------------------------------------------
        r = np.asarray(image, dtype=np.uint8)
        rows, cols = r.shape
        output = np.zeros_like(r)
        ks = kernel_size
        tile_pixels = ks * ks
        clip_threshold = clip_limit * (tile_pixels / L)

        for row_start in range(0, rows, ks):
            for col_start in range(0, cols, ks):
                row_end = min(row_start + ks, rows)
                col_end = min(col_start + ks, cols)
                tile = r[row_start:row_end, col_start:col_end]

                # Build histogram of this tile
                counts = np.bincount(tile.ravel(), minlength=L).astype(np.float64)

                # Clip and redistribute excess uniformly
                excess = np.sum(np.maximum(counts - clip_threshold, 0.0))
                counts = np.minimum(counts, clip_threshold)
                counts += excess / L

                # Build CDF and LUT
                cdf = np.cumsum(counts)
                cdf_min = cdf[cdf > 0][0] if cdf[cdf > 0].size > 0 else 0.0
                tile_total = tile.size
                lut = np.round(
                    (L_MAX * (cdf - cdf_min)) / max(tile_total - cdf_min, 1)
                ).astype(np.uint8)

                output[row_start:row_end, col_start:col_end] = lut[tile]

        return output


# ===========================================================================
# SECTION 3.4 -- SPATIAL SMOOTHING (LOW-PASS) FILTERS
# ===========================================================================

def box_filter(image, kernel_size=3):
    """
    Apply a box (mean / averaging) spatial filter.

    The box filter kernel of size m x n is:
        h(s, t) = 1 / (m * n)   for all (s, t) in the kernel

    The filtered image is the 2D discrete convolution:
        g(x, y) = sum_s sum_t h(s,t) * f(x-s, y-t)

    which is simply the arithmetic mean of all pixel values inside the
    m x n neighbourhood centred at (x, y).

    Background & Use Cases
    ----------------------
    * Noise reduction: averaging uncorrelated additive noise with variance
      sigma^2 over an m x n neighbourhood reduces the variance to
      sigma^2 / (m*n).
    * Blurring / smoothing: a larger kernel produces stronger smoothing.
    * Limitation: the box filter equally weights all neighbours, so it
      blurs edges as aggressively as it smooths flat regions.
      High-frequency edge information is lost because the filter is a
      low-pass filter whose frequency response is a 2D sinc function.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 or float.
    kernel_size : int, optional
        Side length of the square kernel.  Default 3.

    Returns
    -------
    np.ndarray
        Smoothed image, uint8, same shape as input.
    """
    r = np.asarray(image, dtype=np.float64)
    m = n = kernel_size
    kernel = np.ones((m, n), dtype=np.float64) / (m * n)
    smoothed = convolve(r, kernel, mode='reflect')
    return np.clip(np.round(smoothed), 0, L_MAX).astype(np.uint8)


def gaussian_filter_spatial(image, kernel_size=5, sigma=1.0):
    """
    Apply a Gaussian spatial smoothing filter.

    The 2D Gaussian kernel (Gonzalez & Woods, Eq. 3.5-7):
        G(s, t) = exp( -(s^2 + t^2) / (2 * sigma^2) )

    normalised so that all coefficients sum to 1.

    Key Properties
    --------------
    1. Spatial separability: the 2D Gaussian is the outer product of two 1D
       Gaussians (G(s,t) = g(s)*g(t)), so it can be applied as two sequential
       1D convolutions, reducing complexity from O(k^2*N) to O(2*k*N).
    2. Isotropy: the Gaussian is circularly symmetric, so it blurs equally in
       all directions (no directional artefacts).
    3. No ringing: unlike ideal brick-wall low-pass filters, the Gaussian has
       no negative side lobes in either domain, so it never produces the
       oscillatory ringing artefacts seen near sharp edges.
    4. Frequency-domain correspondence: a Gaussian in the spatial domain is
       also a Gaussian in the frequency domain (self-dual under the Fourier
       transform).  Larger sigma -> narrower low-pass filter cut-off.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 or float.
    kernel_size : int, optional
        Side length of the square kernel window in pixels.  Should be odd
        and >= 3.  Typical rule-of-thumb: kernel_size >= 6*sigma + 1.
        Default 5.
    sigma : float, optional
        Standard deviation of the Gaussian (in pixels).  Default 1.0.

    Returns
    -------
    np.ndarray
        Gaussian-smoothed image, uint8, same shape as input.
    """
    r = np.asarray(image, dtype=np.float64)
    half = kernel_size // 2
    s = np.arange(-half, half + 1, dtype=np.float64)
    t = np.arange(-half, half + 1, dtype=np.float64)
    S, T = np.meshgrid(s, t)
    kernel = np.exp(-(S**2 + T**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()   # normalise to unit sum
    smoothed = convolve(r, kernel, mode='reflect')
    return np.clip(np.round(smoothed), 0, L_MAX).astype(np.uint8)


def median_filter_custom(image, kernel_size=3):
    """
    Apply a median spatial filter using a sliding rectangular window.

    For each pixel (x, y) the output is:
        g(x, y) = median{ f(x+s, y+t) : (s,t) in W }

    where W is a kernel_size x kernel_size neighbourhood.

    Background & Use Cases
    ----------------------
    The median filter is the classic remedy for salt-and-pepper noise
    (also called impulse noise), where random pixels are set to either
    0 (pepper) or 255 (salt).

    Advantages over linear mean filter:
    * Edge preservation: an isolated outlier pixel (salt or pepper) is
      replaced by the median of its neighbourhood.  The median ignores
      extreme values and picks the central value, which is much more likely
      to be part of the true signal near an edge.
    * Non-linear: the median is not a linear operation, so it does NOT
      sharpen or amplify noise the way gradient operators do.
    * Statistical robustness: the median is a robust estimator with 50%
      breakdown point -- up to half the neighbourhood can be corrupted before
      the estimate breaks down.

    Disadvantage: more computationally expensive than linear convolution
    (O(k^2 log k) per pixel vs. O(k^2) for box filter), and not separable.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 or float.
    kernel_size : int, optional
        Side length of the square filter window.  Default 3.

    Returns
    -------
    np.ndarray
        Median-filtered image, uint8, same shape as input.

    Examples
    --------
    >>> img = np.array([[0, 0, 255, 0, 0]], dtype=np.uint8)  # single salt pixel
    >>> median_filter_custom(img, kernel_size=3)
    array([[0, 0, 0, 0, 0]], dtype=uint8)   # noise removed
    """
    r = np.asarray(image, dtype=np.uint8)
    result = median_filter(r, size=kernel_size, mode='reflect')
    return result.astype(np.uint8)


def max_filter(image, kernel_size=3):
    """
    Apply a sliding-window maximum (dilation) filter.

    For each pixel (x, y):
        g(x, y) = max{ f(x+s, y+t) : (s,t) in W }

    Background & Use Cases
    ----------------------
    * Removes pepper (dark) noise: a pepper pixel (value 0) in a
      neighbourhood full of brighter pixels is replaced by the neighbourhood
      maximum, eliminating the dark outlier.
    * Morphological dilation: the max filter is equivalent to a grey-scale
      morphological dilation with a flat structuring element, which expands
      bright regions and fills small dark holes.
    * Side-effect: because large pixel values propagate, the overall image
      becomes slightly brighter and bright objects appear enlarged.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 or float.
    kernel_size : int, optional
        Side length of the square filter window.  Default 3.

    Returns
    -------
    np.ndarray
        Maximum-filtered image, uint8, same shape as input.
    """
    r = np.asarray(image, dtype=np.uint8)
    result = maximum_filter(r, size=kernel_size, mode='reflect')
    return result.astype(np.uint8)


def min_filter(image, kernel_size=3):
    """
    Apply a sliding-window minimum (erosion) filter.

    For each pixel (x, y):
        g(x, y) = min{ f(x+s, y+t) : (s,t) in W }

    Background & Use Cases
    ----------------------
    * Removes salt (bright) noise: a salt pixel (value 255) in a
      neighbourhood of darker pixels is replaced by the neighbourhood
      minimum, eliminating the bright outlier.
    * Morphological erosion: the min filter is equivalent to grey-scale
      morphological erosion with a flat structuring element, which shrinks
      bright regions and expands dark regions.
    * Side-effect: the overall image becomes slightly darker.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 or float.
    kernel_size : int, optional
        Side length of the square filter window.  Default 3.

    Returns
    -------
    np.ndarray
        Minimum-filtered image, uint8, same shape as input.
    """
    r = np.asarray(image, dtype=np.uint8)
    result = minimum_filter(r, size=kernel_size, mode='reflect')
    return result.astype(np.uint8)


# ===========================================================================
# SECTION 3.5 -- SPATIAL SHARPENING (HIGH-PASS / DERIVATIVE) FILTERS
# ===========================================================================

def laplacian_sharpen(image, mode='4-neighbor'):
    """
    Sharpen an image using the discrete Laplacian operator.

    The Laplacian is the second-order isotropic derivative operator:
        nabla^2 f = d^2f/dx^2 + d^2f/dy^2

    Two discrete approximations are supported (Gonzalez & Woods, Fig. 3.37):

    4-neighbour (non-diagonal):
        W4 = [[ 0,  1,  0],
              [ 1, -4,  1],
              [ 0,  1,  0]]

    8-neighbour (diagonal included):
        W8 = [[ 1,  1,  1],
              [ 1, -8,  1],
              [ 1,  1,  1]]

    Enhanced (sharpened) image formula:
        g(x,y) = f(x,y) - nabla^2 f(x,y)   [for kernels with negative centre]

    Subtracting the Laplacian (which has a negative centre) is equivalent to
    adding detail -- edges and fine structures are enhanced.

    Background & Use Cases
    ----------------------
    * Second-order derivative: the Laplacian responds to rapid intensity
      changes (edges, lines, isolated points) and is zero in regions of
      constant or linearly changing intensity.
    * Zero crossings at edges: the sign change of nabla^2 f (zero crossing)
      marks the precise location of an edge.
    * Isotropic: both W4 and W8 are symmetric under 90-degree rotation (W4)
      or 45-degree rotation (W8), so they detect edges regardless of
      orientation.
    * Sensitive to noise: because it is a second-order operator, the
      Laplacian amplifies noise more than first-order gradient operators
      (Sobel, Prewitt).  Pre-smoothing (e.g., Gaussian) is often applied.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    mode : str, optional
        '4-neighbor' (default) or '8-neighbor'.  Determines which discrete
        Laplacian kernel to use.

    Returns
    -------
    np.ndarray
        Sharpened image, uint8, same shape as input.

    Raises
    ------
    ValueError
        If mode is not one of '4-neighbor' or '8-neighbor'.
    """
    if mode == '4-neighbor':
        kernel = np.array([[0,  1,  0],
                           [1, -4,  1],
                           [0,  1,  0]], dtype=np.float64)
    elif mode == '8-neighbor':
        kernel = np.array([[1,  1,  1],
                           [1, -8,  1],
                           [1,  1,  1]], dtype=np.float64)
    else:
        raise ValueError(
            "mode must be '4-neighbor' or '8-neighbor', got '{}'".format(mode)
        )

    f = np.asarray(image, dtype=np.float64)
    laplacian = convolve(f, kernel, mode='reflect')
    # Both kernels have a negative centre; the sharpening formula is g = f - nabla^2 f
    g = f - laplacian
    return np.clip(np.round(g), 0, L_MAX).astype(np.uint8)


def unsharp_masking(image, blur_sigma=1.0, k=1.0):
    """
    Sharpen an image via Unsharp Masking (USM) / Highboost Filtering.

    Algorithm (Gonzalez & Woods, Section 3.6.3):

    Step 1 -- Blur the image to obtain a smooth approximation:
        f_bar(x,y) = Gaussian_blur(f(x,y), sigma=blur_sigma)

    Step 2 -- Compute the unsharp mask (high-frequency residual):
        g_mask(x,y) = f(x,y) - f_bar(x,y)

    Step 3 -- Add the weighted mask back to the original:
        g(x,y) = f(x,y) + k * g_mask(x,y)

    Combining steps 1-3:
        g(x,y) = (1 + k) * f(x,y) - k * f_bar(x,y)

    Interpretation by value of k
    ----------------------------
    * k = 1  ->  Unsharp masking: the classic darkroom technique.
      Subtracting the blurred version leaves only the high-frequency detail.
      Adding that detail back once sharpens the image to the original
      edge strength.
    * k > 1  ->  Highboost filtering: amplifies the high-frequency mask
      beyond the original amplitude, producing a "crisper" or over-sharpened
      result.  k -> infinity approaches a pure high-pass filter response.
    * k = 0  ->  no sharpening (output equals input).

    Background
    ----------
    USM originated in analogue photography and printing where a blurred
    negative (the "unsharp mask") was laid on top of the original negative
    to increase contrast at edges.  The digital version replicates this by
    computing the blur programmatically.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    blur_sigma : float, optional
        Standard deviation of the Gaussian blur used to create f_bar.
        Larger values blur more -> wider sharpening halo.  Default 1.0.
    k : float, optional
        Weight of the mask (k=1 -> unsharp masking; k>1 -> highboost).
        Default 1.0.

    Returns
    -------
    np.ndarray
        Sharpened image, uint8, same shape as input.  Values are clipped
        to [0, 255].
    """
    f = np.asarray(image, dtype=np.float64)
    # Determine kernel size automatically from sigma (at least 3, always odd)
    ks = max(3, int(6 * blur_sigma + 1))
    if ks % 2 == 0:
        ks += 1
    f_blur = gaussian_filter_spatial(image, kernel_size=ks, sigma=blur_sigma).astype(np.float64)
    g_mask = f - f_blur
    g = f + k * g_mask
    return np.clip(np.round(g), 0, L_MAX).astype(np.uint8)


def sobel_gradient(image):
    """
    Compute image gradients using the Sobel operator.

    The Sobel masks (Gonzalez & Woods, Fig. 3.41) are:

        Gx = [[-1, -2, -1],     Gy = [[-1, 0, 1],
               [ 0,  0,  0],           [-2, 0, 2],
               [ 1,  2,  1]]           [-1, 0, 1]]

    (Gx detects horizontal edges; Gy detects vertical edges.)

    Gradient magnitude (Euclidean norm approximation):
        M(x,y) = sqrt( Gx^2 + Gy^2 )

    Gradient direction (angle of steepest ascent):
        theta(x,y) = arctan2( Gy, Gx )   [in radians, range (-pi, pi)]

    Note: the gradient direction is perpendicular to the edge itself.

    Why Sobel?
    ----------
    * Noise robustness: by using weights [1, 2, 1] in the direction
      perpendicular to differentiation, the Sobel operator effectively
      applies a 1D Gaussian smoothing (sigma ~= 0.85) before differentiation,
      making it more noise-resistant than the simpler Prewitt operator.
    * First-order derivative: while the Laplacian detects edges via zero
      crossings of the second derivative, the Sobel operator detects them
      at the peaks of the first derivative magnitude.
    * Directional sensitivity: the separate Gx and Gy components reveal
      the dominant orientation of edges in the image.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].

    Returns
    -------
    dict with keys:
        'magnitude' : np.ndarray (float64) -- gradient magnitude M(x,y)
        'gx'        : np.ndarray (float64) -- horizontal gradient Gx * f
        'gy'        : np.ndarray (float64) -- vertical gradient Gy * f
        'angle'     : np.ndarray (float64) -- gradient direction in radians
    """
    f = np.asarray(image, dtype=np.float64)
    Gx_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)
    Gy_kernel = np.array([[-1,  0,  1],
                           [-2,  0,  2],
                           [-1,  0,  1]], dtype=np.float64)
    gx = convolve(f, Gx_kernel, mode='reflect')
    gy = convolve(f, Gy_kernel, mode='reflect')
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return {
        'magnitude': magnitude,
        'gx': gx,
        'gy': gy,
        'angle': angle,
    }


def prewitt_gradient(image):
    """
    Compute image gradients using the Prewitt operator.

    The Prewitt masks (Gonzalez & Woods, Section 3.6.1) are:

        Gx = [[-1, -1, -1],     Gy = [[-1, 0, 1],
               [ 0,  0,  0],           [-1, 0, 1],
               [ 1,  1,  1]]           [-1, 0, 1]]

    Gradient magnitude:
        M(x,y) = sqrt( Gx^2 + Gy^2 )

    Gradient direction:
        theta(x,y) = arctan2( Gy, Gx )

    Comparison with Sobel
    ---------------------
    * Equal weighting: the Prewitt operator uses uniform weights [1,1,1] in
      the smoothing direction (vs. [1,2,1] in Sobel), so it applies a simple
      box average rather than a Gaussian.
    * Slightly more noise-sensitive: the lack of central weighting means
      Prewitt does not down-weight the influence of diagonal neighbours as
      much as Sobel.
    * Simpler computation: useful when computational resources are very
      limited (all coefficients are +/-1 or 0, requiring only additions).
    * Both operators are mathematically first-order directional derivatives
      and produce equivalent results on ideal step edges.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].

    Returns
    -------
    dict with keys:
        'magnitude' : np.ndarray (float64) -- gradient magnitude M(x,y)
        'gx'        : np.ndarray (float64) -- horizontal gradient
        'gy'        : np.ndarray (float64) -- vertical gradient
        'angle'     : np.ndarray (float64) -- gradient direction in radians
    """
    f = np.asarray(image, dtype=np.float64)
    Gx_kernel = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float64)
    Gy_kernel = np.array([[-1,  0,  1],
                           [-1,  0,  1],
                           [-1,  0,  1]], dtype=np.float64)
    gx = convolve(f, Gx_kernel, mode='reflect')
    gy = convolve(f, Gy_kernel, mode='reflect')
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return {
        'magnitude': magnitude,
        'gx': gx,
        'gy': gy,
        'angle': angle,
    }


# ===========================================================================
# SECTION 3.3.3 -- ENHANCEMENT USING LOCAL STATISTICS
# ===========================================================================

def enhance_with_local_stats(
    image,
    kernel_size=3,
    k0=0.4,
    k1=0.02,
    k2=0.4,
    C=22.8
):
    """
    Selectively enhance low-contrast, dark regions using local image statistics.

    Algorithm (Gonzalez & Woods, Section 3.3.3):

    For each pixel (x, y), compute:
        m(x,y)      = local mean in the kernel_size x kernel_size neighbourhood
        sigma(x,y)  = local standard deviation in the same neighbourhood

    Global statistics:
        m_G         = global mean of the entire image
        sigma_G     = global standard deviation of the entire image

    Enhancement rule:
        IF   m(x,y)     <= k0 * m_G        (pixel is in a relatively dark region)
        AND  sigma(x,y) >= k1 * sigma_G    (some minimum texture is present)
        AND  sigma(x,y) <= k2 * sigma_G    (pixel is in a low-contrast region)
        THEN
            g(x,y) = C * f(x,y)            (amplify by constant C)
        ELSE
            g(x,y) = f(x,y)                (leave unchanged)

    Background & Use Cases
    ----------------------
    * X-ray lesion enhancement: small, dark, low-contrast masses (such as
      early-stage tumours) can be selectively brightened without amplifying
      noise in the bright bone or lung regions.
    * Adaptive local amplification: global histogram equalisation would
      amplify noise across the entire image; this method restricts
      amplification to only those pixels that are both dark and low-contrast,
      which are the most likely to contain subtle diagnostic detail.
    * Parameter tuning:
        k0 (default 0.4) -- lower -> fewer pixels selected (more selective).
        k1 (default 0.02) -- ensures some minimum texture is present.
        k2 (default 0.4) -- upper bound on local contrast for selection.
        C  (default 22.8) -- empirically chosen by Gonzalez & Woods.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image, uint8 in [0, 255].
    kernel_size : int, optional
        Side length of the local neighbourhood window.  Default 3.
    k0 : float, optional
        Threshold on local mean relative to global mean.  Default 0.4.
    k1 : float, optional
        Lower threshold on local std relative to global std.  Default 0.02.
    k2 : float, optional
        Upper threshold on local std relative to global std.  Default 0.4.
    C : float, optional
        Amplification constant applied to selected pixels.  Default 22.8.

    Returns
    -------
    np.ndarray
        Locally enhanced image, uint8 in [0, 255].
    """
    f = np.asarray(image, dtype=np.float64)

    # Global statistics
    m_G = f.mean()
    sigma_G = f.std()

    # Local mean via box convolution
    ks = kernel_size
    kernel = np.ones((ks, ks), dtype=np.float64) / (ks * ks)
    local_mean = convolve(f, kernel, mode='reflect')

    # Local standard deviation: sqrt(E[x^2] - E[x]^2)
    local_mean_sq = convolve(f**2, kernel, mode='reflect')
    local_var = local_mean_sq - local_mean**2
    local_var = np.maximum(local_var, 0.0)   # guard against numerical negatives
    local_std = np.sqrt(local_var)

    # Enhancement condition (all three clauses must hold)
    cond = (
        (local_mean <= k0 * m_G) &
        (local_std  >= k1 * sigma_G) &
        (local_std  <= k2 * sigma_G)
    )

    g = f.copy()
    g[cond] = C * f[cond]

    return np.clip(np.round(g), 0, L_MAX).astype(np.uint8)


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    # Intensity transformations
    'image_negative',
    'log_transform',
    'power_law_transform',
    'piecewise_linear_contrast_stretch',
    'gray_level_slicing',
    'bit_plane_slicing',
    # Histogram processing
    'compute_histogram',
    'histogram_equalization',
    'histogram_specification',
    'local_histogram_equalization',
    # Smoothing filters
    'box_filter',
    'gaussian_filter_spatial',
    'median_filter_custom',
    'max_filter',
    'min_filter',
    # Sharpening filters
    'laplacian_sharpen',
    'unsharp_masking',
    'sobel_gradient',
    'prewitt_gradient',
    # Local statistics enhancement
    'enhance_with_local_stats',
]
