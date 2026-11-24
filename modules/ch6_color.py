"""
Chapter 6: Color Image Processing
Based on: Gonzalez & Woods, "Digital Image Processing", 3rd Edition

Topics:
- Color models (RGB, HSI, HSV, CMY, CMYK)
- Pseudocolor processing
- Full-color image processing
- Color image smoothing and sharpening
- Color segmentation
"""
import numpy as np
from scipy.ndimage import gaussian_filter, convolve

__all__ = [
    "rgb_to_hsi",
    "hsi_to_rgb",
    "rgb_to_hsv",
    "hsv_to_rgb",
    "rgb_to_cmy",
    "cmy_to_cmyk",
    "intensity_slicing_pseudocolor",
    "continuous_pseudocolor",
    "color_complement",
    "color_slicing",
    "color_balance",
    "smooth_color_image",
    "sharpen_color_image",
    "segment_by_color",
]


# ===========================================================================
# SECTION 6.2–6.3 — COLOR MODELS
# ===========================================================================

def rgb_to_hsi(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSI color space.

    Input must be a float64 image with values in [0, 1].

    Formulae (Gonzalez & Woods, Section 6.2.3):

        theta = arccos( 0.5 * ((R-G) + (R-B))
                        / sqrt((R-G)^2 + (R-B)*(G-B) + eps) )

        H = theta          if B <= G
            360 - theta    if B  > G
        H normalized to [0, 1] by dividing by 360.

        S = 1 - 3 / (R + G + B + eps) * min(R, G, B)

        I = (R + G + B) / 3

    HSI separates chromatic (H, S) from achromatic (I) information:
        H: dominant wavelength / hue.
        S: purity (saturation) of the color.
        I: average brightness (intensity).
    Processing I alone preserves color fidelity.

    Parameters
    ----------
    image : np.ndarray
        Float64 RGB image of shape (H, W, 3), values in [0, 1].

    Returns
    -------
    np.ndarray
        Float64 HSI image of shape (H, W, 3).
        H: [0, 1]  (normalized from 0°–360°)
        S: [0, 1]
        I: [0, 1]
    """
    img = image.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    eps = 1e-10

    # Numerator and denominator for theta
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B) + eps)

    theta = np.arccos(np.clip(num / (den + eps), -1.0, 1.0))  # radians [0, pi]
    theta_deg = np.degrees(theta)                               # degrees [0, 180]

    # H in degrees [0, 360]
    H = np.where(B <= G, theta_deg, 360.0 - theta_deg)
    H = H / 360.0  # normalize to [0, 1]

    # Saturation
    total = R + G + B + eps
    S = 1.0 - (3.0 / total) * np.minimum(np.minimum(R, G), B)
    S = np.clip(S, 0.0, 1.0)

    # Intensity
    I = (R + G + B) / 3.0
    I = np.clip(I, 0.0, 1.0)

    return np.stack([H, S, I], axis=2)


def hsi_to_rgb(hsi_image: np.ndarray) -> np.ndarray:
    """Convert an HSI image back to RGB color space.

    Input must be float64 with:
        H: [0, 1] (will be converted to degrees [0°, 360°])
        S: [0, 1]
        I: [0, 1]

    Formulae (Gonzalez & Woods, Section 6.2.3):

    Sector 1 — H in [0°, 120°):
        B = I * (1 - S)
        R = I * (1 + S*cos(H) / cos(60° - H))
        G = 3*I - (R + B)

    Sector 2 — H in [120°, 240°):  (use H' = H - 120°)
        R = I * (1 - S)
        G = I * (1 + S*cos(H') / cos(60° - H'))
        B = 3*I - (R + G)

    Sector 3 — H in [240°, 360°):  (use H'' = H - 240°)
        G = I * (1 - S)
        B = I * (1 + S*cos(H'') / cos(60° - H''))
        R = 3*I - (G + B)

    Inverse mapping reconstructs RGB after processing in HSI space.

    Parameters
    ----------
    hsi_image : np.ndarray
        Float64 HSI image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Float64 RGB image of shape (H, W, 3), values clipped to [0, 1].
    """
    hsi = hsi_image.astype(np.float64)
    H_norm = hsi[:, :, 0]
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    H_deg = H_norm * 360.0  # convert to degrees

    R = np.zeros_like(H_deg)
    G = np.zeros_like(H_deg)
    B = np.zeros_like(H_deg)

    # --- Sector 1: 0° <= H < 120° ---
    mask1 = (H_deg >= 0) & (H_deg < 120)
    if mask1.any():
        H1 = np.radians(H_deg[mask1])
        s1 = S[mask1]
        i1 = I[mask1]
        cos60 = np.cos(np.radians(60) - H1)
        cosH = np.cos(H1)
        b1 = i1 * (1 - s1)
        r1 = i1 * (1 + s1 * cosH / (cos60 + 1e-10))
        g1 = 3 * i1 - (r1 + b1)
        R[mask1] = r1
        G[mask1] = g1
        B[mask1] = b1

    # --- Sector 2: 120° <= H < 240° ---
    mask2 = (H_deg >= 120) & (H_deg < 240)
    if mask2.any():
        H2 = np.radians(H_deg[mask2] - 120)
        s2 = S[mask2]
        i2 = I[mask2]
        cos60 = np.cos(np.radians(60) - H2)
        cosH = np.cos(H2)
        r2 = i2 * (1 - s2)
        g2 = i2 * (1 + s2 * cosH / (cos60 + 1e-10))
        b2 = 3 * i2 - (r2 + g2)
        R[mask2] = r2
        G[mask2] = g2
        B[mask2] = b2

    # --- Sector 3: 240° <= H < 360° ---
    mask3 = (H_deg >= 240) & (H_deg < 360)
    if mask3.any():
        H3 = np.radians(H_deg[mask3] - 240)
        s3 = S[mask3]
        i3 = I[mask3]
        cos60 = np.cos(np.radians(60) - H3)
        cosH = np.cos(H3)
        g3 = i3 * (1 - s3)
        b3 = i3 * (1 + s3 * cosH / (cos60 + 1e-10))
        r3 = 3 * i3 - (g3 + b3)
        R[mask3] = r3
        G[mask3] = g3
        B[mask3] = b3

    rgb = np.stack([R, G, B], axis=2)
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSV color space (standard algorithm).

    Input must be float64 with values in [0, 1].

    Formulae:
        V = max(R, G, B)
        S = (V - min(R,G,B)) / V   (0 when V = 0)

        H = 60 * (G - B) / (V - min)           if V = R
            60 * (B - R) / (V - min) + 120      if V = G
            60 * (R - G) / (V - min) + 240      if V = B
        H normalized to [0, 1] by dividing by 360.

    HSV vs HSI:
        V = max(R, G, B)       — Hexcone model, intuitive for artists.
        I = mean(R, G, B)      — Better suited for image processing math.

    Parameters
    ----------
    image : np.ndarray
        Float64 RGB image of shape (H, W, 3), values in [0, 1].

    Returns
    -------
    np.ndarray
        Float64 HSV image of shape (H, W, 3).
        H: [0, 1], S: [0, 1], V: [0, 1].
    """
    img = image.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    V = np.max(img, axis=2)
    C = V - np.min(img, axis=2)   # chroma

    # Saturation
    S = np.where(V > 0, C / (V + 1e-10), 0.0)

    # Hue (in degrees [0, 360])
    H = np.zeros_like(V)
    mask_r = (V == R) & (C > 0)
    mask_g = (V == G) & (C > 0)
    mask_b = (V == B) & (C > 0)

    H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / (C[mask_r] + 1e-10))) % 360
    H[mask_g] = (60 * ((B[mask_g] - R[mask_g]) / (C[mask_g] + 1e-10)) + 120) % 360
    H[mask_b] = (60 * ((R[mask_b] - G[mask_b]) / (C[mask_b] + 1e-10)) + 240) % 360

    H_norm = H / 360.0
    return np.stack([H_norm, S, V], axis=2)


def hsv_to_rgb(hsv_image: np.ndarray) -> np.ndarray:
    """Convert an HSV image back to RGB color space (standard algorithm).

    Input must be float64 with:
        H: [0, 1]  (will be converted to degrees [0°, 360°))
        S: [0, 1]
        V: [0, 1]

    Parameters
    ----------
    hsv_image : np.ndarray
        Float64 HSV image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Float64 RGB image of shape (H, W, 3), values clipped to [0, 1].
    """
    hsv = hsv_image.astype(np.float64)
    H_norm = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    H = H_norm * 360.0
    h_i = (H / 60).astype(int) % 6   # sector index 0–5
    f = H / 60 - np.floor(H / 60)    # fractional part

    p = V * (1 - S)
    q = V * (1 - f * S)
    t = V * (1 - (1 - f) * S)

    R = np.select(
        [h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i == 5],
        [V, q, p, p, t, V]
    )
    G = np.select(
        [h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i == 5],
        [t, V, V, q, p, p]
    )
    B = np.select(
        [h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i == 5],
        [p, p, t, V, V, q]
    )

    rgb = np.stack([R, G, B], axis=2)
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_cmy(image: np.ndarray) -> np.ndarray:
    """Convert a normalized RGB image to CMY color space.

    Formula (Gonzalez & Woods, Eq. 6.2-1):
        C = 1 - R
        M = 1 - G
        Y = 1 - B

    Subtractive color model used in printing.
    CMY channels are the complements of RGB:
        C (Cyan)    absorbs Red.
        M (Magenta) absorbs Green.
        Y (Yellow)  absorbs Blue.

    Parameters
    ----------
    image : np.ndarray
        Float64 RGB image of shape (H, W, 3), values in [0, 1].

    Returns
    -------
    np.ndarray
        Float64 CMY image of shape (H, W, 3), values in [0, 1].
    """
    img = np.clip(image.astype(np.float64), 0.0, 1.0)
    return 1.0 - img


def cmy_to_cmyk(cmy_image: np.ndarray) -> np.ndarray:
    """Convert a CMY image to CMYK color space.

    Formula (Gonzalez & Woods, Eq. 6.2-2):
        K  = min(C, M, Y)
        C' = (C - K) / (1 - K)
        M' = (M - K) / (1 - K)
        Y' = (Y - K) / (1 - K)

    When K = 1 (pure black): C' = M' = Y' = 0.

    Adding the Key/Black (K) channel saves ink and improves dark
    reproduction: pure black from equal CMY produces muddy brown in practice.

    Parameters
    ----------
    cmy_image : np.ndarray
        Float64 CMY image of shape (H, W, 3), values in [0, 1].

    Returns
    -------
    np.ndarray
        Float64 CMYK image of shape (H, W, 4), values in [0, 1].
    """
    cmy = np.clip(cmy_image.astype(np.float64), 0.0, 1.0)
    C = cmy[:, :, 0]
    M = cmy[:, :, 1]
    Y = cmy[:, :, 2]

    K = np.minimum(np.minimum(C, M), Y)
    denom = 1.0 - K + 1e-10

    C_prime = (C - K) / denom
    M_prime = (M - K) / denom
    Y_prime = (Y - K) / denom

    # When K=1, set remaining channels to 0
    C_prime = np.where(K >= 1.0, 0.0, C_prime)
    M_prime = np.where(K >= 1.0, 0.0, M_prime)
    Y_prime = np.where(K >= 1.0, 0.0, Y_prime)

    return np.stack([C_prime, M_prime, Y_prime, K], axis=2)


# ===========================================================================
# SECTION 6.4 — PSEUDOCOLOR IMAGE PROCESSING
# ===========================================================================

def intensity_slicing_pseudocolor(
    gray_image: np.ndarray,
    thresholds: list,
    colors: list,
) -> np.ndarray:
    """Apply intensity slicing pseudocolor mapping to a grayscale image.

    The gray range [0, 255] is divided by the thresholds into len(colors)
    intervals. Each interval is assigned a distinct color.

    Example with thresholds=[85, 170] and 3 colors:
        [0,  85)  → colors[0]
        [85, 170) → colors[1]
        [170,255] → colors[2]

    Enhances perception: human eye distinguishes thousands of colors
    but only ~20 distinct gray levels. Used in medical imaging, satellite
    imagery, and thermal maps.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale uint8 image of shape (H, W).
    thresholds : list of float
        Sorted boundary values [l1, l2, ..., ln-1].
        len(thresholds) must equal len(colors) - 1.
    colors : list of tuple
        List of (R, G, B) tuples (uint8 values, 0–255) for each interval.
        len(colors) = len(thresholds) + 1.

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape (H, W, 3).
    """
    gray = gray_image.astype(np.float64)
    H, W = gray.shape
    output = np.zeros((H, W, 3), dtype=np.uint8)

    # Build boundary list: -inf, l1, l2, ..., ln-1, +inf
    boundaries = [-np.inf] + list(thresholds) + [np.inf]

    for i, color in enumerate(colors):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        mask = (gray >= lo) & (gray < hi)
        for c_idx in range(3):
            output[:, :, c_idx][mask] = int(color[c_idx])

    # Handle the last boundary as inclusive
    if len(thresholds) > 0:
        mask_last = gray == 255
        for c_idx in range(3):
            output[:, :, c_idx][mask_last] = int(colors[-1][c_idx])

    return output


def continuous_pseudocolor(gray_image: np.ndarray) -> np.ndarray:
    """Apply sinusoidal continuous pseudocolor mapping to a grayscale image.

    Formula:
        I = pixel / 255   (normalized to [0, 1])
        R = sin(pi * I)
        G = sin(pi * I + pi/3)
        B = sin(pi * I + 2*pi/3)

    The sinusoidal functions create smooth, natural color transitions.
    Useful for thermal imaging, elevation maps, and scientific visualization.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale uint8 image of shape (H, W).

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape (H, W, 3).
    """
    I = gray_image.astype(np.float64) / 255.0
    R = np.sin(np.pi * I)
    G = np.sin(np.pi * I + np.pi / 3)
    B = np.sin(np.pi * I + 2 * np.pi / 3)
    # Sinusoidal values may be negative; take abs for display, then scale
    rgb = np.abs(np.stack([R, G, B], axis=2))
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


# ===========================================================================
# SECTION 6.5 — FULL-COLOR IMAGE PROCESSING
# ===========================================================================

def color_complement(image: np.ndarray) -> np.ndarray:
    """Compute the complement of a color image.

    For each channel c:
        c' = (L - 1) - c,  where L = 256 for uint8.

    Analogous to the photographic negative but applied independently to
    each RGB channel. Useful for enhancing white/light detail in dark
    regions of a color image.

    Parameters
    ----------
    image : np.ndarray
        uint8 RGB image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        uint8 RGB complement image of shape (H, W, 3).
    """
    return (255 - image.astype(np.uint8)).astype(np.uint8)


def color_slicing(
    image: np.ndarray,
    target_color: list,
    radius: float = 50,
) -> np.ndarray:
    """Isolate colors within a spherical region in RGB space.

    For each pixel z:
        D(z, a) = sqrt((z_R - a_R)^2 + (z_G - a_G)^2 + (z_B - a_B)^2)
        If D <= radius : keep original color.
        Else           : convert to grayscale (mean of channels).

    Used in product inspection, medical image analysis, and color-based
    object isolation. Euclidean distance in RGB space defines the region
    of interest.

    Parameters
    ----------
    image : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    target_color : array-like
        Target [R, G, B] color (uint8 values, 0–255).
    radius : float
        Maximum Euclidean distance from target_color to keep.

    Returns
    -------
    np.ndarray
        uint8 RGB image with non-target pixels converted to grayscale.
    """
    img = image.astype(np.float64)
    a = np.array(target_color, dtype=np.float64).reshape(1, 1, 3)

    # Euclidean distance in RGB space
    D = np.sqrt(np.sum((img - a) ** 2, axis=2))  # shape (H, W)

    # Grayscale version (mean of channels)
    gray = img.mean(axis=2, keepdims=True)
    gray_rgb = np.repeat(gray, 3, axis=2)

    # Keep original if within radius, else gray
    mask = D <= radius  # (H, W)
    output = np.where(mask[:, :, np.newaxis], img, gray_rgb)
    return np.clip(output, 0, 255).astype(np.uint8)


def color_balance(
    image: np.ndarray,
    r_gain: float = 1.0,
    g_gain: float = 1.0,
    b_gain: float = 1.0,
) -> np.ndarray:
    """Apply white balance correction by scaling individual color channels.

    Each channel is multiplied by its gain factor:
        R' = R * r_gain
        G' = G * g_gain
        B' = B * b_gain

    Compensates for different illuminant color temperatures (e.g. tungsten,
    fluorescent, daylight). Gains > 1 boost a channel; < 1 reduce it.

    Parameters
    ----------
    image : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    r_gain : float
        Gain factor for the Red channel.
    g_gain : float
        Gain factor for the Green channel.
    b_gain : float
        Gain factor for the Blue channel.

    Returns
    -------
    np.ndarray
        uint8 RGB image with adjusted white balance.
    """
    img = image.astype(np.float64)
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float64)
    result = img * gains[np.newaxis, np.newaxis, :]
    return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# SECTION 6.6 — COLOR IMAGE SMOOTHING AND SHARPENING
# ===========================================================================

def smooth_color_image(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    in_hsi: bool = True,
) -> np.ndarray:
    """Smooth a color image using a Gaussian filter.

    Two modes (Gonzalez & Woods, Section 6.6.1):

    in_hsi=True (recommended):
        1. Convert RGB → HSI.
        2. Apply Gaussian smoothing only to the Intensity (I) channel.
        3. Convert back HSI → RGB.
        → Preserves hue and saturation; avoids color shifts at edges.

    in_hsi=False:
        Apply Gaussian smoothing to each RGB channel independently.
        → May cause color shifts at high-contrast edges.

    Parameters
    ----------
    image : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    kernel_size : int
        Not directly used (sigma drives gaussian_filter); kept for API
        consistency with other filter functions.
    sigma : float
        Standard deviation of the Gaussian kernel.
    in_hsi : bool
        If True, smooth in HSI space; otherwise smooth in RGB space.

    Returns
    -------
    np.ndarray
        uint8 smoothed RGB image of shape (H, W, 3).
    """
    img_float = image.astype(np.float64) / 255.0

    if in_hsi:
        hsi = rgb_to_hsi(img_float)
        # Smooth only the Intensity channel
        hsi[:, :, 2] = gaussian_filter(hsi[:, :, 2], sigma=sigma)
        hsi[:, :, 2] = np.clip(hsi[:, :, 2], 0.0, 1.0)
        rgb = hsi_to_rgb(hsi)
    else:
        rgb = np.zeros_like(img_float)
        for c in range(3):
            rgb[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=sigma)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def sharpen_color_image(image: np.ndarray, in_hsi: bool = True) -> np.ndarray:
    """Sharpen a color image using the Laplacian operator.

    Laplacian sharpening (Gonzalez & Woods, Section 6.6.2):
        sharpened = original - c * Laplacian(original)
        where c = 1 (standard un-sharp masking / Laplacian sharpening).

    Laplacian kernel (4-connectivity):
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]

    Two modes:

    in_hsi=True (recommended):
        Apply Laplacian sharpening only to the Intensity (I) channel in HSI.
        → Same rationale as HSI smoothing: preserves hue/saturation.

    in_hsi=False:
        Apply Laplacian sharpening to each RGB channel independently.

    Parameters
    ----------
    image : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    in_hsi : bool
        If True, sharpen in HSI space; otherwise sharpen in RGB space.

    Returns
    -------
    np.ndarray
        uint8 sharpened RGB image of shape (H, W, 3).
    """
    laplacian_kernel = np.array(
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]], dtype=np.float64
    )

    img_float = image.astype(np.float64) / 255.0

    if in_hsi:
        hsi = rgb_to_hsi(img_float)
        I = hsi[:, :, 2]
        lap = convolve(I, laplacian_kernel, mode="reflect")
        # Laplacian enhancement: subtract Laplacian (using the sign convention
        # where the center coefficient is positive → subtract for sharpening)
        I_sharp = I - lap
        hsi[:, :, 2] = np.clip(I_sharp, 0.0, 1.0)
        rgb = hsi_to_rgb(hsi)
    else:
        rgb = np.zeros_like(img_float)
        for c in range(3):
            ch = img_float[:, :, c]
            lap = convolve(ch, laplacian_kernel, mode="reflect")
            rgb[:, :, c] = np.clip(ch - lap, 0.0, 1.0)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ===========================================================================
# SECTION 6.7 — COLOR SEGMENTATION
# ===========================================================================

def segment_by_color(
    image: np.ndarray,
    target_color: list,
    threshold: float = 50,
) -> np.ndarray:
    """Segment an image by color using Euclidean distance in RGB space.

    For each pixel z:
        D(z, a) = sqrt((z_R - a_R)^2 + (z_G - a_G)^2 + (z_B - a_B)^2)
        mask(z) = 1  if D < threshold
                  0  otherwise

    Simple color-based segmentation assuming a unimodal distribution of
    the target color. Threshold is in Euclidean RGB distance units.
    Commonly used in product inspection and medical image analysis.

    Parameters
    ----------
    image : np.ndarray
        uint8 RGB image of shape (H, W, 3).
    target_color : array-like
        Mean/representative [R, G, B] color of the region of interest
        (uint8 values, 0–255).
    threshold : float
        Maximum Euclidean RGB distance for a pixel to be included.

    Returns
    -------
    np.ndarray
        Binary uint8 mask of shape (H, W): 255 for target pixels, 0 otherwise.
    """
    img = image.astype(np.float64)
    a = np.array(target_color, dtype=np.float64).reshape(1, 1, 3)
    D = np.sqrt(np.sum((img - a) ** 2, axis=2))
    mask = (D < threshold).astype(np.uint8) * 255
    return mask
