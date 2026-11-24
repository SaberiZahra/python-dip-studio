"""
Chapter 5: Image Restoration & Noise Reduction
Based on: Gonzalez & Woods, "Digital Image Processing", 3rd Edition

Topics:
- Noise models (PDF-based generation)
- Spatial domain restoration filters (mean, order-statistic, adaptive)
- Frequency domain restoration (inverse, Wiener, CLS)
- Degradation model: g(x,y) = h(x,y)*f(x,y) + eta(x,y)
  In frequency: G(u,v) = H(u,v)*F(u,v) + N(u,v)
"""
import numpy as np
from scipy.ndimage import convolve, minimum_filter, maximum_filter, median_filter

__all__ = [
    # Noise generators
    "add_gaussian_noise",
    "add_rayleigh_noise",
    "add_erlang_noise",
    "add_exponential_noise",
    "add_uniform_noise",
    "add_salt_pepper_noise",
    "add_periodic_noise",
    # Spatial mean filters
    "arithmetic_mean_filter",
    "geometric_mean_filter",
    "harmonic_mean_filter",
    "contraharmonic_mean_filter",
    # Order-statistic filters
    "midpoint_filter",
    "alpha_trimmed_mean_filter",
    # Adaptive filters
    "adaptive_local_noise_filter",
    "adaptive_median_filter",
    # Frequency domain kernels & restoration
    "motion_blur_kernel",
    "gaussian_blur_kernel",
    "inverse_filter",
    "wiener_filter",
    "constrained_least_squares_filter",
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sliding_windows(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Return a view of all (kernel_size x kernel_size) sliding windows.

    Returns shape (H, W, kernel_size, kernel_size).
    """
    pad = kernel_size // 2
    img_pad = np.pad(image.astype(np.float64), pad, mode="reflect")
    H, W = image.shape
    # Build strided view
    shape = (H, W, kernel_size, kernel_size)
    strides = (img_pad.strides[0], img_pad.strides[1],
               img_pad.strides[0], img_pad.strides[1])
    windows = np.lib.stride_tricks.as_strided(img_pad, shape=shape, strides=strides)
    return windows  # shape (H, W, ks, ks)


# ===========================================================================
# SECTION 5.2 — NOISE MODELS (PDF-based generation)
# ===========================================================================

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Add Gaussian (Normal) noise to an image.

    eta ~ N(mean, sigma^2)

    Most common noise model. Results from electronic noise in sensors.
    Central Limit Theorem: sum of many independent noise sources → Gaussian.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    mean : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255] as uint8.
    """
    img = image.astype(np.float64)
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_rayleigh_noise(image: np.ndarray, a: float = 0, b: float = 2000) -> np.ndarray:
    """Add Rayleigh noise to an image.

    PDF: p(z) = (2/b) * (z - a) * exp(-(z - a)^2 / b)  for z >= a

    Generation via inverse CDF:
        eta = a + sqrt(-b * ln(1 - U)),  U ~ Uniform(0, 1)

    Mean = a + sqrt(pi * b / 4)

    Models range noise in radar and ultrasound imaging.
    Positively skewed distribution.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    a : float
        Location (minimum) parameter.
    b : float
        Scale parameter (b > 0).

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255] as uint8.
    """
    img = image.astype(np.float64)
    U = np.random.uniform(0, 1, img.shape)
    # Avoid log(0) by clipping U away from 1
    U = np.clip(U, 0, 1 - 1e-10)
    noise = a + np.sqrt(-b * np.log(1 - U))
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_erlang_noise(image: np.ndarray, a: float = 10, b: int = 5) -> np.ndarray:
    """Add Erlang (Gamma) noise to an image.

    PDF: p(z) = (a^b * z^(b-1) * exp(-a*z)) / (b-1)!  for z >= 0

    Generated using np.random.gamma(shape=b, scale=1/a).

    Models laser speckle and some medical imaging modalities.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    a : float
        Rate parameter (a > 0). Mean = b/a.
    b : int
        Shape parameter (positive integer for Erlang).

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255] as uint8.
    """
    img = image.astype(np.float64)
    noise = np.random.gamma(shape=b, scale=1.0 / a, size=img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_exponential_noise(image: np.ndarray, a: float = 0.1) -> np.ndarray:
    """Add Exponential noise to an image.

    PDF: p(z) = a * exp(-a * z)  for z >= 0

    Generated using np.random.exponential(scale=1/a).

    Special case of the Gamma distribution with b=1.
    Models certain types of coherent illumination noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    a : float
        Rate parameter (a > 0). Mean = 1/a.

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255] as uint8.
    """
    img = image.astype(np.float64)
    noise = np.random.exponential(scale=1.0 / a, size=img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_uniform_noise(image: np.ndarray, a: float = -50, b: float = 50) -> np.ndarray:
    """Add Uniform noise to an image.

    PDF: p(z) = 1/(b - a)  for a <= z <= b

    Models quantization noise resulting from A/D conversion.
    PDF is flat, so all values in [a, b] are equally likely.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    a : float
        Lower bound of the uniform distribution.
    b : float
        Upper bound of the uniform distribution (b > a).

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255] as uint8.
    """
    img = image.astype(np.float64)
    noise = np.random.uniform(a, b, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(
    image: np.ndarray, pa: float = 0.05, pb: float = 0.05
) -> np.ndarray:
    """Add Salt-and-Pepper (impulse) noise to an image.

    With probability pa: pixel = 0   (pepper / dark impulse)
    With probability pb: pixel = 255 (salt / light impulse)

    Caused by sharp/sudden disturbances: faulty sensors, timing errors,
    bit errors in transmission.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    pa : float
        Probability of pepper (black) noise.
    pb : float
        Probability of salt (white) noise.

    Returns
    -------
    np.ndarray
        Noisy image as uint8.
    """
    img = image.copy().astype(np.uint8)
    r = np.random.random(img.shape)
    img[r < pa] = 0           # pepper
    img[(r >= pa) & (r < pa + pb)] = 255  # salt
    return img


def add_periodic_noise(
    image: np.ndarray,
    amplitude: float = 20,
    u0: float = 0.1,
    v0: float = 0.1,
) -> np.ndarray:
    """Add sinusoidal periodic noise to an image.

    eta(x, y) = amplitude * sin(2*pi*u0*x + 2*pi*v0*y)

    u0, v0 are spatial frequencies in cycles per pixel (range 0 to 0.5).

    Caused by electrical or mechanical interference during acquisition.
    Appears as bright conjugate spots in the DFT magnitude spectrum.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    amplitude : float
        Amplitude of the sinusoidal pattern.
    u0 : float
        Horizontal frequency (cycles per pixel).
    v0 : float
        Vertical frequency (cycles per pixel).

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 255] as uint8.
    """
    img = image.astype(np.float64)
    H, W = img.shape
    x = np.arange(W)   # column indices → spatial x
    y = np.arange(H)   # row indices    → spatial y
    xx, yy = np.meshgrid(x, y)
    noise = amplitude * np.sin(2 * np.pi * u0 * xx + 2 * np.pi * v0 * yy)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ===========================================================================
# SECTION 5.3 — SPATIAL MEAN FILTERS
# ===========================================================================

def arithmetic_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply an Arithmetic Mean filter.

    For each m×n window:
        f_hat(x, y) = (1 / mn) * sum_{(r,c) in window} g(r, c)

    Blurs noise but also blurs edges. Best suited for Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    kernel_size : int
        Side length of the square window (must be odd).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    mn = kernel_size * kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / mn
    img = image.astype(np.float64)
    filtered = convolve(img, kernel, mode="reflect")
    return np.clip(filtered, 0, 255).astype(np.uint8)


def geometric_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a Geometric Mean filter.

    For each m×n window:
        f_hat(x, y) = ( prod_{(r,c) in window} g(r, c) )^(1/mn)

    A small epsilon is added to avoid log(0).

    Achieves similar smoothing to the arithmetic mean but loses less detail.
    Performs better than arithmetic mean for some noise distributions.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    kernel_size : int
        Side length of the square window (must be odd).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    eps = 1e-6
    mn = kernel_size * kernel_size
    windows = _sliding_windows(image, kernel_size)  # (H, W, ks, ks)
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    log_sum = np.sum(np.log(flat + eps), axis=2)
    result = np.exp(log_sum / mn)
    return np.clip(result, 0, 255).astype(np.uint8)


def harmonic_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a Harmonic Mean filter.

    For each m×n window:
        f_hat(x, y) = mn / sum_{(r,c) in window} (1 / g(r, c))

    Epsilon added to denominators to avoid division by zero.

    Works well for salt (white) noise but fails for pepper (zero) noise.
    Also effective for Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    kernel_size : int
        Side length of the square window (must be odd).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    eps = 1e-6
    mn = kernel_size * kernel_size
    windows = _sliding_windows(image, kernel_size)
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    result = mn / np.sum(1.0 / (flat + eps), axis=2)
    return np.clip(result, 0, 255).astype(np.uint8)


def contraharmonic_mean_filter(
    image: np.ndarray, kernel_size: int = 3, Q: float = 1.5
) -> np.ndarray:
    """Apply a Contraharmonic Mean filter.

    For each m×n window:
        f_hat(x, y) = sum(g^(Q+1)) / sum(g^Q)

    Special cases:
        Q = 0  → Arithmetic mean
        Q = -1 → Harmonic mean
        Q > 0  → Eliminates pepper (dark) noise
        Q < 0  → Eliminates salt (white) noise

    Cannot simultaneously remove both salt and pepper noise.
    Versatile filter controlled by the order Q.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    kernel_size : int
        Side length of the square window (must be odd).
    Q : float
        Order of the filter.

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    eps = 1e-6
    windows = _sliding_windows(image, kernel_size)
    flat = (windows.reshape(windows.shape[0], windows.shape[1], -1) + eps)
    numerator = np.sum(flat ** (Q + 1), axis=2)
    denominator = np.sum(flat ** Q, axis=2)
    result = numerator / (denominator + eps)
    return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# SECTION 5.3.2 — ORDER-STATISTIC FILTERS
# ===========================================================================

def midpoint_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a Midpoint filter.

    For each m×n window:
        f_hat(x, y) = (z_max + z_min) / 2

    Works best for uniformly or Gaussian-distributed noise.
    The midpoint of the range is a reasonable estimate of central tendency.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    kernel_size : int
        Side length of the square window (must be odd).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    windows = _sliding_windows(image, kernel_size)
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    z_min = flat.min(axis=2)
    z_max = flat.max(axis=2)
    result = (z_min + z_max) / 2.0
    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_trimmed_mean_filter(
    image: np.ndarray, kernel_size: int = 3, d: int = 2
) -> np.ndarray:
    """Apply an Alpha-Trimmed Mean filter.

    For each m×n window:
        1. Sort the mn pixel values.
        2. Delete the d//2 lowest and d//2 highest values.
        3. Average the remaining mn - d values.

    Special cases:
        d = 0        → Arithmetic mean filter
        d = mn - 1   → Median filter

    Bridges median and arithmetic mean filtering.
    Effective for mixed noise (Gaussian + impulse / salt-and-pepper).

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    kernel_size : int
        Side length of the square window (must be odd).
    d : int
        Total number of extreme values to trim (must be even, 0 <= d < mn).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    mn = kernel_size * kernel_size
    trim = d // 2
    windows = _sliding_windows(image, kernel_size)
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    flat_sorted = np.sort(flat, axis=2)
    if trim > 0:
        trimmed = flat_sorted[:, :, trim: mn - trim]
    else:
        trimmed = flat_sorted
    result = trimmed.mean(axis=2)
    return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# SECTION 5.3.3 — ADAPTIVE FILTERS
# ===========================================================================

def adaptive_local_noise_filter(
    image: np.ndarray, noise_variance: float, kernel_size: int = 7
) -> np.ndarray:
    """Apply an Adaptive Local Noise Reduction filter.

    At each pixel (x, y):
        1. Compute local mean m_L and local variance sigma_L^2 in the window.
        2. Compute ratio = noise_var / sigma_L^2  (clamped to [0, 1]).
        3. Output: g(x,y) - ratio * (g(x,y) - m_L)

    Behaviour:
        - Local variance >> noise variance → ratio ≈ 0 → preserve pixel (edge/detail).
        - Local variance ≈ noise variance  → ratio ≈ 1 → heavy smoothing (flat region).

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    noise_variance : float
        Estimated global noise variance (sigma_eta^2).
    kernel_size : int
        Side length of the square local window (must be odd).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    img = image.astype(np.float64)
    windows = _sliding_windows(image, kernel_size)
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    m_L = flat.mean(axis=2)        # local mean
    sigma_L2 = flat.var(axis=2)    # local variance

    # Clamp ratio to [0, 1] when sigma_L^2 < noise_var (avoids over-correction)
    ratio = noise_variance / (sigma_L2 + 1e-10)
    ratio = np.clip(ratio, 0.0, 1.0)

    result = img - ratio * (img - m_L)
    return np.clip(result, 0, 255).astype(np.uint8)


def adaptive_median_filter(image: np.ndarray, s_max: int = 7) -> np.ndarray:
    """Apply an Adaptive Median filter (Algorithm 5.1, Gonzalez & Woods).

    For each pixel, starts with a 3×3 window and grows up to s_max × s_max.

    Level A — Find a valid (non-impulse) median:
        z_min = minimum value in window
        z_med = median value in window
        z_max = maximum value in window
        If z_min < z_med < z_max: proceed to Level B.
        Else: grow window size and repeat Level A (up to s_max).
              If window reaches s_max: output z_med regardless.

    Level B — Decide whether the current pixel is an impulse:
        z_xy  = current pixel value
        If z_min < z_xy < z_max: output z_xy (pixel is NOT an impulse).
        Else: output z_med       (pixel IS an impulse, replace with median).

    Advantages over standard median filter:
        - Only replaces pixels that are detected as impulses (preserves edges).
        - Adaptively enlarges the window to find a non-impulse median.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    s_max : int
        Maximum allowed window size (must be odd and >= 3).

    Returns
    -------
    np.ndarray
        Filtered image as uint8.
    """
    img = image.astype(np.uint8)
    result = np.zeros_like(img)
    unresolved = np.ones(img.shape, dtype=bool)

    for s in range(3, s_max + 1, 2):
        if not np.any(unresolved):
            break
        z_min = minimum_filter(img, size=s, mode='reflect')
        z_max = maximum_filter(img, size=s, mode='reflect')
        z_med = median_filter(img, size=s, mode='reflect')

        # Level A: Check if median is non-impulse
        valid_med = (z_med > z_min) & (z_med < z_max)
        can_decide = unresolved & (valid_med | (s == s_max))

        # Level B: Check if original pixel is non-impulse
        not_impulse = (img > z_min) & (img < z_max)
        result[can_decide & not_impulse] = img[can_decide & not_impulse]
        result[can_decide & ~not_impulse] = z_med[can_decide & ~not_impulse]

        unresolved[can_decide] = False

    if np.any(unresolved):
        result[unresolved] = img[unresolved]

    return result.astype(np.uint8)


# ===========================================================================
# SECTION 5.5–5.9 — FREQUENCY DOMAIN RESTORATION
# ===========================================================================

def motion_blur_kernel(
    M: int, N: int, T: float = 1.0, a: float = 0.1, b: float = 0.1
) -> np.ndarray:
    """Compute the frequency-domain motion blur degradation function H(u,v).

    Formula (Gonzalez & Woods, Eq. 5.6-10):
        phi(u,v) = pi * (u*a + v*b)
        H(u,v)   = (T / phi) * sin(phi) * exp(-j * phi)

    When phi = 0: H(u,v) = T  (L'Hôpital limit).

    Models uniform linear motion during exposure.
        a = horizontal velocity component (pixels/frame).
        b = vertical velocity component   (pixels/frame).
        T = exposure time.

    Frequencies u, v are in the range [-0.5, 0.5] (normalized),
    consistent with np.fft.fftfreq output.

    Parameters
    ----------
    M : int
        Height of the output kernel (matches image height).
    N : int
        Width of the output kernel (matches image width).
    T : float
        Exposure time.
    a : float
        Horizontal motion parameter (velocity × time, in pixels).
    b : float
        Vertical motion parameter.

    Returns
    -------
    np.ndarray
        Complex-valued degradation function H of shape (M, N).
    """
    u = np.fft.fftfreq(M)   # range [-0.5, 0.5)
    v = np.fft.fftfreq(N)
    V, U = np.meshgrid(v, u)   # U ~ rows (height), V ~ cols (width)

    phi = np.pi * (U * a + V * b)

    # Avoid division by zero; will fix those pixels below
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(
            phi == 0,
            T + 0j,
            (T / phi) * np.sin(phi) * np.exp(-1j * phi),
        )
    return H.astype(np.complex128)


def gaussian_blur_kernel(M: int, N: int, sigma: float = 5) -> np.ndarray:
    """Compute the frequency-domain Gaussian blur degradation function H(u,v).

    Formula:
        D(u,v) = sqrt(u^2 + v^2)   (Euclidean distance from center)
        H(u,v) = exp(-D^2 / (2 * sigma^2))

    Models atmospheric turbulence blur (and is widely used as a general
    low-pass degradation for Wiener filter demonstrations).

    Parameters
    ----------
    M : int
        Height of the output kernel.
    N : int
        Width of the output kernel.
    sigma : float
        Standard deviation of the Gaussian (larger → more blur).

    Returns
    -------
    np.ndarray
        Real-valued (cast to complex128) degradation function H of shape (M, N).
    """
    u = np.fft.fftfreq(M)
    v = np.fft.fftfreq(N)
    V, U = np.meshgrid(v, u)
    D2 = U ** 2 + V ** 2
    H = np.exp(-D2 / (2 * sigma ** 2))
    return H.astype(np.complex128)


def inverse_filter(
    G: np.ndarray, H: np.ndarray, threshold: float = 0.1
) -> np.ndarray:
    """Apply a truncated Inverse filter in the frequency domain.

    F_hat(u,v) = G(u,v) / H(u,v)

    When |H(u,v)| < threshold: set F_hat(u,v) = 0
    (avoids catastrophic noise amplification where H ≈ 0).

    Direct inversion but very sensitive to noise. When H ≈ 0, division
    amplifies noise to infinity. The threshold truncates the filter
    at spectral components with weak response.

    Parameters
    ----------
    G : np.ndarray
        DFT of the degraded image, shape (M, N), complex.
    H : np.ndarray
        Degradation function, shape (M, N), complex.
    threshold : float
        Minimum magnitude of H below which output is set to zero.

    Returns
    -------
    np.ndarray
        Restored image as uint8.
    """
    H_abs = np.abs(H)
    # Avoid divide by zero; pixels below threshold are zeroed after
    H_safe = np.where(H_abs >= threshold, H, 1.0 + 0j)
    F_hat = np.where(H_abs >= threshold, G / H_safe, 0.0 + 0j)
    f_hat = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_hat, 0, 255).astype(np.uint8)


def wiener_filter(G: np.ndarray, H: np.ndarray, K: float = 0.01) -> np.ndarray:
    """Apply a Wiener (MMSE) filter in the frequency domain.

    Formula (Gonzalez & Woods, Eq. 5.8-4):
        F_hat(u,v) = [ H*(u,v) / (|H(u,v)|^2 + K) ] * G(u,v)

    K ≈ Sn(u,v) / Sf(u,v)  — ratio of noise power spectrum to signal
    power spectrum (assumed constant here for simplicity).

    Special cases:
        K = 0        → Inverse filter
        K → ∞        → Zero output (maximum smoothing)
        Optimal K    = noise_power / signal_power

    Minimum Mean Square Error (MMSE) filter. Balances sharpness against
    noise amplification. Preferred over inverse filter when noise is present.

    Parameters
    ----------
    G : np.ndarray
        DFT of the degraded image, shape (M, N), complex.
    H : np.ndarray
        Degradation function, shape (M, N), complex.
    K : float
        Noise-to-signal power ratio (regularization parameter).

    Returns
    -------
    np.ndarray
        Restored image as uint8.
    """
    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2
    W = H_conj / (H_abs2 + K)
    F_hat = W * G
    f_hat = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_hat, 0, 255).astype(np.uint8)


def constrained_least_squares_filter(
    G: np.ndarray, H: np.ndarray, gamma: float = 0.001
) -> np.ndarray:
    """Apply a Constrained Least Squares (CLS) filter in the frequency domain.

    Formula (Gonzalez & Woods, Eq. 5.9-6):
        P(u,v) = DFT{ p(x,y) },  where p(x,y) is the Laplacian kernel:
                 [[0, -1,  0],
                  [-1,  4, -1],
                  [0, -1,  0]]
        padded to the same size as G.

        F_hat(u,v) = [ H*(u,v) / (|H(u,v)|^2 + gamma * |P(u,v)|^2) ] * G(u,v)

    The Laplacian P penalizes rough (noisy) solutions.
    gamma controls the degree of regularization:
        - Larger gamma → smoother (more regularized) result.
        - gamma → 0    → Inverse filter (no regularization).

    Parameters
    ----------
    G : np.ndarray
        DFT of the degraded image, shape (M, N), complex.
    H : np.ndarray
        Degradation function, shape (M, N), complex.
    gamma : float
        Regularization parameter controlling smoothness.

    Returns
    -------
    np.ndarray
        Restored image as uint8.
    """
    M, N = G.shape

    # Laplacian kernel (Gonzalez & Woods, p. 358)
    p = np.array([[0, -1,  0],
                  [-1,  4, -1],
                  [0, -1,  0]], dtype=np.float64)

    # Pad Laplacian kernel to size (M, N), centred at top-left
    p_padded = np.zeros((M, N), dtype=np.float64)
    ph, pw = p.shape
    p_padded[:ph, :pw] = p

    P = np.fft.fft2(p_padded)

    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2
    P_abs2 = np.abs(P) ** 2

    F_hat = (H_conj / (H_abs2 + gamma * P_abs2)) * G
    f_hat = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_hat, 0, 255).astype(np.uint8)
