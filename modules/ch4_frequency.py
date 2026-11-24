"""
Chapter 4: Filtering in the Frequency Domain
Based on: Gonzalez & Woods, "Digital Image Processing", 3rd Edition

This module implements the complete frequency-domain filtering pipeline
as described in Section 4.7 of the textbook, plus all standard filters.

Standard Processing Steps (Section 4.7):
1. Load image f(x,y) of size M x N
2. Pad to P x Q (P=2M, Q=2N) to avoid wraparound error
3. Multiply padded image by (-1)^(x+y) to center DFT
4. Compute 2D DFT: F(u,v) = DFT{fp(x,y) * (-1)^(x+y)}
5. Construct filter H(u,v) of same size P x Q
6. G(u,v) = H(u,v) * F(u,v)  [element-wise multiplication]
7. Compute IDFT and multiply by (-1)^(x+y): gp(x,y) = real(IDFT{G}) * (-1)^(x+y)
8. Crop top-left M x N portion to get final result g(x,y)

References:
    Gonzalez, R.C. & Woods, R.E. (2008). Digital Image Processing (3rd ed.).
    Prentice Hall. Chapter 4, Sections 4.1–4.10.
"""

import numpy as np
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

__author__ = "Generated for DIP Course (Gonzalez & Woods, 3rd Ed.)"
__version__ = "1.0.0"


# ===========================================================================
# SECTION 1: CORE PIPELINE  (Gonzalez & Woods, Section 4.7)
# ===========================================================================


def pad_image(image: np.ndarray) -> np.ndarray:
    """
    Zero-pad a 2-D image from (M, N) to (2M, 2N).

    **Why padding?**
    The 2-D DFT / IDFT pair implicitly assumes the signal is *periodic*.
    When we multiply two DFTs and invert, the resulting spatial-domain
    product is a *circular* (wrap-around) convolution.  For linear
    (non-circular) convolution – which is what we want – the padded size
    must satisfy P >= M + m − 1 and Q >= N + n − 1, where (m, n) is the
    filter size.  Using P = 2M, Q = 2N is the standard conservative choice
    (Section 4.7, Gonzalez & Woods).

    Without padding, edges of the filtered image are contaminated by
    wrap-around energy ("ghosting artefacts").

    Args:
        image (np.ndarray): Grayscale 2-D image of shape (M, N).
                            Any numeric dtype; will be converted to float64.

    Returns:
        np.ndarray: Zero-padded array of shape (2M, 2N), dtype float64.
                    The original image occupies the top-left (M, N) block.

    Example:
        >>> import numpy as np
        >>> img = np.ones((4, 6), dtype=np.uint8)
        >>> padded = pad_image(img)
        >>> padded.shape
        (8, 12)
    """
    image = np.asarray(image, dtype=np.float64)
    M, N = image.shape[:2]
    P, Q = 2 * M, 2 * N
    if image.ndim == 2:
        padded = np.zeros((P, Q), dtype=np.float64)
        padded[:M, :N] = image
    else:
        raise ValueError("pad_image expects a 2-D (grayscale) image.")
    return padded


def compute_dft(image: np.ndarray) -> dict:
    """
    Compute the 2-D Discrete Fourier Transform using the full Section 4.7
    pipeline: pad → center via (-1)^(x+y) → FFT.

    **Centering trick (Section 4.6.3):**
    Multiplying the spatial-domain image by (-1)^(x+y) before the FFT is
    mathematically equivalent to applying ``np.fft.fftshift`` after the FFT.
    This places the DC component (zero frequency) at the centre of the
    array, making filter design intuitive (distance from centre = spatial
    frequency).

        F_centered(u, v) = FFT{ f_padded(x,y) · (−1)^(x+y) }

    **Log-magnitude spectrum:**
    The dynamic range of |F| is typically enormous.  A log transform
    compresses it for display:

        S(u, v) = c · log(1 + |F(u, v)|)

    where c is chosen so the spectrum spans [0, 255] (uint8 display range).

    Args:
        image (np.ndarray): 2-D grayscale image of shape (M, N).

    Returns:
        dict with keys:
            * ``'F'``          – complex128 DFT array, shape (2M, 2N),
                                 DC at centre.
            * ``'spectrum'``   – float64 log-magnitude scaled to [0, 255].
            * ``'phase'``      – float64 phase in radians, shape (2M, 2N).
            * ``'padded_size'``– tuple (P, Q) = (2M, 2N).

    Example:
        >>> result = compute_dft(np.eye(8))
        >>> result['F'].shape
        (16, 16)
    """
    image = np.asarray(image, dtype=np.float64)
    M, N = image.shape

    # Step 1 – zero-pad to P x Q
    fp = pad_image(image)          # shape (2M, 2N)
    P, Q = fp.shape

    # Step 2 – centre DFT by multiplying by (-1)^(x+y)
    x = np.arange(P).reshape(-1, 1)   # column vector
    y = np.arange(Q).reshape(1, -1)   # row vector
    centering = (-1.0) ** (x + y)
    fp_centered = fp * centering       # shape (P, Q)

    # Step 3 – 2-D FFT
    F = np.fft.fft2(fp_centered)      # complex128, shape (P, Q)

    # Log-magnitude spectrum, normalised to [0, 255]
    magnitude = np.abs(F)
    log_mag = np.log1p(magnitude)      # log(1 + |F|)
    max_val = log_mag.max()
    if max_val > 0:
        spectrum = (log_mag / max_val) * 255.0
    else:
        spectrum = log_mag.copy()

    # Phase spectrum
    phase = np.arctan2(F.imag, F.real)

    return {
        'F': F,
        'spectrum': spectrum,
        'phase': phase,
        'padded_size': (P, Q),
    }


def compute_distance_matrix(P: int, Q: int) -> np.ndarray:
    """
    Compute the Euclidean distance from each point (u, v) to the centre
    (P/2, Q/2) of a P×Q grid.

    **Definition (Section 4.7):**

        D(u, v) = sqrt( (u − P/2)^2 + (v − Q/2)^2 )

    With the DFT centred at (P/2, Q/2), D(u,v) directly represents the
    *radial spatial frequency* at grid point (u, v).  All circularly
    symmetric frequency-domain filters are expressed as functions of D.

    Args:
        P (int): Number of rows (height of padded image).
        Q (int): Number of columns (width of padded image).

    Returns:
        np.ndarray: float64 array of shape (P, Q) containing D(u, v).

    Example:
        >>> D = compute_distance_matrix(4, 4)
        >>> D[2, 2]   # centre point
        0.0
    """
    u = np.arange(P, dtype=np.float64).reshape(-1, 1)
    v = np.arange(Q, dtype=np.float64).reshape(1, -1)
    D = np.sqrt((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2)
    return D


def apply_filter(
    F_centered: np.ndarray,
    H: np.ndarray,
    original_shape: tuple,
) -> np.ndarray:
    """
    Apply a frequency-domain filter and return the filtered spatial image.

    Implements Steps 6–8 of the Section 4.7 pipeline:

    **Step 6** – Multiply:
        G(u, v) = H(u, v) · F(u, v)   (element-wise)

    **Step 7** – Inverse DFT and de-centre:
        g_p(x, y) = real{ IFFT2{ G(u, v) } } · (−1)^(x+y)

    The (−1)^(x+y) multiplication undoes the centering applied before the
    forward DFT (Gonzalez & Woods Eq. 4.6-18).

    **Step 8** – Crop:
        g(x, y) = g_p[0:M, 0:N]   (top-left block)

    The result is clipped to [0, 255] and cast to uint8 for display.

    Args:
        F_centered (np.ndarray): Complex DFT array (output of ``compute_dft``
                                 key ``'F'``), shape (P, Q).
        H (np.ndarray):          Real-valued filter array, shape (P, Q),
                                 values in [0, 1] (or any real gain).
        original_shape (tuple):  (M, N) – spatial size to crop to.

    Returns:
        np.ndarray: Filtered image, dtype uint8, shape (M, N),
                    pixel values clipped to [0, 255].

    Example:
        >>> result = compute_dft(img)
        >>> H = gaussian_lpf(*result['padded_size'], D0=30)
        >>> out = apply_filter(result['F'], H, img.shape)
    """
    P, Q = F_centered.shape
    M, N = original_shape

    # Step 6 – filter in frequency domain
    G = H * F_centered                       # element-wise, complex result

    # Step 7 – inverse FFT
    g_padded = np.fft.ifft2(G).real          # take real part

    # De-centre: multiply by (-1)^(x+y)
    x = np.arange(P, dtype=np.float64).reshape(-1, 1)
    y = np.arange(Q, dtype=np.float64).reshape(1, -1)
    centering = (-1.0) ** (x + y)
    g_padded *= centering

    # Step 8 – crop to original size and clip
    g = g_padded[:M, :N]
    g = np.clip(g, 0, 255)
    return g.astype(np.uint8)


# ===========================================================================
# SECTION 2: LOWPASS FILTERS  (Section 4.4)
# ===========================================================================


def ideal_lpf(P: int, Q: int, D0: float) -> np.ndarray:
    """
    Ideal (Brick-wall) Lowpass Filter.

    **Definition (Eq. 4.4-5):**

        H(u, v) = 1   if D(u, v) ≤ D0
                  0   otherwise

    where D(u, v) is the distance from the DFT centre and D0 is the
    *cutoff frequency*.

    **Why it is NOT used in practice – Gibbs phenomenon:**
    The sharp discontinuity at D = D0 is equivalent (via the IDFT) to
    convolving the image with a 2-D *sinc* function, which has significant
    negative side-lobes.  These lobes produce visible ringing (Gibbs
    phenomenon) around sharp edges – a classic artefact demonstrated in
    Gonzalez & Woods Figure 4.19.

    Args:
        P   (int):   Padded image height.
        Q   (int):   Padded image width.
        D0  (float): Cutoff radius (pixels in the frequency domain).

    Returns:
        np.ndarray: float64 filter array of shape (P, Q) with values in {0, 1}.
    """
    D = compute_distance_matrix(P, Q)
    H = np.where(D <= D0, 1.0, 0.0)
    return H.astype(np.float64)


def butterworth_lpf(P: int, Q: int, D0: float, n: int = 2) -> np.ndarray:
    """
    Butterworth Lowpass Filter (BLF) of order *n*.

    **Definition (Eq. 4.4-10):**

        H(u, v) = 1 / (1 + (D(u,v) / D0)^(2n))

    **Properties (Section 4.4.2):**
    *   At D = D0: H = 0.5 (3-dB point), regardless of order.
    *   The passband is *maximally flat* – there are no ripples.
    *   As n → ∞ the BLF approaches the ideal LPF and ringing appears.
    *   n = 1 or 2: practically no ringing, smooth spatial impulse response.
    *   Unlike the ideal LPF, the BLF has no abrupt transition, so its
        spatial impulse response decays without large negative lobes.

    Args:
        P   (int):   Padded image height.
        Q   (int):   Padded image width.
        D0  (float): Cutoff frequency (D where H = 0.5).
        n   (int):   Filter order.  Default is 2.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in (0, 1].
    """
    D = compute_distance_matrix(P, Q)
    # Avoid division by zero at D=0 when D0=0 (edge case)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(D0 > 0, D / D0, 0.0)
    H = 1.0 / (1.0 + ratio ** (2 * n))
    return H.astype(np.float64)


def gaussian_lpf(P: int, Q: int, D0: float) -> np.ndarray:
    """
    Gaussian Lowpass Filter (GLF).

    **Definition (Eq. 4.4-16):**

        H(u, v) = exp( −D^2(u,v) / (2 · D0^2) )

    **Properties (Section 4.4.3):**
    *   The Fourier transform of a Gaussian is itself a Gaussian.
      Therefore, the spatial-domain impulse response is also Gaussian.
    *   A Gaussian has **no negative lobes** → **no ringing** (Gibbs).
    *   H = e^(−1/2) ≈ 0.607 at D = D0 (not the 0.5 / 3-dB convention).
    *   Provides the smoothest possible transition between pass and stop
      bands at the cost of a less sharp cutoff than BLF of order ≥ 2.

    Args:
        P   (int):   Padded image height.
        Q   (int):   Padded image width.
        D0  (float): Cutoff frequency (H = 0.607 at D = D0).

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in (0, 1].
    """
    D = compute_distance_matrix(P, Q)
    H = np.exp(-(D ** 2) / (2.0 * D0 ** 2))
    return H.astype(np.float64)


# ===========================================================================
# SECTION 3: HIGHPASS FILTERS  (Section 4.5)
# ===========================================================================


def ideal_hpf(P: int, Q: int, D0: float) -> np.ndarray:
    """
    Ideal Highpass Filter.

    **Relationship to Ideal LPF (Eq. 4.5-1):**

        H_HP(u, v) = 1 − H_LP(u, v)

    So H_HP = 0 if D ≤ D0, 1 otherwise.

    **Effect:**
    Removes the low-frequency DC component and slowly varying illumination.
    Passes fine detail (edges, texture).  Suffers from the **same ringing
    artefacts** as the ideal LPF (Gibbs phenomenon), because the
    complementary filter still has a sharp discontinuity at D0.

    Args:
        P   (int):   Padded image height.
        Q   (int):   Padded image width.
        D0  (float): Cutoff radius.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in {0, 1}.
    """
    return 1.0 - ideal_lpf(P, Q, D0)


def butterworth_hpf(P: int, Q: int, D0: float, n: int = 2) -> np.ndarray:
    """
    Butterworth Highpass Filter (BHF) of order *n*.

    **Definition (Eq. 4.5-5):**

        H_HP(u, v) = 1 / (1 + (D0 / D(u,v))^(2n))

    Equivalently: ``1 − butterworth_lpf(P, Q, D0, n)``.

    Special case D = 0 (DC component): H is set to 0 to remove the DC
    term entirely (consistent with Gonzalez & Woods convention).

    **Properties (Section 4.5.2):**
    *   Smooth transition; for n = 1, 2 there is essentially no ringing.
    *   At D = D0, H = 0.5 (same 3-dB cutoff convention as BLF).
    *   Higher n → sharper transition but increasing ringing tendency.
    *   Commonly used in edge detection pre-processing and feature
      extraction pipelines.

    Args:
        P   (int):   Padded image height.
        Q   (int):   Padded image width.
        D0  (float): Cutoff frequency (D where H = 0.5).
        n   (int):   Filter order.  Default is 2.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in [0, 1).
    """
    D = compute_distance_matrix(P, Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(D > 0, D0 / D, np.inf)
    H = 1.0 / (1.0 + ratio ** (2 * n))
    # Explicitly zero the DC term (D=0 → H=0)
    H[D == 0] = 0.0
    return H.astype(np.float64)


def gaussian_hpf(P: int, Q: int, D0: float) -> np.ndarray:
    """
    Gaussian Highpass Filter (GHF).

    **Definition (Eq. 4.5-9):**

        H_HP(u, v) = 1 − exp( −D^2(u,v) / (2 · D0^2) )

    Equivalently: ``1 − gaussian_lpf(P, Q, D0)``.

    **Properties & Relationship to LoG (Section 4.5.3):**
    *   The GHF is **ringing-free** (Gaussian spatial impulse response).
    *   For small D0 (narrow pass-band centred on DC subtracted), the
      filter approximates the **Laplacian of Gaussian (LoG)** operator:
      both highlight edges via second-order differentiation.
    *   The spatial-domain equivalent is an image minus its Gaussian-
      smoothed version, i.e. an *unsharp masking* operation.

    Args:
        P   (int):   Padded image height.
        Q   (int):   Padded image width.
        D0  (float): Cutoff frequency.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in [0, 1).
    """
    return 1.0 - gaussian_lpf(P, Q, D0)


# ===========================================================================
# SECTION 4: ADVANCED / DERIVATIVE FILTERS  (Section 4.6 & 4.9)
# ===========================================================================


def laplacian_frequency(P: int, Q: int) -> np.ndarray:
    """
    Laplacian Filter in the Frequency Domain.

    **Definition (Eq. 4.6-9):**

        H(u, v) = −4π² · D²(u, v)

    **Mathematical basis:**
    Differentiation in the spatial domain corresponds to multiplication
    by (j·2π·D) in the frequency domain.  The *Laplacian* is the sum of
    second-order partial derivatives, so its frequency-domain equivalent
    is multiplication by:

        (j·2πu)·(j·2πu) + (j·2πv)·(j·2πv) = −4π²(u² + v²) = −4π²D²

    **Image enhancement recipe (Eq. 4.6-13):**

        g(x,y) = f(x,y) − c · IDFT{ H(u,v) · F(u,v) }

    where c = 1 when H as defined above (the Laplacian sharpens edges;
    subtracting it from f restores the low-frequency information while
    enhancing edges).

    **Display note:**
    The raw H values are very large.  For visualisation, normalise by
    dividing by ``|H|.max()``.  The normalised array is returned here;
    for actual filtering, multiply by the raw scale factor.

    Args:
        P (int): Padded image height.
        Q (int): Padded image width.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q) with values
                    H(u,v) = −4π²·D²(u,v).  (Raw, un-normalised.)
    """
    D = compute_distance_matrix(P, Q)
    H = -4.0 * (np.pi ** 2) * (D ** 2)
    return H.astype(np.float64)


def high_freq_emphasis(
    P: int,
    Q: int,
    D0: float,
    a: float = 0.5,
    b: float = 0.75,
    filter_type: str = 'butterworth',
    n: int = 2,
) -> np.ndarray:
    """
    High-Frequency Emphasis (HFE) Filter.

    **Definition (Eq. 4.9-9):**

        H_HFE(u, v) = a + b · H_HP(u, v)

    where H_HP is a highpass filter and a, b are constants with a ≥ 0,
    b > a.

    **Motivation (Section 4.9.1):**
    A pure highpass filter forces H(0,0) = 0, which zeroes the DC
    component and produces a dark image with no mean value.  HFE adds an
    offset *a* so that:

    *   Low frequencies are attenuated (not eliminated) → preserved contrast.
    *   High frequencies are further boosted by factor *b* → enhanced edges.

    When *a = 0*, *b = 1* → pure HPF.
    When *a = 1*, *b = 1* → all-pass + HPF boost = unsharp masking.

    **Combined with histogram equalisation:**
    Applying histogram equalisation *after* HFE simultaneously corrects
    uneven illumination and enhances local contrast – a powerful pipeline
    for medical or aerial imagery.

    Args:
        P           (int):   Padded image height.
        Q           (int):   Padded image width.
        D0          (float): Cutoff frequency for the HPF component.
        a           (float): DC offset (additive constant). Default 0.5.
        b           (float): High-frequency boost factor. Default 0.75.
        filter_type (str):   ``'butterworth'`` (default) or ``'gaussian'``.
        n           (int):   Butterworth order (used only when
                             filter_type='butterworth'). Default 2.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q).
    """
    if filter_type == 'butterworth':
        H_hp = butterworth_hpf(P, Q, D0, n)
    elif filter_type == 'gaussian':
        H_hp = gaussian_hpf(P, Q, D0)
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'. "
                         "Choose 'butterworth' or 'gaussian'.")
    H_hfe = a + b * H_hp
    return H_hfe.astype(np.float64)


def homomorphic_filter(
    image: np.ndarray,
    D0: float,
    gamma_L: float = 0.25,
    gamma_H: float = 2.0,
    c: float = 1.0,
) -> np.ndarray:
    """
    Homomorphic Filter for simultaneous illumination correction and
    contrast enhancement.

    **Illumination-Reflectance Model (Section 4.9.3):**

        f(x,y) = i(x,y) · r(x,y)

    where i is the illumination (slow, low-frequency) and r is the
    reflectance (fine detail, high-frequency).  Taking the natural log
    *separates* them:

        z(x,y) = ln(f(x,y)) = ln(i(x,y)) + ln(r(x,y))

    A frequency-domain filter H applied to Z(u,v) = DFT{z} now acts
    *independently* on the two components.

    **Filter function (Modified Gaussian HPF, Eq. 4.9-11):**

        H(u,v) = (γ_H − γ_L) · (1 − exp(−c · D²(u,v) / D0²)) + γ_L

    *   γ_L < 1 → attenuates the illumination band → compresses dynamic
      range.
    *   γ_H > 1 → boosts the reflectance band → enhances local contrast.

    **Full pipeline:**
    1. z = ln(f + 1)   [+1 avoids log(0)]
    2. DFT of centred padded z  → Z(u,v)
    3. G(u,v) = H(u,v) · Z(u,v)
    4. g_p = real(IDFT{G}) · (−1)^(x+y)  [de-centre]
    5. Output = exp(g_p) − 1
    6. Crop to original size and normalise to [0, 255]

    Args:
        image   (np.ndarray): Input grayscale image (M, N), any numeric dtype.
        D0      (float):      Cutoff frequency.
        gamma_L (float):      Low-frequency gain (< 1). Default 0.25.
        gamma_H (float):      High-frequency gain (> 1). Default 2.0.
        c       (float):      Filter sharpness constant. Default 1.0.

    Returns:
        np.ndarray: Filtered image, dtype uint8, shape (M, N),
                    normalised to [0, 255].
    """
    image = np.asarray(image, dtype=np.float64)
    M, N = image.shape

    # Step 1 – log transform (+1 avoids log(0) for zero-valued pixels)
    z = np.log1p(image)

    # Step 2 – pad, centre, and DFT
    result = compute_dft(z)
    Z = result['F']             # complex, shape (P, Q)
    P, Q = result['padded_size']

    # Step 3 – construct homomorphic filter
    D = compute_distance_matrix(P, Q)
    H = (gamma_H - gamma_L) * (1.0 - np.exp(-c * (D ** 2) / (D0 ** 2))) + gamma_L

    # Step 4 – filter and inverse DFT (re-use apply_filter internals)
    G = H * Z
    g_padded = np.fft.ifft2(G).real

    # De-centre
    x = np.arange(P, dtype=np.float64).reshape(-1, 1)
    y = np.arange(Q, dtype=np.float64).reshape(1, -1)
    centering = (-1.0) ** (x + y)
    g_padded *= centering

    # Step 5 – inverse log transform
    g_padded = np.expm1(g_padded)   # exp(x) - 1

    # Step 6 – crop and normalise
    g = g_padded[:M, :N]
    g_min, g_max = g.min(), g.max()
    if g_max > g_min:
        g = (g - g_min) / (g_max - g_min) * 255.0
    else:
        g = np.zeros_like(g)
    return np.clip(g, 0, 255).astype(np.uint8)


# ===========================================================================
# SECTION 5: SELECTIVE / BAND FILTERS  (Section 4.8)
# ===========================================================================


def bandreject_gaussian(P: int, Q: int, D0: float, W: float) -> np.ndarray:
    """
    Gaussian Bandreject Filter (BRF).

    **Definition (Eq. 4.8-7):**

        H_BR(u,v) = 1 − exp( −[(D²(u,v) − D0²) / (D(u,v)·W)]² )

    *   D0 is the *centre frequency* of the rejected band.
    *   W  is the *bandwidth* (radial extent of the band).

    **Application (Section 4.8):**
    Bandreject filters remove a specific ring (or annulus) of frequencies.
    Typical use cases:
    *   Power-line interference (50 / 60 Hz banding)
    *   Scanner drum patterns (periodic horizontal / vertical lines)
    *   Moiré patterns from screen printing or textile scanning

    **Practical note:**
    At D = 0 the formula has 0/0 (D in denominator).  Handled by setting
    H = 1 at D = 0 (zero-frequency / DC is passed through).

    Args:
        P  (int):   Padded image height.
        Q  (int):   Padded image width.
        D0 (float): Centre frequency of the rejected band.
        W  (float): Bandwidth.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in [0, 1].
    """
    D = compute_distance_matrix(P, Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        # At D=0: numerator (0 - D0^2) is non-zero, denominator is 0 → inf
        ratio = np.where(
            D > 0,
            (D ** 2 - D0 ** 2) / (D * W),
            np.inf
        )
    H = 1.0 - np.exp(-(ratio ** 2))
    return H.astype(np.float64)


def bandpass_gaussian(P: int, Q: int, D0: float, W: float) -> np.ndarray:
    """
    Gaussian Bandpass Filter (BPF).

    **Relationship to BRF (Eq. 4.8-9):**

        H_BP(u,v) = 1 − H_BR(u,v)

    **Application (Section 4.8):**
    Passes only the frequencies in the annulus centred at D0 with
    half-width W/2.  Used to:
    *   Isolate a specific periodic texture component.
    *   Analyse the energy distribution within a frequency band.
    *   Extract the periodic noise pattern itself (see also notch pass).

    Args:
        P  (int):   Padded image height.
        Q  (int):   Padded image width.
        D0 (float): Centre frequency of the passed band.
        W  (float): Bandwidth.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in [0, 1].
    """
    return 1.0 - bandreject_gaussian(P, Q, D0, W)


def notch_reject_filter(
    P: int,
    Q: int,
    notch_points: list,
    D0: float = 10.0,
    filter_type: str = 'gaussian',
) -> np.ndarray:
    """
    Notch Reject Filter (NRF).

    **Theory (Section 4.8.2):**
    Periodic noise in an image manifests as bright isolated spots in the
    DFT magnitude spectrum at positions (u_k, v_k) and (−u_k, −v_k)
    (they always occur in conjugate pairs because the image is real-valued).

    A notch reject filter zeros out (or attenuates) these specific
    frequency pairs while leaving all other frequencies intact.

    **Distance definition (Eq. 4.8-14):**

        D_k(u,v)     = sqrt( (u − P/2 − u_k)² + (v − Q/2 − v_k)² )
        D_{−k}(u,v)  = sqrt( (u − P/2 + u_k)² + (v − Q/2 + v_k)² )

    **Filter (product form, Eq. 4.8-15):**

        H_NR(u,v) = ∏_k  H_k(u,v) · H_{−k}(u,v)

    where each H_k is a *highpass* Gaussian (or Butterworth) evaluated at
    D_k (a small circular HPF centred on the spike location).

    **Gaussian notch (Eq. 4.8-16):**

        H_k = 1 − exp( −D_k² / (2·D0²) )

    **Usage workflow:**
    1. Display log-magnitude DFT spectrum.
    2. Identify bright interference spots → record their (u, v) offsets
       from the centre (P//2, Q//2).
    3. Pass those offsets as ``notch_points``.
    4. Apply filter → periodic noise greatly reduced.

    Args:
        P             (int):   Padded image height.
        Q             (int):   Padded image width.
        notch_points  (list):  List of (u_k, v_k) tuples, coordinates
                               *relative to the DFT centre*.
        D0            (float): Notch radius (std-dev for Gaussian).
        filter_type   (str):   ``'gaussian'`` (default) or ``'butterworth'``.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in [0, 1].
    """
    # Grid of (u, v) coordinates
    u_idx = np.arange(P, dtype=np.float64).reshape(-1, 1)
    v_idx = np.arange(Q, dtype=np.float64).reshape(1, -1)

    # Start with all-pass
    H = np.ones((P, Q), dtype=np.float64)

    for (uk, vk) in notch_points:
        # Distance from positive notch location
        D_k = np.sqrt((u_idx - P // 2 - uk) ** 2 + (v_idx - Q // 2 - vk) ** 2)
        # Distance from conjugate notch location
        D_neg_k = np.sqrt((u_idx - P // 2 + uk) ** 2 + (v_idx - Q // 2 + vk) ** 2)

        if filter_type == 'gaussian':
            Hk = 1.0 - np.exp(-(D_k ** 2) / (2.0 * D0 ** 2))
            H_neg_k = 1.0 - np.exp(-(D_neg_k ** 2) / (2.0 * D0 ** 2))
        elif filter_type == 'butterworth':
            # Order-2 Butterworth highpass evaluated at D_k
            with np.errstate(divide='ignore', invalid='ignore'):
                r_k = np.where(D_k > 0, D0 / D_k, np.inf)
                r_neg = np.where(D_neg_k > 0, D0 / D_neg_k, np.inf)
            Hk = 1.0 / (1.0 + r_k ** 4)
            H_neg_k = 1.0 / (1.0 + r_neg ** 4)
        else:
            raise ValueError(f"Unknown filter_type '{filter_type}'. "
                             "Choose 'gaussian' or 'butterworth'.")

        H *= Hk * H_neg_k

    return H.astype(np.float64)


def notch_pass_filter(
    P: int,
    Q: int,
    notch_points: list,
    D0: float = 10.0,
) -> np.ndarray:
    """
    Notch Pass Filter (NPF).

    **Relationship to NRF (Eq. 4.8-21):**

        H_NP(u,v) = 1 − H_NR(u,v)

    **Application:**
    Passes *only* the notch frequencies (the bright spectral spikes)
    and blocks everything else.  The resulting spatial-domain image
    is the extracted **periodic noise pattern** itself.

    This extracted noise n̂(x,y) can then be subtracted from the original
    to refine noise removal:

        f_restored(x,y) = f(x,y) − n̂(x,y)

    This two-step approach (Gonzalez & Woods Section 4.8.2) often gives
    better results than NRF alone because it allows the user to inspect and
    scale the noise estimate before subtraction.

    Args:
        P             (int):   Padded image height.
        Q             (int):   Padded image width.
        notch_points  (list):  List of (u_k, v_k) tuples (centred coords).
        D0            (float): Notch radius for the underlying NRF.

    Returns:
        np.ndarray: float64 filter array of shape (P, Q), values in [0, 1].
    """
    H_NR = notch_reject_filter(P, Q, notch_points, D0=D0, filter_type='gaussian')
    return (1.0 - H_NR).astype(np.float64)


# ===========================================================================
# SECTION 6: IMAGE QUALITY METRICS  (Appendix / Section 4.10)
# ===========================================================================


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).

    **Definition:**

        MSE = (1 / (MN)) · Σ_{x,y} [f(x,y) − f̂(x,y)]²

    MSE is the most basic full-reference quality metric.  It measures the
    average squared pixel-level deviation between the reference image f and
    the processed / reconstructed image f̂.

    **Limitations:**
    *   MSE treats all pixel differences equally regardless of spatial
      structure → poor correlation with human perception.
    *   An image with low MSE can look visually poor (e.g. blurring and
      ringing may both yield similar MSE).
    *   Use alongside PSNR and SSIM for a fuller quality assessment.

    Args:
        original  (np.ndarray): Reference image, shape (M, N).
        processed (np.ndarray): Distorted / processed image, shape (M, N).

    Returns:
        float: MSE value (lower is better; 0 = identical images).
    """
    f = np.asarray(original, dtype=np.float64)
    f_hat = np.asarray(processed, dtype=np.float64)
    if f.shape != f_hat.shape:
        raise ValueError("original and processed must have the same shape.")
    return float(np.mean((f - f_hat) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in decibels.

    **Definition:**

        PSNR = 10 · log10( MAX² / MSE )

    where MAX = 255 for uint8 imagery.

    **Interpretation:**
    *   PSNR → ∞ when MSE → 0 (identical images).
    *   Typical "good quality" values: > 30 dB for most applications;
      > 40 dB for lossless / near-lossless compression.
    *   PSNR is widely used because of simplicity, but like MSE it does
      not always correlate well with subjective quality.

    Args:
        original  (np.ndarray): Reference image, shape (M, N).
        processed (np.ndarray): Distorted image, shape (M, N).

    Returns:
        float: PSNR in dB.  Returns ``inf`` if MSE = 0.
    """
    mse = compute_mse(original, processed)
    if mse == 0.0:
        return float('inf')
    max_val = 255.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).

    **Definition (Wang et al., 2004; referenced in Gonzalez & Woods):**

        SSIM(x, y) = [L(x,y)]^α · [C(x,y)]^β · [S(x,y)]^γ

    With α = β = γ = 1 and the combined form:

        SSIM = (2·μ_x·μ_y + C1)(2·σ_xy + C2)
               ─────────────────────────────────────
               (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    **Constants** to avoid division by zero (dynamic-range based):

        C1 = (0.01 · 255)²  =  6.5025
        C2 = (0.03 · 255)²  = 58.5225

    **Computation strategy:**
    1. Apply an 11×11 Gaussian window (σ = 1.5) to compute local means,
       variances, and cross-covariance across the image.
    2. Compute SSIM map.
    3. Return the mean SSIM over all windows (the MSSIM index).

    **Why SSIM is better than MSE/PSNR:**
    *   MSE is not correlated with human visual quality judgements on
      structured distortions (blur, ringing, blocking).
    *   SSIM captures *luminance*, *contrast*, and *structural* similarity
      separately, matching the human visual system more closely.
    *   SSIM ∈ [−1, 1]; values close to 1 indicate high similarity.

    Implementation uses ``scipy.ndimage.uniform_filter`` as a tractable
    approximation to Gaussian windowing.  For the full 11×11 Gaussian-
    weighted implementation see ``skimage.metrics.structural_similarity``.

    Args:
        original  (np.ndarray): Reference image, shape (M, N).
        processed (np.ndarray): Distorted image, shape (M, N).

    Returns:
        float: Mean SSIM index ∈ [−1, 1].  1.0 = identical images.
    """
    f = np.asarray(original, dtype=np.float64)
    g = np.asarray(processed, dtype=np.float64)
    if f.shape != g.shape:
        raise ValueError("original and processed must have the same shape.")

    # Stability constants (Eq. in Wang et al. 2004)
    C1 = (0.01 * 255.0) ** 2   # 6.5025
    C2 = (0.03 * 255.0) ** 2   # 58.5225

    window_size = 11  # 11×11 sliding window (standard)

    # Local means via uniform filter (approximation of Gaussian weighting)
    mu_x  = uniform_filter(f, size=window_size, mode='reflect')
    mu_y  = uniform_filter(g, size=window_size, mode='reflect')

    mu_x2  = mu_x ** 2
    mu_y2  = mu_y ** 2
    mu_xy  = mu_x * mu_y

    # Local second moments
    sigma_x2  = uniform_filter(f * f, size=window_size, mode='reflect') - mu_x2
    sigma_y2  = uniform_filter(g * g, size=window_size, mode='reflect') - mu_y2
    sigma_xy  = uniform_filter(f * g, size=window_size, mode='reflect') - mu_xy

    # Clip negative variances (can occur near boundaries due to numerics)
    sigma_x2 = np.maximum(sigma_x2, 0.0)
    sigma_y2 = np.maximum(sigma_y2, 0.0)

    # SSIM map
    numerator   = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    # Core pipeline
    'pad_image',
    'compute_dft',
    'compute_distance_matrix',
    'apply_filter',
    # Lowpass filters
    'ideal_lpf',
    'butterworth_lpf',
    'gaussian_lpf',
    # Highpass filters
    'ideal_hpf',
    'butterworth_hpf',
    'gaussian_hpf',
    # Advanced filters
    'laplacian_frequency',
    'high_freq_emphasis',
    'homomorphic_filter',
    # Selective / band filters
    'bandreject_gaussian',
    'bandpass_gaussian',
    'notch_reject_filter',
    'notch_pass_filter',
    # Quality metrics
    'compute_mse',
    'compute_psnr',
    'compute_ssim',
]
