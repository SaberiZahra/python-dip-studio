"""
Chapter 9: Morphological Image Processing
Based on: Gonzalez & Woods, "Digital Image Processing", 3rd Edition

Binary and grayscale morphological operations using structuring elements.
All binary operations work on boolean/uint8 images.
Grayscale operations work on float64 images.
"""
import numpy as np
from scipy.ndimage import (
    binary_erosion, binary_dilation,
    grey_erosion, grey_dilation,
    label as nd_label,
)

__all__ = [
    'get_structuring_element',
    'binary_erosion_custom',
    'binary_dilation_custom',
    'opening',
    'closing',
    'boundary_extraction',
    'morphological_gradient',
    'hit_or_miss_transform',
    'region_filling',
    'thinning',
    'gray_erosion_custom',
    'gray_dilation_custom',
    'gray_opening',
    'gray_closing',
    'tophat_transform',
    'bottomhat_transform',
    'morphological_reconstruction',
    'otsu_threshold',
    'local_thresholding',
]


# ---------------------------------------------------------------------------
# §9.2  Structuring Elements
# ---------------------------------------------------------------------------

def get_structuring_element(shape: str = 'disk', size: int = 3) -> np.ndarray:
    """
    Create a flat structuring element (SE) of the requested shape.

    The structuring element (SE) defines the neighborhood shape used in every
    morphological operation.  It is a small binary mask centered at its origin
    (the center pixel).

    Parameters
    ----------
    shape : str
        One of ``'disk'``, ``'square'``, ``'cross'``, ``'diamond'``.

        * **disk** – Isotropic, circular SE.  The most widely used shape
          because it treats all directions equally and avoids the corner
          artifacts introduced by square SEs.
          ``B[i,j] = 1  if  sqrt((i-r)^2 + (j-r)^2) <= r``,  r = (size-1)/2.
        * **square** – All entries are 1.  Fast to compute (separable), but
          causes corner artifacts along diagonal boundaries.
        * **cross** – Only the centre row and centre column are 1, giving
          4-connectivity (N, S, E, W) rather than 8-connectivity.
        * **diamond** – All pixels whose Manhattan distance from the centre
          is <= r are 1.  Intermediate between cross and disk.

    size : int
        Side length of the bounding box (must be odd ≥ 1).  For a disk,
        square, and diamond the radius is ``r = (size - 1) / 2``.

    Returns
    -------
    B : numpy.ndarray of dtype bool, shape (size, size)
        The structuring element mask.

    Raises
    ------
    ValueError
        If *shape* is not recognised or *size* is even / < 1.

    Examples
    --------
    >>> B = get_structuring_element('disk', 5)
    >>> B.shape
    (5, 5)
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size}.")
    r = (size - 1) / 2
    # Build coordinate grids centred at the origin
    y, x = np.ogrid[-r:r + 1, -r:r + 1]

    shape = shape.lower()
    if shape == 'disk':
        B = (x ** 2 + y ** 2) <= r ** 2
    elif shape == 'square':
        B = np.ones((size, size), dtype=bool)
    elif shape == 'cross':
        B = np.zeros((size, size), dtype=bool)
        cx = size // 2
        B[cx, :] = True
        B[:, cx] = True
    elif shape == 'diamond':
        B = (np.abs(x) + np.abs(y)) <= r
    else:
        raise ValueError(
            f"Unknown shape '{shape}'. Choose from 'disk', 'square', "
            f"'cross', 'diamond'."
        )
    return B.astype(bool)


# ---------------------------------------------------------------------------
# §9.3  Binary Morphological Operations
# ---------------------------------------------------------------------------

def binary_erosion_custom(image: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Binary erosion: ``A ⊖ B = { z | (B)_z ⊆ A }``.

    A pixel *z* survives erosion only if the structuring element *B*, when
    placed with its origin at *z*, fits **entirely** inside the foreground set
    *A*.

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * **Shrinks** foreground objects — removes pixels on the object boundary.
    * Removes bright noise smaller than the SE.
    * Separates weakly connected objects (narrow bridges are severed).
    * The eroded set is always a subset of the original.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input image (bool or uint8, non-zero = foreground).
    B : numpy.ndarray
        Structuring element (bool array).  Its shape defines the neighbourhood.

    Returns
    -------
    numpy.ndarray of dtype bool
        Eroded binary image.

    Notes
    -----
    Delegates to :func:`scipy.ndimage.binary_erosion` with ``border_value=0``
    (image is padded with background outside its boundary).

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.2.1, Eq. (9-1).
    """
    img_bool = image.astype(bool)
    return binary_erosion(img_bool, structure=B, border_value=False)


def binary_dilation_custom(image: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Binary dilation: ``A ⊕ B = { z | (B̂)_z ∩ A ≠ ∅ }``.

    A pixel *z* is included in the dilation if the **reflection** of *B*,
    centred at *z*, has at least one element that overlaps the foreground *A*.

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * **Grows** foreground objects — adds pixels at the object boundary.
    * Fills small holes and thin gaps inside objects.
    * Bridges narrow gaps between nearby objects.
    * For a symmetric SE (most common), ``B̂ = B``.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input image (bool or uint8).
    B : numpy.ndarray
        Structuring element (bool array).

    Returns
    -------
    numpy.ndarray of dtype bool
        Dilated binary image.

    Notes
    -----
    Delegates to :func:`scipy.ndimage.binary_dilation`.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.2.2, Eq. (9-5).
    """
    img_bool = image.astype(bool)
    return binary_dilation(img_bool, structure=B)


def opening(image: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Morphological opening: ``A ∘ B = (A ⊖ B) ⊕ B``.

    Opening = erosion followed by dilation **with the same SE**.

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Removes **small bright objects** (noise, spurs, protrusions) that are
      smaller than the SE.
    * Does not significantly change the shape or size of larger objects.
    * Idempotent: ``(A ∘ B) ∘ B = A ∘ B``.
    * ``A ∘ B ⊆ A`` — opening never adds new pixels.
    * Primary use: **noise removal** while preserving object shape.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input (bool or uint8).
    B : numpy.ndarray
        Structuring element.

    Returns
    -------
    numpy.ndarray of dtype bool

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.3.1, Eq. (9-14).
    """
    eroded = binary_erosion_custom(image, B)
    return binary_dilation_custom(eroded, B)


def closing(image: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Morphological closing: ``A • B = (A ⊕ B) ⊖ B``.

    Closing = dilation followed by erosion **with the same SE**.

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Fills **small dark holes** and gaps within foreground objects.
    * Bridges narrow gaps between nearby objects.
    * Does not significantly change the overall shape of larger objects.
    * Idempotent: ``(A • B) • B = A • B``.
    * ``A ⊆ A • B`` — closing never removes existing pixels.
    * Primary use: **closing small dark gaps** in bright objects.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input (bool or uint8).
    B : numpy.ndarray
        Structuring element.

    Returns
    -------
    numpy.ndarray of dtype bool

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.3.2, Eq. (9-16).
    """
    dilated = binary_dilation_custom(image, B)
    return binary_erosion_custom(dilated, B)


def boundary_extraction(image: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Morphological boundary extraction: ``β(A) = A − (A ⊖ B)``.

    Subtracts the eroded image from the original to yield a **1-pixel-wide
    outline** of each foreground object.

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Produces a thin border exactly one SE radius wide.
    * Useful for perimeter measurement, contour display, and object counting.
    * Unlike gradient operators, the result is guaranteed binary.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input (bool or uint8).
    B : numpy.ndarray
        Structuring element (typically a small 3×3 disk or cross).

    Returns
    -------
    numpy.ndarray of dtype bool
        Binary boundary mask.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.4.1, Eq. (9-19).
    """
    img_bool = image.astype(bool)
    eroded = binary_erosion_custom(img_bool, B)
    return img_bool & ~eroded


def morphological_gradient(image: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Morphological gradient: ``gradient = (A ⊕ B) − (A ⊖ B)``.

    The difference between the dilation and erosion of the input highlights
    transitions between foreground and background — i.e., edges.

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Thicker than ``boundary_extraction`` (which uses only erosion).
    * Works on both binary and grayscale images.
    * Symmetric: measures the spread around the boundary in both directions.

    Parameters
    ----------
    image : numpy.ndarray
        Binary (bool/uint8) or grayscale (float64) input image.
    B : numpy.ndarray
        Structuring element.

    Returns
    -------
    numpy.ndarray
        Gradient image (same dtype promotion as input differences).

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.3 (grayscale gradient discussion).
    """
    img = image.astype(float)
    # For binary images the result is 0 or 1; for grayscale it is continuous
    dil = binary_dilation(image.astype(bool), structure=B).astype(float)
    ero = binary_erosion(image.astype(bool), structure=B, border_value=False).astype(float)
    return dil - ero


def hit_or_miss_transform(
    image: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
) -> np.ndarray:
    """
    Hit-or-miss transform (HMT): ``HMT(A) = (A ⊖ B1) ∩ (Aᶜ ⊖ B2)``.

    Detects **specific local configurations** (shapes) in a binary image.

    * **B1** probes the foreground pattern (pixels that *must* be ON).
    * **B2** probes the background pattern (pixels that *must* be OFF), applied
      to the complement ``Aᶜ``.
    * B1 and B2 must be **disjoint** (no pixel can be simultaneously required
      foreground and background).

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Only pixels where the local neighbourhood matches *both* B1 and B2
      simultaneously survive.
    * Used for: specific shape detection, corner finding, isolated pixel
      detection, skeletonisation sub-operations.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input (bool or uint8).
    B1 : numpy.ndarray
        Foreground structuring element.
    B2 : numpy.ndarray
        Background structuring element (applied to complement of image).

    Returns
    -------
    numpy.ndarray of dtype bool
        Binary image marking matched locations.

    Raises
    ------
    ValueError
        If B1 and B2 have different shapes.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.4.4, Eq. (9-22).
    """
    if B1.shape != B2.shape:
        raise ValueError(
            f"B1 and B2 must have the same shape; got {B1.shape} and {B2.shape}."
        )
    img_bool = image.astype(bool)
    comp = ~img_bool  # complement Aᶜ
    erode_fg = binary_erosion(img_bool, structure=B1, border_value=False)
    erode_bg = binary_erosion(comp, structure=B2, border_value=False)
    return erode_fg & erode_bg


def region_filling(
    image: np.ndarray,
    seed_point: tuple,
    B: np.ndarray | None = None,
) -> np.ndarray:
    """
    Region filling via iterative geodesic dilation.

    Algorithm (Gonzalez & Woods §9.4.2)::

        X₀  = {seed_point}
        X_k = (X_{k-1} ⊕ B) ∩ Aᶜ    for k = 1, 2, 3, …
        stop when X_k == X_{k-1}
        Result = X_k ∪ A

    Starting from a *seed point* **inside** a bounded region, the foreground
    iteratively expands (dilation) but is constrained to stay outside the
    boundary set ``A``.  The process terminates when no new pixels are added.

    Parameters
    ----------
    image : numpy.ndarray
        Binary boundary image (bool or uint8).  The boundary pixels are ON;
        the region interior is OFF.
    seed_point : tuple of int
        ``(row, col)`` coordinate inside the region to fill.  Must lie in the
        OFF area of *image*.
    B : numpy.ndarray or None
        Structuring element.  Defaults to a 3×3 cross (4-connectivity).

    Returns
    -------
    numpy.ndarray of dtype bool
        ``image`` with the enclosed region (and original boundary) filled.

    Notes
    -----
    For 8-connectivity use a 3×3 square SE; for 4-connectivity use a cross.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.4.2, Eq. (9-20).
    """
    if B is None:
        B = get_structuring_element('cross', 3)

    img_bool = image.astype(bool)
    comp = ~img_bool  # Aᶜ — the fillable interior + exterior

    # Initialise seed
    X = np.zeros_like(img_bool)
    X[seed_point[0], seed_point[1]] = True

    for _ in range(max(image.shape) * 2):
        X_prev = X.copy()
        X = binary_dilation(X, structure=B) & comp
        if np.array_equal(X, X_prev):
            break

    return X | img_bool


# ---------------------------------------------------------------------------
# §9.5.7  Zhang-Suen Thinning
# ---------------------------------------------------------------------------

def thinning(image: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Zhang-Suen thinning algorithm for binary images.

    Iteratively removes border pixels from the foreground while preserving
    the **topological connectivity** of the skeleton, until no more pixels can
    be removed.

    Algorithm (two-sub-iteration per pass)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each foreground pixel P1 with 8-neighbours P2…P9 (clockwise):

    **Sub-iteration 1** — delete P1 if ALL of:
      1. ``2 ≤ N(P1) ≤ 6``  (N = number of non-zero neighbours)
      2. ``T(P1) = 1``  (T = transitions 0→1 in the cyclic order P2,P3,…P9,P2)
      3. ``P2 · P4 · P6 = 0``
      4. ``P4 · P6 · P8 = 0``

    **Sub-iteration 2** — delete P1 if ALL of:
      1. ``2 ≤ N(P1) ≤ 6``
      2. ``T(P1) = 1``
      3. ``P2 · P4 · P8 = 0``
      4. ``P2 · P6 · P8 = 0``

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Reduces binary objects to **1-pixel-wide skeletons**.
    * Preserves the number of connected components and holes.
    * Used in: character recognition, fingerprint analysis, medical imaging.

    Parameters
    ----------
    image : numpy.ndarray
        Binary input (bool or uint8), non-zero = foreground.
    max_iter : int
        Maximum number of full passes (each pass = 2 sub-iterations).
        Useful to limit runtime on large images.

    Returns
    -------
    numpy.ndarray of dtype bool
        Thinned (skeletonised) binary image.

    References
    ----------
    Zhang, T. Y. & Suen, C. Y. (1984). "A fast parallel algorithm for
    thinning digital patterns." *Commun. ACM*, 27(3), 236–239.
    Gonzalez & Woods, 3rd ed., §9.5.7.
    """
    img = image.astype(np.uint8)
    img[img > 0] = 1

    def _neighbours(r, c, arr):
        # P2…P9 clockwise starting from the top
        return [
            arr[r - 1, c],      # P2
            arr[r - 1, c + 1],  # P3
            arr[r,     c + 1],  # P4
            arr[r + 1, c + 1],  # P5
            arr[r + 1, c],      # P6
            arr[r + 1, c - 1],  # P7
            arr[r,     c - 1],  # P8
            arr[r - 1, c - 1],  # P9
        ]

    def _transitions(nbrs):
        """Count 0→1 transitions in cyclic order."""
        n = nbrs + [nbrs[0]]
        return sum(1 for i in range(8) if n[i] == 0 and n[i + 1] == 1)

    for _ in range(max_iter):
        changed = False
        for step in (1, 2):
            to_delete = []
            for r in range(1, img.shape[0] - 1):
                for c in range(1, img.shape[1] - 1):
                    if img[r, c] == 0:
                        continue
                    nbrs = _neighbours(r, c, img)
                    n_count = sum(nbrs)
                    if not (2 <= n_count <= 6):
                        continue
                    if _transitions(nbrs) != 1:
                        continue
                    p2, p3, p4, p5, p6, p7, p8, p9 = nbrs
                    if step == 1:
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
                            continue
                    to_delete.append((r, c))
            if to_delete:
                changed = True
                for r, c in to_delete:
                    img[r, c] = 0
        if not changed:
            break

    return img.astype(bool)


# ---------------------------------------------------------------------------
# §9.6  Grayscale Morphology
# ---------------------------------------------------------------------------

def gray_erosion_custom(
    image: np.ndarray,
    B_flat: bool = True,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Grayscale erosion with a flat structuring element.

    For a **flat** SE (all structuring-element values ``b(s,t) = 0``), the
    operation reduces to a **sliding minimum**::

        [f ⊖ b](x, y) = min_{(s,t)∈B}  f(x+s, y+t)

    For a non-flat SE the general form is::

        [f ⊖ b](x, y) = min_{(s,t)∈B}  [ f(x+s, y+t) − b(s,t) ]

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * **Darkens** the image — reduces bright features.
    * Removes bright peaks (local maxima) smaller than the SE.
    * Expands dark regions.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (float64 recommended; uint8 also accepted).
    B_flat : bool
        If ``True`` (default) use a flat disk SE of radius ``kernel_size//2``.
        The ``size`` parameter of :func:`scipy.ndimage.grey_erosion` is used.
    kernel_size : int
        Side length of the SE bounding box (odd integer).

    Returns
    -------
    numpy.ndarray of dtype float64
        Eroded grayscale image.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.1, Eq. (9-45).
    """
    img = image.astype(np.float64)
    if B_flat:
        footprint = get_structuring_element('disk', kernel_size)
        return grey_erosion(img, footprint=footprint)
    else:
        return grey_erosion(img, size=kernel_size)


def gray_dilation_custom(
    image: np.ndarray,
    B_flat: bool = True,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Grayscale dilation with a flat structuring element.

    For a **flat** SE (``b(s,t) = 0``), the operation is a **sliding maximum**::

        [f ⊕ b](x, y) = max_{(s,t)∈B}  f(x−s, y−t)

    For a non-flat SE::

        [f ⊕ b](x, y) = max_{(s,t)∈B}  [ f(x−s, y−t) + b(s,t) ]

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * **Brightens** the image — expands bright features.
    * Fills dark valleys (local minima) smaller than the SE.
    * Expands bright regions.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (float64 recommended; uint8 also accepted).
    B_flat : bool
        If ``True`` (default) use a flat disk SE of radius ``kernel_size//2``.
    kernel_size : int
        Side length of the SE bounding box (odd integer).

    Returns
    -------
    numpy.ndarray of dtype float64
        Dilated grayscale image.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.2, Eq. (9-47).
    """
    img = image.astype(np.float64)
    if B_flat:
        footprint = get_structuring_element('disk', kernel_size)
        return grey_dilation(img, footprint=footprint)
    else:
        return grey_dilation(img, size=kernel_size)


def gray_opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Grayscale opening: erosion followed by dilation.

    ``f ∘ b = (f ⊖ b) ⊕ b``

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Removes **bright peaks** and thin protrusions narrower than the SE.
    * Suppresses bright noise while preserving the overall intensity landscape.
    * Useful for **background estimation**: if the background is smooth and
      objects are brighter than it, opening approximates the background.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (float64 or uint8).
    kernel_size : int
        SE size (odd integer).

    Returns
    -------
    numpy.ndarray of dtype float64

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.3, Eq. (9-52).
    """
    eroded = gray_erosion_custom(image, B_flat=True, kernel_size=kernel_size)
    return gray_dilation_custom(eroded, B_flat=True, kernel_size=kernel_size)


def gray_closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Grayscale closing: dilation followed by erosion.

    ``f • b = (f ⊕ b) ⊖ b``

    Morphological effects
    ~~~~~~~~~~~~~~~~~~~~~
    * Fills **dark holes** and valleys narrower than the SE.
    * Suppresses dark noise while preserving bright structure.
    * Useful for **background estimation** when objects are darker than the
      background (complementary to :func:`gray_opening`).

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (float64 or uint8).
    kernel_size : int
        SE size (odd integer).

    Returns
    -------
    numpy.ndarray of dtype float64

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.3, Eq. (9-54).
    """
    dilated = gray_dilation_custom(image, B_flat=True, kernel_size=kernel_size)
    return gray_erosion_custom(dilated, B_flat=True, kernel_size=kernel_size)


def tophat_transform(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    White top-hat transform: ``T_hat(f) = f − (f ∘ b)``.

    Subtracts the grayscale opening from the original image, isolating
    bright features that are **smaller than the SE**.

    Morphological effects / use-cases
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Extracts small **bright objects** on a slowly varying background.
    * When illumination varies slowly (features wider than the SE are removed
      by opening), the difference isolates only the small bright structures.
    * Used for **background normalisation**: bright = object, dark = background.
    * Typical SE size: larger than the objects of interest, but smaller than
      background variations.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (float64 or uint8).
    kernel_size : int
        SE size (odd integer). Should be larger than the objects to suppress.

    Returns
    -------
    numpy.ndarray of dtype float64
        Top-hat image (non-negative values).

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.4, Eq. (9-55).
    """
    img = image.astype(np.float64)
    opened = gray_opening(img, kernel_size=kernel_size)
    return img - opened


def bottomhat_transform(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Black bottom-hat transform: ``B_hat(f) = (f • b) − f``.

    Subtracts the original from its grayscale closing, isolating
    **dark features smaller than the SE**.

    Morphological effects / use-cases
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Extracts small **dark objects** on a slowly varying bright background.
    * Equivalent to applying the white top-hat to the photometric complement.
    * Used to find dark spots, scratches, or voids on bright surfaces.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (float64 or uint8).
    kernel_size : int
        SE size (odd integer).

    Returns
    -------
    numpy.ndarray of dtype float64
        Bottom-hat image (non-negative values).

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.6.4, Eq. (9-56).
    """
    img = image.astype(np.float64)
    closed = gray_closing(img, kernel_size=kernel_size)
    return closed - img


def morphological_reconstruction(
    marker: np.ndarray,
    mask: np.ndarray,
    method: str = 'dilation',
) -> np.ndarray:
    """
    Morphological reconstruction by dilation or erosion.

    **Reconstruction by dilation** (default)::

        R^D_G(F):  R₀ = F
                   R_k = (R_{k-1} ⊕ B) ∧ G   (element-wise minimum with mask G)
                   stop when R_k = R_{k-1}

    **Reconstruction by erosion**::

        R^E_G(F):  R₀ = F
                   R_k = (R_{k-1} ⊖ B) ∨ G   (element-wise maximum with mask G)
                   stop when R_k = R_{k-1}

    Here *F* is the **marker** and *G* is the **mask**.  For reconstruction by
    dilation the marker must satisfy ``F ≤ G`` (pixel-wise); for erosion
    ``F ≥ G``.

    Use-cases
    ~~~~~~~~~
    * **Peak detection**: set marker = image − δ, mask = image → geodesic
      dilation eliminates regional maxima smaller than δ; difference reveals peaks.
    * **Hole filling**: initialise marker from boundary of image complement.
    * **Reconstruction-based opening/closing**: remove objects not connected to
      the seed without altering the shape of retained objects.

    Parameters
    ----------
    marker : numpy.ndarray
        Starting image F.  Must be ≤ mask (for dilation) or ≥ mask (for erosion).
    mask : numpy.ndarray
        Constraining mask G.  Same shape as *marker*.
    method : str
        ``'dilation'`` (default) or ``'erosion'``.

    Returns
    -------
    numpy.ndarray of dtype float64
        Reconstructed image.

    Raises
    ------
    ValueError
        If *method* is not ``'dilation'`` or ``'erosion'``.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §9.5.3 (Geodesic dilation).
    """
    method = method.lower()
    if method not in ('dilation', 'erosion'):
        raise ValueError(f"method must be 'dilation' or 'erosion', got '{method}'.")

    B = get_structuring_element('disk', 3)
    R = marker.astype(np.float64)
    G = mask.astype(np.float64)

    for _ in range(max(marker.shape) * 10):
        R_prev = R.copy()
        if method == 'dilation':
            R = np.minimum(grey_dilation(R, footprint=B), G)
        else:
            R = np.maximum(grey_erosion(R, footprint=B), G)
        if np.allclose(R, R_prev):
            break

    return R


# ---------------------------------------------------------------------------
# §10.3  Otsu Thresholding
# ---------------------------------------------------------------------------

def otsu_threshold(image: np.ndarray) -> dict:
    """
    Otsu's optimal global thresholding.

    Finds the threshold ``k*`` that **maximises the between-class variance**
    ``σ_B²(k)`` (equivalently minimises the weighted sum of within-class
    variances).

    Mathematical derivation
    ~~~~~~~~~~~~~~~~~~~~~~~
    Given the normalised histogram ``p_i = n_i / N``  for  ``i ∈ [0, L-1]``:

    Cumulative quantities up to threshold *k*::

        ω₁(k) = Σ p_i            (i = 0 … k)   — class 1 probability
        μ(k)  = Σ i·p_i          (i = 0 … k)   — cumulative mean
        μ_T   = Σ i·p_i          (i = 0 … L-1) — global mean

    Between-class variance (compact form)::

        σ_B²(k) = [μ_T·ω₁(k) − μ(k)]²
                  ─────────────────────
                   ω₁(k)·[1 − ω₁(k)]

    Optimal threshold::

        k* = argmax_k  σ_B²(k)

    Separability measure::

        η = σ_B²(k*) / σ_T²       ∈ [0, 1]

    ``η = 1`` → perfectly separated bimodal histogram.
    ``η → 0`` → unimodal or heavily overlapping histogram.

    Assumptions
    ~~~~~~~~~~~
    * The histogram is **bimodal** (two classes: background and foreground).
    * For multi-modal images use iterative or multi-level Otsu.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale uint8 image (values 0–255) or float64 (scaled to 0–255).

    Returns
    -------
    dict with keys:
        * ``'threshold'``    : int   — optimal threshold k*
        * ``'separability'`` : float — η ∈ [0, 1]
        * ``'binary'``       : ndarray of bool — ``image > k*``

    References
    ----------
    Gonzalez & Woods, 3rd ed., §10.3.3, Eq. (10-50)–(10-57).
    Otsu, N. (1979). "A threshold selection method from gray-level histograms."
    *IEEE Trans. SMC*, 9(1), 62–66.
    """
    # Work with uint8 intensity levels 0–255
    img = np.clip(image, 0, 255).astype(np.uint8)
    L = 256

    # 1. Normalised histogram
    hist, _ = np.histogram(img.ravel(), bins=L, range=(0, L))
    p = hist / hist.sum()   # p_i  for i in 0…L-1

    # 2. Global mean and total variance
    levels = np.arange(L, dtype=np.float64)
    mu_T = np.sum(levels * p)                           # global mean
    sigma_T2 = np.sum((levels - mu_T) ** 2 * p)        # total variance

    # 3. Cumulative sums for vectorised computation
    omega = np.cumsum(p)        # ω₁(k)
    mu    = np.cumsum(levels * p)   # μ(k)  — cumulative mean

    # 4. Between-class variance for every k
    #    σ_B²(k) = (μ_T·ω₁(k) − μ(k))² / (ω₁(k)·(1−ω₁(k)))
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator   = (mu_T * omega - mu) ** 2
        denominator = omega * (1.0 - omega)
        sigma_B2    = np.where(denominator > 0, numerator / denominator, 0.0)

    # 5. Optimal threshold
    k_star = int(np.argmax(sigma_B2))
    sigma_B2_max = sigma_B2[k_star]

    # 6. Separability measure η
    eta = float(sigma_B2_max / sigma_T2) if sigma_T2 > 0 else 0.0

    return {
        'threshold':    k_star,
        'separability': eta,
        'binary':       img > k_star,
    }


def local_thresholding(
    image: np.ndarray,
    block_size: int = 51,
    offset: int = 10,
) -> np.ndarray:
    """
    Local (adaptive) thresholding with per-block Otsu thresholds.

    The image is divided into non-overlapping rectangular blocks.  Within each
    block, Otsu's method is applied independently.  The resulting binary blocks
    are stitched back together.  An *offset* is **subtracted** from each
    local threshold before applying it, biasing the decision slightly toward
    including more foreground pixels.

    Motivation
    ~~~~~~~~~~
    Global thresholding fails when **illumination is non-uniform** (e.g.,
    document scans, microscopy images with vignetting).  Local thresholding
    adapts to spatial intensity variations by computing a separate threshold
    for each image region.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input (uint8 or float64).
    block_size : int
        Width and height (in pixels) of each processing block.  Larger blocks
        are less sensitive to local variation; smaller blocks adapt more
        aggressively.
    offset : int
        Value subtracted from the per-block Otsu threshold before binarising.
        A positive offset makes the threshold lower (more foreground pixels).

    Returns
    -------
    numpy.ndarray of dtype bool
        Binary image composed of locally thresholded blocks.

    Notes
    -----
    Blocks near the right/bottom boundary may be smaller than ``block_size``.

    References
    ----------
    Gonzalez & Woods, 3rd ed., §10.3 (local thresholding discussion).
    """
    img = np.clip(image, 0, 255).astype(np.uint8)
    h, w = img.shape
    result = np.zeros((h, w), dtype=bool)

    for r0 in range(0, h, block_size):
        for c0 in range(0, w, block_size):
            block = img[r0:r0 + block_size, c0:c0 + block_size]
            otsu_result = otsu_threshold(block)
            local_thr = max(0, otsu_result['threshold'] - offset)
            result[r0:r0 + block_size, c0:c0 + block_size] = block > local_thr

    return result
