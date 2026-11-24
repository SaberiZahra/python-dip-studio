"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Digital Image Processing — Gonzalez & Woods (3rd Edition)         ║
║                   Interactive Learning & Demonstration Tool                  ║
║                                                                              ║
║  Covers: Chapters 2 (Fundamentals), 3 (Spatial Filtering),                 ║
║          4 (Frequency Domain), 5 (Restoration), 6 (Color),                  ║
║          9 (Morphology), 10 (Segmentation)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from PIL import Image
import io
import sys
import os

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from modules import (
    ch2_fundamentals as ch2,
    ch3_spatial      as ch3,
    ch4_frequency    as ch4,
    ch5_restoration  as ch5,
    ch6_color        as ch6,
    ch9_morphology   as ch9,
    utils,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DIP — Gonzalez & Woods",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size:12px; }

/* Headers */
h1 { color: #38bdf8; font-weight: 800; }
h2 { color: #7dd3fc; font-weight: 700; border-bottom: 2px solid #0369a1; padding-bottom:4px; }
h3 { color: #bae6fd; }

/* Info boxes */
.theory-box {
    background: #0c1a2e;
    border-left: 4px solid #0ea5e9;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    font-size: 13.5px;
    line-height: 1.6;
}
.formula-box {
    background: #0d1f1a;
    border-left: 4px solid #34d399;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: monospace;
    font-size: 13px;
}
.metric-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def show_theory(text: str):
    st.markdown(f'<div class="theory-box">📖 <b>Theory:</b> {text}</div>',
                unsafe_allow_html=True)

def show_formula(text: str):
    st.markdown(f'<div class="formula-box">📐 {text}</div>',
                unsafe_allow_html=True)

def fig_to_st(fig, caption=""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)
    plt.close(fig)

def imshow_dark(ax, img, title="", cmap=None):
    if cmap is None:
        cmap = "gray" if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, color="#bae6fd", fontsize=10, pad=4)
    ax.axis("off")

def dark_fig(ncols=3, nrows=1, h=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, h * nrows))
    fig.patch.set_facecolor("#0f172a")
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#1e293b")
    return fig, np.array(axes).flatten() if nrows * ncols > 1 else axes

def spectrum_fig(image_gray, title="DFT Spectrum"):
    img = image_gray.astype(np.float64)
    M, N = img.shape
    fp = img * ((-1) ** (np.arange(M)[:, None] + np.arange(N)[None, :]))
    F = np.fft.fft2(fp, s=(2 * M, 2 * N))
    spec = np.log1p(np.abs(F))
    spec = (spec / spec.max() * 255).astype(np.uint8)
    return spec, F

def upload_or_sample(key="main"):
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("📁 Upload Custom Image", type=["png","jpg","jpeg","bmp","tif"],
                                    key=f"upload_{key}")
    with col2:
        sample_choice = st.selectbox("🎨 Or Choose Benchmark Sample",
            ["Cameraman", "Moon", "Coins", "Astronaut (Color)", "Checkerboard", "Gradient", "Circles", "Text Pattern"],
            key=f"sample_{key}")

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imdecode(
            np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8),
            cv2.IMREAD_COLOR)
        if img_color is not None:
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        else:
            img_color = np.stack([img, img, img], axis=-1)
        return img, img_color
    else:
        samples = utils.get_sample_images()
        name_map = {
            "Cameraman": "cameraman",
            "Moon": "moon",
            "Coins": "coins",
            "Astronaut (Color)": "astronaut",
            "Checkerboard": "checkerboard",
            "Gradient": "gradient",
            "Circles": "circles",
            "Text Pattern": "text_pattern",
        }
        key_name = name_map.get(sample_choice, "cameraman")
        sample_raw = samples.get(key_name, samples["cameraman"])
        if sample_raw.ndim == 3:
            img_color = sample_raw
            img = cv2.cvtColor(sample_raw, cv2.COLOR_RGB2GRAY)
        else:
            img = sample_raw
            img_color = np.stack([img, img, img], axis=-1)
        return img, img_color

def metrics_row(original, processed, label="Processed"):
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64) if processed is not None else orig_f
    if orig_f.shape != proc_f.shape:
        return
    mse  = utils.compute_mse(orig_f, proc_f)
    psnr = utils.compute_psnr(orig_f, proc_f)
    ssim = utils.compute_ssim(orig_f, proc_f)
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE",  f"{mse:.2f}")
    c2.metric("PSNR", f"{psnr:.2f} dB")
    c3.metric("SSIM", f"{ssim:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

CHAPTERS = {
    "🏠 Home":               "home",
    "📐 Ch.2 — Fundamentals":          "ch2",
    "🎨 Ch.3 — Intensity & Spatial":   "ch3",
    "🌊 Ch.4 — Frequency Domain":      "ch4",
    "🔧 Ch.5 — Restoration":           "ch5",
    "🎭 Ch.6 — Color Processing":      "ch6",
    "🔬 Ch.9 — Morphology":            "ch9",
    "✂️ Ch.10 — Segmentation":          "ch10",
    "📊 Quality Metrics":              "metrics",
    "⛓️ Pipeline Builder":             "pipeline",
}

with st.sidebar:
    st.markdown("## 🔬 DIP — Gonzalez & Woods")
    st.markdown("---")
    chapter_label = st.radio("**Navigate**", list(CHAPTERS.keys()),
                             label_visibility="collapsed")
    page = CHAPTERS[chapter_label]
    st.markdown("---")
    st.markdown("""
    **👥 Project Team:**
    - **Zahra Saberi**
    - **Yasamin Sam**
    - **Mina Torfipour**
    """)
    st.markdown("---")
    st.caption("Based on: Gonzalez & Woods\n*Digital Image Processing*, 3rd Ed.")
    st.caption("Implemented in Python · NumPy · OpenCV · SciPy · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════

if page == "home":
    st.title("🔬 Digital Image Processing")
    st.subheader("Interactive Learning Tool — Gonzalez & Woods, 3rd Edition")
    st.markdown("---")

    st.markdown("""
    This application implements **all major algorithms** from the textbook
    *Digital Image Processing* by Gonzalez & Woods (3rd Edition).
    Each section provides:
    - 📖 **Theory explanations** matching the textbook
    - 📐 **Exact mathematical formulas** as implemented
    - 🎛️ **Interactive controls** (sliders, dropdowns) to explore parameters
    - 📊 **Side-by-side comparisons** of original, intermediate and result images
    - 📈 **Quality metrics** (MSE, PSNR, SSIM)
    """)

    cols = st.columns(4)
    chapters_info = [
        ("📐", "Ch.2", "Fundamentals", "Interpolation, quantization, arithmetic ops"),
        ("🎨", "Ch.3", "Spatial Filtering", "Intensity transforms, histogram, smoothing, sharpening"),
        ("🌊", "Ch.4", "Frequency Domain", "DFT, LPF/HPF, homomorphic, notch filters"),
        ("🔧", "Ch.5", "Restoration", "Noise models, Wiener filter, inverse filter"),
        ("🎭", "Ch.6", "Color", "RGB↔HSI↔CMY, pseudocolor, color slicing"),
        ("🔬", "Ch.9", "Morphology", "Erosion, dilation, opening, closing, top-hat"),
        ("✂️", "Ch.10","Segmentation", "Otsu thresholding, local adaptive"),
        ("⛓️", "",     "Pipeline Builder", "Chain multiple operations step-by-step"),
    ]
    for i, (icon, num, name, desc) in enumerate(chapters_info):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:8px;padding:12px;margin:4px 0;
                        border-top:3px solid #0ea5e9;">
            <b style="color:#38bdf8;font-size:16px">{icon} {num}</b><br>
            <b style="color:#e2e8f0">{name}</b><br>
            <span style="color:#94a3b8;font-size:12px">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between chapters. Upload your own image or use a sample image.")

    st.markdown("---")
    st.markdown("""
    ### 👥 Project Team & Authors
    * **Zahra Saberi**
    * **Yasamin Sam**
    * **Mina Torfipour**
    
    *Digital Image Processing & Computer Vision Course Project*
    """)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 2 — FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch2":
    st.title("📐 Chapter 2: Digital Image Fundamentals")
    img_gray, img_color = upload_or_sample("ch2")
    if img_gray is None:
        st.warning("Please upload an image."); st.stop()

    tab1, tab2, tab3 = st.tabs([
        "🔍 Interpolation & Scaling",
        "📉 Quantization & Sampling",
        "➕ Arithmetic Operations"
    ])

    # ── Tab 1: Interpolation ─────────────────────────────────────────────────
    with tab1:
        st.header("Image Interpolation Methods")
        show_theory(
            "When resizing an image, we must estimate pixel values at non-integer "
            "coordinates. Three methods are compared: "
            "<b>Nearest Neighbor</b> (fastest, blockiest), "
            "<b>Bilinear</b> (4 neighbors, smooth), "
            "<b>Bicubic</b> (16 neighbors, sharpest edges). "
            "The scale factor first downsamples then upsamples to show quality differences."
        )
        show_formula("NN: x = round(x'/sx) &nbsp;|&nbsp; Bilinear: f(x,y) = weighted avg of 4 neighbors &nbsp;|&nbsp; Bicubic: 16-neighbor polynomial fit")

        col1, col2 = st.columns([1, 2])
        with col1:
            scale_down = st.slider("Downsample factor (then upscale back)", 2, 8, 4, key="interp_scale")
        with col2:
            st.caption("Images are downsampled then upsampled to original size, revealing quality differences.")

        with st.spinner("Computing interpolations..."):
            nn   = ch2.nearest_neighbor_interpolation(
                       ch2.nearest_neighbor_interpolation(img_gray, 1/scale_down, 1/scale_down),
                       scale_down, scale_down)
            bil  = ch2.bilinear_interpolation(
                       ch2.bilinear_interpolation(img_gray, 1/scale_down, 1/scale_down),
                       scale_down, scale_down)
            bic  = ch2.bicubic_interpolation(
                       ch2.bicubic_interpolation(img_gray, 1/scale_down, 1/scale_down),
                       scale_down, scale_down)

        # Align sizes
        h, w = img_gray.shape
        nn  = cv2.resize(nn.astype(np.uint8),  (w, h))
        bil = cv2.resize(bil.astype(np.uint8), (w, h))
        bic = cv2.resize(bic.astype(np.uint8), (w, h))

        fig, axes = dark_fig(ncols=4, h=4)
        for ax, im, t in zip(axes,
            [img_gray, nn, bil, bic],
            ["Original", "Nearest Neighbor", "Bilinear", "Bicubic"]):
            imshow_dark(ax, im, t)
        fig.tight_layout()
        fig_to_st(fig)

        st.subheader("Quality Metrics vs Original")
        c1, c2, c3 = st.columns(3)
        for col, name, proc in zip([c1,c2,c3], ["Nearest","Bilinear","Bicubic"], [nn,bil,bic]):
            with col:
                st.caption(f"**{name}**")
                metrics_row(img_gray, proc, name)

    # ── Tab 2: Quantization ──────────────────────────────────────────────────
    with tab2:
        st.header("Spatial & Intensity Resolution")
        show_theory(
            "Reducing <b>intensity resolution</b> (bit depth) decreases the number of "
            "gray levels. Below ~4 bits (16 levels), <b>false contouring</b> appears — "
            "smooth gradients show visible bands. "
            "Reducing <b>spatial resolution</b> (subsampling) demonstrates aliasing: "
            "high-frequency patterns become Moire patterns."
        )

        col1, col2 = st.columns(2)
        with col1:
            bits = st.slider("Gray levels (bit depth)", 1, 8, 4, key="quant_bits")
            reduced_q = ch2.reduce_gray_levels(img_gray, bits)
            fig, axes = dark_fig(ncols=2, h=4)
            imshow_dark(axes[0], img_gray,  f"Original (8 bits, 256 levels)")
            imshow_dark(axes[1], reduced_q, f"{bits} bits → {2**bits} levels")
            fig.tight_layout(); fig_to_st(fig)
            show_formula(f"f_k(x,y) = floor(f / 2^(8-{bits})) × 255 / (2^{bits}-1)")

        with col2:
            factor = st.slider("Subsampling factor (then upscale)", 2, 16, 4, key="sub_factor")
            subsampled = ch2.subsample_image(img_gray, factor)
            fig, axes = dark_fig(ncols=2, h=4)
            imshow_dark(axes[0], img_gray,   "Original")
            imshow_dark(axes[1], subsampled, f"Subsampled ×{factor} (Aliasing)")
            fig.tight_layout(); fig_to_st(fig)
            show_theory("Aliasing occurs when sampling rate < 2 × max_frequency (Nyquist). "
                        "High-frequency details become low-frequency Moire patterns.")

    # ── Tab 3: Arithmetic ────────────────────────────────────────────────────
    with tab3:
        st.header("Arithmetic & Logic Operations")
        show_theory(
            "Arithmetic operations on images enable: "
            "<b>averaging</b> (noise reduction by sqrt(K)), "
            "<b>subtraction</b> (background removal, change detection), "
            "<b>multiplication</b> (masking, shading correction)."
        )

        op = st.selectbox("Operation", [
            "Averaging (noise reduction across K frames)",
            "Image Subtraction (background removal)",
            "Multiplication / Masking",
            "Geometric Transform (rotation, flip, scale)"
        ])

        if "Averaging" in op:
            K = st.slider("Number of noisy frames to average (K)", 1, 50, 10)
            sigma = st.slider("Noise std dev (σ)", 5, 60, 25)
            show_formula(f"ḡ(x,y) = (1/{K}) Σᵢ gᵢ(x,y) · noise_var_reduction = σ²/{K} = {sigma**2//K}")
            rng = np.random.default_rng(42)
            noisy_frames = [
                np.clip(img_gray.astype(np.float64) + rng.normal(0, sigma, img_gray.shape), 0, 255)
                for _ in range(K)
            ]
            averaged = ch2.image_average(noisy_frames)
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray,                       "Original (clean)")
            imshow_dark(axes[1], noisy_frames[0].astype(np.uint8), f"One noisy frame (σ={sigma})")
            imshow_dark(axes[2], averaged.astype(np.uint8),       f"Averaged (K={K} frames)")
            fig.tight_layout(); fig_to_st(fig)
            metrics_row(img_gray, averaged.astype(np.uint8))

        elif "Subtraction" in op:
            show_theory("Subtracting frames detects changes. Used in security cameras, "
                        "medical imaging (subtraction angiography), and motion detection.")
            sigma2 = st.slider("Background shift σ", 5, 50, 20)
            rng = np.random.default_rng(0)
            bg   = np.clip(img_gray.astype(np.float64) + rng.normal(0, sigma2, img_gray.shape), 0, 255).astype(np.uint8)
            diff = ch2.image_subtract(img_gray, bg)
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray, "Frame 1")
            imshow_dark(axes[1], bg,       "Frame 2 (shifted)")
            imshow_dark(axes[2], diff,     "|Frame1 - Frame2|")
            fig.tight_layout(); fig_to_st(fig)

        elif "Multiplication" in op:
            show_theory("Multiplying by a mask isolates ROI. Multiplying by illumination inverse corrects shading.")
            # Create a circular mask
            h, w = img_gray.shape
            Y, X = np.ogrid[:h, :w]
            mask = ((X - w//2)**2 + (Y - h//2)**2 <= (min(h,w)//3)**2).astype(np.uint8) * 255
            result = ch2.image_multiply(img_gray, mask)
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray, "Original")
            imshow_dark(axes[1], mask,     "Circular Mask")
            imshow_dark(axes[2], result,   "Masked Image")
            fig.tight_layout(); fig_to_st(fig)

        else:  # Geometric
            col1, col2 = st.columns(2)
            with col1:
                t_type = st.selectbox("Transform", ["rotation", "flip_h", "flip_v", "scale"])
                if t_type == "rotation":
                    angle = st.slider("Angle (degrees)", -180, 180, 45)
                    result = ch2.geometric_transform(img_gray, "rotation", angle=angle)
                elif t_type == "scale":
                    sx = st.slider("Scale X", 0.25, 3.0, 1.5)
                    sy = st.slider("Scale Y", 0.25, 3.0, 1.5)
                    result = ch2.geometric_transform(img_gray, "scale", sx=sx, sy=sy)
                else:
                    result = ch2.geometric_transform(img_gray, t_type)
            with col2:
                fig, axes = dark_fig(ncols=2, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], result,   f"After {t_type}")
                fig.tight_layout(); fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 3 — INTENSITY TRANSFORMS & SPATIAL FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch3":
    st.title("🎨 Chapter 3: Intensity Transforms & Spatial Filtering")
    img_gray, img_color = upload_or_sample("ch3")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💡 Intensity Transforms",
        "📊 Histogram Processing",
        "🌊 Spatial Smoothing",
        "⚡ Spatial Sharpening"
    ])

    # ── Tab 1: Intensity Transforms ──────────────────────────────────────────
    with tab1:
        st.header("Point Intensity Transformation Functions")
        show_theory(
            "Point (pixel-by-pixel) transforms map each input gray level r to an output "
            "level s = T(r) independently of spatial position. The transform function T "
            "defines the contrast manipulation."
        )

        transform = st.selectbox("Select Transform", [
            "Image Negative",
            "Log Transform",
            "Power-Law (Gamma)",
            "Piecewise-Linear Contrast Stretch",
            "Gray-Level Slicing",
            "Bit-Plane Slicing"
        ])

        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            if transform == "Image Negative":
                show_formula("s = (L-1) - r")
                show_theory("Reverses intensity. Enhances white detail in dark regions. "
                            "Used in medical X-ray imaging.")
                result = ch3.image_negative(img_gray)

            elif transform == "Log Transform":
                show_formula("s = c · log(1 + r), c = (L-1) / log(1 + max_r)")
                show_theory("Compresses dynamic range. Maps narrow low-intensity range "
                            "to wider output. Essential for displaying Fourier spectrum.")
                result = ch3.log_transform(img_gray)

            elif transform == "Power-Law (Gamma)":
                gamma = st.slider("Gamma (γ)", 0.1, 5.0, 1.0, 0.1)
                show_formula(f"s = c · r^γ, γ = {gamma:.1f}")
                if gamma < 1:
                    show_theory(f"γ < 1: maps dark inputs to lighter output (brightens). Used to correct washed-out images.")
                elif gamma > 1:
                    show_theory(f"γ > 1: maps bright inputs to darker output (darkens). Used to correct over-bright images.")
                else:
                    show_theory("γ = 1: identity (no change). CRT monitors have γ ≈ 2.2.")
                result = ch3.power_law_transform(img_gray, gamma)

            elif transform == "Piecewise-Linear Contrast Stretch":
                show_formula("3-segment: (0,0)→(r1,s1)→(r2,s2)→(255,255)")
                r1 = st.slider("r1 (input)", 0, 254, 70)
                s1 = st.slider("s1 (output)", 0, 254, 20)
                r2 = st.slider("r2 (input)", r1+1, 255, 180)
                s2 = st.slider("s2 (output)", s1+1, 255, 235)
                show_theory("Stretches or compresses different ranges. Setting s1=0, s2=255 "
                            "maps full contrast from [r1,r2] window.")
                result = ch3.piecewise_linear_contrast_stretch(img_gray, r1, s1, r2, s2)

            elif transform == "Gray-Level Slicing":
                A = st.slider("Range low (A)", 0, 254, 100)
                B = st.slider("Range high (B)", A+1, 255, 180)
                highlight = st.checkbox("Highlight only (black outside)", value=True)
                show_formula("If A ≤ r ≤ B: s = L-1, else: s = 0 (or s = r)")
                show_theory("Highlights a specific band of gray levels. Used to enhance "
                            "specific features like lesions in medical images.")
                result = ch3.gray_level_slicing(img_gray, A, B, highlight)

            else:  # Bit-Plane Slicing
                bit = st.slider("Bit plane (0=LSB, 7=MSB)", 0, 7, 7)
                show_formula(f"b_{bit}(x,y) = floor(f / 2^{bit}) mod 2")
                show_theory("MSB (bit 7) carries most visual information. Lower bits "
                            "are noisy. Used in image compression and watermarking.")
                result = ch3.bit_plane_slicing(img_gray, bit)

        with col_view:
            # Also plot T(r) curve
            r_vals = np.arange(256)
            if transform == "Image Negative":
                s_vals = 255 - r_vals
            elif transform == "Log Transform":
                c = 255 / np.log(1 + 255)
                s_vals = c * np.log(1 + r_vals)
            elif transform == "Power-Law (Gamma)":
                s_vals = np.clip((r_vals / 255.0) ** gamma * 255, 0, 255)
            elif transform == "Piecewise-Linear Contrast Stretch":
                s_vals = np.interp(r_vals, [0, r1, r2, 255], [0, s1, s2, 255])
            elif transform == "Gray-Level Slicing":
                s_vals = np.where((r_vals >= A) & (r_vals <= B), 255,
                                   r_vals if not highlight else 0)
            else:
                s_vals = ((r_vals // (2 ** bit)) % 2) * 255

            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray, "Original")
            imshow_dark(axes[1], result.astype(np.uint8), transform)
            ax = axes[2]
            ax.set_facecolor("#1e293b")
            ax.plot(r_vals, s_vals, color="#38bdf8", lw=2)
            ax.plot([0, 255], [0, 255], "--", color="#475569", lw=1, alpha=0.5)
            ax.set_title("T(r) transform function", color="#bae6fd", fontsize=10)
            ax.set_xlabel("Input r", color="#94a3b8")
            ax.set_ylabel("Output s", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.set_xlim(0, 255); ax.set_ylim(0, 255)
            fig.tight_layout()
            fig_to_st(fig)
            metrics_row(img_gray, result.astype(np.uint8))

    # ── Tab 2: Histogram Processing ──────────────────────────────────────────
    with tab2:
        st.header("Histogram Processing")
        show_theory(
            "The image histogram p_r(r_k) = n_k / (MN) describes the probability of "
            "each gray level. Histogram Equalization redistributes levels to maximize "
            "contrast by using the CDF as the transform function. "
            "CLAHE limits amplification to avoid noise enhancement."
        )

        h_method = st.selectbox("Method", [
            "Global Histogram Equalization",
            "Histogram Specification (Matching)",
            "CLAHE (Local Adaptive)",
        ])

        if h_method == "Global Histogram Equalization":
            show_formula("s_k = T(r_k) = (L-1) · Σⱼ₌₀ᵏ p_r(rⱼ)")
            eq_img, t_func = ch3.histogram_equalization(img_gray)
            hist_orig, _ = ch3.compute_histogram(img_gray)
            hist_eq,   _ = ch3.compute_histogram(eq_img)

            fig = plt.figure(figsize=(15, 8), facecolor="#0f172a")
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
            axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
            for ax in axes: ax.set_facecolor("#1e293b")

            imshow_dark(axes[0], img_gray, "Original Image")
            imshow_dark(axes[1], eq_img,   "Equalized Image")
            ax = axes[2]; ax.plot(np.arange(256), t_func, color="#38bdf8", lw=2)
            ax.set_title("Transform T(r)", color="#bae6fd", fontsize=10)
            ax.set_xlabel("Input r", color="#94a3b8"); ax.set_ylabel("Output s", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")

            ax = axes[3]
            ax.bar(np.arange(256), hist_orig / hist_orig.max(), color="#38bdf8", alpha=0.7, width=1)
            ax.set_title("Histogram: Original", color="#bae6fd", fontsize=10)
            ax.tick_params(colors="#94a3b8")

            ax = axes[4]
            ax.bar(np.arange(256), hist_eq / hist_eq.max(), color="#34d399", alpha=0.7, width=1)
            ax.set_title("Histogram: Equalized", color="#bae6fd", fontsize=10)
            ax.tick_params(colors="#94a3b8")

            ax = axes[5]
            cdf_orig = np.cumsum(hist_orig) / hist_orig.sum()
            cdf_eq   = np.cumsum(hist_eq)   / hist_eq.sum()
            ax.plot(cdf_orig, color="#38bdf8", lw=2, label="Original CDF")
            ax.plot(cdf_eq,   color="#34d399", lw=2, label="Equalized CDF")
            ax.legend(fontsize=8, labelcolor="#e2e8f0")
            ax.set_title("CDF Comparison", color="#bae6fd", fontsize=10)
            ax.tick_params(colors="#94a3b8")

            fig_to_st(fig)
            metrics_row(img_gray, eq_img)

        elif h_method == "Histogram Specification (Matching)":
            show_formula("z = G⁻¹(T(r))  where T=source CDF, G=target CDF")
            show_theory("Maps source histogram shape to match a target distribution. "
                        "More flexible than equalization. Used in remote sensing "
                        "to make images taken at different times comparable.")
            target_type = st.selectbox("Target Distribution",
                ["Gaussian (bell-shaped)", "Uniform", "Right-skewed (bright preference)"])

            bins = np.linspace(0, 255, 256)
            if "Gaussian" in target_type:
                mean_t = st.slider("Mean", 50, 200, 128)
                std_t  = st.slider("Std Dev", 10, 80, 40)
                target_hist = np.exp(-0.5 * ((bins - mean_t) / std_t) ** 2)
            elif "Uniform" in target_type:
                target_hist = np.ones(256)
            else:
                target_hist = np.exp(-bins / 60)

            target_hist = target_hist / target_hist.sum()
            matched = ch3.histogram_specification(img_gray, target_hist)
            hist_m, _ = ch3.compute_histogram(matched)

            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray, "Original")
            imshow_dark(axes[1], matched,  "Matched")
            ax = axes[2]
            ax.set_facecolor("#1e293b")
            ax.plot(np.arange(256), target_hist / target_hist.max(), color="#f59e0b", lw=2, label="Target")
            ax.bar(np.arange(256), hist_m / (hist_m.max()+1e-9), color="#38bdf8", alpha=0.5, width=1, label="Result")
            ax.legend(fontsize=8, labelcolor="#e2e8f0")
            ax.set_title("Histogram Match", color="#bae6fd", fontsize=10)
            ax.tick_params(colors="#94a3b8")
            fig.tight_layout(); fig_to_st(fig)
            metrics_row(img_gray, matched)

        else:  # CLAHE
            clip = st.slider("Clip Limit", 1.0, 10.0, 2.0, 0.5)
            tile = st.select_slider("Tile Size", [4, 8, 16, 32, 64], value=8)
            show_formula(f"Local HE in {tile}×{tile} tiles, clip_limit={clip:.1f}")
            show_theory("CLAHE limits the contrast amplification (clip limit) to prevent "
                        "noise from being amplified in smooth regions. The redistributed "
                        "histogram bins are spread uniformly.")
            clahe_img = ch3.local_histogram_equalization(img_gray, kernel_size=tile, clip_limit=clip)
            global_eq, _ = ch3.histogram_equalization(img_gray)

            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray,   "Original")
            imshow_dark(axes[1], global_eq,  "Global HE")
            imshow_dark(axes[2], clahe_img,  f"CLAHE (tile={tile}, clip={clip:.1f})")
            fig.tight_layout(); fig_to_st(fig)
            metrics_row(img_gray, clahe_img)

    # ── Tab 3: Spatial Smoothing ─────────────────────────────────────────────
    with tab3:
        st.header("Spatial Smoothing Filters")
        show_theory(
            "Smoothing filters reduce noise by averaging neighbors. "
            "<b>Linear filters</b> (box, Gaussian) are optimal for Gaussian noise. "
            "<b>Nonlinear order-statistic filters</b> (median) are superior for "
            "impulse (salt-and-pepper) noise because they can't be overwhelmed by outliers."
        )

        filter_type = st.selectbox("Filter Type", [
            "Box Filter (Uniform Average)",
            "Gaussian Filter",
            "Median Filter",
            "Max Filter (removes pepper)",
            "Min Filter (removes salt)",
            "Compare All"
        ])

        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            ksize = st.slider("Kernel size", 3, 21, 5, 2)
            add_noise = st.checkbox("Add salt-and-pepper noise first", value=True)
            if add_noise:
                pa = st.slider("Pepper probability", 0.0, 0.2, 0.05, 0.01)
                pb = st.slider("Salt probability",   0.0, 0.2, 0.05, 0.01)
                noisy = ch5.add_salt_pepper_noise(img_gray, pa, pb)
            else:
                noisy = img_gray.copy()

        with col_view:
            sigma_g = 1.5 if ksize < 7 else ksize / 4
            if filter_type == "Box Filter (Uniform Average)":
                show_formula(f"h(x,y) = 1/{ksize}×{ksize} · 1  (uniform kernel)")
                result = ch3.box_filter(noisy, ksize)
            elif filter_type == "Gaussian Filter":
                show_formula(f"G(s,t) = exp(-(s²+t²) / 2σ²), σ = {sigma_g:.1f}")
                result = ch3.gaussian_filter_spatial(noisy, ksize, sigma_g)
            elif filter_type == "Median Filter":
                show_formula(f"f̂(x,y) = median{{g(s,t) | (s,t) ∈ S_xy}}, window={ksize}×{ksize}")
                result = ch3.median_filter_custom(noisy, ksize)
            elif filter_type == "Max Filter (removes pepper)":
                show_formula(f"f̂(x,y) = max{{g(s,t)}}")
                result = ch3.max_filter(noisy, ksize)
            elif filter_type == "Min Filter (removes salt)":
                show_formula(f"f̂(x,y) = min{{g(s,t)}}")
                result = ch3.min_filter(noisy, ksize)
            else:  # Compare All
                box    = ch3.box_filter(noisy, ksize)
                gauss  = ch3.gaussian_filter_spatial(noisy, ksize, sigma_g)
                med    = ch3.median_filter_custom(noisy, ksize)
                fig, axes = dark_fig(ncols=3, nrows=2, h=3.5)
                ax_list = axes
                for ax, im, t in zip(ax_list,
                    [img_gray, noisy, box, gauss, med, noisy],
                    ["Original", "Noisy", "Box Filter", "Gaussian", "Median", ""]):
                    imshow_dark(ax, im, t)
                ax_list[-1].axis("off")
                fig.tight_layout(); fig_to_st(fig)
                st.subheader("MSE vs Original")
                c1, c2, c3 = st.columns(3)
                for col, name, proc in zip([c1,c2,c3], ["Box","Gaussian","Median"],[box,gauss,med]):
                    with col:
                        st.caption(f"**{name}**")
                        metrics_row(img_gray, proc)
                result = None

            if filter_type != "Compare All" and result is not None:
                fig, axes = dark_fig(ncols=3, h=4)
                imshow_dark(axes[0], img_gray, "Original (clean)")
                imshow_dark(axes[1], noisy,    "Noisy input")
                imshow_dark(axes[2], result,   f"{filter_type}")
                fig.tight_layout(); fig_to_st(fig)
                metrics_row(img_gray, result.astype(np.uint8) if result is not None else img_gray)

    # ── Tab 4: Spatial Sharpening ────────────────────────────────────────────
    with tab4:
        st.header("Spatial Sharpening Filters")
        show_theory(
            "Sharpening enhances edges and fine detail by emphasizing high-frequency components. "
            "<b>Laplacian</b> uses 2nd derivatives (zero-crossings mark edges). "
            "<b>Sobel/Prewitt</b> uses 1st derivatives (gradient magnitude = edge strength). "
            "<b>Unsharp masking / Highboost</b> subtracts blurred version and adds the residual back."
        )

        sharp_method = st.selectbox("Method", [
            "Laplacian (2nd derivative)",
            "Unsharp Masking & Highboost",
            "Sobel Gradient",
            "Prewitt Gradient",
            "Compare All Sharpening"
        ])

        if sharp_method == "Laplacian (2nd derivative)":
            mode = st.radio("Kernel type", ["4-neighbor", "8-neighbor"])
            show_formula(
                "W4=[[0,1,0],[1,-4,1],[0,1,0]] &nbsp;|&nbsp; W8=[[1,1,1],[1,-8,1],[1,1,1]]"
                "<br>g(x,y) = f(x,y) - ∇²f(x,y)")
            result = ch3.laplacian_sharpen(img_gray, mode)
            fig, axes = dark_fig(ncols=3, h=4)
            lap_only = ch3.laplacian_sharpen(img_gray, mode).astype(np.int16) - img_gray.astype(np.int16)
            imshow_dark(axes[0], img_gray, "Original")
            imshow_dark(axes[1], np.clip(np.abs(lap_only), 0, 255).astype(np.uint8), "Laplacian (edges)")
            imshow_dark(axes[2], result,   "Sharpened")
            fig.tight_layout(); fig_to_st(fig)
            metrics_row(img_gray, result.astype(np.uint8))

        elif sharp_method == "Unsharp Masking & Highboost":
            k     = st.slider("Boost factor k (k=1: Unsharp, k>1: Highboost)", 0.5, 5.0, 1.0, 0.1)
            sigma = st.slider("Blur sigma for mask", 0.5, 5.0, 1.5, 0.1)
            show_formula(f"g_mask = f - gaussian(f, σ={sigma:.1f})<br>g = f + k·g_mask, k={k:.1f}")
            show_theory("The mask is the high-frequency residual. Adding it back amplifies details. "
                        "k=1 is unsharp masking; k>1 is highboost (overboosts edges).")
            result = ch3.unsharp_masking(img_gray, sigma, k)
            fig, axes = dark_fig(ncols=3, h=4)
            blurred = ch3.gaussian_filter_spatial(img_gray, kernel_size=15, sigma=sigma)
            mask = np.clip(img_gray.astype(np.int16) - blurred.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            imshow_dark(axes[0], img_gray, "Original")
            imshow_dark(axes[1], mask,     "High-freq Mask (centered)")
            imshow_dark(axes[2], result,   f"Highboost k={k:.1f}")
            fig.tight_layout(); fig_to_st(fig)
            metrics_row(img_gray, result.astype(np.uint8))

        elif "Sobel" in sharp_method or "Prewitt" in sharp_method:
            fn   = ch3.sobel_gradient if "Sobel" in sharp_method else ch3.prewitt_gradient
            name = "Sobel" if "Sobel" in sharp_method else "Prewitt"
            show_formula(f"{name}: M(x,y) = √(Gx² + Gy²) ≈ |Gx| + |Gy|")
            grad = fn(img_gray)
            fig, axes = dark_fig(ncols=4, h=4)
            imshow_dark(axes[0], img_gray, "Original")
            imshow_dark(axes[1], np.abs(grad['gx']).astype(np.uint8), f"Gx (horizontal edges)")
            imshow_dark(axes[2], np.abs(grad['gy']).astype(np.uint8), f"Gy (vertical edges)")
            imshow_dark(axes[3], grad['magnitude'].astype(np.uint8), "Gradient Magnitude")
            fig.tight_layout(); fig_to_st(fig)

        else:  # Compare All
            lap  = ch3.laplacian_sharpen(img_gray)
            sob  = ch3.sobel_gradient(img_gray)['magnitude'].astype(np.uint8)
            ush  = ch3.unsharp_masking(img_gray, 1.5, 1.0)
            hb   = ch3.unsharp_masking(img_gray, 1.5, 2.5)
            fig, axes = dark_fig(ncols=3, nrows=2, h=3.5)
            for ax, im, t in zip(axes,
                [img_gray, lap, sob, ush, hb, img_gray],
                ["Original", "Laplacian Sharp.", "Sobel Edges",
                 "Unsharp Masking (k=1)", "Highboost (k=2.5)", ""]):
                imshow_dark(ax, im, t)
            axes[-1].axis("off")
            fig.tight_layout(); fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 4 — FREQUENCY DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch4":
    st.title("🌊 Chapter 4: Filtering in the Frequency Domain")
    img_gray, _ = upload_or_sample("ch4")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔭 DFT Explorer",
        "📉 Lowpass Filters",
        "📈 Highpass Filters",
        "🎯 Selective Filters"
    ])

    # Shared DFT computation
    dft_data = ch4.compute_dft(img_gray)
    F_centered = dft_data['F']
    P, Q = dft_data['padded_size']
    D_mat = ch4.compute_distance_matrix(P, Q)
    spec  = dft_data['spectrum']

    # ── Tab 1: DFT Explorer ──────────────────────────────────────────────────
    with tab1:
        st.header("2D DFT — Spectrum Analysis")
        show_theory(
            "The 2D Discrete Fourier Transform decomposes an image into its frequency components. "
            "The <b>magnitude spectrum</b> shows which frequencies are present (how much power). "
            "The <b>phase spectrum</b> encodes spatial structure (positions of edges). "
            "<b>Low frequencies</b> (center) = slow background variations. "
            "<b>High frequencies</b> (edges) = fine detail and edges."
        )
        show_formula(
            "F(u,v) = Σₓ Σᵧ f(x,y)·e^{-j2π(ux/M + vy/N)}<br>"
            "Display: S(u,v) = c·log(1 + |F(u,v)|)  [centered at (P/2,Q/2)]"
        )

        fig, axes = dark_fig(ncols=4, h=4)
        imshow_dark(axes[0], img_gray, "Original f(x,y)")
        imshow_dark(axes[1], spec,     "Log Magnitude |F(u,v)|")
        imshow_dark(axes[2], ((dft_data['phase'] + np.pi) / (2*np.pi) * 255).astype(np.uint8),
                    "Phase ∠F(u,v)")
        # Reconstruct from spectrum only (zero phase) vs only phase
        F_mag_only = np.abs(F_centered) * np.exp(1j * 0)
        from_mag = np.real(np.fft.ifft2(F_mag_only))
        from_mag = (from_mag[:img_gray.shape[0], :img_gray.shape[1]])
        from_mag = utils.normalize_image(from_mag).astype(np.uint8)
        imshow_dark(axes[3], from_mag, "Reconstructed (mag only, zero phase)")
        fig.tight_layout()
        fig_to_st(fig)

        st.markdown("### Padding Demonstration")
        show_theory(
            "Zero-padding from M×N to 2M×2N prevents <b>wraparound error</b> (circular convolution artifact). "
            "Without padding, filtered images show ghost copies at borders."
        )
        pad_before = img_gray
        pad_after  = ch4.pad_image(img_gray)
        fig, axes = dark_fig(ncols=2, h=4)
        imshow_dark(axes[0], pad_before, f"Original ({img_gray.shape[0]}×{img_gray.shape[1]})")
        imshow_dark(axes[1], pad_after,  f"Padded ({pad_after.shape[0]}×{pad_after.shape[1]})")
        fig.tight_layout(); fig_to_st(fig)

    # ── Tab 2: Lowpass Filters ───────────────────────────────────────────────
    with tab2:
        st.header("Lowpass Filters (Smoothing)")
        show_theory(
            "Lowpass filters pass low frequencies (smooth regions) and attenuate high frequencies "
            "(edges, noise). Controlled by cutoff frequency D₀. "
            "<b>Ideal</b>: sharp cutoff → ringing (Gibbs phenomenon). "
            "<b>Butterworth</b>: smooth cutoff, order n controls roll-off. "
            "<b>Gaussian</b>: no ringing (Gaussian in freq = Gaussian in space)."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            lpf_type = st.selectbox("Filter Type", [
                "Ideal LPF", "Butterworth LPF", "Gaussian LPF", "Compare All 3"
            ], key="lpf_type")
            D0 = st.slider("Cutoff D₀ (pixels)", 5, 200, 50, key="lpf_D0")
            if "Butterworth" in lpf_type:
                n_order = st.slider("Order n", 1, 10, 2, key="lpf_n")
            show_formula(
                "ILPF: H=1 if D≤D₀ else 0<br>"
                "BLPF: H = 1/(1+(D/D₀)^2n)<br>"
                "GLPF: H = exp(-D²/2D₀²)"
            )

        with col2:
            if lpf_type == "Ideal LPF":
                H = ch4.ideal_lpf(P, Q, D0)
                result = ch4.apply_filter(F_centered, H, img_gray.shape)
                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], (H * 255).astype(np.uint8), "Ideal LPF H(u,v)")
                spec_filtered = utils.normalize_image(np.log1p(np.abs(F_centered * H))).astype(np.uint8)
                imshow_dark(axes[2], spec_filtered, "Filtered Spectrum")
                imshow_dark(axes[3], result, f"Result (D₀={D0})")
                fig.tight_layout(); fig_to_st(fig)

            elif lpf_type == "Butterworth LPF":
                H = ch4.butterworth_lpf(P, Q, D0, n_order)
                result = ch4.apply_filter(F_centered, H, img_gray.shape)
                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], (H * 255).astype(np.uint8), f"BLPF (n={n_order})")
                # Cross-section of H
                ax = axes[2]
                ax.set_facecolor("#1e293b")
                mid = H[P//2, :]
                ax.plot(mid, color="#38bdf8", lw=2)
                ax.axvline(Q//2 - D0, color="#f59e0b", ls="--", alpha=0.7, label="D₀")
                ax.axvline(Q//2 + D0, color="#f59e0b", ls="--", alpha=0.7)
                ax.set_title("H cross-section", color="#bae6fd", fontsize=10)
                ax.tick_params(colors="#94a3b8"); ax.legend(fontsize=8, labelcolor="#e2e8f0")
                imshow_dark(axes[3], result, f"BLPF Result (D₀={D0}, n={n_order})")
                fig.tight_layout(); fig_to_st(fig)

            elif lpf_type == "Gaussian LPF":
                H = ch4.gaussian_lpf(P, Q, D0)
                result = ch4.apply_filter(F_centered, H, img_gray.shape)
                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], (H * 255).astype(np.uint8), "GLPF H(u,v)")
                ax = axes[2]
                ax.set_facecolor("#1e293b")
                mid = H[P//2, :]
                ax.plot(mid, color="#34d399", lw=2)
                ax.axvline(Q//2 - D0, color="#f59e0b", ls="--", alpha=0.7, label="D₀ (H=0.607)")
                ax.axvline(Q//2 + D0, color="#f59e0b", ls="--", alpha=0.7)
                ax.set_title("H cross-section", color="#bae6fd", fontsize=10)
                ax.tick_params(colors="#94a3b8"); ax.legend(fontsize=8, labelcolor="#e2e8f0")
                imshow_dark(axes[3], result, f"GLPF Result (D₀={D0})")
                fig.tight_layout(); fig_to_st(fig)

            else:  # Compare All
                H_i = ch4.ideal_lpf(P, Q, D0)
                H_b = ch4.butterworth_lpf(P, Q, D0, 2)
                H_g = ch4.gaussian_lpf(P, Q, D0)
                ri = ch4.apply_filter(F_centered, H_i, img_gray.shape)
                rb = ch4.apply_filter(F_centered, H_b, img_gray.shape)
                rg = ch4.apply_filter(F_centered, H_g, img_gray.shape)
                fig, axes = dark_fig(ncols=4, nrows=2, h=3.5)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], ri,       f"Ideal LPF (D₀={D0})")
                imshow_dark(axes[2], rb,       f"Butterworth (n=2)")
                imshow_dark(axes[3], rg,       f"Gaussian")
                ax = axes[4]; ax.set_facecolor("#1e293b")
                ax.plot(H_i[P//2, :], color="#ef4444", lw=2, label="Ideal")
                ax.plot(H_b[P//2, :], color="#38bdf8", lw=2, label="Butterworth")
                ax.plot(H_g[P//2, :], color="#34d399", lw=2, label="Gaussian")
                ax.legend(fontsize=8, labelcolor="#e2e8f0")
                ax.set_title("Filter cross-sections H[P/2, :]", color="#bae6fd", fontsize=10)
                ax.tick_params(colors="#94a3b8")
                for ax, im, t in zip(axes[5:8],
                    [(H_i*255).astype(np.uint8),
                     (H_b*255).astype(np.uint8),
                     (H_g*255).astype(np.uint8)],
                    ["Ideal H(u,v)", "Butterworth H(u,v)", "Gaussian H(u,v)"]):
                    imshow_dark(ax, im, t)
                axes[-1].axis("off")
                fig.tight_layout(); fig_to_st(fig)

                st.subheader("Metrics vs Original")
                c1,c2,c3 = st.columns(3)
                for col, name, proc in zip([c1,c2,c3],["Ideal","Butterworth","Gaussian"],[ri,rb,rg]):
                    with col:
                        st.caption(f"**{name}**"); metrics_row(img_gray, proc)
                result = rg

            if lpf_type != "Compare All 3":
                metrics_row(img_gray, result)

    # ── Tab 3: Highpass Filters ──────────────────────────────────────────────
    with tab3:
        st.header("Highpass Filters (Sharpening)")
        show_theory(
            "Highpass filters are complementary to lowpass: H_HP = 1 - H_LP. "
            "They pass high frequencies (edges, noise) and attenuate low frequencies "
            "(illumination, background). Used for edge detection, feature extraction. "
            "<b>High-Frequency Emphasis</b> adds an offset to preserve some low-frequency content. "
            "<b>Homomorphic filtering</b> simultaneously brightens shadows and enhances details."
        )

        hpf_type = st.selectbox("Filter Type", [
            "Ideal HPF", "Butterworth HPF", "Gaussian HPF",
            "High-Frequency Emphasis (HFE)",
            "Laplacian in Freq Domain",
            "Homomorphic Filter",
        ], key="hpf_type")

        col1, col2 = st.columns([1, 2])
        with col1:
            D0h = st.slider("Cutoff D₀", 5, 200, 50, key="hpf_D0")

            if hpf_type == "Ideal HPF":
                H = ch4.ideal_hpf(P, Q, D0h)
                show_formula("H_HP = 1 - H_ILPF")
            elif hpf_type == "Butterworth HPF":
                n_h = st.slider("Order n", 1, 10, 2, key="hpf_n")
                H = ch4.butterworth_hpf(P, Q, D0h, n_h)
                show_formula("H_BHPF = 1/(1+(D₀/D)^2n)")
            elif hpf_type == "Gaussian HPF":
                H = ch4.gaussian_hpf(P, Q, D0h)
                show_formula("H_GHPF = 1 - exp(-D²/2D₀²)")
            elif hpf_type == "High-Frequency Emphasis (HFE)":
                a = st.slider("a (DC offset)", 0.0, 1.0, 0.5, 0.05, key="hfe_a")
                b = st.slider("b (HP gain)", a+0.01, 3.0, 1.5, 0.05, key="hfe_b")
                H = ch4.high_freq_emphasis(P, Q, D0h, a, b, 'butterworth', 2)
                show_formula(f"H_HFE = {a:.2f} + {b:.2f}·H_HP")
            elif hpf_type == "Laplacian in Freq Domain":
                H = ch4.laplacian_frequency(P, Q)
                H_disp = -H / (H.min() + 1e-9)
                H_disp = np.clip(H_disp, 0, 1)
                show_formula("H(u,v) = -4π²·D²(u,v)")
            else:  # Homomorphic
                gamma_L = st.slider("γL (illumination attenuation)", 0.0, 1.0, 0.25, 0.05)
                gamma_H = st.slider("γH (reflectance boost)", 1.0, 4.0, 2.0, 0.1)
                c_h    = st.slider("c (sharpness)", 0.1, 5.0, 1.0, 0.1)
                show_formula(f"H = (γH-γL)·(1-exp(-c·D²/D₀²)) + γL")
                show_theory(
                    "Illumination-reflectance model: f=i·r → ln(f)=ln(i)+ln(r). "
                    f"γL={gamma_L:.2f} attenuates slow illumination; "
                    f"γH={gamma_H:.2f} boosts high-freq reflectance detail."
                )
                result = ch4.homomorphic_filter(img_gray, D0h, gamma_L, gamma_H, c_h)
                H = None

        with col2:
            if hpf_type != "Homomorphic Filter" and H is not None:
                if hpf_type == "Laplacian in Freq Domain":
                    # Enhance via laplacian
                    laplacian_filtered = ch4.apply_filter(F_centered, H, img_gray.shape)
                    result = np.clip(img_gray.astype(np.int32) - laplacian_filtered.astype(np.int32), 0, 255).astype(np.uint8)
                    H_show = H_disp
                else:
                    result = ch4.apply_filter(F_centered, H, img_gray.shape)
                    H_show = H

                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], (np.clip(H_show, 0, 1) * 255).astype(np.uint8), "H(u,v) filter")
                spec_filtered = utils.normalize_image(np.log1p(np.abs(F_centered * H))).astype(np.uint8)
                imshow_dark(axes[2], spec_filtered, "Filtered Spectrum")
                imshow_dark(axes[3], result, hpf_type)
                fig.tight_layout(); fig_to_st(fig)
            else:
                fig, axes = dark_fig(ncols=3, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], spec,     "Original Spectrum")
                imshow_dark(axes[2], result,   "Homomorphic Result")
                fig.tight_layout(); fig_to_st(fig)

            metrics_row(img_gray, result)

    # ── Tab 4: Selective Filters ─────────────────────────────────────────────
    with tab4:
        st.header("Selective / Band Filters")
        show_theory(
            "Selective filters target specific frequency bands. "
            "<b>Bandreject</b>: removes a ring of frequencies (periodic noise from power lines). "
            "<b>Bandpass</b>: keeps only frequencies in the ring. "
            "<b>Notch Reject</b>: removes specific frequency pairs "
            "(conjugate pairs from sinusoidal interference)."
        )

        sel_type = st.selectbox("Filter Type", [
            "Bandreject (Gaussian)",
            "Bandpass (Gaussian)",
            "Notch Reject",
            "Notch Pass"
        ], key="sel_type")

        col1, col2 = st.columns([1, 2])
        with col1:
            D0s = st.slider("Center frequency D₀", 5, 150, 40, key="sel_D0")

            if sel_type in ["Bandreject (Gaussian)", "Bandpass (Gaussian)"]:
                W = st.slider("Bandwidth W", 2, 50, 10, key="sel_W")
                show_formula("H_BR = 1 - exp(-((D²-D₀²)/(D·W))²)<br>H_BP = 1 - H_BR")
                show_theory("Bandreject removes periodic noise. Bandpass isolates "
                            "periodic components. W controls ring width.")
                if "Bandreject" in sel_type:
                    H = ch4.bandreject_gaussian(P, Q, D0s, W)
                else:
                    H = ch4.bandpass_gaussian(P, Q, D0s, W)
                result = ch4.apply_filter(F_centered, H, img_gray.shape)

            else:
                # Add periodic noise then filter it
                st.markdown("**Periodic Noise Frequencies**")
                u0 = st.slider("Periodic noise u₀ (frequency x)", -60, 60, 30, key="notch_u")
                v0 = st.slider("Periodic noise v₀ (frequency y)", -60, 60, 20, key="notch_v")
                D0n = st.slider("Notch radius D₀", 5, 30, 10, key="notch_D0")
                show_formula("D_k = √((u-P/2-u_k)² + (v-Q/2-v_k)²)<br>NRF = Π H_k · H_{-k}")

                # Add periodic noise
                noisy_periodic = ch5.add_periodic_noise(img_gray, amplitude=30,
                                                         u0=u0/(img_gray.shape[0]),
                                                         v0=v0/(img_gray.shape[1]))
                dft_noisy = ch4.compute_dft(noisy_periodic)
                F_noisy   = dft_noisy['F']
                spec_noisy= dft_noisy['spectrum']

                notch_pts = [(u0, v0)]
                if "Reject" in sel_type:
                    H = ch4.notch_reject_filter(P, Q, notch_pts, D0n)
                else:
                    H = ch4.notch_pass_filter(P, Q, notch_pts, D0n)
                result = ch4.apply_filter(F_noisy, H, img_gray.shape)

        with col2:
            if sel_type in ["Bandreject (Gaussian)", "Bandpass (Gaussian)"]:
                fig, axes = dark_fig(ncols=4, h=4)
                noisy_p = ch5.add_periodic_noise(img_gray, 25, 0.1, 0.1)
                dft_noisy2 = ch4.compute_dft(noisy_p)
                F_n2 = dft_noisy2['F']
                res2 = ch4.apply_filter(F_n2, H, img_gray.shape)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], noisy_p, "With Periodic Noise")
                imshow_dark(axes[2], (H*255).astype(np.uint8), f"Filter H(u,v)")
                imshow_dark(axes[3], res2, sel_type)
                fig.tight_layout(); fig_to_st(fig)
                metrics_row(img_gray, res2)
            else:
                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], img_gray,         "Original (clean)")
                imshow_dark(axes[1], noisy_periodic,   "Periodic Noise Added")
                imshow_dark(axes[2], spec_noisy,       "Noisy Spectrum (see dots)")
                imshow_dark(axes[3], result,           sel_type)
                fig.tight_layout(); fig_to_st(fig)
                metrics_row(img_gray, result)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 5 — IMAGE RESTORATION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch5":
    st.title("🔧 Chapter 5: Image Restoration & Noise Reduction")
    img_gray, _ = upload_or_sample("ch5")

    tab1, tab2, tab3 = st.tabs([
        "🌧️ Noise Models",
        "🧹 Spatial Restoration Filters",
        "🔭 Deblurring (Freq Domain)"
    ])

    # ── Tab 1: Noise Models ──────────────────────────────────────────────────
    with tab1:
        st.header("Statistical Noise Models")
        show_theory(
            "Different acquisition conditions produce different noise distributions. "
            "Understanding the noise model is crucial for choosing the right restoration filter. "
            "The PDF characterizes the statistical properties of the noise."
        )

        noise_type = st.selectbox("Noise Model", [
            "Gaussian (σ, μ)",
            "Rayleigh",
            "Gamma / Erlang",
            "Exponential",
            "Uniform",
            "Salt & Pepper (Impulse)",
            "Periodic (Sinusoidal)",
            "Compare All Noise Types"
        ])

        col1, col2 = st.columns([1, 2])
        with col1:
            if noise_type == "Gaussian (σ, μ)":
                mu = st.slider("Mean μ", -50, 50, 0)
                sigma_n = st.slider("Std Dev σ", 1, 80, 25)
                show_formula("p(z) = (1/√(2πσ))·exp(-(z-μ)²/2σ²)")
                show_theory("Most common noise model. Models thermal electronic noise in sensors.")
                noisy = ch5.add_gaussian_noise(img_gray, mu, sigma_n)
            elif noise_type == "Rayleigh":
                a_r = st.slider("a (location)", 0, 50, 0)
                b_r = st.slider("b (scale²)", 100, 5000, 2000)
                show_formula("p(z) = (2/b)(z-a)·exp(-(z-a)²/b) for z≥a")
                show_theory("Asymmetric, right-skewed. Models range noise in radar and ultrasound imaging.")
                noisy = ch5.add_rayleigh_noise(img_gray, a_r, b_r)
            elif noise_type == "Gamma / Erlang":
                a_g = st.slider("a (rate)", 1, 20, 10)
                b_g = st.slider("b (order/shape)", 1, 10, 5)
                show_formula("p(z) = a^b·z^(b-1)·exp(-az)/(b-1)! for z≥0")
                show_theory("Models laser speckle. When b=1 it reduces to Exponential noise.")
                noisy = ch5.add_erlang_noise(img_gray, a_g, b_g)
            elif noise_type == "Exponential":
                a_e = st.slider("a (rate λ)", 1, 50, 10)
                show_formula("p(z) = a·exp(-az) for z≥0")
                show_theory("Special case of Gamma (b=1). Models coherent illumination interference.")
                noisy = ch5.add_exponential_noise(img_gray, a_e / 100.0)
            elif noise_type == "Uniform":
                a_u = st.slider("Range low a", -100, 0, -50)
                b_u = st.slider("Range high b", 1, 100, 50)
                show_formula("p(z) = 1/(b-a)  for a≤z≤b")
                show_theory("Models quantization noise from A/D conversion. Flat, equal-probability distribution.")
                noisy = ch5.add_uniform_noise(img_gray, a_u, b_u)
            elif noise_type == "Salt & Pepper (Impulse)":
                pa = st.slider("Pepper probability Pa", 0.0, 0.3, 0.05, 0.01)
                pb = st.slider("Salt probability Pb",   0.0, 0.3, 0.05, 0.01)
                show_formula("P(z=0)=Pa (pepper), P(z=255)=Pb (salt)")
                show_theory("Caused by faulty sensors, bit errors, or ADC glitches. "
                            "Best removed by median filter (non-linear).")
                noisy = ch5.add_salt_pepper_noise(img_gray, pa, pb)
            elif noise_type == "Periodic (Sinusoidal)":
                amp = st.slider("Amplitude A", 5, 100, 25)
                u0  = st.slider("Frequency u₀", 0.0, 0.5, 0.1, 0.01)
                v0  = st.slider("Frequency v₀", 0.0, 0.5, 0.0, 0.01)
                show_formula("η(x,y) = A·sin(2π·u₀·x + 2π·v₀·y)")
                show_theory("Caused by electrical interference during scanning. Visible as "
                            "conjugate bright spots in the DFT spectrum.")
                noisy = ch5.add_periodic_noise(img_gray, amp, u0, v0)
            else:  # Compare all
                noisy_g  = ch5.add_gaussian_noise(img_gray, 0, 20)
                noisy_sp = ch5.add_salt_pepper_noise(img_gray, 0.05, 0.05)
                noisy_p  = ch5.add_periodic_noise(img_gray, 25, 0.1, 0.07)
                noisy_u  = ch5.add_uniform_noise(img_gray, -40, 40)
                fig, axes = dark_fig(ncols=3, nrows=2, h=3.5)
                for ax, im, t in zip(axes,
                    [img_gray, noisy_g, noisy_sp, noisy_p, noisy_u, img_gray],
                    ["Original", "Gaussian", "Salt & Pepper", "Periodic", "Uniform", ""]):
                    imshow_dark(ax, im, t)
                axes[-1].axis("off")
                fig.tight_layout(); fig_to_st(fig)
                noisy = noisy_g

        with col2:
            if noise_type != "Compare All Noise Types":
                dft_noisy = ch4.compute_dft(noisy)
                fig, axes = dark_fig(ncols=3, h=4)
                imshow_dark(axes[0], img_gray, "Original")
                imshow_dark(axes[1], noisy,    f"Noisy ({noise_type})")
                imshow_dark(axes[2], dft_noisy['spectrum'], "Noise Spectrum")
                fig.tight_layout(); fig_to_st(fig)

                # PDF plot
                fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor="#0f172a")
                ax2.set_facecolor("#1e293b")
                noise_only = noisy.astype(np.float32) - img_gray.astype(np.float32)
                ax2.hist(noise_only.ravel(), bins=100, color="#38bdf8", alpha=0.8, density=True)
                ax2.set_title("Noise PDF (histogram)", color="#bae6fd", fontsize=10)
                ax2.tick_params(colors="#94a3b8")
                fig2.tight_layout(); fig_to_st(fig2, "Noise distribution")
                metrics_row(img_gray, noisy)

    # ── Tab 2: Spatial Restoration ───────────────────────────────────────────
    with tab2:
        st.header("Spatial Domain Restoration Filters")
        show_theory(
            "Restoration filters attempt to reverse the effect of degradation. "
            "Mean filters work for Gaussian/uniform noise. "
            "Order-statistic filters are more robust to impulse noise. "
            "Adaptive filters adjust their behavior based on local image statistics."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            noise_choice = st.selectbox("Add noise first", [
                "Salt & Pepper", "Gaussian", "Both (mixed)"
            ], key="rest_noise")
            if noise_choice == "Salt & Pepper":
                p_sp = st.slider("S&P probability", 0.01, 0.2, 0.05, 0.01)
                noisy_r = ch5.add_salt_pepper_noise(img_gray, p_sp, p_sp)
            elif noise_choice == "Gaussian":
                sig_r = st.slider("σ", 5, 60, 20)
                noisy_r = ch5.add_gaussian_noise(img_gray, 0, sig_r)
            else:
                p2 = st.slider("S&P prob", 0.02, 0.1, 0.04, 0.01)
                s2 = st.slider("Gaussian σ", 5, 40, 15)
                noisy_r = ch5.add_salt_pepper_noise(
                    ch5.add_gaussian_noise(img_gray, 0, s2), p2, p2)

            rest_type = st.selectbox("Restoration Filter", [
                "Arithmetic Mean",
                "Geometric Mean",
                "Harmonic Mean",
                "Contraharmonic Mean",
                "Alpha-Trimmed Mean",
                "Midpoint Filter",
                "Adaptive Local Noise Filter",
                "Adaptive Median Filter",
                "Compare All Mean Filters"
            ], key="rest_type")
            ksize_r = st.slider("Kernel size", 3, 15, 5, 2, key="rest_ksize")

            if rest_type == "Contraharmonic Mean":
                Q_ch = st.slider("Q parameter", -3.0, 3.0, 1.5, 0.1)
                show_formula(f"f̂ = Σg^(Q+1) / Σg^Q, Q={Q_ch:.1f}")
                if Q_ch > 0: show_theory("Q>0: removes pepper (dark) noise")
                elif Q_ch < 0: show_theory("Q<0: removes salt (bright) noise")
            elif rest_type == "Alpha-Trimmed Mean":
                d_alpha = st.slider("d (pixels trimmed from each end)", 0, ksize_r**2//4, 2)
                show_formula(f"f̂ = Σ g_trimmed / (mn - d),  d={d_alpha}")
            elif rest_type == "Adaptive Local Noise Filter":
                nv = st.slider("Noise variance σ²η", 1, 500, 100)
                show_formula("f̂ = g - (σ²η/σ²L)·(g - mL)")
            elif rest_type == "Adaptive Median Filter":
                smax_r = st.slider("Max window size S_max", 5, 15, 7, 2)
                show_formula("Adaptively grows window to find non-impulse median")

        with col2:
            with st.spinner("Computing restoration..."):
                if rest_type == "Arithmetic Mean":
                    result_r = ch5.arithmetic_mean_filter(noisy_r, ksize_r)
                    show_formula(f"f̂ = (1/{ksize_r}²) · Σg(s,t)")
                elif rest_type == "Geometric Mean":
                    result_r = ch5.geometric_mean_filter(noisy_r, ksize_r)
                    show_formula("f̂ = [Πg(s,t)]^(1/mn)")
                elif rest_type == "Harmonic Mean":
                    result_r = ch5.harmonic_mean_filter(noisy_r, ksize_r)
                    show_formula("f̂ = mn / Σ(1/g(s,t))")
                elif rest_type == "Contraharmonic Mean":
                    result_r = ch5.contraharmonic_mean_filter(noisy_r, ksize_r, Q_ch)
                elif rest_type == "Alpha-Trimmed Mean":
                    result_r = ch5.alpha_trimmed_mean_filter(noisy_r, ksize_r, d_alpha)
                elif rest_type == "Midpoint Filter":
                    result_r = ch5.midpoint_filter(noisy_r, ksize_r)
                    show_formula("f̂ = (z_min + z_max) / 2")
                elif rest_type == "Adaptive Local Noise Filter":
                    result_r = ch5.adaptive_local_noise_filter(noisy_r, nv, ksize_r)
                elif rest_type == "Adaptive Median Filter":
                    result_r = ch5.adaptive_median_filter(noisy_r, smax_r)
                else:  # Compare all
                    am  = ch5.arithmetic_mean_filter(noisy_r, ksize_r)
                    gm  = ch5.geometric_mean_filter(noisy_r, ksize_r)
                    med = ch5.alpha_trimmed_mean_filter(noisy_r, ksize_r, 2)
                    adm = ch5.adaptive_median_filter(noisy_r, 7)
                    fig, axes = dark_fig(ncols=3, nrows=2, h=3.5)
                    for ax, im, t in zip(axes,
                        [img_gray, noisy_r, am, gm, med, adm],
                        ["Original", "Noisy", "Arithmetic Mean",
                         "Geometric Mean", "Alpha-Trimmed", "Adaptive Median"]):
                        imshow_dark(ax, im, t)
                    fig.tight_layout(); fig_to_st(fig)
                    st.subheader("Metrics")
                    c1,c2,c3,c4 = st.columns(4)
                    for col, name, proc in zip([c1,c2,c3,c4],
                        ["Arith.","Geom.","Alpha-trim","Adapt.Med."],
                        [am, gm, med, adm]):
                        with col:
                            st.caption(f"**{name}**"); metrics_row(img_gray, proc)
                    result_r = adm

            if rest_type != "Compare All Mean Filters":
                fig, axes = dark_fig(ncols=3, h=4)
                imshow_dark(axes[0], img_gray,  "Original (clean)")
                imshow_dark(axes[1], noisy_r,   f"Noisy ({noise_choice})")
                imshow_dark(axes[2], result_r,  rest_type)
                fig.tight_layout(); fig_to_st(fig)
                metrics_row(img_gray, result_r)

    # ── Tab 3: Deblurring ────────────────────────────────────────────────────
    with tab3:
        st.header("Frequency Domain Deblurring")
        show_theory(
            "Deblurring (image restoration) reverses a known or estimated degradation H(u,v). "
            "In frequency domain: G = H·F + N → estimate F̂. "
            "<b>Inverse filter</b>: F̂ = G/H (noise-sensitive). "
            "<b>Wiener filter</b>: balances sharpness vs noise (optimal in MMSE sense). "
            "<b>CLS filter</b>: regularized inverse using Laplacian penalty."
        )
        show_formula("G(u,v) = H(u,v)·F(u,v) + N(u,v)")

        col1, col2 = st.columns([1, 2])
        with col1:
            blur_type = st.selectbox("Degradation (Blur) Type", [
                "Gaussian Blur", "Motion Blur"
            ])
            if blur_type == "Gaussian Blur":
                blur_sigma = st.slider("Blur sigma", 1, 20, 5)
                show_formula("H_blur(u,v) = exp(-D²/2σ²)")
                blur_kernel = ch5.gaussian_blur_kernel(*img_gray.shape, blur_sigma)
            else:
                a_m = st.slider("Horizontal velocity a", 0.0, 0.3, 0.1, 0.01)
                b_m = st.slider("Vertical velocity b",   0.0, 0.3, 0.0, 0.01)
                show_formula("H_motion = T·sin(π(ua+vb))/(π(ua+vb))·e^{-jπ(ua+vb)}")
                blur_kernel = ch5.motion_blur_kernel(*img_gray.shape, a=a_m, b=b_m)

            noise_level = st.slider("Add Gaussian noise σ", 0, 30, 5)
            rest_method = st.selectbox("Restoration Method", [
                "Inverse Filter (truncated)",
                "Wiener Filter",
                "CLS Filter",
                "Compare All 3"
            ])
            if rest_method == "Inverse Filter (truncated)":
                thresh = st.slider("H threshold", 0.001, 0.5, 0.05, 0.001)
            elif rest_method == "Wiener Filter":
                K_w = st.slider("K (noise/signal power ratio)", 0.0001, 0.5, 0.01,
                                format="%.4f", key="wiener_K")
                show_formula(f"F̂ = [H*/(|H|²+K)]·G, K={K_w:.4f}")
            elif rest_method == "CLS Filter":
                gamma_cls = st.slider("γ (regularization)", 0.0001, 1.0, 0.001, format="%.4f")
                show_formula("F̂ = [H*/(|H|²+γ|P|²)]·G")

        with col2:
            # Apply blur in frequency domain
            F_orig = np.fft.fft2(img_gray.astype(np.float64))
            H_blur = blur_kernel  # already freq domain
            G_blurred = H_blur * F_orig
            if noise_level > 0:
                rng = np.random.default_rng(42)
                noise_freq = np.fft.fft2(rng.normal(0, noise_level, img_gray.shape))
                G_blurred = G_blurred + noise_freq

            blurred_img = np.real(np.fft.ifft2(G_blurred))
            blurred_img = np.clip(blurred_img, 0, 255).astype(np.uint8)

            if rest_method != "Compare All 3":
                if rest_method == "Inverse Filter (truncated)":
                    F_hat = ch5.inverse_filter(G_blurred, H_blur, thresh)
                elif rest_method == "Wiener Filter":
                    F_hat = ch5.wiener_filter(G_blurred, H_blur, K_w)
                else:
                    F_hat = ch5.constrained_least_squares_filter(G_blurred, H_blur, gamma_cls)

                restored = np.real(np.fft.ifft2(F_hat))
                restored = np.clip(restored, 0, 255).astype(np.uint8)

                fig, axes = dark_fig(ncols=3, h=4)
                imshow_dark(axes[0], img_gray,   "Original f(x,y)")
                imshow_dark(axes[1], blurred_img, f"Degraded g(x,y)")
                imshow_dark(axes[2], restored,    rest_method)
                fig.tight_layout(); fig_to_st(fig)
                metrics_row(img_gray, restored)
            else:
                inv = ch5.inverse_filter(G_blurred, H_blur, 0.05)
                wie = ch5.wiener_filter(G_blurred, H_blur, 0.01)
                cls = ch5.constrained_least_squares_filter(G_blurred, H_blur, 0.001)

                r_inv = np.clip(np.real(np.fft.ifft2(inv)), 0, 255).astype(np.uint8)
                r_wie = np.clip(np.real(np.fft.ifft2(wie)), 0, 255).astype(np.uint8)
                r_cls = np.clip(np.real(np.fft.ifft2(cls)), 0, 255).astype(np.uint8)

                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], img_gray,   "Original")
                imshow_dark(axes[1], blurred_img,"Degraded")
                imshow_dark(axes[2], r_inv,      "Inverse Filter")
                imshow_dark(axes[3], r_wie,      "Wiener Filter")
                fig.tight_layout(); fig_to_st(fig)
                st.subheader("Metrics")
                c1,c2,c3 = st.columns(3)
                for col, name, proc in zip([c1,c2,c3],["Inverse","Wiener","CLS"],[r_inv,r_wie,r_cls]):
                    with col: st.caption(f"**{name}**"); metrics_row(img_gray, proc)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 6 — COLOR
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch6":
    st.title("🎭 Chapter 6: Color Image Processing")
    img_gray, img_color = upload_or_sample("ch6")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Color Spaces",
        "🌈 Pseudocolor",
        "🖌️ Full-Color Filtering",
        "✂️ Color Segmentation"
    ])

    # ── Tab 1: Color Spaces ──────────────────────────────────────────────────
    with tab1:
        st.header("Color Space Transformations")
        show_theory(
            "Different color spaces separate image information differently. "
            "<b>RGB</b>: device-oriented, all channels coupled. "
            "<b>HSI</b>: separates hue (color), saturation (purity), intensity (brightness). "
            "<b>HSV</b>: similar to HSI but value=max(R,G,B) instead of mean. "
            "<b>CMY/CMYK</b>: subtractive model for printing."
        )

        cs_type = st.selectbox("Show", [
            "RGB Channels",
            "RGB → HSI → RGB (round-trip)",
            "RGB → HSV",
            "RGB → CMY",
            "Compare All Channels"
        ])

        img_c = img_color.astype(np.float32) / 255.0

        if cs_type == "RGB Channels":
            fig, axes = dark_fig(ncols=4, h=4)
            imshow_dark(axes[0], img_color,              "RGB Original")
            imshow_dark(axes[1], img_color[:,:,0],       "R channel (Red)")
            imshow_dark(axes[2], img_color[:,:,1],       "G channel (Green)")
            imshow_dark(axes[3], img_color[:,:,2],       "B channel (Blue)")
            fig.tight_layout(); fig_to_st(fig)

        elif "HSI" in cs_type:
            show_formula(
                "H = arccos((0.5((R-G)+(R-B)))/√((R-G)²+(R-B)(G-B)))<br>"
                "S = 1 - 3·min(R,G,B)/(R+G+B)<br>"
                "I = (R+G+B)/3"
            )
            hsi = ch6.rgb_to_hsi(img_c)
            H, S, I = hsi[:,:,0], hsi[:,:,1], hsi[:,:,2]
            rgb_back = ch6.hsi_to_rgb(hsi)
            fig, axes = dark_fig(ncols=4, nrows=2, h=3.5)
            imshow_dark(axes[0], img_color, "Original RGB")
            imshow_dark(axes[1], (H*255).astype(np.uint8), "Hue H")
            imshow_dark(axes[2], (S*255).astype(np.uint8), "Saturation S")
            imshow_dark(axes[3], (I*255).astype(np.uint8), "Intensity I")
            imshow_dark(axes[4], (np.clip(rgb_back,0,1)*255).astype(np.uint8), "Reconstructed RGB")
            diff = np.abs(img_c - np.clip(rgb_back,0,1))
            imshow_dark(axes[5], (diff*10*255).astype(np.uint8), "Round-trip Error ×10")
            axes[6].axis("off"); axes[7].axis("off")
            fig.tight_layout(); fig_to_st(fig)

        elif "HSV" in cs_type:
            hsv = ch6.rgb_to_hsv(img_c)
            fig, axes = dark_fig(ncols=4, h=4)
            imshow_dark(axes[0], img_color,                       "Original RGB")
            imshow_dark(axes[1], (hsv[:,:,0]*255).astype(np.uint8), "Hue H")
            imshow_dark(axes[2], (hsv[:,:,1]*255).astype(np.uint8), "Saturation S")
            imshow_dark(axes[3], (hsv[:,:,2]*255).astype(np.uint8), "Value V")
            fig.tight_layout(); fig_to_st(fig)

        elif "CMY" in cs_type:
            show_formula("C = 1-R, M = 1-G, Y = 1-B  (normalize to [0,1])")
            cmy = ch6.rgb_to_cmy(img_c)
            fig, axes = dark_fig(ncols=4, h=4)
            imshow_dark(axes[0], img_color,                       "Original RGB")
            imshow_dark(axes[1], (cmy[:,:,0]*255).astype(np.uint8), "Cyan C")
            imshow_dark(axes[2], (cmy[:,:,1]*255).astype(np.uint8), "Magenta M")
            imshow_dark(axes[3], (cmy[:,:,2]*255).astype(np.uint8), "Yellow Y")
            fig.tight_layout(); fig_to_st(fig)

        else:
            hsi = ch6.rgb_to_hsi(img_c)
            hsv = ch6.rgb_to_hsv(img_c)
            cmy = ch6.rgb_to_cmy(img_c)
            fig, axes = dark_fig(ncols=4, nrows=3, h=3)
            imgs = [
                img_color[:,:,0], img_color[:,:,1], img_color[:,:,2], img_color,
                hsi[:,:,0]*255,   hsi[:,:,1]*255,   hsi[:,:,2]*255,   img_color,
                cmy[:,:,0]*255,   cmy[:,:,1]*255,   cmy[:,:,2]*255,   img_color,
            ]
            titles = [
                "R","G","B","RGB",
                "Hue H","Sat S","Int I","[HSI]",
                "Cyan C","Magenta M","Yellow Y","[CMY]"
            ]
            for ax, im, t in zip(axes, imgs, titles):
                imshow_dark(ax, np.clip(im, 0, 255).astype(np.uint8), t)
            fig.tight_layout(); fig_to_st(fig)

    # ── Tab 2: Pseudocolor ───────────────────────────────────────────────────
    with tab2:
        st.header("Pseudocolor (False Color) Processing")
        show_theory(
            "The human eye can distinguish thousands of colors but only ~20 gray levels. "
            "Pseudocolor maps gray values to distinct colors, dramatically enhancing "
            "perceptual discrimination. Used in medical imaging, thermal cameras, satellite data."
        )

        pc_type = st.selectbox("Pseudocolor Method", [
            "Intensity Slicing",
            "Continuous Color Mapping (Sinusoidal)",
            "Matplotlib Colormaps"
        ])

        if pc_type == "Intensity Slicing":
            n_levels = st.slider("Number of color bands", 2, 16, 6)
            thresholds = [int(i * 255 / n_levels) for i in range(1, n_levels)]
            # Generate distinct colors
            cmap = plt.get_cmap("hsv")
            colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255))
                      for c in [cmap(i/n_levels) for i in range(n_levels)]]
            show_formula(f"Thresholds: {thresholds}")
            result_pc = ch6.intensity_slicing_pseudocolor(img_gray, thresholds, colors)
            fig, axes = dark_fig(ncols=2, h=4)
            imshow_dark(axes[0], img_gray,  "Grayscale Original")
            imshow_dark(axes[1], result_pc, f"Pseudocolor ({n_levels} bands)")
            fig.tight_layout(); fig_to_st(fig)

        elif pc_type == "Continuous Color Mapping (Sinusoidal)":
            show_formula("R=sin(πI), G=sin(πI+π/3), B=sin(πI+2π/3)  where I∈[0,1]")
            result_pc = ch6.continuous_pseudocolor(img_gray)
            fig, axes = dark_fig(ncols=2, h=4)
            imshow_dark(axes[0], img_gray,  "Grayscale Original")
            imshow_dark(axes[1], result_pc, "Continuous Pseudocolor")
            fig.tight_layout(); fig_to_st(fig)

        else:
            cmap_name = st.selectbox("Colormap", [
                "jet", "hot", "cool", "rainbow", "viridis", "plasma", "inferno"
            ])
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0f172a")
            for a in ax: a.set_facecolor("#1e293b")
            ax[0].imshow(img_gray, cmap="gray")
            ax[0].set_title("Grayscale", color="#bae6fd", fontsize=10)
            ax[0].axis("off")
            im2 = ax[1].imshow(img_gray, cmap=cmap_name)
            ax[1].set_title(f"Pseudocolor ({cmap_name})", color="#bae6fd", fontsize=10)
            ax[1].axis("off")
            plt.colorbar(im2, ax=ax[1])
            fig.tight_layout(); fig_to_st(fig)

    # ── Tab 3: Full-Color Filtering ──────────────────────────────────────────
    with tab3:
        st.header("Full-Color Image Filtering")
        show_theory(
            "Color images can be filtered in RGB (channel-by-channel) or in HSI space "
            "(only the Intensity channel). HSI filtering preserves color fidelity since "
            "H and S are unchanged. RGB filtering may cause color shifts at edges."
        )

        cfilt_type = st.selectbox("Filter", [
            "Smooth in HSI vs RGB",
            "Sharpen in HSI vs RGB",
            "Color Complement",
            "Color Balance"
        ])

        if cfilt_type == "Smooth in HSI vs RGB":
            sigma_c = st.slider("Gaussian σ", 0.5, 10.0, 2.0, 0.1)
            show_formula("HSI: filter I only. RGB: filter R, G, B separately.")
            rgb_smooth = ch6.smooth_color_image(img_color, sigma=sigma_c, in_hsi=False)
            hsi_smooth = ch6.smooth_color_image(img_color, sigma=sigma_c, in_hsi=True)
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_color, "Original")
            imshow_dark(axes[1], rgb_smooth,"RGB smoothing")
            imshow_dark(axes[2], hsi_smooth,"HSI smoothing (I channel only)")
            fig.tight_layout(); fig_to_st(fig)

        elif cfilt_type == "Sharpen in HSI vs RGB":
            rgb_sharp = ch6.sharpen_color_image(img_color, in_hsi=False)
            hsi_sharp = ch6.sharpen_color_image(img_color, in_hsi=True)
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_color, "Original")
            imshow_dark(axes[1], rgb_sharp, "RGB sharpening")
            imshow_dark(axes[2], hsi_sharp, "HSI sharpening (I only)")
            fig.tight_layout(); fig_to_st(fig)

        elif cfilt_type == "Color Complement":
            comp = ch6.color_complement(img_color)
            fig, axes = dark_fig(ncols=2, h=4)
            imshow_dark(axes[0], img_color, "Original")
            imshow_dark(axes[1], comp,      "Color Complement")
            fig.tight_layout(); fig_to_st(fig)

        else:
            r_g = st.slider("Red gain",   0.5, 2.0, 1.0, 0.05)
            g_g = st.slider("Green gain", 0.5, 2.0, 1.0, 0.05)
            b_g = st.slider("Blue gain",  0.5, 2.0, 1.0, 0.05)
            balanced = ch6.color_balance(img_color, r_g, g_g, b_g)
            fig, axes = dark_fig(ncols=2, h=4)
            imshow_dark(axes[0], img_color, "Original")
            imshow_dark(axes[1], balanced,  "Color Balanced")
            fig.tight_layout(); fig_to_st(fig)

    # ── Tab 4: Color Segmentation ────────────────────────────────────────────
    with tab4:
        st.header("Color-Based Segmentation")
        show_theory(
            "Colors can be used to segment/isolate objects of interest. "
            "Euclidean distance in RGB or HSI space measures color similarity. "
            "Pixels within radius W₀ of the target color are kept; others are grayed out."
        )
        show_formula("D(z, a) = √((zR-aR)² + (zG-aG)² + (zB-aB)²) ≤ W₀")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Target Color (RGB)**")
            tR = st.slider("Target R", 0, 255, 150, key="cseg_R")
            tG = st.slider("Target G", 0, 255, 50,  key="cseg_G")
            tB = st.slider("Target B", 0, 255, 50,  key="cseg_B")
            radius_seg = st.slider("Radius W₀", 10, 150, 60)

            # Show target color swatch
            swatch = np.full((50, 200, 3), [tR, tG, tB], dtype=np.uint8)
            st.image(swatch, caption=f"Target: RGB({tR},{tG},{tB})", width=200)

        with col2:
            sliced = ch6.color_slicing(img_color, [tR, tG, tB], radius_seg)
            mask   = ch6.segment_by_color(img_color, [tR, tG, tB], radius_seg)
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_color, "Original")
            imshow_dark(axes[1], mask * 255,"Binary Mask")
            imshow_dark(axes[2], sliced,    "Color Sliced")
            fig.tight_layout(); fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 9 — MORPHOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch9":
    st.title("🔬 Chapter 9: Morphological Image Processing")
    img_gray, _ = upload_or_sample("ch9")

    tab1, tab2, tab3 = st.tabs([
        "⬛ Binary Morphology",
        "🔲 Grayscale Morphology",
        "🎩 Top-Hat & Gradient"
    ])

    # Binary thresholding
    threshold_m = st.sidebar.slider("Binary threshold (for morphology)", 0, 255, 127)
    binary_img = (img_gray > threshold_m).astype(np.uint8)

    with tab1:
        st.header("Binary Morphological Operations")
        show_theory(
            "Binary morphology operates on binary (foreground/background) images using "
            "a structuring element (SE) B. The SE's shape and size determine what "
            "structures are affected. All operations are defined set-theoretically."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            se_shape = st.selectbox("Structuring Element Shape",
                ["disk", "square", "cross", "diamond"], key="morph_se")
            se_size  = st.select_slider("SE Size", [3, 5, 7, 9, 11, 15], value=5, key="morph_size")
            morph_op = st.selectbox("Operation", [
                "Erosion (A⊖B)",
                "Dilation (A⊕B)",
                "Opening (A∘B = Erosion then Dilation)",
                "Closing (A•B = Dilation then Erosion)",
                "Boundary Extraction β(A) = A - A⊖B",
                "Morphological Gradient",
                "Compare All Basic Ops"
            ], key="morph_op")

            B = ch9.get_structuring_element(se_shape, se_size)
            se_display = B.astype(np.uint8) * 255

        with col2:
            show_formulas = {
                "Erosion (A⊖B)":           ("A⊖B = {z | (B)_z ⊆ A}", "Shrinks objects, removes protrusions smaller than SE, separates touching objects."),
                "Dilation (A⊕B)":          ("A⊕B = {z | (B̂)_z ∩ A ≠ ∅}", "Grows objects, fills holes smaller than SE, bridges small gaps."),
                "Opening (A∘B = Erosion then Dilation)": ("A∘B = (A⊖B)⊕B", "Removes small bright objects/noise. Smooths contours. Preserves overall shape."),
                "Closing (A•B = Dilation then Erosion)": ("A•B = (A⊕B)⊖B", "Fills small holes. Bridges narrow gaps. Smooth dark contours."),
                "Boundary Extraction β(A) = A - A⊖B": ("β(A) = A − (A⊖B)", "1-pixel-wide boundary of objects. Used for perimeter measurement."),
                "Morphological Gradient":  ("grad = (A⊕B) − (A⊖B)", "Highlights edges of objects. Wider than boundary extraction."),
            }

            if morph_op != "Compare All Basic Ops":
                formula, theory = show_formulas.get(morph_op, ("", ""))
                show_formula(formula)
                show_theory(theory)

                if "Erosion" in morph_op:
                    result_m = ch9.binary_erosion_custom(binary_img, B).astype(np.uint8) * 255
                elif "Dilation" in morph_op:
                    result_m = ch9.binary_dilation_custom(binary_img, B).astype(np.uint8) * 255
                elif "Opening" in morph_op:
                    result_m = ch9.opening(binary_img, B).astype(np.uint8) * 255
                elif "Closing" in morph_op:
                    result_m = ch9.closing(binary_img, B).astype(np.uint8) * 255
                elif "Boundary" in morph_op:
                    result_m = ch9.boundary_extraction(binary_img, B).astype(np.uint8) * 255
                else:  # Gradient
                    result_m = ch9.morphological_gradient(binary_img, B).astype(np.uint8) * 255

                fig, axes = dark_fig(ncols=4, h=4)
                imshow_dark(axes[0], binary_img * 255, "Binary Input")
                imshow_dark(axes[1], se_display,       f"SE ({se_shape} {se_size}×{se_size})")
                imshow_dark(axes[2], result_m,          morph_op)
                # Overlay: show what changed
                diff_m = np.abs(result_m.astype(np.int32) - (binary_img*255).astype(np.int32)).astype(np.uint8)
                imshow_dark(axes[3], diff_m, "Change (|result - input|)")
                fig.tight_layout(); fig_to_st(fig)

            else:
                er  = ch9.binary_erosion_custom(binary_img, B).astype(np.uint8) * 255
                di  = ch9.binary_dilation_custom(binary_img, B).astype(np.uint8) * 255
                op  = ch9.opening(binary_img, B).astype(np.uint8) * 255
                cl  = ch9.closing(binary_img, B).astype(np.uint8) * 255
                bnd = ch9.boundary_extraction(binary_img, B).astype(np.uint8) * 255
                fig, axes = dark_fig(ncols=3, nrows=2, h=3.5)
                for ax, im, t in zip(axes,
                    [binary_img*255, er, di, op, cl, bnd],
                    ["Binary Input", "Erosion ⊖", "Dilation ⊕",
                     "Opening ∘", "Closing •", "Boundary β"]):
                    imshow_dark(ax, im, t)
                fig.tight_layout(); fig_to_st(fig)

    with tab2:
        st.header("Grayscale Morphological Operations")
        show_theory(
            "Grayscale morphology extends binary morphology to intensity images. "
            "Dilation uses the max operator; erosion uses the min operator. "
            "These provide more nuanced control over image texture and contrast."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            gs_ksize = st.select_slider("Kernel size", [3, 5, 7, 9, 11, 15, 21], value=7, key="gs_ksize")
            gs_op = st.selectbox("Operation", [
                "Grayscale Erosion (darkens)",
                "Grayscale Dilation (brightens)",
                "Grayscale Opening",
                "Grayscale Closing",
            ])

        with col2:
            if "Erosion" in gs_op:
                show_formula("(f⊖b)(x,y) = min{f(x+s,y+t) - b(s,t)}")
                result_gs = ch9.gray_erosion_custom(img_gray, kernel_size=gs_ksize)
                show_theory("Darkens image. Removes bright peaks smaller than kernel. Used to find dark objects.")
            elif "Dilation" in gs_op:
                show_formula("(f⊕b)(x,y) = max{f(x-s,y-t) + b(s,t)}")
                result_gs = ch9.gray_dilation_custom(img_gray, kernel_size=gs_ksize)
                show_theory("Brightens image. Fills dark valleys. Used to find bright objects.")
            elif "Opening" in gs_op:
                show_formula("f∘b = (f⊖b)⊕b")
                result_gs = ch9.gray_opening(img_gray, gs_ksize)
                show_theory("Removes bright features smaller than SE. Used for background estimation.")
            else:
                show_formula("f•b = (f⊕b)⊖b")
                result_gs = ch9.gray_closing(img_gray, gs_ksize)
                show_theory("Fills dark features smaller than SE. Removes dark spots.")

            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray,   "Original")
            imshow_dark(axes[1], result_gs,  gs_op)
            diff_gs = np.clip(img_gray.astype(np.int16) - result_gs.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            imshow_dark(axes[2], diff_gs,    "Difference (centered at 128)")
            fig.tight_layout(); fig_to_st(fig)
            metrics_row(img_gray, result_gs)

    with tab3:
        st.header("Top-Hat Transform & Morphological Gradient")
        show_theory(
            "Top-Hat transform isolates bright features (or dark features with Bottom-Hat) "
            "that are smaller than the structuring element. Essential when illumination is "
            "uneven: T_hat(f) = f - (f∘b) removes the slowly-varying background, "
            "leaving small bright objects."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            th_ksize = st.select_slider("SE kernel size", [15, 25, 35, 51, 75], value=25, key="th_ksize")
            show_formula("Top-Hat: T_hat(f) = f - (f∘b)<br>Bottom-Hat: B_hat(f) = (f•b) - f")
            show_theory("Large SE: captures large-scale background. "
                        "Result contains only features smaller than SE. "
                        "Used in PCB inspection, cell counting, text extraction.")

        with col2:
            tophat    = ch9.tophat_transform(img_gray, th_ksize)
            bottomhat = ch9.bottomhat_transform(img_gray, th_ksize)
            grad      = ch9.gray_erosion_custom(img_gray, kernel_size=5)  # for display
            grad_img  = ch9.morphological_gradient(binary_img, ch9.get_structuring_element('disk', 3))

            fig, axes = dark_fig(ncols=4, h=4)
            imshow_dark(axes[0], img_gray,                   "Original")
            imshow_dark(axes[1], tophat,                     f"Top-Hat (k={th_ksize})")
            imshow_dark(axes[2], bottomhat,                  f"Bottom-Hat (k={th_ksize})")
            imshow_dark(axes[3], grad_img.astype(np.uint8)*255, "Morph. Gradient (binary)")
            fig.tight_layout(); fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHAPTER 10 — SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ch10":
    st.title("✂️ Chapter 10: Image Segmentation — Thresholding")
    img_gray, _ = upload_or_sample("ch10")

    show_theory(
        "Thresholding partitions an image into regions. The simplest approach: "
        "b(x,y) = 1 if f(x,y) > T, else 0. The challenge: how to choose T? "
        "<b>Otsu's method</b> automatically finds the optimal T by maximizing "
        "the between-class variance of the two resulting regions."
    )

    tab1, tab2 = st.tabs(["🎯 Otsu Global Thresholding", "🗺️ Local Adaptive Thresholding"])

    with tab1:
        st.header("Otsu's Global Thresholding (Section 10.3)")
        show_formula(
            "σ²_B(k) = [μ_T·ω(k) − μ(k)]² / [ω(k)·(1−ω(k))]<br>"
            "k* = argmax_k σ²_B(k)  [optimal threshold]"
        )
        show_theory(
            "Otsu's method: maximize between-class variance (= minimize within-class variance). "
            "Assumes bimodal histogram. Separability measure η = σ²_B(k*) / σ²_T ∈ [0,1]. "
            "η=1: perfectly separated. η<0.5: poor thresholding candidate."
        )

        result_otsu = ch9.otsu_threshold(img_gray)
        T_star      = result_otsu['threshold']
        eta         = result_otsu['separability']
        binary_otsu = result_otsu['binary'].astype(np.uint8) * 255

        # Manual threshold for comparison
        T_manual = st.slider("Manual threshold (compare)", 0, 255, int(T_star))

        col1, col2 = st.columns(2)
        with col1:
            hist, _ = ch3.compute_histogram(img_gray)
            sigma_B2 = []
            mu_T = sum(i * hist[i] for i in range(256)) / hist.sum()
            cumhist = np.cumsum(hist)
            cummean = np.cumsum(np.arange(256) * hist)
            for k in range(256):
                omega = cumhist[k] / hist.sum()
                mu_k  = cummean[k] / hist.sum()
                if omega > 0 and omega < 1:
                    sb2 = (mu_T * omega - mu_k) ** 2 / (omega * (1 - omega) + 1e-10)
                else:
                    sb2 = 0
                sigma_B2.append(sb2)

            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
            ax2.set_facecolor("#1e293b")
            ax2.bar(np.arange(256), hist / hist.max(), color="#38bdf8", alpha=0.5, width=1, label="Histogram")
            ax2.plot(np.array(sigma_B2) / max(sigma_B2), color="#f59e0b", lw=2, label="σ²_B(k)")
            ax2.axvline(T_star,  color="#34d399", lw=2, ls="--", label=f"Otsu T*={T_star}")
            ax2.axvline(T_manual, color="#ef4444", lw=1.5, ls=":", label=f"Manual T={T_manual}")
            ax2.legend(fontsize=8, labelcolor="#e2e8f0")
            ax2.set_title("Histogram + Between-class variance σ²_B(k)", color="#bae6fd", fontsize=10)
            ax2.tick_params(colors="#94a3b8")
            fig2.tight_layout(); fig_to_st(fig2)

        with col2:
            binary_manual = (img_gray > T_manual).astype(np.uint8) * 255
            fig, axes = dark_fig(ncols=3, h=4)
            imshow_dark(axes[0], img_gray,      "Original")
            imshow_dark(axes[1], binary_otsu,   f"Otsu (T*={T_star})")
            imshow_dark(axes[2], binary_manual, f"Manual (T={T_manual})")
            fig.tight_layout(); fig_to_st(fig)

            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | **Otsu threshold T*** | {T_star} |
            | **Separability η** | {eta:.4f} |
            """)

    with tab2:
        st.header("Local Adaptive Thresholding")
        show_theory(
            "Global threshold fails when illumination is non-uniform. "
            "Local thresholding computes a different threshold for each small block/window. "
            "This handles varying background and lighting conditions."
        )

        blk = st.select_slider("Block size", [11, 21, 31, 51, 71, 101, 151], value=51)
        off = st.slider("Offset", 0, 50, 10)
        show_formula(f"T_local(x,y) = local_otsu(block_{blk}×{blk}) - {off}")

        local_binary = ch9.local_thresholding(img_gray, blk, off)
        _, global_otsu = ch3.histogram_equalization(img_gray)
        global_bin = (img_gray > T_star).astype(np.uint8) * 255

        fig, axes = dark_fig(ncols=3, h=4)
        imshow_dark(axes[0], img_gray,                     "Original")
        imshow_dark(axes[1], global_bin,                   f"Global Otsu (T*={T_star})")
        imshow_dark(axes[2], local_binary.astype(np.uint8)*255, f"Local Adaptive (blk={blk})")
        fig.tight_layout(); fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "metrics":
    st.title("📊 Image Quality Metrics")
    img_gray, _ = upload_or_sample("metrics")

    show_theory(
        "Quality metrics quantify the difference between a reference and processed image. "
        "<b>MSE</b>: mean squared pixel difference (lower = better). "
        "<b>PSNR</b>: logarithmic ratio of max signal to noise (higher = better, >40dB is excellent). "
        "<b>SSIM</b>: perceptual similarity [0,1] based on luminance, contrast, and structure "
        "(higher = better, >0.9 is excellent)."
    )
    show_formula(
        "MSE = (1/MN) Σ (f-f̂)²<br>"
        "PSNR = 10·log₁₀(255²/MSE) [dB]<br>"
        "SSIM = (2μₓμᵧ+C₁)(2σₓᵧ+C₂) / ((μₓ²+μᵧ²+C₁)(σₓ²+σᵧ²+C₂))"
    )

    # Compare original with various degradations
    noises = [
        ("Gaussian σ=10",   ch5.add_gaussian_noise(img_gray, 0, 10)),
        ("Gaussian σ=25",   ch5.add_gaussian_noise(img_gray, 0, 25)),
        ("Salt&Pepper 5%",  ch5.add_salt_pepper_noise(img_gray, 0.05, 0.05)),
        ("Blurred (σ=3)",   ch3.gaussian_filter_spatial(img_gray, 15, 3)),
        ("JPEG-like (γ=1.5)", ch3.power_law_transform(img_gray, 1.5)),
        ("Neg+Equalized",   ch3.histogram_equalization(ch3.image_negative(img_gray))[0]),
    ]

    rows = []
    for name, proc in noises:
        mse  = utils.compute_mse(img_gray.astype(np.float64), proc.astype(np.float64))
        psnr = utils.compute_psnr(img_gray.astype(np.float64), proc.astype(np.float64))
        ssim = utils.compute_ssim(img_gray.astype(np.float64), proc.astype(np.float64))
        rows.append({"Operation": name, "MSE": f"{mse:.2f}", "PSNR (dB)": f"{psnr:.2f}", "SSIM": f"{ssim:.4f}"})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Visual comparison
    fig, axes = dark_fig(ncols=4, nrows=2, h=3.5)
    imshow_dark(axes[0], img_gray, "Original")
    for i, (name, proc) in enumerate(noises[:7]):
        if i + 1 < len(axes):
            imshow_dark(axes[i+1], proc, name)
    fig.tight_layout(); fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "pipeline":
    st.title("⛓️ Pipeline Builder — Step-by-Step Processing")
    img_gray, img_color = upload_or_sample("pipeline")

    show_theory(
        "Chain multiple operations together to build a custom processing pipeline. "
        "Each step shows the intermediate result, allowing you to visualize the "
        "cumulative effect of each operation on the image."
    )

    st.markdown("### 🔧 Add Steps to Pipeline")

    if "pipeline_steps" not in st.session_state:
        st.session_state.pipeline_steps = []

    available_ops = {
        "Histogram Equalization": lambda img, **kw: ch3.histogram_equalization(img)[0],
        "Gaussian Smoothing (σ=2)": lambda img, **kw: ch3.gaussian_filter_spatial(img, 9, 2),
        "Median Filter (5×5)": lambda img, **kw: ch3.median_filter_custom(img, 5),
        "Laplacian Sharpen": lambda img, **kw: ch3.laplacian_sharpen(img),
        "Sobel Edges": lambda img, **kw: ch3.sobel_gradient(img)['magnitude'].astype(np.uint8),
        "Gamma 0.5 (brighten)": lambda img, **kw: ch3.power_law_transform(img, 0.5),
        "Gamma 2.0 (darken)": lambda img, **kw: ch3.power_law_transform(img, 2.0),
        "GLPF (D₀=50)": lambda img, **kw: ch4.apply_filter(
            ch4.compute_dft(img)['F'],
            ch4.gaussian_lpf(*ch4.pad_image(img).shape, 50),
            img.shape),
        "GHPF (D₀=50)": lambda img, **kw: ch4.apply_filter(
            ch4.compute_dft(img)['F'],
            ch4.gaussian_hpf(*ch4.pad_image(img).shape, 50),
            img.shape),
        "Add Gaussian Noise": lambda img, **kw: ch5.add_gaussian_noise(img, 0, 20),
        "Add S&P Noise (3%)": lambda img, **kw: ch5.add_salt_pepper_noise(img, 0.03, 0.03),
        "Top-Hat Transform": lambda img, **kw: ch9.tophat_transform(img, 25),
        "Otsu Threshold": lambda img, **kw: ch9.otsu_threshold(img)['binary'].astype(np.uint8) * 255,
        "Morphological Opening": lambda img, **kw: ch9.gray_opening(img, 5),
        "Image Negative": lambda img, **kw: ch3.image_negative(img),
        "Homomorphic (γL=0.25, γH=2)": lambda img, **kw: ch4.homomorphic_filter(img, 50, 0.25, 2.0),
    }

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_op = st.selectbox("Select operation", list(available_ops.keys()), key="pipe_op")
    with col2:
        if st.button("➕ Add Step", use_container_width=True):
            st.session_state.pipeline_steps.append(new_op)
    with col3:
        if st.button("🗑️ Clear Pipeline", use_container_width=True):
            st.session_state.pipeline_steps = []

    if st.session_state.pipeline_steps:
        st.markdown("**Current Pipeline:**")
        for i, step in enumerate(st.session_state.pipeline_steps):
            sc1, sc2 = st.columns([6, 1])
            sc1.markdown(f"`{i+1}.` **{step}**")
            if sc2.button("❌", key=f"rm_{i}"):
                st.session_state.pipeline_steps.pop(i)
                st.rerun()

        st.markdown("---")
        st.subheader("📸 Pipeline Results")

        current = img_gray.copy()
        intermediates = [(current.copy(), "Input")]

        for step_name in st.session_state.pipeline_steps:
            fn = available_ops[step_name]
            try:
                current = fn(current)
                if current is not None:
                    current = np.clip(current, 0, 255).astype(np.uint8)
                    intermediates.append((current.copy(), step_name))
            except Exception as e:
                st.error(f"Error in step '{step_name}': {e}")
                break

        # Display all intermediates
        n = len(intermediates)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = dark_fig(ncols=n_cols, nrows=n_rows, h=3.5)
        ax_list = np.array(axes).flatten() if n > 1 else [axes]
        for i, (im, title) in enumerate(intermediates):
            if i < len(ax_list):
                imshow_dark(ax_list[i], im, f"[{i}] {title}")
        for j in range(len(intermediates), len(ax_list)):
            ax_list[j].axis("off")
        fig.tight_layout(); fig_to_st(fig)

        # Metrics: final vs original
        if len(intermediates) > 1:
            st.subheader("Final Image Quality")
            metrics_row(img_gray, intermediates[-1][0], "Final")
    else:
        st.info("👆 Add steps above to build your pipeline.")
