# 🔬 DIP-Studio: Digital Image Processing & Computer Vision Interactive Platform
> **Comprehensive Implementation of Gonzalez & Woods ("Digital Image Processing", 3rd Edition)**  
> Developed as a Course Project for Computer Vision / Digital Image Processing.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.14+-8CAAE6.svg)](https://scipy.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 👥 Authors & Project Team
* **Zahra Saberi**
* **Yasamin Sam**
* **Mina Torfipour**

---

## 📖 About the Project

**DIP-Studio** is an open-source, interactive web platform and educational workbench implementing the foundational algorithms of digital image processing based on the reference textbook **"Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods (3rd Edition)**.

The platform provides a unified multi-panel graphical interface that connects spatial domain operations, 2D Discrete Fourier Transforms (DFT), noise generation models, image restoration filters, color spaces, and morphological analysis with real-time parameter tuning and quantitative evaluation metrics (**MSE, PSNR, SSIM**).

---

## ✨ Key Features & Curriculum Coverage

### 📐 Chapter 2: Digital Image Fundamentals
* **Interpolation & Resampling**: Nearest Neighbor, Bilinear (4-neighbor weighted average), and Bicubic (16-neighbor cubic polynomial fitting) with zoom/scale testing.
* **Quantization & Spatial Resolution**: Bit-depth reduction ($8 \to 1$ bit) showcasing false contouring; subsampling demonstrating spatial aliasing (Moiré patterns).
* **Arithmetic Operations**: Multi-frame image averaging ($\sigma^2/K$ noise reduction), background subtraction, masking, and affine geometric transformations.

### 🎨 Chapter 3: Intensity Transformations & Spatial Filtering
* **Point Transformations**: Image Negative ($s = L-1-r$), Log Transformation ($s = c\log(1+r)$), Gamma / Power-Law correction ($s = c \cdot r^\gamma$), Piecewise-Linear contrast stretching, Gray-level slicing, and Bit-plane slicing.
* **Histogram Processing**: Global Histogram Equalization (maximizing entropy), Histogram Specification/Matching to target distributions, and Local Adaptive Histogram Equalization (CLAHE).
* **Spatial Smoothing**: Box filter, 2D Gaussian filter, Median filter, Minimum filter, and Maximum filter.
* **Spatial Sharpening**: 2nd derivative Laplacian operator ($4$- and $8$-neighbor kernels), Unsharp Masking & Highboost filtering ($g = f + k \cdot g_{\text{mask}}$), Sobel and Prewitt gradient magnitude operators.

### 🌊 Chapter 4: Filtering in the Frequency Domain
* **Complete Standard Pipeline**: $2M \times 2N$ Zero-Padding, Spectrum Centering via $(-1)^{x+y}$, 2D Fast Fourier Transform (FFT), element-wise multiplication, inverse FFT, and cropping.
* **Lowpass Filters (LPF)**: Ideal Lowpass (ILPF) demonstrating Gibbs ringing, Butterworth Lowpass (BLPF) with adjustable order $n$, and Gaussian Lowpass (GLPF) with smooth cutoff.
* **Highpass Filters (HPF)**: Ideal Highpass (IHPF), Butterworth Highpass (BHPF), Gaussian Highpass (GHPF), Frequency Laplacian, and High-Frequency Emphasis (HFE).
* **Homomorphic Filtering**: Separation of illumination ($i$) and reflectance ($r$) via logarithmic transformation for shadow correction and detail enhancement.
* **Selective Filtering**: Gaussian Bandreject and Bandpass filters, Symmetric Conjugate Notch Reject and Notch Pass filters for sinusoidal interference elimination.

### 🔧 Chapter 5: Image Restoration & Reconstruction
* **Statistical Noise Generation**: Gaussian noise ($\mu, \sigma$), Rayleigh noise, Erlang/Gamma speckle, Exponential noise, Uniform quantization noise, Bipolar Salt & Pepper impulse noise, and Periodic sinusoidal noise.
* **Spatial Restoration**: Arithmetic Mean, Geometric Mean, Harmonic Mean, Contraharmonic Mean (with positive/negative $Q$ control), Midpoint, and Alpha-Trimmed Mean filters.
* **Adaptive Filtering**: Local Adaptive Noise Reduction filter ($\sigma_\eta^2 / \sigma_L^2$), and Adaptive Median filter (Gonzalez & Woods Algorithm 5.1).
* **Frequency Domain Deblurring**: Uniform Linear Motion blur simulation, Atmospheric Turbulence Gaussian blur, Truncated Inverse filtering, Wiener / Minimum Mean Square Error (MMSE) filtering, and Constrained Least Squares (CLS) regularized restoration.

### 🎭 Chapter 6: Color Image Processing
* **Color Models & Conversion**: RGB $\leftrightarrow$ HSI (exact trigonometric conversion), RGB $\leftrightarrow$ HSV, and RGB $\leftrightarrow$ CMY/CMYK.
* **Pseudocolor (False Color)**: Intensity slicing and continuous sinusoidal pseudocolor mapping.
* **Full-Color Processing**: HSI Intensity-only smoothing/sharpening vs. independent RGB channel filtering (preventing chromatic edge shifts), Color Complement, White Balance gain adjustment, and Euclidean spherical Color Slicing.

### 🔬 Chapters 9 & 10: Morphological Processing & Segmentation
* **Binary Morphology**: Structuring element generation (disk, square, cross, diamond), Erosion ($A \ominus B$), Dilation ($A \oplus B$), Opening ($A \circ B$), Closing ($A \bullet B$), Boundary Extraction ($\beta(A)$), and Morphological Gradient.
* **Grayscale Morphology**: Grayscale erosion, dilation, opening, closing, Top-Hat transform (background flattening), and Bottom-Hat transform.
* **Optimal Segmentation**: Otsu's Global Thresholding (maximizing between-class variance $\sigma_B^2$ with separability index $\eta$), and Local Adaptive Thresholding.

### ⛓️ Interactive Pipeline Builder & Quality Evaluation
* **Multi-Step Pipeline**: Build, chain, reorder, and evaluate custom sequences of image operations with immediate intermediate step previews.
* **Quantitative Quality Metrics**: Instant calculation of Mean Squared Error (**MSE**), Peak Signal-to-Noise Ratio (**PSNR** in dB), and Structural Similarity Index (**SSIM**).

---

## 🚀 Getting Started

### 1. Prerequisites
Python 3.10 or higher installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive App
```bash
streamlit run app.py
```
Open your web browser and navigate to `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app.py                      # Multi-panel interactive Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
├── README.md                   # Project documentation & reference
└── modules/                    # Algorithmic core modules
    ├── ch2_fundamentals.py     # Sampling, interpolation, arithmetic operations
    ├── ch3_spatial.py          # Intensity transforms, histogram, spatial filters
    ├── ch4_frequency.py        # 2D DFT, LPF, HPF, homomorphic, notch filters
    ├── ch5_restoration.py      # Noise models, restoration filters, Wiener deblurring
    ├── ch6_color.py            # Color space conversions, pseudocolor, color slicing
    ├── ch9_morphology.py       # Binary & grayscale morphology, Otsu thresholding
    └── utils.py                # Quality metrics (MSE/PSNR/SSIM), benchmarks, helpers
```

---

## 📜 Academic Reference
* Gonzalez, R. C., & Woods, R. E. (2008). *Digital Image Processing* (3rd ed.). Pearson Prentice Hall.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
