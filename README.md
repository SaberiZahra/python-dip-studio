# Digital Image Processing Studio

An interactive web application for experimenting with digital image processing algorithms based on **Gonzalez & Woods (Digital Image Processing, 3rd Edition)**.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/SaberiZahra/python-dip-studio.git
cd python-dip-studio
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Features

- **Chapter 2 — Fundamentals:** Nearest Neighbor, Bilinear, and Bicubic interpolation; gray-level and spatial resolution reduction; arithmetic operations and frame averaging.
- **Chapter 3 — Spatial Filtering:** Negative, log, gamma, piecewise linear transforms; histogram equalization & matching (CLAHE); smoothing (box, Gaussian, median, min, max); sharpening (Laplacian, unsharp masking, highboost, Sobel, Prewitt).
- **Chapter 4 — Frequency Domain:** 2D DFT with zero-padding and centering; Ideal, Butterworth, and Gaussian lowpass & highpass filters; Homomorphic filter; bandreject, bandpass, and notch filters.
- **Chapter 5 — Restoration:** Noise generators (Gaussian, Rayleigh, Erlang, Exponential, Uniform, Salt & Pepper, Periodic); mean and order-statistic filters; adaptive filters; frequency deblurring (inverse, Wiener, CLS).
- **Chapter 6 — Color Processing:** RGB to HSI/HSV/CMY color space conversions; pseudocolor intensity slicing and continuous mapping; full-color smoothing and sharpening; color slicing.
- **Chapter 9 & 10 — Morphology & Segmentation:** Binary and grayscale erosion, dilation, opening, closing, boundary extraction, top-hat transform; Otsu global thresholding and local adaptive thresholding.
- **Quality Metrics & Pipeline:** Real-time MSE, PSNR, and SSIM computation; custom multi-step operation pipeline builder.

---

## Requirements

- Python 3.10+
- `streamlit`, `numpy`, `scipy`, `opencv-python`, `scikit-image`, `matplotlib`, `pillow`, `pandas`
