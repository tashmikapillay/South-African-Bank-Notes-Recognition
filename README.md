# South-African-Bank-Notes-Recognition

This project builds a full image processing and computer vision pipeline to classify South African bank notes across five denominations: R10, R20, R50, R100, and R200, covering both old and new series notes.

The pipeline follows four stages:

1. **Preprocessing and Enhancement** — grayscale conversion, CLAHE, and Gaussian blur to normalise images
2. **Segmentation** — Otsu's thresholding and Canny edge detection to isolate the note from its background
3. **Feature Extraction** — HOG, LBP, and colour histograms combined into one feature vector per image
4. **Classification** — SVM, Random Forest, and KNN trained and compared on the extracted features

The system is invariant to the side photographed (front or back), scale, and rotation.

---

## Repository Structure

```
.
├── South African Bank Note Recognition.ipynb       # Main notebook — run this
├── dataset/               # Place your image dataset here (see below)
│   ├── R10/
│   ├── R20/
│   ├── R50/
│   ├── R100/
│   └── R200/
└── README.md
```

---

## Requirements

The notebook runs on Python 3.10 or later. Install dependencies with:

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn Pillow ipywidgets joblib
```

Or if you are on Google Colab, all of these are already available except `ipywidgets`, which Colab also includes by default.

---

## Dataset Setup

The notebook expects images organised into denomination subfolders. Each subfolder should be named after its denomination exactly as shown below.

**For local execution**, place the `dataset/` folder in the same directory as the notebook:

```
South African Bank Note Recognition.ipynb
dataset/
    R10/   ← images of R10 notes (front and back, old and new series)
    R20/
    R50/
    R100/
    R200/
```

**For Google Colab**, upload the `dataset/` folder to your Google Drive at the following path before running:

```
MyDrive/COMP702/dataset/
```

The notebook automatically detects whether it is running on Colab or locally and sets the dataset path accordingly. No manual path changes are needed.

Accepted image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/) and upload `South African Bank Note Recognition.ipynb`, or open it directly from GitHub via the Colab interface.
2. Upload the `dataset/` folder to `MyDrive/dataset/` in your Google Drive.
3. Run all cells in order: **Runtime → Run all**.
4. When the last cell runs, a file upload button will appear. Use it to upload any bank note image and get a live prediction.

### Option 2 — Local Jupyter

1. Clone or download this repository.
2. Place the `dataset/` folder next to the notebook as described above.
3. Install dependencies (see Requirements).
4. Launch Jupyter and open the notebook:

```bash
South African Bank Note Recognition.ipynb
```

5. Run all cells in order. The live prediction cell at the end will show a file path input widget where you can type the path to any image you want to classify.

---

## Pipeline Summary

| Stage | Technique |
|---|---|
| Preprocessing | Grayscale, CLAHE, Gaussian blur |
| Segmentation | Otsu's thresholding (primary), Canny + contour (comparison) |
| Feature Extraction | HOG + LBP + Colour histogram (concatenated) |
| Classification | SVM (RBF kernel), Random Forest (200 trees), KNN (k=5) |

---

## Results

| Classifier | Test Accuracy |
|---|---|
| Random Forest | 98.92% |
| Support Vector Machine | 98.20% |
| K-Nearest Neighbours | 88.13% |

Tested on a stratified 80/20 train-test split of 1 386 images (154 originals expanded via augmentation).

---

## Notes

- Random seeds are fixed (`seed=42`) throughout so results are fully reproducible.
- The notebook saves several snapshot images to disk during execution (e.g. `snapshot_preprocessing.png`). These are used in the project report.
- The best trained model is saved to `best_model.pkl` via `joblib` at the end of the notebook and can be loaded for inference without retraining.
