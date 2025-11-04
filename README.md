# CS446/ECE449 Machine Learning — Homeworks (2025)

This repo contains coursework for CS446/ECE449. Each homework has its own folder with code, figures, and the writeup (LaTeX/PDF). Use the per‑folder run notes below.

## Repo Structure
- `CS446_2025_hw1/` — Perceptron toy example + LaTeX writeup.
- `CS446_2025_hw2/` — Gaussian Naive Bayes (PyTorch) and Logistic Regression demo; data files included; LaTeX writeup.
- `CS446_2025_hw3/` — SVM (kernelized, PyTorch) and Linear/Ridge/Lasso regression pipeline; code in `hw3_code/`; LaTeX writeup.
- `CS446_2025_hw4/` — Bias–variance analysis (written) and polynomial regression model selection; code in `hw4_code/`; LaTeX writeup.

## Environment
- Python 3.10+ recommended.
- Common Python dependencies used across homeworks:
  - numpy, matplotlib, scikit-learn, pandas
  - torch, torchvision, pillow
  - scipy

## Per‑Homework Usage

### HW1 (`CS446_2025_hw1`)
- Perceptron 3D toy problem with decision boundary visualization:
  - `cd CS446_2025_hw1`
  - `python test.py`
- Writeup sources: `main.tex` (compiled PDF: `main.pdf`).

### HW2 (`CS446_2025_hw2`)
- Q2 — Gaussian Naive Bayes (uses provided `gaussian_train.pth` / `gaussian_test.pth`):
  - `cd CS446_2025_hw2`
  - Example quick eval in one line:
    - `python -c "import hw2_utils as u; ypred,ytest=u.gaussian_eval(); import torch; print('Accuracy:', float((ypred==ytest).float().mean()))"`
  - Or open a Python REPL and call `hw2_utils.gaussian_eval()`.
- Q4 — Logistic Regression tutorial with feature engineering, regularization, grid search, and GD variants:
  - `cd CS446_2025_hw2`
  - `python hw2_q4.py`
  - This script shows multiple plots and (when plots are enabled) saves images like `submission_4a_*.png`, `submission_4b.png`, `submission_4c.png`, `submission_4d.png`.
- Dependencies: numpy, matplotlib, scikit-learn, torch, scipy.
- Writeup source: `main.tex` (compiled PDF: `main.pdf`).

### HW3 (`CS446_2025_hw3/hw3_code`)
- Q2 — Kernel SVM (dual, PyTorch) with kernels in `hw3_utils.py`:
  - `cd CS446_2025_hw3/hw3_code`
  - `python test.py` (trains small SVMs with linear/poly/RBF kernels and plots decision regions)
- Q4 — Linear regression pipeline (OLS/Ridge closed‑form, Lasso via ISTA, log‑transform + Duan’s smearing):
  - `cd CS446_2025_hw3/hw3_code`
  - `python hw3_q4.py`
  - Downloads dataset from OpenML on first run; set `HIDE_PLOTS=1` to suppress windows.
- Dependencies: numpy, matplotlib, scikit-learn, pandas, torch, torchvision, pillow.
- Writeup source: `../hw3.tex` (compiled PDF: `../hw3.pdf`).

### HW4 (`CS446_2025_hw4/hw4_code`)
- Q1 — Bias–variance in Ridge Regression (written):
  - Writeup sources in `../hw4.tex` (compiled PDF: `../hw4.pdf`).
- Q3 — Model selection via k-fold CV for polynomial regression (Linear/Ridge/Lasso):
  - Code: `hw4_q3.py`, helpers in `hw4_utils.py`.
  - Typical use (example):
    - `from hw4_q3 import select_best_model`
    - `best = select_best_model(X_train, y_train)`
    - `best.fit(X_train, y_train); ypred = best.predict(X_test)`
  - Cross-validation: `cross_validate_model(X, y, model, k_folds=5)` returns mean/std MSE.
- Dependencies: numpy, scikit-learn. (Matplotlib only needed for the optional `test.py` visualization in the folder.)
- Submission (per instructions): upload only `hw4_q3.py` and `hw4_utils.py` to Gradescope for the programming part.
