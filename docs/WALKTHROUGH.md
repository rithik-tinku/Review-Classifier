# Review Sentiment Classifier Walkthrough

This walkthrough details the final validation, hardening, and bug resolutions for the Review Classifier project.

---

## Bug Fix: AttributeError Resolution

- **Bug:** `AttributeError: 'Pandas' object has no attribute '_5'` in `app.py`.
- **Root Cause:** Using `.itertuples()` to access the `Issue Category` column via `row._5` is highly fragile and crashes if columns shift position or are loaded differently.
- **Fix:** Replaced `.itertuples()` with `.iterrows()` and used key-based dictionary lookup:
  - `review_text = row[review_col]`
  - `category_val = row.get("Issue Category", "Other")`
- **Verification:** Tested dashboard launch using both the default `dataset.csv` and uploaded CSV files. The dashboard runs crash-free.

---

## Final Verification Result

```text
> python tests/test_predictor.py
Loading weights: 100%|##########| 104/104 [00:00<00:00, 2924.35it/s]
All unit tests and business rules validation passed successfully!
```
- **Confusion Matrix Plots:**
  - Baseline Matrix: [baseline_confusion_matrix.png](file:///c:/Users/ACER/Desktop/PROJECTS/ReviewClassifier/Review-Classifier/plots/baseline_confusion_matrix.png)
  - Comparison Report: [model_comparison.md](file:///c:/Users/ACER/Desktop/PROJECTS/ReviewClassifier/Review-Classifier/reports/model_comparison.md)

---

## Environment Validation Details

- **Environment:** Isolated virtual environment `.venv-validation`
- **Python Version:** 3.13.14
- **Packages:**
  - NumPy: `2.5.1`
  - Pandas: `3.0.3`
  - PyTorch: `2.13.0+cpu`
  - Transformers: `5.14.1`
  - Streamlit: `1.59.2`

---

## Environment Recovery History
During audits, package dependency conflicts and PyTorch DLL errors (`WinError 4551`/`126`) arose in the old virtual environment due to AppLocker restriction rules. This was fixed by setting up a fresh, clean environment `.venv-validation` and re-installing from verified wheel builds using the system Python interpreter alias.

---

## Verification Policy & Workflow

1. **Imports:** Run `python -c "import numpy; import pandas; import sklearn; import transformers; import torch"` to verify no dynamic library load issues.
2. **Tests:** Run `python tests/test_predictor.py` to ensure unit tests succeed.
3. **Launch:** Run `python -m streamlit run app.py` to verify the dashboard executes successfully.
4. **Browser Rendering:** Real browser screenshots of the active running application must be used for verification (illustrated concept mockups are invalid).
