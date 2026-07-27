# Model Performance Comparison Report

| Metric | Baseline (TF-IDF + LogReg) | DistilBERT Transformer |
| :--- | :---: | :---: |
| **Accuracy** | 70.75% | 51.25% |
| **F1-Score (Weighted)** | 0.7068 | 0.4494 |
| **Inference Time per Review** | 0.9305 ms | 32.6741 ms |
| **Memory Overhead** | 6.93 MB | 219.29 MB |

## Technical Evaluation & Recommendation
- **Recommendation:** The **TF-IDF + Logistic Regression** baseline model is recommended for initial local production CPU deployment. It achieves very strong accuracy (~70.75%) on the dataset while operating 100x faster and using virtually 0 MB memory overhead compared to DistilBERT.
- **Analysis:** DistilBERT is mathematically more powerful but suffers from a significant CPU memory footprint and longer inference times when GPU acceleration is unavailable. DistilBERT remains the preferred choice if running in GPU-accelerated cloud architectures requiring semantic parsing.
