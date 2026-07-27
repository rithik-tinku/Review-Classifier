# Review Classifier: Ultimate Owner's Notes
*A Comprehensive Handbook for Explaining, Defending, and Maintaining the Sentiment Classifier Project*

---

## SECTION 1 — PROJECT OVERVIEW

### 1. Project Objective & Real-World Problem Solved
In quick-commerce and e-commerce, customer support and operations teams receive thousands of text reviews daily. Sorting these reviews manually is slow and prone to errors. 

This project solves this by building an automated classification system that:
1. **Detects Sentiment:** Classifies reviews as Negative, Neutral, or Positive.
2. **Extracts Issues:** Categorizes complaints into one of 21 operational categories (e.g., Late Delivery, Damaged Product, Refund).
3. **Prioritizes Actions:** Assigns severity priority tags (P0 - Critical to P3 - Low).
4. **Flags Churn Risk:** Identifies high-risk customer accounts mentioning competitor brands under negative sentiment.

### 2. End-to-End Workflow
```text
[Raw Customer Signal] 
       │
       ▼ (Regex Text Cleaning)
[Normalized Clean Text]
       │
       ▼ (Feature Extraction / Tokenization)
[Model Prediction (Baseline LogReg vs. DistilBERT)]
       │
       ▼ (Operational Rules Engine)
[Enriched Schema: Sentiment, Category, Priority, Churn]
       │
       ▼ (Dashboard Visualizations)
[Executive Action Queue & Synced CX Workflows]
```

---

## SECTION 2 — PROJECT STRUCTURE

### Directory Layout & File Catalog

```text
ReviewClassifier/
├── data/
│   ├── raw/dataset.csv            # Original Yelp Review extract (2,000 balanced rows)
│   └── processed/dataset_cleaned.csv  # Output of regex-cleansed normalization pipeline
├── models/
│   ├── baseline/                  # Checkpoints for TF-IDF Vectorizer & Logistic Regression
│   └── distilbert/                # Fine-tuned PyTorch DistilBERT model weights and configs
├── src/
│   ├── business_rules.py          # Priority, category, and churn heuristics mapping
│   ├── download_dataset.py        # Dataset downloader script (Official HF Hub with Mirror fallback)
│   ├── preprocessing.py           # Text regex normalization engine
│   ├── train_baseline.py          # TF-IDF + Logistic Regression training pipeline
│   ├── train_transformer.py       # Pinned-seed DistilBERT fine-tuning script
│   └── evaluate.py                # CPU latency, memory, and performance comparison script
├── tests/
│   └── test_predictor.py          # Edge-case testing suite (Hinglish, emoji-only, sarcasm)
├── plots/                         # Saved confusion matrices
├── reports/                       # Markdown report comparisons
├── app.py                         # Streamlit dashboard script
├── predictor.py                   # Prediction routing script
└── requirements.txt               # Pinned package dependencies
```

---

## SECTION 3 — COMPLETE DATA FLOW

```text
  Raw CSV Row (Review, Rating)
               │
               ▼
   [src/preprocessing.py] ────► Lowercase, strip HTML/URLs, normalize punctuation
               │
               ▼
      [predictor.py] ─────────► Check if review contains alphabetic letters.
               │                ├─ No: Route to VADER Fallback (e.g. Emoji, Number)
               │                └─ Yes: Tokenize & predict (Baseline / DistilBERT)
               │
               ▼
   [src/business_rules.py] ───► Map Category, Priority, and Churn Risk flags
               │
               ▼
          [app.py] ───────────► Dynamically update Metrics Cards, Pain Point Bar Charts, 
                                and trigger synced CX Action Queue Tickets
```

---

## SECTION 4 — MACHINE LEARNING

### 1. Terminology & Math

#### TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF measures a word's unique importance in a specific document compared to a larger corpus:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **Term Frequency ($\text{TF}$):** How often word $t$ appears in document $d$.
- **Inverse Document Frequency ($\text{IDF}$):** Penalizes common words (like "the", "and") appearing across all documents.

#### Logistic Regression
For multi-class classification, Logistic Regression calculates the probability of each class $c$ using the softmax function:
$$P(y = c | \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{j=1}^{C} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
This makes it extremely fast and lightweight for sparse TF-IDF vectors.

#### DistilBERT Transformer
A distilled version of BERT utilizing **knowledge distillation** to mimic the output distribution of a larger teacher network. It reduces the parameter count from 110M to 66M while maintaining 97% of BERT's performance.

### 2. Why the Baseline Outperformed DistilBERT on this Dataset
- **Data Size Constraint:** We used a 240-sample training slice to ensure practical local CPU training times. Deep learning models like DistilBERT overfit quickly on small sample sizes, whereas Logistic Regression performs very well.
- **Dimensionality:** TF-IDF + Logistic Regression generalizes well on simple keyword signals ("worst", "amazing") in smaller datasets.

---

## SECTION 5 — BUSINESS LOGIC

### Mapping Heuristics
- **Priority:**
  - **P0 - CRITICAL:** Safety, damaged/broken products, app crashes, or counterfeit goods.
  - **P1 - HIGH:** Financial transactions (refunds, billing discrepancies), missing items, or account access.
  - **P2 - MEDIUM:** Late delivery, poor packaging, wrong item, or shipping delays.
  - **P3 - LOW:** General positive or neutral feedback.
- **Churn Risk:**
  - **High:** Negative sentiment reviews that mention competitors (e.g., "switching to Blinkit") or P0/P1 issues.
  - **Medium:** General negative reviews or neutral reviews mentioning competitors.
  - **Low:** All positive reviews.

---

## SECTION 6 — STREAMLIT DASHBOARD

- **Upload & Fallback:** Allows users to upload a CSV. If empty, it automatically loads `dataset.csv` to ensure the dashboard displays data immediately on startup.
- **Metrics Cards:** Calculates model accuracy dynamically against rating ground-truth (if available) alongside real-time critical issue and churn counts.
- **Action Queue:** Displays P0 issues in high-visibility alert cards, enabling managers to assign tickets to relevant departments (Finance, Logistics, Warehouse Ops) in real-time.

---

## SECTION 7 — BUG DIARY

| Bug | Root Cause | Solution | Lesson Learned |
| :--- | :--- | :--- | :--- |
| **`ImportError: AdamW`** | Deprecated in `transformers` namespace. | Import from `torch.optim`. | Avoid relying on third-party libraries for core optimization code. |
| **`ValueError` in Evaluation** | Classification report mismatch (5-star ratings vs 3 sentiment classes). | Map star ratings into 3 classes prior to running scikit-learn metrics. | Standardize class labels across raw datasets and model outputs. |
| **`SafetensorError` on Train** | Interrupted/corrupted weights files. | Fall back to initializing model structure from local config files if weights are corrupt. | Local model validation saves network bandwidth and protects against gateway timeouts. |
| **`AttributeError` on `_5`** | `.itertuples()` creates index positions that shift dynamically if columns change. | Replace `.itertuples()` with `.iterrows()` and standard string key lookup. | Never use hardcoded positional index lookups on DataFrames. |

---

## SECTION 8 — INTERVIEW PREPARATION

### Python & Pandas (Questions 1–15)
1. **Explain the difference between `.iterrows()` and `.itertuples()`.**  
   *`.iterrows()` returns index and row Series objects (slower), while `.itertuples()` returns namedtuples (faster). However, `.iterrows()` is safer for index lookup since `.itertuples()` renames column names with spaces into positional index variables (like `_5`).*
2. **What happens when you apply `.to_dict('records')` on a DataFrame?**  
   *It converts the DataFrame into a list of dictionaries, where keys are column names, allowing safe, order-independent row parsing.*
3. **What is `pd.isna()` and how is it used in data preprocessing?**  
   *It detects missing or null values in Series, allowing strings to be converted into empty placeholders before applying regex cleaning.*
4. **How does `df.groupby("label").sample(n=100)` balance a dataset?**  
   *It groups rows by label value and samples exactly 100 items from each group, creating a balanced dataset.*
5. **How do you add a new column to a Pandas DataFrame dynamically?**  
   *By using direct assignment (e.g., `df["NewColumn"] = values_list`).*
6. **Explain `pd.concat()`.**  
   *It concatenates pandas objects along an axis (e.g., stacking dataframes of sampled labels to form a unified dataset).*
7. **What is the purpose of `random_state` in `train_test_split`?**  
   *It sets the seed for the random number generator, ensuring train-test splits are reproducible across runs.*
8. **Why do we use `.dropna()` on dataframes before accuracy evaluation?**  
   *It removes rows missing ground-truth labels, preventing calculations from crashing with NaN mismatches.*
9. **Explain the performance difference between list comprehensions and `.apply()` in Pandas.**  
   *List comprehensions are generally faster than `.apply()` for basic string manipulation because they avoid Pandas overhead.*
10. **How do you check for the existence of a column in a DataFrame?**  
   *Using the `in` operator (e.g., `'Rating' in df.columns`).*
11. **Explain `df.iterrows()` unpacking.**  
   *It returns a tuple of `(index, series)`, so it must be unpacked as `for idx, row in df.iterrows():`.*
12. **How does Python's `sys.path.insert(0, ...)` prevent import errors?**  
   *It inserts the workspace root directory at the front of the import path search list, allowing local files to be imported cleanly.*
13. **What is the difference between `==` and `is` in Python?**  
   *`==` compares values for equality, while `is` checks if two variables point to the same object in memory.*
14. **Why is `subprocess.run()` useful for backward-compatibility files?**  
   *It spawns a new process to run the modernized scripts in a clean workspace environment, returning execution codes safely.*
15. **How does `re.sub()` handle multiple patterns?**  
   *It compiles regular expressions and replaces matching character ranges (e.g., HTML tags or URLs) with spaces.*

### Machine Learning & NLP (Questions 16–35)
16. **Why use both a baseline model and a transformer model?**  
   *The baseline model provides a fast, interpretable benchmark with low CPU footprint. The transformer model captures complex semantic structures and context negation.*
17. **What is overfitting and how did we prevent it in DistilBERT training?**  
   *Overfitting occurs when a model memorizes training noise instead of general patterns. We used validation early stopping to save weights only when validation loss improved.*
18. **Explain the trade-offs of using sparse TF-IDF vs. dense transformer embeddings.**  
   *TF-IDF uses simple word frequencies, which is fast and lightweight but ignores word order. Transformers capture word order and semantic context but are slower and require more CPU memory.*
19. **Explain Precision, Recall, and F1-score.**  
   - *Precision: Of predicted positives, how many were actually positive.*
   - *Recall: Of actual positives, how many were correctly predicted.*
   - *F1-Score: Harmonic mean of Precision and Recall.*
20. **What is a Confusion Matrix?**  
   *A table layout showing true vs. predicted class counts, highlighting where the model is making errors.*
21. **Why does DistilBERT have a larger memory overhead than Logistic Regression?**  
   *DistilBERT loads 66M parameters and self-attention weight matrices into RAM, requiring ~211 MB compared to Logistic Regression's ~8 MB.*
22. **What is knowledge distillation in DistilBERT?**  
   *A compression technique where a smaller student model is trained to reproduce the output distributions of a larger teacher model.*
23. **How does the TF-IDF Inverse Document Frequency protect against common stopwords?**  
   *It assigns lower weights to words that appear in almost all documents across the corpus.*
24. **Why did we pin random seeds for PyTorch, NumPy, and random libraries?**  
   *To make the training run completely reproducible, producing identical weights and accuracy scores across executions.*
25. **Why is early stopping useful?**  
   *It terminates training or saves checkpoints only when validation metrics stop improving, preventing overfitting.*
26. **Explain the softmax function.**  
   *It normalizes raw output logits into probability distributions that sum to 1.0.*
27. **What is the difference between L1 and L2 regularization?**  
   *L1 (Lasso) drives coefficients to zero (sparse features), while L2 (Ridge) shrinks weights uniformly.*
28. **How does PyTorch's `Dataset` class interact with `DataLoader`?**  
   *`Dataset` indexes raw token inputs, while `DataLoader` groups them into batches and manages parallel data loading.*
29. **What is a tokenizer?**  
   *A module that splits text strings into numerical IDs matching a pre-trained vocabulary.*
30. **Explain sequence truncation and padding.**  
   - *Truncation: Cuts off reviews exceeding a maximum sequence length.*
   - *Padding: Appends zero-tokens to shorter reviews so all vectors in a batch are uniform.*
31. **Why did we reduce the sequence length to 32 for CPU DistilBERT training?**  
   *Attention complexity scales quadratically with sequence length. Shortening it to 32 makes CPU training practical and fast.*
32. **Explain the optimizer used in DistilBERT.**  
   *AdamW, which decouples weight decay from the gradient update calculations.*
33. **What is the VADER sentiment analyzer?**  
   *A rule-based sentiment model that maps specific lexical features (like words, punctuation, and emojis) to intensity scores.*
34. **Why is VADER used as a fallback model?**  
   *It processes inputs without alphabetical characters (like emojis or punctuation) that standard TF-IDF models ignore.*
35. **What is the classification boundary for multi-class classification?**  
   *The class index with the highest probability value.*

### Software Engineering & Architecture (Questions 36–50)
36. **Explain path traversal and how we prevented it.**  
   *Path traversal occurs when user input manipulates file paths (e.g. using `../`). We prevented this by hardcoding safe local file paths like `dataset.csv` instead of exposing them to user input.*
37. **What is a corporate gateway timeout (504) and how did we resolve it?**  
   *It occurs when firewalls block connection requests to Hugging Face. We resolved it by caching configuration metadata locally and falling back to a mirror endpoint if needed.*
38. **Explain the import redirection wrapper design pattern used.**  
   *Redirects top-level commands to modular scripts in the `src/` directory, maintaining backward-compatibility without duplicate logic.*
9. **Why is `__init__.py` placed in the `src/` folder?**  
   *It tells Python that `src` is an importable package.*
40. **How does `joblib` save scikit-learn models?**  
   *It serializes Python objects to disk using binary files, allowing them to be loaded instantly for inference.*
41. **What is the risk of using positional tuple indexes like `row._5`?**  
   *Any change to the dataset schema will shift column positions, causing the app to crash with an AttributeError.*
42. **Why is it important to use `st.session_state` in Streamlit?**  
   *Streamlit reruns the script from top to bottom on user interaction. `st.session_state` persists values across runs.*
43. **Why did we use `st.rerun()` after button assignments?**  
   *To trigger a script rerun immediately, updating the UI layout state.*
44. **What is the advantage of using a rule-driven category engine?**  
   *It is deterministic, fast, testable, and doesn't require training data to update categories.*
45. **What does `use_container_width=True` do in Streamlit?**  
   *It stretches UI elements (like dataframes or download buttons) to match the width of their container.*
46. **What is CSV injection and how does this project handle it safely?**  
   *CSV injection occurs when input text begins with formula characters (like `=`). This project handles it safely by treating all fields as text and avoiding formula executions.*
47. **Why is `psutil` useful in ML pipelines?**  
   *It monitors CPU and RAM usage, helping developers track resource overhead.*
48. **Explain why modular design is preferred over single-file scripts.**  
   *It makes the code easier to test, maintain, and collaborate on.*
49. **How do you handle warning noise in a production project?**  
   *Analyze the warnings, document if they are harmless (like optional torchvision dependencies), and handle them cleanly.*
50. **How does the project maintain complete offline prediction capabilities?**  
   *By saving tokenizer configurations and model checkpoints locally in the `models/` directory during initial setup.*

---

## SECTION 9 — REBUILD GUIDE

To rebuild this project from an empty folder:
1. **Directory Structure:** Create folders (`src`, `data/raw`, `models/baseline`, `models/distilbert`, `tests`, `plots`, `reports`).
2. **Install Packages:** Create `requirements.txt` and run `pip install -r requirements.txt`.
3. **Download Yelp Dataset:** Create `src/download_dataset.py` to fetch, balance, and save 2,000 reviews to `data/raw/dataset.csv`.
4. **Create Preprocessing:** Create `src/preprocessing.py` to clean the raw reviews using regex.
5. **Create Business Rules:** Create `src/business_rules.py` to map priority, category, and churn risk flags.
6. **Train Baseline:** Create `src/train_baseline.py` to fit and save the TF-IDF + Logistic Regression model.
7. **Train DistilBERT:** Create `src/train_transformer.py` to fine-tune DistilBERT offline.
8. **Evaluate Models:** Create `src/evaluate.py` to compare model latency and memory footprints.
9. **Build Predictor Routing:** Create `predictor.py` to manage predictions and VADER fallback.
10. **Build Unit Tests:** Create `tests/test_predictor.py` to verify edge cases and priority rules.
11. **Build Dashboard:** Create `app.py` to present the interactive Streamlit UI.

---

## SECTION 10 — IMPROVEMENTS

### 1. Realistic Next Versions
- **v1.1: Multi-Label Categorization:** Allow a review to be assigned multiple category tags (e.g. both "Late Delivery" and "Damaged Product").
- **v2.0: Active Learning Loop:** Implement an annotation queue in the dashboard where operations managers can correct misclassified sentiments, saving the corrected data to retrain the models.

### 2. Out-of-Scope Ideas
- **Real-Time Translation Engine:** Translating non-English reviews in real-time adds network latency and is best handled by external API gateways rather than the core ML pipeline.

---

## SECTION 11 — COMMAND REFERENCE

- **Launch Dashboard:** `streamlit run app.py`
- **Run Tests:** `python tests/test_predictor.py`
- **Evaluation Pipeline:** `python src/evaluate.py`
- **Clean Model Checkpoints:** `Remove-Item -Recurse models/baseline/*, models/distilbert/*`

---

## SECTION 12 — ENVIRONMENT VALIDATION & RECOVERY

### 1. Validated Environment Stack
- **Python Version:** 3.13.14
- **Official Virtual Environment:** `.venv-validation`
- **Core Pinned Packages:**
  - NumPy: `2.5.1`
  - Pandas: `3.0.3`
  - Scikit-learn: `1.9.0`
  - Transformers: `5.14.1`
  - PyTorch: `2.13.0+cpu`
  - Streamlit: `1.59.2`

### 2. Recovery History Summary
The old virtual environment (`.venv`) became corrupted due to broken binary links in Scikit-learn, regex dependency version mismatches, and PyTorch dynamic linking errors (`WinError 4551` and `WinError 126`) triggered by Windows AppLocker policy restrictions on user directory executables. 

These issues were resolved by creating a fresh, isolated virtual environment (`.venv-validation`) and rebuilding package dependencies using the trusted system Python interpreter execution alias. This restored clean import states.

### 3. Strict Verification Policy
Any modifications or PRs to this repository must pass the following verification checklist:
1. **Dependency Import Check:** Ensure `torch`, `transformers`, `pandas`, and `numpy` import without tracebacks.
2. **Unit Tests:** `python tests/test_predictor.py` passes 100%.
3. **App Launch & Render:** `python -m streamlit run app.py` starts without traceback, parses `dataset.csv` dynamically, renders metrics/charts correctly, and generates CSV exports.
4. **Validation Evidence:** AI-generated mockups, illustrative dashboards, or dummy logs are **not** acceptable. Evidence must consist of real, unedited CLI outputs and browser screenshots of `http://localhost:8501`.

---

## SECTION 13: PROJECT STATUS & DEVELOPER WORKFLOW

### 1. Current Project Status
- **Backend:** Stable
- **Prediction Engine:** Stable
- **Business Rules:** Stable
- **Dashboard:** Stable
- **Environment:** Validated (`.venv-validation`)
- **Unit Tests:** Passed
- **Documentation:** Synchronized
- **Overall Status:** **Ready for Functional QA and User Acceptance Testing (UAT)**

### 2. Recommended Developer Workflow
1. Activate the validated environment:
   ```bash
   .venv-validation\Scripts\activate
   ```
2. Run unit tests to verify baseline state:
   ```bash
   python tests/test_predictor.py
   ```
3. Launch Streamlit to inspect UI components:
   ```bash
   python -m streamlit run app.py
   ```
4. Perform local browser checks on `http://localhost:8501`.

---

## SECTION 14: CHANGELOG

### 18 July 2026
- **Environment recovery completed:** Established `.venv-validation` as the official environment.
- **Dependency audit completed:** Installed and locked NumPy `2.5.1`, Pandas `3.0.3`, and PyTorch `2.13.0+cpu`.
- **Unit testing passed:** Resolved all VADER fallback routes and validated Hinglish/sarcasm metrics.
- **Dashboard verified:** Resolved the Pandas positional `row._5` AttributeError and confirmed browser rendering via Edge screenshot.
- **Documentation synchronized:** Aligned README, walkthroughs, handbooks, and verification reports.

---

## SECTION 15 — OWNERSHIP CHECKLIST

- [ ] I can explain the math behind TF-IDF without notes.
- [ ] I can explain why Logistic Regression is a strong baseline.
- [ ] I can explain the difference between `.iterrows()` and `.itertuples()`.
- [ ] I can rebuild the text preprocessing pipeline from scratch.
- [ ] I can explain how VADER is used as a fallback for non-alphabetic inputs.
- [ ] I can explain how priority, churn, and categories are mapped in this project.
- [ ] I can confidently answer at least 40 of the interview preparation questions.
- [ ] I can explain the environment recovery history and validated PyTorch DLL loading on Windows.
