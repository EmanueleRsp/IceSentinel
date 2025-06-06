# ❄️ IceSentinel

**A Machine Learning Approach to Avalanche Danger Level Prediction**
*Emanuele Respino | Università di Pisa, Master’s Degree in Artificial Intelligence and Data Engineering*
Academic Year 2024/2025

Welcome to **IceSentinel**! 🚀 This repository contains everything you need—code, documentation, datasets, pretrained models, and notebook workflows—to nowcast and forecast avalanche danger levels at IMIS stations in the Swiss Alps using machine learning.

---

## 🌟 Project Overview

Avalanches are one of the most lethal natural hazards in mountainous regions. Accurately nowcasting avalanche danger levels—on a five-point scale from 1 (Low) to 5 (Very High)—is crucial for public safety and alpine tourism. Traditionally, experts combine meteorological observations, snowpack measurements, and physics-based simulations (e.g., SNOWPACK) to assign a daily danger level per region.

**IceSentinel** aims to automate and streamline this process by leveraging ML pipelines. By ingesting hourly IMIS station data alongside multi-layer SNOWPACK outputs (snow-stratigraphy profiles), IceSentinel delivers real-time, reproducible avalanche danger nowcasts *without* rerunning full physics-based models for each update. This accelerates decision-making, reduces subjective bias, and enables finer spatial 👀 and temporal resolution.

**Key objectives**:

* 🧰 Build and compare multiple end-to-end ML pipelines (Random Forest, XGBoost, SVM) using various oversampling, scaling, and dimensionality-reduction strategies.
* 🔍 Perform rigorous time-aware cross-validation (expanding-window) to prevent temporal leakage.
* ⚙️ Conduct hyperparameter optimization, feature selection, and final model retraining on the full dataset.
* 🧩 Provide model interpretability via feature importance and SHAP values (global & local).
* 📊 Deploy a simple GUI (Streamlit-based) for real-time predictions at each IMIS station.

---

## 📂 Repository Structure

```plaintext
IceSentinel/
├─ documentation/
│  ├─ Project Documentation.pdf          # Full project report (chapters, methodologies, results)
│  ├─ Project Guidelines.pdf             # DMML course guidelines
│  ├─ Project Proposal.pdf               # Original project proposal
│  └─ Project Presentation.pdf           # Project report presentation slides
│
├─ interface/
│  └─ avalanche_app.py                   # Streamlit-based GUI 
│
├─ models/
│  ├─ Paper_Optimized.pkl                # Reference model from Pérez-Guillén et al. (2022) 📑
│  ├─ RF_ROS_SS_NoDR.pkl                 # Random Forest (ROSampler + StandardScaler, no PCA) 🥈
│  ├─ XGB_ROS_RS_NoDR.pkl                # XGBoost (ROSampler + RobustScaler, no PCA) 🥇
│  └─ XGB_ROS_RS_NoDR_Opt.pkl            # Final XGBoost pipeline (ROSampler + RobustScaler, no PCA, optimized) 🔥
│
├─ notebooks/
│  ├─ 01 - Exploratory Data Analysis.ipynb     # EDA: class imbalances, missing data, correlations, PCA 📊
│  ├─ 02 - Preprocessing.ipynb                 # Feature pruning, missing-value removal ✂️
│  ├─ 03 - Training.ipynb                      # Pipeline screening, hyperparameter tuning, model selection 🤖
│  ├─ 04 - Second Stage Analysis.ipynb         # Focus on discussing class 5 discrimination 🎯
│  ├─ 05 - Interpretability.ipynb              # Feature importance, SHAP analyses (global & local) 🧩
│  └─ 06 - Interface.ipynb                     # GUI prototype (Streamlit) 🌐
│
├─ resources/
│  ├─ data_rf2_tidy.csv                # Raw dataset ✨
│  └─ reduced_data_rf2_tidy.csv        # Preprocessed dataset (≈22 600 samples, 68 features) 📈
│
├─ results/                            # Summary CSVs and images (plots, tables) 🖼️
│
├─ requirements.txt                    # Python dependencies (pandas, scikit-learn, xgboost, shap, streamlit, etc.) 📦
└─ README.md                           # ← You are here 😄
```

---

## 🚀 Getting Started

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Emanuele.rsp/IceSentinel.git
   cd IceSentinel
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate         # Linux/macOS
   venv\Scripts\activate            # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip  
   pip install -r requirements.txt
   ```

   * Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `streamlit`, `matplotlib`, `seaborn`.

---

## 📝 Workflow & Notebooks

All data preprocessing and modeling steps are fully reproducible via the notebooks in `notebooks/`. In brief:

* **01 – Exploratory Data Analysis (EDA)**
  Inspect the cleaned dataset (`data_rf2_tidy.csv`), handle missing values, assess class balance, explore feature correlations, and perform PCA to gauge redundancy.

* **02 – Preprocessing**
  Drop irrelevant or fully missing columns, remove rows with incomplete data, merge rare classes, encode labels, and split data into time-aware training/validation/test sets.

* **03 – Training**
  Define and evaluate ML pipelines (sampling → scaling → optional dimensionality reduction → classifier) using expanding-window CV; tune hyperparameters for top candidates; select and retrain final models on historical data (2001–2020); model optimization via importance ranking and RFE to build a reduced-feature pipeline (≈30 variables); validate performance on both CV and hold-out.

* **04 – Second-Stage Analysis**
  Short discussion about Discriminating Class 5.

* **05 – Interpretability**
  Generate global and local explanations (feature-importance metrics, SHAP summary and dependence plots) to understand model behavior and key drivers behind each danger level.

* **06 – Interface**
  Prototype a Streamlit GUI for station/aspect selection, nowcast vs. 24 h forecast, and visualization of predicted danger level, class probabilities, and SHAP contributions.

For complete details on each step, please refer to the full project documentation in `documentation/`. 📚

---

## 📦 Pre-trained Models

All final, pipeline-optimized models are saved under `models/`. Their key details:

| Model File                | Description                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `XGB_ROS_RS_NoDR_Opt.pkl` | XGBoost classifier with RandomOverSampler + RobustScaler + no PCA; 20 features; leakage-free hyperparams 🔥  |
| `RF_ROS_RS_NoDR.pkl`      | Random Forest (ROS + RobustScaler + NoDR; included for comparison) 🌲                                        |
| `RF_ROS_SS_NoDR.pkl`      | Random Forest (ROS + StandardScaler + NoDR; included for comparison) ⚖️                                      |
| `Paper_Optimized.pkl`     | Reference RF pipeline from Pérez-Guillén et al. (2022), tuned on 30 features with their proposed workflow 📑 |

**Loading a model in Python**:

```python
import pickle

with open("models/XGB_ROS_RS_NoDR_Opt.pkl", "rb") as f:
    xgb_model = pickle.load(f)
```

---

## 🌐 Interface & Deployment

We built a simple Streamlit app (`avalanche_app.py`) to demonstrate model usage interactively.

### Running the App Locally

```bash
streamlit run avalanche_app.py
```

* The app will launch in your default browser (or at [http://localhost:8501](http://localhost:8501)).
* Ensure `results/` is in your working directory and that `results/shap/csv/<model_name>/winter2020/*.csv` are accessible: they're pre-computed shap values used for Global Explainability to avoid unnecessary computational and time costs.

---

## 📚 Documentation

Full documentation is packaged under `documentation/`:

* **Project Documentation.pdf**
  The complete write-up of methods, experiments, and results (chapters 1–6). Contains in-depth descriptions of:

  1. Introduction & objectives
  2. Data sources (IMIS stations, SNOWPACK outputs, SLF bulletins)
  3. Exploratory data analysis (EDA), feature distributions, PCA, correlation
  4. Detailed preprocessing steps (missing values, class merging, train/hold-out splits)
  5. ML methodology (pipelines, CV schemes, hyperparameter tuning)
  6. Model explainability (feature importance, SHAP global & local)
  7. Real-time GUI architecture and user flow
  8. Results & conclusions

* **Project Guidelines.pdf**
  DMML course guidelines (proposal template, evaluation criteria, timeline). 📑

* **Project Proposal.pdf**
  Initial proposal slide deck (problem statement, dataset overview, workflow outline, references). 💡

* **Project Presentation.pdf**
  Presentation slide deck (project report summary). ✨
  
---

## 📖 Citation & References

Reference papers:

```bibtex
@article{Pérez-Guillén2022,
  title   = {Data-driven automated predictions of the avalanche danger level for dry-snow conditions in Switzerland},
  author  = {P{\'e}rez-Guill{\'e}n, C. and Techel, F. and Hendrick, M. and Volpi, M. and van Herwijnen, A. and Olevski, T. and Obozinski, G. and P{\'e}rez-Cruz, F. and Schweizer, J.},
  journal = {Nat. Hazards Earth Syst. Sci.},
  volume  = {22},
  pages   = {2031--2056},
  year    = {2022},
  doi     = {10.5194/nhess-22-2031-2022}
}

@article{Pérez-Guillén2025,
  title   = {Assessing the performance and explainability of an avalanche danger forecast model},
  author  = {P{\'e}rez-Guill{\'e}n, C. and Techel, F. and Volpi, M. and van Herwijnen, A.},
  journal = {Nat. Hazards Earth Syst. Sci.},
  volume  = {25},
  pages   = {1331--1351},
  year    = {2025},
  doi     = {10.5194/nhess-25-1331-2025}
}
```

---

🙏 Thank you for exploring **IceSentinel**! For issues, questions, or contributions, please open an issue or submit a pull request on the GitHub repository:
[https://github.com/Emanuele.rsp/IceSentinel](https://github.com/Emanuele.rsp/IceSentinel)
