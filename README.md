# 🛒 MartPulse - AI-Powered Mart Sales Prediction | DSN Hackathon 2026

[Python](https://img.shields.io/badge/Python-3.11%2B-blue)
[Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B)
[Model](https://img.shields.io/badge/Model-RandomForest_120_Trees-success)
[RMSE](https://img.shields.io/badge/RMSE-1128.15-brightgreen)
[Status](https://img.shields.io/badge/Status-Competition_Ready-blueviolet)

> **An end-to-end machine learning pipeline that predicts total sales for retail products across Nigerian supermarket chains, with a deployable Streamlit intelligence dashboard featuring batch test.csv upload.**

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [Exploratory Data Analysis - Key Findings](#-exploratory-data-analysis---key-findings)
- [Feature Engineering](#-feature-engineering)
- [Modeling Strategy](#-modeling-strategy)
- [Results & Leaderboard](#-results--leaderboard)
- [Deployment - Streamlit Dashboard](#-deployment---streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Submission Generation](#-submission-generation)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement
Mart Sales dataset presents a regression challenge: **Predict `total_sales` (₦)** for products across different stores.

Business Goal: Help Nigerian retailers optimize pricing, shelf allocation, and store format strategy to maximize revenue. A 1% improvement in forecast accuracy can save millions in inventory waste.

**Target:** `total_sales` (Continuous)  
**Metric:** RMSE (Root Mean Squared Error) - Lower is better

## 📦 Dataset Overview

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | ~6,800 | Historical sales with target |
| `test.csv` | ~2,900 | Holdout set for leaderboard |

**Features:**
- `product_weight_kg`, `product_price`, `shelf_visibility` (Numerical)
- `fat_content` (Low Fat / Regular)
- `product_category` - 15+ categories with dirty casing (e.g., `DAIRY`, `dairy`, `Dairy`)
- `store_code` (OUT001 - OUT049), `store_age_years`, `store_size` (Small/Medium/Large)
- `store_location_tier` (Tier_1/2/3), `store_format` (Flagship Hypermarket, etc.)

## 🔍 Exploratory Data Analysis - Key Findings

### Factor 1: Price is King
Strong positive correlation **(r=0.57)** between `product_price` and `total_sales`. Higher-priced premium products drive disproportionately higher revenue. This is the #1 feature in our model.

### Factor 2: Location Tier Paradox
Tier_2 stores have the highest median sales (~₦1,900), followed by Tier_3, while Tier_1 has the lowest. However, Tier_3 shows extreme high-value outliers >₦12,000, suggesting luxury buying pockets.

### Factor 3: Store Format Dominates
`Flagship Hypermarket` median ~₦3,200 (2x higher than Standard Supermarket), `Corner Shop` median <₦500. Store format is a game-changer and must be one-hot encoded.

### Factor 4: Shelf Visibility Paradox
Negative correlation (r=-0.11). Highest sales concentrated at very low visibility (0.00-0.05). Beyond 0.15 visibility, sales collapse. Indicates slow-moving products are given more shelf space - classic retail trap.

### Factor 5: Product Category Dirty Data
`seafood` (2,850 mean sales) is top seller, `Breakfast` lowest. **Critical Issue:** `STARCHY FOODS`, `Starchy Foods`, `starchy foods` exist as 3 separate categories. Standardizing this gave +5% LB boost.

### Factor 6: Store Size & Age
- Medium & Large stores: ~₦2,250 mean vs Small: ₦1,900
- Older stores (35-47 years) show perfectly stable high sales ~₦2,260, with anomaly dip at 34 years (data error to cap).

## 🛠️ Feature Engineering

```python
# 1. Dirty Category Cleaning - Most Important
df['product_category'] = df['product_category'].str.lower().map(mapping).fillna('Others')

# 2. Store Code Label Encoding
store_code_map = {'OUT001':0, 'OUT002':1, ... } # 17 stores

# 3. Interaction Features for Competition Edge
- price_per_weight = product_price / product_weight_kg
- price_per_visibility = product_price / (shelf_visibility + 0.01)
- is_flagship = (store_format == Flagship Hypermarket)
- is_old_store = (store_age_years > 35)
- price_x_tier = product_price * store_location_tier

# 4. Outlier Handling
- Cap store_age 34 years anomaly
- Clip shelf_visibility > 0.25
```

## 🤖 Modeling Strategy

We experimented with 4 models using 5-Fold CV:

| Model | CV RMSE | Size | Deployment |
|-------|---------|------|------------|
| Linear Regression | 1450.2 | 1KB | Too weak |
| LightGBM 500 trees | 1125.8 | 3.5MB | Best score |
| RandomForest 800 trees | 1128.15 | 850MB | Too big - EOF error |
| **RandomForest 120 trees (Final)** | **1128.5** | **25MB** | **Best Deployable** |

**Final Choice:** `RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_leaf=5, max_features='sqrt')`

Why RF for deployment?
- Most stable across folds
- No Python version mismatch issues (unlike LGBM with Python 3.14)
- 25MB with `compress=3` - instant Streamlit load
- Interpretable for business stakeholders

## 🏆 Results & Leaderboard

- **Public LB RMSE:** 1128.15
- **Improvement vs Baseline (Mean):** -22%
- **Model Size:** 25MB (vs 850MB original)
- **Inference Time:** <50ms per prediction

Feature Importance from RF:
1. product_price (0.42)
2. store_format_Flagship (0.18)
3. product_category_Seafood (0.12)
4. store_location_tier (0.08)
5. shelf_visibility (0.07)

## 🚀 Deployment - Streamlit Dashboard

Professional dashboard built to impress DSN judges:

**Features that make you stand out:**
1.  **Single Prediction:** Beautiful gradient card with Gauge chart, ROI calculator
2.  **📤 Batch Upload (Killer Feature):** Upload raw `test.csv` → Auto-cleans dirty categories, handles `OUT001` string → Predicts all rows → Downloads `submission.csv` ready for leaderboard
3.  **Business Insights:** Plotly charts showing price correlation, store format impact

**Colors:** Professional Navy `#0F172A` → Royal Blue `#1E3A8A` gradient with Electric Blue `#0EA5E9` accents.

Run locally:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
DSN-Mart-Sales-Prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── dsn_best_model.joblib (25MB - RF 120 trees)
│   └── model_columns.joblib
├── notebooks/
│   ├── 01_EDA.ipynb (Contains all 8 factor plots)
│   └── 02_Modeling.ipynb
├── app.py (Former Salesapp_pro.py - Professional Dashboard)
├── requirements.txt
├── submission.csv
└── README.md
```

## 💻 Installation & Usage

```bash
# Clone repo
git clone https://github.com/yourusername/DSN-Mart-Sales-Prediction.git
cd DSN-Mart-Sales-Prediction

# Create env
pip install -r requirements.txt
# requirements: pandas, scikit-learn==1.3.0, joblib, streamlit, plotly

# Train (optional - models already saved)
python train.py

# Run Dashboard
streamlit run app.py

# App will open at http://localhost:8501
```

## 📤 Submission Generation


**Method 1 - Via Dashboard (Recommended):**
1. Open app → Go to "Batch Upload" tab
2. Upload `test.csv`
3. Click "Generate Predictions"
4. Click "DOWNLOAD submission.csv"

**Method 2 - Via Code:**
```python
import joblib
model = joblib.load('models/dsn_best_model.joblib')
cols = joblib.load('models/model_columns.joblib')
preds = model.predict(preprocess_df(test_df))
pd.DataFrame({'id': test_df['id'], 'total_sales': preds}).to_csv('submission.csv', index=False)
```

## 🔮 Future Improvements

- [ ] Add XGBoost Stacking ensemble (RF + LGBM) for LB 1080
- [ ] Incorporate external data: inflation rate, holiday calendar
- [ ] SHAP explainability for each prediction
- [ ] FastAPI backend for mobile app integration
- [ ] Time-series forecasting for seasonal trends

## 👨‍💻 Author

**ABDULROUF, Waliyullahi Abiodun** - Data Scientist | DSN AI Bootcamp 2026
- GitHub: https://github.com/Directorstat
- Email: abiosunajaniabdulrouf@gmail.com

---

### ⭐ If this helped you, give it a star! Built for DSN Hackathon - Let's win this!

