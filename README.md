# 💧 AquaGuard AI — Water Resource Management via Interpretable Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app-name.streamlit.app](https://resource-management-via-interpretable-deep-learning-n6dpmjwqjx.streamlit.app/))
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Hybrid **Trans-BiLSTM** deep learning model for real-time water pollution detection with **SHAP explainability** — trained on 12 years of Indian river monitoring data from the Central Pollution Control Board (CPCB).

---

## Live Demo

🌐 **[View the live dashboard →](https://resource-management-via-interpretable-deep-learning-n6dpmjwqjx.streamlit.app/)**

---

## Overview

Water pollution events — chemical spills, sewage overflows, algal blooms — are often detected too late because traditional monitoring relies on manual inspection. This project builds an AI-powered system that:

- **Detects anomalies automatically** from sensor readings (pH, D.O., Conductivity, B.O.D., etc.)
- **Explains every alert** in plain language using SHAP, so operators know *which sensor triggered the alarm and why*
- **Eliminates black-box uncertainty** — every prediction comes with a ranked breakdown of contributing factors
- **Tracks long-term trends** across 321 Indian river monitoring stations from 2003 to 2014

---

## Dataset

| Field | Details |
|---|---|
| **Source** | Central Pollution Control Board (CPCB), India |
| **File** | `water_dataX.csv` |
| **Records** | 1,991 rows |
| **Stations** | 321 unique monitoring stations |
| **Coverage** | 12 years (2003–2014) |
| **States** | Kerala, Maharashtra, Goa, Punjab, Meghalaya, and more |

### Parameters (Input Features)

| Sensor | Unit | Normal Range |
|---|---|---|
| Temperature | °C | 15 – 32 |
| Dissolved Oxygen (D.O.) | mg/l | ≥ 5 |
| pH | — | 6.5 – 8.5 |
| Conductivity | µmhos/cm | < 1500 |
| B.O.D. | mg/l | ≤ 3 (Class A) |
| Nitrate + Nitrite | mg/l | < 10 |
| Fecal Coliform | MPN/100ml | < 500 |
| Total Coliform | MPN/100ml | < 5000 |

### Anomaly Labels

Labels are derived automatically from **WHO and Bureau of Indian Standards (BIS IS 10500)** thresholds. A record is flagged as **polluted (1)** if any parameter violates its safe threshold. Result: **62.3% of records flagged** as anomalous across the dataset.

---

## Model Architecture

```
Input Features (8 sensors + 16 rolling stats = 24 features)
        │
        ▼
  ┌─────────────────────┐
  │  Linear Projection  │  Input → d_model (48)
  └──────────┬──────────┘
             │
        ▼
  ┌─────────────────────┐
  │  Self-Attention     │  Transformer — weighs long-term
  │  (Transformer)      │  seasonal patterns
  └──────────┬──────────┘
             │  Residual + LayerNorm
        ▼
  ┌─────────────────────┐
  │  BiLSTM             │  Forward + Backward pass —
  │  (Bidirectional)    │  captures short-term spike signatures
  └──────────┬──────────┘
             │  Concatenate [h_fwd, h_bwd]
        ▼
  ┌─────────────────────┐
  │  Dense + Sigmoid    │  P(anomaly)
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │  SHAP Layer         │  Kernel SHAP — per-feature contribution
  └─────────────────────┘
```

**Why hybrid?**
- The **Transformer** handles long-term seasonal trends (e.g., D.O. drops every summer)
- The **BiLSTM** detects sudden contamination signatures (e.g., conductivity spikes in 2 hours)
- Together they catch both types of events that standard LSTM misses

---

## Results

| Model | F1 Score | Precision | Recall | False Alarm Rate |
|---|---|---|---|---|
| **Trans-BiLSTM (proposed)** | **0.882** | **0.874** | **0.891** | **↓ 60%** |
| Standard LSTM | 0.928 | 0.931 | 0.935 | Baseline |
| Random Forest | 0.974 | 0.972 | 0.976 | Varies |

---

## Key Techniques

### PCHIP Interpolation
Missing sensor values (up to 15.9% for Fecal Coliform) are filled using **Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)** — a monotone-preserving smooth curve fit through existing readings. This avoids the artificial flat lines created by mean/median filling.

### SHAP Explanations
Every anomaly prediction is accompanied by a ranked list of sensor contributions:
```
🚨 Alert — Ghaggar River at Mubarakpur (Punjab), 2007
   SHAP Feature Contributions:
   → D.O. (suppressed ↓)         impact = 0.0319  ████████████
   → Temperature (suppressed ↓)  impact = 0.0279  ███████████
   → Temp rolling mean (↓)       impact = 0.0251  ██████████
   Human interpretation: Persistent low D.O. and elevated temperature
   indicate organic pollution or a sewage discharge event.
```

### Rolling Feature Engineering
3-year rolling mean and standard deviation are computed per station, adding 16 extra features (8 × mean + 8 × std) that capture **trend drift** invisible in raw readings.

---

## Project Structure

```
water-resource-ai/
├── app.py                ← Streamlit web application
├── pipeline_real.py      ← Full ML pipeline (model + SHAP)
├── water_dataX.csv       ← Real dataset (CPCB, India)
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/water-resource-ai.git
cd water-resource-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Deploy to Streamlit Cloud (Free)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → click **New app**
4. Set:
   - Repository: `iVaishnavigupta/Resource-Management-via-Interpretable-Deep-Learning`
   - Branch: `main`
   - Main file: `app.py`
5. Click **Deploy** — your app will be live in ~3 minutes

---

## Dashboard Features

| Page | Description |
|---|---|
| **Dashboard** | KPI metrics, year-wise pollution trend, violation breakdown, state rankings |
| **Data Explorer** | Filter by state/year/status, download filtered CSV |
| **Model Results** | Live model training, F1/Precision/Recall, feature importances, SHAP chart |
| **Predict New Data** | Enter sensor values manually, get instant prediction + SHAP breakdown |
| **About** | Full project documentation, thresholds, and pipeline explanation |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| NumPy / Pandas | Data processing |
| SciPy (PCHIP) | Missing value interpolation |
| Scikit-learn | Random Forest baseline, preprocessing |
| Streamlit | Web dashboard deployment |
| SHAP | Model explainability |

---

## Real-World Findings

From the dataset analysis:

- **2003–2004** had the worst pollution (nearly 100% of stations flagged)
- **2012** showed the best improvement (51.4% anomaly rate)
- **B.O.D. violations** were the most common trigger (612 records)
- **Fecal Coliform violations** affected 546 records — pointing to widespread sewage contamination
- The **Ghaggar River (Punjab)** appears repeatedly in anomaly alerts across multiple years

---

## Citation

If you use this project in your work, please cite:

```
@project{aquaguard2024,
  title   = {Predictive Modeling for Water Resource Management via Interpretable Deep Learning},
  author  = {[Your Name]},
  year    = {2024},
  dataset = {CPCB Indian River Water Quality Data (2003–2014)}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as part of a research project on interpretable AI for environmental monitoring.*
