# 🏠 Mumbai House Price Prediction

An end-to-end Machine Learning web application for predicting residential property prices in Mumbai using property features, locality intelligence, explainable AI, anomaly detection, similar-property analysis, confidence estimation, interactive maps, deal analysis, and automated PDF valuation reports.

---

## 📌 Project Overview

Mumbai's real-estate market varies significantly across different localities, property types, sizes, and amenities.

This project builds a complete property intelligence system that goes beyond simple house-price prediction.

The system can:

* Predict property prices
* Analyze locality-level market information
* Calculate property intelligence scores
* Detect unusual/anomalous properties
* Find similar properties
* Analyze price per square foot
* Estimate prediction confidence
* Perform what-if analysis
* Search and filter properties
* Display properties on interactive maps
* Analyze whether a listing appears overpriced or underpriced
* Explain predictions using SHAP
* Generate PDF property valuation reports
* Maintain prediction history
* Provide a web-based dashboard

---

## ✨ Key Features

### 💰 1. House Price Prediction

Predicts the estimated market value of a Mumbai property using a trained Random Forest regression model.

Input features include:

* Location / Locality
* Property type
* Area
* Bedrooms
* Bathrooms
* Balcony
* Property age
* Total floors
* Other engineered property features

---

### 📍 2. Locality Intelligence

The system maintains locality-level information to provide additional context around the predicted price.

It can provide information such as:

* Average price
* Price per square foot
* Locality activity
* Market position
* Property availability

---

### 🧠 3. Property Intelligence Score

Each property receives an overall intelligence score based on multiple factors.

The system considers:

* Property quality
* Price efficiency
* Market activity
* Market position
* Risk indicators

The dashboard also provides component-level explanations.

---

### 🚨 4. Anomaly Detection

The system identifies properties whose characteristics or prices appear unusual compared with the broader property dataset.

This helps detect potentially:

* Unusually expensive properties
* Unusually cheap properties
* Unusual property configurations
* Potential pricing anomalies

---

### 🔎 5. Similar Property Finder

The application can find properties with characteristics similar to the selected property.

Similarity can be based on features such as:

* Location
* Area
* Bedrooms
* Bathrooms
* Property type
* Price
* Other property characteristics

---

### 📊 6. Price Per Square Foot Analysis

The project analyzes:

* Price per square foot
* Locality-level price differences
* Property-level price efficiency
* Relative pricing patterns

---

### 🎯 7. Prediction Confidence

Instead of showing only a single predicted price, the application can provide a confidence/range-based interpretation of the prediction.

This gives users additional context around the estimated property value.

---

### 🗺️ 8. Interactive Property Map

The application provides an interactive map-based view of Mumbai properties.

The mapping system can display:

* Property locations
* Localities
* Property information
* Price-related information
* Search results

---

### 🔍 9. Property Search & Filtering

Users can search and filter properties based on different criteria.

Examples:

* Locality
* Price
* Area
* Bedrooms
* Bathrooms
* Property type
* Other property attributes

---

### 💼 10. Deal Analyzer

Users can provide a property's listed/asking price and compare it with the model's estimated value.

The system can help identify whether the asking price is:

* Below the estimated model value
* Close to the estimated model value
* Above the estimated model value

The analysis is intended as a data-driven comparison rather than financial advice.

---

### 🤖 11. Explainable AI with SHAP

SHAP-based explainability is used to understand how individual features contribute to a prediction.

This helps answer questions such as:

* Which features increased the predicted price?
* Which features reduced the predicted price?
* Which property characteristics had the strongest influence?

The implementation uses a memory-conscious approach suitable for local systems.

---

### 📄 12. PDF Property Valuation Report

The application can generate a property valuation report containing information such as:

* Property details
* Predicted price
* Price per square foot
* Property intelligence
* Confidence information
* Deal analysis
* Additional property insights

---

### 📈 13. Prediction History

The project maintains prediction history so previous predictions can be analyzed and reviewed.

---

### 📊 14. Model Performance Dashboard

The project includes model-performance analysis covering regression evaluation and model comparison.

---

## 🧠 Machine Learning Pipeline

The overall workflow follows:

```text
Raw Data
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Property Intelligence
   ↓
Explainable AI
   ↓
Anomaly Detection
   ↓
Similar Property Analysis
   ↓
Price / Locality Analysis
   ↓
Prediction Confidence
   ↓
Flask Web Application
   ↓
Property Intelligence Dashboard
```

---

## 📚 Project Notebooks

The project contains the following notebooks:

| Notebook                                 | Purpose                             |
| ---------------------------------------- | ----------------------------------- |
| `01_data_understanding.ipynb`            | Understanding the dataset           |
| `02_data_cleaning.ipynb`                 | Cleaning and preprocessing          |
| `03_eda.ipynb`                           | Exploratory Data Analysis           |
| `04_feature_engineering.ipynb`           | Feature creation and transformation |
| `05_model_training.ipynb`                | Initial model training              |
| `06_model_tuning.ipynb`                  | Hyperparameter tuning               |
| `07_property_intelligence.ipynb`         | Property intelligence analysis      |
| `08_location_intelligence.ipynb`         | Locality/location analysis          |
| `09_explainable_ai.ipynb`                | SHAP-based explainability           |
| `10_anomaly_detection.ipynb`             | Anomaly detection                   |
| `11_similar_property_finder.ipynb`       | Similar property search             |
| `12_price_per_sqft_analysis.ipynb`       | Price/sqft analysis                 |
| `13_interactive_locality_map.ipynb`      | Interactive property mapping        |
| `14_property_search_filter.ipynb`        | Property search and filtering       |
| `15_prediction_confidence.ipynb`         | Prediction confidence analysis      |
| `16_model_performance_dashboard.ipynb`   | Model performance                   |
| `17_prediction_history.ipynb`            | Prediction history                  |
| `18_pdf_property_valuation_report.ipynb` | PDF report generation               |

---

## 🏗️ Project Structure

```text
mumbai-house-price-prediction/
│
├── app/
│   ├── app.py
│   │
│   ├── templates/
│   │   ├── dashboard.html
│   │   ├── deal_analyzer.html
│   │   ├── index.html
│   │   ├── property_finder.html
│   │   ├── property_map.html
│   │   └── result.html
│   │
│   ├── static/
│   └── outputs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── ...
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── ...
│   └── 18_pdf_property_valuation_report.ipynb
│
├── outputs/
│
├── .gitignore
├── README.md
└── requirements.txt
```

> Large trained model files are intentionally excluded from GitHub using `.gitignore` because of GitHub's individual file-size limitations.

---

## 🛠️ Technologies Used

### Programming

* Python 3.12

### Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Joblib
* SHAP

### Visualization

* Matplotlib
* Seaborn
* Folium

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Data & Reports

* OpenPyXL
* ReportLab

### Development Environment

* VS Code
* Jupyter Notebook
* Conda

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/tdalvi708-debug/mumbai-house-price-prediction.git
```

```bash
cd mumbai-house-price-prediction
```

---

### 2. Create the Conda environment

```bash
conda create -n mumbai-house python=3.12 -y
```

Activate it:

```bash
conda activate mumbai-house
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Navigate to the application directory:

```bash
cd app
```

Start Flask:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 🖥️ Application Modules

The web application provides multiple modules:

```text
Dashboard
   │
   ├── Price Prediction
   ├── Property Intelligence
   ├── Deal Analyzer
   ├── Similar Properties
   ├── Property Search
   ├── Interactive Property Map
   ├── Confidence Analysis
   ├── What-If Analysis
   ├── Anomaly Detection
   └── PDF Valuation Report
```

---

## 📈 Model

The primary prediction model used by the application is:

```text
RandomForestRegressor
```

The model was trained using engineered property and locality-related features.

The project also includes model tuning and performance evaluation notebooks.

---

## 🔐 Large Model Files

The trained model files are intentionally excluded from the Git repository because some files are hundreds of megabytes in size.

Examples include:

```text
models/random_forest_log_model.joblib
models/random_forest_model.joblib
models/best_tuned_model.joblib
outputs/similar_property_model.pkl
```

These files are listed in `.gitignore`.

For local execution, the required model artifacts must be available inside the appropriate `models/` and `outputs/` directories.

---

## 🎯 Project Objective

The main objective is to build a practical Mumbai real-estate intelligence platform rather than a simple machine-learning prediction script.

The project combines:

```text
Machine Learning
      +
Data Analysis
      +
Explainable AI
      +
Anomaly Detection
      +
Geospatial Analysis
      +
Property Similarity
      +
Market Intelligence
      +
Web Application
      +
Automated Reporting
```

---

## 🚀 Future Improvements

Possible future improvements include:

* Cloud deployment
* Real-time property listing integration
* More advanced geospatial features
* Deep learning models
* Automated model retraining
* Real-time market data
* User authentication
* Database integration
* Property recommendation engine
* Mobile application
* Automated model monitoring

---

## ⚠️ Disclaimer

This project is intended for educational, research, and demonstration purposes.

Predicted property values are machine-learning estimates and should not be treated as guaranteed market prices or financial advice.

Actual property prices may vary depending on factors that are not represented in the dataset.

---

## 👨‍💻 Author

**Tejas Dalvi**

Computer Engineering / Data Science

Lokmanya Tilak College of Engineering, Navi Mumbai

---

## ⭐ Acknowledgement

This project was developed as an end-to-end machine-learning and web-application project to explore real-world property price prediction and property intelligence.
