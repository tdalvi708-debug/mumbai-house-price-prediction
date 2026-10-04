from flask import Flask, render_template, request, jsonify, session, send_file

import os
import json
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap

from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


# ============================================================
# FLASK APP
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "templates")
)
app.secret_key = "mumbai-house-price-prediction-secret-key"


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "render_optimized_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model.joblib")
LOCALITY_DATA_PATH = os.path.join(PROCESSED_DIR, "locality_intelligence.csv")
PROPERTY_DATA_PATH = os.path.join(PROCESSED_DIR, "property_intelligence.csv")
ANOMALY_PROPERTY_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "property_intelligence_with_anomalies.csv"
)


# ============================================================
# STARTUP PATH DEBUGGING
# ============================================================

print("\n" + "=" * 70)
print("MUMBAI HOUSE PRICE PREDICTION SYSTEM")
print("=" * 70)
print("APP_DIR       :", APP_DIR)
print("BASE_DIR      :", BASE_DIR)
print("MODEL_DIR     :", MODEL_DIR)
print("PROCESSED_DIR :", PROCESSED_DIR)
print("OUTPUT_DIR    :", OUTPUT_DIR)
print("MODEL_PATH    :", MODEL_PATH)
print("PREPROCESSOR  :", PREPROCESSOR_PATH)
print("=" * 70)


# ============================================================
# GLOBAL MODEL/DATA OBJECTS
# ============================================================

model = None
preprocessor = None
anomaly_model = None
shap_explainer = None
model_error = None
preprocessor_error = None

locality_data = pd.DataFrame()
property_data = pd.DataFrame()

similarity_scaler = None
similarity_model = None
similarity_data = None
similarity_columns = [
    "area",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "age",
    "total_floors",
    "latitude",
    "longitude",
]


# ============================================================
# FILE LOADER
# ============================================================

def load_joblib_artifact(path, label, required=False):
    """Load a joblib file and print a useful diagnostic on failure."""
    if not os.path.exists(path):
        message = f"{label} file not found: {path}"
        print("WARNING:", message)
        if required:
            return None, message
        return None, message

    try:
        obj = joblib.load(path)
        print(f"{label} loaded successfully: {path}")
        return obj, None
    except Exception as exc:
        message = (
            f"Could not load {label}. Path: {path}. "
            f"Error: {type(exc).__name__}: {exc}"
        )
        print("\n" + "!" * 70)
        print(message)
        traceback.print_exc()
        print("!" * 70 + "\n")
        return None, message


# ============================================================
# HUGGING FACE MODEL FALLBACK
# ============================================================

HF_REPO_ID = "tdalvi/mumbai-house-price-model"
HF_MODEL_FILENAME = "render_optimized_model.joblib"


def ensure_model_available():
    """
    Make the production model available on Render.

    The optimized model is intentionally not stored in GitHub because it is
    a binary ML artifact. Render downloads it from the public Hugging Face
    repository when the file is not present locally.
    """
    if os.path.exists(MODEL_PATH):
        print("Render model found locally:", MODEL_PATH)
        return True

    print("Render model not found locally.")
    print(
        "Downloading production model from Hugging Face:",
        f"{HF_REPO_ID}/{HF_MODEL_FILENAME}"
    )

    try:
        from huggingface_hub import hf_hub_download

        os.makedirs(MODEL_DIR, exist_ok=True)

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            local_dir=MODEL_DIR,
        )

        # hf_hub_download normally places the file in MODEL_DIR when
        # local_dir is supplied. Keep this copy operation as a safety net
        # for environments where the returned path points elsewhere.
        if os.path.abspath(downloaded_path) != os.path.abspath(MODEL_PATH):
            import shutil
            shutil.copy2(downloaded_path, MODEL_PATH)

        if os.path.exists(MODEL_PATH):
            print("Production model downloaded successfully:", MODEL_PATH)
            return True

        print("ERROR: Hugging Face download completed but model file is missing.")
        return False

    except Exception as exc:
        print("\n" + "!" * 70)
        print("HUGGING FACE MODEL DOWNLOAD FAILED")
        print(repr(exc))
        traceback.print_exc()
        print("!" * 70 + "\n")
        return False


# ============================================================
# LOAD MAIN MODEL
# ============================================================

ensure_model_available()

model, model_error = load_joblib_artifact(
    MODEL_PATH,
    "ML model",
    required=True
)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

preprocessor, preprocessor_error = load_joblib_artifact(
    PREPROCESSOR_PATH,
    "Preprocessor",
    required=True
)


# ============================================================
# LOAD LOCALITY DATA
# ============================================================

if os.path.exists(LOCALITY_DATA_PATH):
    try:
        locality_data = pd.read_csv(LOCALITY_DATA_PATH)
        print("Locality intelligence loaded:", locality_data.shape)
    except Exception as exc:
        print("WARNING: locality_intelligence.csv could not be read:", exc)
else:
    print("WARNING: locality_intelligence.csv not found:", LOCALITY_DATA_PATH)


# ============================================================
# LOAD PROPERTY DATA
# ============================================================

property_path_to_use = PROPERTY_DATA_PATH
if not os.path.exists(property_path_to_use) and os.path.exists(ANOMALY_PROPERTY_DATA_PATH):
    property_path_to_use = ANOMALY_PROPERTY_DATA_PATH

if os.path.exists(property_path_to_use):
    try:
        property_data = pd.read_csv(property_path_to_use)
        print("Property intelligence loaded:", property_data.shape)
        print("Property columns:", list(property_data.columns))
    except Exception as exc:
        print("WARNING: property intelligence could not be read:", exc)
else:
    print("WARNING: property_intelligence.csv not found:", PROPERTY_DATA_PATH)


# ============================================================
# LOAD ANOMALY MODEL
# ============================================================

anomaly_model, _ = load_joblib_artifact(
    ANOMALY_MODEL_PATH,
    "Anomaly detection model",
    required=False
)


# ============================================================
# SIMILAR PROPERTY SYSTEM
# ============================================================

if not property_data.empty:
    available_columns = [
        col for col in similarity_columns
        if col in property_data.columns
    ]

    if available_columns:
        try:
            similarity_data = property_data[available_columns].copy()
            similarity_data = similarity_data.replace([np.inf, -np.inf], np.nan)
            similarity_data = similarity_data.apply(
                pd.to_numeric,
                errors="coerce"
            )
            similarity_data = similarity_data.fillna(
                similarity_data.median(numeric_only=True)
            ).fillna(0)

            similarity_scaler = StandardScaler()
            similarity_matrix = similarity_scaler.fit_transform(similarity_data)

            similarity_model = NearestNeighbors(
                n_neighbors=min(6, len(similarity_data)),
                metric="euclidean"
            )
            similarity_model.fit(similarity_matrix)
            similarity_columns = available_columns
            print("Similar property system ready.")
        except Exception as exc:
            print("WARNING: Similar property system failed:", exc)
            similarity_scaler = None
            similarity_model = None
            similarity_data = None


# ============================================================
# SHAP EXPLAINER
# ============================================================

if model is not None:
    try:
        shap_explainer = shap.TreeExplainer(model)
        print("SHAP TreeExplainer loaded successfully.")
    except Exception as exc:
        shap_explainer = None
        print("WARNING: SHAP explainer could not be created:", repr(exc))


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def clean_string(value, default=""):
    if value is None:
        return default
    return str(value).strip()


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def format_currency(value):
    value = safe_float(value, 0)
    return f"₹{value:,.0f}"


def format_lakh_crore(value):
    value = safe_float(value, 0)
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    return format_currency(value)


# ============================================================
# LOCALITY NORMALIZATION
# ============================================================

def normalize_locality(value):
    value = clean_string(value).lower()
    value = " ".join(value.split())

    replacements = {
        "dombivali": "dombivli",
        "dombivli": "dombivli",
        "nala sopara": "nalasopara",
        "nala-sopara": "nalasopara",
        "nalasopara": "nalasopara",
        "ville parle": "vile parle",
        "vile parle": "vile parle",
        "vikroli": "vikhroli",
        "vikhroli": "vikhroli",
        "taloje": "taloja",
        "taloja": "taloja",
    }
    return replacements.get(value, value)


def find_locality_row(locality):
    target = normalize_locality(locality)
    if not locality_data.empty and "locality" in locality_data.columns:
        temp = locality_data.copy()
        temp["locality_clean"] = temp["locality"].apply(normalize_locality)

        exact = temp[temp["locality_clean"] == target]
        if not exact.empty:
            return exact.iloc[0]

        # Fuzzy fallback
        best_row = None
        best_score = 0
        for _, row in temp.iterrows():
            candidate = normalize_locality(row.get("locality", ""))
            score = SequenceMatcher(None, target, candidate).ratio()
            if target in candidate or candidate in target:
                score += 0.15
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None and best_score >= 0.55:
            return best_row

    return None


# ============================================================
# AGE GROUP
# ============================================================

def get_age_group(age):
    age = safe_int(age, 0)
    if age <= 5:
        return "New"
    if age <= 15:
        return "Moderately New"
    if age <= 30:
        return "Old"
    return "Very Old"


# ============================================================
# LOCALITY COORDINATES
# ============================================================

def get_locality_coordinates(locality):
    default_latitude = 19.0760
    default_longitude = 72.8777

    row = find_locality_row(locality)
    if row is not None:
        lat = safe_float(row.get("average_latitude"))
        lon = safe_float(row.get("average_longitude"))
        if lat is not None and lon is not None:
            return lat, lon

    matched = _locality_property_matches(locality)
    if not matched.empty:
        lat = pd.to_numeric(matched.get("latitude"), errors="coerce").median()
        lon = pd.to_numeric(matched.get("longitude"), errors="coerce").median()
        if pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon)

    if not property_data.empty:
        lat = pd.to_numeric(property_data.get("latitude"), errors="coerce").median()
        lon = pd.to_numeric(property_data.get("longitude"), errors="coerce").median()
        if pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon)

    return default_latitude, default_longitude


# ============================================================
# LOCALITY STATISTICS
# ============================================================

def _locality_property_matches(locality):
    """Return property rows matching a locality using the same normalization everywhere."""
    if property_data.empty or "locality" not in property_data.columns:
        return pd.DataFrame()

    target = normalize_locality(locality)
    temp = property_data.copy()
    temp["locality_clean"] = temp["locality"].fillna("").apply(normalize_locality)

    matched = temp[temp["locality_clean"] == target]

    if matched.empty and target:
        matched = temp[
            temp["locality_clean"].str.contains(target, regex=False, na=False)
        ]

    if matched.empty and target:
        # Fuzzy fallback for small spelling differences.
        best_name = None
        best_score = 0.0
        for candidate in temp["locality_clean"].dropna().unique():
            score = SequenceMatcher(None, target, str(candidate)).ratio()
            if target in str(candidate) or str(candidate) in target:
                score += 0.15
            if score > best_score:
                best_score = score
                best_name = candidate

        if best_name is not None and best_score >= 0.55:
            matched = temp[temp["locality_clean"] == best_name]

    return matched


def _build_locality_stats_from_properties(locality):
    """Build reliable locality statistics directly from the property dataset."""
    matched = _locality_property_matches(locality)
    if matched.empty:
        return None

    price = pd.to_numeric(matched.get("price"), errors="coerce")
    ppsf = pd.to_numeric(matched.get("price_per_sqft"), errors="coerce")

    if ppsf.isna().all() and "predicted_price" in matched.columns:
        area_series = pd.to_numeric(matched.get("area"), errors="coerce")
        predicted = pd.to_numeric(matched.get("predicted_price"), errors="coerce")
        ppsf = predicted / area_series.replace(0, np.nan)

    lat = pd.to_numeric(matched.get("latitude"), errors="coerce") if "latitude" in matched.columns else pd.Series(dtype=float)
    lon = pd.to_numeric(matched.get("longitude"), errors="coerce") if "longitude" in matched.columns else pd.Series(dtype=float)

    first_locality = clean_string(
        matched.iloc[0].get("locality"),
        clean_string(locality, "Unknown")
    )

    return make_json_safe({
        "locality": first_locality,
        "listing_count": int(len(matched)),
        "average_price": safe_float(price.mean()),
        "median_price": safe_float(price.median()),
        "average_price_per_sqft": safe_float(ppsf.mean()),
        "average_predicted_price": safe_float(
            pd.to_numeric(matched.get("predicted_price"), errors="coerce").mean()
        ) if "predicted_price" in matched.columns else None,
        "average_latitude": safe_float(lat.mean()) if not lat.empty else None,
        "average_longitude": safe_float(lon.mean()) if not lon.empty else None,
    })


def get_locality_statistics(locality):
    """
    Get complete locality statistics.

    Important: locality_intelligence.csv is used as a supplementary source,
    but property_intelligence.csv remains the authoritative fallback. This
    prevents a partial locality row from causing 'market data unavailable'.
    """
    property_stats = _build_locality_stats_from_properties(locality)
    locality_row = find_locality_row(locality)

    merged = {}
    if locality_row is not None:
        merged.update(make_json_safe(locality_row.to_dict()))

    if property_stats:
        # Property-level calculations fill missing or unusable values.
        for key, value in property_stats.items():
            if value is not None:
                merged[key] = value

    if not merged:
        return None

    merged["locality"] = clean_string(
        merged.get("locality"),
        clean_string(locality, "Unknown")
    )
    return make_json_safe(merged)


# ============================================================
# MODEL FEATURE PREPARATION
# ============================================================

def build_model_input(
    area,
    locality,
    bedroom_num,
    bathroom_num,
    balcony_num,
    age,
    total_floors,
    property_type,
    furnished,
    latitude,
    longitude,
):
    if bedroom_num > 0:
        area_per_bedroom = area / bedroom_num
        bathroom_per_bedroom = bathroom_num / bedroom_num
    else:
        area_per_bedroom = area
        bathroom_per_bedroom = bathroom_num

    input_data = {
        "area": area,
        "locality": locality,
        "bedroom_num": bedroom_num,
        "bathroom_num": bathroom_num,
        "balcony_num": balcony_num,
        "age": age,
        "total_floors": total_floors,
        "latitude": latitude,
        "longitude": longitude,
        "property_type": property_type,
        "furnished": furnished,
        "area_per_bedroom": area_per_bedroom,
        "bathroom_per_bedroom": bathroom_per_bedroom,
        "age_group": get_age_group(age),
        "city": "Mumbai",
    }

    input_df = pd.DataFrame([input_data])

    if hasattr(preprocessor, "feature_names_in_"):
        expected_features = list(preprocessor.feature_names_in_)
        final_input = pd.DataFrame(index=[0])

        for feature in expected_features:
            if feature in input_df.columns:
                final_input[feature] = input_df[feature].values
            else:
                final_input[feature] = np.nan
    else:
        final_input = input_df.copy()

    return final_input


# ============================================================
# SHAP
# ============================================================

def generate_shap_explanation(processed_input):
    empty = ([], [], [], None, None)
    if shap_explainer is None:
        return empty

    try:
        if hasattr(processed_input, "toarray"):
            shap_input = processed_input.toarray()
        else:
            shap_input = np.asarray(processed_input)

        shap_input = np.asarray(shap_input, dtype=float)
        if shap_input.ndim == 1:
            shap_input = shap_input.reshape(1, -1)

        shap_values = shap_explainer.shap_values(shap_input)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        elif shap_values.ndim > 2:
            shap_values = shap_values.reshape(-1)

        shap_values = np.asarray(shap_values, dtype=float)

        try:
            feature_names = [
                str(x) for x in preprocessor.get_feature_names_out()
            ]
        except Exception:
            feature_names = [f"Feature_{i}" for i in range(len(shap_values))]

        n = min(len(feature_names), len(shap_values))
        if n == 0:
            return empty

        shap_df = pd.DataFrame({
            "feature": feature_names[:n],
            "shap_value": shap_values[:n],
        })
        shap_df["absolute_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values(
            "absolute_shap", ascending=False
        ).reset_index(drop=True)

        top = shap_df.head(10).to_dict("records")
        positive = (
            shap_df[shap_df["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(5)
            .to_dict("records")
        )
        negative = (
            shap_df[shap_df["shap_value"] < 0]
            .sort_values("shap_value", ascending=True)
            .head(5)
            .to_dict("records")
        )

        expected_value = shap_explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            arr = np.asarray(expected_value).reshape(-1)
            expected_value = float(arr[0]) if len(arr) else None
        else:
            expected_value = safe_float(expected_value)

        shap_prediction = None
        if expected_value is not None:
            shap_prediction = safe_float(
                np.sum(shap_values) + expected_value
            )

        return (
            make_json_safe(positive),
            make_json_safe(negative),
            make_json_safe(top),
            shap_prediction,
            expected_value,
        )

    except Exception as exc:
        print("SHAP ERROR:", repr(exc))
        return empty


# ============================================================
# CONFIDENCE RANGE
# ============================================================

def get_prediction_confidence(processed_input, predicted_price):
    result = {
        "lower": predicted_price,
        "upper": predicted_price,
        "spread": 0,
        "confidence_percent": None,
        "method": "Model estimate",
    }

    if model is None or not hasattr(model, "estimators_"):
        return result

    try:
        tree_predictions = []
        for estimator in model.estimators_:
            value = estimator.predict(processed_input)[0]
            if np.isfinite(value):
                tree_predictions.append(float(value))

        if len(tree_predictions) >= 5:
            arr = np.asarray(tree_predictions)
            lower = float(np.percentile(arr, 10))
            upper = float(np.percentile(arr, 90))
            result["lower"] = max(0, lower)
            result["upper"] = max(result["lower"], upper)
            result["spread"] = result["upper"] - result["lower"]
            result["confidence_percent"] = round(
                max(0, min(100, 100 - ((result["spread"] / max(predicted_price, 1)) * 100))),
                1
            )
            result["method"] = "Random Forest tree prediction range"
    except Exception as exc:
        print("Confidence error:", repr(exc))

    return make_json_safe(result)


# ============================================================
# MARKET POSITION / DEAL ANALYSIS
# ============================================================

def calculate_market_position(predicted_price, locality_average_price):
    if locality_average_price is None or locality_average_price <= 0:
        return {
            "label": "Locality benchmark unavailable",
            "difference_percent": None,
            "difference": None,
        }

    difference = predicted_price - locality_average_price
    percentage = (difference / locality_average_price) * 100

    if percentage > 10:
        label = "Above locality average"
    elif percentage < -10:
        label = "Below locality average"
    else:
        label = "Near locality average"

    return {
        "label": label,
        "difference_percent": safe_float(percentage),
        "difference": safe_float(difference),
    }


def calculate_deal_analysis(predicted_price, listed_price, locality_average_price):
    analysis = {
        "listed_price": safe_float(listed_price),
        "difference": None,
        "difference_percent": None,
        "status": "No listing price supplied",
        "message": "Add a listed price to compare the asking price with the model valuation.",
    }

    if listed_price is not None and listed_price > 0:
        difference = predicted_price - listed_price
        percentage = (difference / listed_price) * 100
        analysis["difference"] = safe_float(difference)
        analysis["difference_percent"] = safe_float(percentage)

        if percentage >= 10:
            analysis["status"] = "Listed below model estimate"
            analysis["message"] = "The asking price is below the model estimate."
        elif percentage <= -10:
            analysis["status"] = "Listed above model estimate"
            analysis["message"] = "The asking price is above the model estimate."
        else:
            analysis["status"] = "Close to model estimate"
            analysis["message"] = "The asking price is relatively close to the model estimate."

    return make_json_safe(analysis)


# ============================================================
# PROPERTY INTELLIGENCE SCORE
# ============================================================

def calculate_intelligence_score(
    predicted_price,
    locality_average_price,
    confidence,
    anomaly_status,
    similar_properties,
    locality_listing_count=0,
    locality_average_price_per_sqft=None,
    predicted_price_per_sqft=None,
    area=None,
    bedroom_num=None,
    bathroom_num=None,
    balcony_num=None,
    age=None,
    total_floors=None,
):
    """Calculate the Property Intelligence Score and explain every component."""

    def clamp(value, low=0.0, high=100.0):
        try:
            return float(max(low, min(high, float(value))))
        except (TypeError, ValueError):
            return low

    def num(value, default=0.0):
        try:
            value = float(value)
            return default if not np.isfinite(value) else value
        except (TypeError, ValueError):
            return default

    area_value = num(area)
    bedrooms = num(bedroom_num)
    bathrooms = num(bathroom_num)
    balconies = num(balcony_num)
    age_value = num(age)
    floors = num(total_floors)

    # 1) PROPERTY QUALITY -------------------------------------------------
    area_score = clamp((area_value / 2500.0) * 100.0) if area_value > 0 else 50.0
    room_score = clamp((bedrooms / 4.0) * 65.0 + (bathrooms / 3.0) * 25.0)
    balcony_score = clamp((balconies / 2.0) * 100.0) if balconies > 0 else 45.0
    age_score = clamp(100.0 - age_value * 2.2) if age_value > 0 else 65.0
    floor_score = clamp(45.0 + min(floors, 20.0) * 2.5) if floors > 0 else 55.0

    property_quality = clamp(
        area_score * 0.25
        + room_score * 0.30
        + balcony_score * 0.10
        + age_score * 0.25
        + floor_score * 0.10
    )

    quality_good = []
    quality_limits = []
    if area_value >= 1000:
        quality_good.append(f"spacious area of {area_value:,.0f} sq.ft")
    elif area_value > 0:
        quality_limits.append(f"area is {area_value:,.0f} sq.ft")
    if bedrooms >= 3:
        quality_good.append(f"{int(bedrooms)} bedrooms")
    elif bedrooms >= 2:
        quality_good.append(f"{int(bedrooms)} bedrooms")
    else:
        quality_limits.append(f"only {int(bedrooms)} bedroom")
    if bathrooms >= 2:
        quality_good.append(f"{int(bathrooms)} bathrooms")
    elif bathrooms < 2:
        quality_limits.append(f"{int(bathrooms)} bathroom")
    if balconies >= 1:
        quality_good.append(f"{int(balconies)} balcony/balconies")
    else:
        quality_limits.append("no balcony supplied")
    if age_value <= 5:
        quality_good.append(f"relatively new ({age_value:.0f} years old)")
    elif age_value >= 15:
        quality_limits.append(f"older property ({age_value:.0f} years old)")
    if floors >= 5:
        quality_good.append(f"{int(floors)} total floors")

    quality_summary = (
        "Strengths: " + ", ".join(quality_good[:3]) + ". " if quality_good else "Strengths: limited positive signals available. "
    ) + (
        "Watch: " + ", ".join(quality_limits[:2]) + "." if quality_limits else "No major structural limitation is visible from the supplied inputs."
    )

    # 2) PRICE EFFICIENCY -------------------------------------------------
    efficiency_difference = None
    locality_ppsf = num(locality_average_price_per_sqft)
    predicted_ppsf = num(predicted_price_per_sqft)
    if locality_ppsf > 0 and predicted_ppsf > 0:
        efficiency_difference = abs(predicted_ppsf - locality_ppsf) / locality_ppsf * 100.0
    else:
        locality_price = num(locality_average_price)
        predicted = num(predicted_price)
        if locality_price > 0 and predicted > 0:
            efficiency_difference = abs(predicted - locality_price) / locality_price * 100.0

    price_efficiency = 50.0 if efficiency_difference is None else clamp(100.0 - efficiency_difference * 1.5)

    if efficiency_difference is None:
        price_summary = "Locality price benchmark is unavailable, so price efficiency is based on limited market information."
    elif predicted_ppsf < locality_ppsf:
        price_summary = f"Model value is about {efficiency_difference:.1f}% below the locality average price/sq.ft, which improves price efficiency."
    elif predicted_ppsf > locality_ppsf:
        price_summary = f"Model value is about {efficiency_difference:.1f}% above the locality average price/sq.ft, so pricing is less efficient relative to the local benchmark."
    else:
        price_summary = "Model value is very close to the locality price/sq.ft benchmark."

    # 3) MARKET ACTIVITY -------------------------------------------------
    listing_count = max(0.0, num(locality_listing_count))
    market_activity = clamp(25.0 + np.log1p(listing_count) * 13.0)
    if listing_count >= 1000:
        market_summary = f"Strong market coverage: {int(listing_count):,} analysed listings are available for this locality."
    elif listing_count >= 100:
        market_summary = f"Moderate market coverage: {int(listing_count):,} analysed listings provide a useful local sample."
    elif listing_count > 0:
        market_summary = f"Limited market coverage: only {int(listing_count):,} analysed listings are available for this locality."
    else:
        market_summary = "No locality listing count was available, so market activity has limited evidence."

    # 4) MARKET POSITION --------------------------------------------------
    locality_price = num(locality_average_price)
    predicted = num(predicted_price)
    if locality_price > 0 and predicted > 0:
        position_difference = (predicted - locality_price) / locality_price * 100.0
        market_position = clamp(100.0 - abs(position_difference) * 2.0)
        if position_difference > 10:
            market_summary_position = f"The model value is {position_difference:.1f}% above the locality average."
        elif position_difference < -10:
            market_summary_position = f"The model value is {abs(position_difference):.1f}% below the locality average."
        else:
            market_summary_position = f"The model value is only {abs(position_difference):.1f}% away from the locality average."
    else:
        position_difference = None
        market_position = 50.0
        market_summary_position = "Locality average price is unavailable, so market position cannot be strongly established."

    # 5) RISK INDICATOR ---------------------------------------------------
    confidence_percent = num(confidence.get("confidence_percent"), 50.0) if isinstance(confidence, dict) else 50.0
    confidence_component = clamp(confidence_percent)
    if anomaly_status == "Potentially Unusual":
        anomaly_component = 25.0
        anomaly_text = "the anomaly detector found an unusual pattern"
    elif anomaly_status == "Within Normal Range":
        anomaly_component = 90.0
        anomaly_text = "the anomaly detector found the property within the normal range"
    else:
        anomaly_component = 55.0
        anomaly_text = "anomaly evidence is limited"
    similarity_component = 85.0 if similar_properties else 45.0

    risk_indicator = clamp(
        confidence_component * 0.60
        + anomaly_component * 0.30
        + similarity_component * 0.10
    )
    risk_summary = (
        f"Model confidence is {confidence_percent:.1f}%, {anomaly_text}, "
        + ("and comparable properties were found." if similar_properties else "and no comparable properties were found.")
    )

    overall = clamp(
        property_quality * 0.20
        + price_efficiency * 0.20
        + market_activity * 0.20
        + market_position * 0.20
        + risk_indicator * 0.20
    )
    overall = round(overall, 1)

    if overall >= 80:
        label = "High intelligence coverage"
    elif overall >= 60:
        label = "Good intelligence coverage"
    else:
        label = "Limited intelligence coverage"

    explanations = {
        "property_quality": quality_summary,
        "price_efficiency": price_summary,
        "market_activity": market_summary,
        "market_position": market_summary_position,
        "risk_indicator": risk_summary,
    }

    return {
        "score": overall,
        "overall_score": overall,
        "percentage": overall,
        "score_percentage": overall,
        "label": label,
        "status": label,
        "interpretation": label,
        "property_quality": round(property_quality, 1),
        "property_quality_percentage": round(property_quality, 1),
        "price_efficiency": round(price_efficiency, 1),
        "price_efficiency_percentage": round(price_efficiency, 1),
        "market_activity": round(market_activity, 1),
        "market_activity_percentage": round(market_activity, 1),
        "market_position": round(market_position, 1),
        "market_position_percentage": round(market_position, 1),
        "risk_indicator": round(risk_indicator, 1),
        "risk_percentage": round(risk_indicator, 1),
        "risk_score": round(risk_indicator, 1),
        "explanations": explanations,
        "property_quality_explanation": quality_summary,
        "price_efficiency_explanation": price_summary,
        "market_activity_explanation": market_summary,
        "market_position_explanation": market_summary_position,
        "risk_indicator_explanation": risk_summary,
    }


# ============================================================
# SIMILAR PROPERTIES
# ============================================================

def find_similar_properties(
    area,
    bedroom_num,
    bathroom_num,
    balcony_num,
    age,
    total_floors,
    latitude,
    longitude,
):
    results = []

    if similarity_model is None or similarity_scaler is None or similarity_data is None:
        return results

    try:
        values = []
        for column in similarity_columns:
            mapping = {
                "area": area,
                "bedroom_num": bedroom_num,
                "bathroom_num": bathroom_num,
                "balcony_num": balcony_num,
                "age": age,
                "total_floors": total_floors,
                "latitude": latitude,
                "longitude": longitude,
            }
            values.append(mapping.get(column, 0))

        X = np.asarray(values, dtype=float).reshape(1, -1)
        X = similarity_scaler.transform(X)
        distances, indices = similarity_model.kneighbors(X)

        for distance, index in zip(distances[0], indices[0]):
            index = int(index)
            if index >= len(property_data):
                continue

            row = property_data.iloc[index]
            results.append({
                "locality": clean_string(row.get("locality"), "Unknown"),
                "price": safe_float(row.get("price")),
                "area": safe_float(row.get("area")),
                "bedroom_num": safe_int(row.get("bedroom_num"), 0),
                "bathroom_num": safe_int(row.get("bathroom_num"), 0),
                "property_type": clean_string(row.get("property_type"), "Unknown"),
                "similarity_distance": round(float(distance), 4),
            })
    except Exception as exc:
        print("Similar property error:", repr(exc))

    return make_json_safe(results)


# ============================================================
# WHAT-IF AREA ANALYSIS
# ============================================================

def generate_what_if_analysis(
    area,
    locality,
    bedroom_num,
    bathroom_num,
    balcony_num,
    age,
    total_floors,
    property_type,
    furnished,
    latitude,
    longitude,
):
    results = []

    if model is None or preprocessor is None:
        return results

    candidate_areas = sorted(set([
        max(100, round(area * 0.75)),
        max(100, round(area * 0.90)),
        round(area),
        round(area * 1.10),
        round(area * 1.25),
    ]))

    for candidate_area in candidate_areas:
        try:
            frame = build_model_input(
                candidate_area,
                locality,
                bedroom_num,
                bathroom_num,
                balcony_num,
                age,
                total_floors,
                property_type,
                furnished,
                latitude,
                longitude,
            )
            transformed = preprocessor.transform(frame)
            price = max(0, float(model.predict(transformed)[0]))
            results.append({
                "area": int(candidate_area),
                "predicted_price": safe_float(price),
                "price_per_sqft": safe_float(price / candidate_area),
            })
        except Exception as exc:
            print("What-if error:", repr(exc))

    return make_json_safe(results)


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomaly(
    area,
    predicted_price,
    bedroom_num,
    bathroom_num,
    balcony_num,
    age,
    total_floors,
    latitude,
    longitude,
    predicted_price_per_sqft,
):
    result = {
        "status": "Not Available",
        "score": None,
        "is_anomaly": False,
    }

    if anomaly_model is None:
        return result

    try:
        features = np.array([
            area,
            predicted_price,
            bedroom_num,
            bathroom_num,
            balcony_num,
            age,
            total_floors,
            latitude,
            longitude,
            predicted_price,
            predicted_price_per_sqft,
            predicted_price_per_sqft,
        ], dtype=float).reshape(1, -1)

        expected = getattr(anomaly_model, "n_features_in_", None)
        if expected is not None and expected != features.shape[1]:
            print(
                "Anomaly model expects",
                expected,
                "features but received",
                features.shape[1],
            )
            return result

        prediction = anomaly_model.predict(features)[0]
        score = anomaly_model.decision_function(features)[0]

        result["score"] = safe_float(score)
        result["is_anomaly"] = bool(prediction == -1)
        result["status"] = (
            "Potentially Unusual"
            if prediction == -1
            else "Within Normal Range"
        )
    except Exception as exc:
        print("Anomaly detection error:", repr(exc))

    return result


# ============================================================
# MAP DATA
# ============================================================

def build_map_properties(
    locality,
    latitude,
    longitude,
    predicted_price,
    area,
    bedroom_num,
    bathroom_num,
    property_type,
    furnished,
    similar_properties,
):
    data = [{
        "type": "predicted",
        "latitude": safe_float(latitude),
        "longitude": safe_float(longitude),
        "locality": locality,
        "price": safe_float(predicted_price),
        "area": safe_float(area),
        "bedrooms": int(bedroom_num),
        "bathrooms": int(bathroom_num),
        "property_type": property_type,
        "furnished": furnished,
    }]

    for item in similar_properties:
        sim_locality = clean_string(item.get("locality"), "Unknown")
        sim_lat, sim_lon = get_locality_coordinates(sim_locality)
        data.append({
            "type": "similar",
            "latitude": safe_float(sim_lat),
            "longitude": safe_float(sim_lon),
            "locality": sim_locality,
            "price": safe_float(item.get("price")),
            "area": safe_float(item.get("area")),
            "bedrooms": safe_int(item.get("bedroom_num"), 0),
            "bathrooms": safe_int(item.get("bathroom_num"), 0),
            "property_type": clean_string(item.get("property_type"), "Unknown"),
        })

    return make_json_safe(data)


# ============================================================
# PROPERTY SEARCH / FINDER HELPERS
# ============================================================

def get_available_localities():
    """Return clean, unique locality names from both datasets."""
    names = set()

    if not property_data.empty and "locality" in property_data.columns:
        names.update(
            clean_string(x)
            for x in property_data["locality"].dropna().unique()
            if clean_string(x)
        )

    if not locality_data.empty and "locality" in locality_data.columns:
        names.update(
            clean_string(x)
            for x in locality_data["locality"].dropna().unique()
            if clean_string(x)
        )

    return sorted(names, key=lambda x: x.lower())


def get_available_property_types():
    if property_data.empty or "property_type" not in property_data.columns:
        return []

    values = [
        clean_string(x)
        for x in property_data["property_type"].dropna().unique()
        if clean_string(x)
    ]
    return sorted(set(values), key=lambda x: x.lower())


def serialize_property_row(row, index=None):
    """Convert one property-data row into frontend-safe search data."""
    return make_json_safe({
        "id": int(index) if index is not None else None,
        "locality": clean_string(row.get("locality"), "Unknown"),
        "city": clean_string(row.get("city"), "Mumbai"),
        "price": safe_float(row.get("price")),
        "predicted_price": safe_float(row.get("predicted_price")),
        "area": safe_float(row.get("area")),
        "price_per_sqft": safe_float(row.get("price_per_sqft")),
        "bedrooms": safe_int(row.get("bedroom_num"), 0),
        "bathrooms": safe_int(row.get("bathroom_num"), 0),
        "balconies": safe_int(row.get("balcony_num"), 0),
        "property_type": clean_string(row.get("property_type"), "Unknown"),
        "furnished": clean_string(row.get("furnished"), "Unknown"),
        "age": safe_int(row.get("age"), 0),
        "total_floors": safe_int(row.get("total_floors"), 0),
        "latitude": safe_float(row.get("latitude")),
        "longitude": safe_float(row.get("longitude")),
        "is_anomaly": bool(safe_int(row.get("is_anomaly"), 0)),
    })


def search_properties(
    locality="",
    min_budget=None,
    max_budget=None,
    bedrooms=None,
    property_type="",
    limit=100,
):
    """
    Search the full property intelligence dataset.

    Budget is based on the actual listing `price`, not the model prediction.
    This keeps the property finder useful as a market-search feature rather
    than mixing it with the prediction workflow.
    """
    if property_data.empty:
        return []

    df = property_data.copy()

    if "locality" in df.columns:
        df["_locality_clean"] = df["locality"].fillna("").apply(normalize_locality)

    # Numeric columns used for filtering.
    if "price" in df.columns:
        df["_price_num"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["_price_num"] = np.nan

    if "bedroom_num" in df.columns:
        df["_bedrooms_num"] = pd.to_numeric(df["bedroom_num"], errors="coerce")
    else:
        df["_bedrooms_num"] = np.nan

    # Locality
    locality = clean_string(locality)
    if locality:
        target = normalize_locality(locality)
        exact = df[df["_locality_clean"] == target]

        if exact.empty:
            contains = df[
                df["_locality_clean"].str.contains(
                    target, regex=False, na=False
                )
            ]
            df = contains
        else:
            df = exact

    # Budget
    if min_budget is not None:
        df = df[df["_price_num"] >= float(min_budget)]

    if max_budget is not None:
        df = df[df["_price_num"] <= float(max_budget)]

    # Bedrooms
    if bedrooms is not None:
        df = df[df["_bedrooms_num"] >= float(bedrooms)]

    # Property type
    property_type = clean_string(property_type)
    if property_type and "property_type" in df.columns:
        target_type = property_type.lower()
        df = df[
            df["property_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(target_type)
        ]

    if df.empty:
        return []

    # Most useful listings first: actual price ascending.
    df = df.sort_values(
        by="_price_num",
        ascending=True,
        na_position="last",
    )

    # Avoid sending thousands of rows to the browser.
    limit = max(1, min(int(limit), 300))
    results = []

    for idx, row in df.head(limit).iterrows():
        results.append(serialize_property_row(row, idx))

    return results


def build_search_summary(results):
    prices = [
        safe_float(item.get("price"))
        for item in results
        if safe_float(item.get("price")) is not None
    ]
    areas = [
        safe_float(item.get("area"))
        for item in results
        if safe_float(item.get("area")) is not None
    ]

    return make_json_safe({
        "count": len(results),
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
        "average_price": float(np.mean(prices)) if prices else None,
        "average_area": float(np.mean(areas)) if areas else None,
    })


def parse_optional_float(value):
    value = clean_string(value)
    if not value:
        return None
    number = safe_float(value)
    if number is None or number < 0:
        return None
    return number


def parse_optional_int(value):
    value = clean_string(value)
    if not value:
        return None
    number = safe_int(value, -1)
    if number < 0:
        return None
    return number


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        model_error=model_error,
        preprocessor_error=preprocessor_error,
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    total_properties = 0
    total_localities = 0
    average_price = 0
    anomaly_count = 0
    top_localities = []
    price_sqft_localities = []

    if not property_data.empty:
        total_properties = int(len(property_data))
        if "locality" in property_data.columns:
            total_localities = int(property_data["locality"].dropna().nunique())
        if "price" in property_data.columns:
            average_price = safe_float(
                pd.to_numeric(property_data["price"], errors="coerce").mean(),
                0,
            )
        if "is_anomaly" in property_data.columns:
            anomaly_count = int(
                pd.to_numeric(
                    property_data["is_anomaly"], errors="coerce"
                ).fillna(0).sum()
            )

    if not locality_data.empty:
        temp = locality_data.copy()
        if "listing_count" in temp.columns:
            temp["listing_count"] = pd.to_numeric(
                temp["listing_count"], errors="coerce"
            )
            top_localities = (
                temp.dropna(subset=["listing_count"])
                .sort_values("listing_count", ascending=False)
                .head(10)
                .to_dict("records")
            )
        if "average_price_per_sqft" in temp.columns:
            temp["average_price_per_sqft"] = pd.to_numeric(
                temp["average_price_per_sqft"], errors="coerce"
            )
            price_sqft_localities = (
                temp.dropna(subset=["average_price_per_sqft"])
                .sort_values("average_price_per_sqft", ascending=False)
                .head(10)
                .to_dict("records")
            )

    # If locality_intelligence.csv is incomplete, build dashboard locality
    # tables from the same property dataset used by the search/prediction system.
    if not top_localities and not property_data.empty and "locality" in property_data.columns:
        temp = property_data.copy()
        temp["_price"] = pd.to_numeric(temp.get("price"), errors="coerce")
        grouped = (
            temp.groupby("locality", dropna=True)
            .agg(
                listing_count=("locality", "size"),
                average_price=("_price", "mean"),
            )
            .reset_index()
            .sort_values("listing_count", ascending=False)
            .head(10)
        )
        top_localities = grouped.to_dict("records")

    if not price_sqft_localities and not property_data.empty and "locality" in property_data.columns:
        temp = property_data.copy()
        if "price_per_sqft" in temp.columns:
            temp["_ppsf"] = pd.to_numeric(
                temp["price_per_sqft"], errors="coerce"
            )
            grouped = (
                temp.groupby("locality", dropna=True)
                .agg(average_price_per_sqft=("_ppsf", "mean"))
                .reset_index()
                .dropna(subset=["average_price_per_sqft"])
                .sort_values("average_price_per_sqft", ascending=False)
                .head(10)
            )
            price_sqft_localities = grouped.to_dict("records")

    return render_template(
        "dashboard.html",
        total_properties=total_properties,
        total_localities=total_localities,
        average_price=average_price,
        anomaly_count=anomaly_count,
        top_localities=make_json_safe(top_localities),
        price_sqft_localities=make_json_safe(price_sqft_localities),
        available_localities=get_available_localities(),
        property_types=get_available_property_types(),
    )


# ============================================================
# LOCALITY EXPLORER API
# ============================================================

@app.route("/api/localities")
def api_localities():
    return jsonify(make_json_safe({
        "localities": get_available_localities(),
        "property_types": get_available_property_types(),
    }))


@app.route("/api/locality-data")
def api_locality_data():
    locality = clean_string(request.args.get("locality"))

    if not locality:
        return jsonify({
            "success": False,
            "message": "Please provide a locality.",
        }), 400

    stats = get_locality_statistics(locality)
    latitude, longitude = get_locality_coordinates(locality)

    if not stats:
        return jsonify(make_json_safe({
            "success": False,
            "message": "No market data found for this locality.",
            "locality": locality,
        }))

    stats = make_json_safe(stats)
    stats["latitude"] = latitude
    stats["longitude"] = longitude

    return jsonify(make_json_safe({
        "success": True,
        "data": stats,
    }))


# ============================================================
# PROPERTY FINDER
# ============================================================

@app.route("/property-finder")
def property_finder():
    return render_template(
        "property_finder.html",
        localities=get_available_localities(),
        property_types=get_available_property_types(),
    )


@app.route("/api/property-search")
def api_property_search():
    try:
        locality = clean_string(request.args.get("locality"))
        min_budget = parse_optional_float(request.args.get("min_budget"))
        max_budget = parse_optional_float(request.args.get("max_budget"))
        bedrooms = parse_optional_int(request.args.get("bedrooms"))
        property_type = clean_string(request.args.get("property_type"))
        limit = safe_int(request.args.get("limit"), 100)

        if (
            min_budget is not None
            and max_budget is not None
            and min_budget > max_budget
        ):
            return jsonify({
                "success": False,
                "message": "Minimum budget cannot be greater than maximum budget.",
            }), 400

        results = search_properties(
            locality=locality,
            min_budget=min_budget,
            max_budget=max_budget,
            bedrooms=bedrooms,
            property_type=property_type,
            limit=limit,
        )

        return jsonify(make_json_safe({
            "success": True,
            "filters": {
                "locality": locality,
                "min_budget": min_budget,
                "max_budget": max_budget,
                "bedrooms": bedrooms,
                "property_type": property_type,
            },
            "summary": build_search_summary(results),
            "results": results,
        }))

    except Exception as exc:
        print("Property search error:", repr(exc))
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Could not search the property dataset.",
        }), 500


# ============================================================
# PROPERTY MAP
# ============================================================

@app.route("/property-map")
def property_map():
    map_path = os.path.join(PROCESSED_DIR, "mumbai_property_map.html")
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as file:
            return file.read()
    return "<h2>Property map is not available.</h2><p>Please run the location intelligence notebook.</p>"


@app.route("/property-density-heatmap")
def property_density_heatmap():
    heatmap_path = os.path.join(PROCESSED_DIR, "property_density_heatmap.html")
    if os.path.exists(heatmap_path):
        with open(heatmap_path, "r", encoding="utf-8") as file:
            return file.read()
    return "<h2>Property density heatmap is not available.</h2><p>Please run the location intelligence notebook.</p>"


@app.route("/api/property-map")
def api_property_map():
    """Backward-compatible map API with optional search filters."""
    try:
        locality = clean_string(request.args.get("locality"))
        min_budget = parse_optional_float(request.args.get("min_budget"))
        max_budget = parse_optional_float(request.args.get("max_budget"))
        bedrooms = parse_optional_int(request.args.get("bedrooms"))
        property_type = clean_string(request.args.get("property_type"))
        limit = safe_int(request.args.get("limit"), 300)

        results = search_properties(
            locality=locality,
            min_budget=min_budget,
            max_budget=max_budget,
            bedrooms=bedrooms,
            property_type=property_type,
            limit=min(limit, 500),
        )

        data = []
        for item in results:
            if item.get("latitude") is None or item.get("longitude") is None:
                continue
            data.append({
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "locality": item.get("locality"),
                "price": item.get("price"),
                "predicted_price": item.get("predicted_price"),
                "area": item.get("area"),
                "bedrooms": item.get("bedrooms"),
                "bathrooms": item.get("bathrooms"),
                "property_type": item.get("property_type"),
                "furnished": item.get("furnished"),
                "is_anomaly": item.get("is_anomaly"),
            })

        return jsonify(make_json_safe(data))

    except Exception as exc:
        print("Property map API error:", repr(exc))
        return jsonify([]), 500


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            raise RuntimeError(
                model_error or
                "ML model could not be loaded. Check the terminal for the exact error."
            )
        if preprocessor is None:
            raise RuntimeError(
                preprocessor_error or
                "Preprocessor could not be loaded."
            )

        area = float(request.form.get("area", "0"))
        locality = clean_string(request.form.get("locality"))
        bedroom_num = int(request.form.get("bedroom_num", "0"))
        bathroom_num = int(request.form.get("bathroom_num", "0"))
        balcony_num = int(request.form.get("balcony_num", "0"))
        age = int(request.form.get("age", "0"))
        total_floors = int(request.form.get("total_floors", "1"))
        property_type = clean_string(request.form.get("property_type"), "Apartment")
        furnished = clean_string(request.form.get("furnished"), "Unfurnished")

        listed_price_raw = (
            request.form.get("listed_price", "")
            or request.form.get("asking_price", "")
            or request.form.get("current_listing_price", "")
        )
        listed_price = None
        if listed_price_raw not in (None, ""):
            listed_price = float(listed_price_raw)
            if listed_price <= 0:
                listed_price = None

        if area <= 0:
            raise ValueError("Area must be greater than 0.")
        if bedroom_num < 0:
            raise ValueError("Bedrooms cannot be negative.")
        if bathroom_num < 0:
            raise ValueError("Bathrooms cannot be negative.")
        if balcony_num < 0:
            raise ValueError("Balconies cannot be negative.")
        if age < 0:
            raise ValueError("Property age cannot be negative.")
        if total_floors <= 0:
            raise ValueError("Total floors must be greater than 0.")
        if not locality:
            raise ValueError("Please enter a locality.")

        latitude, longitude = get_locality_coordinates(locality)

        final_input = build_model_input(
            area,
            locality,
            bedroom_num,
            bathroom_num,
            balcony_num,
            age,
            total_floors,
            property_type,
            furnished,
            latitude,
            longitude,
        )

        processed_input = preprocessor.transform(final_input)
        predicted_price = max(0, float(model.predict(processed_input)[0]))
        predicted_price_per_sqft = predicted_price / area

        locality_stats = get_locality_statistics(locality)
        locality_average_price = None
        locality_median_price = None
        locality_average_price_per_sqft = None
        locality_listing_count = None

        if locality_stats:
            locality_average_price = safe_float(locality_stats.get("average_price"))
            locality_median_price = safe_float(locality_stats.get("median_price"))
            locality_average_price_per_sqft = safe_float(
                locality_stats.get("average_price_per_sqft")
            )
            locality_listing_count = safe_int(
                locality_stats.get("listing_count"), 0
            )

        price_difference = None
        price_difference_percent = None
        if locality_average_price and locality_average_price != 0:
            price_difference = predicted_price - locality_average_price
            price_difference_percent = (
                price_difference / locality_average_price
            ) * 100

        similar_properties = find_similar_properties(
            area,
            bedroom_num,
            bathroom_num,
            balcony_num,
            age,
            total_floors,
            latitude,
            longitude,
        )

        map_properties = build_map_properties(
            locality,
            latitude,
            longitude,
            predicted_price,
            area,
            bedroom_num,
            bathroom_num,
            property_type,
            furnished,
            similar_properties,
        )

        (
            shap_positive,
            shap_negative,
            shap_top_features,
            shap_prediction,
            shap_expected_value,
        ) = generate_shap_explanation(processed_input)

        if shap_prediction is None:
            shap_prediction = predicted_price

        anomaly = detect_anomaly(
            area,
            predicted_price,
            bedroom_num,
            bathroom_num,
            balcony_num,
            age,
            total_floors,
            latitude,
            longitude,
            predicted_price_per_sqft,
        )

        confidence = get_prediction_confidence(
            processed_input,
            predicted_price,
        )

        market_position = calculate_market_position(
            predicted_price,
            locality_average_price,
        )

        deal_analysis = calculate_deal_analysis(
            predicted_price,
            listed_price,
            locality_average_price,
        )

        intelligence_score = calculate_intelligence_score(
            predicted_price,
            locality_average_price,
            confidence,
            anomaly["status"],
            similar_properties,
            locality_listing_count=locality_listing_count,
            locality_average_price_per_sqft=locality_average_price_per_sqft,
            predicted_price_per_sqft=predicted_price_per_sqft,
            area=area,
            bedroom_num=bedroom_num,
            bathroom_num=bathroom_num,
            balcony_num=balcony_num,
            age=age,
            total_floors=total_floors,
        )

        what_if = generate_what_if_analysis(
            area,
            locality,
            bedroom_num,
            bathroom_num,
            balcony_num,
            age,
            total_floors,
            property_type,
            furnished,
            latitude,
            longitude,
        )

        context = {
            "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "property": {
                "area": area,
                "locality": locality,
                "bedroom_num": bedroom_num,
                "bathroom_num": bathroom_num,
                "balcony_num": balcony_num,
                "age": age,
                "total_floors": total_floors,
                "property_type": property_type,
                "furnished": furnished,
                "latitude": latitude,
                "longitude": longitude,
            },
            "prediction": {
                "price": predicted_price,
                "price_per_sqft": predicted_price_per_sqft,
            },
            "locality": {
                "stats": locality_stats,
                "average_price": locality_average_price,
                "median_price": locality_median_price,
                "average_price_per_sqft": locality_average_price_per_sqft,
                "listing_count": locality_listing_count,
            },
            "comparison": {
                "price_difference": price_difference,
                "price_difference_percent": price_difference_percent,
                "market_position": market_position,
            },
            "listed_price": listed_price,
            "confidence": confidence,
            "deal_analysis": deal_analysis,
            "intelligence_score": intelligence_score,
            "anomaly": anomaly,
            "shap": {
                "positive": shap_positive,
                "negative": shap_negative,
                "top_features": shap_top_features,
                "prediction": shap_prediction,
                "expected_value": shap_expected_value,
            },
            "similar_properties": similar_properties,
            "what_if": what_if,
            "map_properties": map_properties,
        }

        # Keep the Flask session small enough for cookie storage.
        session["deal_context"] = make_json_safe(context)

        return render_template(
            "result.html",
            predicted_price=predicted_price,
            predicted_price_per_sqft=predicted_price_per_sqft,
            area=area,
            locality=locality,
            bedroom_num=bedroom_num,
            bathroom_num=bathroom_num,
            balcony_num=balcony_num,
            age=age,
            total_floors=total_floors,
            property_type=property_type,
            furnished=furnished,
            latitude=latitude,
            longitude=longitude,
            locality_stats=locality_stats,
            locality_average_price=locality_average_price,
            locality_median_price=locality_median_price,
            locality_average_price_per_sqft=locality_average_price_per_sqft,
            locality_listing_count=locality_listing_count,
            price_difference=price_difference,
            price_difference_percent=price_difference_percent,
            similar_properties=similar_properties,
            map_properties=map_properties,
            shap_positive=shap_positive,
            shap_negative=shap_negative,
            shap_top_features=shap_top_features,
            shap_prediction=shap_prediction,
            shap_expected_value=shap_expected_value,
            anomaly_status=anomaly["status"],
            anomaly_score=anomaly["score"],
            is_anomaly=anomaly["is_anomaly"],
            confidence=confidence,
            market_position=market_position,
            deal_analysis=deal_analysis,
            intelligence_score=intelligence_score,
            what_if=what_if,
            listed_price=listed_price,
        )

    except ValueError as exc:
        return render_template("index.html", error=str(exc))
    except Exception as exc:
        print("\n" + "=" * 70)
        print("PREDICTION ERROR:")
        print(repr(exc))
        traceback.print_exc()
        print("=" * 70 + "\n")
        return render_template(
            "index.html",
            error=(
                "Something went wrong while calculating the prediction. "
                f"Error: {str(exc)}"
            ),
        )


# ============================================================
# DEAL ANALYZER
# ============================================================

@app.route("/deal-analyzer")
@app.route("/deal_analyzer")
def deal_analyzer():
    context = session.get("deal_context", {})
    return render_template("deal_analyzer.html", context=context)


# ============================================================
# PDF HELPERS
# ============================================================

def pdf_money(value):
    value = safe_float(value, 0)
    return format_currency(value)


def pdf_number(value):
    value = safe_float(value, 0)
    return f"{value:,.2f}"


def build_pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=7,
        textColor=colors.HexColor("#0b2b55"),
    ))
    return styles


def pdf_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2b55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def create_valuation_pdf(context, output_path):
    styles = build_pdf_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    p = context.get("property", {})
    pred = context.get("prediction", {})
    locality = context.get("locality", {})
    confidence = context.get("confidence", {})
    deal = context.get("deal_analysis", {})
    intelligence = context.get("intelligence_score", {})
    anomaly = context.get("anomaly", {})
    shap_data = context.get("shap", {})
    similar = context.get("similar_properties", [])
    what_if = context.get("what_if", [])
    comparison = context.get("comparison", {})

    story = []

    story.append(Paragraph(
        "Mumbai Property Valuation Report",
        styles["ReportTitleCustom"]
    ))
    story.append(Paragraph(
        f"Generated: {clean_string(context.get('created_at'), datetime.now().strftime('%d %b %Y'))}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Property Overview", styles["SectionCustom"]))
    overview = [
        ["Field", "Value"],
        ["Locality", clean_string(p.get("locality"), "Unknown")],
        ["Area", f"{safe_float(p.get('area'), 0):,.0f} sq.ft"],
        ["Bedrooms", str(p.get("bedroom_num", 0))],
        ["Bathrooms", str(p.get("bathroom_num", 0))],
        ["Balconies", str(p.get("balcony_num", 0))],
        ["Property Age", f"{p.get('age', 0)} years"],
        ["Total Floors", str(p.get("total_floors", 0))],
        ["Property Type", clean_string(p.get("property_type"), "Unknown")],
        ["Furnished", clean_string(p.get("furnished"), "Unknown")],
    ]
    story.append(pdf_table(overview, [55 * mm, 115 * mm]))

    story.append(Paragraph("2. Valuation Summary", styles["SectionCustom"]))
    valuation_data = [
        ["Metric", "Value"],
        ["Estimated Property Value", pdf_money(pred.get("price", 0))],
        ["Estimated Price / Sq.ft", pdf_money(pred.get("price_per_sqft", 0))],
        ["Estimated Lower Range", pdf_money(confidence.get("lower", 0))],
        ["Estimated Upper Range", pdf_money(confidence.get("upper", 0))],
    ]
    story.append(pdf_table(valuation_data, [75 * mm, 95 * mm]))

    story.append(Paragraph("3. Market Comparison", styles["SectionCustom"]))
    comparison_data = [
        ["Metric", "Value"],
        ["Locality Average Price", pdf_money(locality.get("average_price", 0))],
        ["Locality Median Price", pdf_money(locality.get("median_price", 0))],
        ["Locality Average / Sq.ft", pdf_money(locality.get("average_price_per_sqft", 0))],
        ["Listing Count", str(locality.get("listing_count", "N/A"))],
        ["Difference vs Locality Average", pdf_money(comparison.get("price_difference", 0))],
        ["Difference %", f"{safe_float(comparison.get('price_difference_percent'), 0):.2f}%"],
        ["Market Position", clean_string(comparison.get("market_position", {}).get("label"), "N/A")],
    ]
    story.append(pdf_table(comparison_data, [75 * mm, 95 * mm]))

    story.append(Paragraph("4. Deal Analysis", styles["SectionCustom"]))
    deal_data = [
        ["Metric", "Value"],
        ["Listed Price", pdf_money(deal.get("listed_price", 0)) if deal.get("listed_price") else "Not supplied"],
        ["Difference", pdf_money(deal.get("difference", 0))],
        ["Difference %", f"{safe_float(deal.get('difference_percent'), 0):.2f}%"],
        ["Status", clean_string(deal.get("status"), "N/A")],
        ["Message", clean_string(deal.get("message"), "N/A")],
    ]
    story.append(pdf_table(deal_data, [55 * mm, 115 * mm]))

    story.append(Paragraph("5. Property Intelligence Score", styles["SectionCustom"]))
    score_data = [
        ["Metric", "Value"],
        ["Score", f"{safe_float(intelligence.get('score'), 0):.1f} / 100"],
        ["Interpretation", clean_string(intelligence.get("label"), "N/A")],
        ["Confidence Method", clean_string(confidence.get("method"), "N/A")],
    ]
    story.append(pdf_table(score_data, [65 * mm, 105 * mm]))

    story.append(Paragraph("6. Anomaly Detection", styles["SectionCustom"]))
    anomaly_data = [
        ["Metric", "Value"],
        ["Status", clean_string(anomaly.get("status"), "N/A")],
        ["Anomaly Score", pdf_number(anomaly.get("score", 0))],
        ["Flagged", "Yes" if anomaly.get("is_anomaly") else "No"],
    ]
    story.append(pdf_table(anomaly_data, [65 * mm, 105 * mm]))

    story.append(Paragraph("7. Explainable AI - SHAP", styles["SectionCustom"]))
    shap_rows = [["Feature", "SHAP Value"]]
    for item in shap_data.get("top_features", [])[:10]:
        shap_rows.append([
            clean_string(item.get("feature"), "Unknown"),
            pdf_number(item.get("shap_value", 0)),
        ])
    if len(shap_rows) == 1:
        shap_rows.append(["Not available", "-"])
    story.append(pdf_table(shap_rows, [120 * mm, 50 * mm]))

    story.append(Paragraph("8. Similar Properties", styles["SectionCustom"]))
    similar_rows = [["Locality", "Area", "Bedrooms", "Bathrooms", "Price"]]
    for item in similar[:6]:
        similar_rows.append([
            clean_string(item.get("locality"), "Unknown"),
            f"{safe_float(item.get('area'), 0):,.0f}",
            str(item.get("bedroom_num", "-")),
            str(item.get("bathroom_num", "-")),
            pdf_money(item.get("price", 0)),
        ])
    if len(similar_rows) == 1:
        similar_rows.append(["Not available", "-", "-", "-", "-"])
    story.append(pdf_table(similar_rows, [45 * mm, 30 * mm, 25 * mm, 25 * mm, 45 * mm]))

    story.append(Paragraph("9. What-If Area Analysis", styles["SectionCustom"]))
    what_if_rows = [["Area (sq.ft)", "Predicted Price", "Price / Sq.ft"]]
    for item in what_if:
        what_if_rows.append([
            f"{safe_float(item.get('area'), 0):,.0f}",
            pdf_money(item.get("predicted_price", 0)),
            pdf_money(item.get("price_per_sqft", 0)),
        ])
    if len(what_if_rows) == 1:
        what_if_rows.append(["Not available", "-", "-"])
    story.append(pdf_table(what_if_rows, [55 * mm, 65 * mm, 50 * mm]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Disclaimer: This report is an AI/ML-based estimate generated from the project's trained model and available property data. It is not a legal, banking, or professional property appraisal.",
        styles["Normal"]
    ))

    doc.build(story)


# ============================================================
# DOWNLOAD PDF
# ============================================================

@app.route("/download-report")
@app.route("/generate-report")
def download_report():
    context = session.get("deal_context")

    if not context:
        return render_template(
            "index.html",
            error="Please generate a property prediction first, then download the report."
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        "current_property_valuation_report.pdf"
    )

    try:
        create_valuation_pdf(context, output_path)
        return send_file(
            output_path,
            as_attachment=True,
            download_name="Mumbai_Property_Valuation_Report.pdf",
            mimetype="application/pdf",
        )
    except Exception as exc:
        print("PDF generation error:", repr(exc))
        traceback.print_exc()

        p = context.get("property", {})
        pred = context.get("prediction", {})
        loc = context.get("locality", {})
        comparison = context.get("comparison", {})
        anomaly = context.get("anomaly", {})
        shap_data = context.get("shap", {})
        return render_template(
            "result.html",
            error=f"Could not generate PDF: {exc}",
            predicted_price=pred.get("price", 0),
            predicted_price_per_sqft=pred.get("price_per_sqft", 0),
            area=p.get("area", 0),
            locality=p.get("locality", "Unknown"),
            bedroom_num=p.get("bedroom_num", 0),
            bathroom_num=p.get("bathroom_num", 0),
            balcony_num=p.get("balcony_num", 0),
            age=p.get("age", 0),
            total_floors=p.get("total_floors", 0),
            property_type=p.get("property_type", "Unknown"),
            furnished=p.get("furnished", "Unknown"),
            latitude=p.get("latitude"),
            longitude=p.get("longitude"),
            locality_stats=loc.get("stats"),
            locality_average_price=loc.get("average_price"),
            locality_median_price=loc.get("median_price"),
            locality_average_price_per_sqft=loc.get("average_price_per_sqft"),
            locality_listing_count=loc.get("listing_count"),
            price_difference=comparison.get("price_difference"),
            price_difference_percent=comparison.get("price_difference_percent"),
            similar_properties=context.get("similar_properties", []),
            map_properties=context.get("map_properties", []),
            shap_positive=shap_data.get("positive", []),
            shap_negative=shap_data.get("negative", []),
            shap_top_features=shap_data.get("top_features", []),
            shap_prediction=shap_data.get("prediction"),
            shap_expected_value=shap_data.get("expected_value"),
            anomaly_status=anomaly.get("status", "Not Available"),
            anomaly_score=anomaly.get("score"),
            is_anomaly=anomaly.get("is_anomaly", False),
            confidence=context.get("confidence", {}),
            market_position=comparison.get("market_position", {}),
            deal_analysis=context.get("deal_analysis", {}),
            intelligence_score=context.get("intelligence_score", {}),
            what_if=context.get("what_if", []),
            listed_price=context.get("listed_price"),
        )


# ============================================================
# HEALTH / DEBUG ROUTE
# ============================================================

@app.route("/health")
def health():
    return jsonify(make_json_safe({
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "anomaly_model_loaded": anomaly_model is not None,
        "property_data_loaded": not property_data.empty,
        "locality_data_loaded": not locality_data.empty,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "preprocessor_path": PREPROCESSOR_PATH,
        "preprocessor_exists": os.path.exists(PREPROCESSOR_PATH),
        "model_error": model_error,
        "preprocessor_error": preprocessor_error,
    }))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\nStarting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/health")
    app.run(debug=True)
