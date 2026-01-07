from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI(title="TrustNet ML API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL & SHAP ----------------
model = joblib.load("model/trustnet_model.pkl")
explainer = joblib.load("model/shap_explainer.pkl")

FEATURES = list(model.feature_names_in_)

@app.get("/")
def root():
    return {"status": "TrustNet ML server running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI(title="TrustNet ML API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL & SHAP ----------------
model = joblib.load("model/trustnet_model.pkl")
explainer = joblib.load("model/shap_explainer.pkl")

FEATURES = list(model.feature_names_in_)

@app.get("/")
def root():
    return {"status": "TrustNet ML server running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI(title="TrustNet ML API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL & SHAP ----------------
model = joblib.load("model/trustnet_model.pkl")
explainer = joblib.load("model/shap_explainer.pkl")

FEATURES = list(model.feature_names_in_)

@app.get("/")
def root():
    return {"status": "TrustNet ML server running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI(title="TrustNet ML API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL & SHAP ----------------
model = joblib.load("model/trustnet_model.pkl")
explainer = joblib.load("model/shap_explainer.pkl")

FEATURES = list(model.feature_names_in_)

@app.get("/")
def root():
    return {"status": "TrustNet ML server running"}

@app.post("/predict")
def predict(features: dict):
    # 1. Create input vector
    X = np.array([[float(features[f]) for f in FEATURES]])

    # 2. Predict trust score (convert explicitly)
    trust_score = float(model.predict(X)[0])

    # 3. SHAP explanation (convert to Python list)
    shap_values = explainer.shap_values(X)[0].tolist()

    # ---------------- CONFIDENCE SCORE ----------------
    abs_shap = np.abs(np.array(shap_values))
    signal_strength = float(abs_shap.sum())

    top_k = np.sort(abs_shap)[-5:]
    dominance = float(top_k.sum() / (signal_strength + 1e-8))

    confidence_score = round(dominance * 100, 2)
    # --------------------------------------------------

    # 4. Top reasons
    reasons = sorted(
        zip(FEATURES, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # 5. Convert to sentences
    explanation = []
    for f, v in reasons:
        v = float(v)  # 🚨 critical
        if v > 0:
            explanation.append(f"{f} increases trust")
        else:
            explanation.append(f"{f} reduces trust")

    # 6. Final response (PURE PYTHON TYPES ONLY)
    return {
        "trust_score": round(trust_score, 2),
        "confidence_score": confidence_score,
        "reasons": explanation
    }

