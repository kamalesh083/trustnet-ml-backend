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
    X = np.array([[features[f] for f in FEATURES]])

    # 2. Predict trust score
    trust_score = float(model.predict(X)[0])

    # 3. SHAP explanation
    shap_values = explainer.shap_values(X)[0]

    # 4. Top reasons
    reasons = sorted(
        zip(FEATURES, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # 5. Convert to sentences
    explanation = []
    for f, v in reasons:
        if v > 0:
            explanation.append(f"{f} increases trust")
        else:
            explanation.append(f"{f} reduces trust")

    return {
        "trust_score": round(trust_score, 2),
        "reasons": explanation
    }
