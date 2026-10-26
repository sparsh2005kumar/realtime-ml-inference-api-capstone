
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Small synthetic demonstration dataset
X = pd.DataFrame({
    "feature_01": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0,
                   0.15, 0.25, 0.85, 0.95],
    "feature_02": [1.0, 0.9, 0.8, 0.2, 0.1, 0.0,
                   0.95, 0.85, 0.15, 0.05]
})
y = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X, y)

model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "champion_model.joblib")

print("Model saved successfully!")
print("Location:", (model_dir / "champion_model.joblib").resolve())
print("Feature names:", list(X.columns))
