from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.joblib")

# Extract expected feature names from the model
EXPECTED_FEATURES = list(model.feature_names_in_)

app = FastAPI(
    title="Game of Thrones House Predictor",
    description="Decision Tree model trained using Azure ML pipeline",
    version="1.0",
)

class CharacterInput(BaseModel):
    region: str
    primary_role: str
    alignment: str
    status: str
    species: str
    honour_1to5: int
    ruthlessness_1to5: int
    intelligence_1to5: int
    combat_skill_1to5: int
    diplomacy_1to5: int
    leadership_1to5: int
    trait_loyal: bool
    trait_scheming: bool


@app.post("/predict")
def predict_house(data: CharacterInput):
    # 1️⃣ Create empty feature row with ALL expected columns
    row = {col: 0 for col in EXPECTED_FEATURES}

    # 2️⃣ Fill numeric / boolean features
    row["honour_1to5"] = data.honour_1to5
    row["ruthlessness_1to5"] = data.ruthlessness_1to5
    row["intelligence_1to5"] = data.intelligence_1to5
    row["combat_skill_1to5"] = data.combat_skill_1to5
    row["diplomacy_1to5"] = data.diplomacy_1to5
    row["leadership_1to5"] = data.leadership_1to5
    row["trait_loyal"] = int(data.trait_loyal)
    row["trait_scheming"] = int(data.trait_scheming)

    # 3️⃣ Handle categorical one-hot columns safely
    def set_onehot(prefix, value):
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1

    set_onehot("region", data.region)
    set_onehot("primary_role", data.primary_role)
    set_onehot("alignment", data.alignment)
    set_onehot("status", data.status)
    set_onehot("species", data.species)

    df = pd.DataFrame([row])

    prediction = model.predict(df)[0]

    return {"predicted_house": prediction}
