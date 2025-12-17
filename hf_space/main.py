from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("model.joblib")

EXPECTED_FEATURES = list(model.feature_names_in_)

app = FastAPI(
    title="Game of Thrones House Predictor API",
    version="1.0"
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
def predict(data: CharacterInput):
    # Build full feature vector
    row = {col: 0 for col in EXPECTED_FEATURES}

    # Numeric + boolean
    row["honour_1to5"] = data.honour_1to5
    row["ruthlessness_1to5"] = data.ruthlessness_1to5
    row["intelligence_1to5"] = data.intelligence_1to5
    row["combat_skill_1to5"] = data.combat_skill_1to5
    row["diplomacy_1to5"] = data.diplomacy_1to5
    row["leadership_1to5"] = data.leadership_1to5
    row["trait_loyal"] = int(data.trait_loyal)
    row["trait_scheming"] = int(data.trait_scheming)

    # One-hot categorical
    def set_onehot(prefix, val):
        key = f"{prefix}_{val}"
        if key in row:
            row[key] = 1

    set_onehot("region", data.region)
    set_onehot("primary_role", data.primary_role)
    set_onehot("alignment", data.alignment)
    set_onehot("status", data.status)
    set_onehot("species", data.species)

    df = pd.DataFrame([row])
    pred = model.predict(df)[0]

    return {"predicted_house": pred}
