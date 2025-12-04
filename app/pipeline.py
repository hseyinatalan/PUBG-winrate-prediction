import joblib
import pandas as pd

model = joblib.load("lgb_optimized_pubg.pkl")

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    # gerekli feature engineering burada yapılmalı
    return model.predict(df)[0]
