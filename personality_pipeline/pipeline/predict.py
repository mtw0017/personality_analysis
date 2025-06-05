import joblib
import pandas as pd

def predict_personality(input_dict, model_path="model/personality_rf.pkl"):
    clf = joblib.load(model_path)
    input_df = pd.DataFrame([input_dict])
    result = clf.predict(input_df)[0]
    return "Introvert" if result == 1 else "Extrovert"
