# predict_and_export.py
import torch
import pandas as pd
from src.model import DiabetesNet
from src.preprocess import load_and_preprocess_data
import os

# === Config ===
DATA_PATH = "data/diabetes_prediction_dataset.csv"
MODEL_PATH = "diabetes_model.pth"
OUTPUT_PATH = "outputs/diabetes_predictions.csv"

# === Ensure output folder exists ===
os.makedirs("outputs", exist_ok=True)

# === Step 1: Load and preprocess data ===
X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
input_dim = X_test.shape[1]

# === Step 2: Load trained model ===
model = DiabetesNet(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# === Step 3: Make predictions ===
X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    probs = model(X_tensor).squeeze().numpy()
    preds = (probs > 0.5).astype(int)

# === Step 4: Prepare export dataframe ===
df_raw = pd.read_csv(DATA_PATH)
split_idx = int(0.8 * len(df_raw))  # assuming test_size=0.2

df_output = df_raw.iloc[split_idx:].copy()
df_output["diabetes_prob"] = probs
df_output["diabetes_predicted"] = preds

# === Step 5: Save to CSV for Power BI ===
df_output.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions exported to {OUTPUT_PATH}")
