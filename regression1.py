import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.model import DiabetesNet
from src.preprocess import load_and_preprocess_data

# ----- Step 1: Load Preprocessed Data -----
filepath = "data/diabetes_prediction_dataset.csv"
X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
input_dim = X_test.shape[1]

# ----- Step 2: Load Trained Model (.pth) -----
model = DiabetesNet(input_dim)
model.load_state_dict(torch.load("diabetes_model.pth", map_location=torch.device('cpu')))
model.eval()

# ----- Step 3: Make Predictions -----
X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    probs = model(X_tensor).squeeze().numpy()

# ----- Step 4: Get Original "Age" Column for Test Set -----
df_raw = pd.read_csv(filepath)
test_indices = int(0.2 * len(df_raw))  # assuming test size = 0.2
age_column = df_raw.iloc[-test_indices:]['age'].values

# ----- Step 5: Plot Age vs Predicted Diabetes Probability -----
df_plot = pd.DataFrame({
    'age': age_column,
    'diabetes_prob': probs
})

plt.figure(figsize=(10, 6))
sns.regplot(x='age', y='diabetes_prob', data=df_plot, logistic=True,
            scatter_kws={"s": 20, "alpha": 0.4}, line_kws={"color": "red"})
plt.title("Age vs Predicted Diabetes Probability")
plt.xlabel("Age")
plt.ylabel("Predicted Probability of Diabetes")
plt.grid(True)
plt.tight_layout()
plt.show()
