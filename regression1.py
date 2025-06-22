import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.model import DiabetesNet

# Step 1: Load dataset
df_raw = pd.read_csv("data/diabetes_prediction_dataset.csv")
df = df_raw.copy()

# Step 2: Label encode categorical variables
label_encoders = {}
for col in ['gender', 'smoking_history']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Step 3: Normalize numeric columns
numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 4: Split dataset (80% train, 20% test)
X = df.drop('diabetes', axis=1)
y = df['diabetes']
split_idx = int(0.8 * len(df))
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Get original values of selected columns before normalization (for plotting)
age_orig = df_raw.iloc[split_idx:]['age'].values
hba1c_orig = df_raw.iloc[split_idx:]['HbA1c_level'].values
glucose_orig = df_raw.iloc[split_idx:]['blood_glucose_level'].values

# Step 5: Load the trained model
input_dim = X.shape[1]
model = DiabetesNet(input_dim)
model.load_state_dict(torch.load("diabetes_model.pth", map_location=torch.device('cpu')))
model.eval()

# Step 6: Predict diabetes probabilities
X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
with torch.no_grad():
    probs = model(X_tensor).squeeze().numpy()

# Step 7: Prepare DataFrame for plotting
df_plot = pd.DataFrame({
    'age': age_orig,
    'HbA1c_level': hba1c_orig,
    'blood_glucose_level': glucose_orig,
    'diabetes_prob': probs
})

# Step 8: Plotting
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Age vs Diabetes Probability
sns.regplot(ax=axes[0], x='age', y='diabetes_prob', data=df_plot, logistic=True,
            scatter_kws={"s": 20, "alpha": 0.4}, line_kws={"color": "red"})
axes[0].set_title("Age vs Diabetes Probability")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Probability")

# 2. HbA1c Level vs Diabetes Probability
sns.regplot(ax=axes[1], x='HbA1c_level', y='diabetes_prob', data=df_plot, logistic=True,
            scatter_kws={"s": 20, "alpha": 0.4}, line_kws={"color": "red"})
axes[1].set_title("HbA1c Level vs Diabetes Probability")
axes[1].set_xlabel("HbA1c Level")

# 3. Blood Glucose vs Diabetes Probability
sns.regplot(ax=axes[2], x='blood_glucose_level', y='diabetes_prob', data=df_plot, logistic=True,
            scatter_kws={"s": 20, "alpha": 0.4}, line_kws={"color": "red"})
axes[2].set_title("Blood Glucose Level vs Diabetes Probability")
axes[2].set_xlabel("Blood Glucose Level")

plt.tight_layout()
plt.show()
