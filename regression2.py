import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from src.model import DiabetesNet
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
df = pd.read_csv("data/diabetes_prediction_dataset.csv")

# Encode categorical variables
le_gender = LabelEncoder()
le_smoke = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])

# Normalize numeric features
scaler = MinMaxScaler()
features = df.drop("diabetes", axis=1)
X_scaled = scaler.fit_transform(features)
y = df["diabetes"].values

# Load trained model
model = DiabetesNet(X_scaled.shape[1])
model.load_state_dict(torch.load("diabetes_model.pth", map_location=torch.device('cpu')))
model.eval()

# Predict probabilities
with torch.no_grad():
    inputs = torch.tensor(X_scaled, dtype=torch.float32)
    outputs = model(inputs).numpy().flatten()

# Add predictions to the DataFrame
df['diabetes_prob'] = outputs

# Plotting logistic regression lines for selected features
features_to_plot = ['age', 'bmi', 'blood_glucose_level']
sns.set(style="whitegrid")

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.regplot(x=feature, y='diabetes_prob', data=df, logistic=True, ci=None)
    plt.title(f'Logistic Regression: {feature.capitalize()} vs. Diabetes Probability')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Predicted Diabetes Probability')
    plt.tight_layout()
    plt.show()
