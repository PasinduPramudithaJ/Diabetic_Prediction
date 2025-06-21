import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Encode categorical features
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])
    df['diabetes'] = df['diabetes'].astype(int)

    X = df.drop('diabetes', axis=1).values
    y = df['diabetes'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
