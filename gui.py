import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
from src.model import DiabetesNet

# Label encoding maps
GENDER_MAP = {'Male': 1, 'Female': 0}
SMOKING_MAP = {
    'never': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'No Info': 3
}
YES_NO_MAP = {'Yes': 1, 'No': 0}

# MinMaxScaler params (replace with your actual min/max from training data)
SCALER_MIN = np.array([0, 0, 0, 0, 0, 12, 4.0, 70])
SCALER_MAX = np.array([1, 120, 1, 1, 3, 60, 15.0, 250])

def minmax_scale(x):
    x = (x - SCALER_MIN) / (SCALER_MAX - SCALER_MIN)
    return np.clip(x, 0, 1)

def load_model(input_dim, model_path="diabetes_model.pth"):
    model = DiabetesNet(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_diabetes(inputs):
    inputs = np.array(inputs, dtype=float)
    inputs_scaled = minmax_scale(inputs)
    model = load_model(len(inputs_scaled))
    with torch.no_grad():
        input_tensor = torch.tensor([inputs_scaled], dtype=torch.float32)
        output = model(input_tensor)
        pred = (output.item() > 0.5)
    return pred

class DiabetesPredictorApp:
    def __init__(self, root):
        self.root = root
        root.title("Diabetes Prediction")
        root.geometry("400x400")
        root.resizable(False, False)
        padding = {'padx': 10, 'pady': 6}

        # Dictionary to hold entry widgets for numeric inputs
        self.entries = {}

        # Gender dropdown
        ttk.Label(root, text="Gender:").grid(row=0, column=0, sticky=tk.W, **padding)
        self.gender_var = tk.StringVar()
        self.gender_combo = ttk.Combobox(root, textvariable=self.gender_var, values=list(GENDER_MAP.keys()), state="readonly")
        self.gender_combo.current(0)
        self.gender_combo.grid(row=0, column=1, **padding)

        # Age
        self._add_label_entry("Age (0-120):", 1)

        # Hypertension dropdown
        ttk.Label(root, text="Hypertension:").grid(row=2, column=0, sticky=tk.W, **padding)
        self.hyper_var = tk.StringVar()
        self.hyper_combo = ttk.Combobox(root, textvariable=self.hyper_var, values=list(YES_NO_MAP.keys()), state="readonly")
        self.hyper_combo.current(1)
        self.hyper_combo.grid(row=2, column=1, **padding)

        # Heart Disease dropdown
        ttk.Label(root, text="Heart Disease:").grid(row=3, column=0, sticky=tk.W, **padding)
        self.heart_var = tk.StringVar()
        self.heart_combo = ttk.Combobox(root, textvariable=self.heart_var, values=list(YES_NO_MAP.keys()), state="readonly")
        self.heart_combo.current(1)
        self.heart_combo.grid(row=3, column=1, **padding)

        # Smoking history dropdown
        ttk.Label(root, text="Smoking History:").grid(row=4, column=0, sticky=tk.W, **padding)
        self.smoke_var = tk.StringVar()
        self.smoke_combo = ttk.Combobox(root, textvariable=self.smoke_var, values=list(SMOKING_MAP.keys()), state="readonly")
        self.smoke_combo.current(0)
        self.smoke_combo.grid(row=4, column=1, **padding)

        # BMI entry
        self._add_label_entry("BMI (12-60):", 5)

        # HbA1c level entry
        self._add_label_entry("HbA1c Level (4.0-15.0):", 6)

        # Blood Glucose level entry
        self._add_label_entry("Blood Glucose Level (70-250):", 7)

        # Predict button
        self.predict_button = ttk.Button(root, text="Predict Diabetes", command=self.on_predict)
        self.predict_button.grid(row=8, column=0, columnspan=2, pady=15)

    def _add_label_entry(self, label_text, row):
        ttk.Label(self.root, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=10, pady=6)
        entry = ttk.Entry(self.root)
        entry.grid(row=row, column=1, padx=10, pady=6)
        # Save entry by key without punctuation and lowercase (e.g. 'age', 'bmi', etc.)
        key = label_text.split()[0].lower()
        self.entries[key] = entry

    def on_predict(self):
        try:
            gender = GENDER_MAP[self.gender_var.get()]
            age = self._validate_float(self.entries["age"].get(), 0, 120, "Age")
            hypertension = YES_NO_MAP[self.hyper_var.get()]
            heart_disease = YES_NO_MAP[self.heart_var.get()]
            smoking_history = SMOKING_MAP[self.smoke_var.get()]
            bmi = self._validate_float(self.entries["bmi"].get(), 12, 60, "BMI")
            hba1c = self._validate_float(self.entries["hba1c"].get(), 4.0, 15.0, "HbA1c Level")
            glucose = self._validate_float(self.entries["blood"].get(), 70, 250, "Blood Glucose Level")

            inputs = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]

            prediction = predict_diabetes(inputs)
            result = "🩸 Diabetic" if prediction else "✅ Not Diabetic"
            messagebox.showinfo("Prediction Result", f"Prediction: {result}")

        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error:\n{e}")

    def _validate_float(self, value, min_val, max_val, field_name):
        if not value.strip():
            raise ValueError(f"{field_name} cannot be empty.")
        try:
            val = float(value)
        except:
            raise ValueError(f"{field_name} must be a number.")

        if val < 0:
            raise ValueError(f"{field_name} cannot be negative.")
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}.")
        return val

def run_app():
    root = tk.Tk()
    app = DiabetesPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()
