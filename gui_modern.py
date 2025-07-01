import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import torch
from PIL import Image, ImageTk
from src.model import DiabetesNet

# Color scheme
PRIMARY_COLOR = "#3498db"
SECONDARY_COLOR = "#2980b9"
ACCENT_COLOR = "#e74c3c"
BG_COLOR = "#ecf0f1"
TEXT_COLOR = "#2c3e50"
SUCCESS_COLOR = "#2ecc71"
WARNING_COLOR = "#f39c12"

# Label encoding maps
GENDER_MAP = {'Male': 1, 'Female': 0}
SMOKING_MAP = {
    'never': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'No Info': 3
}
YES_NO_MAP = {'Yes': 1, 'No': 0}

# MinMaxScaler params
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
        root.title("Diabetes Risk Assessment")
        root.geometry("800x600")
        root.configure(bg=BG_COLOR)
        root.resizable(True, True)
        
        # Custom fonts
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=10)
        self.button_font = font.Font(family="Helvetica", size=12, weight="bold")
        
        # Header Frame
        self.header_frame = tk.Frame(root, bg=PRIMARY_COLOR)
        self.header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Header Label
        self.header_label = tk.Label(
            self.header_frame,
            text="Diabetes Risk Prediction Tool",
            font=self.title_font,
            fg="white",
            bg=PRIMARY_COLOR,
            pady=15
        )
        self.header_label.pack()
        
        # Main container
        self.main_frame = tk.Frame(root, bg=BG_COLOR)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame (inputs)
        self.input_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right frame (info/visuals)
        self.info_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Dictionary to hold entry widgets
        self.entries = {}
        
        # Create input fields
        self.create_input_fields()
        
        # Create info panel
        self.create_info_panel()
        
        # Predict button
        self.predict_button = tk.Button(
            self.input_frame,
            text="ASSESS DIABETES RISK",
            command=self.on_predict,
            bg=ACCENT_COLOR,
            fg="white",
            font=self.button_font,
            padx=20,
            pady=10,
            bd=0,
            activebackground=WARNING_COLOR,
            activeforeground="white"
        )
        self.predict_button.grid(row=9, column=0, columnspan=2, pady=20, sticky="ew")
        
        # Footer
        self.footer_label = tk.Label(
            root,
            text="© 2025 Diabetes Prediction System | For educational purposes only",
            font=("Helvetica", 8),
            fg=TEXT_COLOR,
            bg=BG_COLOR
        )
        self.footer_label.pack(side=tk.BOTTOM, pady=5)
        
        # Tooltip system
        self.tooltip = tk.Label(root, bg="yellow", relief="solid", borderwidth=1, wraplength=200)
        self.tooltip_active = False
        
    def create_input_fields(self):
        """Create all input fields with modern styling"""
        # Gender dropdown
        self.create_label("Gender:", 0)
        self.gender_var = tk.StringVar()
        self.gender_combo = ttk.Combobox(
            self.input_frame,
            textvariable=self.gender_var,
            values=list(GENDER_MAP.keys()),
            state="readonly",
            font=self.label_font
        )
        self.gender_combo.current(0)
        self.gender_combo.grid(row=0, column=1, pady=5, sticky="ew")
        self.add_tooltip(self.gender_combo, "Select your biological gender")
        
        # Age
        self.create_label("Age (years):", 1)
        self.entries["age"] = self.create_entry(1, "Enter age between 0-120")
        
        # Hypertension dropdown
        self.create_label("Hypertension:", 2)
        self.hyper_var = tk.StringVar()
        self.hyper_combo = ttk.Combobox(
            self.input_frame,
            textvariable=self.hyper_var,
            values=list(YES_NO_MAP.keys()),
            state="readonly",
            font=self.label_font
        )
        self.hyper_combo.current(1)
        self.hyper_combo.grid(row=2, column=1, pady=5, sticky="ew")
        self.add_tooltip(self.hyper_combo, "Do you have high blood pressure?")
        
        # Heart Disease dropdown
        self.create_label("Heart Disease:", 3)
        self.heart_var = tk.StringVar()
        self.heart_combo = ttk.Combobox(
            self.input_frame,
            textvariable=self.heart_var,
            values=list(YES_NO_MAP.keys()),
            state="readonly",
            font=self.label_font
        )
        self.heart_combo.current(1)
        self.heart_combo.grid(row=3, column=1, pady=5, sticky="ew")
        self.add_tooltip(self.heart_combo, "Do you have any heart conditions?")
        
        # Smoking history dropdown
        self.create_label("Smoking History:", 4)
        self.smoke_var = tk.StringVar()
        self.smoke_combo = ttk.Combobox(
            self.input_frame,
            textvariable=self.smoke_var,
            values=list(SMOKING_MAP.keys()),
            state="readonly",
            font=self.label_font
        )
        self.smoke_combo.current(0)
        self.smoke_combo.grid(row=4, column=1, pady=5, sticky="ew")
        self.add_tooltip(self.smoke_combo, "Select your smoking history")
        
        # BMI entry
        self.create_label("BMI:", 5)
        self.entries["bmi"] = self.create_entry(5, "Enter BMI between 12-60")
        
        # HbA1c level entry
        self.create_label("HbA1c Level:", 6)
        self.entries["hba1c"] = self.create_entry(6, "Enter HbA1c between 4.0-15.0")
        
        # Blood Glucose level entry
        self.create_label("Blood Glucose (mg/dL):", 7)
        self.entries["blood"] = self.create_entry(7, "Enter glucose level between 70-250")
        
        # Configure grid weights
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=2)
        
    def create_label(self, text, row):
        """Create a styled label"""
        label = tk.Label(
            self.input_frame,
            text=text,
            font=self.label_font,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            anchor="w"
        )
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        return label
        
    def create_entry(self, row, placeholder):
        """Create a styled entry field"""
        entry = ttk.Entry(
            self.input_frame,
            font=self.label_font
        )
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        entry.insert(0, placeholder)
        entry.bind("<FocusIn>", lambda e: entry.delete('0', 'end') if entry.get() == placeholder else None)
        entry.bind("<FocusOut>", lambda e: entry.insert(0, placeholder) if not entry.get() else None)
        return entry
        
    def create_info_panel(self):
        """Create the right panel with information and visuals"""
        # Risk meter frame
        self.meter_frame = tk.Frame(self.info_frame, bg=BG_COLOR)
        self.meter_frame.pack(fill=tk.X, pady=10)
        
        self.meter_label = tk.Label(
            self.meter_frame,
            text="YOUR RISK METER",
            font=self.button_font,
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        self.meter_label.pack()
        
        # Canvas for risk meter
        self.meter_canvas = tk.Canvas(
            self.meter_frame,
            width=200,
            height=30,
            bg=BG_COLOR,
            highlightthickness=0
        )
        self.meter_canvas.pack(pady=10)
        self.draw_meter(0)  # Initial state
        
        # Info text
        self.info_text = tk.Text(
            self.info_frame,
            width=40,
            height=10,
            wrap=tk.WORD,
            bg="white",
            fg=TEXT_COLOR,
            font=("Helvetica", 10),
            padx=10,
            pady=10,
            relief="flat"
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_info_text("""
        Welcome to the Diabetes Risk Assessment Tool!
        
        This tool helps you understand your risk of developing diabetes 
        based on key health indicators. Fill in your information on the 
        left and click the button to get your personalized assessment.
        
        Regular check-ups with your doctor are recommended regardless 
        of the results shown here.
        """)
        
        # Prevention tips button
        self.tips_button = tk.Button(
            self.info_frame,
            text="View Prevention Tips",
            command=self.show_prevention_tips,
            bg=SECONDARY_COLOR,
            fg="white",
            font=self.label_font,
            padx=10,
            pady=5,
            bd=0
        )
        self.tips_button.pack(pady=5)
        
    def draw_meter(self, risk_level):
        """Draw a risk meter visualization"""
        self.meter_canvas.delete("all")
        width = 200
        height = 30
        
        # Draw meter background
        self.meter_canvas.create_rectangle(0, 0, width, height, fill="#e0e0e0", outline="")
        
        # Draw colored segments
        segment_width = width / 3
        self.meter_canvas.create_rectangle(0, 0, segment_width, height, fill=SUCCESS_COLOR, outline="")
        self.meter_canvas.create_rectangle(segment_width, 0, 2*segment_width, height, fill=WARNING_COLOR, outline="")
        self.meter_canvas.create_rectangle(2*segment_width, 0, 3*segment_width, height, fill=ACCENT_COLOR, outline="")
        
        # Draw labels
        self.meter_canvas.create_text(segment_width/2, height/2, text="Low", fill="white", font=("Helvetica", 9, "bold"))
        self.meter_canvas.create_text(1.5*segment_width, height/2, text="Medium", fill="white", font=("Helvetica", 9, "bold"))
        self.meter_canvas.create_text(2.5*segment_width, height/2, text="High", fill="white", font=("Helvetica", 9, "bold"))
        
        # Draw indicator
        indicator_pos = risk_level * width
        self.meter_canvas.create_line(indicator_pos, 0, indicator_pos, height, fill="black", width=2)
        self.meter_canvas.create_polygon(
            indicator_pos-5, height,
            indicator_pos+5, height,
            indicator_pos, height+10,
            fill="black"
        )
        
    def update_info_text(self, text):
        """Update the info text area"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)
        
    def show_prevention_tips(self):
        """Show diabetes prevention tips"""
        tips = """
        DIABETES PREVENTION TIPS:
        
        1. Maintain a healthy weight
        2. Exercise regularly (30 mins/day)
        3. Eat a balanced diet (more veggies, whole grains)
        4. Limit sugary foods and drinks
        5. Don't smoke
        6. Control blood pressure
        7. Get regular check-ups
        
        Small lifestyle changes can significantly reduce your risk!
        """
        messagebox.showinfo("Prevention Tips", tips)
        
    def add_tooltip(self, widget, text):
        """Add hover tooltip to a widget"""
        widget.bind("<Enter>", lambda e: self.show_tooltip(e, text))
        widget.bind("<Leave>", lambda e: self.hide_tooltip())
        
    def show_tooltip(self, event, text):
        """Show tooltip on hover"""
        if not self.tooltip_active:
            x, y, _, _ = event.widget.bbox("insert")
            x += event.widget.winfo_rootx() + 25
            y += event.widget.winfo_rooty() + 25
            
            self.tooltip.config(text=text)
            self.tooltip.place(x=x, y=y)
            self.tooltip_active = True
            
    def hide_tooltip(self):
        """Hide the tooltip"""
        self.tooltip.place_forget()
        self.tooltip_active = False
        
    def on_predict(self):
        """Handle prediction button click"""
        try:
            # Get and validate inputs
            gender = GENDER_MAP[self.gender_var.get()]
            age = self._validate_float(self.entries["age"].get(), 0, 120, "Age")
            hypertension = YES_NO_MAP[self.hyper_var.get()]
            heart_disease = YES_NO_MAP[self.heart_var.get()]
            smoking_history = SMOKING_MAP[self.smoke_var.get()]
            bmi = self._validate_float(self.entries["bmi"].get(), 12, 60, "BMI")
            hba1c = self._validate_float(self.entries["hba1c"].get(), 4.0, 15.0, "HbA1c Level")
            glucose = self._validate_float(self.entries["blood"].get(), 70, 250, "Blood Glucose Level")

            inputs = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose]

            # Make prediction
            prediction = predict_diabetes(inputs)
            
            # Update UI with results
            if prediction:
                result = "🛑 HIGH RISK OF DIABETES"
                advice = """
                Our assessment indicates you may be at higher risk for diabetes.
                
                Recommendations:
                - Consult with your doctor
                - Consider lifestyle changes
                - Monitor blood sugar regularly
                - Get a comprehensive check-up
                """
                risk_level = 0.8  # 80% on meter
            else:
                result = "✅ LOW RISK OF DIABETES"
                advice = """
                Our assessment indicates you're at lower risk for diabetes.
                
                To maintain good health:
                - Continue healthy habits
                - Get regular check-ups
                - Monitor risk factors
                """
                risk_level = 0.2  # 20% on meter
                
            # Update the display
            self.draw_meter(risk_level)
            self.update_info_text(f"{result}\n\n{advice}")
            
            # Show popup with basic result
            messagebox.showinfo(
                "Assessment Complete",
                f"Result: {result.split(' ')[0]}\n\nSee the right panel for details and recommendations."
            )

        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def _validate_float(self, value, min_val, max_val, field_name):
        """Validate numeric input fields"""
        if not value.strip() or value in ["Enter age between 0-120", "Enter BMI between 12-60", 
                                        "Enter HbA1c between 4.0-15.0", "Enter glucose level between 70-250"]:
            raise ValueError(f"{field_name} cannot be empty.")
        try:
            val = float(value)
        except:
            raise ValueError(f"{field_name} must be a number.")

        if val < min_val or val > max_val:
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}.")
        return val

def run_app():
    root = tk.Tk()
    
    # Set window icon (replace with your own icon if available)
    try:
        img = Image.open("icon.png")  # Add your icon file
        photo = ImageTk.PhotoImage(img)
        root.tk.call('wm', 'iconphoto', root._w, photo)
    except:
        pass
        
    # Set theme
    style = ttk.Style()
    style.theme_use("clam")
    
    # Configure styles
    style.configure("TCombobox", padding=5, relief="flat", background="white")
    style.configure("TEntry", padding=5, relief="flat")
    style.map("TCombobox", fieldbackground=[("readonly", "white")])
    
    app = DiabetesPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()