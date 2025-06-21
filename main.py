import torch
from src.preprocess import load_and_preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Path to your dataset CSV
    filepath = "data/diabetes_prediction_dataset.csv"

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Step 2: Train the model
    input_dim = X_train.shape[1]
    model = train_model(X_train, y_train, input_dim)

    # ✅ Step 3: Save the trained model to a .pth file
    model_path = "diabetes_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n🔥 Trained model saved as '{model_path}'")

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
