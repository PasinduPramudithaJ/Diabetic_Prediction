import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.train import DiabetesDataset

def evaluate_model(model, X_test, y_test):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = DiabetesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            y_pred.extend(preds)
            y_true.extend(batch_y.numpy())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
