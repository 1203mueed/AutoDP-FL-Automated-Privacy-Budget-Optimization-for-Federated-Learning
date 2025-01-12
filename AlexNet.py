import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import alexnet
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import json

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
model_dir = "alexNet"
comparison_report_path = "alexNet/comparison_report.txt"

# Custom Label Encoder for multiple columns
class MultiLabelEncoder:
    def __init__(self):
        self.label_encoders = {}

    def fit_transform(self, df, columns):
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def transform(self, df, columns):
        for col in columns:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])
        return df

# Custom dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_extensions = [".jpeg", ".jpg", ".png"]

    def __len__(self):
        return len(self.df)

    def find_image_path(self, image_id):
        for ext in self.image_extensions:
            img_path = os.path.join(self.img_dir, f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"Image file not found for ID: {image_id}")

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        img_name = self.find_image_path(image_id)
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = {
            "super_class": torch.tensor(self.df.iloc[idx]["super_class"], dtype=torch.long),
            "malignancy": torch.tensor(self.df.iloc[idx]["malignancy"], dtype=torch.long),
            "main_class_1": torch.tensor(self.df.iloc[idx]["main_class_1"], dtype=torch.long),
            "main_class_2": torch.tensor(self.df.iloc[idx]["main_class_2"], dtype=torch.long),
            "sub_class": torch.tensor(self.df.iloc[idx]["sub_class"], dtype=torch.long),
        }

        return image, labels

# AlexNet model with task-specific heads and Dropout regularization
class AlexNetModel(nn.Module):
    def __init__(self, num_classes_super_class=2, num_classes_malignancy=3, num_classes_main_class_1=7,
                 num_classes_main_class_2=15, num_classes_sub_class=33):
        super(AlexNetModel, self).__init__()

        # Load pre-trained AlexNet
        self.base_model = alexnet(pretrained=True)
        self.base_model.classifier[6] = nn.Linear(self.base_model.classifier[6].in_features, 1024)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

        # Task-specific heads
        self.fc_super_class = nn.Linear(1024, num_classes_super_class)
        self.fc_malignancy = nn.Linear(1024, num_classes_malignancy)
        self.fc_main_class_1 = nn.Linear(1024, num_classes_main_class_1)
        self.fc_main_class_2 = nn.Linear(1024, num_classes_main_class_2)
        self.fc_sub_class = nn.Linear(1024, num_classes_sub_class)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)

        # Outputs for each task
        out_super_class = self.fc_super_class(x)
        out_malignancy = self.fc_malignancy(x)
        out_main_class_1 = self.fc_main_class_1(x)
        out_main_class_2 = self.fc_main_class_2(x)
        out_sub_class = self.fc_sub_class(x)

        return {
            "super_class": out_super_class,
            "malignancy": out_malignancy,
            "main_class_1": out_main_class_1,
            "main_class_2": out_main_class_2,
            "sub_class": out_sub_class,
        }

# Evaluation function
def evaluate_model(model, loader, criterion_dict, device):
    model.eval()
    val_loss = 0
    val_true = {task: [] for task in criterion_dict.keys()}
    val_pred = {task: [] for task in criterion_dict.keys()}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}
            outputs = model(images)

            loss = sum([criterion_dict[task](outputs[task], labels[task]) for task in outputs.keys()])
            val_loss += loss.item()

            for task in outputs.keys():
                val_pred[task].extend(outputs[task].argmax(dim=1).cpu().numpy())
                val_true[task].extend(labels[task].cpu().numpy())

    classification_reports = {
        task: classification_report(val_true[task], val_pred[task], output_dict=True)
        for task in val_true
    }

    avg_metrics = {
        "precision": sum(precision_score(val_true[task], val_pred[task], average="weighted") for task in val_true) / len(val_true),
        "recall": sum(recall_score(val_true[task], val_pred[task], average="weighted") for task in val_true) / len(val_true),
        "f1_score": sum(f1_score(val_true[task], val_pred[task], average="weighted") for task in val_true) / len(val_true),
        "accuracy": sum(accuracy_score(val_true[task], val_pred[task]) for task in val_true) / len(val_true),
    }

    return val_loss / len(loader), avg_metrics, classification_reports


class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models_registry_path = os.path.join(model_dir, "models_registry.json")
        self.models = self._load_registry()

    def _load_registry(self):
        """Load the existing model registry if it exists"""
        if os.path.exists(self.models_registry_path):
            try:
                with open(self.models_registry_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_registry(self):
        """Save the current model registry"""
        with open(self.models_registry_path, 'w') as f:
            json.dump(self.models, f)

    def save_model(self, model, name):
        path = os.path.join(self.model_dir, f"{name}.pth")
        torch.save(model.state_dict(), path)
        if os.path.getsize(path) > 0:
            # Add model name and path to the models dictionary
            self.models[name] = path
            self._save_registry()  # Save the updated registry
            print(f"Saved model: {name} at {path}")
        else:
            raise IOError(f"Failed to save the model: {name}. File is empty.")

    def load_model(self, model_class, name, device, **kwargs):
        """
        Loads a saved model.
        """
        path = self.models.get(name)
        if path is None:
            raise ValueError(f"Model {name} not found in registry. Available models: {list(self.models.keys())}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        model = model_class(**kwargs).to(device)
        model.load_state_dict(torch.load(path))
        print(f"Loaded model: {name} from {path}")
        return model

    def compare_models(self, model_class, test_loader, device, label_columns, **kwargs):
        """Compare all registered models."""
        if not self.models:
            print("No models available for comparison.")
            return

        results = []
        criterion_dict = {task: nn.CrossEntropyLoss().to(device) for task in label_columns}
        best_accuracy = 0
        best_model_name = None
        best_classification_reports = None

        for name in self.models.keys():
            try:
                model = self.load_model(model_class, name, device, **kwargs)
                _, metrics, classification_reports = evaluate_model(model, test_loader, criterion_dict, device)
                results.append({"name": name, **metrics})

                # Check for the best accuracy model
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model_name = name
                    best_classification_reports = classification_reports

            except Exception as e:
                print(f"Error evaluating model {name}: {str(e)}")
                continue

        if not results:
            print("No models were successfully evaluated.")
            return

        # Save the classification report for the best model
        if best_classification_reports is not None:
            report_path = os.path.join(self.model_dir, f"{best_model_name}_classification_report.json")
            with open(report_path, "w") as f:
                json.dump(best_classification_reports, f, indent=4)
            print(f"Best model classification report saved at {report_path}")

        # Write overall results to file
        with open(comparison_report_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
            f.flush()

        print(f"Comparison report saved at {comparison_report_path}")

        # Print comparison summary
        print("\nModel Comparison Summary:")
        for result in results:
            print(f"\nModel: {result['name']}")
            for metric, value in result.items():
                if metric != 'name':
                    print(f"{metric}: {value:.4f}")


# Main training loop
def train_specific_model(train_loader, val_loader, device):
    # Hyperparameters from Optuna
    lr = 6.672307011698221e-05
    weight_decay = 4.50619125285372e-06

    # Model
    model = AlexNetModel().to(device)

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    # Loss functions for each task
    criterion_dict = {
        "super_class": nn.CrossEntropyLoss().to(device),
        "malignancy": nn.CrossEntropyLoss().to(device),
        "main_class_1": nn.CrossEntropyLoss().to(device),
        "main_class_2": nn.CrossEntropyLoss().to(device),
        "sub_class": nn.CrossEntropyLoss().to(device),
    }

    best_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    model_manager = ModelManager(model_dir)

    # Training and validation loop
    for epoch in range(1, 20):  # 20 epochs
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)

            loss = sum([criterion_dict[task](outputs[task], labels[task]) for task in outputs.keys()])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss, avg_metrics, _ = evaluate_model(model, val_loader, criterion_dict, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"F1 Score: {avg_metrics['f1_score']:.4f}, "
              f"Accuracy: {avg_metrics['accuracy']:.4f}")

        # Save best models
        for metric in best_metrics:
            if avg_metrics[metric] > best_metrics[metric]:
                best_metrics[metric] = avg_metrics[metric]
                model_manager.save_model(model, f"best_val_{metric}")


if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Define label columns
    label_columns = ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]

    # Encode labels
    label_encoder = MultiLabelEncoder()
    train_df = label_encoder.fit_transform(train_df, label_columns)
    test_df = label_encoder.transform(test_df, label_columns)

    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Dataset and augmentation
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=222),
        transforms.ColorJitter(brightness=1.2802390610488672),
        transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = SkinLesionDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, train_img_dir, transform=test_transform)
    test_dataset = SkinLesionDataset(test_df, test_img_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Train and save the specific model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_specific_model(train_loader, val_loader, device)

    # Compare models - now with the required label_columns argument
    model_manager = ModelManager(model_dir)
    model_manager.compare_models(
        model_class=AlexNetModel,
        test_loader=test_loader,
        device=device,
        label_columns=label_columns,
        num_classes_super_class=2,
        num_classes_malignancy=3,
        num_classes_main_class_1=7,
        num_classes_main_class_2=15,
        num_classes_sub_class=33
    )

