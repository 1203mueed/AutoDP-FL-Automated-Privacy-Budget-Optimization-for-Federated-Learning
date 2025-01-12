import os
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

# Privacy hyperparameters for DP-SGD
MAX_GRAD_NORM = 1.2
DELTA = 1e-5
BATCH_SIZE = 32
MAX_PHYSICAL_BATCH_SIZE = 32
EPOCHS = 5

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
save_model_path = "EfficientNet_Opacus_epsilon_more_info.pth"
log_file = "opacus_EfficientNet_epsilon_training_log_more_info.txt"

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

# DenseNet model with task-specific heads
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes_super_class=2, num_classes_malignancy=3, num_classes_main_class_1=7,
                 num_classes_main_class_2=15, num_classes_sub_class=33):
        super(EfficientNetModel, self).__init__()

        # Load pre-trained EfficientNet-B0
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 1024)

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

    metrics = {}
    for task in val_true.keys():
        metrics[task] = {
            "accuracy": accuracy_score(val_true[task], val_pred[task]),
            "precision": precision_score(val_true[task], val_pred[task], average="weighted"),
            "recall": recall_score(val_true[task], val_pred[task], average="weighted"),
            "f1_score": f1_score(val_true[task], val_pred[task], average="weighted"),
        }

    average_metrics = {key: sum(task[key] for task in metrics.values()) / len(metrics) for key in metrics[next(iter(metrics))]}
    return val_loss / len(loader), average_metrics, metrics

# Logging function
def log_results(epoch, train_loss, val_loss, average_metrics, task_metrics, epsilon, delta, cumulative_privacy_loss, sigma, duration):
    with open(log_file, "a") as log:
        log.write(f"Epoch {epoch + 1}\n")
        log.write(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
        log.write(f"Avg Metrics: {average_metrics}\n")
        log.write(f"Task Metrics: {task_metrics}\n")
        log.write(f"Epsilon: {epsilon:.2f}, Delta: {delta:.2e}, Cumulative Privacy Loss: {cumulative_privacy_loss:.2f}, Sigma: {sigma:.2f}\n")
        log.write(f"Training Duration: {duration:.2f}s\n")
        log.write(f"Privacy-Performance Tradeoff: {epsilon / average_metrics['f1_score']:.2f}\n\n")

# Main training loop
def train_with_opacus(train_loader, val_loader, device, epsilon_values):
    for epsilon in epsilon_values:
        # Initialize model
        model = EfficientNetModel().to(device)

        # Validate model for Opacus compatibility
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print(f"Errors found in the model: {errors}")

        # Optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=8.672307011698221e-05)
        criterion_dict = {task: nn.CrossEntropyLoss().to(device) for task in ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]}

        # Attach Opacus Privacy Engine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=EPOCHS,
            target_epsilon=epsilon,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        sigma = optimizer.noise_multiplier

        best_f1 = 0
        for epoch in range(EPOCHS):
            start_time = time.time()
            model.train()
            train_loss = 0

            with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, optimizer=optimizer) as memory_safe_data_loader:
                for images, labels in tqdm(memory_safe_data_loader, desc=f"Training Epoch {epoch + 1}"):
                    images = images.to(device)
                    labels = {key: value.to(device) for key, value in labels.items()}
                    optimizer.zero_grad()

                    outputs = model(images)
                    loss = sum([criterion_dict[task](outputs[task], labels[task]) for task in outputs.keys()])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

            duration = time.time() - start_time
            val_loss, average_metrics, task_metrics = evaluate_model(model, val_loader, criterion_dict, device)
            epsilon = privacy_engine.get_epsilon(DELTA)

            log_results(epoch, train_loss, val_loss, average_metrics, task_metrics, epsilon, DELTA, epsilon * DELTA, sigma, duration)

            if average_metrics["f1_score"] > best_f1:
                best_f1 = average_metrics["f1_score"]
                torch.save(model.state_dict(), save_model_path)
                print(f"New best model saved with F1 Score: {best_f1:.4f}")
if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Encode labels
    label_columns = ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
    label_encoder = MultiLabelEncoder()
    train_df = label_encoder.fit_transform(train_df, label_columns)
    test_df = label_encoder.transform(test_df, label_columns)

    # Dataset and augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = SkinLesionDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(test_df, test_img_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment with different epsilon values for DP-SGD
    epsilon_values = [0.01, 0.05,  0.1, 0.2,  0.3,0.4,  0.5,0.6,  0.7, 0.8,0.9,  1, 2, 3, 4,5,  5.0, 10.0, 20, 30, 40, 50.0, 100]

    # Train the model with Opacus
    train_with_opacus(train_loader, val_loader, device, epsilon_values)
