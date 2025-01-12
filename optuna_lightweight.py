import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import json
import optuna

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
model_dir = "OptunaLightweightCustomConvModel"
comparison_report_path = os.path.join(model_dir, "comparison_report.txt")


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


# Lightweight Custom Conv Model
class LightweightCustomConvModel(nn.Module):
    def __init__(self, num_classes_super_class=2, num_classes_malignancy=3, num_classes_main_class_1=7,
                 num_classes_main_class_2=15, num_classes_sub_class=33, scale_factor=1.0):
        super(LightweightCustomConvModel, self).__init__()
        self.scale_factor = scale_factor
        self.base_channels = int(32 * scale_factor)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True)
        )

        self.dw_sep_conv1 = self._depthwise_separable_conv(self.base_channels, self.base_channels * 2)
        self.dw_sep_conv2 = self._depthwise_separable_conv(self.base_channels * 2, self.base_channels * 4)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc_super_class = nn.Linear(self.base_channels * 4, num_classes_super_class)
        self.fc_malignancy = nn.Linear(self.base_channels * 4, num_classes_malignancy)
        self.fc_main_class_1 = nn.Linear(self.base_channels * 4, num_classes_main_class_1)
        self.fc_main_class_2 = nn.Linear(self.base_channels * 4, num_classes_main_class_2)
        self.fc_sub_class = nn.Linear(self.base_channels * 4, num_classes_sub_class)

    def _depthwise_separable_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_sep_conv1(x)
        x = self.dw_sep_conv2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return {
            "super_class": self.fc_super_class(x),
            "malignancy": self.fc_malignancy(x),
            "main_class_1": self.fc_main_class_1(x),
            "main_class_2": self.fc_main_class_2(x),
            "sub_class": self.fc_sub_class(x),
        }


# Objective function for Optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightCustomConvModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler and criterion
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    criterion_dict = {
        "super_class": nn.CrossEntropyLoss().to(device),
        "malignancy": nn.CrossEntropyLoss().to(device),
        "main_class_1": nn.CrossEntropyLoss().to(device),
        "main_class_2": nn.CrossEntropyLoss().to(device),
        "sub_class": nn.CrossEntropyLoss().to(device),
    }

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(5):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1} Training"):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss = sum([criterion_dict[k](outputs[k], labels[k]) for k in outputs])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss = 0
        accuracy_dict = {k: 0 for k in criterion_dict.keys()}
        total_samples = {k: 0 for k in criterion_dict.keys()}

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Trial {trial.number} Epoch {epoch + 1} Validation"):
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                outputs = model(images)

                loss = sum([criterion_dict[k](outputs[k], labels[k]) for k in outputs])
                val_loss += loss.item()

                # Calculate accuracy
                for k in outputs:
                    preds = outputs[k].argmax(dim=1)
                    accuracy_dict[k] += (preds == labels[k]).sum().item()
                    total_samples[k] += labels[k].size(0)

        val_loss /= len(val_loader)
        accuracy = {k: accuracy_dict[k] / total_samples[k] for k in accuracy_dict}
        scheduler.step(val_loss)

        # Logging results
        accuracy_str = ", ".join([f"{k}_acc = {v:.4f}" for k, v in accuracy.items()])
        print(
            f"[Trial {trial.number}] Epoch {epoch + 1}: val_loss = {val_loss:.4f}, lr = {optimizer.param_groups[0]['lr']:.6f}, {accuracy_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


# Run Optuna
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
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and loaders
    train_dataset = SkinLesionDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(val_df, train_img_dir, transform=val_transform)
    global train_loader
    global val_loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Save the best hyperparameters
    best_params = study.best_params
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best hyperparameters:", best_params)

