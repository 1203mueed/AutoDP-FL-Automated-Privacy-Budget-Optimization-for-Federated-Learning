import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import json

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
save_model_path = "inception/inception_optuna_optimized_specific_model.pth"
classification_report_path = "inception/classification_report.json"


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

# Inception model with task-specific heads and Dropout regularization
class InceptionModel(nn.Module):
    def __init__(self, num_classes_super_class=2, num_classes_malignancy=3, num_classes_main_class_1=7,
                 num_classes_main_class_2=15, num_classes_sub_class=33):
        super(InceptionModel, self).__init__()

        # Load pre-trained Inception-v3
        self.base_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        self.base_model.aux_logits = False  # Manually disable aux_logits after initialization
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1024)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.30237685159816907)  # Specific dropout value

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

    task_f1_scores = {task: f1_score(val_true[task], val_pred[task], average="weighted") for task in val_true}
    average_f1 = sum(task_f1_scores.values()) / len(task_f1_scores)

    classification_report_dict = {
        task: classification_report(val_true[task], val_pred[task], output_dict=True)
        for task in val_true
    }

    return val_loss / len(loader), average_f1, task_f1_scores, classification_report_dict

# Main training loop
def train_specific_model(train_loader, val_loader, device):
    # Hyperparameters from Optuna
    lr = 8.672307011698221e-05
    weight_decay = 4.50619125285372e-06

    # Model
    model = InceptionModel().to(device)

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    criterion_dict = {task: nn.CrossEntropyLoss().to(device) for task in label_columns}

    num_epochs = 2  # Number of epochs
    best_f1 = 0
    best_model_state = None
    best_classification_report = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images = images.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}
            optimizer.zero_grad()

            outputs = model(images)
            loss = sum([criterion_dict[task](outputs[task], labels[task]) for task in outputs.keys()])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss, val_f1, task_f1_scores, classification_report_dict = evaluate_model(model, val_loader, criterion_dict, device)
        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            best_classification_report = classification_report_dict

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    # Save the best model and classification report
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(best_model_state, save_model_path)
    print(f"Best model saved at {save_model_path}")

    with open(classification_report_path, "w") as f:
        json.dump(best_classification_report, f, indent=4)
    print(f"Classification report saved at {classification_report_path}")


    # Save the best model and classification report
    torch.save(best_model_state, save_model_path)
    print(f"Best model saved at {save_model_path}")

    with open(classification_report_path, "w") as f:
        json.dump(best_classification_report, f, indent=4)
    print(f"Classification report saved at {classification_report_path}")

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
        transforms.Resize((299, 299)),  # Adjusted for Inception-v3 input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=222),
        transforms.ColorJitter(brightness=1.2802390610488672),
        transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Adjusted for Inception-v3 input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = SkinLesionDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = SkinLesionDataset(test_df, test_img_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Train and save the specific model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_specific_model(train_loader, val_loader, device)

