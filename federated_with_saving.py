import os
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
import flwr as fl
from models import DenseNet121
from flwr.common import Context

# Privacy hyperparameters
MAX_GRAD_NORM = 1.2
DELTA = 1e-5
BATCH_SIZE = 16
MAX_PHYSICAL_BATCH_SIZE = 16
EPOCHS = 3

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
save_model_path = "densenet_Opacus_epsilon_more_info.pth"
log_file = "opacus_densenet_epsilon_training_log_more_info.txt"
round_accuracy_log_file = "federated_round_accuracies.txt"

# Federated learning parameters
NUM_CLIENTS = 3  # Number of clients
EPSILON_VALUES = [ 0.1,0.4, 11.5, 10.0]
w1, w2 = 0.5, 0.5  # Weights for optimal epsilon calculation

# Ensure log files exist
open(log_file, "w").close()
open(round_accuracy_log_file, "w").close()


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

# Federated client implementation
# Utility function for logging
def log_message(message, file_path, mode="a"):
    """Log a message to both the console and a specified log file."""
    print(message)  # Print to console
    with open(file_path, mode) as f:
        f.write(message + "\n")  # Append message to log file

# Federated client implementation
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, device, epsilon):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epsilon = epsilon

        # Initialize model
        self.model = DenseNet121().to(device)
        self.model = ModuleValidator.fix(self.model)

        # Privacy engine setup
        self.privacy_engine = PrivacyEngine()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8.672307011698221e-05)
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=EPOCHS,
            target_epsilon=self.epsilon,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        self.criterion_dict = {
            task: nn.CrossEntropyLoss().to(device)
            for task in ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
        }

    def get_parameters(self, config=None):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        log_message(f"Starting training on client with epsilon {self.epsilon}", log_file)
        self.set_parameters(parameters)

        for epoch in range(EPOCHS):
            log_message(f"Epoch {epoch + 1}/{EPOCHS} on client...", log_file)
            self.model.train()
            total_train_loss = 0
            total_batches = 0

            with BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=self.optimizer
            ) as memory_safe_loader:
                for i, (images, labels) in enumerate(memory_safe_loader, 1):
                    images = images.to(self.device)
                    labels = {key: value.to(self.device) for key, value in labels.items()}

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = sum(
                        [self.criterion_dict[task](outputs[task], labels[task]) for task in outputs.keys()]
                    )
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item()
                    total_batches += 1

                    if i % 10 == 0:
                        print(f"  Batch {i}: Loss = {loss.item():.4f}")

            avg_train_loss = total_train_loss / total_batches
            log_message(f"  Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}", log_file)

            # Perform validation after each epoch
            self.model.eval()
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = {key: value.to(self.device) for key, value in labels.items()}

                    outputs = self.model(images)
                    for task in outputs.keys():
                        preds = outputs[task].argmax(dim=1)
                        total_correct += (preds == labels[task]).sum().item()
                        total_samples += labels[task].size(0)

            val_accuracy = total_correct / total_samples
            log_message(f"  Epoch {epoch + 1} Validation Accuracy: {val_accuracy:.4f}", log_file)

        log_message(f"Training completed for epsilon {self.epsilon}.", log_file)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        log_message("Evaluating model...", log_file)
        self.set_parameters(parameters)
        self.model.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = {key: value.to(self.device) for key, value in labels.items()}

                outputs = self.model(images)
                for task in outputs.keys():
                    preds = outputs[task].argmax(dim=1)
                    total_correct += (preds == labels[task]).sum().item()
                    total_samples += labels[task].size(0)

        accuracy = total_correct / total_samples
        log_message(f"Evaluation accuracy: {accuracy:.4f}", log_file)
        return 0.0, total_samples, {"accuracy": accuracy}

# Federated learning setup
def federated(train_loaders, val_loaders, epsilon_values, num_clients, device):
    def client_fn(client_id: str):
        client_index = int(client_id)
        train_loader = train_loaders[client_index]
        val_loader = val_loaders[client_index]
        epsilon = EPSILON_VALUES[client_index]
        return FlowerClient(train_loader, val_loader, device, epsilon)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    global_model = DenseNet121().to(device)

    # Simulation with logging per round
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Log metrics for each round
    with open(round_accuracy_log_file, "a") as f:
        for server_round, metrics in enumerate(history.metrics_centralized.get("accuracy", []), start=1):
            log_message(f"Round {server_round}: Accuracy = {metrics:.4f}", round_accuracy_log_file)
            f.write(log_message)

    # Save the final global model
    log_message(f"Saving global model to {save_model_path}...", log_file)
    torch.save(global_model.state_dict(), save_model_path)
    log_message("Model saved successfully.", log_file)




if __name__ == "__main__":
    print("Starting Federated Learning Script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    encoder = MultiLabelEncoder()
    columns_to_encode = ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
    train_df = encoder.fit_transform(train_df, columns_to_encode)
    test_df = encoder.transform(test_df, columns_to_encode)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = SkinLesionDataset(train_df, train_img_dir, transform=train_transform)

    num_samples = len(full_dataset)
    split_sizes = [num_samples // NUM_CLIENTS] * NUM_CLIENTS
    split_sizes[-1] += num_samples % NUM_CLIENTS
    client_data_splits = torch.utils.data.random_split(full_dataset, split_sizes)
    print(f"Dataset split into {NUM_CLIENTS} clients.")

    train_loaders = [
        DataLoader(client_split, batch_size=BATCH_SIZE, shuffle=True)
        for client_split in client_data_splits
    ]
    val_loaders = [
        DataLoader(client_split, batch_size=BATCH_SIZE, shuffle=False)
        for client_split in client_data_splits
    ]

    print("DataLoaders created. Starting federated learning...")
    federated(train_loaders, val_loaders, EPSILON_VALUES, NUM_CLIENTS, device)
    print("Federated learning completed.")