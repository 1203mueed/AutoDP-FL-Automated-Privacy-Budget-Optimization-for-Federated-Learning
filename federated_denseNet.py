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
BATCH_SIZE = 32
MAX_PHYSICAL_BATCH_SIZE = 32
EPOCHS = 5

# Paths
train_csv_path = "train.csv"
train_img_dir = "train_images"
test_csv_path = "test.csv"
test_img_dir = "test_images"
save_model_path = "densenet_Opacus_epsilon_more_info.pth"
log_file = "opacus_densenet_epsilon_training_log_more_info.txt"

# Federated learning parameters
NUM_CLIENTS = 3  # Number of clients
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
w1, w2 = 0.7, 0.3  # Weights for optimal epsilon calculation

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

# # DenseNet model with task-specific heads
# class DenseNet121(nn.Module):
#     def __init__(self, num_classes_super_class=2, num_classes_malignancy=3, num_classes_main_class_1=7,
#                  num_classes_main_class_2=15, num_classes_sub_class=33):
#         super(DenseNet121, self).__init__()
#
#         # Load pre-trained DenseNet-121
#         self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
#         self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 1024)
#
#         # Task-specific heads
#         self.fc_super_class = nn.Linear(1024, num_classes_super_class)
#         self.fc_malignancy = nn.Linear(1024, num_classes_malignancy)
#         self.fc_main_class_1 = nn.Linear(1024, num_classes_main_class_1)
#         self.fc_main_class_2 = nn.Linear(1024, num_classes_main_class_2)
#         self.fc_sub_class = nn.Linear(1024, num_classes_sub_class)
#
#     def forward(self, x):
#         x = self.base_model(x)
#
#         # Outputs for each task
#         out_super_class = self.fc_super_class(x)
#         out_malignancy = self.fc_malignancy(x)
#         out_main_class_1 = self.fc_main_class_1(x)
#         out_main_class_2 = self.fc_main_class_2(x)
#         out_sub_class = self.fc_sub_class(x)
#
#         return {
#             "super_class": out_super_class,
#             "malignancy": out_malignancy,
#             "main_class_1": out_main_class_1,
#             "main_class_2": out_main_class_2,
#             "sub_class": out_sub_class,
#         }

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
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

    def get_parameters(self, config=None):  # Accept the optional 'config' argument
        """
        Return the model's parameters as a list of NumPy arrays.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"Starting training on client with epsilon {self.epsilon}")
        self.set_parameters(parameters)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS} on client...")
            self.model.train()
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

                    if i % 10 == 0:
                        print(f"  Batch {i}: Loss = {loss.item():.4f}")

        print(f"Training completed for epsilon {self.epsilon}.")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("Evaluating model...")
        self.set_parameters(parameters)
        self.model.eval()

        val_loss, f1_scores = 0, []
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader, 1):
                images = images.to(self.device)
                labels = {key: value.to(self.device) for key, value in labels.items()}

                outputs = self.model(images)
                loss = sum(
                    [self.criterion_dict[task](outputs[task], labels[task]) for task in outputs.keys()]
                )
                val_loss += loss.item()

                for task in outputs.keys():
                    preds = outputs[task].argmax(dim=1).cpu().numpy()
                    truths = labels[task].cpu().numpy()
                    f1_scores.append(f1_score(truths, preds, average="weighted"))

                if i % 10 == 0:
                    print(f"  Batch {i}: Loss = {loss.item():.4f}")

        print("Evaluation completed.")
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        return float(val_loss / len(self.val_loader)), len(self.val_loader.dataset), {"f1_score": avg_f1_score}


# Federated learning setup
# Federated learning setup
def federated(train_loaders, val_loaders, epsilon_values, num_clients, device):
    """
    Sets up and runs federated learning with differential privacy.

    Args:
        train_loaders (list): List of DataLoader objects for each client's training data.
        val_loaders (list): List of DataLoader objects for each client's validation data.
        epsilon_values (list): List of privacy budget (epsilon) values for each client.
        num_clients (int): Number of clients participating in federated learning.
        device (torch.device): Device to use for computation (CPU or GPU).
    """

    def client_fn(client_id: str):
        """
        Create a Flower client based on the client ID.

        Args:
            client_id (str): Client ID as a string (e.g., "0", "1", ...).

        Returns:
            FlowerClient: The Flower federated learning client.
        """
        client_index = int(client_id)  # Convert client ID to integer
        train_loader = train_loaders[client_index]
        val_loader = val_loaders[client_index]
        epsilon = EPSILON_VALUES[client_index]
        return FlowerClient(train_loader, val_loader, device, epsilon)

    # Set up the federated learning strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )




if __name__ == "__main__":
    print("Starting Federated Learning Script...")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess the data
    print("Loading training and testing data...")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Encode labels
    print("Encoding labels...")
    encoder = MultiLabelEncoder()
    columns_to_encode = ["super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"]
    train_df = encoder.fit_transform(train_df, columns_to_encode)
    test_df = encoder.transform(test_df, columns_to_encode)

    # Define data transformations
    print("Defining data transformations...")
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

    # Prepare datasets
    print("Preparing datasets...")
    full_dataset = SkinLesionDataset(train_df, train_img_dir, transform=train_transform)

    # Splitting datasets among clients
    num_samples = len(full_dataset)
    split_sizes = [num_samples // NUM_CLIENTS] * NUM_CLIENTS
    split_sizes[-1] += num_samples % NUM_CLIENTS
    client_data_splits = torch.utils.data.random_split(full_dataset, split_sizes)
    print(f"Dataset split into {NUM_CLIENTS} clients.")

    # Create DataLoaders
    train_loaders = [
        DataLoader(client_split, batch_size=BATCH_SIZE, shuffle=True)
        for client_split in client_data_splits
    ]
    val_loaders = [
        DataLoader(client_split, batch_size=BATCH_SIZE, shuffle=False)
        for client_split in client_data_splits
    ]

    print("DataLoaders created. Starting federated learning...")

    # Run federated learning
    federated(train_loaders, val_loaders, EPSILON_VALUES, NUM_CLIENTS, device)
    print("Federated learning completed.")

