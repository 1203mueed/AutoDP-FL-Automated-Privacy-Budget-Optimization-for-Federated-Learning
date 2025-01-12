
# AutoDP-FL: Automated Privacy Budget Optimization for Federated Learning

**Description:**  
AutoDP-FL automates privacy budget (ε) tuning for federated learning. It uses Opacus for Differential Privacy (DP) in PyTorch, Flower for federated training, and popular architectures (e.g., AlexNet, ResNet). This ensures privacy-preserving multi-task classification with minimal utility loss.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Differential Privacy with Opacus](#differential-privacy-with-opacus)
5. [Federated Learning with Flower](#federated-learning-with-flower)
6. [Optimal Epsilon Discovery](#optimal-epsilon-discovery)
7. [Dataset](#dataset)
8. [Contributing](#contributing)

---

## Key Features
- **Automated Epsilon Tuning**: AutoDP-FL finds an optimal privacy budget to balance model performance and privacy.
- **Federated Learning**: Uses Flower to simulate or run multiple clients.
- **Differential Privacy (DP)**: Opacus integration for privacy-preserving training in PyTorch.
- **Multi-Task Classification**: Predict multiple aspects (e.g., super_class, malignancy) simultaneously.
- **Popular Architectures**: AlexNet, ResNet, DenseNet, EfficientNet, MobileNet, VGG16, ShuffleNetV2, RegNetX, Inception, Vision Transformer, etc.
- **Custom Models**: Lightweight and custom CNN examples included.
- **Hyperparameter Optimization**: Example with Optuna for tuning learning rate, weight decay, etc.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YourUsername/AutoDP-FL-Automated-Privacy-Budget-Optimization-for-Federated-Learning.git
   cd AutoDP-FL-Automated-Privacy-Budget-Optimization-for-Federated-Learning
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   # or
   venv\Scriptsctivate       # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional**: Check PyTorch GPU availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   If `True`, your environment can run scripts on the GPU.

---

## Usage

### Local Training
Pick a model script (e.g., `AlexNet.py`) and run:
```bash
python AlexNet.py
```
This trains AlexNet for multi-task classification and saves results in a corresponding results folder (e.g., `AlexNet_results/`).

### Differential Privacy with Opacus
For instance, `Opacus_EfficientNet.py`:
```bash
python Opacus_EfficientNet.py
```
Attaches a PrivacyEngine, enabling differential privacy with a chosen or auto-optimized ε.

### Federated Learning with Flower
Use `federated_with_saving.py` to simulate multiple federated clients:
```bash
python federated_with_saving.py
```
This spawns local clients, each training a model in a privacy-preserving manner (if configured), and aggregates them with Flower’s FedAvg strategy.

### Hyperparameter Tuning (Optuna)
See `optuna_lightweight.py`:
```bash
python optuna_lightweight.py
```
Tries different hyperparameters (e.g., learning rate, weight decay) and picks the best via Optuna.

---

## Differential Privacy with Opacus
Opacus integrates with PyTorch to protect user data by:
- Bounding per-sample gradients (`max_grad_norm`).
- Adding noise to updates based on the chosen ε and δ.
- Tracking total privacy expenditure as training progresses.

---

## Federated Learning with Flower
Flower orchestrates federated rounds by:
- Spawning multiple clients (or connecting real client machines).
- Having each client train locally, possibly under DP.
- Aggregating updates with FedAvg or another strategy on the server side.

---

## Optimal Epsilon Discovery
1. **Train with Multiple Epsilons**: The code tries a range of ε values (e.g., 0.1, 1.0, 5.0).
2. **Monitor Performance**: Logs accuracy, F1-score, etc., for each ε.
3. **Select the Best**: Chooses the ε that yields the best trade-off between model utility and privacy.

---

## Dataset
This project uses the [DERM12345 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DAXZ7P):  
**DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 40 Subclasses**.

---

## Contributing
1. **Fork** the repo and **clone** it locally.
2. **Create a feature branch** for your additions or bug fixes.
3. **Push** your branch to GitHub and **open a Pull Request**.

---

**Happy Federated Learning with Privacy!**
