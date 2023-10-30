Hyperparameter sweeping can be done in PyTorch using libraries like Optuna, Ray Tune, or even manually by iterating over different hyperparameter combinations. Below is a simple example using Optuna to find the best learning rate and batch size for a simple neural network on the MNIST dataset.

### Install Optuna
First, install Optuna:
```bash
pip install optuna
```

### Python Code
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import optuna

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Define objective function for Optuna
def objective(trial):
    # Hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Data loading
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(3):  # Just 3 epochs for demonstration
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Return the final loss
    return loss.item()

# Create Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Results
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```

In this example, Optuna will try 50 different combinations of learning rates and batch sizes, aiming to minimize the loss returned by the `objective` function. The best hyperparameters will be printed at the end.