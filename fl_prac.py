import numpy as np
import torch as tc
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
import random
from datetime import datetime

# Set random seed for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed(seed)
        tc.cuda.manual_seed_all(seed)
    tc.backends.cudnn.deterministic = True
    tc.backends.cudnn.benchmark = False

set_random_seed(42)

# Load MNIST dataset
train = MNIST("", download=True, train=True)
test = MNIST("", download=True, train=False)

X_train = train.data.numpy()
y_train = train.targets.numpy()
X_test = test.data.numpy()
y_test = test.targets.numpy()

# Reshape and normalize the training and testing data
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255

# Use StandardScaler for normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train = tc.from_numpy(X_train).float()
y_train = tc.from_numpy(y_train)
X_test = tc.from_numpy(X_test).float()
y_test = tc.from_numpy(y_test)

# Move tensors to GPU if available
X_train = X_train.cuda()
y_train = y_train.cuda()
X_test = X_test.cuda()
y_test = y_test.cuda()

# Split dataset into clients using Dirichlet distribution
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i].append(idcs)
    
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

n_clients = 40
alpha = 0.1
opt ="SGD"
#opt = 'Adam'
client_idcs = dirichlet_split_noniid(y_train.cpu().numpy(), alpha, n_clients)

# Calculate class distribution for each client
def calculate_client_class_distribution(y_train, client_idcs, n_classes=10):
    client_distribution = []
    for idcs in client_idcs:
        client_labels = y_train[idcs].cpu().numpy()
        class_count = np.bincount(client_labels, minlength=n_classes)
        client_distribution.append(class_count)
    return client_distribution

client_distribution = calculate_client_class_distribution(y_train, client_idcs)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
df = pd.DataFrame(client_distribution, columns=[f'Class_{i}' for i in range(10)])
df.index = [f'Client_{i+1}' for i in range(len(client_distribution))]
filename = f"{current_time}_{alpha}_client_data_distribution.csv"
df.to_csv(filename, index=True)
print(f"Client data distribution saved to {filename}")

# Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc1_dropout = nn.Dropout(0.2)
        init.kaiming_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2_dropout = nn.Dropout(0.1)
        init.kaiming_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3_dropout = nn.Dropout(0.1)
        init.kaiming_normal_(self.fc3.weight)
        
        self.fc4 = nn.Linear(128, 10)
        init.xavier_normal_(self.fc4.weight)

        self.group_norm2 = nn.GroupNorm(num_groups=16, num_channels=256)
        self.group_norm3 = nn.GroupNorm(num_groups=16, num_channels=128)

    def forward(self, x):
        x = self.fc1(x)
        x = tc.relu(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        else:
            x = self.group_norm2(x)
        x = tc.relu(x)
        x = self.fc2_dropout(x)

        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        else:
             x = self.group_norm3(x)
        x = tc.relu(x)
        x = self.fc3_dropout(x)

        x = self.fc4(x)
        return x

# Prepare client datasets
clients_datasets = []
for idcs in client_idcs:
    X_client = X_train[idcs]
    y_client = y_train[idcs]
    clients_datasets.append(TensorDataset(X_client, y_client))

# Function to train a client model
def train_one_client(client_dataset, model):
    criterion = nn.CrossEntropyLoss()
    if(opt == "Adam" ):
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-6)
    else:
        # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=3e-6)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=3e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    
    train_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    def cal_accu(outputs, targets):
        _, predicted = tc.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)

    epochs = 6
    losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct_train = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() * inputs.size(0)
            correct_train += cal_accu(outputs, targets) * inputs.size(0)
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_accuracy = correct_train / len(client_dataset)
        train_accuracies.append(train_accuracy)
        losses.append(epoch_loss / len(client_dataset))

    print(f"Final Client Training Accuracy: {train_accuracies[-1]:.4f}")

    return model, losses

# Federated learning rounds

n_rounds = 100
k = 10  # Number of selected clients per round
n_iterations = 5  # Number of iterations per round
global_model = Net().cuda()
clients_models = []
accuracies = []

for round_num in range(n_rounds):
    print(f"\nRound {round_num + 1}")

    round_accuracies = []

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1} in Round {round_num + 1}")
        
        clients_models = []
        clients_losses = []

        selected_clients = random.sample(range(n_clients), k)
        selected_clients_data_sizes = [len(clients_datasets[i]) for i in selected_clients]

        # 在這裡對每個 client 訓練多次
        iteration_accuracies = []
        for i in selected_clients:
            print(f"\nTraining Model for Client {i + 1}")
            client_dataset = clients_datasets[i]
            
            # 重新初始化 client model 並載入 global model 的權重
            local_model = Net().cuda()
            local_model.load_state_dict(global_model.state_dict())
            
            # 執行多次訓練
            model, losses = train_one_client(client_dataset, local_model)
            clients_models.append(model)
            clients_losses.append(losses)
            
            # 計算這次 iteration 的準確率
            correct = 0
            total = 0
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
            with tc.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    _, predicted = tc.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            iteration_accuracies.append(accuracy)

        # 計算多次迭代後的平均準確率
        avg_accuracy = sum(iteration_accuracies) / len(iteration_accuracies)
        round_accuracies.append(avg_accuracy)
        print(f'Average Client Model Accuracy in Round {round_num + 1}: {avg_accuracy:.2f}%')

        # Federated Averaging：對選定的 client 進行模型聚合
        global_state_dict = global_model.state_dict()
        total_data_size = sum(selected_clients_data_sizes)

        for key in global_state_dict.keys():
            global_state_dict[key] = sum(
                (selected_clients_data_sizes[i] / total_data_size) * clients_models[i].state_dict()[key]
                for i in range(k)
            )

        global_model.load_state_dict(global_state_dict)

    # 評估 global model 的準確率
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    correct = 0
    total = 0
    with tc.no_grad():
        for inputs, targets in test_loader:
            outputs = global_model(inputs)
            _, predicted = tc.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    global_accuracy = 100 * correct / total
    accuracies.append(global_accuracy)
    print(f'Global Model Accuracy after Round {round_num + 1}: {global_accuracy:.2f}%')


# Save accuracies to a CSV file
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#filename = f"{current_time}_accuracy.csv"
filename = f"{alpha}_n{n_clients}_k{k}_{opt}_accuracy.csv"
df = pd.DataFrame(accuracies, columns=['Accuracy'])
df.to_csv(filename, index=False)

print(f"Accuracies saved to {filename}")
