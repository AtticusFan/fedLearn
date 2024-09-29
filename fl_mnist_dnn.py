import numpy as np
import torch as tc
print(tc.__version__)
import pandas as pd
import matplotlib.pyplot as plt
tc.cuda.is_available()
from torchvision.datasets import MNIST
from datetime import datetime
import os


##### dataset divided among clients #####
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

train = MNIST("", download=True, train=True)
test = MNIST("", download=True, train=False)
X_train = train.data
y_train = train.targets
X_test = test.data
y_test = test.targets
image = X_train[10000]
label = y_train[10000]
'''
plt.imshow(image, cmap="gray")
plt.show()
print("The number in the image is", label.numpy())
'''
y = tc.cat((y_train, y_test)).numpy()
pd.Series(y).value_counts().sort_index()
pd.Series(y).value_counts().sort_values()

X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()
print("Size of X_train is", X_train.shape)
print("Size of y_train is", y_train.shape)
print("Size of X_test is", X_test.shape)
print("Size of y_test is", y_test.shape)

##### reshape to 2-dimension
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print("Size of X_train is", X_train.shape)
print("Size of X_test is", X_test.shape)



X_train = X_train / 255
X_test = X_test / 255


n_clients = 20
alpha = 0.5
client_idcs = dirichlet_split_noniid(y_train, alpha, n_clients)
for i, idcs in enumerate(client_idcs):
    print(f"client {i+1} has {len(idcs)} samples")
# 展示不同client上的label分布
classes = train.classes
n_classes = len(classes)
plt.figure(figsize=(12, 8))
label_distribution = [[] for _ in range(n_classes)]
for c_id, idc in enumerate(client_idcs):
    for idx in idc:
        label_distribution[y_train[idx]].append(c_id)

plt.hist(label_distribution, stacked=True,
            bins=np.arange(-0.5, n_clients + 1.5, 1),
            label=classes, rwidth=0.5)
plt.xticks(np.arange(n_clients), [" %d " %
                                    c_id for c_id in range(n_clients)])
plt.xlabel("Client ID")
plt.ylabel("Number of samples")
plt.legend()
plt.title("Display Label Distribution on Different Clients")

now = datetime.now()
date_folder = now.strftime("%Y%m%d")
filename = now.strftime("%H%M.png")
folder_path = os.path.join('label-distribution', date_folder)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# 保存圖形到指定資料夾
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path)
plt.show()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




X_train = tc.from_numpy(X_train).float()
y_train = tc.from_numpy(y_train)
X_test = tc.from_numpy(X_test).float()
y_test = tc.from_numpy(y_test)




X_train = X_train.cuda()
y_train = y_train.cuda()
X_test = X_test.cuda()
y_test = y_test.cuda()

#clients_datasets = []
#for idcs in client_idcs:
   # X_client = X_train[idcs]
   # y_client = y_train[idcs]
    #clients_datasets.append(TensorDataset(X_client, y_client))




import torch.nn as nn
import torch.nn.init as init


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

    def forward(self, x):
        x = self.fc1(x)
        x = tc.relu(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = tc.relu(x)
        x = self.fc2_dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = tc.relu(x)
        x = self.fc3_dropout(x)

        x = self.fc4(x)

        return x




import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

global_model=Net().cuda()
clients_datasets = []
for idcs in client_idcs:
    X_client = X_train[idcs]
    y_client = y_train[idcs]
    clients_datasets.append(TensorDataset(X_client, y_client))




test_dataset = TensorDataset(X_test, y_test)
def train_one_client(client_dataset, global_model_state):
    global y_test
    model = Net().cuda()
    model.load_state_dict(global_model_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=3e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)




#from torch.utils.data import TensorDataset, DataLoader, random_split

#train_dataset = TensorDataset(X_train, y_train)
#test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    #calculate the accuracy of each training round
    def cal_accu(outputs,targets):
        _, predicted = tc.max(outputs,1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)


    ## epochs = 10 ##
    epochs = 10
    losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_val_loss = 0
        correct_train = 0
        correct_val = 0


        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() * inputs.size(0)

            correct_train += cal_accu(outputs,targets)*inputs.size(0)

            loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_loss /= len(client_dataset)
        losses.append(epoch_loss)
        train_accuracy = correct_train/len(client_dataset)
        train_accuracies.append(train_accuracy)


        model.eval()

        with tc.no_grad():
            for inputs, targets in test_loader:
                val_outputs = model(inputs)
                val_loss = criterion(val_outputs, targets)
                epoch_val_loss += val_loss.item() * inputs.size(0)

                correct_val += cal_accu(val_outputs,targets)*inputs.size(0)

        epoch_val_loss /= len(test_dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = correct_val/len(test_dataset)
        val_accuracies.append(val_accuracy)

        model.train()
    
        print(
            f"Epoch: {epoch + 1}/{epochs}, Training Loss: {losses[epoch]:.4f}, Validation Loss: {val_losses[epoch]:.4f},Train Accuracy: {train_accuracies[epoch]:.4f},Validation Accuracy: {val_accuracies[epoch]:.4f}"
    )


    '''
    plt.plot(range(1, epochs + 1), losses, color="red", label="train loss")
    plt.plot(range(1, epochs + 1), val_losses, color="blue", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    '''

    train_pred = []
    test_pred = []

    train_loader = DataLoader(X_client, batch_size=1024)
    test_loader = DataLoader(X_test, batch_size=1024)

    model.eval()

    with tc.no_grad():
        for inputs in train_loader:
            outputs = model(inputs)
            predictions = tc.argmax(outputs, dim=1)
            train_pred.append(predictions)

        for inputs in test_loader:
            outputs = model(inputs)
            predictions = tc.argmax(outputs, dim=1)
            test_pred.append(predictions)

    train_pred = tc.cat(train_pred)
    test_pred = tc.cat(test_pred)
    y_train = y_client.cpu()
    y_test = y_test.cpu()
    train_pred = train_pred.cpu()
    test_pred = test_pred.cpu()

    from sklearn.metrics import accuracy_score

    print(f"Training accuracy is {100 * accuracy_score(y_train, train_pred):.2f}%")
    print(f"Testing accuracy is {100 * accuracy_score(y_test, test_pred):.2f}%")




    from sklearn.metrics import precision_score, recall_score, f1_score

    print(f"Precision score is {100 * precision_score(y_test, test_pred, average='weighted'):.2f}%")
    print(f"Recall score is {100 * recall_score(y_test, test_pred, average='weighted'):.2f}%")
    print(f"F1 score is {100 * f1_score(y_test, test_pred, average='weighted'):.2f}%")


    
    import seaborn as sb
    from sklearn.metrics import confusion_matrix

    cfm = confusion_matrix(y_test, test_pred)

    #sb.heatmap(cfm / np.sum(cfm, axis=1), annot=True, fmt=".2", cmap="Blues")
    #plt.show()
    
    from PIL import Image

    image = np.asarray(Image.open("number.png").resize((28, 28)).convert("L"), dtype=np.float32)

    #plt.imshow(image)
    #plt.show()



    image = image.reshape(1, -1)
    image = image / 255
    image = scaler.transform(image)
    image = tc.from_numpy(image).float()
    image = image.cuda()




    model.eval()

    with tc.no_grad():
        prediction = tc.argmax(model(image), dim=1)

    print("The model predicts the number", np.squeeze(prediction.cpu().numpy()), "in the image")




    #tc.save(model, "model.pth")
    return model, losses

# 多次round訓練
num_rounds = 100

output_dir = 'global-model-accuracy'
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
filename = f'{timestamp}_accuracy.txt'
filepath = os.path.join(output_dir, filename)
with open(filepath, 'a') as f:
    f.write(f'Total_clients: {n_clients}\n')
    f.write(f'Communication rounds: {num_rounds}\n')
    f.write(f'\n')
        
for round in range(num_rounds):
    print(f"\nRound {round + 1}/{num_rounds}")
    
    clients_models = []
    clients_losses = []

    # 訓練每個客戶端的模型
    for i, client_dataset in enumerate(clients_datasets):
        print(f"Training Model {i+1}")
        model, losses=train_one_client(client_dataset, global_model.state_dict())
        clients_models.append(model)
        clients_losses.append(losses)

    #global_model = Net().cuda()
    global_state_dict = global_model.state_dict()

    clients_data_sizes = [len(dataset) for dataset in clients_datasets]
    total_data_size = sum(clients_data_sizes)
    #print(clients_data_sizes)
    #print(total_data_size)

    # 更新全局模型參數
    for key in global_state_dict.keys():
        global_state_dict[key] = sum(
            (len(clients_datasets[i]) / total_data_size) * clients_models[i].state_dict()[key]
            for i in range(len(clients_datasets))
        )

    global_model.load_state_dict(global_state_dict)

    # 評估全局模型
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    global_model.eval()
    correct = 0
    total = 0
    total_loss = 0.0 
    total_samples = 0
    with tc.no_grad():
        for inputs, targets in test_loader:
            outputs = global_model(inputs)
            loss = tc.nn.functional.cross_entropy(outputs, targets)  # 計算損失
            total_loss += loss.item() * targets.size(0)  # 乘上樣本數
            total_samples += targets.size(0)
            _, predicted = tc.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    #average_loss = total_loss / len(test_loader)
    average_loss = total_loss / total_samples  # 除以總樣本數
    print(f'Accuracy of the global model: {accuracy:.2f}%')
    print(f'Loss of the global model: {average_loss:.4f}')
    
    with open(filepath, 'a') as f:
        f.write(f'Round: {round + 1}\n')
        f.write(f'Accuracy: {accuracy:.2f}%\n')
        f.write(f'Loss: {average_loss:.4f}\n')
        f.write(f'\n')

print("Training complete.")
print("=======================================")

