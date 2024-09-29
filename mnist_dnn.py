#!/usr/bin/env python
# coding: utf-8

# In[34]:


#%pip install torch
#%pip install torchvision
#%pip install scikit-learn
#%pip install matplotlib
#%pip install seaborn
#%pip install pandas

# In[35]:


#%matplotlib inline

# In[36]:


#from google.colab import drive

#drive.mount('/content/drive')

# In[37]:


import numpy as np
import torch as tc
import pandas as pd
import matplotlib.pyplot as plt

# In[38]:


tc.cuda.is_available()

# In[39]:


from torchvision.datasets import MNIST

train = MNIST("", download=True, train=True)
test = MNIST("", download=True, train=False)

# In[40]:


X_train = train.data
y_train = train.targets
X_test = test.data
y_test = test.targets

# In[41]:


image = X_train[10000]
label = y_train[10000]

plt.imshow(image, cmap="gray")
plt.show()
print("The number in the image is", label.numpy())

# In[42]:


y = tc.cat((y_train, y_test)).numpy()

# In[43]:


pd.Series(y).value_counts().sort_index()

# In[44]:


pd.Series(y).value_counts().sort_values()

# In[45]:


X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

# In[46]:


print("Size of X_train is", X_train.shape)
print("Size of y_train is", y_train.shape)
print("Size of X_test is", X_test.shape)
print("Size of y_test is", y_test.shape)

# In[47]:


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# In[48]:


print("Size of X_train is", X_train.shape)
print("Size of X_test is", X_test.shape)

# In[49]:


X_train = X_train / 255
X_test = X_test / 255

# In[50]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# In[51]:


X_train = tc.from_numpy(X_train).float()
y_train = tc.from_numpy(y_train)
X_test = tc.from_numpy(X_test).float()
y_test = tc.from_numpy(y_test)

# In[52]:


X_train = X_train.cuda()
y_train = y_train.cuda()
X_test = X_test.cuda()
y_test = y_test.cuda()

# In[53]:


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

# In[54]:


import torch.optim as optim

model = Net()
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=9e-3, weight_decay=3e-6)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

# In[55]:


from torch.utils.data import TensorDataset, DataLoader, random_split

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

#calculate the accuracy of each training round
def cal_accu(outputs,targets):
    _, predicted = tc.max(outputs,1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


epochs = 5
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

    epoch_loss /= len(train_dataset)
    losses.append(epoch_loss)
    train_accuracy = correct_train/len(train_dataset)
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

# In[56]:


plt.plot(range(1, epochs + 1), losses, color="red", label="train loss")
plt.plot(range(1, epochs + 1), val_losses, color="blue", label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# In[57]:


train_pred = []
test_pred = []

train_loader = DataLoader(X_train, batch_size=1024)
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

# In[58]:


y_train = y_train.cpu()
y_test = y_test.cpu()
train_pred = train_pred.cpu()
test_pred = test_pred.cpu()

# In[59]:


from sklearn.metrics import accuracy_score

print(f"Training accuracy is {100 * accuracy_score(y_train, train_pred):.2f}%")
print(f"Testing accuracy is {100 * accuracy_score(y_test, test_pred):.2f}%")

# In[60]:


from sklearn.metrics import precision_score, recall_score, f1_score

print(f"Precision score is {100 * precision_score(y_test, test_pred, average='weighted'):.2f}%")
print(f"Recall score is {100 * recall_score(y_test, test_pred, average='weighted'):.2f}%")
print(f"F1 score is {100 * f1_score(y_test, test_pred, average='weighted'):.2f}%")

# In[61]:


import seaborn as sb
from sklearn.metrics import confusion_matrix

cfm = confusion_matrix(y_test, test_pred)

sb.heatmap(cfm / np.sum(cfm, axis=1), annot=True, fmt=".2", cmap="Blues")
plt.show()

# In[62]:


from PIL import Image

image = np.array(Image.open("number.png").resize((28, 28)).convert("L"))

plt.imshow(image)
plt.show()

# In[63]:


image = image.reshape(1, -1)
image = image / 255
image = scaler.transform(image)
image = tc.from_numpy(image).float()
image = image.cuda()

# In[64]:


model.eval()

with tc.no_grad():
    prediction = tc.argmax(model(image), dim=1)

print("The model predicts the number", np.squeeze(prediction.cpu().numpy()), "in the image")

# In[65]:


tc.save(model, "model.pth")

# In[66]:


#from google.colab import files

#files.download("model.pth")
