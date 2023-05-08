import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

from utils.feature_select import process_data


def evaluate():
    # Evaluate model
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        for i in range(len(X_test)):
            inputs = torch.from_numpy(X_test[i]).float().unsqueeze(0).to(device)
            labels = torch.from_numpy(np.array(y_train[i])).unsqueeze(0).to(device)
            outputs = net(inputs)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(X_train)}")
        print(f"Accuracy: {correct / total}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)

        x = x.view(-1)  # adjust the shape of the tensor
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


# Load data
data = pd.read_csv("data/train.csv")
data = process_data(data)

# Preprocess data
X = data.drop("is_BPH", axis=1)
y = data["is_BPH"]
X = np.array(X)
y = np.array(y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize training and testing input
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
net = Net()
net.to(device)
# criterion = nn.BCEWithLogitsLoss()
# criterion = torch.nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

for epoch in range(200):
    running_loss = 0.0
    for i in range(len(X_train)):
        inputs = torch.from_numpy(X_train[i]).float().unsqueeze(0).to(device)
        labels = torch.from_numpy(np.array(y_train[i])).unsqueeze(0).to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1} loss: {running_loss / len(X_train)}")
    evaluate()
