import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

file_path = './checkpoints/'
# Load the extracted features and original data
train_features_rf = torch.load(file_path+'train_features_rf.pt')
train_features_svm = torch.load(file_path+'train_features_svm.pt')
train_labels = torch.load(file_path+'train_labels.pt')

test_features_rf = torch.load(file_path+'test_features_rf.pt')
test_features_svm = torch.load(file_path+'test_features_svm.pt')
test_labels = torch.load(file_path+'test_labels.pt')

train_data = torch.load('train_data.pt')
test_data = torch.load('test_data.pt')

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x1, x2):
        x1 = torch.relu(self.fc1(x1))
        x1 = torch.relu(self.fc2(x1))

        x2 = torch.relu(self.fc1(x2))
        x2 = torch.relu(self.fc2(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.fc3(x)
        return x

# Instantiate the neural network model
net = Net()

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Create a DataLoader for training data
train_dataset = TensorDataset(train_features_rf, train_features_svm, train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the neural network model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        features_rf, features_svm, data, labels = data

        optimizer.zero_grad()

        outputs = net(features_rf, features_svm, data)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

print('Finished training')

# Evaluate the trained neural network model
test_dataset = TensorDataset(test_features_rf, test_features_svm, test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        features_rf, features_svm, data, labels = data
        outputs = net(features_rf, features_svm, data)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels.float().view(-1, 1)).sum().item()

accuracy = correct / total
print('Accuracy on test set: %.3f' % accuracy)
