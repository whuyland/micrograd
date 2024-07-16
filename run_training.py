import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from micrograd.engine import Value
from micrograd.nn import MLP

# Generate the dataset
X, y = make_blobs(n_samples=100, n_features=2)
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
plt.show()


def plot_result(generator, what):
    # Visualization
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    Z = generator(Xmesh)

    # Plot the decision boundary
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(what)
    plt.show()


def train_by_torch():
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define the neural network model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(2, 4)
            self.fc2 = nn.Linear(4, 3)  # Assuming 3 classes in the blobs dataset

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def generator(Xmesh):
        # Convert the meshgrid to a PyTorch tensor
        Xmesh_tensor = torch.tensor(Xmesh, dtype=torch.float32)

        # Get the model's predictions on the meshgrid
        with torch.no_grad():
            scores = model(Xmesh_tensor)
            _, Z = torch.max(scores, 1)
            Z = Z.numpy()
        return Z

    plot_result(generator, 'torch')


def train_by_micrograd():
    model = MLP(2, [4, 4, 3])  # 2-layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))

    def get_label(values):
        max_index = 0
        for i in range(1, len(values)):
            if values[i].data > values[max_index].data:
                max_index = i

        return max_index

    def softmax(logits):
        counts = [logit.exp() for logit in logits]
        denominator = sum(counts)
        out = [c / denominator for c in counts]
        return out

    # loss function
    def loss(batch_size=None):

        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # forward the model to get scores
        values = list(map(model, inputs))
        scores = [softmax(value) for value in values]

        losses = [-scorei[yi].log() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))

        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = data_loss + reg_loss

        # also get accuracy
        accuracy = [get_label(scorei) == yi for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    for k in range(200):
        # forward
        total_loss, acc = loss()

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (sgd)
        learning_rate = 0.01
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 10 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")

    def generator(Xmesh):
        inputs = [list(map(Value, xrow)) for xrow in Xmesh]
        scores = list(map(model, inputs))
        Z = np.array([get_label(s) for s in scores])
        return Z

    plot_result(generator, "micrograd")


train_by_torch()
train_by_micrograd()
