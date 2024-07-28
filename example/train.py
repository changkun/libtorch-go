# Copyright 2024 Changkun Ou <changkun.de>. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def generate_data(num_samples=1000):
    X = np.random.rand(num_samples, 4)
    y = np.sum(X**2, axis=1, keepdims=True)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_train, y_train = generate_data()

model = NN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

X_test, y_test = generate_data(100)
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)

print(f'Test Loss: {test_loss.item():.4f}')
torch.jit.save(torch.jit.script(model), 'model.pt')