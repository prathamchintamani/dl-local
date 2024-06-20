import pandas as pd
import numpy as np

import torch
from torch import nn

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

df = pd.read_csv("data_banknote_authentication.txt")
df.columns = ["variance","skewness","curtosis","entropy","target"]
df = df.sample(frac = 1)
df = df.reset_index(drop = True)

y = df.target
X = df.drop(["target"],axis = 1)

def train_test_split(X,y,train_size=0.80):
  n = int(np.floor(train_size*len(X)))
  X_train = X[0:n]
  X_test = X[n:]
  y_train = y[0:n]
  y_test = y[n:]
  
  return X_train,X_test,y_train,y_test
  
X_train ,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.70)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

n = np.shape(X_train)[0]
m = np.shape(X_test)[0]

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

class neural(nn.Module):
  def __init__(self, in_size, hidden_size = 8):
    super().__init__()
    self.output = nn.Sequential(
        nn.Linear(in_features=in_size,out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size,out_features=1),
        nn.Sigmoid()
    )
  def forward(self, X):
    return self.output(X)

model = neural(in_size = 4, hidden_size = 10).to(device)

print(model)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(y_logits) # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model(X_test).squeeze()
      test_pred = torch.round(test_logits) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")