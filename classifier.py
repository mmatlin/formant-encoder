import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

training_formants_path = "train_formants.csv"

# Create Tensors to hold dependent/independent variable data
train_ind = pd.read_csv(training_formants_path, usecols=[1,2,3,4,5], header=None)
train_dep = pd.read_csv(training_formants_path, usecols=[0], header=None)
x = torch.from_numpy(train_ind.values).float()
y = torch.from_numpy(train_dep.values).long()
print(x)
print(y)

# Create a TensorDataset and DataLoader to provide the model with batches of data
train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds, batch_size=32)

##### Set model layer dimensions
### D_in is the input dimension (5, one for each estimated formant)
D_in = x.shape[1]
### H is the hidden layer dimension
H = 16
### C is the number of final categories (there are 14 monophthongs)
C = 14

##### Define the model:
### 1 linear NN layer with 5 nodes
### 1 hidden layer with 16 nodes and ReLU activation
### 1 linear layer with 14 nodes (one for each possible monophthong) and log-softmax activation

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, C),
    torch.nn.LogSoftmax()
)

print(model)

# Define loss as negative log-likelihood
loss_fn = torch.nn.NLLLoss(reduction='sum')

# Set up model learning parameters
learning_rate = 1e-3
epochs = 1#25
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)

for epoch in range(epochs):
    # train_dl provides a batch of independent data and dependent data
    for xb, yb in train_dl:
        # Compute and print loss for each batch every 5 epochs
        pred = model(xb)
        yb = yb.squeeze()
        loss = loss_fn(pred, yb)
        if epoch % 5 == 0:
            print(epoch, loss.item())

        # Zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

# Print model weights and biases
print()
for parameter in model.parameters():
    print(parameter.data)