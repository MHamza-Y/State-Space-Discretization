from time import time

import torch


def train_step(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    # hc = model.init_hidden()
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_step(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    #hc = model.init_hidden()
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def train_model(model, test_loader, train_loader, loss_function=None, optimizer=None, learning_rate=0.001, n_epochs=10):
    if loss_function is None:
        loss_function = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_step(test_loader, model, loss_function)
    print()

    for ix_epoch in range(n_epochs):
        start_time = time()
        print(f"Epoch {ix_epoch}\n---------")
        train_step(train_loader, model, loss_function, optimizer=optimizer)
        test_step(test_loader, model, loss_function)
        end_time = time()
        epoch_time = end_time - start_time
        print(f"Epoch time: {epoch_time = :.3f}s")
