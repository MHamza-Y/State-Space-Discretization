from time import time

import torch
from torch.optim.lr_scheduler import MultiStepLR


def train_step(data_loader, model, loss_function, optimizer, load_to_gpu):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        if load_to_gpu:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_step(data_loader, model, loss_function, load_to_gpu):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            if load_to_gpu:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss


def train_model(model, test_loader, train_loader, loss_function=None, optimizer=None, learning_rate=0.001, n_epochs=10,
                load_to_gpu=False, gamma=0.1, lr_milestones=[25], train_step_fn=train_step, test_step_fn=test_step):
    if loss_function is None:
        loss_function = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

    print("Untrained test\n--------")
    test_step(test_loader, model, loss_function, load_to_gpu=load_to_gpu)
    print()

    for ix_epoch in range(n_epochs):
        start_time = time()
        print(f"Epoch {ix_epoch}\n---------")
        train_step_fn(train_loader, model, loss_function, optimizer=optimizer, load_to_gpu=load_to_gpu)
        test_step_fn(test_loader, model, loss_function, load_to_gpu=load_to_gpu)
        scheduler.step()
        print(f'lr: {scheduler.get_last_lr()}')
        end_time = time()
        epoch_time = end_time - start_time
        print(f"Epoch time: {epoch_time = :.3f}s")
