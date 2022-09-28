from time import time

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR


def train_step(data_loader, model, loss_function, optimizer, load_to_gpu):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    # hc = model.init_hidden()
    for X, y in data_loader:
        if load_to_gpu:
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_step(data_loader, model, loss_function, load_to_gpu):
    num_batches = len(data_loader)
    total_loss = 0

    # hc = model.init_hidden()
    model.eval()
    first_iter = True
    with torch.no_grad():
        for X, y in data_loader:
            if load_to_gpu:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            output = model(X)
            total_loss += loss_function(output, y).item()
            if first_iter:
                print(f"Actual: {y[0, 9, :]}")
                print(f"Predicted: {output[0, 9, :]}")
                first_iter = False

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def train_model(model, test_loader, train_loader, loss_function=None, optimizer=None, learning_rate=0.001, n_epochs=10,
                load_to_gpu=False, gamma=0.1, lr_milestones=[25]):
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
        train_step(train_loader, model, loss_function, optimizer=optimizer, load_to_gpu=load_to_gpu)
        test_step(test_loader, model, loss_function, load_to_gpu=load_to_gpu)
        scheduler.step()
        print(f'lr: {scheduler.get_last_lr()}')
        end_time = time()
        epoch_time = end_time - start_time
        print(f"Epoch time: {epoch_time = :.3f}s")
