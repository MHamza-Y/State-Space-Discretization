from time import time

import torch
from torch.utils.tensorboard import SummaryWriter

from state_quantization.forcasting_quantization_models import ForcastingQuant


class Trainer:

    def __init__(self, model, train_loader, test_loader, load_to_gpu, comment):
        self.writer = SummaryWriter(comment=comment)
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.load_to_gpu = load_to_gpu
        self.update_graph()

    def update_graph(self):
        for X, y in self.train_loader:
            if self.load_to_gpu:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            self.writer.add_graph(model=self.model, input_to_model=X)
            break

    def training_step(self):
        raise NotImplementedError

    def evaluation_step(self):
        raise NotImplementedError

    def post_epoch_hook(self):
        pass

    def pre_epoch_hook(self):
        pass

    def train(self, n_epochs):
        print("Untrained test\n--------")
        self.evaluation_step()

        for epoch in range(n_epochs):
            self.epoch = epoch
            start_time = time()
            print('--------------------------------------')
            print('--------------------------------------')
            print(f"Epoch {epoch + 1}\n---------")
            self.pre_epoch_hook()
            self.training_step()
            self.evaluation_step()
            self.post_epoch_hook()
            self.writer.flush()
            end_time = time()
            epoch_time = end_time - start_time
            print(f"Epoch time: {epoch_time = :.3f}s")


class ForcastingQuantTrainer(Trainer):
    def __init__(self, forcasting_quant_model: ForcastingQuant, train_loader, test_loader, autoencoder_training_start,
                 load_to_gpu=False, forcasting_loss_function=None, forecasting_optimizer=None,
                 forecasting_learning_rate=1e-4, forecasting_lr_scheduler=None, autoencoder_lr_scheduler=None,
                 autoencoder_learning_rate=1e-4, autoencoder_loss_function=None, autoencoder_optimizer=None):

        comment = f'model={forcasting_quant_model.__class__},forecasting_learning_rate={forecasting_learning_rate},autoencoder_learning_rate={autoencoder_learning_rate}'
        super().__init__(forcasting_quant_model, train_loader, test_loader, load_to_gpu, comment)
        self.model = forcasting_quant_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.load_to_gpu = load_to_gpu

        if forcasting_loss_function is None:
            forcasting_loss_function = torch.nn.MSELoss()
        if forecasting_optimizer is None:
            forecasting_optimizer = torch.optim.Adam(self.model.forcasting_model.parameters(),
                                                     lr=forecasting_learning_rate)
        if autoencoder_loss_function is None:
            autoencoder_loss_function = torch.nn.MSELoss()
        if autoencoder_optimizer is None:
            autoencoder_optimizer = torch.optim.Adam(self.model.autoencoder_quant_model.parameters(),
                                                     lr=autoencoder_learning_rate)
        self.forcasting_loss_function = forcasting_loss_function
        self.forecasting_optimizer = forecasting_optimizer
        self.forecasting_lr_scheduler = forecasting_lr_scheduler
        self.autoencoder_lr_scheduler = autoencoder_lr_scheduler
        self.autoencoder_loss_function = autoencoder_loss_function
        self.autoencoder_optimizer = autoencoder_optimizer
        self.autoencoder_training_start = autoencoder_training_start
        self.epoch = 0

    def post_epoch_hook(self):
        print('--------------------------------------')
        print(f'Forecasting lr: {self.forecasting_lr_scheduler.get_last_lr()}')
        print(f'Autoencoder lr: {self.autoencoder_lr_scheduler.get_last_lr()}')
        self.autoencoder_lr_scheduler.step()
        self.forecasting_lr_scheduler.step()

    def training_step(self):
        num_batches = len(self.train_loader)
        total_forecasting_loss = 0
        total_autoencoder_loss = 0
        self.model.train()

        for X, y in self.train_loader:
            if self.load_to_gpu:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            forcasting_out, autoencoder_out = self.model(X)
            if self.epoch < self.autoencoder_training_start:
                forecasting_loss = self.forcasting_loss_function(forcasting_out, y)
                self.forecasting_optimizer.zero_grad()
                forecasting_loss.backward()
                self.forecasting_optimizer.step()
                total_forecasting_loss += forecasting_loss.item()
                self.writer.add_scalar("Forecasting/train/loss", forecasting_loss, self.epoch)
            else:

                autoencoder_loss = self.autoencoder_loss_function(autoencoder_out, self.model.autoencoder_in)
                self.autoencoder_optimizer.zero_grad()
                autoencoder_loss.backward()
                self.autoencoder_optimizer.step()
                total_autoencoder_loss += autoencoder_loss.item()
                self.writer.add_scalar("Autoencoder/train/loss", autoencoder_loss, self.epoch)

        print('--------------------------------------')
        if self.epoch < self.autoencoder_training_start:
            avg_forecasting_loss = total_forecasting_loss / num_batches
            print(f"Forcasting Train loss: {avg_forecasting_loss}")
        else:
            avg_autoencoder_loss = total_autoencoder_loss / num_batches
            print(f"Autoencoder Train loss: {avg_autoencoder_loss}")

    def evaluation_step(self):
        num_batches = len(self.test_loader)
        total_forecasting_loss = 0
        total_autoencoder_loss = 0

        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                if self.load_to_gpu:
                    X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                forcasting_out, autoencoder_out = self.model(X)
                forecasting_loss = self.forcasting_loss_function(forcasting_out, y)
                autoencoder_loss = self.autoencoder_loss_function(autoencoder_out,
                                                                  self.model.autoencoder_in)
                total_forecasting_loss += forecasting_loss.item()
                total_autoencoder_loss += autoencoder_loss.item()

        print('--------------------------------------')
        if self.epoch < self.autoencoder_training_start:
            avg_forecasting_loss = total_forecasting_loss / num_batches
            self.writer.add_scalar("Forecasting/Eval/loss", forecasting_loss, self.epoch)
            print(f"Forcasting Test loss: {avg_forecasting_loss}")
        else:
            self.writer.add_scalar("Autoencoder/Eval/loss", autoencoder_loss, self.epoch)
            avg_autoencoder_loss = total_autoencoder_loss / num_batches
            print(f"Autoencoder Test loss: {avg_autoencoder_loss}")