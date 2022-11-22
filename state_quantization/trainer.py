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


class NNTrainer(Trainer):
    def __init__(self, model, train_loader, test_loader,
                 load_to_gpu=False, loss_function=None, optimizer=None,
                 learning_rate=1e-4, lr_scheduler=None, eval_loss_graph_tags='Model/Eval/loss',
                 train_loss_graph_tags='Model/train/loss'):

        comment = f'model={model.__class__},learning_rate={learning_rate},lr_scheduler={lr_scheduler}'
        super().__init__(model, train_loader, test_loader, load_to_gpu, comment)

        if loss_function is None:
            loss_function = torch.nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        if eval_loss_graph_tags is None:
            eval_loss_graph_tags = ['Model/Eval/loss']
        if train_loss_graph_tags is None:
            train_loss_graph_tags = ['Model/train/loss']
        self.eval_loss_graph_tags = eval_loss_graph_tags
        self.train_loss_graph_tags = train_loss_graph_tags

    def post_epoch_hook(self):
        print('--------------------------------------')
        print(f'lr: {self.lr_scheduler.get_last_lr()}')
        self.lr_scheduler.step()

    def training_step(self):
        num_batches = len(self.train_loader)
        total_loss = 0
        self.model.train()

        for X, y in self.train_loader:
            if self.load_to_gpu:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            output = self.model(X)

            loss = self.loss_function(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            for tag in self.train_loss_graph_tags:
                self.writer.add_scalar(tag, loss, self.epoch)

        print('--------------------------------------')

        avg_loss = total_loss / num_batches
        print(f"Model Train loss: {avg_loss}")

    def evaluation_step(self):
        num_batches = len(self.test_loader)
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                if self.load_to_gpu:
                    X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                output = self.model(X)

                loss = self.loss_function(output, y)
                total_loss += loss.item()

        print('--------------------------------------')

        avg_loss = total_loss / num_batches
        for tag in self.eval_loss_graph_tags:
            self.writer.add_scalar(tag, avg_loss, self.epoch)
        print(f"Model Test loss: {avg_loss}")


class ForcastingQuantTrainer(Trainer):
    def __init__(self, forcasting_quant_model: ForcastingQuant, train_loader, test_loader, autoencoder_training_start,
                 load_to_gpu=False, forcasting_loss_function=None, forecasting_optimizer=None,
                 forecasting_learning_rate=1e-4, forecasting_lr_scheduler=None, autoencoder_lr_scheduler=None,
                 autoencoder_learning_rate=1e-4, autoencoder_loss_function=None, autoencoder_optimizer=None,
                 additional_eval_model=None):

        comment = f'model={forcasting_quant_model.__class__},hidden_size={forcasting_quant_model.forcasting_model.hidden_size},bits={forcasting_quant_model.autoencoder_quant_model.bottleneck_size},forecasting_learning_rate={forecasting_learning_rate},autoencoder_learning_rate={autoencoder_learning_rate}'
        super().__init__(forcasting_quant_model, train_loader, test_loader, load_to_gpu, comment)

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
        self.additional_eval_model = additional_eval_model
        if self.additional_eval_model:
            self.additional_eval_model_loss_func = torch.nn.MSELoss()

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

            else:
                # self.model.forcasting_model.eval()
                autoencoder_loss = self.autoencoder_loss_function(autoencoder_out, self.model.autoencoder_in)
                self.autoencoder_optimizer.zero_grad()
                autoencoder_loss.backward()
                self.autoencoder_optimizer.step()
                total_autoencoder_loss += autoencoder_loss.item()

        print('--------------------------------------')
        if self.epoch < self.autoencoder_training_start:

            avg_forecasting_loss = total_forecasting_loss / num_batches
            self.writer.add_scalar("Forecasting/train/loss", avg_forecasting_loss, self.epoch)
            print(f"Forcasting Train loss: {avg_forecasting_loss}")
        else:
            avg_autoencoder_loss = total_autoencoder_loss / num_batches
            self.writer.add_scalar("Autoencoder/train/loss", avg_autoencoder_loss,
                                   abs(self.epoch - self.autoencoder_training_start))
            print(f"Autoencoder Train loss: {avg_autoencoder_loss}")

    def evaluation_step(self):
        num_batches = len(self.test_loader)
        total_forecasting_loss = 0
        total_autoencoder_loss = 0
        total_eval_model_loss = 0
        total_first_out_loss = 0

        self.model.eval()
        if self.additional_eval_model:
            self.additional_eval_model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                if self.load_to_gpu:
                    X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                forcasting_out, autoencoder_out = self.model(X)
                if self.additional_eval_model:  # and self.epoch >= self.autoencoder_training_start:
                    output = self.additional_eval_model(X)
                    eval_model_loss = self.additional_eval_model_loss_func(output, y)
                    total_eval_model_loss += eval_model_loss.item()

                forecasting_loss = self.forcasting_loss_function(forcasting_out, y)
                autoencoder_loss = self.autoencoder_loss_function(autoencoder_out,
                                                                  self.model.autoencoder_in)
                first_out_loss = self.forcasting_loss_function(forcasting_out[:, 0, :], y[:, 0, :])
                total_forecasting_loss += forecasting_loss.item()
                total_autoencoder_loss += autoencoder_loss.item()
                total_first_out_loss += first_out_loss.item()

        print('--------------------------------------')
        if self.epoch < self.autoencoder_training_start:
            avg_forecasting_loss = total_forecasting_loss / num_batches
            self.writer.add_scalar("Forecasting/Eval/loss", avg_forecasting_loss, self.epoch)
            print(f"Forcasting Test loss: {avg_forecasting_loss}")
        else:
            avg_autoencoder_loss = total_autoencoder_loss / num_batches
            self.writer.add_scalar("Autoencoder/Eval/loss", avg_autoencoder_loss,
                                   abs(self.epoch - self.autoencoder_training_start))
            print(f"Autoencoder Test loss: {avg_autoencoder_loss}")
            if self.additional_eval_model:
                avg_eval_model_loss = total_eval_model_loss / num_batches
                self.writer.add_scalar("Model/Eval/loss", avg_eval_model_loss,
                                       abs(self.epoch - self.autoencoder_training_start))
                avg_first_out_loss = total_first_out_loss / num_batches
                self.writer.add_scalar("Model/Eval/first_out_loss", avg_first_out_loss,
                                       abs(self.epoch - self.autoencoder_training_start))
                print(f"Eval Model Test loss: {avg_eval_model_loss}")
                print(f"First Out Loss: {avg_first_out_loss}")
