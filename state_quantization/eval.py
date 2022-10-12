import matplotlib.pyplot as plt
import torch


def eval_model(model, x, y, value_keys):
    model.eval()
    print(x.unsqueeze(0).shape)
    with torch.no_grad():
        pred_y = model(x.unsqueeze(0))
        print(y)
        print(pred_y)
    pred_y = pred_y[0].cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    print(pred_y.shape)
    print(y.shape)

    fig, axs = plt.subplots(y.shape[1], figsize=(30, 60))
    for i in range(y.shape[1]):
        axs[i].plot(pred_y[:, i], label='y_pred')
        axs[i].plot(y[:, i], label='y')
        plt.legend()
        axs[i].set_title(value_keys[i])
        axs[i].yaxis.set_tick_params(labelsize=20)
        axs[i].set_ylim([-2, 2])
        axs[i].plot()


def compare_models(models, x, y, value_keys):
    pred = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred_y = model(x.unsqueeze(0))
        pred_y = pred_y[0].cpu().detach().numpy()
        pred.append(pred_y)
    y = y.cpu().detach().numpy()
    fig, axs = plt.subplots(y.shape[1], figsize=(30, 60))

    for i in range(y.shape[1]):

        for j, pred_y in enumerate(pred):
            axs[i].plot(pred_y[:, i], label=models[j].__class__.__name__)
        axs[i].plot(y[:, i], label='y')
        axs[i].set_title(value_keys[i])
        axs[i].yaxis.set_tick_params(labelsize=20)
        axs[i].set_ylim([-2, 2])
        axs[i].legend()
        axs[i].plot()
