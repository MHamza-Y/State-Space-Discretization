import matplotlib.pyplot as plt
import torch

value_keys = ['setpoint', 'velocity', 'gain', 'shift', 'fatigue', 'consumption']


def eval_model(model, x, y):
    model.eval()
    print(x.unsqueeze(0).shape)
    with torch.no_grad():
        pred_y = model(x.unsqueeze(0))
        print(pred_y)
    pred_y = pred_y[0].cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    print(pred_y.shape)
    print(y.shape)

    fig, axs = plt.subplots(y.shape[1], figsize=(30, 60))
    for i in range(y.shape[1]):
        axs[i].plot(pred_y[:, i], label='y_pred')
        axs[i].plot(y[:, i], label='y')
        axs[i].set_title(value_keys[i])
        axs[i].yaxis.set_tick_params(labelsize=20)
        axs[i].plot()
