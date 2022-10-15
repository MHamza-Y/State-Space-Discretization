import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

model_path = 'state_quantization/model'

model_dict_path = 'state_quantization/model_dict.pt'
model = torch.load(model_path)
