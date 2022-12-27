import torch

a = torch.zeros((1,1,2,3))
a = a.squeeze()
print(a.shape)