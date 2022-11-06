import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data import random_split


class CNN(Module):

    def __init__(
        self,
        number_classes,
        loss_function=F.cross_entropy,
        optimizer=torch.optim.SGD,
    ):
        super().__init__()
        self.network = Sequential(
            Conv2d()
        )


if __name__ == '__main__':
    print('Training')
