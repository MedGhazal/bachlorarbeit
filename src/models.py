# import tracemalloc
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.optim import Adam
from torch.nn.functional import mse_loss, normalize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset

from dataset import MotionDataset, activities_dictionary
from plotters import plot
from utils import (
    Model,
    CLASSES_MAPPING,
    accuracy,
    normatize_dataset,
    to_device,
)


class MLP(Model):

    def __init__(
        self,
        num_classes=len(CLASSES_MAPPING),
        num_features=None,
        num_frames=None,
        optimizer=None,
        loss_function=None,
    ):
        super().__init__(
            num_classes=num_classes,
            num_features=num_features,
            num_frames=num_frames,
            optimizer=optimizer,
            loss_function=loss_function,
        )
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(self.num_features, 64)
        self.hidden1 = nn.Linear(64, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64 * self.num_frames, self.num_classes)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.flatten(F.normalize(x))
        x = self.relu(x)
        x = self.output(x)
        return x


class CNN(Model):

    def __init__(
        self,
        num_classes=None,
        num_features=None,
        num_frames=None,
        optimizer=None,
        loss_function=None,
    ):
        super().__init__(
            num_classes=num_classes,
            num_features=num_features,
            num_frames=num_frames,
            optimizer=optimizer,
            loss_function=loss_function,
        )
        self.network = Sequential(
            nn.Conv1d(num_frames, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
            nn.ReLU(),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_):
        return self.network(input_)


if __name__ == '__main__':
    if not os.path.exists('dataset.pt'):
        dataset = MotionDataset(
            # get_matrixified_root_positions=True,
            get_matrixified_joint_positions=True,
            frequency=1,
            max_length=1000,
            min_length=1000,
        )
        try:
            dataset.parse()
        except FileNotFoundError:
            dataset.extract()
            dataset.parse()

        dataset = dataset.matrix_represetations
        dataset = normatize_dataset(dataset)
        torch.save(dataset, 'dataset.pt')
    else:
        print('Loading the dataset...')
        dataset = torch.load('dataset.pt')

    accuracies, histories = [], []
    num_folds = 10
    folds = random_split(dataset, [1/num_folds]*num_folds)
    for i in range(num_folds):
        model = MLP(
            num_classes=len(activities_dictionary),
            num_features=44,
            # num_features=3,
            num_frames=1000,
            optimizer=Adam,
            loss_function=mse_loss,
        )
        print(f'Next fold {i+1} as validation set...')
        train_dataset = ConcatDataset([
            fold for index, fold in enumerate(folds) if index != i
        ])
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        validation_loader = DataLoader(
            folds[i],
            batch_size=16,
            drop_last=True,
            num_workers=8,
        )
        history = model.fit(5, .0001, train_loader, validation_loader)
        accuracies.append(history[-1]['valueAccuracy'])
        histories.append(history)
    plot(histories)
    print(sum(accuracies)/len(accuracies))
