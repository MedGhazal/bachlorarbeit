import os
import sys
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.optim import Adam, SGD, ASGD
from torch.nn.functional import mse_loss, normalize, cross_entropy, soft_margin_loss, nll_loss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset

from dataset import MotionDataset, activities_dictionary, Classification
from plotters import plot
from utils import (
    Model,
    CLASSES_MAPPING,
    accuracy,
    prepare_dataset,
    to_device,
    visualize_class_distribution,
    # visualize_length_distribution,
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
        self.softmax = nn.Softmax(0)
        self.input_layer = nn.Linear(num_features * num_frames, 16)
        self.hidden1 = nn.Linear(16, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.output = nn.Linear(64, self.num_classes)

    def forward(self, input_, device):
        x = self.relu(self.input_layer(self.flatten(input_)))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.flatten(F.normalize(x))
        x = self.softmax(x)
        x = self.output(x)
        x = self.softmax(x)
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
            nn.Conv1d(num_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Flatten(),
            nn.Linear(144, num_classes),
            nn.Softmax(0),
            # nn.Linear(num_classes, num_classes),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_, device):
        return self.network(input_)


class CNN_2d(Model):

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
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, 3),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, 3),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(277, num_classes),
            nn.ReLU(),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_, device):
        return self.network(input_)


class RNN(Model):

    def __init__(
        self,
        num_classes=None,
        num_features=None,
        num_frames=None,
        num_layers=None,
        hidden_size=None,
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
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            num_frames,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        output, _ = self.rnn(input_, initial_hidden_state)
        return self.softmax(self.linear(output[:, -1, :]))


class GRU(Model):

    def __init__(
        self,
        num_classes=None,
        num_features=None,
        num_frames=None,
        num_layers=None,
        hidden_size=None,
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
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            num_frames,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        output, _ = self.gru(input_, initial_hidden_state)
        return self.softmax(self.linear(output[:, -1, :]))


class LSTM(Model):

    def __init__(
        self,
        num_classes=None,
        num_features=None,
        num_frames=None,
        num_layers=None,
        hidden_size=None,
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
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            num_frames,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        initial_cell_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        output, _ = self.lstm(input_, (initial_hidden_state, initial_cell_state))
        return self.softmax(self.linear(output[:, -1, :]))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = 'cpu'
    frequency = 1
    length = 10000
    if not os.path.exists('dataset.pt'):

        dataset = MotionDataset(
            classification=Classification(2),
            # get_matrixified_root_positions=True,
            get_matrixified_joint_positions=True,
            frequency=frequency,
            max_length=length,
            min_length=length,
        )

        try:
            dataset.parse()
        except FileNotFoundError:
            dataset.extract()
            dataset.parse()

        dataset = dataset.matrix_represetations
        label_frequency = visualize_class_distribution(dataset)
        num_classified_motions = len(dataset)
        label_ratio = {
            label: frequency / num_classified_motions
            for label, frequency in label_frequency.items()
        }
        with open('label_ratio.json', 'w') as json_file:
            json.dump(label_ratio, json_file)
        weights = [
            ratio for label, ratio in label_ratio.items()
            if label in activities_dictionary
        ]
        dataset = prepare_dataset(dataset, normalize=True)
        torch.save(dataset, 'dataset.pt')
    else:
        print('Loading the dataset...')
        dataset = torch.load('dataset.pt')
        with open('label_ratio.json', 'r') as json_file:
            label_ratio = json.load(json_file)
        weights = [
            ratio for label, ratio in label_ratio.items()
            if label in activities_dictionary
        ]

    accuracies, histories = [], []
    num_folds = 10 # 10, 30, 2
    if num_folds == 2:
        folds = random_split(dataset, [.1, .9])
    else:
        folds = random_split(dataset, [1/num_folds]*num_folds)
    labels_, predictions_, training_losses_ = [], [], []
    for i in range(num_folds):
        model = LSTM(
            num_classes=len(activities_dictionary),
            num_features=44,
            # num_features=3,
            num_frames=length//frequency,
            optimizer=Adam,
            # optimizer=SGD,
            # optimizer=ASGD,
            # loss_function=mse_loss,
            loss_function=cross_entropy,
            num_layers=10,
            hidden_size=10,
        ).to(device)
        print(f'Next fold {i+1} as validation set...')
        train_dataset = ConcatDataset([
            fold for index, fold in enumerate(folds) if index != i
        ])
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )
        validation_loader = DataLoader(
            folds[i],
            batch_size=16,
            drop_last=False,
            num_workers=8,
        )
        training_losses, history, labels, predictions = model.fit(
            50,
            .00001,
            # .01,
            device,
            train_loader,
            validation_loader,
            weights=torch.tensor(weights),
        )
        training_losses_.append(training_losses)
        labels_.append(labels)
        predictions_.append(predictions)
        accuracies.append(history[-1]['valueAccuracy'])
        histories.append(history)
        if i == 0:
            break
    plot('MLP', histories, labels_, predictions_, training_losses_)
    print(f'Average accuracy is {sum(accuracies)/len(accuracies):.2f}%')
