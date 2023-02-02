import os
import json
# from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset

from dataset import MotionDataset, activities_dictionary, Classification
from plotters import plot
from utils import (
    Model,
    CLASSES_MAPPING,
    prepare_dataset,
)


class MLP(Model):
    is_seqential = False
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
        self.input_layer = nn.Linear(num_features * num_frames, 1024)
        self.hidden1 = nn.Linear(1024, 2048)
        self.hidden2 = nn.Linear(2048, 4096)
        self.output = nn.Linear(4096, self.num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_, device):
        x = self.relu(self.input_layer(self.flatten(input_)))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(self.flatten(F.normalize(x)))
        return self.softmax(x)


class FCN(Model):
    is_seqential = False
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
            nn.Conv1d(num_features, 3, kernel_size=3, padding=3),
            nn.ReLU(),
            # nn.BatchNorm1d(27),
            # nn.Conv1d(27, 9, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm1d(9),
            # nn.Conv1d(9, 3, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm1d(3),
            # nn.Conv1d(3, num_classes, kernel_size=3, padding=1),
            # nn.BatchNorm1d(num_classes),
            # nn.AvgPool1d(num_classes),
            # nn.Flatten(),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_, device):
        output = self.network(input_)
        print()
        print(input_.size())
        print(output.size())
        self.output_layer = nn.Linear(output.size()[-1], self.num_classes)
        self.softmax = nn.Softmax(0)
        # return self.softmax(self.output_layer(output))


class CNN(Model):
    is_seqential = False
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
            nn.Conv1d(num_features, 27, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(27),
            nn.Conv1d(27, 9, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(9),
            nn.Conv1d(9, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(3),
            nn.Conv1d(3, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_classes),
            nn.AvgPool1d(num_classes),
            nn.Flatten(),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_, device):
        output = self.network(input_)
        self.output_layer = nn.Linear(output.size()[-1], self.num_classes)
        self.softmax = nn.Softmax(0)
        return self.softmax(self.output_layer(output))


class ResNetBlock(nn.Module):

    def __init__(self, shape_in=None, shape_out=None,):
        super().__init__()
        self.network = Sequential(
            nn.Conv1d(shape_in, 9, 1),
            nn.ReLU(),
            nn.BatchNorm1d(9),
            nn.Conv1d(9, 18, 1),
            nn.ReLU(),
            nn.BatchNorm1d(18),
            nn.Conv1d(18, shape_out, 1),
            nn.ReLU(),
            nn.BatchNorm1d(shape_out),
        )

    def forward(self, input_, device):
        return self.network(input_)


class ResNet(Model):
    is_seqential = False
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
        self.block_1 = ResNetBlock(num_features, num_features)
        self.block_2 = ResNetBlock(num_features, num_features)
        self.block_3 = ResNetBlock(num_features, num_features)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(0)

    def forward(self, motion, device):
        output_block_1 = self.block_1(motion, device)
        output_block_2 = self.block_1(output_block_1 + motion, device)
        output_block_3 = self.block_1(output_block_1 + output_block_2, device)
        self.output_layer = nn.Linear(output_block_3.size()[-1], self.num_classes)
        self.softmax = nn.Softmax(0)
        return self.softmax(self.output_layer(output_block_3))


class RNN(Model):
    is_seqential = True
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
            num_features,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        input_ = torch.transpose(input_, 1, 2)
        output, _ = self.rnn(input_, initial_hidden_state)
        return self.softmax(self.linear(output[:, -1, :]))


class GRU(Model):
    is_seqential = True
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
            num_features,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        output, _ = self.gru(input_, initial_hidden_state)
        return self.softmax(self.linear(output[:, -1, :]))


class LSTM(Model):
    is_seqential = True
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
            num_features,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        initial_cell_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        input_ = torch.transpose(input_, 1, 2)
        output, _ = self.lstm(
            input_, (initial_hidden_state, initial_cell_state),
        )
        return self.softmax(self.linear(output[:, -1, :]))


class CNN_LSTM(Model):
    is_seqential = True

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
        padding, kernel_size, stride, dilation = [1, 2, 1, 1]
        self.conv1d = nn.Conv1d(
            num_features,
            16,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        shape_out = num_frames + 2 * padding - dilation * (kernel_size - 1) - 1
        shape_out /= stride
        shape_out += 1
        shape_out = int(shape_out)
        self.lstm = nn.LSTM(
            shape_out,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        cnn_output = self.conv1d(input_)
        initial_cell_state = torch.zeros(
            self.num_layers, input_.size(0), self.hidden_size,
        ).to(device)
        output, _ = self.lstm(
            cnn_output, (initial_hidden_state, initial_cell_state),
        )
        return self.softmax(self.linear(output[:, -1, :]))


def train_model_on(
    model,
    device,
    weights,
    frequency,
    min_length,
    max_length,
    num_epochs,
    train_loader,
    validation_loader,
    num_features,
    num_layers=None,
    hidden_size=None,
):

    if model.is_seqential:
        model_ = model(
            num_classes=len(activities_dictionary),
            num_features=num_features,
            num_frames=max_length//frequency,
            optimizer=Adam,
            loss_function=cross_entropy,
            num_layers=num_layers,
            hidden_size=hidden_size,
        ).to(device)
    else:
        model_ = model(
            num_classes=len(activities_dictionary),
            num_features=num_features,
            num_frames=max_length//frequency,
            optimizer=Adam,
            loss_function=cross_entropy,
        ).to(device)
    return model_.fit(
        num_epochs,
        10 ** -5,
        device,
        train_loader,
        validation_loader,
        weights=weights,
    )


def train_model(
    device,
    model,
    weights,
    folds,
    num_epochs,
    frequency,
    min_length,
    max_length,
    num_features,
    num_layers=None,
    hidden_size=None,
):

    accuracies, histories = [], []
    labels_, predictions_, training_losses_ = [], [], []
    weights = torch.tensor(weights).to(device)

    for i, fold in enumerate(folds):
        print(f'Next fold {i+1} as validation set...')
        train_dataset = ConcatDataset([
            fold for index, fold in enumerate(folds) if index != i
        ])
        train_loader = DataLoader(
            train_dataset,
            batch_size=1 if model.is_seqential else 64,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        validation_loader = DataLoader(
            fold,
            batch_size=1 if model.is_seqential else 64,
            drop_last=False,
            num_workers=0,
        )
        training_losses, history, labels, predictions = train_model_on(
            model,
            device,
            weights,
            frequency,
            min_length,
            max_length,
            num_epochs,
            train_loader,
            validation_loader,
            num_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        training_losses_.append(training_losses)
        labels_.append(labels)
        predictions_.append(predictions)
        accuracies.append(history[-1]['valueAccuracy'])
        histories.append(history)

    plot(
        model.__name__,
        frequency,
        num_features,
        histories,
        labels_,
        predictions_,
        training_losses_,
    )
    print(
        f'Average accuracy is {sum(accuracies)/len(accuracies):.2f}%'
        f' with the model {model.__name__}'
    )
