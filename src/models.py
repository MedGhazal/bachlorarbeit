import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.nn import Sequential
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset


from utils import (
    Model,
    CLASSES_MAPPING,
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

        def calculate_output_shape(input_shape, kernel_size):
            return (input_shape - kernel_size + 1)

        def calculate_output_shape_strided(input_shape, kernel_size):
            return floor(input_shape / kernel_size)
        output_shape = calculate_output_shape_strided(
            calculate_output_shape(
                calculate_output_shape(
                    calculate_output_shape(
                        calculate_output_shape(
                            calculate_output_shape(num_frames, 3), 3,
                        ), 3
                    ), 3
                ), 3
            ), num_classes,
        ) - 16
        self.network = Sequential(
            nn.Conv1d(num_frames, 4, 3),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 8, 3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, num_classes, 3),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.AvgPool1d(num_classes),
            nn.Flatten(),
            nn.Linear(output_shape, num_classes),
            nn.Softmax(0),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_, device):
        return self.network(input_)


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

        def calculate_output_shape(input_shape, kernel_size):
            return (input_shape - kernel_size + 1)

        def calculate_output_shape_strided(input_shape, kernel_size):
            return floor(input_shape / kernel_size)
        output_shape = calculate_output_shape_strided(
            calculate_output_shape(
                calculate_output_shape(
                    calculate_output_shape(
                        calculate_output_shape(
                            calculate_output_shape(
                                calculate_output_shape(num_frames, 3), 3,
                            ), 3
                        ), 3
                    ), 3
                ), 3
            ), num_classes,
        ) * num_classes // 12
        self.network = Sequential(
            nn.Conv1d(num_frames, 4, 3),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 3),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
            nn.AvgPool1d(num_classes),
            nn.Flatten(),
            nn.Linear(output_shape, num_classes),
            nn.Softmax(0),
        )
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_, device):
        return self.network(input_)


class ResNetBlock(nn.Module):

    def __init__(self, shape_in=None, shape_out=None,):
        super().__init__()
        self.network = Sequential(
            nn.Conv1d(shape_in, 4, 1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, shape_out, 1),
            nn.BatchNorm1d(shape_out),
            nn.ReLU(),
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

        self.block_1 = ResNetBlock(num_frames, num_frames)
        self.block_2 = ResNetBlock(num_frames, num_frames)
        self.block_3 = ResNetBlock(num_frames, num_frames)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_frames * num_features, num_classes)
        self.softmax = nn.Softmax(0)
        self.relu = nn.ReLU()

    def forward(self, motion, device):
        output_block_1 = self.block_1(motion, device)
        output_block_2 = self.block_2(output_block_1 + motion, device)
        output_block_3 = self.block_3(output_block_1 + output_block_2, device)
        output_blocks = self.flatten(self.flatten(output_block_3))
        return self.softmax(self.linear(output_blocks))


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
        bidirectional=False,
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
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(
            hidden_size * (2 if self.rnn.bidirectional else 1),
            num_classes,
        )
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        if self.rnn.bidirectional:
            initial_hidden_state = torch.zeros(
                self.num_layers * 2, input_.size(0), self.hidden_size,
            ).to(device)
        else:
            initial_hidden_state = torch.zeros(
                self.num_layers, input_.size(0), self.hidden_size,
            ).to(device)
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
        bidirectional=False,
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
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(
            hidden_size * (2 if self.gru.bidirectional else 1),
            num_classes,
        )
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        if self.gru.bidirectional:
            initial_hidden_state = torch.zeros(
                self.num_layers * 2, input_.size(0), self.hidden_size,
            ).to(device)
        else:
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
        bidirectional=False,
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
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(
            hidden_size * (2 if self.lstm.bidirectional else 1),
            num_classes,
        )
        self.softmax = nn.Softmax(0)

    def forward(self, input_, device):
        initial_hidden_state = torch.zeros(
            self.num_layers * (2 if self.lstm.bidirectional else 1),
            input_.size(0),
            self.hidden_size,
        ).to(device)
        initial_cell_state = torch.zeros(
            self.num_layers * (2 if self.lstm.bidirectional else 1),
            input_.size(0),
            self.hidden_size,
        ).to(device)
        output, _ = self.lstm(
            input_, (initial_hidden_state, initial_cell_state),
        )
        return self.softmax(self.linear(output[:, -1, :]))


def train_model_on(
    model,
    num_classes,
    device,
    weights,
    num_frames,
    num_epochs,
    learning_rate,
    train_loader,
    validation_loader,
    num_features,
    num_layers=None,
    hidden_size=None,
    bidirectional=False,
):

    if model.is_seqential:
        model_ = model(
            num_classes=num_classes,
            num_features=num_features,
            num_frames=num_frames,
            optimizer=Adam,
            loss_function=cross_entropy,
            num_layers=num_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
        ).to(device)
    else:
        model_ = model(
            num_classes=num_classes,
            num_features=num_features,
            num_frames=num_frames,
            optimizer=Adam,
            loss_function=cross_entropy,
        ).to(device)
    return model_.fit(
        num_epochs,
        learning_rate,
        device,
        train_loader,
        validation_loader,
        weights=weights,
    )


def train_model(
    device,
    model,
    num_classes,
    weights,
    folds,
    num_epochs,
    learning_rate,
    num_frames,
    num_features,
    num_layers=None,
    hidden_size=None,
    bidirectional=False,
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
            batch_size=32 if model.is_seqential else 32,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        validation_loader = DataLoader(
            fold,
            batch_size=64,
            drop_last=False,
            num_workers=0,
        )
        training_losses, history, labels, predictions = train_model_on(
            model,
            num_classes,
            device,
            weights,
            num_frames,
            num_epochs,
            learning_rate,
            train_loader,
            validation_loader,
            num_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
        )
        training_losses_.append(training_losses)
        labels_.append(labels)
        predictions_.append(predictions)
        accuracies.append(history[-1]['valueAccuracy'])
        histories.append(history)

    print(
        f'Average accuracy is {sum(accuracies)/len(accuracies):.2f}%'
        f' with the model {model.__name__}'
    )
    return histories, labels_, predictions_, training_losses_
