# import tracemalloc
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset

from dataset import MotionDataset, activities_dictionary
from plotters import plot

CLASSES_MAPPING = {
    activity: number for number, activity in enumerate(activities_dictionary)
}


class MLP(nn.Module):

    def __init__(self, num_classes=11, num_features=44, num_frames=1000):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(num_features, 64)
        self.hidden1 = nn.Linear(64, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64 * num_frames, num_classes)
        self.optimizer = torch.optim.Adam
        self.loss_function = F.mse_loss

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.flatten(F.normalize(x))
        x = self.relu(x)
        x = self.output(x)
        return x

    def trainingStep(self, batch):
        motions, labels = batch
        targets = self.forward(motions)
        return self.loss_function(labels, targets)

    def validationStep(self, batch):
        motions, labels = batch
        output = self.forward(motions)
        loss = self.trainingStep(batch)
        return {'valueLoss': loss, 'valueAccuracy': accuracy(output, labels)}

    def validationEpochEnd(self, outputs):
        batch_losses = [x['valueLoss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['valueAccuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() * 100
        return {
            'valueLoss': epoch_loss.item(),
            'valueAccuracy': epoch_acc.item(),
        }

    def epochEnd(self, epoch, result):
        print(
            f"{epoch}. epoch, the loss value: {result['valueLoss']:.2f}"
            f" with model accuracy: {result['valueAccuracy']:.2f}%"
        )

    def evaluate(self, valuationSetLoader):
        outputs = []
        print('Evaluate...')
        for batch in tqdm(valuationSetLoader, ncols=100):
            outputs.append(self.validationStep(batch))
        return self.validationEpochEnd(outputs)

    def fit(self, epochs, learning_rate, train_loader, valuation_loader):
        history = []
        optimizer = self.optimizer(self.parameters(), learning_rate)
        print('Training...')
        for epoch in range(1, epochs):
            for batch in tqdm(train_loader, ncols=100,):
                loss = self.trainingStep(batch)
                loss.backward()
                optimizer.step()
            result = self.evaluate(valuation_loader)
            self.epochEnd(epoch, result)
            history.append(result)
        return history


class CNN(Module):

    def __init__(
        self,
        num_classes=11,
        num_features=44,
        num_frames=1000,
        loss_function=F.mse_loss,
        optimizer=torch.optim.ASGD,
    ):
        super().__init__()
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

    def trainingStep(self, batch):
        motions, labels = batch
        targets = self.forward(motions)
        return self.loss_function(labels, targets)

    def validationStep(self, batch):
        motions, labels = batch
        outputs = self.forward(motions)
        loss = self.trainingStep(batch)
        return {'valueLoss': loss, 'valueAccuracy': accuracy(outputs, labels)}

    def validationEpochEnd(self, outputs):
        batch_losses = [x['valueLoss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['valueAccuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() * 100
        return {
            'valueLoss': epoch_loss.item(),
            'valueAccuracy': epoch_acc.item(),
        }

    def epochEnd(self, epoch, result):
        print(
            f"{epoch}. epoch, the loss value: {result['valueLoss']:.2f}"
            f" with model accuracy: {result['valueAccuracy']:.2f}%"
        )

    def evaluate(self, valuationSetLoader):
        outputs = []
        print('Evaluate...')
        for batch in tqdm(valuationSetLoader, ncols=100):
            outputs.append(self.validationStep(batch))
        return self.validationEpochEnd(outputs)

    def fit(self, epochs, learning_rate, train_loader, valuation_loader):
        history = []
        optimizer = self.optimizer(self.parameters(), learning_rate)
        print('Training...')
        for epoch in range(1, epochs):
            for batch in tqdm(train_loader, ncols=100,):
                loss = self.trainingStep(batch)
                loss.backward()
                optimizer.step()
            result = self.evaluate(valuation_loader)
            self.epochEnd(epoch, result)
            history.append(result)
        return history


def accuracy(outputs, labels):
    predictions = torch.argmax(outputs, dim=1)
    labels = torch.argmax(labels, dim=1)
    return torch.tensor(
        torch.sum(predictions == labels).item() / len(predictions)
    )


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def normatize_dataset(dataset):
    normalized_dataset = []
    for matrix_positions, label in dataset:
        onehot_presentation = torch.zeros((len(CLASSES_MAPPING)))
        onehot_presentation[CLASSES_MAPPING[label]] = 1
        normalizes_positions = F.normalize(torch.tensor(matrix_positions))
        normalized_dataset.append(
            [normalizes_positions.float(), onehot_presentation]
        )
    return normalized_dataset


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
        print(f'Next fold {i+1} as validation set...')
        model = MLP(
            num_classes=len(activities_dictionary),
            num_features=44,
            # num_features=3,
            num_frames=1000
        )
        train_dataset = ConcatDataset([
            fold for index, fold in enumerate(folds) if index != 0
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
