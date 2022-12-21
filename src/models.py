# import tracemalloc
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from dataset import MotionDataset, activities_dictionary
from plotters import plot

CLASSES_MAPPING = {
    activity: number for number, activity in enumerate(activities_dictionary)
}


class MLP(nn.Module):

    def __init__(self, num_classes=11, num_features=3, num_frames=1000):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(3, 64)
        self.hidden1 = nn.Linear(64, 256)
        self.hidden2 = nn.Linear(256, 512)
        self.hidden3 = nn.Linear(512, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.output = nn.Linear(64 * num_frames, num_classes)
        self.optimizer = torch.optim.SGD
        self.loss_function = F.mse_loss

    def forward(self, x):
        x = torch.nn.functional.relu(self.input_layer(x))
        x = torch.nn.functional.relu(self.hidden1(x))
        x = torch.nn.functional.relu(self.hidden2(x))
        x = torch.nn.functional.relu(self.hidden3(x))
        x = torch.nn.functional.relu(self.hidden4(x))
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.output(x))
        return x

    def trainingStep(self, batch):
        motions, labels = batch
        targets = self.forward(motions)
        return self.loss_function(labels, targets)

    def validationStep(self, batch):
        motions, labels = batch
        output = self.forward(motions)
        print(labels, output)
        loss = self.trainingStep(batch)
        return {'valueLoss': loss, 'valueAccuracy': accuracy(output, labels)}

    def validationEpochEnd(self, outputs):
        batch_losses = [x['valueLoss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['valueAccuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
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
            motion, class_ = batch
            onehot_presentation = torch.full((1, len(CLASSES_MAPPING)), .01)
            onehot_presentation[0][CLASSES_MAPPING[class_[0]]] = .9
            batch = [motion, onehot_presentation]
            outputs.append(self.validationStep(batch))
        return self.validationEpochEnd(outputs)

    def fit(self, epochs, learning_rate, train_loader, valuation_loader):
        history = []
        optimizer = self.optimizer(self.parameters(), learning_rate)
        print('Training...')
        for epoch in range(1, epochs):
            for batch in tqdm(train_loader, ncols=100,):
                motion, class_ = batch
                onehot_presentation = torch.full((1, len(CLASSES_MAPPING)), .01)
                onehot_presentation[0][CLASSES_MAPPING[class_[0]]] = .9
                batch = [motion, onehot_presentation]
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
        number_classes,
        loss_function=F.cross_entropy,
        optimizer=torch.optim.ASGD,
    ):
        super().__init__()
        self.network = Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1, dtype=float),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=4, padding=1, dtype=float),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, dtype=float),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, dtype=float),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, dtype=float),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, dtype=float),
            nn.ReLU(),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(5, number_classes, dtype=float),
        )
        self.num_classes = number_classes
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
        output = self.forward(motions)
        loss = self.trainingStep(batch)
        return {'valueLoss': loss, 'valueAccuracy': accuracy(output, labels)}

    def validationEpochEnd(self, outputs):
        batch_losses = [x['valueLoss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['valueAccuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
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
            motion, class_ = batch
            onehot_presentation = torch.zeros((1, len(CLASSES_MAPPING)))
            onehot_presentation[0][CLASSES_MAPPING[class_[0]]] = 1
            batch = [motion, onehot_presentation]
            outputs.append(self.validationStep(batch))
        return self.validationEpochEnd(outputs)

    def fit(self, epochs, learning_rate, train_loader, valuation_loader):
        history = []
        optimizer = self.optimizer(self.parameters(), learning_rate)
        print('Training...')
        for epoch in range(1, epochs):
            for batch in tqdm(train_loader, ncols=100,):
                motion, class_ = batch
                onehot_presentation = torch.zeros((1, len(CLASSES_MAPPING)))
                onehot_presentation[0][CLASSES_MAPPING[class_[0]]] = 1
                batch = [motion, onehot_presentation]
                loss = self.trainingStep(batch)
                loss.backward()
                optimizer.step()
            result = self.evaluate(valuation_loader)
            self.epochEnd(epoch, result)
            history.append(result)
        return history


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def normatize_dataset(dataset):
    normalized_dataset = []
    for matrix_positions, label in dataset:
        normalizes_positions = F.normalize(torch.tensor(matrix_positions))
        normalized_dataset.append(
            [normalizes_positions.float(), label]
        )
    return normalized_dataset


if __name__ == '__main__':
    dataset = MotionDataset(
        get_matrixified_root_positions=True,
        frequency=1,
        max_length=1000,
        min_length=1000,
    )

    try:
        dataset.parse()
    except FileNotFoundError:
        dataset.extract()
        dataset.parse()

    # tracemalloc.start()
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # print('Top 10')
    # for stat in top_stats[:10]:
    #     print(stat)

    dataset = dataset.matrix_represetations
    dataset = normatize_dataset(dataset)

    model = MLP(
        num_classes=len(activities_dictionary),
        num_features=100,
        num_frames=1000
    )
    train_data_set, validation_data_set = random_split(
        dataset,
        [int(len(dataset)*.9)+1, int(len(dataset)*.1)],
    )
    train_loader = DataLoader(
        train_data_set,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    validation_loader = DataLoader(
        validation_data_set,
        batch_size=1,
        drop_last=True,
        num_workers=8,
    )
    history = model.fit(10, .01, train_loader, validation_loader)
    plot(history)
