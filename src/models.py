from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from dataset import MotionDataset, activities_dictionary

CLASSES_MAPPING = {
    activity: number+1 for number, activity in enumerate(activities_dictionary)
}


class CNN(Module):

    def __init__(
        self,
        number_classes,
        loss_function=F.cross_entropy,
        optimizer=torch.optim.SGD,
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
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(30, 128, dtype=float),
            nn.ReLU(),
            nn.Linear(128, 44, dtype=float),
            nn.Linear(44, number_classes, dtype=float),
        )
        self.num_classes = number_classes
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, input_):
        return self.network(input_)

    def trainingStep(self, batch):
        motions, labels = batch
        print(motions, labels, sep='\n')
        return self.loss_function(motions, labels)

    def validationStep(self, batch):
        motions, labels = batch
        output = self(motions)
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
            f"{epoch}. epoch, the loss value: {result['valuelos']:.2f}"
            f"with model accuracy: {result['valueAccuracy']:.2f}%"
        )

    def evaluate(self, valuationSetLoader):
        outputs = [self.validationStep(batch) for batch in valuationSetLoader]
        return self.validationEpochEnd(outputs)

    def fit(self, epochs, learning_rate, train_loader, valuation_loader):
        history = []
        optimizer = self.optimizer(self.parameters(), learning_rate)
        for epoch in tqdm(range(epochs), ncols=100,):
            for batch in tqdm(train_loader, ncols=100,):
                motion, class_ = batch
                onehot_presentation = torch.zeros((1, len(CLASSES_MAPPING)))
                onehot_presentation[0][CLASSES_MAPPING[class_[0]]] = 1
                batch = [motion, onehot_presentation]
                loss = self.trainingStep(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
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


if __name__ == '__main__':
    dataset = MotionDataset(
        matrixfy=True,
        frequency=10,
        max_length=1000,
        min_length=1000,
    )

    try:
        dataset.parse()
    except FileNotFoundError:
        dataset.extract()
        dataset.parse()

    dataset = dataset.matrix_represetations
    print(torch.tensor(dataset[0][0].size))

    model = CNN(len(activities_dictionary))
    print(
        len(dataset),
        [int(len(dataset)*.8), int(len(dataset)*.2)],
        sum(
            [int(len(dataset)*.8), int(len(dataset)*.2)],
        ),
    )
    train_data_set, validation_data_set = random_split(
        dataset,
        [int(len(dataset)*.8)+1, int(len(dataset)*.2)],
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
    history = model.fit(1, .1, train_loader, validation_loader)

    print('Training...')
