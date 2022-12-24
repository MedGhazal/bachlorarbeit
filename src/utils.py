from tqdm import tqdm
import os
from functools import wraps
import torch
import torch.nn as nn
from torch.nn.functional import normalize


activities_dictionary = {
    'walk',
    'turn',
    'run',
    'stand',
    'jump',
    'wave',
    'stumbl',
    'danc',
    'throw',
    'kneel',
    'kick',
}
CLASSES_MAPPING = {
    activity: number for number, activity in enumerate(activities_dictionary)
}


def check_exists():
    return os.path.isdir(
        os.path.join(
            os.path.expanduser(''),
            'motion_dataset',
        )
    )


def change_to(path):
    def decorator(function):
        @wraps(function)
        def func(*args, **kwargs):
            current_path = os.getcwd()
            if not current_path == os.path.join(
                os.sep.join(current_path.split(os.sep)[:-2]),
                path,
            ):
                os.chdir(os.path.join(current_path, path))
                returns = function(*args, **kwargs)
                os.chdir(current_path)
            else:
                returns = function(*args, **kwargs)
            return returns
        return func
    return decorator


class Model(nn.Module):

    def __init__(
        self,
        num_classes=None,
        num_features=None,
        num_frames=None,
        optimizer=None,
        loss_function=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_frames = num_frames
        self.optimizer = optimizer
        self.loss_function = loss_function

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
        normalizes_positions = normalize(torch.tensor(matrix_positions))
        normalized_dataset.append(
            [normalizes_positions.float(), onehot_presentation]
        )
    return normalized_dataset
