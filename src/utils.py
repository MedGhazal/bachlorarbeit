from tqdm import tqdm
import os
from functools import wraps
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import normalize, cross_entropy
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


activities_dictionary = [
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
]
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


def get_device(device_cpu=False):

    if device_cpu and not torch.cuda.is_available():
        print(f'Used device is cpu')
        return torch.device('cpu')
    elif torch.backends.mps.is_available():
        print(f'Used device is mps')
        return torch.device('mps')
    elif torch.cuda.is_available():
        print(f'Used device is cuda')
        return torch.device('cuda')
    else:
        print(f'Used device is cpu')
        return torch.device('cpu')


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

    def training_step(self, batch, device, weights=None):
        motions, labels = batch
        motions, labels = motions.to(device), labels.to(device)
        targets = self.forward(motions, device)
        if self.loss_function == cross_entropy:
            if weights is None:
                return self.loss_function(
                    labels,
                    targets,
                    reduction='mean',
                )
            else:
                return self.loss_function(
                    labels,
                    targets,
                    reduction='mean',
                    weight=weights,
                )
        else:
            return self.loss_function(labels, targets)

    def validation_step(self, batch, device, weights=None):
        motions, labels = batch
        motions = motions.to(device)
        labels = labels.to(device)
        outputs = self.forward(motions, device)
        with torch.no_grad():
            loss = self.training_step(batch, device, weights=weights)
        accuracy_ = accuracy(outputs, labels)
        return {'valueLoss': loss, 'valueAccuracy': accuracy_}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['valueLoss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['valueAccuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() * 100
        return {'valueLoss': epoch_loss.item(), 'valueAccuracy': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            f"Final epoch, the loss value: {result['valueLoss']:.4f}"
            f" with model accuracy: {result['valueAccuracy']:.2f}%"
        )

    def evaluate(self, valuationSetLoader, device, weights=None):
        outputs = []
        labels, predictions = None, None
        for batch in valuationSetLoader:
            outputs.append(self.validation_step(batch, device, weights=weights))
            motions, batch_labels = batch
            motions = motions.to(device)
            batch_outputs = self.forward(motions, device)
            batch_predictions = torch.argmax(batch_outputs, dim=1)
            batch_labels = torch.argmax(batch_labels, dim=1)
            if labels is None and predictions is None:
                labels = batch_labels
                predictions = batch_predictions
            else:
                labels = torch.cat((labels, batch_labels))
                predictions = torch.cat((predictions, batch_predictions))
        return self.validation_epoch_end(outputs), labels, predictions

    def fit(
        self,
        epochs,
        learning_rate,
        device,
        train_loader,
        validation_loader,
        weights=None,
    ):
        history, training_losses = [], []
        optimizer = self.optimizer(self.parameters(), learning_rate)
        learning_scheduler = ReduceLROnPlateau(optimizer, patience=2,)
        print('Training...')
        for epoch in tqdm(range(1, epochs), ncols=50,):
            for batch in train_loader:
                loss = self.training_step(batch, device, weights=weights)
                training_losses.append(float(loss))
                loss.backward()
                optimizer.step()
            adjust_learning_rate(optimizer, epoch, learning_rate)
            result, labels, predictions = self.evaluate(
                validation_loader,
                device,
                weights=weights,
            )
            # learning_scheduler.step(training_losses[-1])
            history.append(result)
        self.epoch_end(epoch, result)
        return training_losses, history, labels, predictions


def accuracy(outputs, labels):
    predictions = torch.argmax(outputs, dim=1)
    labels = torch.argmax(labels, dim=1)
    return torch.tensor(
        torch.sum(predictions == labels).item() / len(predictions)
    )


def normalize_dataset(
    dataset,
    labels,
    oversample=False,
    oversample_values=None,
):
    normalized_dataset = []
    CLASSES_MAPPING = {
        activity: number for number, activity in enumerate(labels)
    }
    for matrix_positions, label in dataset:
        if label in CLASSES_MAPPING.keys():
            onehot_presentation = torch.zeros((len(CLASSES_MAPPING)))
            onehot_presentation[CLASSES_MAPPING[label]] = 1.0
            item = [
                normalize(torch.tensor(matrix_positions)).float(),
                onehot_presentation,
            ]
            if oversample:
                normalized_dataset += oversample_values[label] * [item]
            else:
                normalized_dataset.append(item)
    return normalized_dataset


def prepare_dataset(
    dataset,
    labels,
    normalize=False,
    oversample=True,
    oversample_values=None,
):
    if normalize:
        return normalize_dataset(
            dataset,
            labels,
            oversample=oversample,
            oversample_values=oversample_values,
        )
    prepared_dataset = []
    CLASSES_MAPPING = {
        activity: number for number, activity in enumerate(labels)
    }
    for matrix_positions, label in dataset:
        if label in CLASSES_MAPPING.keys():
            onehot_presentation = torch.zeros((len(CLASSES_MAPPING)))
            onehot_presentation[CLASSES_MAPPING[label]] = 1.0
            item = [torch.tensor(matrix_positions).float(), onehot_presentation]
            if oversample:
                prepared_dataset += oversample_values[label] * [item]
            else:
                prepared_dataset.append(item)
    return prepared_dataset


def adjust_learning_rate(optimizer, epoch, base_learning_rate):
    learning_rate = base_learning_rate / epoch
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = learning_rate
