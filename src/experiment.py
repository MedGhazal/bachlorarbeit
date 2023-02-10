import os
import json
from bokeh.io import output_file, show
from bokeh.models import Tabs, TabPanel

from dataset import get_folds
from utils import get_device, change_to
from models import MLP, CNN, FCN, ResNet, RNN, GRU, LSTM, train_model
from plotters import plot_experiment_set


class ExperimentSet:

    def __init__(
        self,
        labels,
        frequency,
        min_length,
        max_length,
        get_matrixified_root_infomation,
        get_matrixified_joint_positions,
        get_matrixified_all,
        padding,
        inverse,
        normalize,
        oversample,
    ):
        self.labels = labels
        self.frequency = frequency
        self.min_length = min_length
        self.max_length = max_length
        self.get_matrixified_all = get_matrixified_all
        self.get_matrixified_joint_positions = get_matrixified_joint_positions
        self.get_matrixified_root_infomation = get_matrixified_root_infomation
        self.padding = padding
        self.inverse = inverse
        self.normalize = normalize
        self.oversample = oversample
        self.experiments = []

    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def run(self):
        weights, folds = get_folds(
            num_folds=5,
            labels=self.labels,
            frequency=self.frequency,
            min_length=self.min_length,
            max_length=self.max_length,
            get_matrixified_root_infomation=self.get_matrixified_root_infomation,
            get_matrixified_joint_positions=self.get_matrixified_joint_positions,
            get_matrixified_all=self.get_matrixified_all,
            padding=self.padding,
            inverse=self.inverse,
            normalize=self.normalize,
            oversample=self.oversample,
        )
        if self.get_matrixified_root_infomation:
            num_features = 6
        if self.get_matrixified_joint_positions:
            num_features = 44
        if self.get_matrixified_all:
            num_features = 50
        tabs = []
        extention = (
            f'{"n" if self.normalize else ""}'
            f'{"o" if self.oversample else ""}'
        )
        output_file(
            f'plots/Experiements_{num_features}'
            f'_{self.frequency}{extention}.html'
        )
        for experiment in self.experiments:
            (
                histories,
                labels_,
                predictions_,
                training_losses_,
            ) = experiment.run(
                folds,
                weights,
                num_features,
                num_frames=self.max_length//self.frequency,
            )
            tabs.append(
                TabPanel(
                    child=plot_experiment_set(
                        experiment.model,
                        self.labels,
                        self.frequency,
                        num_features,
                        histories,
                        labels_,
                        predictions_,
                        training_losses_,
                        bidirectional=experiment.bidirectional,
                    ),
                    title=(
                        f'{"bi-" if experiment.bidirectional else ""}'
                        f'{experiment.model}'
                    ),
                ),
            )
        show(Tabs(tabs=tabs))
        os.remove('dataset.pt')


class Experiment:

    def __init__(
        self,
        model,
        num_classes,
        device,
        num_epochs,
        learning_rate,
        num_layers=None,
        hidden_size=None,
        bidirectional=False,
    ):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)

    def get_model(self):
        models = {
            'MLP': MLP,
            'CNN': CNN,
            'FCN': FCN,
            'ResNet': ResNet,
            'RNN': RNN,
            'GRU': GRU,
            'LSTM': LSTM,
        }
        return models[self.model]

    def run(self, folds, weights, num_features, num_frames=None):
        device = get_device(device_cpu=self.device == 'cpu')
        model = self.get_model()
        return train_model(
            device,
            model,
            self.num_classes,
            weights,
            folds,
            self.num_epochs,
            self.learning_rate,
            num_frames,
            num_features,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
        )


# @change_to('experiments_one2one/all_labels')
@change_to('.')
def get_experiement_sets():
    with open('Experiments.json', 'r') as json_file:
        experiment_sets = json.load(json_file)
    with open('labels.json', 'r') as json_file:
        labels = json.load(json_file)
    return experiment_sets, labels


def main():
    experiment_sets, labels = get_experiement_sets()
    num_classes = len(labels)
    for experiments in experiment_sets:
        params = experiments['params']
        experiment_set = ExperimentSet(
            labels,
            params['frequency'],
            params['min_length'],
            params['max_length'],
            params['get_matrixified_root_infomation'],
            params['get_matrixified_joint_positions'],
            params['get_matrixified_all'],
            params['padding'],
            params['inverse'],
            params['normalize'],
            params['oversample'],
        )
        models = experiments['models']
        for model in models:
            experiment_set.add_experiment(
                Experiment(
                    model['model'],
                    num_classes,
                    model['device'],
                    model['num_epochs'],
                    model['learning_rate'],
                    num_layers=model.get('num_layers'),
                    hidden_size=model.get('hidden_size'),
                    bidirectional=model.get('bidirectional'),
                )
            )
        experiment_set.run()


if __name__ == '__main__':
    main()
