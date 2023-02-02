import json
from dataset import get_folds
from utils import get_device
from models import MLP, CNN, FCN, ResNet, RNN, GRU, LSTM, CNN_LSTM, train_model


class Experiment:

    def __init__(
        self,
        model,
        device,
        num_epochs,
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
        num_layers=None,
        hidden_size=None,
    ):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.frequency = frequency
        self.min_length = min_length
        self.max_length = max_length
        self.get_matrixified_root_infomation = get_matrixified_root_infomation
        self.get_matrixified_joint_positions = get_matrixified_joint_positions
        self.get_matrixified_all = get_matrixified_all
        self.padding = padding
        self.inverse = inverse
        self.normalize = normalize
        self.oversample = oversample
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def get_model(self):
        models = {
            'MLP' : MLP,
            'CNN' : CNN,
            'FCN' : FCN,
            'ResNet' : ResNet,
            'RNN' : RNN,
            'GRU' : GRU,
            'LSTM' : LSTM,
            'CNN_LSTM' : CNN_LSTM,
        }
        return models[self.model]

    def run(self):
        device = get_device(device_cpu=self.device=='cpu')
        model = self.get_model()
        weights, folds = get_folds(
            num_folds=5,
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
        train_model(
            device,
            model,
            weights,
            folds,
            self.num_epochs,
            self.frequency,
            self.min_length,
            self.max_length,
            num_features,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
        )


def main():
    with open('Experiments.json', 'r') as json_file:
        experiments = json.load(json_file)
    for experiment_data in experiments:
        experiment = Experiment(
            experiment_data['model'],
            experiment_data['device'],
            experiment_data['num_epochs'],
            experiment_data['frequency'],
            experiment_data['min_length'],
            experiment_data['max_length'],
            experiment_data['get_matrixified_root_infomation'],
            experiment_data['get_matrixified_joint_positions'],
            experiment_data['get_matrixified_all'],
            experiment_data['padding'],
            experiment_data['inverse'],
            experiment_data['normalize'],
            experiment_data['oversample'],
            num_layers=experiment_data.get('num_layers'),
            hidden_size=experiment_data.get('hidden_size'),
        )
        experiment.run()


if __name__ == '__main__':
    main()
