import pandas as pd
import numpy as np
from random import randrange
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import grid, row

from utils import activities_dictionary

def get_random_hex():
    return hex(randrange(17, 255))[2:].upper()


def get_random_color():
    return f'#{get_random_hex()}{get_random_hex()}{get_random_hex()}'


def get_element_confusion_matrix(position, labels, predictions):
    row, column = position
    return sum(
        1 for index, label in enumerate(labels)
        if label == row and column == predictions[index]
    )


def get_row(row, labels, predictions):
    return [
        get_element_confusion_matrix((row, column), labels, predictions)
        for column in range(len(activities_dictionary))
    ]


def get_confusion_matrix(labels, predictions):
    confusion_matrix = [
        get_row(row, labels, predictions) for row in range(len(activities_dictionary))
    ]
    return np.array(confusion_matrix)


def visualize_confusion_matrix(labels, predictions, fold):
    confusion_matrix = get_confusion_matrix(labels, predictions)
    print(confusion_matrix)
    data = {
        'labels': labels,
        'predictions': predictions,
    }
    data_frame = pd.DataFrame(data)
    confusion_matrix = pd.crosstab(data['labels'], data['predictions'], rownames=['Labels'], colnames=['Predictions'])
    print(confusion_matrix)


def visualize_losses(figure_losses, fold, loss_values, color):
    figure_losses.line(
        x=list(range(len(loss_values))),
        y=loss_values,
        name=f'{fold}',
        legend_label=f'{fold}. fold',
        line_color=color,
    )


def visualize_accuracies(figure_accuracies, fold, accuracy_values, color):
    figure_accuracies.line(
        x=list(range(len(accuracy_values))),
        y=accuracy_values,
        name=f'{fold}',
        legend_label=f'{fold}. fold',
        line_color=color,
    )


def plot(model, histories, labels, predictions):
    output_file('plots/Experiements.html')
    figures = []
    figure_losses = figure(
        title=f'The loss values the {model} over epochs',
        # width=300,
        # height=300,
    )
    figure_accuracies = figure(
        title=f'The accuracy the {model} over epochs',
        # width=300,
        # height=300,
    )
    figures.append(figure_accuracies)
    figures.append(figure_losses)
    for fold, history in enumerate(histories):
        color = get_random_color()
        loss_values = [values['valueLoss'] for values in history]
        accuracy_values = [values['valueAccuracy'] for values in history]
        visualize_accuracies(figure_accuracies, fold+1, accuracy_values, color)
        visualize_losses(figure_losses, fold+1, loss_values, color)
    for fold, labels_, predictions_ in enumerate(zip(labels, predictions)):
        visualize_confusion_matrix(labels, predictions, fold+1)
    grid_ = grid(row(*figures))
    show(grid_)


if __name__ == '__main__':
    labels = [4, 1, 1, 3, 7, 9, 1, 7, 7, 6, 1, 6, 1, 4, 0, 4, 1, 1, 1, 1, 1, 1, 7, 1, 7, 7, 3, 1, 7, 2, 2, 4]
    predictions = [4, 1, 1, 3, 7, 9, 1, 7, 7, 6, 1, 6, 1, 4, 0, 4, 1, 1, 1, 1, 1, 1, 7, 1, 7, 7, 3, 1, 7, 2, 2, 4]
    print(get_confusion_matrix(labels, predictions))
    data = {
        'labels': labels,
        'predictions': predictions,
    }
    data_frame = pd.DataFrame(data)
    confusion_matrix = pd.crosstab(data['labels'], data['predictions'], rownames=['Labels'], colnames=['Predictions'])
    print(confusion_matrix)
