import pandas as pd
import numpy as np
from random import randrange
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import grid, row, column
from bokeh.palettes import Viridis, Iridescent
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
from bokeh.transform import transform

from utils import activities_dictionary, CLASSES_MAPPING

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


def visualize_confusion_matrix(labels_, predictions_, final_accuracy_values):
    figures = []
    for fold, (labels, predictions) in enumerate(zip(labels_, predictions_)):
        confusion_matrix = pd.DataFrame(
            get_confusion_matrix(labels, predictions),
            index=list(CLASSES_MAPPING.keys()),
            columns=list(CLASSES_MAPPING.keys()),
        )
        confusion_matrix.index.name = 'labels'
        confusion_matrix.columns.name = 'predictions'
        confusion_matrix = confusion_matrix.stack().rename("value").reset_index()
        figure_ = figure(
            width=550,
            height=400,
            title=(
                f'Confusion matrices for fold {fold+1} '
                f'with accuracy {final_accuracy_values[fold]:.2f}%'
            ),
            x_range=list(CLASSES_MAPPING.keys()),
            y_range=list(CLASSES_MAPPING.keys())[::-1],
            x_axis_label='Predictions',
            y_axis_label='Labels',
            x_axis_location="above",
            tooltips=[
                ('Num', '@value'),
                ('Labels', '@labels'),
                ('Predictions', '@predictions'),
            ],
        )
        color_mapper = LinearColorMapper(
            palette=Iridescent[23][::-1],
            low=confusion_matrix.value.min(),
            high=confusion_matrix.value.max(),
        )
        color_bar = ColorBar(
                color_mapper=color_mapper,
                location=(0, 0),
                ticker=BasicTicker(desired_num_ticks=10),
        )
        figure_.rect(
            x='predictions',
            y='labels',
            width=1,
            height=1,
            source=ColumnDataSource(confusion_matrix),
            fill_color=transform('value', color_mapper),
            line_color=transform('value', color_mapper),
        )
        figure_.add_layout(color_bar, 'right')
        figures.append(figure_)
    return figures


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


def plot(model, histories, labels_, predictions_, training_losses_):
    output_file('plots/Experiements.html')
    figures = []
    figure_validation_losses = figure(
        title=f'The validation loss values the {model} over epochs',
        # width=300,
        # height=300,
    )
    figure_training_losses = figure(
        title=f'The training loss values the {model} over epochs',
        # width=300,
        # height=300,
    )
    figure_accuracies = figure(
        title=f'The accuracy the {model} over epochs',
        # width=300,
        # height=300,
    )
    figures.append(figure_accuracies)
    figures.append(figure_validation_losses)
    figures.append(figure_training_losses)
    final_accuracy_values = []
    for fold, (history, training_losses) in enumerate(
        zip(
            histories,
            training_losses_,
        )
    ):
        color = get_random_color()
        loss_values = [values['valueLoss'] for values in history]
        accuracy_values = [values['valueAccuracy'] for values in history]
        final_accuracy_values.append(accuracy_values[-1])
        visualize_accuracies(figure_accuracies, fold+1, accuracy_values, color)
        visualize_losses(figure_training_losses, fold+1, training_losses, color)
        visualize_losses(figure_validation_losses, fold+1, loss_values, color)
    figures_confusion_matrices =  visualize_confusion_matrix(
        labels_, predictions_, final_accuracy_values
    )
    grid_ = grid(column(row(*figures), row(*figures_confusion_matrices)))
    show(grid_)


def visualize_length_distribution(motions_lengths):
    output_file('plots/length_distribution.html')
    figures = [
        figure(title='Frames length distribution'),
        figure(
            title='Length of frames',
            tooltips=[('Motion', '$y'),('Length', '$x')],
        )
    ]
    lengths = list(motions_lengths.values())
    bins = np.linspace(min(lengths), max(lengths), 40)
    histogram_, edges = np.histogram(lengths, density=True, bins=bins)
    figures[0].quad(
        top=histogram_*len(lengths),
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color='#000000',
        line_color='#ffffff',
    )
    figures[1].scatter(
        x=lengths,
        y=list(map(lambda x: int(x), motions_lengths.keys())),
        fill_color='#000000',
        line_color='#000000',
        fill_alpha=.3,
        line_alpha=.4,
        size=10,
    )
    grid_ = grid(row(*figures))
    show(grid_)
    return figures
