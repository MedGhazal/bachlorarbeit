import matplotlib.pyplot as plt


def plot(histories):
    for history in histories:
        # loss_values = [values['valueLoss'] for values in history]
        accuracy_values = [values['valueAccuracy'] for values in history]
        plt.plot(list(range(1, len(history)+1)), accuracy_values)
        plt.title('Accuracy values per epoch')
    plt.show()
