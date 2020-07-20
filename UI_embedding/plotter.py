import matplotlib.pyplot as plt
import numpy as np

def plot_loss(training_loss, testing_loss, output_path=''):
    epochs = list(enumerate(range(len(training_loss))))
    fig = plt.figure()
    plot = fig.add_subplot(1,1,1)
    plot.plot(epochs, training_loss, color='blue')
    if testing_loss:
        plot.plot(epochs, testing_loss, color='red')
    plot.set_title('Loss across epochs')
    fig.savefig(output_path+ "loss_figure.png")