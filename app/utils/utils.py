import matplotlib.pyplot as plt
import os


def plot(data, data_avg, title, file_name, output_directory, legend=None):
    plt.plot(data)
    plt.plot(data_avg)
    if legend is not None:
        plt.legend(legend, loc='upper left')
    plt.title(title)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig("{}/{}.pdf".format(output_directory, file_name))
    plt.clf()
