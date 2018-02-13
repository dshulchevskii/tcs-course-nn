from torch import utils
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def loader(train=True, batch_size=50, shuffle=True, normalize=True):
    if normalize:
        transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    else:
        transform = transforms.ToTensor()
    return utils.data.DataLoader(
    datasets.MNIST('./dataset', train=train, download=True,
                   transform=transform), batch_size=batch_size, shuffle=shuffle)
                   
def plot_mnist(images, shape):
    fig = plt.figure(figsize=shape, dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[1], shape[0], j)
        ax.matshow(images[j - 1, 0, :, :], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()