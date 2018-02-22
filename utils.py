from torch import utils
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

def loader(train=True, batch_size=50, shuffle=True, normalize=None, path='./dataset'):
    if normalize is not None:
        transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(normalize[0], normalize[1])
                   ])
    else:
        transform = transforms.ToTensor()
    return utils.data.DataLoader(
    datasets.MNIST(path, train=train, download=True,
                   transform=transform), batch_size=batch_size, shuffle=shuffle)
                   
def plot_mnist(images, shape):
    fig = plt.figure(figsize=shape[::-1], dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[0], shape[1], j)
        ax.matshow(images[j - 1, 0, :, :], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()
    
def plot_results(model, loader, shape):
    data, target = next(iter(loader))
    data, target = Variable(data, volatile=True), Variable(target)
    
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    plot_mnist(data.data.numpy(), shape)
    print(pred.numpy().reshape(shape))
    
def plot_graphs(log, tpe='loss'):
    train_log = [z for z in zip(*log['train'])]
    test_log = [z for z in zip(*log['test'])]
    train_epochs = range(len(log['train']))
    test_epochs = range(len(log['test']))
    
    if tpe == 'loss':
        train_handler, = plt.plot(train_epochs, train_log[0], color='r', label='train')
        test_handler, = plt.plot(test_epochs, test_log[0], color='b', label='test')
        plt.title('errors')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend(handles=[train_handler, test_handler])
        plt.show()
    elif tpe == 'accuracy':
        train_handler, = plt.plot(train_epochs, train_log[1], color='r', label='train')
        test_handler, = plt.plot(test_epochs, test_log[1], color='b', label='test')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(handles=[train_handler, test_handler])
        plt.show()