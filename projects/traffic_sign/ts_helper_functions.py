# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
import matplotlib.pyplot as plt 

def plot_accuracies(train_accuracies, test_accurcaries):
    """ Plot the history of accuracies"""
    plt.plot(train_accuracies, '-bx')
    plt.plot(test_accurcaries, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')    
    plt.legend(['Training', 'Testing'])
    plt.title('Accuracy vs. No. of epochs')
    plt.show()
    

def plot_losses(train_losses, test_losses):
    """ Plot the losses in each epoch"""
    plt.plot(train_losses, '-bx')
    plt.plot(test_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Testing'])
    plt.title('Loss vs. No. of epochs')
    plt.show()
