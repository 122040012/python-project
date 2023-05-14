import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
import numpy as np
import os

# Hyperparameters
batch_size = 100
learning_rate = 0.001
num_epochs = 15

# Transforms to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train():
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Save and print loss and accuracy
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')
            train_losses.append(loss.item())
            train_accuracies.append(accuracy * 100)

    return train_losses, train_accuracies


def test():
    model.eval()
    test_loss = []
    test_accuracy = []

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save and print loss and accuracy for each batch
            batch_loss = loss.item()
            batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
            print(f'Step [{i + 1}/{len(test_loader)}], Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%')
            test_loss.append(batch_loss)
            test_accuracy.append(batch_accuracy)

    return test_loss, test_accuracy


def plot_individual(train_loss, train_accuracy, test_loss, test_accuracy, seed):
    # Create a directory to save individual plots
    os.makedirs(f'plots_seed_{seed}', exist_ok=True)

    # Plot individual training loss
    plt.figure(seed + 1)
    plt.plot(train_loss)
    plt.title(f"Training Loss - Seed {seed}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(f'plots_seed_{seed}/training_loss.png')

    # Plot individual training accuracy
    plt.figure(seed + 2)
    plt.plot(train_accuracy)
    plt.title(f"Training Accuracy - Seed {seed}")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.savefig(f'plots_seed_{seed}/training_accuracy.png')

    # Plot individual testing loss
    plt.figure(seed + 3)
    plt.plot(test_loss)
    plt.title(f"Testing Loss - Seed {seed}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(f'plots_seed_{seed}/testing_loss.png')

    # Plot individual testing accuracy
    plt.figure(seed + 4)
    plt.plot(test_accuracy)
    plt.title(f"Testing Accuracy - Seed {seed}")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.savefig(f'plots_seed_{seed}/testing_accuracy.png')

def plot():
    mean_train_loss = np.loadtxt("mean_train_loss.txt")
    mean_train_accuracy = np.loadtxt("mean_train_accuracy.txt")
    mean_test_loss = np.loadtxt("mean_test_loss.txt")
    mean_test_accuracy = np.loadtxt("mean_test_accuracy.txt")

    # Plot training loss
    plt.figure("Mean Training Loss")
    plt.plot(mean_train_loss)
    plt.title("Mean Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("mean_training_loss.png")

    # Plot testing loss
    plt.figure("Mean Testing Loss")
    plt.plot(mean_test_loss)
    plt.title("Mean Testing Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("mean_testing_loss.png")

    # Plot testing accuracy
    plt.figure("Mean Testing Accuracy")
    plt.plot(mean_test_accuracy)
    plt.title("Mean Testing Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.savefig("mean_testing_accuracy.png")

    # Plot mean training accuracy
    plt.figure("Mean Training Accuracy")
    plt.plot(mean_train_accuracy)
    plt.title("Mean Training Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.savefig("mean_training_accuracy.png")

def run_single_seed(seed, train_losses, train_accuracies, test_losses, test_accuracies, seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loss, train_accuracy = train()
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy = test()
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    seeds.append(seed)

if __name__ == '__main__':
    random_seeds = [42, 123, 456]  # Define your random seeds here

    # Use Manager for sharing data between processes
    with Manager() as manager:
        train_losses = manager.list()
        train_accuracies = manager.list()
        test_losses = manager.list()
        test_accuracies = manager.list()
        seeds = manager.list()

        # Create and start processes for each random seed
        processes = []
        for seed in random_seeds:
            process = Process(target=run_single_seed,
                              args=(seed, train_losses, train_accuracies, test_losses, test_accuracies, seeds))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Convert results to NumPy arrays
        train_losses = np.array(train_losses)
        train_accuracies = np.array(train_accuracies)
        test_losses = np.array(test_losses)
        test_accuracies = np.array(test_accuracies)

        # Save the results to files
        np.savetxt("train_losses.txt", train_losses)
        np.savetxt("train_accuracies.txt", train_accuracies)
        np.savetxt("test_losses.txt", test_losses)
        np.savetxt("test_accuracies.txt", test_accuracies)
        np.savetxt("seeds.txt", seeds)

        # Calculate the mean of training loss, testing loss, and testing accuracy
        mean_train_loss = np.mean(train_losses, axis=0)
        mean_train_accuracy = np.mean(train_accuracies, axis=0)
        mean_test_loss = np.mean(test_losses, axis=0)
        mean_test_accuracy = np.mean(test_accuracies, axis=0)

        # Save the mean values to files
        np.savetxt("mean_train_loss.txt", mean_train_loss)
        np.savetxt("mean_train_accuracy.txt", mean_train_accuracy)
        np.savetxt("mean_test_loss.txt", mean_test_loss)
        np.savetxt("mean_test_accuracy.txt", mean_test_accuracy)

        # Plot the mean values
        plot()
        plt.show()
