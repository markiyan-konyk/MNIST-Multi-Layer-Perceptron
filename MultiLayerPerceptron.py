import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


mnist_train = datasets.MNIST(root="./datasets", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root="./datasets", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=True)

# Parametres
W1 = torch.randn(784,500) / np.sqrt(784)
W1.requires_grad_()
W2 = torch.randn(500,10) / np.sqrt(500)
W2.requires_grad_()

b = torch.zeros(10, requires_grad=True)

#The optimizer
optimizer = torch.optim.SGD([W1, W2, b], lr=0.99)


for image, label in tqdm(train_loader):
    optimizer.zero_grad()

    x = image.view(-1, 28*28)
    y1 = torch.matmul(x,W1)
    relu_y1 = F.relu(y1)
    y2 = torch.matmul(relu_y1, W2) + b

    cross_entropy = F.cross_entropy(y2, label)
    cross_entropy.backward()
    optimizer.step()

correct = 0
total = len(mnist_test)

with torch.no_grad():
    for image, label in tqdm(test_loader):
        x = image.view(-1, 28*28)
        y1 = torch.matmul(x, W1)
        y1 = F.relu(y1)
        y2 = torch.matmul(y1, W2) + b 

        predictions = torch.argmax(y2, dim=1)
        correct += torch.sum((predictions == label).float())

print("Test accuracy: {}%".format(100 * correct / total))

with torch.no_grad():
    while True:
        i = random.randint(0, len(mnist_test) - 1)
        image, label = mnist_test[i]

        x = image.view(-1,28*28)
        y1 = torch.matmul(x, W1)
        y1 = F.relu(y1)
        y2 = torch.matmul(y1, W2) + b 
        prediction = torch.argmax(y2, dim=1).item()


        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Predicted: {prediction} | Actual: {label}")
        plt.axis("off")
        plt.show()

        user_input = input("Press Enter to continue, or type 'end' to stop: ").strip().lower()
        if user_input == "end":
            print("Exiting...")
            break

